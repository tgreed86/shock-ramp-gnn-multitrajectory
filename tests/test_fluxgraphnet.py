"""Flux budgets, AMR geometry, open/wall boundaries, limiting, and training."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

import test_cache_only
import test_model
from engine import build_model, carry_state, model_step
from flux import (build_flux_geometry, conservative_remap, leaf_owners, limit_flux_rate,
                  limit_face_transports, load_boundary_segments)
from geometry import dynamic_cells_from_parent_masks
from mesh_geometry import build_amr_face_adjacency_edges, dec_edge_attr_for_dyadic_quads
from model import FluxGraphNet, RAMP_QUANTITIES
from runtime import Mesh, Runtime
from train import load_config, validate_config


def mesh_for(centers, levels, H=2, W=2):
    edges = build_amr_face_adjacency_edges(centers, levels, H, W, (0., 1., 0., 1.),
                                          return_edge_attr=False, backend='cpu')
    attrs = dec_edge_attr_for_dyadic_quads(centers, levels, edges, dx0=1/W, dy0=1/H)
    return Mesh(centers, levels, torch.zeros(len(centers), dtype=torch.long), edges,
                torch.zeros(H, W, dtype=torch.bool), attrs)


class FluxTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(16)
        self.cfg = load_config(Path(__file__).resolve().parents[1]/'config.json')
        self.cfg['model'].update(name='FluxGraphNet', hidden=8, message_passing_steps=2, mlp_layers=1, dropout=0.)
        self.cfg['data'].update(H=2, W=2)
        centers, levels, _, _ = dynamic_cells_from_parent_masks(
            {1: torch.tensor([[True, False], [False, False]])}, 2, 2, 0., 1., 0., 1.)
        self.mesh = mesh_for(centers, levels)

    def geometry(self, mesh=None, **kwargs):
        m = self.mesh if mesh is None else mesh
        return build_flux_geometry(m.centers, m.levels, m.edges, m.edge_attr,
                                   dx0=.5, dy0=.5, bbox=(0., 1., 0., 1.), **kwargs)

    def net(self, **kwargs):
        return FluxGraphNet(4, 4, hidden=8, message_passing_steps=2, mlp_layers=1,
                            dropout=0., length_scale=.5, **kwargs)

    def test_exact_face_sign_length_and_unequal_area(self):
        # One coarse/fine interface of length .5, with cells of areas 1 and .25.
        mesh = SimpleNamespace(centers=torch.tensor([[.5, .5], [1.25, .25]]), levels=torch.tensor([0, 1]),
                               edges=torch.tensor([[0, 1], [1, 0]]))
        mesh.edge_attr = dec_edge_attr_for_dyadic_quads(mesh.centers, mesh.levels, mesh.edges, dx0=1., dy0=1.)
        geometry = build_flux_geometry(mesh.centers, mesh.levels, mesh.edges, mesh.edge_attr,
                                       dx0=1., dy0=1., bbox=(0., 2., 0., 1.))
        net = FluxGraphNet(4, 4, hidden=4, message_passing_steps=1, mlp_layers=1, dropout=0.,
                           use_open_boundary_source=False, use_ramp_boundary_source=False)
        class Sender(nn.Module):
            def forward(self, x):
                return x[:, :4]
        net.decoder = Sender()
        x = torch.tensor([[2., 4., 6., 8.], [0., 0., 0., 0.]], requires_grad=True)
        with patch.object(net, 'encode_process', return_value=(x, torch.zeros(2, 4))):
            rate, flux, boundary = net.predict_rate_and_flux(x, mesh.edges, mesh.centers, geometry)
        torch.testing.assert_close(flux, torch.tensor([[.5, 1., 1.5, 2.]]))
        torch.testing.assert_close(rate, torch.tensor([[-.5, -1., -1.5, -2.], [2., 4., 6., 8.]]))
        self.assertEqual(boundary.count_nonzero(), 0)
        rate.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_amr_geometry_boundary_lengths_and_reverse_pairing(self):
        g = self.geometry()
        self.assertEqual(2 * len(g.forward_edges), self.mesh.edges.shape[1])
        torch.testing.assert_close(self.mesh.edges[:, g.forward_edges], self.mesh.edges.flip(0)[:, g.reverse_edges])
        torch.testing.assert_close(g.area.sum(), torch.tensor(1.))
        torch.testing.assert_close(g.boundary_length.sum(), torch.tensor(4.))
        torch.testing.assert_close(g.boundary_length[g.boundary_open].sum(), torch.tensor(3.))
        # Every exposed face in this full square lies exactly on the bbox.
        owners, normal = g.boundary_owner, g.boundary_normal
        half = .25 / 2**self.mesh.levels[owners]
        position = self.mesh.centers[owners] + normal * half[:, None]
        self.assertTrue(torch.all(torch.any((position == 0) | (position == 1), dim=1)))
        # Shuffling edges does not rely on forward/reverse blocks being aligned.
        order = torch.randperm(self.mesh.edges.shape[1])
        shuffled = copy.copy(self.mesh)
        shuffled.edges, shuffled.edge_attr = self.mesh.edges[:, order], self.mesh.edge_attr[order]
        g2 = self.geometry(shuffled)
        self.assertEqual(len(g.forward_edges), len(g2.forward_edges))
        with self.assertRaisesRegex(ValueError, 'reverse edge'):
            build_flux_geometry(self.mesh.centers, self.mesh.levels, self.mesh.edges[:, :-1], self.mesh.edge_attr[:-1],
                                dx0=.5, dy0=.5, bbox=(0., 1., 0., 1.))

    def test_closed_domain_mass_balance_gradients_and_permutations(self):
        net = self.net(use_open_boundary_source=False, use_ramp_boundary_source=False)
        x = torch.randn(len(self.mesh.centers), 4, requires_grad=True)
        g = self.geometry()
        rate = net(x, self.mesh.edges, self.mesh.centers, g)
        torch.testing.assert_close((rate * g.area[:, None]).sum(0), torch.zeros(4), atol=2e-7, rtol=0)
        self.assertGreater(float(rate.abs().max()), 0.)
        rate.square().mean().backward()
        for p in (x, *net.parameters()):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        nodes = torch.randperm(len(x))
        edges = torch.randperm(self.mesh.edges.shape[1])
        m = copy.copy(self.mesh)
        m.centers, m.levels = m.centers[nodes], m.levels[nodes]
        m.edges, m.edge_attr = torch.argsort(nodes)[m.edges][:, edges], m.edge_attr[edges]
        torch.testing.assert_close(net(x[nodes], m.edges, m.centers, self.geometry(m)), rate[nodes],
                                   rtol=1e-5, atol=1e-6)
        # Dropout may change a face prediction, but never its equal/opposite scatter.
        net = FluxGraphNet(4, 4, hidden=8, message_passing_steps=2, dropout=.4,
                           use_open_boundary_source=False, use_ramp_boundary_source=False)
        torch.testing.assert_close((net(x, m.edges, m.centers, self.geometry(m)) * self.geometry(m).area[:, None]).sum(0),
                                   torch.zeros(4), atol=5e-7, rtol=0)

    def test_open_exchange_wall_masks_and_interior_localization(self):
        y, x = torch.meshgrid((torch.arange(3)+.5)/3, (torch.arange(3)+.5)/3, indexing='ij')
        m = mesh_for(torch.stack([x.flatten(), y.flatten()], 1), torch.zeros(9, dtype=torch.long), H=3, W=3)
        g = build_flux_geometry(m.centers, m.levels, m.edges, m.edge_attr,
                                dx0=1/3, dy0=1/3, bbox=(0., 1., 0., 1.))
        net = self.net()
        with torch.no_grad():
            for p in net.parameters():
                p.zero_()
            net.open_boundary_head[-1].bias.copy_(torch.tensor([1., 2., -3., 4.]))
            net.ramp_boundary_head[-1].bias.fill_(7.)
        rate, internal, boundary = net.predict_rate_and_flux(torch.zeros(9, 4), m.edges, m.centers, g)
        torch.testing.assert_close(internal, torch.zeros_like(internal))
        torch.testing.assert_close(rate[4], torch.zeros(4))  # Interior cell has no source.
        torch.testing.assert_close(boundary[~g.boundary_open][:, [0, 3]], torch.zeros(3, 2))
        expected_outward = torch.tensor([3., 13., -2., 12.]) * net.length_scale
        torch.testing.assert_close((rate * g.area[:, None]).sum(0), -expected_outward)
        torch.testing.assert_close((rate * g.area[:, None]).sum(0), -boundary.sum(0) * net.length_scale**2)
        # Independent open and wall faces both contribute at bottom corners.
        self.assertLess(float(rate[0, 0]), 0.)
        self.assertEqual(float(rate[1, 0]), 0.)
        rate.square().mean().backward()
        self.assertGreater(float(net.open_boundary_head[-1].bias.grad.abs().sum()), 0.)
        self.assertGreater(float(net.ramp_boundary_head[-1].bias.grad.abs().sum()), 0.)

    def test_empty_graph_single_cell(self):
        m = mesh_for(torch.tensor([[.5, .5]]), torch.tensor([0]), H=1, W=1)
        g = build_flux_geometry(m.centers, m.levels, m.edges, m.edge_attr,
                                dx0=1., dy0=1., bbox=(0., 1., 0., 1.))
        net = self.net(use_open_boundary_source=False, use_ramp_boundary_source=False)
        torch.testing.assert_close(net(torch.ones(1, 4), m.edges, m.centers, g), torch.zeros(1, 4))
        self.assertTrue(torch.isfinite(self.net()(torch.ones(1, 4), m.edges, m.centers, g)).all())

    def test_ramp_staircase_and_boundary_heads_disabled(self):
        # A lower-right cutout mimics the staircase of full cells above a ramp.
        y, x = torch.meshgrid((torch.arange(4)+.5)/4, (torch.arange(4)+.5)/4, indexing='ij')
        keep = y >= .5*x
        m = mesh_for(torch.stack([x[keep], y[keep]], 1), torch.ones(int(keep.sum()), dtype=torch.long))
        g = self.geometry(m)
        wall = ~g.boundary_open
        self.assertTrue(torch.any(g.boundary_normal[wall, 0] > 0))
        self.assertTrue(torch.any(g.boundary_normal[wall, 1] < 0))
        self.assertTrue(torch.all(g.boundary_normal[g.boundary_open, 1] >= 0))
        net = self.net(use_open_boundary_source=False)
        rate, _, boundary = net.predict_rate_and_flux(torch.randn(len(m.centers), 4), m.edges, m.centers, g)
        torch.testing.assert_close(boundary[g.boundary_open], torch.zeros_like(boundary[g.boundary_open]))
        torch.testing.assert_close((rate*g.area[:, None]).sum(0)[[0, 3]], torch.zeros(2), atol=1e-7, rtol=0)

    def test_mapped_top_is_open_even_away_from_bbox(self):
        bottom, top = torch.tensor([[0., 0.], [1., .3]]), torch.tensor([[0., .7], [1., 1.]])
        spec = dict(bottom=bottom, top=top, left=torch.stack([bottom[0], top[0]]),
                    right=torch.stack([bottom[1], top[1]]),
                    outer_polygon_ccw=torch.stack([bottom[0], bottom[1], top[1], top[0], bottom[0]]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'spec.pt'
            torch.save(spec, path)
            segments = load_boundary_segments(path, (0., 1., 0., 1.))
        y, x = torch.meshgrid((torch.arange(8)+.5)/8, (torch.arange(8)+.5)/8, indexing='ij')
        keep = (y-.0625 >= .3*(x+.0625)) & (y+.0625 <= .7+.3*(x-.0625))
        m = mesh_for(torch.stack([x[keep], y[keep]], 1), torch.full((int(keep.sum()),), 2))
        mapped, rectangular = self.geometry(m, boundary_segments=segments), self.geometry(m)
        newly_open = mapped.boundary_open & ~rectangular.boundary_open
        self.assertTrue(newly_open.any())
        # Avoid corner faces where either neighboring physical side can be
        # nearest: check the central sloping top and bottom independently.
        owner_x = m.centers[mapped.boundary_owner, 0]
        central = (owner_x > .2) & (owner_x < .8)
        upper = central & (mapped.boundary_normal[:, 1] > 0)
        lower = central & (mapped.boundary_normal[:, 1] < 0)
        self.assertTrue(upper.any() and lower.any())
        self.assertTrue(newly_open[upper].all())
        self.assertFalse(mapped.boundary_open[lower].any())

    def test_limiter_accepts_roundoff_but_rejects_invalid_states(self):
        state = torch.tensor([[.001, 1.0000001, 0., 10.]])
        zero = torch.zeros_like(state)
        torch.testing.assert_close(limit_flux_rate(zero, state, torch.ones(4), torch.tensor(1.), self.cfg), zero)
        state[0, 1] = 2.
        with self.assertRaisesRegex(ValueError, 'physical bounds'):
            limit_flux_rate(zero, state, torch.ones(4), torch.tensor(1.), self.cfg)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_forward_backward_matches_cpu(self):
        cpu, gpu = self.net(), self.net().cuda()
        gpu.load_state_dict(cpu.state_dict())
        m, g = self.mesh, self.geometry()
        from dataclasses import fields, replace
        device_geometry = replace(g, **{f.name: getattr(g, f.name).cuda() for f in fields(g)})
        x = torch.randn(len(m.centers), 4)
        a = cpu(x, m.edges, m.centers, g)
        b = gpu(x.cuda(), m.edges.cuda(), m.centers.cuda(), device_geometry)
        torch.testing.assert_close(a, b.cpu(), rtol=2e-4, atol=1e-5)
        a.square().mean().backward()
        b.square().mean().backward()
        for p, q in zip(cpu.parameters(), gpu.parameters()):
            torch.testing.assert_close(p.grad, q.grad.cpu(), rtol=1e-3, atol=1e-5)

    def test_shared_limiter_and_physical_normalization_preserve_balance(self):
        area = torch.tensor([1., .25])
        rate = torch.tensor([[-100., 20., -5., -80.], [400., -80., 20., 320.]], requires_grad=True)
        state = torch.tensor([[1., .1, .2, 2.], [3., .4, .1, 10.]])
        sigma = torch.tensor([2., .4, 3., 5.])
        self.cfg['train'].update(rate_clip_norm=50., delta_clip_norm=.2)
        limited = limit_flux_rate(rate, state, sigma, torch.tensor(.3), self.cfg)
        self.assertLess(float(limited.abs().max()), 50.)
        self.assertLessEqual(float((limited*.3).abs().max()), .2)
        predicted = state + limited*.3*sigma
        self.assertTrue(torch.all(predicted[:, [0, 3]] >= 1e-6))
        torch.testing.assert_close(((predicted-state)*area[:, None]).sum(0), torch.zeros(4), atol=1e-6, rtol=0)
        self.assertIs(carry_state(predicted, self.cfg), predicted)
        limited.square().sum().backward()
        self.assertTrue(torch.isfinite(rate.grad).all())
        self.cfg['train'].update(rate_clip_norm=0., delta_clip_norm=0.)
        self.cfg['loss'].update(state_rho_max=4., state_E_max=11., state_m_max=.5, u_clip=.5)
        limited = limit_flux_rate(rate, state, sigma, torch.tensor(1.), self.cfg)
        predicted = state + limited*sigma
        self.assertTrue(torch.all(predicted[:, [1, 2]].abs() <= .5))
        self.assertTrue(torch.all(predicted[:, [1, 2]].abs() <= .5 * predicted[:, :1]))
        self.assertTrue(torch.all(predicted[:, 0] <= 4.) and torch.all(predicted[:, 3] <= 11.))

    def test_conservative_remap_refine_coarsen_different_domains_and_gradients(self):
        coarse_c, coarse_l, _, _ = dynamic_cells_from_parent_masks({}, 2, 2, 0., 1., 0., 1.)
        args = dict(H=2, W=2, bbox=(0., 1., 0., 1.), max_level=1)
        a = leaf_owners(coarse_c, coarse_l, **args)
        b = leaf_owners(self.mesh.centers, self.mesh.levels, **args)
        state = torch.arange(16.).reshape(4, 4).requires_grad_()
        refined = conservative_remap(state, a, b, 7)
        torch.testing.assert_close((refined*self.geometry().area[:, None]).sum(0), state.sum(0)*.25)
        restored = conservative_remap(refined, b, a, 4)
        torch.testing.assert_close(restored, state, rtol=0, atol=0)
        restored.sum().backward()
        torch.testing.assert_close(state.grad, torch.ones_like(state))
        b[0] = -1
        with self.assertRaisesRegex(ValueError, 'identical.*domains'):
            conservative_remap(state, a, b, 7)

    def test_local_limiter_floor_cell_does_not_freeze_remote_transfer(self):
        edges = torch.tensor([[0, 2], [1, 3]])
        g = SimpleNamespace(area=torch.tensor([1., .25, .5, 1.]),
                            forward_edges=torch.arange(2), boundary_owner=torch.empty(0, dtype=torch.long))
        state = torch.tensor([[2., 0., 0., 1e-6], [2., 0., 0., 10.],
                              [2., 0., 0., 10.], [2., 0., 0., 10.]])
        flux = torch.tensor([[0., 0., 0., 1e8], [.1, .02, 0., .1]], requires_grad=True)
        sigma = torch.tensor([2., .4, 3., 5.])
        rate, interior, boundary, scales = limit_face_transports(flux, torch.empty(0, 4), edges,
            g, state, sigma, torch.tensor(.1), 1., self.cfg)
        self.assertEqual(float(scales[0]), 0.)
        self.assertEqual(float(scales[1]), 1.)
        torch.testing.assert_close(interior[1], flux[1])
        predicted = state + rate * .1 * sigma
        self.assertTrue(torch.all(predicted[:, [0, 3]] >= 1e-6))
        torch.testing.assert_close((rate * g.area[:, None]).sum(0), torch.zeros(4), atol=1e-7, rtol=0)
        rate[2].sum().backward()
        self.assertGreater(float(flux.grad[1].abs().sum()), 0.)
        self.assertTrue(torch.isfinite(flux.grad).all())

    def test_local_limiter_all_bounds_exterior_balance_and_gradients(self):
        # Exercise independently limited influx/outflux, coarse-fine faces,
        # exterior exchange, and all configured linear state constraints.
        g, m = self.geometry(), self.mesh
        self.cfg['train'].update(rate_clip_norm=2., delta_clip_norm=.1)
        self.cfg['loss'].update(state_rho_max=2.1, state_E_max=5.1, state_m_max=.2, u_clip=.05)
        state = torch.tensor([2., 0., 0., 5.]).repeat(7, 1)
        sigma = torch.tensor([2., .4, 3., 5.])
        for seed in range(5):
            torch.manual_seed(seed)
            interior = (100 * torch.randn(len(g.forward_edges), 4)).requires_grad_()
            exterior = (100 * torch.randn(len(g.boundary_owner), 4)).requires_grad_()
            rate, limited, boundary, scales = limit_face_transports(interior, exterior, m.edges,
                g, state, sigma, torch.tensor(.3), .5, self.cfg)
            predicted = state + rate * .3 * sigma
            self.assertTrue(torch.all(predicted[:, [0, 3]] >= 1e-6))
            self.assertTrue(torch.all(predicted[:, 0] <= 2.1) and torch.all(predicted[:, 3] <= 5.1))
            self.assertTrue(torch.all(predicted[:, 1:3].abs() <= .2))
            self.assertTrue(torch.all(predicted[:, 1:3].abs() <= .05 * predicted[:, :1]))
            self.assertLessEqual(float(rate.abs().max()), 2.)
            self.assertLessEqual(float((rate * .3).abs().max()), .1)
            torch.testing.assert_close((rate * g.area[:, None]).sum(0), -boundary.sum(0) * .5**2,
                                       atol=1e-7, rtol=1e-5)
            rate.square().mean().backward()
            self.assertTrue(torch.isfinite(interior.grad).all() and torch.isfinite(exterior.grad).all())
        state[0, 0] = -1.
        with self.assertRaisesRegex(ValueError, 'admissible state'):
            limit_face_transports(interior, exterior, m.edges, g, state, sigma, torch.tensor(.3), .5, self.cfg)

    def test_local_limiter_inactive_identity_and_derivative(self):
        edges = torch.tensor([[0], [1]])
        g = SimpleNamespace(area=torch.tensor([1., .25], dtype=torch.float64),
                            forward_edges=torch.arange(1), boundary_owner=torch.empty(0, dtype=torch.long))
        state = torch.tensor([[2., 0., 0., 5.], [2., 0., 0., 5.]], dtype=torch.float64)
        sigma, dt = torch.ones(4, dtype=torch.float64), torch.tensor(.1, dtype=torch.float64)
        def advance(flux):
            return limit_face_transports(flux, flux.new_empty(0, 4), edges, g, state, sigma, dt, 1., self.cfg)[0]
        flux = torch.tensor([[.1, .02, -.03, .2]], dtype=torch.float64, requires_grad=True)
        torch.testing.assert_close(advance(flux), torch.cat([-flux, 4*flux]))
        self.assertTrue(torch.autograd.gradcheck(advance, (flux,)))
        # Energy saturation has a differentiable, nontrivial face ratio.
        flux = torch.tensor([[.1, .02, -.03, 100.]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(advance, (flux,)))

    def test_engine_raw_supervision_trains_completely_saturated_flux(self):
        self.cfg['loss'].update(parc=False, flux_raw_rate_weight=.1)
        self.cfg['features']['build'].update(include_ramp_angle=False, include_signed_distance_to_ramp=False)
        self.cfg['train'].update(flux_limiter='local')
        edges = torch.tensor([[0], [1]])
        g = SimpleNamespace(area=torch.ones(2), forward_edges=torch.arange(1),
                            boundary_owner=torch.empty(0, dtype=torch.long))
        mesh = SimpleNamespace(edges=edges, centers=torch.tensor([[.25, .25], [.75, .25]]),
                               levels=torch.zeros(2, dtype=torch.long))
        state = torch.tensor([[2., 0., 0., 1e-6], [2., 0., 0., 5.]])
        class SaturatedFlux(nn.Module):
            length_scale = 1.
            def __init__(self):
                super().__init__()
                self.amplitude = nn.Parameter(torch.tensor(100.))
            def predict_rate_and_flux(self, *args):
                flux = self.amplitude * torch.tensor([[0., 0., 0., 1.]])
                return torch.cat([-flux, flux]), flux, flux.new_empty(0, 4)
        net = SaturatedFlux()
        rt = SimpleNamespace(flux_geometry=lambda mesh: g)
        diagnostics = {}
        loss, predicted = model_step(net, state, state, mesh, SimpleNamespace(ramp={}), self.cfg,
            rt, torch.zeros(4), torch.ones(4), .1, .1, 0, diagnostics=diagnostics)
        torch.testing.assert_close(predicted, state)
        loss.backward()
        self.assertGreater(float(net.amplitude.grad), 0.)  # Descent reduces the harmful raw flux.
        self.assertEqual(float(diagnostics['applied_rate_loss']), 0.)
        self.assertGreater(float(diagnostics['raw_rate_loss']), 0.)

    def test_local_limiter_configuration_and_legacy_defaults(self):
        for weight in (-1., float('inf'), float('nan'), True):
            cfg = copy.deepcopy(self.cfg)
            cfg['loss']['flux_raw_rate_weight'] = weight
            with self.assertRaisesRegex(ValueError, 'flux_raw_rate_weight'):
                validate_config(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg['train']['flux_limiter'] = 'typo'
        with self.assertRaisesRegex(ValueError, 'flux_limiter'):
            validate_config(cfg)
        cfg['train']['flux_limiter'] = 'local'
        cfg['loss']['flux_raw_rate_weight'] = 0.
        with self.assertRaisesRegex(ValueError, 'flux_raw_rate_weight > 0'):
            validate_config(cfg)
        cfg['train'].pop('flux_limiter')
        cfg['loss'].pop('flux_raw_rate_weight')
        cfg['features']['build'].update(include_ramp_angle=False, include_signed_distance_to_ramp=False)
        validate_config(cfg)
        self.assertEqual(cfg['loss']['flux_raw_rate_weight'], 0.)
        net = build_model(cfg)
        state = torch.tensor([2., .1, .1, 10.]).repeat(7, 1)
        rt = Runtime(cfg, 'cpu')
        source = SimpleNamespace(ramp={})
        try:
            with patch('engine.limit_flux_rate', wraps=limit_flux_rate) as legacy, \
                 patch('engine.limit_face_transports', side_effect=AssertionError('Changed legacy rollout')):
                model_step(net, state, state, self.mesh, source, cfg, rt,
                           torch.zeros(4), torch.ones(4), .1, .1, 0)
                self.assertEqual(legacy.call_count, 1)
        finally:
            rt.close()

    def test_factory_and_config_validation(self):
        validate_config(self.cfg)
        self.assertIsInstance(build_model(self.cfg), FluxGraphNet)
        for key, value in [('open_boundary_sides', ['typo']), ('use_open_boundary_source', 'yes'),
                           ('use_ramp_boundary_source', 1), ('ramp_boundary_source_channels', [4]),
                           ('open_boundary_source_channels', [True])]:
            cfg = copy.deepcopy(self.cfg)
            cfg['model'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_config(cfg)
        self.cfg['loss']['blend_weight'] = .1
        with self.assertRaisesRegex(ValueError, 'blend_weight'):
            validate_config(self.cfg)

    def test_all_ramp_conservation_subsets_mask_wall_flux_only(self):
        m, g = self.mesh, self.geometry()
        for bits in range(16):
            conserved = [name for i, name in enumerate(RAMP_QUANTITIES) if bits & (1 << i)]
            with self.subTest(conserved=conserved):
                self.cfg['model']['ramp_conserved_quantities'] = conserved
                validate_config(self.cfg)
                net = build_model(self.cfg)
                with torch.no_grad():
                    for p in net.parameters():
                        p.zero_()
                    net.open_boundary_head[-1].bias.fill_(2.)
                    net.ramp_boundary_head[-1].bias.fill_(7.)
                x = torch.zeros(len(m.centers), net.node_encoder[0].in_features)
                rate, _, boundary = net.predict_rate_and_flux(x, m.edges, m.centers, g)
                allowed = torch.tensor([0. if name in conserved else 1. for name in RAMP_QUANTITIES])
                lengths = g.boundary_length[:, None] / net.length_scale
                torch.testing.assert_close(boundary[~g.boundary_open], 7 * lengths[~g.boundary_open] * allowed)
                torch.testing.assert_close(boundary[g.boundary_open], 2 * lengths[g.boundary_open].expand(-1, 4))
                # The square has three unit-length open sides and one wall.
                torch.testing.assert_close((rate * g.area[:, None]).sum(0), -(6 + 7*allowed) * net.length_scale)
                rate.square().sum().backward()
                grad = net.ramp_boundary_head[-1].bias.grad
                self.assertTrue(torch.all(grad[allowed == 0] == 0))
                self.assertTrue(torch.all(grad[allowed != 0] != 0))

    def test_named_ramp_conservation_preserves_legacy_weights_and_disable_switch(self):
        self.cfg['model']['ramp_boundary_source_channels'] = [1, 2]
        legacy = build_model(self.cfg)
        del self.cfg['model']['ramp_boundary_source_channels']
        self.cfg['model']['ramp_conserved_quantities'] = ['density', 'energy']
        named = build_model(self.cfg)
        named.load_state_dict(legacy.state_dict(), strict=True)
        x = torch.randn(len(self.mesh.centers), named.node_encoder[0].in_features)
        args = (x, self.mesh.edges, self.mesh.centers, self.geometry())
        torch.testing.assert_close(named(*args), legacy(*args), rtol=0, atol=0)
        self.cfg['model'].update(use_ramp_boundary_source=False, ramp_conserved_quantities=[])
        disabled = build_model(self.cfg)
        _, _, boundary = disabled.predict_rate_and_flux(*args)
        self.assertIsNone(disabled.ramp_boundary_head)
        self.assertEqual(int(boundary[~self.geometry().boundary_open].count_nonzero()), 0)

    def test_invalid_or_ambiguous_ramp_conservation_options(self):
        for value in (None, 'energy', [0], ['mass'], ['energy', 'energy'], [['density']]):
            cfg = copy.deepcopy(self.cfg)
            cfg['model']['ramp_conserved_quantities'] = value
            for validate in (validate_config, build_model):
                with self.subTest(value=value, validate=validate.__name__), self.assertRaisesRegex(ValueError, 'ramp_conserved_quantities'):
                    validate(cfg)
        self.cfg['model'].update(ramp_conserved_quantities=[], ramp_boundary_source_channels=[0, 1, 2, 3])
        for validate in (validate_config, build_model):
            with self.assertRaisesRegex(ValueError, 'not both'):
                validate(self.cfg)

    def test_engine_physical_mass_change_matches_exterior_flux(self):
        self.cfg['loss']['parc'] = False
        self.cfg['features']['build'].update(include_ramp_angle=False, include_signed_distance_to_ramp=False)
        self.cfg['train'].update(rate_clip_norm=0., delta_clip_norm=0.)
        net = build_model(self.cfg)
        rt = Runtime(self.cfg, 'cpu')
        state = torch.tensor([10., .2, .3, 100.]).repeat(7, 1)
        mu, sigma = torch.tensor([2., 3., 4., 5.]), torch.tensor([.1, .2, .3, .4])
        captured = []
        original = net.predict_rate_and_flux
        def capture(*args):
            result = original(*args)
            captured.append(result)
            return result
        try:
            with patch.object(net, 'predict_rate_and_flux', side_effect=capture):
                loss, predicted = model_step(net, state, state+1., self.mesh, SimpleNamespace(ramp={}),
                                             self.cfg, rt, mu, sigma, .002, .1, 0)
            _, _, boundary = captured[0]
            expected = -boundary.sum(0) * net.length_scale**2 * sigma * (.002/.1)
            actual = ((predicted-state)*self.geometry().area[:, None]).sum(0)
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-4)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))
        finally:
            rt.close()


class FluxTrainingTests(test_model.MeshGraphNetTrainingTests):
    only_specs = test_cache_only.CacheOnlyTests.only_specs

    def setUp(self):
        super().setUp()
        self.cfg['model']['name'] = 'FluxGraphNet'
        self.cfg['model']['ramp_conserved_quantities'] = ['density', 'energy']

    def test_cache_only_training_without_native_files(self):
        self.tmp.cleanup()
        test_cache_only.CacheOnlyTests.setUp(self)
        self.cfg['model'].update(name='FluxGraphNet', message_passing_steps=2, mlp_layers=1)
        test_cache_only.CacheOnlyTests.exercise_training(self, 'cpu')

    def test_local_history_and_resume_rejects_limiter_change(self):
        import argparse
        from contextlib import redirect_stdout
        import io
        import json
        from train import main
        config = self.root/'config.json'
        config.write_text(json.dumps(self.cfg))
        args = argparse.Namespace(config=str(config), device='cpu', epochs=1,
                                  out_dir=str(self.root/'local'), max_batches=1, dry_run=False, resume=None)
        with redirect_stdout(io.StringIO()):
            main(args)
        payload = torch.load(self.root/'local/last_model.pt', weights_only=False)
        metrics = payload['history'][0]['train']
        flux = metrics['flux']
        self.assertEqual(set(flux), {'raw_rate_rms', 'applied_rate_rms', 'limited_face_fraction',
                                     'raw_rate_loss', 'applied_rate_loss'})
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in flux.values()))
        self.assertTrue(0 <= flux['limited_face_fraction'] <= 1)
        self.assertAlmostEqual(metrics['loss'], flux['applied_rate_loss'] +
                               self.cfg['loss']['flux_raw_rate_weight'] * flux['raw_rate_loss'], places=5)
        self.cfg['train']['flux_limiter'] = 'global'
        config.write_text(json.dumps(self.cfg))
        args.resume, args.epochs = str(self.root/'local/last_model.pt'), 2
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'flux_limiter differs'):
            main(args)


if __name__ == '__main__':
    unittest.main()
