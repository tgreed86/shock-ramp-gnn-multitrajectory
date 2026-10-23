"""AMR transport restriction, time/unit conventions, loss gradients and resume."""
import argparse
import copy
from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import h5py
import torch

import test_cache_only
from test_fluxgraphnet import mesh_for
from data import load_trajectories
from engine import build_model, model_step, run_epoch
from flux import leaf_owners
from flux_supervision import restriction_plan, supervised_flux_loss
from geometry import dynamic_cells_from_parent_masks
from precompute_face_transports import divergence_rate, export_all, export_source
from runtime import Runtime
from train import load_config, main, validate_config


class FluxSupervisionTests(unittest.TestCase):
    tearDown = test_cache_only.CacheOnlyTests.tearDown
    values = test_cache_only.CacheOnlyTests.values

    def setUp(self):
        test_cache_only.CacheOnlyTests.setUp(self)
        self.cfg['model'].update(name='FluxGraphNet', message_passing_steps=1)
        self.cfg['loss'].update(parc=False, flux_supervision_weight=.05)
        self.cfg['data'].update(flux_cache_root=str(self.root/'flux'),
                                excluded_trajectories=['p-1p0_angle-30'])
        self.cfg['split'].update(train=.4, val=.4, test=.2)
        metadata = json.loads(self.times_file.read_text())
        metadata['trajectories']['p-1p0_angle-20']['times'] = [0., .1, .3, .6, 1., 1.5, 2.1]
        self.times_file.write_text(json.dumps(metadata))
        self.path = export_all(self.cfg, self.root/'flux', trajectories=['p-1p0_angle-20'])[0]
        self.source = load_trajectories(self.cfg)[0]

    def mesh(self):
        mask = torch.zeros(4, 4, dtype=torch.bool)
        mask[1, 1] = True
        centers, levels, _, _ = dynamic_cells_from_parent_masks({1: mask}, 4, 4, 0., 1., 0., 1.)
        return mesh_for(centers, levels, H=4, W=4)

    def test_amr_restriction_preserves_every_cell_balance_and_orientation(self):
        runtime = Runtime(self.cfg, 'cpu')
        try:
            mesh = self.mesh()
            g = runtime.flux_geometry(mesh, self.source)
            _, handle, faces = runtime.flux_targets_reader().reader(self.source)
            owners = runtime.remap_owners(mesh)
            ids, targets, count = restriction_plan(faces, mesh, g, owners, self.cfg)
            # Arbitrary independent transports expose orientation or cancellation
            # errors that uniform analytic flow could hide.
            torch.manual_seed(9)
            fine = torch.randn(len(faces.owner), 4, dtype=torch.float64)
            coarse = torch.zeros(count, 4, dtype=torch.float64).index_add_(0, targets, fine[ids])
            a, b = mesh.edges[:, g.forward_edges]
            n = len(a)
            balance = torch.zeros(len(mesh.centers), 4, dtype=torch.float64)
            balance.index_add_(0, a, -coarse[:n])
            balance.index_add_(0, b, coarse[:n])
            balance.index_add_(0, g.boundary_owner, -coarse[n:])
            fine_owners = leaf_owners(faces.centers, torch.ones(len(faces.centers), dtype=torch.long),
                H=4, W=4, bbox=[0., 1., 0., 1.], max_level=1)
            reference = torch.zeros_like(balance).index_add_(0, owners,
                (divergence_rate(fine, faces) * faces.area[:, None])[fine_owners])
            torch.testing.assert_close(balance, reference, rtol=1e-12, atol=1e-12)
            self.assertLess(len(ids), len(fine))  # Internal-to-coarse faces cancelled.
            self.assertGreater(int(torch.bincount(targets).max()), 1)  # Actual sums, not averages.

            sigma, dt_ref, length = torch.tensor([2., 3., 4., 5.]), .3, .25
            for t in (0, 1, 4):
                cached = torch.from_numpy(handle['interval_mean_transport'][t])
                summed = torch.zeros(count, 4).index_add_(0, targets, cached[ids])
                expected = summed * dt_ref / (sigma*length**2)
                actual = runtime.flux_targets(self.source, t, mesh, sigma, dt_ref, length)
                torch.testing.assert_close(actual, expected)
            # Reuse the mapping and CPU values for another normalization.
            with patch('flux_supervision.restriction_plan', side_effect=AssertionError('Rebuilt mapping')):
                actual = runtime.flux_targets(self.source, 4, mesh, sigma, dt_ref*2, length)
                torch.testing.assert_close(actual, expected*2)
            bad_owners = owners.clone()
            bad_owners[0] = -1
            with self.assertRaisesRegex(ValueError, 'identical.*domains'):
                restriction_plan(faces, mesh, g, bad_owners, self.cfg)
        finally:
            runtime.close()

    def test_raw_loss_is_added_without_changing_prediction_and_backpropagates(self):
        runtime = Runtime(self.cfg, 'cpu')
        try:
            mesh = self.mesh()
            runtime.flux_geometry(mesh, self.source)
            state, _ = runtime.gt(self.source, 2, mesh)
            target, _ = runtime.gt(self.source, 3, mesh)
            model = build_model(self.cfg)
            sigma, mu = torch.ones(4), torch.zeros(4)
            cfg_without = copy.deepcopy(self.cfg)
            cfg_without['loss']['flux_supervision_weight'] = 0.
            for limiter in ('local', 'global'):
                self.cfg['train']['flux_limiter'] = cfg_without['train']['flux_limiter'] = limiter
                args = (model, state, target, mesh, self.source)
                base, pred0 = model_step(*args, cfg_without, runtime, mu, sigma, .3, .1, 0)
                captured = []
                original = model.predict_rate_and_flux
                def capture(*a):
                    result = original(*a)
                    captured.append(result)
                    return result
                diagnostics = {}
                with patch.object(model, 'predict_rate_and_flux', side_effect=capture):
                    total, pred = model_step(*args, self.cfg, runtime, mu, sigma, .3, .1, 0,
                                             local_t=2, diagnostics=diagnostics)
                _, interior, exterior = captured[0]
                truth = runtime.flux_targets(self.source, 2, mesh, sigma, .1, model.length_scale)
                expected = supervised_flux_loss(interior, exterior, truth, mesh.flux_geometry, model)
                torch.testing.assert_close(total, base + .05*expected)
                torch.testing.assert_close(pred, pred0)
                torch.testing.assert_close(diagnostics['supervised_flux_loss'], expected.detach())
                model.zero_grad()
                expected.backward()
                for head in (model.decoder, model.open_boundary_head, model.ramp_boundary_head):
                    grads = [p.grad for p in head.parameters() if p.grad is not None]
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                    self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)
            with patch.object(runtime, 'flux_targets', side_effect=AssertionError('Read targets during rollout')):
                loss, _ = model_step(model, state, None, mesh, self.source, self.cfg, runtime,
                                     mu, sigma, .3, .1, 0)
                self.assertIsNone(loss)
                model_step(model, state, target, mesh, self.source, cfg_without, runtime,
                           mu, sigma, .3, .1, 0)
        finally:
            runtime.close()

    def test_hard_boundary_masks_excluded(self):
        runtime = Runtime(self.cfg, 'cpu')
        try:
            mesh = self.mesh()
            g = runtime.flux_geometry(mesh, self.source)
            model = build_model(self.cfg)
            ni, nb = len(g.forward_edges), len(g.boundary_owner)
            interior = torch.full((ni, 4), 100., requires_grad=True)
            exterior = torch.zeros(nb, 4, requires_grad=True)
            truth = torch.zeros(ni+nb, 4)
            wall = (~g.boundary_open).nonzero(as_tuple=True)[0] + ni
            truth[wall, 0] = truth[wall, 3] = 1e4
            expected = (interior.square().sum() /
                        (ni*4 + int(g.boundary_open.sum())*4 + len(wall)*2))
            loss = supervised_flux_loss(interior, exterior, truth, g, model)
            torch.testing.assert_close(loss, expected)
            loss.backward()
            self.assertTrue((interior.grad > 0).all())
            self.assertEqual(float(exterior.grad.abs().sum()), 0.)
            model.open_boundary_head = model.ramp_boundary_head = None
            torch.testing.assert_close(supervised_flux_loss(interior, exterior, truth, g, model),
                                       interior.square().mean())
        finally:
            runtime.close()

    def test_direct_supervision_trains_a_completely_limited_face(self):
        self.cfg['loss']['flux_raw_rate_weight'] = 0.
        g = SimpleNamespace(area=torch.ones(2), forward_edges=torch.arange(1),
            boundary_owner=torch.empty(0, dtype=torch.long), boundary_open=torch.empty(0, dtype=torch.bool))
        mesh = SimpleNamespace(edges=torch.tensor([[0], [1]]),
            centers=torch.tensor([[.25, .25], [.75, .25]]), levels=torch.zeros(2, dtype=torch.long))
        state = torch.tensor([[2., 0., 0., 1e-6], [2., 0., 0., 5.]])
        class SaturatedFlux(torch.nn.Module):
            length_scale = 1.
            open_boundary_head = ramp_boundary_head = None
            open_boundary_mask = ramp_boundary_mask = torch.zeros(4)
            def __init__(self):
                super().__init__()
                self.amplitude = torch.nn.Parameter(torch.tensor(100.))
            def predict_rate_and_flux(self, *args):
                flux = self.amplitude * torch.tensor([[0., 0., 0., 1.]])
                return torch.cat([-flux, flux]), flux, flux.new_empty(0, 4)
        model = SaturatedFlux()
        runtime = SimpleNamespace(flux_geometry=lambda mesh: g, flux_targets=lambda *a: torch.zeros(1, 4))
        diagnostics = {}
        loss, predicted = model_step(model, state, state, mesh, self.source, self.cfg, runtime,
            torch.zeros(4), torch.ones(4), .1, .1, 0, local_t=3, diagnostics=diagnostics)
        torch.testing.assert_close(predicted, state)
        self.assertEqual(float(diagnostics['applied_rate_loss']), 0.)
        loss.backward()
        self.assertGreater(float(model.amplitude.grad), 0.)

    def test_rollout_windows_use_absolute_interval_indices(self):
        runtime = Runtime(self.cfg, 'cpu')
        try:
            calls = []
            original = runtime.flux_targets
            def targets(source, local_t, *args):
                calls.append((source.name, local_t))
                return original(source, local_t, *args)
            with patch.object(runtime, 'flux_targets', side_effect=targets):
                run_epoch(build_model(self.cfg), [(0, 2)], [self.source], self.cfg, runtime,
                          torch.zeros(4), torch.ones(4), .1)
            self.assertEqual(calls, [(self.source.name, 2), (self.source.name, 3)])
        finally:
            runtime.close()

    def test_native_sources_can_use_matching_flux_targets(self):
        (self.root/'unavailable_native_data').rename(self.root/'data')
        self.cfg['data'].update(source='native', root=str(self.root/'data'),
                                flux_cache_root=str(self.root/'native_flux'))
        source = load_trajectories(self.cfg)[0]
        export_source(source, self.cfg, self.root/'native_flux'/source.name/'face_transports_L1.h5',
                      gamma=1.4, rho_floor=1e-6, p_floor=0.)
        runtime = Runtime(self.cfg, 'cpu')
        try:
            manifest = runtime.prepare_flux_supervision([source])
            self.assertEqual(set(manifest['files']), {source.name})
            mesh = self.mesh()
            runtime.flux_geometry(mesh, source)
            values = runtime.flux_targets(source, 2, mesh, torch.ones(4), .1, .25)
            self.assertTrue(torch.isfinite(values).all())
        finally:
            runtime.close()

    def test_validation_rejects_partial_stale_and_invalid_caches(self):
        for key, value in [('snapshots', 2), ('trajectory', 'wrong'), ('cache_sha256', 'stale'),
                           ('mesh_spec_sha256', 'stale'), ('timestamps_sha256', 'stale'),
                           ('orientation', 'reverse')]:
            with h5py.File(self.path, 'r+') as f:
                original = f.attrs['provenance_json']
                p = json.loads(original)
                p[key] = value
                f.attrs['provenance_json'] = json.dumps(p)
            runtime = Runtime(self.cfg, 'cpu')
            try:
                with self.assertRaisesRegex(ValueError, key):
                    runtime.prepare_flux_supervision([self.source])
            finally:
                runtime.close()
            with h5py.File(self.path, 'r+') as f:
                f.attrs['provenance_json'] = original
        with h5py.File(self.path, 'r+') as f:
            f['geometry/normal'][0] *= -1
        runtime = Runtime(self.cfg, 'cpu')
        try:
            with self.assertRaisesRegex(ValueError, 'geometry/normal'):
                runtime.prepare_flux_supervision([self.source])
        finally:
            runtime.close()

    def test_enabled_training_validation_test_resume_and_relocation(self):
        config = self.root/'config.json'
        config.write_text(json.dumps(self.cfg))
        args = argparse.Namespace(config=str(config), device='cpu', epochs=1, out_dir=str(self.root/'run'),
                                  max_batches=1, dry_run=True, resume=None)
        with redirect_stdout(io.StringIO()):
            main(args)
        self.assertFalse((self.root/'run').exists())
        args.dry_run = False
        with redirect_stdout(io.StringIO()) as output:
            main(args)
        self.assertEqual(len(output.getvalue().splitlines()), 1)  # Epoch-only logging.
        checkpoint = self.root/'run/last_model.pt'
        saved = torch.load(checkpoint, weights_only=False)
        self.assertEqual(set(saved['flux_supervision_manifest']['files']), {self.source.name})
        for metrics in (saved['history'][0]['train'], saved['history'][0]['val'],
                        json.loads((self.root/'run/test_metrics.json').read_text())):
            flux = metrics['flux']
            self.assertGreater(flux['supervised_flux_loss'], 0)
            self.assertAlmostEqual(metrics['loss'], flux['applied_rate_loss'] +
                .1*flux['raw_rate_loss'] + .05*flux['supervised_flux_loss'], places=3)
        (self.root/'flux').rename(self.root/'relocated_flux')
        self.cfg['data']['flux_cache_root'] = './relocated_flux'
        config.write_text(json.dumps(self.cfg))
        self.assertEqual(load_config(config)['data']['flux_cache_root'], str((self.root/'relocated_flux').resolve()))
        args.resume, args.epochs = str(checkpoint), 2
        with redirect_stdout(io.StringIO()):
            main(args)
        saved2 = torch.load(checkpoint, weights_only=False)
        self.assertEqual(saved2['epoch'], 2)
        self.assertEqual(saved2['flux_supervision_manifest'], saved['flux_supervision_manifest'])
        new_path = self.root/'relocated_flux'/self.source.name/self.path.name
        with h5py.File(new_path, 'r+') as f:
            f['interval_mean_transport'][0, 0, 0] += 1
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'targets differ'):
            main(args)

    def test_disabled_old_checkpoint_compatibility_and_option_validation(self):
        for weight in (-1, float('nan'), float('inf'), True):
            cfg = copy.deepcopy(self.cfg)
            cfg['loss']['flux_supervision_weight'] = weight
            with self.assertRaisesRegex(ValueError, 'flux_supervision_weight'):
                validate_config(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg['model']['name'] = 'MeshGraphNet'
        with self.assertRaisesRegex(ValueError, 'requires model.name=FluxGraphNet'):
            validate_config(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg['loss']['flux_raw_rate_weight'] = 0
        validate_config(cfg)  # Direct flux supervision also trains limited faces.
        self.cfg['loss'].pop('flux_supervision_weight')
        self.cfg['data'].pop('flux_cache_root')
        validate_config(self.cfg)
        self.assertEqual(self.cfg['loss']['flux_supervision_weight'], 0.)
        config = self.root/'legacy.json'
        config.write_text(json.dumps(self.cfg))
        args = argparse.Namespace(config=str(config), device='cpu', epochs=1, out_dir=str(self.root/'oldrun'),
                                  max_batches=1, dry_run=False, resume=None)
        with patch('flux_supervision.FluxSupervision.reader', side_effect=AssertionError('Opened disabled targets')):
            with redirect_stdout(io.StringIO()):
                main(args)
            checkpoint = self.root/'oldrun/last_model.pt'
            saved = torch.load(checkpoint, weights_only=False)
            saved.pop('flux_supervision_manifest')
            saved['config']['loss'].pop('flux_supervision_weight')
            torch.save(saved, checkpoint)
            args.epochs, args.resume = 2, str(checkpoint)
            with redirect_stdout(io.StringIO()):
                main(args)


if __name__ == '__main__':
    unittest.main()
