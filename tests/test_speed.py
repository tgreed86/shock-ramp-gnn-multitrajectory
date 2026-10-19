"""Numerical parity and bounded-memory tests for the training fast paths."""
import copy
from unittest.mock import patch
import unittest

import h5py
import torch

import test_runtime
from data import load_trajectories
from engine import build_model, run_epoch
from geometry import dynamic_cells_from_parent_masks
from gt_cache import TensorLRU
from kernels import _RuntimeMultiResGtLookup
from mesh_geometry import build_amr_face_adjacency_edges
from runtime import Mesh, Runtime
from train import numerical_runtime_config, validate_config


class CacheTests(unittest.TestCase):
    setUp = test_runtime.RuntimeTests.setUp
    tearDown = test_runtime.RuntimeTests.tearDown
    values = test_runtime.RuntimeTests.values

    def test_tensor_lru_byte_budget(self):
        cache = TensorLRU(32)
        a = torch.ones(4)
        cache.put('a', a)
        cache.put('b', a)
        self.assertEqual(cache.bytes, 32)
        self.assertIs(cache.get('a'), a)
        cache.put('c', a)
        self.assertIsNone(cache.get('b'))
        cache.put('a', torch.ones(2))
        self.assertEqual(cache.bytes, 24)
        cache.put('too_large', torch.ones(20))
        self.assertNotIn('too_large', cache.entries)
        self.assertLessEqual(cache.bytes, cache.max_bytes)
        cache.clear()
        self.assertEqual(cache.bytes, 0)
        disabled = TensorLRU(0)
        disabled.put('a', a)
        self.assertFalse(disabled.entries)

    def test_canonical_time_plan_reuse_and_reader_eviction(self):
        sources = load_trajectories(self.cfg)
        rt = Runtime(self.cfg, 'cpu')
        mesh = Mesh(self.centers[0], torch.zeros(16, dtype=torch.long), None, None, None, None)
        try:
            a, _ = rt.gt(sources[0], 3, mesh, initial=True)
            reader = rt.lookup(sources[0])
            self.assertEqual(reader.disk_reads, 1)
            with patch.object(reader, 'prepare', side_effect=AssertionError('plan rebuilt')):
                b, _ = rt.gt(sources[0], 3, mesh)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(reader.disk_reads, 1)
            rt.gt(sources[0], 4, mesh)
            self.assertEqual(reader.disk_reads, 2)
            different, _ = rt.gt(sources[1], 3, mesh)
            self.assertFalse(torch.equal(a, different))
            again, _ = rt.gt(sources[0], 3, mesh)
            torch.testing.assert_close(a, again, rtol=0, atol=0)
            self.assertEqual(rt.lookup(sources[0]).disk_reads, 0)
        finally:
            rt.close()
        self.assertEqual(rt.cpu_features.bytes, 0)
        self.assertEqual(rt.cuda_features.bytes, 0)

    def exercise_lookup_parity(self, device):
        sources = load_trajectories(self.cfg)
        # A cache boundary hole: L0 cell 0 does not exist in the selected file.
        with h5py.File(sources[0].cache_files[0], 'r+') as f:
            f['static/pred_levels'][0] = 1
        cpu = _RuntimeMultiResGtLookup(level_files=sources[0].cache_files, H=4, W=4,
                    bbox=(0,1,0,1), refine_ratio=2, max_level=1)
        rt = Runtime(self.cfg, device)
        c = torch.cat((self.centers[0][:5], self.centers[1][:9])).to(device)
        levels = torch.tensor([0]*5 + [1]*9, device=device)
        reader = rt.lookup(sources[0])
        plan = reader.prepare(c, levels, device=device)
        try:
            for initial, local_t in ((True, 0), (True, 3), (False, 3), (False, 6)):
                ref, matched, _ = cpu.lookup(t_dst=local_t+1 if initial else local_t,
                    dataset_name='feat_t_on_pred' if initial else 'feat_tp1_on_pred',
                    pred_centers=c.cpu(), pred_levels=levels.cpu())
                actual = reader.gather(local_t, plan, out_device=device).cpu()
                torch.testing.assert_close(plan.missing.cpu(), (~matched).nonzero(as_tuple=True)[0])
                torch.testing.assert_close(actual[matched], ref[matched], rtol=0, atol=0)
                self.assertTrue((actual[~matched] == 0).all())
            self.assertEqual(reader.disk_reads, 6)  # Three physical times, two file levels.
            # Missing cells take the same explicit IDW path, for initial and target states.
            mesh = Mesh(c, levels, None, None, None, None)
            values, missing = rt.gt(sources[0], 3, mesh)
            expected = rt.idw(*sources[0].snapshot(3, self.cfg, rt.device)[:2], c[plan.missing])
            self.assertEqual(missing, 1)
            torch.testing.assert_close(values[plan.missing], expected)
        finally:
            cpu.close()
            rt.close()

    def test_lookup_matches_reference(self):
        self.exercise_lookup_parity('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_lookup_matches_reference(self):
        self.exercise_lookup_parity('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_cache_hit_without_host_features(self):
        sources = load_trajectories(self.cfg)
        rt = Runtime(self.cfg, 'cuda')
        try:
            reader = rt.lookup(sources[0])
            plan = reader.prepare(self.centers[0].cuda(), torch.zeros(16, device='cuda'), device='cuda')
            ref = reader.gather(3, plan, out_device='cuda')
            rt.cpu_features.clear()
            with patch.object(rt.cpu_features, 'get', side_effect=AssertionError('CPU read')):
                actual = reader.gather(3, plan, out_device='cuda')
            torch.testing.assert_close(ref, actual, rtol=0, atol=0)
            self.assertLessEqual(rt.cuda_features.bytes, rt.cuda_features.max_bytes)
        finally:
            rt.close()

    def test_global_cache_budget_across_sources(self):
        self.cfg['train']['runtime_mesh']['multires_gt_lookup'].update(cpu_cache_mb=512/1024**2)
        sources = load_trajectories(self.cfg)
        rt = Runtime(self.cfg, 'cpu')
        mesh = Mesh(self.centers[0], torch.zeros(16, dtype=torch.long), None, None, None, None)
        try:
            for sid, time in ((0, 1), (1, 1), (0, 2), (1, 3), (0, 1)):
                value, _ = rt.gt(sources[sid], time, mesh)
                torch.testing.assert_close(value, self.values(sid, time, 0))
                self.assertLessEqual(rt.cpu_features.bytes, 512)
                self.assertLessEqual(len(rt.cpu_features.entries), 2)
        finally:
            rt.close()

    def test_reuse_unchanged_mesh_and_gradients(self):
        source = load_trajectories(self.cfg)[0]
        rt = Runtime(self.cfg, 'cpu')
        c, x, levels = source.snapshot(0, self.cfg, rt.device)
        try:
            mesh = rt.predict_mesh(source, c, x, levels, 0)
            with patch('runtime.dynamic_cells_from_parent_masks', side_effect=AssertionError('rebuilt')):
                same = rt.predict_mesh(source, c, x, levels, 0, previous_mesh=mesh)
            self.assertIs(mesh, same)
            state = torch.randn(len(mesh.centers), 4, requires_grad=True)
            mapped = rt.remap(mesh, state, same)
            self.assertIs(mapped, state)
            torch.testing.assert_close(torch.autograd.grad(mapped.sum(), state)[0], torch.ones_like(state))
            rt.rt['reuse_unchanged_mesh'] = False
            other = rt.predict_mesh(source, c, x, levels, 0, previous_mesh=mesh)
            self.assertIsNot(other, mesh)
            torch.testing.assert_close(other.edges, mesh.edges, rtol=0, atol=0)
        finally:
            rt.close()

    def test_training_tensor_vs_cpu_adjacency(self):
        sources = load_trajectories(self.cfg)
        weights = build_model(self.cfg).state_dict()
        runs = []
        for backend in ('cpu', 'tensor'):
            cfg = copy.deepcopy(self.cfg)
            cfg['train']['runtime_mesh']['adjacency_backend'] = backend
            cfg['train']['runtime_mesh']['reuse_unchanged_mesh'] = backend == 'tensor'
            net = build_model(cfg)
            net.load_state_dict(weights)
            opt = torch.optim.AdamW(net.parameters(), lr=.001)
            rt = Runtime(cfg, 'cpu')
            try:
                stats = run_epoch(net, [(0,0), (1,0)], sources, cfg, rt, torch.zeros(4), torch.ones(4),
                                  .1, optimizer=opt)
                runs.append((stats, [p.detach().clone() for p in net.parameters()]))
            finally:
                rt.close()
        self.assertEqual(runs[0][0]['loss'], runs[1][0]['loss'])
        for a, b in zip(runs[0][1], runs[1][1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_config_validation_and_resume_options(self):
        original = copy.deepcopy(self.cfg['train']['runtime_mesh'])
        altered = copy.deepcopy(original)
        altered.update(adjacency_backend='cpu', reuse_unchanged_mesh=False)
        altered['multires_gt_lookup'].update(cpu_cache_mb=0, cuda_cache_mb=0, backend='cpu')
        self.assertEqual(numerical_runtime_config(original), numerical_runtime_config(altered))
        altered['update_every_steps'] += 1
        self.assertNotEqual(numerical_runtime_config(original), numerical_runtime_config(altered))
        for key in ('cpu_cache_mb', 'cuda_cache_mb'):
            for value in (-1, float('nan'), float('inf'), 'all', True):
                cfg = copy.deepcopy(self.cfg)
                cfg['train']['runtime_mesh']['multires_gt_lookup'][key] = value
                with self.assertRaisesRegex(ValueError, key):
                    validate_config(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg['train']['runtime_mesh']['multires_gt_lookup']['max_cached_feature_steps'] = 4
        with self.assertWarnsRegex(FutureWarning, 'obsolete'):
            validate_config(cfg)
        cfg['train']['runtime_mesh']['multires_gt_lookup']['backend'] = 'cuda'
        with self.assertRaisesRegex(ValueError, 'CUDA training device'):
            Runtime(cfg, 'cpu')


class AdjacencyTests(unittest.TestCase):
    def compare(self, centers, levels, h, w, bbox, device):
        expected = build_amr_face_adjacency_edges(centers, levels, h, w, bbox, backend='cpu')
        actual = build_amr_face_adjacency_edges(centers.to(device), levels.to(device), h, w, bbox, backend='tensor')
        torch.testing.assert_close(actual[0].cpu(), expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1].cpu(), expected[1])
        no_attrs = build_amr_face_adjacency_edges(centers.to(device), levels.to(device), h, w, bbox,
                                                backend='tensor', return_edge_attr=False)
        torch.testing.assert_close(no_attrs.cpu(), expected[0], rtol=0, atol=0)

    def exercise(self, device):
        torch.set_num_threads(1)
        for seed in range(8):
            gen = torch.Generator().manual_seed(seed)
            h, w = 4, 7
            masks = {1: torch.rand((h, w), generator=gen) < .5}
            for level in (2, 3):
                parent = masks[level-1].repeat_interleave(2, 0).repeat_interleave(2, 1)
                masks[level] = parent & (torch.rand(parent.shape, generator=gen) < .6)
            bbox = (-.5, 1.5, .25, 1.75)
            c, lv, _, _ = dynamic_cells_from_parent_masks(masks, h, w, *bbox)
            # Remove cells to model a wedge boundary and disconnected/empty regions.
            keep = c[:, 1] > .2*c[:, 0] + .4
            c, lv = c[keep], lv[keep]
            order = torch.randperm(len(c), generator=gen)
            self.compare(c[order], lv[order], h, w, bbox, device)
        for count in (0, 1):
            self.compare(torch.tensor([[.5,.5]])[:count], torch.zeros(count, dtype=torch.long),
                         1, 1, (0,1,0,1), device)

    def test_tensor_adjacency_parity(self):
        self.exercise('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_adjacency_parity(self):
        self.exercise('cuda')


if __name__ == '__main__':
    unittest.main()
