import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from data import discover, load_trajectories, normalization, split_windows, validate_cache
from engine import build_model, run_epoch
from geometry import apply_idw_map, build_idw_map
from runtime import Mesh, Runtime
from train import load_config, validate_config


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = load_config(Path(__file__).resolve().parents[1]/'config.json')
        self.cfg['data'].update(root=str(self.root/'data'), H=4, W=4, window_size=3, excluded_trajectories=[])
        self.cfg['mesh']['starting_mesh_path'] = str(self.root/'specs')
        self.cfg['policy']['max_level'] = 1
        self.cfg['policy']['dilate_cells_lmax'] = 0
        self.cfg['model'].update(hidden=8, layers=2, dropout=0.)
        self.cfg['debug']['progress_every_batches'] = 0
        self.cfg['debug']['progress_bar'] = False
        self.cfg['features']['build'].update(include_ramp_angle=False, include_signed_distance_to_ramp=False)
        lookup = self.cfg['train']['runtime_mesh']['multires_gt_lookup']
        lookup.update(root=str(self.root/'cache'), max_open_trajectories=1, cpu_cache_mb=1, cuda_cache_mb=1)
        (self.root/'specs').mkdir()
        self.centers = {}
        for level in (0, 1):
            size = 4 * 2**level
            y, x = torch.meshgrid((torch.arange(size)+.5)/size, (torch.arange(size)+.5)/size, indexing='ij')
            self.centers[level] = torch.stack([x.flatten(), y.flatten()], 1)
        for sid, angle in enumerate([20, 30]):
            name = f'p-1p0_angle-{angle}'
            path = self.root/'data'/'p_1.0'/f'angle_{angle}'/f'{name}.pt'
            path.parent.mkdir(parents=True)
            series = [{'pos': self.centers[1], 'x': self.values(sid, t, 1), 'level': torch.ones(64, dtype=torch.long),
                       'time': t*.1} for t in range(7)]
            torch.save(series, path)
            spec = self.root/'specs'/f'wedge_mesh_spec_4x4_L1_ramp-angle-{angle}.pt'
            bottom = 0. if sid == 0 else .4
            torch.save({'n0_x':4, 'n0_y':4, 'max_level':1,
                'outer_polygon_ccw':torch.tensor([[0.,0.], [1.,bottom], [1.,1.], [0.,1.], [0.,0.]])}, spec)
            cache = self.root/'cache'/name/'uniform_meshes'
            cache.mkdir(parents=True)
            for level in (0, 1):
                with h5py.File(cache/f'shock_ramp_precomp_L{level}_uniform_refine.h5', 'w') as f:
                    meta = f.create_group('meta')
                    for key, value in dict(H=4, W=4, T=7, bbox=[0.,1.,0.,1.], refine_ratio=2, starting_refine_to_level=level).items():
                        meta.attrs[key] = value
                    meta.attrs['uniform_gt_provenance'] = json.dumps(dict(source_path=str(path), source_size=path.stat().st_size,
                        mesh_spec_sha256=hashlib.sha256(spec.read_bytes()).hexdigest(), max_timesteps=None))
                    meta.create_dataset('cfg_json', data=np.bytes_(json.dumps(self.cfg)))
                    static = f.create_group('static')
                    static['pred_centers'] = self.centers[level].numpy()
                    static['pred_levels'] = np.full(len(self.centers[level]), level, dtype=np.int64)
                    for t in range(1, 7):
                        g = f.create_group(f't{t:05d}')
                        g['feat_t_on_pred'] = self.values(sid, t-1, level).numpy()
                        g['feat_tp1_on_pred'] = self.values(sid, t, level).numpy()

    def values(self, sid, t, level):
        c = self.centers[level]
        return torch.stack([1 + .1*c[:,0] + .01*t + sid, .2*c[:,0], .1*c[:,1],
                            10 + c[:,1] + t + 20*sid], dim=1)

    def tearDown(self):
        self.tmp.cleanup()

    def test_exclusions_before_loading(self):
        self.cfg['data']['excluded_trajectories'] = ['p-1p0_angle-30']
        # This excluded cache can be invalid without affecting an included trajectory.
        invalid = self.root/'cache'/'p-1p0_angle-30'/'uniform_meshes'/'shock_ramp_precomp_L0_uniform_refine.h5'
        invalid.write_text('not an HDF5 file')
        sources = load_trajectories(self.cfg)
        self.assertEqual([s.name for s in sources], ['p-1p0_angle-20'])
        self.cfg['data']['excluded_trajectories'] = ['typo']
        with self.assertRaisesRegex(ValueError, 'Unknown excluded'):
            discover(self.cfg)
        self.cfg['data']['excluded_trajectories'] = ['p-1p0_angle-20', 'p-1p0_angle-30']
        with self.assertRaisesRegex(ValueError, 'All trajectories'):
            discover(self.cfg)

    def test_window_split_matches_baseline(self):
        sources = load_trajectories(self.cfg)
        for stride in (1, 2, 3):
            self.cfg['data']['stride'] = stride
            splits = split_windows(sources, self.cfg)
            ids = np.repeat([0, 1], 7)
            starts = np.arange(0, 14-3+1, stride)
            starts = starts[ids[starts] == ids[starts+2]]
            starts = starts[np.random.default_rng(self.cfg['seed']).permutation(len(starts))]
            nt, nv = round(.85*len(starts)), round(.1*len(starts))
            for name, group in [('train', starts[:nt]), ('val', starts[nt:nt+nv]), ('test', starts[nt+nv:])]:
                self.assertEqual(splits[name], [(int(ids[t]), int(t % 7)) for t in group])

    def test_normalization_is_train_window_weighted(self):
        sources = load_trajectories(self.cfg)
        windows = [(0,0), (0,1)]
        mu, sigma = normalization(sources, windows, self.cfg)
        values = torch.cat([sources[s].series[t+j]['x'] for s,t in windows for j in range(3)]).double()
        torch.testing.assert_close(mu, values.mean(0).float())
        torch.testing.assert_close(sigma, torch.sqrt(values.var(0)+1e-8).float())

    def test_local_time_lookup_and_lru(self):
        sources = load_trajectories(self.cfg)
        rt = Runtime(self.cfg, 'cpu')
        mesh = Mesh(self.centers[0], torch.zeros(16, dtype=torch.long), None, None, None, None)
        try:
            for sid in [0, 1, 0, 1]:
                for initial, t in [(True,0), (True,3), (False,1), (False,6)]:
                    result, misses = rt.gt(sources[sid], t, mesh, initial=initial)
                    torch.testing.assert_close(result, self.values(sid, t, 0))
                    self.assertEqual(misses, 0)
                self.assertEqual(len(rt.lookups), 1)
            with self.assertRaises(IndexError):
                rt.gt(sources[0], 7, mesh)
        finally:
            rt.close()

    def test_wrong_source_cache_rejected(self):
        sources = load_trajectories(self.cfg)
        sources[0].cache_files = sources[1].cache_files
        with self.assertRaisesRegex(ValueError, 'trajectory'):
            validate_cache(sources[0], self.cfg)

    def test_idw_gradients(self):
        src = torch.tensor([[0.,0.], [1.,1.]])
        features = torch.tensor([[1.,2.], [3.,4.]], requires_grad=True)
        for backend in ('exact', 'kdtree'):
            idx, weights = build_idw_map(torch.tensor([[.5,.5]]), src, k=2, backend=backend)
            result = apply_idw_map(idx, weights, features)
            gradient, = torch.autograd.grad(result.sum(), features)
            torch.testing.assert_close(gradient, torch.full_like(features, .5))

    def test_remap_gradients_and_fallback(self):
        sources = load_trajectories(self.cfg)
        rt = Runtime(self.cfg, 'cpu')
        c = self.centers[0][:2]
        old = Mesh(c, torch.zeros(2, dtype=torch.long), torch.arange(2), None, None, None)
        query = torch.tensor([[.25,.125]])
        new = Mesh(query, torch.ones(1, dtype=torch.long), torch.zeros(1, dtype=torch.long), None, torch.zeros(4,4), None)
        values = torch.ones(2,4, requires_grad=True)
        mapped = rt.remap(old, values, new)
        self.assertTrue(torch.autograd.grad(mapped.sum(), values)[0].abs().sum() > 0)
        # Clear an index deliberately: this tests missing-cell IDW, not corrupt-file fallback.
        rt.lookup(sources[0])._source_file_for_level.clear()
        try:
            values, missing = rt.gt(sources[0], 1, new)
            self.assertEqual(missing, 1)
            self.assertTrue(torch.isfinite(values).all())
            self.cfg['train']['runtime_mesh']['multires_gt_lookup']['fallback_to_idw'] = False
            with self.assertRaisesRegex(RuntimeError, 'cache misses'):
                rt.gt(sources[0], 1, new)
        finally:
            rt.close()

    def exercise_training(self, device, backend='gradient_fast'):
        self.cfg['train']['runtime_mesh'].update(policy_backend=backend, update_every_steps=2)
        self.cfg['data']['window_size'] = 5
        sources = load_trajectories(self.cfg)
        rt = Runtime(self.cfg, device)
        net = build_model(self.cfg).to(device)
        before = [p.detach().clone() for p in net.parameters()]
        opt = torch.optim.AdamW(net.parameters(), lr=.001)
        try:
            result = run_epoch(net, [(0,0), (1,0)], sources, self.cfg, rt,
                    torch.zeros(4, device=device), torch.ones(4, device=device), .1, optimizer=opt)
            self.assertEqual(result['rebuilds'], 4)
            self.assertEqual(result['steps'], 8)
            self.assertEqual(len(rt.geometry), 2)
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before, net.parameters())))
            self.assertTrue(np.isfinite(result['loss']))
        finally:
            rt.close()

    def test_autoregressive_two_geometries(self):
        self.exercise_training('cpu')

    def test_legacy_gradient_predictor(self):
        self.exercise_training('cpu', 'gradient')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_training(self):
        self.exercise_training('cuda')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS is not available')
    def test_mps_training(self):
        self.exercise_training('mps')

    def test_reject_unsupported_training_modes(self):
        self.cfg['train']['runtime_mesh']['enabled'] = False
        with self.assertRaisesRegex(ValueError, 'runtime-only'):
            validate_config(self.cfg)

    def test_progress_bar_output_and_batch_cap(self):
        sources = load_trajectories(self.cfg)
        self.cfg['debug'].update(epoch_only=False, progress_bar=True, progress_bar_mininterval=0., progress_every_batches=1)
        rt = Runtime(self.cfg, 'cpu')
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                result = run_epoch(build_model(self.cfg), [(0,0), (1,0)], sources, self.cfg, rt,
                    torch.zeros(4), torch.ones(4), .1, epoch=2, split='val', max_batches=1)
            self.assertEqual(result['batches'], 1)
            for text in ('Epoch 2 | val', '100%', '1/1', 'loss=', 'mae=', '[BATCH] val 1/1'):
                self.assertIn(text, output.getvalue())
        finally:
            rt.close()

    def test_disabled_and_empty_progress(self):
        sources = load_trajectories(self.cfg)
        rt = Runtime(self.cfg, 'cpu')
        try:
            rt.constraints(sources[0])
            for enabled, windows in [(False, [(0,0)]), (True, [])]:
                self.cfg['debug']['progress_bar'] = enabled
                output = io.StringIO()
                with redirect_stdout(output):
                    result = run_epoch(build_model(self.cfg), windows, sources, self.cfg, rt,
                        torch.zeros(4), torch.ones(4), .1)
                self.assertEqual(result['batches'], len(windows))
                self.assertEqual(output.getvalue(), '')
        finally:
            rt.close()

    def test_invalid_progress_settings(self):
        for value in (-1, float('nan'), float('inf'), 'five', True):
            self.cfg['debug']['progress_bar_mininterval'] = value
            with self.assertRaisesRegex(ValueError, 'progress_bar_mininterval'):
                validate_config(self.cfg)


if __name__ == '__main__':
    unittest.main()
