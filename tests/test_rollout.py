"""Rollout/training parity, exact native geometry, and the five-GIF CLI."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image
import torch

import engine
import kernels as k
import test_runtime
from data import load_trajectories, split_manifest, split_windows
from eval.plot_rollout import main
from eval.rollout import iter_rollout, load_inputs
from eval.rollout_export import ComparisonGrid
from eval.rollout_geometry import mapped_quads, NativeGeometry
from export_cache_times import export_times
from runtime import Runtime
from train import checkpoint_format


def write_map(path, xy, time, level=0):
    """One Chombo node box, with physical x/y stored in separate planes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ny, nx, _ = xy.shape
    with h5py.File(path, 'w') as f:
        g = f.create_group(f'level_{level}')
        g.attrs['time'] = time
        g['boxes'] = np.array([(0, 0, nx-2, ny-2)], dtype=[(s, '<i4') for s in ('lo_i', 'lo_j', 'hi_i', 'hi_j')])
        g['data:offsets=0'] = np.array([0, xy.size])
        g['data:datatype=0'] = xy.transpose(2, 0, 1).reshape(-1)
        g.create_group('data_attributes').attrs['comps'] = 2


class RolloutTests(unittest.TestCase):
    tearDown = test_runtime.RuntimeTests.tearDown
    values = test_runtime.RuntimeTests.values

    def setUp(self):
        test_runtime.RuntimeTests.setUp(self)

    def cache_cfg(self):
        path = self.root/'times.json'
        if not path.exists():
            export_times(self.cfg, path)
        cfg = copy.deepcopy(self.cfg)
        cfg['data'].update(source='uniform_cache', timestamps_path=str(path))
        return cfg

    def test_rollout_matches_training_states_meshes_and_remesh_cadence(self):
        self.cfg['data']['window_size'] = 5
        for cfg in (self.cfg, self.cache_cfg()):
            for name in ('FeatureNet', 'MeshGraphNet', 'FluxGraphNet'):
                for predictor in ('gradient', 'gradient_fast'):
                    for cadence in (1, 2):
                        with self.subTest(mode=cfg['data'].get('source', 'native'), name=name,
                                          predictor=predictor, cadence=cadence):
                            cfg['model'].update(name=name, message_passing_steps=2, mlp_layers=1)
                            cfg['train']['runtime_mesh'].update(policy_backend=predictor, update_every_steps=cadence)
                            sources = load_trajectories(cfg)
                            model = engine.build_model(cfg)
                            mu, sigma = torch.zeros(4), torch.ones(4)
                            expected = []
                            def capture(*args, **kwargs):
                                result = engine_step(*args, **kwargs)
                                mesh = args[3]
                                expected.append((mesh.centers.clone(), mesh.levels.clone(), mesh.edges.clone(),
                                                 engine.carry_state(result[1], cfg).clone()))
                                return result
                            engine_step = engine.model_step
                            rt = Runtime(cfg, 'cpu')
                            try:
                                with patch('engine.model_step', side_effect=capture):
                                    engine.run_epoch(model, [(0, 1)], sources, cfg, rt, mu, sigma, .1)
                            finally:
                                rt.close()
                            rt = Runtime(cfg, 'cpu')
                            real_gt = rt.gt
                            def initial_only(source, index, *args, **kwargs):
                                self.assertEqual(index, 1, 'Rollout read future GT')
                                self.assertTrue(kwargs.get('initial'))
                                return real_gt(source, index, *args, **kwargs)
                            try:
                                with patch.object(rt, 'gt', side_effect=initial_only) as lookup:
                                    actual = list(iter_rollout(model, sources[0], cfg, rt, mu, sigma, .1,
                                                               start=1, steps=4))
                                self.assertEqual(lookup.call_count, 1)
                                for (index, mesh, state), ref in zip(actual[1:], expected):
                                    for value, target in zip((mesh.centers, mesh.levels, mesh.edges, state), ref):
                                        torch.testing.assert_close(value, target, rtol=0, atol=0)
                                self.assertEqual([frame[0] for frame in actual], [1, 2, 3, 4, 5])
                            finally:
                                rt.close()

    def test_continuous_rollout_exceeds_training_window_without_resets(self):
        cfg = self.cache_cfg()
        source = load_trajectories(cfg)[0]
        rt = Runtime(cfg, 'cpu')
        try:
            with patch.object(rt, 'gt', wraps=rt.gt) as lookup:
                frames = list(iter_rollout(engine.build_model(cfg), source, cfg, rt,
                                          torch.zeros(4), torch.ones(4), .1, start=0, steps=6))
            self.assertEqual(len(frames), 7)
            self.assertEqual(lookup.call_count, 1)
            self.assertGreater(6, cfg['data']['window_size'])
            with self.assertRaisesRegex(ValueError, 'snapshot count'):
                list(iter_rollout(engine.build_model(cfg), source, cfg, rt,
                                  torch.zeros(4), torch.ones(4), .1, start=1, steps=6))
        finally:
            rt.close()

    def test_fluxgraphnet_checkpoint_load_and_rollout(self):
        cfg = self.cache_cfg()
        cfg['model'].update(name='FluxGraphNet', message_passing_steps=2, mlp_layers=1)
        checkpoint, sources = self.save_checkpoint(cfg)
        native = self.root/'data/p_1.0/angle_20/p-1p0_angle-20.pt'
        loaded_cfg, model, source, _, mu, sigma, dt_ref, _ = load_inputs(
            checkpoint, sources[0].path, native, device='cpu')
        rt = Runtime(loaded_cfg, 'cpu')
        try:
            frames = list(iter_rollout(model, source, loaded_cfg, rt, mu, sigma, dt_ref, start=0, steps=4))
            self.assertEqual(len(frames), 5)
            self.assertTrue(all(torch.isfinite(state).all() for _, _, state in frames))
            self.assertTrue(all(mesh.flux_geometry is not None for _, mesh, _ in frames))
        finally:
            rt.close()

    def add_native_geometry(self):
        # Add exact map metadata to the existing synthetic native trajectories.
        grid = np.linspace(0., 1., 9)
        y, x = np.meshgrid(grid, grid, indexing='ij')
        xy = np.stack([x, y], axis=-1)
        j, i = torch.meshgrid(torch.arange(8), torch.arange(8), indexing='ij')
        for native in (self.root/'data').glob('p_*/angle_*/*.pt'):
            series = torch.load(native, weights_only=False)
            for t, snapshot in enumerate(series):
                snapshot.update(ij=torch.stack([i.flatten(), j.flatten()], dim=-1),
                                pos_phys=snapshot['pos'].clone(), t=t,
                                bbox=torch.tensor([0., 1., 0., 1.]), bbox_phys=torch.tensor([0., 1., 0., 1.]))
                write_map(native.parent/'plot'/f'DMR.plot.{t:06d}.2d.map.hdf5', xy, snapshot['time'], level=1)
            torch.save(series, native)
            for cache in (self.root/'cache'/native.stem/'uniform_meshes').glob('*.h5'):
                with h5py.File(cache, 'r+') as f:
                    provenance = json.loads(f['meta'].attrs['uniform_gt_provenance'])
                    provenance['source_size'] = native.stat().st_size
                    f['meta'].attrs['uniform_gt_provenance'] = json.dumps(provenance)

    def save_checkpoint(self, cfg, *, omit_first_source=False):
        sources = load_trajectories(cfg)
        manifest = split_manifest(sources, split_windows(sources, cfg), cfg)
        if omit_first_source:
            manifest['sources'] = manifest['sources'][1:]
        path = self.root/'model.pt'
        torch.save(dict(config=cfg, model=engine.build_model(cfg).state_dict(), format=checkpoint_format(cfg),
                        manifest=manifest, mu=torch.zeros(4), sigma=torch.ones(4), dt_ref=.1, epoch=2), path)
        return path, sources

    def test_cli_outputs_five_looping_gifs_and_manifest(self):
        self.add_native_geometry()
        cfg = self.cache_cfg()
        cfg['model'].update(name='MeshGraphNet', message_passing_steps=2, mlp_layers=1)
        checkpoint, sources = self.save_checkpoint(cfg)
        native = self.root/'data/p_1.0/angle_20/p-1p0_angle-20.pt'
        # Only selected family's files and supplied .pt are needed, despite stale training paths.
        cfg_from_checkpoint = torch.load(checkpoint, weights_only=False)
        cfg_from_checkpoint['config']['data']['timestamps_path'] = '/unavailable/timestamps.json'
        torch.save(cfg_from_checkpoint, checkpoint)
        args = ['--checkpoint', str(checkpoint), '--cache-dir', str(sources[0].path),
                '--native-pt', str(native),
                '--steps', '3', '--frame-stride', '2', '--device', 'cpu', '--dpi', '40']
        with redirect_stdout(io.StringIO()):
            output = main(args)
        self.assertEqual(output, checkpoint.resolve().parent/'rollout_plots'/sources[0].name)
        self.assertEqual({p.name for p in output.iterdir()},
                         {'density.gif', 'x_momentum.gif', 'y_momentum.gif', 'energy.gif', 'mesh.gif',
                          'rollout.json', 'rollout_data.h5'})
        for path in output.glob('*.gif'):
            with Image.open(path) as image:
                self.assertEqual(image.n_frames, 3)  # Initial, snapshot 2, final snapshot 3.
                self.assertEqual(image.info['loop'], 0)
                self.assertEqual(image.info['duration'], 100)
        metadata = json.loads((output/'rollout.json').read_text())
        self.assertEqual(metadata['frame_indices'], [0, 2, 3])
        self.assertEqual(metadata['prediction_steps'], 3)
        self.assertEqual(metadata['native_geometry'], 'exact mapped cells')
        self.assertEqual(metadata['data_source'], 'uniform_cache')
        with h5py.File(output/'rollout_data.h5', 'r') as data:
            self.assertEqual(data.attrs['format'], 'shock-ramp-rollout-v1')
            self.assertTrue(data.attrs['complete'])
            np.testing.assert_array_equal(data['rollout_step'][...], [0, 1, 2, 3])
            np.testing.assert_array_equal(data['is_initial'][...], [True, False, False, False])
            np.testing.assert_allclose(data['time'][...], [0., .1, .2, .3])
            np.testing.assert_allclose(data['dt'][...], [0., .1, .1, .1])
            self.assertEqual(data['grid/prediction'].shape, (4, 8, 8, 4))
            self.assertEqual(data['grid/prediction'].compression, 'gzip')
            self.assertEqual(json.loads(data['metadata_json'][()]), metadata)
            for index in range(4):
                expected = self.values(0, index, 1).numpy().reshape(8, 8, 4)
                for reference in ('uniform_gt', 'native_gt'):
                    np.testing.assert_array_equal(data[f'grid/{reference}'][index], expected)
                    self.assertTrue(data[f'grid/{reference}_valid'][index].all())
                valid = data['grid/prediction_valid'][index]
                predicted = data['grid/prediction'][index]
                self.assertTrue(np.isnan(predicted[~valid]).all())
                self.assertTrue(np.isfinite(predicted[valid]).all())
                self.assertTrue(np.isfinite(np.sqrt(((predicted[valid]-expected[valid])**2).mean(0))).all())
                self.assertEqual(data[f'snapshots/{index:06d}'].attrs['rollout_step'], index)
            values = data['grid/uniform_gt'][...]
            expected_range = values.max(axis=(0, 1, 2))-values.min(axis=(0, 1, 2))
            np.testing.assert_allclose(data['reference_ranges/uniform_gt/data_range'][...], expected_range)
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            main(args)

    def test_comparison_grid_containment_masks_and_fine_over_coarse(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['data'].update(H=1, W=1)
        grid = ComparisonGrid(cfg)
        coarse = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
        fine = [[0., 0.], [.5, 0.], [.5, .5], [0., .5]]
        owners = grid.owners(np.array([fine, coarse]), np.array([1, 0]))
        np.testing.assert_array_equal(owners, [[0, 1], [1, 1]])
        image, valid = grid.sample(owners, np.array([[20.]*4, [10.]*4]))
        np.testing.assert_array_equal(image[..., 0], [[20., 10.], [10., 10.]])
        owners = grid.owners(np.array([fine]), np.array([1]))
        image, valid = grid.sample(owners, np.array([[20.]*4]))
        np.testing.assert_array_equal(valid, [[True, False], [False, False]])
        self.assertTrue(np.isnan(image[~valid]).all())
        # Check mapped, non-axis-aligned (including concave) cells independently.
        from matplotlib.path import Path as Polygon
        grid = ComparisonGrid(self.cfg)
        y, x = np.meshgrid(grid.y, grid.x, indexing='ij')
        points = np.column_stack([x.ravel(), y.ravel()])
        for vertices in (
                [[.05, .07], [.91, .12], [.8, .94], [.09, .83]],
                [[.05, .05], [.94, .08], [.31, .39], [.08, .92]]):
            expected = Polygon(vertices).contains_points(points).reshape(grid.height, grid.width)
            actual = grid.owners(np.array([vertices]), np.array([0])) >= 0
            np.testing.assert_array_equal(actual, expected)

    def test_selection_of_excluded_trajectory_and_identity_validation(self):
        cfg = self.cache_cfg()
        cfg['data']['excluded_trajectories'] = []
        checkpoint, sources = self.save_checkpoint(cfg, omit_first_source=True)
        native = self.root/'data/p_1.0/angle_20/p-1p0_angle-20.pt'
        cache = sources[0].path
        loaded = load_inputs(checkpoint, cache, native, device='cpu')
        self.assertEqual(loaded[2].name, 'p-1p0_angle-20')
        checkpoint, _ = self.save_checkpoint(cfg)
        with h5py.File(sources[0].cache_files[0], 'r+') as f:
            f['t00001/feat_t_on_pred'][0, 0] += 1.
        with self.assertRaisesRegex(ValueError, 'differ from the checkpoint'):
            load_inputs(checkpoint, cache, native, device='cpu')
        with self.assertRaisesRegex(ValueError, 'same.*trajectory'):
            load_inputs(checkpoint, sources[1].path, native, device='cpu')

    def test_maps_required_and_index_time_geometry_validation(self):
        self.add_native_geometry()
        source = load_trajectories(self.cfg)[0]
        geometry = NativeGeometry(source, [0, 1])
        snapshot = source.series[0]
        actual = geometry.quads(0)
        np.testing.assert_allclose(actual.mean(axis=1), snapshot['pos'].numpy())
        self.assertEqual(actual.shape, (64, 4, 2))
        with self.assertRaisesRegex(FileNotFoundError, 'Exact native mesh'):
            NativeGeometry(source, [0], map_dir=self.root/'missing')
        path = geometry.map_dir/'DMR.plot.000000.2d.map.hdf5'
        # Input row order may differ from the map's box order.
        reordered = dict(snapshot)
        for key in ('ij', 'level', 'pos', 'pos_phys'):
            reordered[key] = snapshot[key].flip(0)
        np.testing.assert_allclose(mapped_quads(reordered, path), actual[::-1])
        reordered['time'] = 1.
        with self.assertRaisesRegex(ValueError, 'map time'):
            mapped_quads(reordered, path)
        reordered = dict(snapshot, pos_phys=snapshot['pos_phys']+1.)
        with self.assertRaisesRegex(ValueError, 'cell centers'):
            mapped_quads(reordered, path)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_rollout(self):
        cfg = self.cache_cfg()
        cfg['model'].update(name='MeshGraphNet', message_passing_steps=2, mlp_layers=1)
        source = load_trajectories(cfg)[0]
        rt = Runtime(cfg, 'cuda')
        try:
            frames = list(iter_rollout(engine.build_model(cfg).cuda(), source, cfg, rt,
                                      torch.zeros(4, device='cuda'), torch.ones(4, device='cuda'), .1, start=0, steps=3))
            self.assertTrue(all(torch.isfinite(state).all() for _, _, state in frames))
        finally:
            rt.close()


if __name__ == '__main__':
    unittest.main()
