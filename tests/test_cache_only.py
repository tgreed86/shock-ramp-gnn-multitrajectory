"""Cache-only training must work with the original trajectory directory absent."""
import argparse
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import unittest

import h5py
import numpy as np
import torch

import test_runtime
from data import load_trajectories, normalization, split_manifest, split_windows
from engine import build_model, run_epoch
from export_cache_times import export_times
from runtime import Mesh, Runtime
from train import load_config, main, validate_config


class CacheOnlyTests(unittest.TestCase):
    tearDown = test_runtime.RuntimeTests.tearDown
    values = test_runtime.RuntimeTests.values

    def setUp(self):
        test_runtime.RuntimeTests.setUp(self)
        native = load_trajectories(self.cfg)
        self.native_splits = {}
        for stride in (1, 2, 3):
            self.cfg['data']['stride'] = stride
            self.native_splits[stride] = split_manifest(native, split_windows(native, self.cfg), self.cfg)['windows']
        self.cfg['data']['stride'] = 1
        self.times_file = export_times(self.cfg, self.root/'timestamps.json')
        # No native file remains at its original location for any cache-only operation.
        (self.root/'data').rename(self.root/'unavailable_native_data')
        self.cfg['data'].pop('root')
        self.cfg['data'].update(source='uniform_cache', timestamps_path=str(self.times_file))

    def only_specs(self, path, *args, **kwargs):
        self.assertEqual(Path(path).parent.resolve(), (self.root/'specs').resolve(), 'Tried to load a native trajectory')
        return self.original_load(path, *args, **kwargs)

    def flatten_cache(self, family):
        for path in (family/'uniform_meshes').glob('*.h5'):
            path.rename(family/path.name)

    def test_flat_and_mixed_layouts_preserve_identity_split_and_values(self):
        sources = load_trajectories(self.cfg)
        expected = split_manifest(sources, split_windows(sources, self.cfg), self.cfg)
        for family in sorted((self.root/'cache').iterdir()):
            self.flatten_cache(family)
            sources = load_trajectories(self.cfg)
            self.assertEqual(split_manifest(sources, split_windows(sources, self.cfg), self.cfg), expected)
            rt = Runtime(self.cfg, 'cpu')
            try:
                mesh = Mesh(self.centers[0], torch.zeros(16, dtype=torch.long), None, None, None, None)
                for sid, source in enumerate(sources):
                    values, missing = rt.gt(source, 3, mesh)
                    self.assertEqual(missing, 0)
                    torch.testing.assert_close(values, self.values(sid, 3, 0))
            finally:
                rt.close()

    def test_flat_layout_training(self):
        for family in (self.root/'cache').iterdir():
            self.flatten_cache(family)
        self.exercise_training('cpu')

    def test_ambiguous_layout_rejected_but_exclusions_still_apply_first(self):
        family = self.root/'cache/p-1p0_angle-20'
        name = 'shock_ramp_precomp_L0_uniform_refine.h5'
        shutil.copyfile(family/'uniform_meshes'/name, family/name)
        with self.assertRaisesRegex(ValueError, 'Ambiguous uniform cache layout'):
            load_trajectories(self.cfg)
        self.cfg['data']['excluded_trajectories'] = [family.name]
        self.assertEqual([s.name for s in load_trajectories(self.cfg)], ['p-1p0_angle-30'])

    def test_incomplete_flat_family_is_not_silently_omitted(self):
        family = self.root/'cache/p-1p0_angle-20'
        self.flatten_cache(family)
        (family/'shock_ramp_precomp_L0_uniform_refine.h5').rename(family/'missing-L0.h5')
        with self.assertRaises(FileNotFoundError):
            load_trajectories(self.cfg)

    def test_native_mode_and_timestamp_export_accept_flat_layout(self):
        for family in (self.root/'cache').iterdir():
            self.flatten_cache(family)
        (self.root/'unavailable_native_data').rename(self.root/'data')
        cfg = copy.deepcopy(self.cfg)
        cfg['data'].update(source='native', root=str(self.root/'data'))
        sources = load_trajectories(cfg)
        for source in sources:
            self.assertEqual(source.cache_files[0].parent.name, source.name)
        output = export_times(cfg, self.root/'flat_timestamps.json')
        self.assertEqual(json.loads(output.read_text()), json.loads(self.times_file.read_text()))
        cfg['train']['runtime_mesh']['multires_gt_lookup']['enabled'] = False
        with patch('data.uniform_cache_files', side_effect=AssertionError('Inspected disabled caches')):
            self.assertEqual(len(load_trajectories(cfg)), 2)

    def test_exported_times_and_identical_window_membership(self):
        self.original_load = torch.load
        with patch('torch.load', side_effect=self.only_specs):
            sources = load_trajectories(self.cfg)
        for source in sources:
            torch.testing.assert_close(source.times, torch.tensor([t*.1 for t in range(7)], dtype=torch.float64))
            self.assertFalse(hasattr(source, 'series'))
        for stride in (1, 2, 3):
            self.cfg['data']['stride'] = stride
            actual = split_manifest(sources, split_windows(sources, self.cfg), self.cfg)['windows']
            self.assertEqual(actual, self.native_splits[stride])

    def test_exclusions_before_cache_open_and_metadata_lookup(self):
        self.cfg['data']['excluded_trajectories'] = ['p-1p0_angle-30']
        path = self.root/'cache/p-1p0_angle-30/uniform_meshes/shock_ramp_precomp_L0_uniform_refine.h5'
        path.write_bytes(b'invalid excluded file')
        metadata = json.loads(self.times_file.read_text())
        del metadata['trajectories']['p-1p0_angle-30']
        self.times_file.write_text(json.dumps(metadata))
        self.assertEqual([s.name for s in load_trajectories(self.cfg)], ['p-1p0_angle-20'])
        for excluded, message in [(['typo'], 'Unknown excluded'),
                                  (['p-1p0_angle-20', 'p-1p0_angle-30'], 'All trajectories')]:
            self.cfg['data']['excluded_trajectories'] = excluded
            with self.assertRaisesRegex(ValueError, message):
                load_trajectories(self.cfg)

    def test_normalization_uses_finest_cache_with_window_multiplicity(self):
        sources = load_trajectories(self.cfg)
        windows = [(0,0), (0,1), (1,2)]
        mu, sigma = normalization(sources, windows, self.cfg)
        expected = torch.cat([self.values(sid, t+j, 1) for sid,t in windows for j in range(3)]).double()
        torch.testing.assert_close(mu, expected.mean(0).float())
        torch.testing.assert_close(sigma, (expected.var(0)+1e-8).sqrt().float())

    def test_source_order_follows_native_relative_names(self):
        old_name, new_name = 'p-1p0_angle-30', 'p-10_angle-30'
        old = self.root/'cache'/old_name
        new = self.root/'cache'/new_name
        old.rename(new)
        relative = f'p_10/angle_30/{new_name}.pt'
        for path in (new/'uniform_meshes').glob('*.h5'):
            with h5py.File(path, 'r+') as f:
                provenance = json.loads(f['meta'].attrs['uniform_gt_provenance'])
                provenance['source_path'] = f'/unavailable/{relative}'
                f['meta'].attrs['uniform_gt_provenance'] = json.dumps(provenance)
        metadata = json.loads(self.times_file.read_text())
        record = metadata['trajectories'].pop(old_name)
        record['source_relative_path'] = relative
        metadata['trajectories'][new_name] = record
        self.times_file.write_text(json.dumps(metadata))
        self.assertEqual([s.name for s in load_trajectories(self.cfg)], ['p-1p0_angle-20', new_name])

    def test_initial_snapshot_uses_canonical_finest_features(self):
        source = load_trajectories(self.cfg)[0]
        rt = Runtime(self.cfg, 'cpu')
        try:
            for t in (0, 3, 6):
                centers, values, levels = rt.initial_snapshot(source, t)
                torch.testing.assert_close(centers, self.centers[1])
                torch.testing.assert_close(values, self.values(0, t, 1))
                self.assertTrue((levels == 1).all())
            reader = rt.lookup(source)
            self.assertEqual(reader.disk_reads, 3)
            rt.initial_snapshot(source, 3)
            self.assertEqual(reader.disk_reads, 3)
        finally:
            rt.close()

    def exercise_fallback(self, device):
        source = load_trajectories(self.cfg)[0]
        rt = Runtime(self.cfg, device)
        reader = rt.lookup(source)
        # Force one level to miss without changing the valid finest-cache representation.
        del reader._source_file_for_level[0]
        mesh = Mesh(self.centers[0][:2].to(device), torch.zeros(2, dtype=torch.long, device=device),
                    None, None, None, None)
        try:
            expected = rt.idw(self.centers[1].to(device), self.values(0, 2, 1).to(device), mesh.centers)
            values, missing = rt.gt(source, 2, mesh)
            self.assertEqual(missing, 2)
            torch.testing.assert_close(values, expected)
            expected_next = rt.idw(self.centers[1].to(device), self.values(0, 3, 1).to(device), mesh.centers)
            with patch.object(rt, 'idw_map', side_effect=AssertionError('Rebuilt static fallback map')):
                values, _ = rt.gt(source, 3, mesh)
            torch.testing.assert_close(values, expected_next)
            rt.lookup_cfg['fallback_to_idw'] = False
            with self.assertRaisesRegex(RuntimeError, 'cache misses'):
                rt.gt(source, 3, mesh)
        finally:
            rt.close()

    def test_finest_cache_fallback_and_map_reuse(self):
        self.exercise_fallback('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_finest_cache_fallback(self):
        self.exercise_fallback('cuda')

    def exercise_training(self, device):
        self.original_load = torch.load
        with patch('torch.load', side_effect=self.only_specs):
            sources = load_trajectories(self.cfg)
            for backend in ('gradient_fast', 'gradient'):
                self.cfg['train']['runtime_mesh']['policy_backend'] = backend
                rt = Runtime(self.cfg, device)
                net = build_model(self.cfg).to(device)
                before = [p.detach().clone() for p in net.parameters()]
                opt = torch.optim.AdamW(net.parameters(), lr=.001)
                mu, sigma = normalization(sources, [(0,0), (1,0)], self.cfg)
                try:
                    for optimizer in (opt, None):
                        stats = run_epoch(net, [(0,0), (1,0)], sources, self.cfg, rt,
                                          mu.to(device), sigma.to(device), .1, optimizer=optimizer)
                        self.assertTrue(np.isfinite(stats['loss']))
                    self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before, net.parameters())))
                finally:
                    rt.close()

    def test_training_and_evaluation_without_native_files(self):
        self.exercise_training('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_cache_only_training(self):
        self.exercise_training('cuda')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS is not available')
    def test_mps_cache_only_training(self):
        self.exercise_training('mps')

    def test_ramp_context_uses_cached_geometry(self):
        from kernels import _build_ramp_feature_context
        self.cfg['features']['build'].update(include_ramp_angle=True, include_signed_distance_to_ramp=True)
        source = load_trajectories(self.cfg)[0]
        expected = _build_ramp_feature_context(self.cfg, data_obj=[{'pos': self.centers[1]}],
                                             pt_path=source.name, mesh_spec_path=str(source.spec))
        for key in ('distance_point', 'distance_normal'):
            torch.testing.assert_close(source.ramp[key], expected[key])
        self.assertEqual(source.ramp['angle_deg'], 20.)

    def test_missing_or_invalid_times_fail(self):
        original = json.loads(self.times_file.read_text())
        for times in ([0., .1], [0., .1, .1, .3, .4, .5, .6], [0., .1, float('nan'), .3, .4, .5, .6]):
            metadata = copy.deepcopy(original)
            metadata['trajectories']['p-1p0_angle-20']['times'] = times
            self.times_file.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'timestamps'):
                load_trajectories(self.cfg)
        self.cfg['data']['timestamps_path'] = str(self.root/'missing.json')
        with self.assertRaisesRegex(FileNotFoundError, 'Physical timestamps'):
            load_trajectories(self.cfg)

    def test_cache_provenance_and_all_feature_shapes_checked(self):
        path = self.root/'cache/p-1p0_angle-20/uniform_meshes/shock_ramp_precomp_L0_uniform_refine.h5'
        with h5py.File(path, 'r+') as f:
            del f['t00006/feat_tp1_on_pred']
            f['t00006/feat_tp1_on_pred'] = np.zeros((1,4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'feature shape'):
            load_trajectories(self.cfg)

    def test_cache_only_config_needs_lookup_and_timestamps(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['train']['runtime_mesh']['multires_gt_lookup']['enabled'] = False
        with self.assertRaisesRegex(ValueError, 'multires_gt_lookup'):
            validate_config(cfg)
        del self.cfg['data']['timestamps_path']
        with self.assertRaisesRegex(ValueError, 'timestamps_path'):
            validate_config(self.cfg)

    def test_resume_after_relocation_and_reject_changed_times(self):
        config = self.root/'config_cache.json'
        config.write_text(json.dumps(self.cfg))
        args = argparse.Namespace(config=str(config), device='cpu', epochs=1, out_dir=str(self.root/'run'),
                                  max_batches=1, dry_run=False, resume=None)
        with redirect_stdout(io.StringIO()):
            main(args)
        checkpoint = self.root/'run/last_model.pt'
        payload = torch.load(checkpoint, weights_only=False)
        self.assertEqual(payload['format'], 'shock-ramp-uniform-cache-v1')
        payload['format'] = 'shock-ramp-multitrajectory-v1'
        native_checkpoint = self.root/'wrong_mode.pt'
        torch.save(payload, native_checkpoint)
        args.resume = str(native_checkpoint)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'data mode, dataset'):
            main(args)
        (self.root/'cache').rename(self.root/'relocated_cache')
        for family in (self.root/'relocated_cache').iterdir():
            self.flatten_cache(family)
        self.cfg['train']['runtime_mesh']['multires_gt_lookup']['root'] = str(self.root/'relocated_cache')
        self.times_file.rename(self.root/'relocated_times.json')
        self.times_file = self.root/'relocated_times.json'
        self.cfg['data']['timestamps_path'] = str(self.times_file)
        config.write_text(json.dumps(self.cfg))
        args.epochs, args.resume = 2, str(checkpoint)
        with redirect_stdout(io.StringIO()):
            main(args)
        self.assertEqual(torch.load(checkpoint, weights_only=False)['epoch'], 2)
        metadata = json.loads(self.times_file.read_text())
        original_metadata = copy.deepcopy(metadata)
        metadata['trajectories']['p-1p0_angle-20']['times'][1] += .001
        self.times_file.write_text(json.dumps(metadata))
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'data mode, dataset'):
            main(args)
        self.times_file.write_text(json.dumps(original_metadata))
        changed_cache = self.root/'relocated_cache/p-1p0_angle-20/shock_ramp_precomp_L0_uniform_refine.h5'
        with h5py.File(changed_cache, 'r+') as f:
            f['t00001/feat_t_on_pred'][0,0] += 1
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'data mode, dataset'):
            main(args)


if __name__ == '__main__':
    unittest.main()
