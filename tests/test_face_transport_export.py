"""Analytic fluxes, conservative restriction, variable-time export and provenance."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import h5py
import numpy as np
import torch

import test_cache_only
from precompute_face_transports import (build_faces, divergence_rate, export_all,
                                       reconstruct_transports)


class FaceTransportTests(unittest.TestCase):
    def test_rusanov_shock_and_orientation(self):
        state = torch.tensor([[1., 0., 0., 2.5], [.125, 0., 0., .25]])
        faces = SimpleNamespace(owner=torch.tensor([0]), neighbor=torch.tensor([1]),
            interior_count=1, normal=torch.tensor([[1., 0.]]), length=torch.tensor([.5]),
            boundary_open=torch.tensor([False]))
        actual, floors = reconstruct_transports(state, faces)
        speed = 1.4**.5
        expected = torch.tensor([[.5*speed*.875, .55, 0., .5*speed*2.25]]) * .5
        torch.testing.assert_close(actual, expected)
        faces.owner, faces.neighbor = faces.neighbor, faces.owner
        faces.normal = -faces.normal
        reverse, _ = reconstruct_transports(state, faces)
        torch.testing.assert_close(reverse, -actual)
        self.assertEqual(int(floors.sum()), 0)

    def test_wall_and_open_flux_and_invalid_state(self):
        # rho=2, velocity=(3,4), pressure=2, total energy=30.
        state = torch.tensor([[2., 6., 8., 30.]])
        faces = SimpleNamespace(owner=torch.tensor([0, 0]), neighbor=torch.tensor([-1, -1]),
            interior_count=0, normal=torch.tensor([[-1., 0.], [0., 1.]]),
            length=torch.tensor([.25, .5]), boundary_open=torch.tensor([False, True]))
        actual, _ = reconstruct_transports(state, faces)
        torch.testing.assert_close(actual[0], torch.tensor([0., -.5, 0., 0.]))
        torch.testing.assert_close(actual[1], torch.tensor([8., 24., 34., 128.])*.5)
        state[0, 3] = 1.
        _, counts = reconstruct_transports(state, faces)
        self.assertEqual(int(counts[1]), 1)
        state[0, 0] = 0.
        with self.assertRaisesRegex(ValueError, 'Nonpositive GT density'):
            reconstruct_transports(state, faces)


class FaceExportTests(unittest.TestCase):
    setUp = test_cache_only.CacheOnlyTests.setUp
    tearDown = test_cache_only.CacheOnlyTests.tearDown
    values = test_cache_only.CacheOnlyTests.values

    def faces(self, cfg=None):
        cfg = self.cfg if cfg is None else cfg
        return build_faces(self.centers[1], torch.ones(64, dtype=torch.long), cfg,
                           self.root/'specs/wedge_mesh_spec_4x4_L1_ramp-angle-20.pt')

    def test_constant_state_and_coarse_face_aggregation(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['model']['open_boundary_sides'] = ['left', 'right', 'bottom', 'top']
        faces = self.faces(cfg)
        state = torch.tensor([1., 2., -.5, 10.]).repeat(64, 1)
        transport, _ = reconstruct_transports(state, faces)
        torch.testing.assert_close(divergence_rate(transport, faces), torch.zeros(64, 4, dtype=torch.float64),
                                   atol=1e-6, rtol=0)
        # Aggregate two adjacent L1 segments to an L0 face using integer keys.
        key = faces.lattice_key
        selected = (key[:, 0] == 0) & (key[:, 2] == 2) & (key[:, 1] < 2)
        self.assertEqual(int(selected.sum()), 2)
        pressure = .4*(10.-.5*(4.+.25))
        expected = torch.tensor([2., 4.+pressure, -1., (10.+pressure)*2.]) * .25
        torch.testing.assert_close(transport[selected].sum(0), expected)
        # Summed coarse-cell balance equals the area-weighted fine divergence;
        # all fine faces inside the coarse cell cancel.
        torch.manual_seed(5)
        random_transport = torch.randn_like(transport)
        inside = (faces.centers[:, 0] < .25) & (faces.centers[:, 1] < .25)
        outgoing = inside[faces.owner].double()
        n = faces.interior_count
        outgoing[:n] -= inside[faces.neighbor[:n]].double()
        expected_balance = -(random_transport.double()*outgoing[:, None]).sum(0)
        balance = (divergence_rate(random_transport, faces)[inside] * faces.area[inside, None]).sum(0)
        torch.testing.assert_close(balance, expected_balance, atol=1e-12, rtol=1e-12)

    def test_stream_all_families_variable_dt_and_verified_reuse(self):
        # Preprocessing includes a trajectory even if training excludes it.
        self.cfg['data']['excluded_trajectories'] = ['p-1p0_angle-30']
        metadata = json.loads(self.times_file.read_text())
        for record in metadata['trajectories'].values():
            record['times'] = [0., .1, .3, .6, 1., 1.5, 2.1]
        self.times_file.write_text(json.dumps(metadata))
        out = self.root/'flux'
        paths = export_all(self.cfg, out)
        self.assertEqual(len(paths), 2)
        with h5py.File(paths[0], 'r') as f:
            self.assertTrue(f.attrs['complete'])
            self.assertEqual(f['snapshot_transport'].shape[0], 7)
            self.assertEqual(f['interval_mean_transport'].shape[0], 6)
            np.testing.assert_allclose(f['dt'][...], [.1, .2, .3, .4, .5, .6])
            np.testing.assert_allclose(f['interval_mean_transport'][1],
                                      (f['snapshot_transport'][1]+f['snapshot_transport'][2])*.5, rtol=1e-6)
            faces = self.faces()
            rate = divergence_rate(torch.from_numpy(f['interval_mean_transport'][1]), faces)
            target = (self.values(0, 2, 1).double()-self.values(0, 1, 1).double())/.2
            np.testing.assert_allclose(f['diagnostics/rate_rmse'][1], (rate-target).square().mean(0).sqrt())
            self.assertEqual(json.loads(f.attrs['provenance_json'])['spatial_metric'], 'normalized_cache_cartesian')
        before = [p.stat().st_mtime_ns for p in paths]
        self.assertEqual(export_all(self.cfg, out, skip_existing=True), paths)
        self.assertEqual(before, [p.stat().st_mtime_ns for p in paths])
        with self.assertRaises(FileExistsError):
            export_all(self.cfg, out)
        with self.assertRaisesRegex(ValueError, 'different provenance'):
            export_all(self.cfg, out, skip_existing=True, gamma=1.5)
        with self.assertRaisesRegex(ValueError, 'Unknown trajectories'):
            export_all(self.cfg, out, trajectories=['typo'])

    def test_partial_export_and_failure_do_not_publish(self):
        paths = export_all(self.cfg, self.root/'short', trajectories=['p-1p0_angle-20'], max_intervals=1)
        with h5py.File(paths[0], 'r') as f:
            self.assertEqual(f['snapshot_transport'].shape[0], 2)
        with self.assertRaisesRegex(ValueError, 'different provenance'):
            export_all(self.cfg, self.root/'short', trajectories=['p-1p0_angle-20'], skip_existing=True)
        cache = self.root/'cache/p-1p0_angle-20/uniform_meshes/shock_ramp_precomp_L1_uniform_refine.h5'
        with h5py.File(cache, 'r+') as f:
            f['t00001/feat_t_on_pred'][0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            export_all(self.cfg, self.root/'bad', trajectories=['p-1p0_angle-20'])
        self.assertEqual(list((self.root/'bad').rglob('*.h5')), [])
        self.assertEqual(list((self.root/'bad').rglob('*.partial')), [])


if __name__ == '__main__':
    unittest.main()
