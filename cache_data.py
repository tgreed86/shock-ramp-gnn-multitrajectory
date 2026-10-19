"""Uniform-cache data sources. No native trajectory file is opened or stat'ed here."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np
import torch

from gt_cache import feature_location
from kernels import _build_ramp_feature_context


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class CacheTrajectory:
    path: Path
    spec: Path
    cache_files: dict
    times: torch.Tensor
    ramp: dict
    source_size: int
    identity: dict
    order_key: str

    backend = 'uniform_cache'

    @property
    def name(self):
        return self.path.name

    @property
    def num_steps(self):
        return len(self.times)

    def normalization_features(self, indices, cfg):
        # One canonical representation at Lmax, never all overlapping levels.
        with h5py.File(self.cache_files[cfg['policy']['max_level']], 'r') as f:
            for index in indices:
                group, dataset = feature_location(index)
                yield torch.from_numpy(np.asarray(f[f't{group:05d}/{dataset}'][...], dtype=np.float32))


def validate_static_and_shapes(path, level, cfg, steps):
    """Validate every dataset's shape without retaining timestep feature arrays."""
    with h5py.File(path, 'r') as f:
        centers = f['static/pred_centers'][...]
        levels = f['static/pred_levels'][...]
        n = len(centers)
        lm = cfg['policy']['max_level']
        if (n == 0 or centers.shape != (n, 2) or not np.isfinite(centers).all()
                or levels.shape != (n,) or levels.dtype.kind not in 'iu'
                or levels.min() < level or levels.max() > lm):
            raise ValueError(f'Invalid uniform cache geometry: {path}')
        x0, x1, y0, y1 = cfg['data']['bbox']
        sizes = cfg['policy']['refine_ratio'] ** levels.astype(np.int64)
        cols = (centers[:, 0].astype(np.float64) - x0) * cfg['data']['W'] * sizes / (x1-x0)
        rows = (centers[:, 1].astype(np.float64) - y0) * cfg['data']['H'] * sizes / (y1-y0)
        if (np.any(cols < 0) or np.any(cols >= cfg['data']['W'] * sizes)
                or np.any(rows < 0) or np.any(rows >= cfg['data']['H'] * sizes)
                or not np.allclose(cols % 1, .5, atol=1e-4, rtol=0)
                or not np.allclose(rows % 1, .5, atol=1e-4, rtol=0)):
            raise ValueError(f'Cache centers are not aligned to the configured grid: {path}')
        keys = np.column_stack((levels, rows.astype(np.int64), cols.astype(np.int64)))
        if len(np.unique(keys, axis=0)) != n:
            raise ValueError(f'Duplicate cache cell keys: {path}')
        shape = (n, len(cfg['features']['use_columns']))
        for t in range(1, steps):
            for name in ('feat_t_on_pred', 'feat_tp1_on_pred'):
                ds = f[f't{t:05d}/{name}']
                if ds.shape != shape or ds.dtype.kind != 'f':
                    raise ValueError(f'Invalid feature shape/dtype: {path} t={t} {name}')
        return torch.from_numpy(centers.astype(np.float32, copy=False))


def load_cache_trajectories(cfg):
    from data import uniform_cache_files, validate_cache

    root = Path(cfg['train']['runtime_mesh']['multires_gt_lookup']['root'])
    directories = sorted(d for d in root.glob('p-*_angle-*') if d.is_dir())
    if not directories:
        raise ValueError(f'No uniform cache families found under {root}')
    names = [d.name for d in directories]
    excluded = cfg['data'].get('excluded_trajectories', [])
    if not isinstance(excluded, list) or not all(isinstance(n, str) for n in excluded):
        raise ValueError('data.excluded_trajectories must be a list of exact trajectory stems')
    if set(excluded) - set(names):
        raise ValueError(f'Unknown excluded trajectories: {sorted(set(excluded)-set(names))}')
    directories = [d for d in directories if d.name not in excluded]
    if not directories:
        raise ValueError('All trajectories were excluded')
    times_path = Path(cfg['data']['timestamps_path'])
    if not times_path.is_file():
        raise FileNotFoundError(f'Physical timestamps are required: {times_path}. '
                                'Run export_cache_times.py once where the native trajectories are available, '
                                'then copy the JSON with the caches.')
    metadata = json.loads(times_path.read_text())
    if metadata.get('schema') != 1 or not isinstance(metadata.get('trajectories'), dict):
        raise ValueError(f'Unsupported timestamp metadata: {times_path}')
    sources = []
    for directory in directories:
        name = directory.name
        match = re.fullmatch(r'p-([0-9p.]+)_angle-([0-9.]+)', name)
        if match is None:
            raise ValueError(f'Invalid trajectory name: {name}')
        record = metadata['trajectories'].get(name)
        if not isinstance(record, dict):
            raise ValueError(f'Missing physical timestamps for {name}: {times_path}')
        times = torch.tensor(record['times'], dtype=torch.float64)
        if (times.ndim != 1 or len(times) < cfg['data']['window_size']
                or not torch.isfinite(times).all() or not (times.diff() > 0).all()):
            raise ValueError(f'Non-increasing, invalid, or insufficient timestamps: {name}')
        relative = Path(record['source_relative_path'])
        if (relative.is_absolute() or len(relative.parts) != 3 or '..' in relative.parts
                or relative.stem != name or not relative.parts[0].startswith('p_')
                or not relative.parts[1].startswith('angle_')):
            raise ValueError(f'Invalid source ordering metadata: {name}')
        h, w, lm = cfg['data']['H'], cfg['data']['W'], cfg['policy']['max_level']
        angle = float(match[2])
        spec = Path(cfg['mesh']['starting_mesh_path']) / f'wedge_mesh_spec_{w}x{h}_L{lm}_ramp-angle-{angle:g}.pt'
        # These small bundled geometry specs are not native trajectory data.
        payload = torch.load(spec, map_location='cpu', weights_only=False)
        if (payload.get('n0_x', payload.get('n0')), payload.get('n0_y', payload.get('n0')), payload['max_level']) != (w, h, lm):
            raise ValueError(f'Mesh specification dimensions do not match: {spec}')
        files = uniform_cache_files(root, name, lm)
        source = CacheTrajectory(directory, spec, files, times, {}, int(record['source_size']), {}, str(relative))
        validate_cache(source, cfg)
        finest_centers = None
        hashes = {}
        for level, path in files.items():
            centers = validate_static_and_shapes(path, level, cfg, len(times))
            if level == lm:
                finest_centers = centers
            hashes[str(level)] = file_sha256(path)
        # Deliberately derived from cached geometry, rather than the native fit.
        source.ramp = _build_ramp_feature_context(cfg, data_obj=[{'pos': finest_centers}],
                                                pt_path=name, mesh_spec_path=str(spec))
        source.identity = {'name': name, 'snapshots': len(times), 'source_relative_path': str(relative),
                           'times_sha256': hashlib.sha256(times.numpy().tobytes()).hexdigest(),
                           'mesh_spec_sha256': file_sha256(spec), 'cache_sha256': hashes}
        sources.append(source)
        print(f'[DATA] {name}: snapshots={len(times)} angle={angle:g} source=uniform_cache', flush=True)
    # Sorting cache directory names would change seeded window membership.
    return sorted(sources, key=lambda s: s.order_key)
