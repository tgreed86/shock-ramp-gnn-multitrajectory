"""Trajectory discovery, exclusion, local window indices, and training statistics."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re

import h5py
import numpy as np
import torch

from kernels import _apply_momentum_sigma_mode, _build_ramp_feature_context


def discover(cfg):
    root = Path(cfg['data']['root'])
    paths = sorted(root.glob('p_*/angle_*/*.pt'))
    if not paths:
        raise ValueError(f'No trajectories found under {root}')
    names = [p.stem for p in paths]
    if len(set(names)) != len(names):
        raise ValueError('Trajectory stems must be unique')
    excluded = cfg['data'].get('excluded_trajectories', [])
    if not isinstance(excluded, list) or not all(isinstance(s, str) for s in excluded):
        raise ValueError('data.excluded_trajectories must be a list of exact trajectory stems')
    unknown = set(excluded) - set(names)
    if unknown:
        raise ValueError(f'Unknown excluded trajectories: {sorted(unknown)}')
    selected = [p.resolve() for p in paths if p.stem not in excluded]
    if not selected:
        raise ValueError('All trajectories were excluded')
    return selected


@dataclass
class Trajectory:
    path: Path
    spec: Path
    cache_files: dict
    series: list
    times: torch.Tensor
    ramp: dict

    backend = 'native'

    @property
    def num_steps(self):
        return len(self.series)

    @property
    def name(self):
        return self.path.stem

    def snapshot(self, index, cfg, device):
        snap = self.series[index]
        return (snap['pos'][:, :2].to(device=device, dtype=torch.float32),
                snap['x'][:, cfg['features']['use_columns']].to(device=device, dtype=torch.float32),
                snap['level'].to(device=device, dtype=torch.long).view(-1))

    def normalization_features(self, indices, cfg):
        for index in indices:
            yield self.series[index]['x'][:, cfg['features']['use_columns']]


def validate_cache(source, cfg):
    """Reject wrong-trajectory/time/geometry caches before opening runtime lookups."""
    digest = hashlib.sha256(source.spec.read_bytes()).hexdigest()
    for level, path in source.cache_files.items():
        with h5py.File(path, 'r') as f:
            meta = f['meta']
            provenance = json.loads(meta.attrs['uniform_gt_provenance'])
            stored = json.loads(meta['cfg_json'][()].decode())
            checks = {
                'trajectory': Path(provenance['source_path']).stem == source.name,
                'source size': provenance['source_size'] == (
                    source.path.stat().st_size if source.backend == 'native' else source.source_size),
                'mesh spec': provenance['mesh_spec_sha256'] == digest,
                'full trajectory': provenance.get('max_timesteps') is None,
                'timesteps': int(meta.attrs['T']) == source.num_steps,
                'height': int(meta.attrs['H']) == cfg['data']['H'],
                'width': int(meta.attrs['W']) == cfg['data']['W'],
                'bbox': np.allclose(meta.attrs['bbox'], cfg['data']['bbox'], rtol=0, atol=1e-12),
                'refinement ratio': int(meta.attrs['refine_ratio']) == cfg['policy']['refine_ratio'],
                'uniform level': int(meta.attrs['starting_refine_to_level']) == level,
                'maximum level': stored['policy']['max_level'] == cfg['policy']['max_level'],
                'features': stored['features']['use_columns'] == cfg['features']['use_columns'],
            }
            if source.backend == 'uniform_cache':
                checks['source ordering'] = str(Path(*Path(provenance['source_path']).parts[-3:])) == source.order_key
            failed = [key for key, passed in checks.items() if not passed]
            if failed:
                raise ValueError(f'Cache mismatch ({", ".join(failed)}): {path}')
            expected = {f't{t:05d}' for t in range(1, source.num_steps)}
            if {k for k in f if re.fullmatch(r't\d+', k)} != expected:
                raise ValueError(f'Incomplete timestep groups: {path}')
            if f['t00001/feat_t_on_pred'].shape[1] != len(cfg['features']['use_columns']):
                raise ValueError(f'Wrong channel count: {path}')


def load_trajectories(cfg):
    if cfg['data'].get('source', 'native') == 'uniform_cache':
        from cache_data import load_cache_trajectories
        return load_cache_trajectories(cfg)
    sources = []
    for path in discover(cfg):
        match = re.fullmatch(r'p-([0-9p.]+)_angle-([0-9.]+)', path.stem)
        if match is None:
            raise ValueError(f'Invalid trajectory name: {path.name}')
        angle = float(match[2])
        h, w, lm = cfg['data']['H'], cfg['data']['W'], cfg['policy']['max_level']
        spec = Path(cfg['mesh']['starting_mesh_path']) / f'wedge_mesh_spec_{w}x{h}_L{lm}_ramp-angle-{angle:g}.pt'
        payload = torch.load(spec, map_location='cpu', weights_only=False)
        if (payload.get('n0_x', payload.get('n0')), payload.get('n0_y', payload.get('n0')), payload['max_level']) != (w, h, lm):
            raise ValueError(f'Mesh specification dimensions do not match: {spec}')
        # Trusted project artifacts only. mmap avoids eagerly copying all nine large files.
        series = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        if not isinstance(series, list) or len(series) < cfg['data']['window_size']:
            raise ValueError(f'Expected at least window_size snapshots: {path}')
        times = torch.tensor([float(s.get('time', s.get('t'))) for s in series], dtype=torch.float64)
        if not torch.isfinite(times).all() or not (times.diff() > 0).all():
            raise ValueError(f'Non-increasing or invalid timestamps: {path}')
        lookup_cfg = cfg['train']['runtime_mesh']['multires_gt_lookup']
        cache_dir = Path(lookup_cfg['root']) / path.stem / 'uniform_meshes'
        files = {l: cache_dir / f'shock_ramp_precomp_L{l}_uniform_refine.h5' for l in range(lm + 1)}
        ramp = _build_ramp_feature_context(cfg, data_obj=[series[0]], pt_path=str(path), mesh_spec_path=str(spec))
        source = Trajectory(path, spec, files, series, times, ramp)
        if lookup_cfg['enabled']:
            validate_cache(source, cfg)
        sources.append(source)
        print(f'[DATA] {source.name}: snapshots={len(series)} angle={angle:g} spec={spec.name}', flush=True)
    return sources


def split_windows(sources, cfg):
    """Preserve the original global-stride, seeded random WINDOW split."""
    k, stride = cfg['data']['window_size'], cfg['data']['stride']
    lengths = [s.num_steps for s in sources]
    offsets = np.cumsum([0] + lengths)
    windows = []
    for start in range(0, int(offsets[-1]) - k + 1, stride):
        sid = int(np.searchsorted(offsets, start, side='right') - 1)
        if start + k <= offsets[sid + 1]:
            windows.append((sid, int(start - offsets[sid])))
    if not windows:
        raise ValueError('No windows remain')
    order = np.random.default_rng(cfg['seed']).permutation(len(windows))
    n_train = round(cfg['split']['train'] * len(windows))
    n_val = round(cfg['split']['val'] * len(windows))
    if n_train < 1 or n_train + n_val > len(windows):
        raise ValueError('Split leaves no training windows or exceeds total windows')
    return {name: [windows[i] for i in indices] for name, indices in (
        ('train', order[:n_train]), ('val', order[n_train:n_train+n_val]),
        ('test', order[n_train+n_val:]))}


def split_manifest(sources, splits, cfg):
    if cfg['data'].get('source', 'native') == 'uniform_cache':
        return {
            'data_source': 'uniform_cache',
            'semantics': 'uniform-cache-v1:lmax-initial,lmax-norm,lmax-ramp,lmax-idw',
            'excluded_trajectories': cfg['data']['excluded_trajectories'],
            'sources': [s.identity for s in sources],
            'window_size': cfg['data']['window_size'], 'stride': cfg['data']['stride'], 'seed': cfg['seed'],
            'split_mode': 'random_overlapping_windows',
            'windows': {split: [{'trajectory': sources[sid].name, 't0': t} for sid, t in windows]
                        for split, windows in splits.items()},
        }
    return {
        'excluded_trajectories': cfg['data']['excluded_trajectories'],
        'sources': [{'name': s.name, 'path': str(s.path), 'snapshots': len(s.series),
                     'mesh_spec': str(s.spec), 'cache_directory': str(s.cache_files[0].parent)} for s in sources],
        'window_size': cfg['data']['window_size'], 'stride': cfg['data']['stride'], 'seed': cfg['seed'],
        'split_mode': 'random_overlapping_windows',
        'windows': {split: [{'trajectory': sources[sid].name, 't0': t} for sid, t in windows]
                    for split, windows in splits.items()},
    }


def normalization(sources, windows, cfg):
    """Train-window-weighted statistics on native cells or the finest cache."""
    columns = cfg['features']['use_columns']
    if not cfg['features']['normalize']:
        return torch.zeros(len(columns)), torch.ones(len(columns))
    counts = Counter((sid, t+j) for sid, t in windows for j in range(cfg['data']['window_size']))
    count, mean, m2 = 0, torch.zeros(len(columns), dtype=torch.float64), torch.zeros(len(columns), dtype=torch.float64)
    for sid, source in enumerate(sources):
        times = sorted(t for s, t in counts if s == sid)
        for t, features in zip(times, source.normalization_features(times, cfg)):
            x = features.double()
            if not torch.isfinite(x).all():
                raise ValueError(f'Non-finite GT features: {source.name} t={t}')
            repeats = counts[(sid, t)]
            n = x.shape[0] * repeats
            batch_mean = x.mean(0)
            batch_m2 = ((x - batch_mean)**2).sum(0) * repeats
            delta = batch_mean - mean
            total = count + n
            m2 += batch_m2 + delta.square() * (count * n / total)
            mean += delta * (n / total)
            count = total
    sigma = torch.sqrt(m2 / max(count-1, 1) + 1e-8).float()
    sigma = _apply_momentum_sigma_mode(sigma, mode=cfg['features']['normalization']['momentum_sigma_mode'],
                                      momentum_x_idx=1, momentum_y_idx=2)
    return mean.float(), sigma
