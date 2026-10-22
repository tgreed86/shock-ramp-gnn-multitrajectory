"""Checkpoint loading and a single continuous training-equivalent rollout."""
import copy
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np
import torch

from cache_data import CacheTrajectory, file_sha256, validate_static_and_shapes
from data import Trajectory, uniform_cache_files, validate_cache
from engine import build_model, carry_state, model_step
import kernels as k
from train import checkpoint_format, device_for, validate_config


def load_inputs(checkpoint_path, cache_dir, native_path, *, device='auto', mesh_specs=None):
    """Load only the explicitly selected trajectory, even if excluded in training."""
    checkpoint_path, cache_dir, native_path = map(Path, (checkpoint_path, cache_dir, native_path))
    # Checkpoints and native trajectories must be trusted project artifacts.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = copy.deepcopy(checkpoint['config'])
    if checkpoint['format'] != checkpoint_format(cfg):
        raise ValueError('Unsupported checkpoint format or inconsistent data mode')
    validate_config(cfg)
    device = device_for(device)
    torch.set_num_threads(cfg['cpu_threads'])
    torch.manual_seed(cfg['seed'])
    if cache_dir.name == 'uniform_meshes':
        cache_dir = cache_dir.parent
    name = cache_dir.name
    match = re.fullmatch(r'p-([0-9p.]+)_angle-([0-9.]+)', name)
    if match is None or native_path.stem != name:
        raise ValueError('Cache family and native .pt filename must name the same p-*_angle-* trajectory')
    h, w, lm = cfg['data']['H'], cfg['data']['W'], cfg['policy']['max_level']
    spec_name = f'wedge_mesh_spec_{w}x{h}_L{lm}_ramp-angle-{float(match[2]):g}.pt'
    if mesh_specs is None:
        mesh_specs = Path(cfg['mesh']['starting_mesh_path'])
        if not (mesh_specs / spec_name).is_file():
            mesh_specs = Path(__file__).resolve().parents[1] / 'mesh_specs'
    spec = Path(mesh_specs) / spec_name
    payload = torch.load(spec, map_location='cpu', weights_only=False)
    if (payload.get('n0_x', payload.get('n0')), payload.get('n0_y', payload.get('n0')),
            payload['max_level']) != (w, h, lm):
        raise ValueError(f'Mesh specification dimensions do not match: {spec}')
    files = uniform_cache_files(cache_dir.parent, name, lm)
    series = torch.load(native_path, map_location='cpu', weights_only=False, mmap=True)
    if not isinstance(series, list) or len(series) < 2:
        raise ValueError('Native .pt must contain at least two snapshots')
    times = torch.tensor([float(s.get('time', s.get('t'))) for s in series], dtype=torch.float64)
    if not torch.isfinite(times).all() or not (times.diff() > 0).all():
        raise ValueError('Native physical timestamps must be finite and strictly increasing')
    native = Trajectory(native_path, spec, files, series, times, {})
    # Checks original trajectory size/name, snapshot count, cache/spec/grid identity.
    validate_cache(native, cfg)
    for level, path in files.items():
        centers = validate_static_and_shapes(path, level, cfg, len(times))
        if level == lm:
            finest_centers = centers
    with h5py.File(files[lm], 'r') as f:
        provenance = json.loads(f['meta'].attrs['uniform_gt_provenance'])
    relative = str(Path(*Path(provenance['source_path']).parts[-3:]))
    if cfg['data'].get('source', 'native') == 'uniform_cache':
        source = CacheTrajectory(cache_dir, spec, files, times, {}, native_path.stat().st_size, {}, relative)
        source.ramp = k._build_ramp_feature_context(
            cfg, data_obj=[{'pos': finest_centers}], pt_path=name, mesh_spec_path=str(spec))
        # A selected training source must still match the checkpoint, even after relocation.
        known = next((s for s in checkpoint['manifest']['sources'] if s['name'] == name), None)
        if known is not None:
            identity = {'name': name, 'snapshots': len(times), 'source_relative_path': relative,
                        'times_sha256': hashlib.sha256(times.numpy().tobytes()).hexdigest(),
                        'mesh_spec_sha256': file_sha256(spec),
                        'cache_sha256': {str(level): file_sha256(path) for level, path in files.items()}}
            if identity != known:
                raise ValueError('Selected cache/spec/timestamps differ from the checkpoint training source')
            source.identity = identity
    else:
        source = native
        source.ramp = k._build_ramp_feature_context(
            cfg, data_obj=[series[0]], pt_path=str(native_path), mesh_spec_path=str(spec))
    cfg['mesh']['starting_mesh_path'] = str(spec.parent)
    rt = cfg['train']['runtime_mesh']
    rt['multires_gt_lookup']['root'] = str(cache_dir.parent)
    # Portable execution backends do not change the predictor or lookup semantics.
    rt['multires_gt_lookup']['backend'] = 'auto'
    rt['adjacency_backend'] = 'auto'
    cfg['device'] = str(device)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    mu = checkpoint['mu'].to(device)
    sigma = checkpoint['sigma'].to(device)
    dt_ref = float(checkpoint['dt_ref'])
    if (mu.shape != (4,) or sigma.shape != (4,) or not torch.isfinite(mu).all()
            or not torch.isfinite(sigma).all() or not (sigma > 0).all()
            or not np.isfinite(dt_ref) or dt_ref <= 0):
        raise ValueError('Checkpoint contains invalid normalization or timestep scaling')
    return cfg, model, source, native, mu, sigma, dt_ref, checkpoint


def iter_rollout(model, source, cfg, runtime, mu, sigma, dt_ref, *, start, steps):
    """Yield (snapshot index, mesh, state); no future GT is read by this iterator.

    A frame at t+1 carries the output mesh chosen from the input state at t.
    The mesh may be updated again before advancing the next physical step.
    """
    if start < 0 or steps < 1 or start + steps >= source.num_steps:
        raise ValueError('Rollout requires start >= 0, steps >= 1, and start+steps < snapshot count')
    model.eval()
    # Keep grad mode scoped to computations, not across generator yields.
    with torch.no_grad():
        centers, state, levels = runtime.initial_snapshot(source, start)
        mesh = runtime.predict_mesh(source, centers, state, levels, 0)
        state, _ = runtime.gt(source, start, mesh, initial=True)
    yield start, mesh, state
    for step in range(steps):
        with torch.no_grad():
            if step > 0 and step % cfg['train']['runtime_mesh']['update_every_steps'] == 0:
                old = mesh
                mesh = runtime.predict_mesh(source, mesh.centers, state, mesh.levels, step, previous_mesh=old)
                state = runtime.remap(old, state, mesh)
            index = start + step
            dt = float(source.times[index+1] - source.times[index])
            _, predicted = model_step(model, state, None, mesh, source, cfg, runtime,
                                      mu, sigma, dt, dt_ref, step)
            if not torch.isfinite(predicted).all():
                raise FloatingPointError(f'Non-finite prediction at snapshot {index+1}')
            state = carry_state(predicted, cfg)
            if not torch.isfinite(state).all():
                raise FloatingPointError(f'Non-finite carried state at snapshot {index+1}')
        yield index + 1, mesh, state
