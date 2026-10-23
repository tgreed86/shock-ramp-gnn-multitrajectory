#!/usr/bin/env python3
"""Export approximate Euler face transports from every finest uniform GT cache.

No training split, model, or feature normalization is used. The spatial metric
is the normalized Cartesian cache bbox, as in the current DEC operators, NOT
the original solver's physical/mapped geometry. Transport = normal Euler flux
density * face length. Interval means use endpoint trapezoidal quadrature;
multiply by the saved dt to obtain time-integrated transport.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import h5py
import numpy as np
import torch

from cache_data import CacheTrajectory, file_sha256, validate_static_and_shapes
from data import uniform_cache_files, validate_cache
from flux import BOUNDARY_SIDES, build_flux_geometry, load_boundary_segments, nearest_boundary_side
from gt_cache import feature_location
from mesh_geometry import dec_edge_attr_for_dyadic_quads


SCHEMA = 'shock-ramp-face-transports-v1'
CHANNELS = ['density', 'x_momentum', 'y_momentum', 'energy']


@dataclass
class Faces:
    centers: torch.Tensor
    area: torch.Tensor
    owner: torch.Tensor
    neighbor: torch.Tensor
    normal: torch.Tensor
    length: torch.Tensor
    midpoint: torch.Tensor
    lattice_key: torch.Tensor
    boundary_side: torch.Tensor
    boundary_open: torch.Tensor
    interior_count: int


def build_faces(centers, levels, cfg, spec):
    """Canonical +x/+y interior faces, then outward-oriented boundary faces."""
    h, w = cfg['data']['H'], cfg['data']['W']
    level, rr = cfg['policy']['max_level'], cfg['policy']['refine_ratio']
    if not torch.all(levels == level):
        raise ValueError('Face targets require a uniformly finest-level cache')
    x0, x1, y0, y1 = cfg['data']['bbox']
    height, width = h * rr**level, w * rr**level
    dx, dy = (x1-x0)/width, (y1-y0)/height
    ij = ((centers.double() - centers.new_tensor([x0, y0]).double()) /
          centers.new_tensor([dx, dy]).double()).floor().long()
    table = torch.full((height, width), -1, dtype=torch.long)
    table[ij[:, 1], ij[:, 0]] = torch.arange(len(centers))
    pairs = []
    for a, b in ((table[:, :-1], table[:, 1:]), (table[:-1], table[1:])):
        keep = (a >= 0) & (b >= 0)
        pairs.append(torch.stack([a[keep], b[keep]]))
    forward = torch.cat(pairs, dim=1)
    edges = torch.cat([forward, forward.flip(0)], dim=1)
    ea = dec_edge_attr_for_dyadic_quads(centers, levels, edges,
        dx0=(x1-x0)/w, dy0=(y1-y0)/h, refine_ratio=rr)
    segments = load_boundary_segments(str(spec), cfg['data']['bbox'])
    open_sides = cfg.get('model', {}).get('open_boundary_sides', ['left', 'right', 'top'])
    if not isinstance(open_sides, list) or any(s not in BOUNDARY_SIDES for s in open_sides):
        raise ValueError('model.open_boundary_sides must contain left/right/bottom/top names')
    g = build_flux_geometry(centers, levels, edges, ea, dx0=(x1-x0)/w, dy0=(y1-y0)/h,
        bbox=cfg['data']['bbox'], refine_ratio=rr, boundary_segments=segments, open_boundary_sides=open_sides)
    src, dst = edges[:, g.forward_edges]
    owner = torch.cat([src, g.boundary_owner])
    neighbor = torch.cat([dst, torch.full_like(g.boundary_owner, -1)])
    normal = torch.cat([ea[g.forward_edges, :2], g.boundary_normal])
    length = torch.cat([g.face_length, g.boundary_length])
    midpoint = centers[owner] + normal * centers.new_tensor([dx, dy]) / 2
    # Stable integer keys for AMR aggregation: [axis, row, col]. For x faces,
    # col is a grid line and row is a segment; for y faces it is the reverse.
    axis = normal[:, 1].abs().long()
    cell = ij[owner]
    col = cell[:, 0] + (normal[:, 0] > 0).long()
    row = cell[:, 1] + (normal[:, 1] > 0).long()
    key = torch.stack([axis, row, col], dim=1)
    if len(torch.unique(key, dim=0)) != len(key):
        raise ValueError('Duplicate face lattice keys')
    side = torch.cat([torch.full_like(src, -1), nearest_boundary_side(midpoint[len(src):], segments)])
    opened = torch.cat([torch.zeros_like(src, dtype=torch.bool), g.boundary_open])
    return Faces(centers, g.area, owner, neighbor, normal, length, midpoint, key, side, opened, len(src))


def primitive(state, gamma, rho_floor, p_floor):
    if state.ndim != 2 or state.shape[1] != 4 or not torch.isfinite(state).all():
        raise ValueError('Expected finite [N,4] conserved states')
    if torch.any(state[:, 0] <= 0):
        raise ValueError('Nonpositive GT density cannot define an Euler flux')
    rho = state[:, 0].clamp_min(rho_floor)
    u = torch.cat([rho[:, None], state[:, 1:]], dim=1)
    velocity = u[:, 1:3] / rho[:, None]
    pressure = (gamma - 1) * (u[:, 3] - .5 * (u[:, 1:3] * velocity).sum(1))
    p = pressure.clamp_min(p_floor)
    sound = (gamma * p / rho).sqrt()
    counts = torch.stack([(state[:, 0] < rho_floor).sum(), (pressure < p_floor).sum()])
    return u, velocity, p, sound, counts


def normal_euler_flux(u, velocity, pressure, normal):
    un = (velocity * normal).sum(1)
    return torch.cat([(u[:, 0] * un)[:, None],
        u[:, 1:3] * un[:, None] + pressure[:, None] * normal,
        ((u[:, 3] + pressure) * un)[:, None]], dim=1), un


@torch.no_grad()
def reconstruct_transports(state, faces, *, gamma=1.4, rho_floor=1e-6, p_floor=0.):
    """Piecewise-constant Rusanov interior, extrapolated open, pressure-only wall.

    No learned limiter, normalization, artificial PARC diffusion, or velocity
    clipping is applied. Evaluate in float64; the exporter stores float32.
    """
    u, velocity, pressure, sound, counts = primitive(state.double(), gamma, rho_floor, p_floor)
    src, n = faces.owner, faces.interior_count
    normal = faces.normal.double()
    result, un_left = normal_euler_flux(u[src], velocity[src], pressure[src], normal)
    dst = faces.neighbor[:n]
    right, un_right = normal_euler_flux(u[dst], velocity[dst], pressure[dst], normal[:n])
    speed = torch.maximum(un_left[:n].abs() + sound[src[:n]], un_right.abs() + sound[dst])
    result[:n] = .5 * (result[:n] + right) - .5 * speed[:, None] * (u[dst] - u[src[:n]])
    wall = (faces.neighbor < 0) & ~faces.boundary_open
    result[wall] = 0
    result[wall, 1:3] = pressure[src[wall], None] * normal[wall]
    transport = result * faces.length.double()[:, None]
    if not torch.isfinite(transport).all() or transport.abs().max() > torch.finfo(torch.float32).max:
        raise ValueError('Non-finite or nonrepresentable face transport')
    return transport.float(), counts


def divergence_rate(transport, faces):
    """Finite-volume dU/dt; interior transfers cancel in the area-weighted sum."""
    t = transport.double()
    balance = t.new_zeros((len(faces.centers), 4))
    balance.index_add_(0, faces.owner, -t)
    n = faces.interior_count
    balance.index_add_(0, faces.neighbor[:n], t[:n])
    return balance / faces.area.double()[:, None]


def export_source(source, cfg, output, *, gamma, rho_floor, p_floor, max_intervals=None,
                  skip_existing=False):
    """Stream one trajectory; publish atomically and never replace existing data."""
    level = cfg['policy']['max_level']
    path = source.cache_files[level]
    validate_cache(source, cfg)
    centers = validate_static_and_shapes(path, level, cfg, source.num_steps)
    with h5py.File(path, 'r') as inp:
        levels = torch.from_numpy(inp['static/pred_levels'][...]).long()
    faces = build_faces(centers, levels, cfg, source.spec)
    count = min(source.num_steps - 1, max_intervals or source.num_steps - 1)
    times = source.times[:count+1].numpy()
    provenance = dict(schema=SCHEMA, trajectory=source.name, level=level,
        cache_sha256=file_sha256(path), mesh_spec_sha256=file_sha256(source.spec),
        timestamps_sha256=hashlib.sha256(source.times.numpy().tobytes()).hexdigest(),
        snapshots=count+1, total_source_snapshots=source.num_steps, channels=CHANNELS,
        H=cfg['data']['H'], W=cfg['data']['W'], bbox=cfg['data']['bbox'],
        refine_ratio=cfg['policy']['refine_ratio'], spatial_metric='normalized_cache_cartesian',
        reconstruction='first_order', numerical_flux='rusanov', gamma=gamma,
        rho_floor=rho_floor, p_floor=p_floor, velocity_clipping=False, diffusion=False,
        open_boundary='interior_state_extrapolation', wall_boundary='pressure_only_slip_wall',
        open_boundary_sides=cfg.get('model', {}).get('open_boundary_sides', ['left', 'right', 'top']),
        temporal_average='endpoint_trapezoid', dtype='float32',
        orientation='interior:+x/+y owner_to_neighbor; boundary:outward_from_owner',
        transport_definition='normal_Euler_flux_density * normalized_face_length; no dt or sigma factor',
        model_conversion='model_transport = cached_transport * dt_ref / (sigma * model_length_scale**2)',
        interval_integral='interval_mean_transport[n] * dt[n]',
        approximate=True)
    encoded = json.dumps(provenance, sort_keys=True, allow_nan=False)
    output = Path(output)
    if output.exists():
        if skip_existing:
            with h5py.File(output, 'r') as old:
                if old.attrs.get('complete', False) and old.attrs.get('provenance_json') == encoded:
                    print(f'[FLUX] {source.name}: verified existing {output}', flush=True)
                    return output
            raise ValueError(f'Existing flux cache is incomplete or has different provenance: {output}')
        raise FileExistsError(f'Refusing to overwrite {output}; use --skip-existing for verified reuse')
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=output.name+'.', suffix='.partial', dir=output.parent)
    os.close(handle)
    try:
        with h5py.File(path, 'r') as inp, h5py.File(temporary, 'w') as out:
            out.attrs.update(schema=SCHEMA, complete=False, provenance_json=encoded,
                source_cache=str(path), source_mesh_spec=str(source.spec), interior_count=faces.interior_count)
            out.create_dataset('times', data=times)
            out.create_dataset('dt', data=np.diff(times))
            geometry = out.create_group('geometry')
            for name in ('centers', 'area', 'owner', 'neighbor', 'normal', 'length', 'midpoint',
                         'lattice_key', 'boundary_side', 'boundary_open'):
                geometry.create_dataset(name, data=getattr(faces, name).numpy(), compression='gzip', compression_opts=1)
            geometry.attrs['lattice_key_layout'] = 'axis,row,col; axis=0:x-normal col=gridline; axis=1:y-normal row=gridline'
            geometry.attrs['boundary_side_names'] = json.dumps(BOUNDARY_SIDES)
            nf = len(faces.owner)
            options = dict(dtype='f4', chunks=(1, min(nf, 16384), 4), compression='gzip', compression_opts=1, shuffle=True)
            snapshot = out.create_dataset('snapshot_transport', shape=(count+1, nf, 4), **options)
            interval = out.create_dataset('interval_mean_transport', shape=(count, nf, 4), **options)
            diag = out.create_group('diagnostics')
            floor_counts = diag.create_dataset('floor_counts_rho_pressure', shape=(count+1, 2), dtype='i8')
            rmse = diag.create_dataset('rate_rmse', shape=(count, 4), dtype='f8')
            target_rms = diag.create_dataset('gt_rate_rms', shape=(count, 4), dtype='f8')
            residual = diag.create_dataset('global_balance_residual', shape=(count, 4), dtype='f8')
            diag.attrs['rate_rmse_definition'] = 'cellwise RMS(divergence_rate(interval_mean_transport) - (U[n+1]-U[n])/dt[n])'
            previous_state = previous_transport = None
            for t in range(count+1):
                group, key = feature_location(t)
                state = torch.from_numpy(np.asarray(inp[f't{group:05d}/{key}'][...], dtype=np.float32))
                transport, floors = reconstruct_transports(state, faces, gamma=gamma, rho_floor=rho_floor, p_floor=p_floor)
                snapshot[t] = transport.numpy()
                floor_counts[t] = floors.numpy()
                if t:
                    mean = ((previous_transport.double() + transport.double()) * .5).float()
                    interval[t-1] = mean.numpy()
                    observed = (state.double() - previous_state.double()) / (times[t] - times[t-1])
                    error = divergence_rate(mean, faces) - observed
                    rmse[t-1] = error.square().mean(0).sqrt().numpy()
                    target_rms[t-1] = observed.square().mean(0).sqrt().numpy()
                    residual[t-1] = (error * faces.area.double()[:, None]).sum(0).numpy()
                previous_state, previous_transport = state, transport
            out.attrs['complete'] = True
        # Hard-link publication is atomic and fails if another process created
        # the destination. Temporary and final paths are on the same filesystem.
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f'[FLUX] {source.name}: {count+1} snapshots, {nf} faces -> {output}', flush=True)
    return output


def export_all(cfg, output_dir, *, trajectories=None, gamma=None, rho_floor=None, p_floor=None,
               max_intervals=None, skip_existing=False):
    """All cache families by default, including trajectories excluded in training."""
    level = cfg['policy']['max_level']
    if cfg['features']['use_columns'] != [0, 1, 2, 3]:
        raise ValueError('Flux export requires channels [density, x_momentum, y_momentum, energy] in that order')
    gamma = float(cfg['loss'].get('gamma', 1.4) if gamma is None else gamma)
    rho_floor = float(cfg['loss'].get('rho_floor', 1e-6) if rho_floor is None else rho_floor)
    p_floor = float(cfg['loss'].get('p_floor', 0.) if p_floor is None else p_floor)
    if not np.isfinite([gamma, rho_floor, p_floor]).all() or gamma <= 1 or rho_floor <= 0 or p_floor < 0:
        raise ValueError('Require finite gamma>1, rho_floor>0, and p_floor>=0')
    if max_intervals is not None and (isinstance(max_intervals, bool) or max_intervals < 1):
        raise ValueError('max_intervals must be positive')
    root = Path(cfg['train']['runtime_mesh']['multires_gt_lookup']['root'])
    families = sorted(d for d in root.glob('p-*_angle-*') if d.is_dir())
    if not families:
        raise ValueError(f'No uniform cache families found under {root}')
    if trajectories:
        unknown = set(trajectories) - {d.name for d in families}
        if unknown:
            raise ValueError(f'Unknown trajectories: {sorted(unknown)}')
        families = [d for d in families if d.name in trajectories]
    metadata = json.loads(Path(cfg['data']['timestamps_path']).read_text())
    if metadata.get('schema') != 1:
        raise ValueError('Unsupported timestamp metadata schema')
    paths = []
    for family in families:
        match = re.fullmatch(r'p-([0-9p.]+)_angle-([0-9.]+)', family.name)
        if not match:
            raise ValueError(f'Invalid trajectory name: {family.name}')
        record = metadata['trajectories'][family.name]
        times = torch.tensor(record['times'], dtype=torch.float64)
        if times.ndim != 1 or len(times) < 2 or not torch.isfinite(times).all() or not (times.diff() > 0).all():
            raise ValueError(f'Invalid physical timestamps for {family.name}')
        spec = Path(cfg['mesh']['starting_mesh_path']) / (
            f'wedge_mesh_spec_{cfg["data"]["W"]}x{cfg["data"]["H"]}_L{level}_ramp-angle-{float(match[2]):g}.pt')
        files = uniform_cache_files(root, family.name, level)
        source = CacheTrajectory(family, spec, {level: files[level]}, times, {},
                                 record['source_size'], {}, record['source_relative_path'])
        paths.append(export_source(source, cfg, Path(output_dir)/family.name/f'face_transports_L{level}.h5',
            gamma=gamma, rho_floor=rho_floor, p_floor=p_floor, max_intervals=max_intervals, skip_existing=skip_existing))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config_cache_only.json')))
    parser.add_argument('--output-dir', required=True, help='New output root; input caches are never modified')
    parser.add_argument('--cache-root', help='Override the uniform GT cache root')
    parser.add_argument('--timestamps', help='Override cache_timestamps.json')
    parser.add_argument('--mesh-specs', help='Override the mesh-spec directory')
    parser.add_argument('--trajectory', action='append', help='Optional exact family name; repeat to select several')
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--rho-floor', type=float)
    parser.add_argument('--p-floor', type=float)
    parser.add_argument('--max-intervals', type=int, help='Smoke-test cap; omit to export every interval')
    parser.add_argument('--skip-existing', action='store_true', help='Reuse only completed files with matching provenance')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    path = Path(args.config).resolve()
    cfg = json.loads(path.read_text())
    for section, key, override in [('data', 'timestamps_path', args.timestamps),
                                   ('mesh', 'starting_mesh_path', args.mesh_specs)]:
        value = override or cfg[section].get(key)
        if not value:
            parser.error(f'{section}.{key} or its CLI override is required')
        p = Path(os.path.expandvars(value)).expanduser()
        cfg[section][key] = str(p.resolve() if override or p.is_absolute() else (path.parent/p).resolve())
    lookup = cfg['train']['runtime_mesh']['multires_gt_lookup']
    p = Path(os.path.expandvars(args.cache_root or lookup['root'])).expanduser()
    lookup['root'] = str(p.resolve() if args.cache_root or p.is_absolute() else (path.parent/p).resolve())
    torch.set_num_threads(args.cpu_threads)
    paths = export_all(cfg, args.output_dir, trajectories=args.trajectory, gamma=args.gamma,
        rho_floor=args.rho_floor, p_floor=args.p_floor, max_intervals=args.max_intervals, skip_existing=args.skip_existing)
    print(f'[FLUX] Completed {len(paths)} trajectory files.', flush=True)


if __name__ == '__main__':
    main()
