"""Conservative restriction of cached finest-grid transports to runtime faces."""
from collections import OrderedDict
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from cache_data import file_sha256
from precompute_face_transports import CHANNELS, SCHEMA, Faces, build_faces


def validate_flux_supervision_config(cfg):
    weight = cfg['loss'].get('flux_supervision_weight', 0.)
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not np.isfinite(weight) or weight < 0):
        raise ValueError('loss.flux_supervision_weight must be finite and nonnegative')
    if weight > 0:
        if cfg['model']['name'] != 'FluxGraphNet':
            raise ValueError('Flux supervision requires model.name=FluxGraphNet')
        root = cfg['data'].get('flux_cache_root')
        if not isinstance(root, str) or not root.strip():
            raise ValueError('Flux supervision requires data.flux_cache_root')
        if not cfg['train']['runtime_mesh']['multires_gt_lookup']['enabled']:
            raise ValueError('Flux supervision requires multires_gt_lookup.enabled=true')


def _read_faces(handle):
    return Faces(**{f.name: (int(handle.attrs[f.name]) if f.name == 'interior_count'
                            else torch.from_numpy(handle['geometry/'+f.name][...]))
                    for f in fields(Faces)})


def _validate_file(handle, source, cfg, faces):
    """Fail on stale, partial, or differently oriented targets before training."""
    label = f'Flux cache {handle.filename}'
    if handle.attrs.get('schema') != SCHEMA or not handle.attrs.get('complete', False):
        raise ValueError(f'{label}: unsupported schema or incomplete export')
    p = json.loads(handle.attrs['provenance_json'])
    level = cfg['policy']['max_level']
    identity = getattr(source, 'identity', {})
    cache_hash = identity.get('cache_sha256', {}).get(str(level)) or file_sha256(source.cache_files[level])
    spec_hash = identity.get('mesh_spec_sha256') or file_sha256(source.spec)
    expected = dict(schema=SCHEMA, trajectory=source.name, level=level, channels=CHANNELS,
        snapshots=source.num_steps, total_source_snapshots=source.num_steps,
        cache_sha256=cache_hash, mesh_spec_sha256=spec_hash,
        timestamps_sha256=hashlib.sha256(source.times.cpu().numpy().tobytes()).hexdigest(),
        H=cfg['data']['H'], W=cfg['data']['W'], bbox=list(cfg['data']['bbox']),
        refine_ratio=cfg['policy']['refine_ratio'], spatial_metric='normalized_cache_cartesian',
        temporal_average='endpoint_trapezoid',
        orientation='interior:+x/+y owner_to_neighbor; boundary:outward_from_owner',
        transport_definition='normal_Euler_flux_density * normalized_face_length; no dt or sigma factor',
        open_boundary_sides=cfg['model'].get('open_boundary_sides', ['left', 'right', 'top']))
    for key, value in expected.items():
        if p.get(key) != value:
            raise ValueError(f'{label}: incompatible {key}; regenerate the full flux cache for this dataset')
    times = source.times.cpu().numpy()
    if not np.array_equal(handle['times'][...], times) or not np.array_equal(handle['dt'][...], np.diff(times)):
        raise ValueError(f'{label}: timestamps/dt differ from the trajectory')
    if handle['interval_mean_transport'].shape != (source.num_steps-1, len(faces.owner), 4):
        raise ValueError(f'{label}: invalid interval_mean_transport shape')
    # Rebuild from the checked source geometry: neither array ordering nor face
    # orientation may be guessed from matching dimensions alone.
    with h5py.File(source.cache_files[level], 'r') as original:
        centers = torch.from_numpy(original['static/pred_centers'][...]).float()
        levels = torch.from_numpy(original['static/pred_levels'][...]).long()
    expected_faces = build_faces(centers, levels, cfg, source.spec)
    for f in fields(Faces):
        actual, wanted = getattr(faces, f.name), getattr(expected_faces, f.name)
        equal = actual == wanted if isinstance(wanted, int) else torch.equal(actual, wanted)
        if not equal:
            raise ValueError(f'{label}: incompatible geometry/{f.name}')
    return p


def _match(keys, queries):
    """Indices of unique runtime faces, with no missing-face interpolation."""
    ordered, order = torch.sort(keys)
    pos = torch.searchsorted(ordered, queries)
    if len(queries) and (not len(keys) or torch.any(pos >= len(keys))):
        raise ValueError('Flux target mapping found an unmatched runtime face')
    if not torch.equal(ordered[pos], queries):
        raise ValueError('Flux target mapping found an unmatched runtime face')
    return order[pos]


@torch.no_grad()
def restriction_plan(faces, mesh, geometry, owners, cfg):
    """Return fine face IDs and destination IDs (interior first, then exterior).

    Fine faces inside one adaptive cell cancel. Every remaining fine face goes
    to exactly one adaptive subface; exterior normals retain their outward sign.
    All planning/target sums use CPU; only the restricted result goes to the GPU.
    """
    owners = owners.detach().cpu()
    h, w = cfg['data']['H'], cfg['data']['W']
    factor = cfg['policy']['refine_ratio'] ** cfg['policy']['max_level']
    x0, x1, y0, y1 = cfg['data']['bbox']
    col = ((faces.centers[:, 0].double()-x0) * (w*factor)/(x1-x0)).floor().long()
    row = ((faces.centers[:, 1].double()-y0) * (h*factor)/(y1-y0)).floor().long()
    inside = torch.zeros(h*w*factor**2, dtype=torch.bool)
    inside[row*(w*factor)+col] = True
    if not torch.equal(inside, owners >= 0):
        raise ValueError('Flux supervision requires identical finest-cache and runtime mesh domains')
    cell = owners[row*(w*factor)+col]
    count, ni = len(mesh.centers), faces.interior_count
    src, dst = cell[faces.owner[:ni]], cell[faces.neighbor[:ni]]
    fine_interior = (src != dst).nonzero(as_tuple=True)[0]
    a, b = mesh.edges[:, geometry.forward_edges].detach().cpu()
    target_interior = _match(a*count+b, (src*count+dst)[fine_interior])

    def side(normal):
        return torch.where(normal[:, 0] != 0, (normal[:, 0] > 0).long(), 2+(normal[:, 1] > 0).long())

    fine_boundary = torch.arange(ni, len(faces.owner))
    boundary_keys = geometry.boundary_owner.cpu()*4 + side(geometry.boundary_normal.cpu())
    target_boundary = _match(boundary_keys, cell[faces.owner[ni:]]*4 + side(faces.normal[ni:]))
    if not torch.equal(geometry.boundary_open.cpu()[target_boundary], faces.boundary_open[ni:]):
        raise ValueError('Flux target boundary types differ from runtime open/wall classification')
    ids = torch.cat([fine_interior, fine_boundary])
    targets = torch.cat([target_interior, len(a)+target_boundary])
    lengths = torch.cat([geometry.face_length, geometry.boundary_length]).detach().cpu()
    summed = torch.zeros_like(lengths)
    summed.index_add_(0, targets, faces.length[ids].to(summed))
    if not torch.allclose(summed, lengths, rtol=1e-5, atol=1e-8):
        raise ValueError('Flux target mapping does not cover each runtime face exactly')
    return ids, targets, len(lengths)


class FluxSupervision:
    """Bound open files and share the runtime CPU tensor-cache byte budget."""
    def __init__(self, cfg, cpu_cache):
        self.cfg, self.cpu_cache = cfg, cpu_cache
        self.readers, self.verified = OrderedDict(), {}

    def close(self):
        for handle, _ in self.readers.values():
            handle.close()
        self.readers.clear()

    def reader(self, source):
        level = self.cfg['policy']['max_level']
        path = Path(self.cfg['data']['flux_cache_root'])/source.name/f'face_transports_L{level}.h5'
        if path in self.readers:
            self.readers.move_to_end(path)
            return path, *self.readers[path]
        limit = self.cfg['train']['runtime_mesh']['multires_gt_lookup']['max_open_trajectories']
        while len(self.readers) >= limit:
            _, (old, _) = self.readers.popitem(last=False)
            old.close()
        if not path.is_file():
            raise FileNotFoundError(f'Missing flux targets: {path}; run precompute_face_transports.py first')
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        handle = h5py.File(path, 'r')
        try:
            faces = _read_faces(handle)
            if path not in self.verified:
                provenance = _validate_file(handle, source, self.cfg, faces)
                self.verified[path] = (signature, dict(sha256=file_sha256(path), provenance=provenance))
            elif self.verified[path][0] != signature:
                raise ValueError(f'Flux targets changed during this run: {path}')
        except Exception:
            handle.close()
            raise
        self.readers[path] = handle, faces
        return path, handle, faces

    def manifest(self, sources):
        records = {}
        for source in sources:
            path, _, _ = self.reader(source)
            records[source.name] = self.verified[path][1]
        return {'objective': 'raw-model-transport-mse-v1', 'files': records}

    @torch.no_grad()
    def targets(self, source, local_t, mesh, runtime, sigma, dt_ref, length_scale):
        if not isinstance(local_t, int) or not 0 <= local_t < source.num_steps-1:
            raise IndexError(f'{source.name}: flux supervision requires an absolute local interval index')
        path, handle, faces = self.reader(source)
        key = ('flux', str(path), local_t)
        values = self.cpu_cache.get(key)
        if values is None:
            values = torch.from_numpy(handle['interval_mean_transport'][local_t]).float()
            if not torch.isfinite(values).all():
                raise ValueError(f'Non-finite flux targets: {path} interval={local_t}')
            self.cpu_cache.put(key, values)
        plan_key = ('flux', str(path))
        if plan_key not in mesh.lookup_plans:
            mesh.lookup_plans[plan_key] = restriction_plan(faces, mesh, runtime.flux_geometry(mesh, source),
                                                        runtime.remap_owners(mesh), self.cfg)
        ids, destinations, size = mesh.lookup_plans[plan_key]
        restricted = values.new_zeros(size, 4)
        restricted.index_add_(0, destinations, values[ids])
        # The cached quantity is a transport rate. The interval's actual dt is
        # used by the Euler update, NOT multiplied into this rate target again.
        return restricted.to(sigma) * (dt_ref / (sigma * length_scale**2))


def supervised_flux_loss(transport, exterior, targets, geometry, model):
    """MSE of normalized integrated transports over trainable face/channels.

    Hard boundary masks remain hard constraints; excluded components (including
    disabled boundary heads) cannot learn and must not add a constant penalty.
    No rate-loss RMS weights apply to this separate, explicitly weighted loss.
    """
    prediction = torch.cat([transport, exterior])
    mask = torch.ones_like(prediction)
    boundary = mask[len(transport):]
    for selected, head, channels in (
            (geometry.boundary_open, model.open_boundary_head, model.open_boundary_mask),
            (~geometry.boundary_open, model.ramp_boundary_head, model.ramp_boundary_mask)):
        boundary[selected] = channels if head is not None else 0
    return ((prediction-targets).square()*mask).sum() / mask.sum().clamp_min(1)
