"""Source-aware wedge geometry, bounded HDF5 lookups, and differentiable remapping."""
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

import kernels as k
from geometry import apply_idw_map, build_idw_map, dynamic_cells_from_parent_masks
from gt_cache import MultiResGtLookup, TensorLRU
from mesh_geometry import _load_wedge_path_from_spec, dec_edge_attr_for_dyadic_quads
from policy import predict_masks_hierarchical_from_gt_gradients


@dataclass
class Mesh:
    centers: torch.Tensor
    levels: torch.Tensor
    parents: torch.Tensor
    edges: torch.Tensor
    parent_mask: torch.Tensor
    edge_attr: torch.Tensor
    refinement_masks: dict | None = None
    lookup_plans: dict = field(default_factory=dict)


class Runtime:
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, torch.device(device)
        self.h, self.w = cfg['data']['H'], cfg['data']['W']
        self.bbox, self.rr = tuple(cfg['data']['bbox']), cfg['policy']['refine_ratio']
        self.dx = (self.bbox[1]-self.bbox[0])/self.w
        self.dy = (self.bbox[3]-self.bbox[2])/self.h
        self.rt = cfg['train']['runtime_mesh']
        self.lookup_cfg = self.rt['multires_gt_lookup']
        backend = self.lookup_cfg.get('backend', 'auto')
        use_cuda = backend == 'cuda' or (backend == 'auto' and self.device.type == 'cuda')
        if use_cuda and self.device.type != 'cuda':
            raise ValueError('CUDA GT lookup requires a CUDA training device; use backend=auto or cpu')
        self.lookup_device = self.device if use_cuda else torch.device('cpu')
        self.adjacency_backend = self.rt.get('adjacency_backend', 'auto')
        if self.adjacency_backend == 'auto':
            self.adjacency_backend = 'tensor' if self.device.type == 'cuda' else 'cpu'
        if self.adjacency_backend == 'tensor' and self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Tensor adjacency supports CPU and CUDA; use auto or cpu on MPS')
        self.geometry, self.lookups = {}, OrderedDict()
        self.cpu_features = TensorLRU(int(self.lookup_cfg.get('cpu_cache_mb', 2048) * 1024**2))
        self.cuda_features = TensorLRU(int(self.lookup_cfg.get('cuda_cache_mb', 512) * 1024**2))
        self.rc_grid = k._runtime_build_coarse_rc_grid(self.h, self.w, self.device)

    def close(self):
        for lookup in self.lookups.values():
            lookup.close()
        self.lookups.clear()
        self.cpu_features.clear()
        self.cuda_features.clear()

    def constraints(self, source):
        key = str(source.spec)
        if key not in self.geometry:
            wedge = _load_wedge_path_from_spec(key, self.cfg)
            self.geometry[key] = k._get_runtime_wedge_constraints(
                cfg=self.cfg, H=self.h, W=self.w, device=self.device, wedge_path=wedge)
        return self.geometry[key]

    def lookup(self, source):
        key = str(source.path)
        if key in self.lookups:
            self.lookups.move_to_end(key)
            return self.lookups[key]
        limit = self.lookup_cfg['max_open_trajectories']
        while len(self.lookups) >= limit:
            _, old = self.lookups.popitem(last=False)
            old.close()
        reader = MultiResGtLookup(
            level_files=source.cache_files, H=self.h, W=self.w, bbox=self.bbox,
            refine_ratio=self.rr, max_level=self.cfg['policy']['max_level'],
            cpu_cache=self.cpu_features, device_cache=self.cuda_features)
        self.lookups[key] = reader
        return reader

    def parents(self, centers, levels):
        row, col = k._cell_level_ij_from_centers(centers=centers, levels=levels, H=self.h,
                W=self.w, bbox=self.bbox, refine_ratio=self.rr)
        return k._runtime_parents_from_level_ij(levels=levels, row=row, col=col,
                H=self.h, W=self.w, refine_ratio=self.rr).view(-1)

    @torch.no_grad()
    def predict_mesh(self, source, centers, features, levels, step, previous_mesh=None):
        constraints = self.constraints(source)
        parents = self.parents(centers, levels)
        mask = k._mask_from_parent_indices(parents, self.h, self.w, self.device)
        if step > 0 and self.rt['reset_to_coarse_each_step']:
            mask = torch.zeros_like(mask)
        if self.rt['policy_backend'] == 'gradient_fast':
            masks = k._runtime_predict_masks_from_fast_gradients(
                centers_t=centers, level_t=levels, feat_policy=features.detach(), parents_t=parents,
                mask_t_parent=mask, cfg=self.cfg, H=self.h, W=self.w, dx=self.dx, dy=self.dy,
                device=self.device, physical_parent_mask=constraints['parent_intersect'].get(1),
                coarse_rc_grid=self.rc_grid,
                use_prev_state_masks=(step > 0 or self.cfg['policy']['hysteresis_prev_source'] == 'gt'))
        else:
            masks = predict_masks_hierarchical_from_gt_gradients(
                {'centers_t': centers, 'center_feat_t': features.detach(), 'dyn_feat_t': features.detach(),
                 'dyn_parents': parents, 'mask_t': mask.view(-1), 'level_t': levels},
                self.cfg, self.h, self.w, self.dx, self.dy, device=self.device)
        masks = k._runtime_apply_wedge_constraints_to_masks(masks_by_level=masks, constraints=constraints)
        if (self.rt.get('reuse_unchanged_mesh', True) and previous_mesh is not None
                and previous_mesh.refinement_masks is not None
                and masks.keys() == previous_mesh.refinement_masks.keys()
                and all(torch.equal(m, previous_mesh.refinement_masks[level]) for level, m in masks.items())):
            return previous_mesh
        c, l, p, _ = dynamic_cells_from_parent_masks(masks, self.h, self.w, *self.bbox, refine_ratio=self.rr)
        c, l, p, edges, mask = k._runtime_filter_mesh_with_wedge_constraints(
            pred_centers=c, pred_levels=l, parent_flat=p.view(-1).long(), cfg=self.cfg,
            H=self.h, W=self.w, device=self.device, constraints=constraints)
        if not len(c) or len(c) > self.rt['max_cells_per_step']:
            raise RuntimeError(f'{source.name}: predicted mesh has {len(c)} cells')
        ea = dec_edge_attr_for_dyadic_quads(c, l, edges, dx0=self.dx, dy0=self.dy, refine_ratio=self.rr)
        return Mesh(c, l, p, edges, mask, ea, refinement_masks=masks)

    def initial_snapshot(self, source, local_t):
        if source.backend == 'uniform_cache':
            return self.lookup(source).uniform_snapshot(self.cfg['policy']['max_level'], local_t, self.device)
        return source.snapshot(local_t, self.cfg, self.device)

    def idw_map(self, centers, queries):
        backend = self.rt['idw']['backend']
        device, chunk = k._select_idw_backend(src_n=len(centers), requested_chunk=self.rt['idw']['chunk'],
                                             out_device=self.device)
        if backend == 'kdtree':
            device = torch.device('cpu')
        return build_idw_map(queries.to(device), centers.to(device),
                             k=self.rt['idw']['k'], chunk=chunk, backend=backend)

    def idw(self, centers, features, queries):
        indices, weights = self.idw_map(centers, queries)
        device = indices.device
        return apply_idw_map(indices, weights, features.to(device)).to(self.device)

    @torch.no_grad()
    def gt(self, source, local_t, mesh, *, initial=False):
        if not 0 <= local_t < source.num_steps:
            raise IndexError(f'{source.name}: invalid local timestep {local_t}')
        if self.lookup_cfg['enabled']:
            reader = self.lookup(source)
            lookup_device = self.lookup_device
            key = (str(source.path), str(lookup_device))
            entry = mesh.lookup_plans.get(key)
            if entry is None or entry[0].owner is not reader.plan_token:
                plan = reader.prepare(mesh.centers, mesh.levels, device=lookup_device)
                entry = (plan, plan.missing.to(self.device))
                mesh.lookup_plans[key] = entry
            plan, missing = entry
            values = reader.gather(local_t, plan, out_device=self.device)
            if len(missing) and not self.lookup_cfg['fallback_to_idw']:
                raise RuntimeError(f'{source.name} t={local_t}: {len(missing)}/{len(mesh.centers)} cache misses')
            if not len(missing):
                return values, 0
        else:
            missing = torch.arange(len(mesh.centers), device=self.device)
            values = None
        if source.backend == 'uniform_cache':
            # The static finest-cache geometry makes this map reusable across timesteps.
            c, x, _ = reader.uniform_snapshot(self.cfg['policy']['max_level'], local_t, self.device)
            if plan.fallback_map is None:
                plan.fallback_map = self.idw_map(c, mesh.centers[missing])
            indices, weights = plan.fallback_map
            interpolated = apply_idw_map(indices, weights, x.to(indices.device)).to(self.device)
        else:
            c, x, _ = source.snapshot(local_t, self.cfg, self.device)
            interpolated = self.idw(c, x, mesh.centers[missing])
        if values is None:
            values = interpolated.new_zeros((len(mesh.centers), interpolated.shape[1]))
        values[missing] = interpolated
        return values, len(missing)

    def remap(self, old_mesh, features, new_mesh):
        if old_mesh is new_mesh:
            return features
        return k._map_pred_to_next_pred(old_mesh.centers, features, old_mesh.levels, old_mesh.parents,
                    new_mesh.centers, new_mesh.levels, new_mesh.parents, new_mesh.parent_mask, self.h, self.w,
                    knn_k=self.rt['idw']['k'], chunk=self.rt['idw']['chunk'], bbox=self.bbox,
                    refine_ratio=self.rr, knn_backend=self.rt['idw']['backend'])
