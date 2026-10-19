# Retained wedge and face-adjacency kernels; see PORTING.md.
from __future__ import annotations
from typing import Any
from typing import Dict
from matplotlib.path import Path
import hashlib
import numpy as np
import torch
_WEDGE_CLIP_LOOKUP_CACHE: Dict[str, Dict[str, Any]] = {}

def amr_cell_wh_area_from_levels(levels: torch.Tensor, *, dx0: float, dy0: float, refine_ratio: int=2):
    """
    levels: (N,) int64
    Returns:
      w: (N,) float32
      h: (N,) float32
      area: (N,) float32
      hx: (N,) float32
      hy: (N,) float32
    """
    rr = int(refine_ratio)
    if rr < 2:
        raise ValueError(f'refine_ratio must be >=2, got {refine_ratio}')
    lv = levels.to(dtype=torch.float32)
    scale = torch.pow(torch.tensor(float(rr), device=levels.device), lv)
    w = (float(dx0) / scale).to(torch.float32)
    h = (float(dy0) / scale).to(torch.float32)
    area = (w * h).to(torch.float32)
    hx = (0.5 * w).to(torch.float32)
    hy = (0.5 * h).to(torch.float32)
    return (w, h, area, hx, hy)

def dec_edge_attr_for_dyadic_quads(centers: torch.Tensor, levels: torch.Tensor, edge_index: torch.Tensor, *, dx0: float, dy0: float, refine_ratio: int=2):
    """
    Returns edge_attr: (E,5) float32 with columns:
      [nx, ny, face_len, dual_len, tau]
    """
    device = centers.device
    ei = edge_index.to(device=device, dtype=torch.int64)
    u = ei[0]
    v = ei[1]
    w, h, _area, hx, hy = amr_cell_wh_area_from_levels(levels, dx0=dx0, dy0=dy0, refine_ratio=refine_ratio)
    du = centers[v] - centers[u]
    dx = du[:, 0]
    dy = du[:, 1]
    exp_dx = hx[u] + hx[v]
    exp_dy = hy[u] + hy[v]
    err_x = (dx.abs() - exp_dx).abs()
    err_y = (dy.abs() - exp_dy).abs()
    is_lr = err_x <= err_y
    nx = torch.zeros_like(dx, dtype=torch.float32)
    ny = torch.zeros_like(dy, dtype=torch.float32)
    dual_len = torch.empty_like(dx, dtype=torch.float32)
    face_len = torch.empty_like(dx, dtype=torch.float32)
    nx[is_lr] = torch.sign(dx[is_lr]).to(torch.float32)
    dual_len[is_lr] = exp_dx[is_lr].to(torch.float32)
    face_len[is_lr] = torch.minimum(h[u[is_lr]], h[v[is_lr]]).to(torch.float32)
    inv = ~is_lr
    ny[inv] = torch.sign(dy[inv]).to(torch.float32)
    dual_len[inv] = exp_dy[inv].to(torch.float32)
    face_len[inv] = torch.minimum(w[u[inv]], w[v[inv]]).to(torch.float32)
    eps = torch.tensor(1e-12, device=device, dtype=torch.float32)
    tau = face_len / torch.maximum(dual_len, eps)
    edge_attr = torch.stack([nx, ny, face_len, dual_len, tau], dim=1).to(torch.float32)
    return edge_attr

def _wedge_lookup_cache_key(*, wedge_path, H: int, W: int, Lmax: int, refine_ratio: int, bbox: tuple[float, float, float, float], radius: float) -> str:
    verts = np.asarray(wedge_path.vertices, dtype=np.float64)
    vhash = hashlib.sha1(verts.tobytes()).hexdigest()[:20]
    xmin, xmax, ymin, ymax = [float(v) for v in bbox]
    return f'v={vhash}|H={int(H)}|W={int(W)}|Lmax={int(Lmax)}|rr={int(refine_ratio)}|bbox={xmin:.16g},{xmax:.16g},{ymin:.16g},{ymax:.16g}|r={float(radius):.16g}'

def _build_wedge_clip_level_lookup(*, wedge_path, H: int, W: int, Lmax: int, refine_ratio: int, bbox: tuple[float, float, float, float], radius: float) -> Dict[str, Any]:
    """
    Precompute per-level cell classification masks:
      - full[L][j,i]: all four corners inside wedge
      - intersect[L][j,i]: center-inside OR any-corner-inside
    """
    xmin, xmax, ymin, ymax = [float(v) for v in bbox]
    xspan = float(xmax - xmin)
    yspan = float(ymax - ymin)
    rr = int(refine_ratio)
    full_cpu: Dict[int, torch.Tensor] = {}
    intersect_cpu: Dict[int, torch.Tensor] = {}
    for l in range(0, int(Lmax) + 1):
        HH = int(H) * rr ** int(l)
        WW = int(W) * rr ** int(l)
        dxL = xspan / float(WW)
        dyL = yspan / float(HH)
        x_edges = xmin + np.arange(WW + 1, dtype=np.float64) * dxL
        y_edges = ymin + np.arange(HH + 1, dtype=np.float64) * dyL
        xv, yv = np.meshgrid(x_edges, y_edges, indexing='xy')
        v_pts = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1)
        v_in = wedge_path.contains_points(v_pts, radius=float(radius)).reshape(HH + 1, WW + 1)
        c00 = v_in[:-1, :-1]
        c10 = v_in[1:, :-1]
        c01 = v_in[:-1, 1:]
        c11 = v_in[1:, 1:]
        full = c00 & c10 & c01 & c11
        any_corner = c00 | c10 | c01 | c11
        x_ctr = xmin + (np.arange(WW, dtype=np.float64) + 0.5) * dxL
        y_ctr = ymin + (np.arange(HH, dtype=np.float64) + 0.5) * dyL
        xc, yc = np.meshgrid(x_ctr, y_ctr, indexing='xy')
        c_pts = np.stack([xc.reshape(-1), yc.reshape(-1)], axis=1)
        center_in = wedge_path.contains_points(c_pts, radius=float(radius)).reshape(HH, WW)
        intersect = center_in | any_corner
        full_cpu[int(l)] = torch.from_numpy(full.astype(np.bool_))
        intersect_cpu[int(l)] = torch.from_numpy(intersect.astype(np.bool_))
    return {'full_cpu': full_cpu, 'intersect_cpu': intersect_cpu, 'device_cache': {}}

def _get_wedge_clip_level_lookup(*, wedge_path, H: int, W: int, Lmax: int, refine_ratio: int, bbox: tuple[float, float, float, float], radius: float) -> Dict[str, Any]:
    key = _wedge_lookup_cache_key(wedge_path=wedge_path, H=H, W=W, Lmax=Lmax, refine_ratio=refine_ratio, bbox=bbox, radius=radius)
    entry = _WEDGE_CLIP_LOOKUP_CACHE.get(key, None)
    if entry is None:
        entry = _build_wedge_clip_level_lookup(wedge_path=wedge_path, H=H, W=W, Lmax=Lmax, refine_ratio=refine_ratio, bbox=bbox, radius=radius)
        _WEDGE_CLIP_LOOKUP_CACHE[key] = entry
        print(f'[WEDGE-LOOKUP] built cache for H={int(H)} W={int(W)} Lmax={int(Lmax)} rr={int(refine_ratio)}', flush=True)
    return entry

def _lookup_level_masks_on_device(lookup: Dict[str, Any], device: torch.device) -> tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    dev = torch.device(device)
    if dev.type == 'cpu':
        return (lookup['full_cpu'], lookup['intersect_cpu'])
    dkey = f'{dev.type}:{(dev.index if dev.index is not None else -1)}'
    dcache = lookup.get('device_cache', {})
    got = dcache.get(dkey, None)
    if got is None:
        got = {'full': {L: m.to(device=dev, dtype=torch.bool) for L, m in lookup['full_cpu'].items()}, 'intersect': {L: m.to(device=dev, dtype=torch.bool) for L, m in lookup['intersect_cpu'].items()}}
        dcache[dkey] = got
        lookup['device_cache'] = dcache
    return (got['full'], got['intersect'])

@torch.no_grad()
def build_amr_face_adjacency_edges(centers: torch.Tensor, levels: torch.Tensor, H: int, W: int, bbox: tuple[float, float, float, float], *, max_L_guard: int=6, return_edge_attr: bool=True, refine_ratio: int=2, backend: str='auto'):
    if backend not in ('auto', 'cpu', 'tensor'):
        raise ValueError(f'Unknown adjacency backend: {backend}')
    use_tensor = backend == 'tensor' or (backend == 'auto' and centers.device.type == 'cuda')
    if use_tensor and centers.device.type not in ('cpu', 'cuda'):
        raise ValueError('Tensor adjacency supports CPU and CUDA; use auto or cpu on MPS')
    builder = _build_amr_face_adjacency_tensor if use_tensor else _build_amr_face_adjacency_cpu
    return builder(centers, levels, H, W, bbox, max_L_guard=max_L_guard,
                   return_edge_attr=return_edge_attr, refine_ratio=refine_ratio)


@torch.no_grad()
def _build_amr_face_adjacency_tensor(centers, levels, H, W, bbox, *, max_L_guard=6,
                                     return_edge_attr=True, refine_ratio=2):
    """Rasterize leaves by level, rather than looping over cells on the host.

    Input must be a non-overlapping dyadic leaf mesh, as produced by the runtime.
    Preserve the reference's four direction blocks and sorted edge order within
    each block so message-passing reduction order does not change unnecessarily.
    """
    device, count = centers.device, len(centers)
    if not count:
        edges = torch.empty((2, 0), dtype=torch.long, device=device)
        attrs = torch.empty((0, 5), dtype=torch.float32, device=device)
        return (edges, attrs) if return_edge_attr else edges
    rr = int(refine_ratio)
    if rr < 2:
        raise ValueError('refine_ratio must be >=2')
    levels = levels.long()
    max_level = int(levels.max().item())
    if max_level > max_L_guard or max_level < 0:
        raise ValueError(f'Invalid/maximal refinement {max_level}; adjacency guard={max_L_guard}')
    xmin, xmax, ymin, ymax = map(float, bbox)
    dx0, dy0 = (xmax-xmin)/W, (ymax-ymin)/H
    occupancy = torch.full((H, W), -1, dtype=torch.long, device=device)
    for level in range(max_level + 1):
        if level:
            occupancy = occupancy.repeat_interleave(rr, 0).repeat_interleave(rr, 1)
        ids = (levels == level).nonzero(as_tuple=True)[0]
        xy = centers[ids].float()
        col = ((xy[:, 0]-xmin)/(dx0/rr**level)).floor().long().clamp(0, W*rr**level-1)
        row = ((xy[:, 1]-ymin)/(dy0/rr**level)).floor().long().clamp(0, H*rr**level-1)
        occupancy[row, col] = ids

    def faces(a, b, length):
        keep = (a != b) & (a >= 0) & (b >= 0)
        a, b = a[keep], b[keep]
        keys, counts = torch.unique(a*count+b, sorted=True, return_counts=True)
        reverse, order = torch.sort((keys % count)*count + keys//count)
        return (keys, reverse), (counts.float()*length, counts[order].float()*length)

    horizontal, lengths_x = faces(occupancy[:, :-1], occupancy[:, 1:], dy0/rr**max_level)
    vertical, lengths_y = faces(occupancy[:-1, :], occupancy[1:, :], dx0/rr**max_level)
    keys = torch.cat((*horizontal, *vertical))
    edges = torch.stack((keys//count, keys % count))
    if not return_edge_attr:
        return edges
    delta = centers.float()[edges[1]] - centers.float()[edges[0]]
    dist = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-12)
    use_x = delta[:, 0].abs() >= delta[:, 1].abs()
    nx = torch.where(use_x, delta[:, 0].sign(), 0.)
    ny = torch.where(use_x, 0., delta[:, 1].sign())
    length = torch.cat((*lengths_x, *lengths_y))
    return edges, torch.stack((nx, ny, length, dist, length/dist), dim=1)


def _build_amr_face_adjacency_cpu(centers: torch.Tensor, levels: torch.Tensor, H: int, W: int, bbox: tuple[float, float, float, float], *, max_L_guard: int=6, return_edge_attr: bool=True, refine_ratio: int=2):
    """
    Face-adjacency edge builder for axis-aligned AMR quads.

    Uses a max-grid occupancy rasterization to detect which *original* cells share a face,
    including coarse–fine partial overlaps. Nodes remain original cells (true centers).

    Returns:
      edge_index: (2,E) int64, directed, includes both directions, deduplicated, no self-loops.

    If return_edge_attr=True, also returns:
      edge_attr: (E,5) float32 columns:
        [nx, ny, face_len, center_dist, w_diff]
      where:
        (nx,ny)     = unit outward normal for the directed edge src->dst (axis-aligned),
        face_len    = shared face length between src and dst (exact on max-grid),
        center_dist = ||c_dst - c_src||,
        w_diff      = face_len / center_dist   (canonical two-point diffusion weight).
    """
    device = centers.device
    N = int(centers.shape[0])
    if N == 0:
        ei = torch.empty((2, 0), dtype=torch.long, device=device)
        if return_edge_attr:
            ea = torch.empty((0, 5), dtype=torch.float32, device=device)
            return (ei, ea)
        return ei
    xmin, xmax, ymin, ymax = map(float, bbox)
    rr = int(refine_ratio)
    if rr < 2:
        raise ValueError(f'refine_ratio must be >=2, got {refine_ratio}')
    levels_i64 = levels.to(torch.int64)
    Lmax = int(levels_i64.max().item()) if levels_i64.numel() else 0
    if Lmax > int(max_L_guard):
        raise RuntimeError(f'Lmax={Lmax} exceeds guard={max_L_guard}. Raster-based face adjacency will be too large; raise guard or use a hierarchical neighbor finder.')
    dx0 = (xmax - xmin) / float(W)
    dy0 = (ymax - ymin) / float(H)
    Hmax = int(H) * rr ** Lmax
    Wmax = int(W) * rr ** Lmax
    dx_max = dx0 / float(rr) ** Lmax
    dy_max = dy0 / float(rr) ** Lmax
    occ = np.full((Hmax, Wmax), fill_value=-1, dtype=np.int32)
    c = centers.detach().cpu().to(torch.float32).numpy()
    l = levels_i64.detach().cpu().numpy()
    for cid in range(N):
        L = int(l[cid])
        scale = rr ** (Lmax - L)
        dxL = dx0 / float(rr) ** L
        dyL = dy0 / float(rr) ** L
        ixL = int(np.floor((c[cid, 0] - xmin) / dxL))
        iyL = int(np.floor((c[cid, 1] - ymin) / dyL))
        ixL = max(0, min(ixL, W * rr ** L - 1))
        iyL = max(0, min(iyL, H * rr ** L - 1))
        ix0 = ixL * scale
        iy0 = iyL * scale
        occ[iy0:iy0 + scale, ix0:ix0 + scale] = cid

    def _agg_directed(src_ids: np.ndarray, dst_ids: np.ndarray, seg_len: float):
        keys = src_ids.astype(np.int64) * N + dst_ids.astype(np.int64)
        uniq, counts = np.unique(keys, return_counts=True)
        src_u = (uniq // N).astype(np.int64)
        dst_u = (uniq % N).astype(np.int64)
        face_len = counts.astype(np.float32) * float(seg_len)
        return (src_u, dst_u, face_len)
    left = occ[:, :-1]
    right = occ[:, 1:]
    m = (left != right) & (left >= 0) & (right >= 0)
    src_lr = left[m]
    dst_lr = right[m]
    s1, d1, fl1 = _agg_directed(src_lr, dst_lr, dy_max)
    s2, d2, fl2 = _agg_directed(dst_lr, src_lr, dy_max)
    bot = occ[:-1, :]
    top = occ[1:, :]
    m = (bot != top) & (bot >= 0) & (top >= 0)
    src_bt = bot[m]
    dst_bt = top[m]
    s3, d3, fl3 = _agg_directed(src_bt, dst_bt, dx_max)
    s4, d4, fl4 = _agg_directed(dst_bt, src_bt, dx_max)
    src_all = np.concatenate([s1, s2, s3, s4], axis=0)
    dst_all = np.concatenate([d1, d2, d3, d4], axis=0)
    fl_all = np.concatenate([fl1, fl2, fl3, fl4], axis=0)
    keep = src_all != dst_all
    src_all = src_all[keep]
    dst_all = dst_all[keep]
    fl_all = fl_all[keep]
    if src_all.size == 0:
        ei = torch.empty((2, 0), dtype=torch.long, device=device)
        if return_edge_attr:
            ea = torch.empty((0, 5), dtype=torch.float32, device=device)
            return (ei, ea)
        return ei
    keys = src_all.astype(np.int64) * N + dst_all.astype(np.int64)
    uniq, inv = np.unique(keys, return_inverse=True)
    if uniq.size != keys.size:
        fl_sum = np.zeros((uniq.size,), dtype=np.float32)
        np.add.at(fl_sum, inv, fl_all)
        src_all = (uniq // N).astype(np.int64)
        dst_all = (uniq % N).astype(np.int64)
        fl_all = fl_sum
    edge_index = torch.as_tensor(np.stack([src_all, dst_all], axis=0), dtype=torch.int64, device=device)
    if not return_edge_attr:
        return edge_index
    c_t = centers.to(torch.float32)
    src_t = edge_index[0]
    dst_t = edge_index[1]
    delta = c_t[dst_t] - c_t[src_t]
    dist = torch.linalg.norm(delta, dim=1).clamp_min(1e-12)
    use_x = delta[:, 0].abs() >= delta[:, 1].abs()
    nx = torch.zeros((edge_index.shape[1],), dtype=torch.float32, device=device)
    ny = torch.zeros((edge_index.shape[1],), dtype=torch.float32, device=device)
    nx[use_x] = torch.sign(delta[use_x, 0])
    ny[~use_x] = torch.sign(delta[~use_x, 1])
    face_len = torch.as_tensor(fl_all, dtype=torch.float32, device=device)
    w_diff = face_len / dist
    edge_attr = torch.stack([nx, ny, face_len, dist, w_diff], dim=1)
    return (edge_index, edge_attr)

@torch.no_grad()
def _load_wedge_path_from_spec(mesh_spec_path: str, cfg: Dict[str, Any]):
    """
    Load the wedge polygon from wedge_mesh_spec.pt and return a
    matplotlib.path.Path object, rescaled into the same coordinate
    system as cfg["data"]["bbox"].

    We infer the native wedge extents directly from outer_polygon_ccw,
    rather than relying on spec["left"/"right"/"bottom"/"top"].
    """
    import numpy as np
    if mesh_spec_path is None:
        raise ValueError('mesh_spec_path is None in _load_wedge_path_from_spec')
    spec = torch.load(mesh_spec_path, map_location='cpu', weights_only=False)
    if 'outer_polygon_ccw' not in spec:
        raise KeyError(f"Expected key 'outer_polygon_ccw' in mesh spec {mesh_spec_path}, found keys={list(spec.keys())}")
    outer = spec['outer_polygon_ccw']
    if isinstance(outer, torch.Tensor):
        outer = outer.detach().cpu().numpy()
    else:
        outer = np.asarray(outer, dtype=np.float64)
    if outer.ndim != 2 or outer.shape[1] != 2:
        raise ValueError(f"'outer_polygon_ccw' must be (M,2), got shape {outer.shape}")
    left = float(outer[:, 0].min())
    right = float(outer[:, 0].max())
    bottom = float(outer[:, 1].min())
    top = float(outer[:, 1].max())
    xmin, xmax, ymin, ymax = cfg['data']['bbox']
    dx_src = right - left
    dy_src = top - bottom
    if dx_src == 0.0 or dy_src == 0.0:
        raise ValueError(f'Degenerate wedge extents inferred from outer_polygon_ccw: left={left}, right={right}, bottom={bottom}, top={top}')
    dx_tgt = xmax - xmin
    dy_tgt = ymax - ymin
    outer_scaled = outer.copy()
    outer_scaled[:, 0] = xmin + (outer[:, 0] - left) / dx_src * dx_tgt
    outer_scaled[:, 1] = ymin + (outer[:, 1] - bottom) / dy_src * dy_tgt
    return Path(outer_scaled)
