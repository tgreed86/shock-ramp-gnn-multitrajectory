# Retained IDW and dyadic mesh kernels; feature gathers preserve gradients.
from __future__ import annotations
import torch.nn.functional as F
import torch

@torch.no_grad()
def dynamic_cells_from_parent_masks(masks_by_level: dict[int, torch.Tensor], H: int, W: int, xmin: float, xmax: float, ymin: float, ymax: float, build_edges: bool=False, refine_ratio: int=2):
    """
    Convert hierarchical *parent* masks into the final leaf mesh.

    Input:
      masks_by_level[L]: bool tensor with shape
        (H*refine_ratio^(L-1), W*refine_ratio^(L-1))
        - L=1: parents at level-0 to be refined -> produce level-1 children
        - L=2: parents at level-1 to be refined -> produce level-2 children
        - ... (up to Lmax)

    Output:
      centers: (N,2) in bbox units
      levels:  (N,)  int64, level of each leaf cell (0..Lmax)
      parents: (N,)  int64, coarse (level-0) parent index in [0..H*W-1]
      edge_index: (2,E) (optional) 4-neighbor within-level connectivity
    """
    rr = int(refine_ratio)
    if rr < 2:
        raise ValueError(f'refine_ratio must be >=2, got {refine_ratio}')
    device = next(iter(masks_by_level.values())).device if masks_by_level else torch.device('cpu')
    Lmax = max(masks_by_level.keys(), default=0)
    refine = [None] * (Lmax + 1)
    for L, M in masks_by_level.items():
        Mh, Mw = (H * rr ** (L - 1), W * rr ** (L - 1))
        assert M.shape == (Mh, Mw), f'mask[{L}] has shape {tuple(M.shape)}; expected {(Mh, Mw)}'
        refine[L] = M.to(torch.bool)
    leaves_by_level: list[torch.Tensor] = []
    if Lmax == 0:
        leaves_by_level.append(torch.ones((H, W), dtype=torch.bool, device=device))
    else:
        leaves_by_level.append(~refine[1])
        for l in range(1, Lmax):
            parent_refined = F.interpolate(refine[l].float().unsqueeze(0).unsqueeze(0), scale_factor=float(rr), mode='nearest').squeeze(0).squeeze(0).to(torch.bool)
            leaves_l = parent_refined & ~refine[l + 1]
            leaves_by_level.append(leaves_l)
        leaf_Lmax = F.interpolate(refine[Lmax].float().unsqueeze(0).unsqueeze(0), scale_factor=float(rr), mode='nearest').squeeze(0).squeeze(0).to(torch.bool)
        leaves_by_level.append(leaf_Lmax)
    xs = xmax - xmin
    ys = ymax - ymin
    centers_list, levels_list, parents_list = ([], [], [])
    for l, leaf in enumerate(leaves_by_level):
        HH = H * rr ** l
        WW = W * rr ** l
        jj, ii = torch.nonzero(leaf, as_tuple=True)
        if jj.numel() == 0:
            continue
        cx = xmin + (ii.to(torch.float32) + 0.5) * (xs / float(WW))
        cy = ymin + (jj.to(torch.float32) + 0.5) * (ys / float(HH))
        centers_list.append(torch.stack([cx, cy], dim=-1))
        levels_list.append(torch.full((jj.numel(),), l, dtype=torch.int64, device=leaf.device))
        i0 = (ii // rr ** l).to(torch.int64)
        j0 = (jj // rr ** l).to(torch.int64)
        parents_list.append(j0 * W + i0)
    if centers_list:
        centers = torch.cat(centers_list, dim=0)
        levels = torch.cat(levels_list, dim=0)
        parents = torch.cat(parents_list, dim=0)
    else:
        centers = torch.empty(0, 2, dtype=torch.float32, device=device)
        levels = torch.empty(0, dtype=torch.int64, device=device)
        parents = torch.empty(0, dtype=torch.int64, device=device)
    if not build_edges or centers.numel() == 0:
        ei = torch.empty(2, 0, dtype=torch.int64, device=device)
        return (centers, levels, parents, ei)
    ei_src, ei_dst = ([], [])
    start = 0
    for l, leaf in enumerate(leaves_by_level):
        HH = H * rr ** l
        WW = W * rr ** l
        idx_map = -torch.ones((HH, WW), dtype=torch.int64, device=device)
        jj, ii = torch.nonzero(leaf, as_tuple=True)
        if jj.numel() == 0:
            continue
        count = jj.numel()
        idx_map[jj, ii] = torch.arange(start, start + count, device=device, dtype=torch.int64)
        start += count
        jj_r, ii_r = (jj, ii + 1)
        m_r = (ii_r < WW) & (idx_map[jj, ii] >= 0) & (idx_map[jj_r, ii_r] >= 0)
        if m_r.any():
            a = idx_map[jj[m_r], ii[m_r]]
            b = idx_map[jj_r[m_r], ii_r[m_r]]
            ei_src.append(a)
            ei_dst.append(b)
        jj_u, ii_u = (jj + 1, ii)
        m_u = (jj_u < HH) & (idx_map[jj, ii] >= 0) & (idx_map[jj_u, ii_u] >= 0)
        if m_u.any():
            a = idx_map[jj[m_u], ii[m_u]]
            b = idx_map[jj_u[m_u], ii_u[m_u]]
            ei_src.append(a)
            ei_dst.append(b)
    if ei_src:
        src = torch.cat(ei_src)
        dst = torch.cat(ei_dst)
        ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    else:
        ei = torch.empty(2, 0, dtype=torch.int64, device=device)
    return (centers, levels, parents, ei)

@torch.no_grad()
def build_idw_map(dst_xy, src_xy, k=8, chunk=8192, eps=1e-08, *, backend='exact'):
    if backend == 'exact':
        return _build_idw_map_exact(dst_xy, src_xy, k=k, chunk=chunk, eps=eps)
    if backend == 'kdtree':
        return _build_idw_map_kdtree(dst_xy, src_xy, k=k, chunk=chunk, eps=eps)
    raise ValueError(f'Unsupported IDW backend: {backend}')

def _idw_weights_from_dist(d_sel: torch.Tensor, eps: float) -> torch.Tensor:
    d_sel = torch.nan_to_num(d_sel, nan=float('inf'), posinf=float('inf'), neginf=float('inf'))
    w_sel = torch.empty_like(d_sel)
    exact = d_sel[:, 0] <= eps
    if exact.any():
        w_sel[exact] = 0.0
        w_sel[exact, 0] = 1.0
    if (~exact).any():
        nz = ~exact
        inv = 1.0 / torch.clamp(d_sel[nz], min=eps)
        inv_sum = inv.sum(dim=1, keepdim=True)
        w_norm = inv / (inv_sum + eps)
        w_sel[nz] = torch.nan_to_num(w_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return w_sel

@torch.no_grad()
def _build_idw_map_kdtree(dst_xy: torch.Tensor, src_xy: torch.Tensor, k: int=8, chunk: int=8192, eps: float=1e-08) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact CPU neighbor search with the same 1/d weights as the torch backend.

    Results return to the query device. Equal-distance neighbor ties may select
    different source indices than torch.topk, as with other search backends.
    """
    from scipy.spatial import cKDTree
    n, m = (int(dst_xy.shape[0]), int(src_xy.shape[0]))
    k = min(int(k), m)
    if n == 0 or k <= 0:
        return (torch.empty((n, 0), dtype=torch.long, device=dst_xy.device), torch.empty((n, 0), dtype=torch.float32, device=dst_xy.device))
    if int(chunk) < 1:
        raise ValueError('IDW chunk must be positive.')
    src = src_xy.detach().to(device='cpu', dtype=torch.float64).numpy()
    dst = dst_xy.detach().to(device='cpu', dtype=torch.float64).numpy()
    tree = cKDTree(src)
    indices, weights = ([], [])
    for start in range(0, n, int(chunk)):
        distance, index = tree.query(dst[start:start + int(chunk)], k=k, eps=0.0)
        index_t = torch.as_tensor(index.copy(), dtype=torch.long).reshape(-1, k)
        distance_t = torch.as_tensor(distance.copy(), dtype=torch.float32).reshape(-1, k)
        indices.append(index_t)
        weights.append(_idw_weights_from_dist(distance_t, eps=eps))
    return (torch.cat(indices).to(dst_xy.device), torch.cat(weights).to(dst_xy.device))

@torch.no_grad()
def _build_idw_map_exact(dst_xy: torch.Tensor, src_xy: torch.Tensor, k: int=8, chunk: int=8192, eps: float=1e-08) -> tuple[torch.Tensor, torch.Tensor]:
    N = int(dst_xy.shape[0])
    M = int(src_xy.shape[0])
    k = min(k, M)
    if N == 0 or k <= 0:
        dev = dst_xy.device
        return (torch.empty((N, 0), dtype=torch.long, device=dev), torch.empty((N, 0), dtype=torch.float32, device=dev))
    idx_all, w_all = ([], [])
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        q = dst_xy[start:end]
        d = torch.cdist(q, src_xy, p=2)
        d = torch.nan_to_num(d, nan=float('inf'), posinf=float('inf'), neginf=float('inf'))
        tk = torch.topk(d, k=k, dim=1, largest=False)
        d_sel = tk.values.contiguous()
        i_sel = tk.indices.contiguous()
        w_sel = _idw_weights_from_dist(d_sel, eps=eps)
        idx_all.append(i_sel)
        w_all.append(w_sel)
    return (torch.cat(idx_all, dim=0), torch.cat(w_all, dim=0))

def apply_idw_map(idx: torch.Tensor, w: torch.Tensor, src_feat: torch.Tensor) -> torch.Tensor:
    N, K = idx.shape
    F = src_feat.shape[1]
    gathered = src_feat.index_select(0, idx.reshape(-1)).view(N, K, F)
    vals = (gathered * w.unsqueeze(-1)).sum(dim=1)
    return torch.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
