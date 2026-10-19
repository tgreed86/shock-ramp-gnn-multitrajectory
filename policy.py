# Focused numerical kernels retained from Physics-Aware; see PORTING.md.
from __future__ import annotations

from typing import Dict
import torch.nn.functional as F
import numpy as np
import time
import torch

def _get_refine_ratio(cfg: dict) -> int:
    pol = cfg.get('policy', {}) or {}
    mesh = cfg.get('mesh', {}) or {}
    rr = pol.get('refine_ratio', mesh.get('refine_ratio', 2))
    rr = int(rr)
    if rr < 2:
        raise ValueError(f'refine_ratio must be >=2, got {rr}')
    return rr

def _resolve_channel_indices(cfg: dict, Fdim: int) -> torch.Tensor:
    """
    Decide which feature channels to use when combining gradients.

    Looks for cfg["policy"]["grad_channels"], which should be either:
      - None / missing  -> use all channels [0..Fdim-1]
      - a single int    -> use [that] (clipped to valid range)
      - a list of ints  -> use those, filtered to [0..Fdim-1]

    Returns a 1D LongTensor of indices.
    """
    pol = cfg.get('policy', {})
    ch = pol.get('grad_channels', None)
    if ch is None:
        return torch.arange(Fdim, dtype=torch.long)
    if isinstance(ch, (int, np.integer)):
        ch_list = [int(ch)]
    else:
        ch_list = [int(c) for c in ch]
    idx = torch.as_tensor(ch_list, dtype=torch.long)
    mask = (idx >= 0) & (idx < Fdim)
    idx = idx[mask]
    if idx.numel() == 0:
        return torch.arange(Fdim, dtype=torch.long)
    return idx

def _parents_from_level_ij(levels: torch.Tensor, ij: torch.Tensor, H: int, W: int, refine_ratio: int=2) -> torch.Tensor:
    """
    Map each level-l cell with indices (i,j) at its own resolution
    (H*refine_ratio^l, W*refine_ratio^l)
    to its coarse parent index in a flattened (H*W) coarse grid.

    Works for arbitrary l >= 0. Assumes ij are integer cell indices at the cell's level.
    """
    if levels.dim() != 1:
        levels = levels.view(-1)
    i = ij[:, 0].long()
    j = ij[:, 1].long()
    rr = int(refine_ratio)
    if rr < 2:
        raise ValueError(f'refine_ratio must be >=2, got {refine_ratio}')
    scale = torch.pow(torch.tensor(rr, device=levels.device, dtype=torch.long), levels)
    row = torch.div(i, scale, rounding_mode='floor').clamp_(0, H - 1)
    col = torch.div(j, scale, rounding_mode='floor').clamp_(0, W - 1)
    parents = row * W + col
    return parents.long()

def _to_tensor_2d(x, H, W, device=None, dtype=torch.float32):
    """
    Accepts: torch.Tensor or np.ndarray (flat or 2D) and returns a 2D torch tensor (H,W).
    """
    if isinstance(x, torch.Tensor):
        t = x.detach()
        if t.numel() == H * W and t.dim() != 2:
            t = t.view(H, W)
        elif t.dim() == 2:
            if t.shape != (H, W):
                t = t.reshape(H, W)
        else:
            t = t.reshape(H, W)
        return t.to(device=device, dtype=dtype, copy=False)
    arr = np.asarray(x)
    if arr.size == H * W and arr.ndim != 2:
        arr = arr.reshape(H, W)
    elif arr.ndim == 2 and arr.shape != (H, W):
        arr = arr.reshape(H, W)
    else:
        arr = arr.reshape(H, W)
    return torch.from_numpy(arr).to(device=device, dtype=dtype)

def _combine_level(per_lvl: torch.Tensor, use_cols: torch.Tensor, combine: str='l2', w: torch.Tensor | None=None) -> torch.Tensor:
    """
        per_lvl: (h, w, F) tensor of per-feature |∇f_k| (CPU or device); may contain NaNs.
        returns: (h, w) combined map (same device as input).
        """
    dev_local = per_lvl.device
    sel = per_lvl.index_select(-1, use_cols.to(dev_local))
    sel = torch.nan_to_num(sel, nan=0.0)
    if combine == 'max':
        return sel.max(dim=2).values
    elif combine in ('sum', 'weighted_sum'):
        ww = w.to(dev_local) if w is not None else torch.ones(sel.shape[-1], device=dev_local, dtype=sel.dtype)
        return (sel * ww.view(1, 1, -1)).sum(dim=2)
    else:
        ww = w.to(dev_local) if w is not None else torch.ones(sel.shape[-1], device=dev_local, dtype=sel.dtype)
        return torch.sqrt(((sel * ww.view(1, 1, -1)) ** 2).sum(dim=2))

@torch.no_grad()
def compute_hierarchical_gradients_from_gt_raster(batch: dict, cfg: dict, H: int, W: int, dx0: float, dy0: float, *, feature_idx=None, feature_names=None):
    """
    Fast vectorized gradient backend:
      1) Compose selected channels to finest grid Hf×Wf (Lmax).
      2) ∇ via conv2d (central-diff) on GPU if available.
      3) L2 combine per-channel to (Hf,Wf); then max-pool down to each level.

    Returns:
      {
        "grad_by_level":     {L: (h,w) },
        "grad_feat_by_level":{L: (h,w,C)}
      }
    """
    from raster import amr_composite_to_finest_grid
    dev = next((v.device for v in batch.values() if torch.is_tensor(v)), torch.device('cpu'))
    pol = cfg.get('policy', {})
    Lmax = int(pol.get('max_level', 2))
    rr = _get_refine_ratio(cfg)
    xmin, xmax, ymin, ymax = cfg['data']['bbox']
    '\n    F_total = batch["dyn_feat_t"].shape[-1]\n    if feature_idx is not None:\n        use_cols = torch.as_tensor(list(feature_idx), dtype=torch.long)\n    elif feature_names is not None:\n        chmap = cfg.get("channels", {})\n        use_cols = torch.as_tensor([chmap[nm] for nm in feature_names], dtype=torch.long)\n    else:\n        use_cols = torch.arange(F_total, dtype=torch.long)\n    use_cols = use_cols[(use_cols >= 0) & (use_cols < F_total)]\n    C = int(use_cols.numel())\n    if C == 0: raise ValueError("No valid feature channels.")\n    '
    F_total = batch['dyn_feat_t'].shape[-1]
    use_cols = _resolve_channel_indices(cfg, F_total).detach().cpu()
    C = int(use_cols.numel())
    if C == 0:
        raise ValueError('No valid feature channels.')
    imgs = []
    valid2d = None
    for c in use_cols.tolist():
        img_flat, valid_flat, Hc, Wc = amr_composite_to_finest_grid(batch['centers_t'].detach().cpu(), None if 'level_t' not in batch else batch['level_t'].detach().cpu(), batch['center_feat_t'][:, [c]].detach().cpu(), H, W, (xmin, xmax, ymin, ymax), refine_ratio=rr)
        img2d = _to_tensor_2d(img_flat, Hc, Wc, device=dev, dtype=torch.float32)
        imgs.append(img2d)
        if valid2d is None:
            valid2d = _to_tensor_2d(valid_flat, Hc, Wc, device=dev, dtype=torch.bool)
    X = torch.stack(imgs, dim=0).to(dev, dtype=torch.float32)
    dx_f = dx0 / float(rr) ** Lmax
    dy_f = dy0 / float(rr) ** Lmax
    kx = torch.tensor([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], dtype=torch.float32, device=dev).view(1, 1, 3, 3) / dx_f
    ky = torch.tensor([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]], dtype=torch.float32, device=dev).view(1, 1, 3, 3) / dy_f
    try:
        Xp = F.pad(X, (1, 1, 1, 1), mode='replicate').unsqueeze(0)
    except NotImplementedError:
        Xp = F.pad(X, (1, 1, 1, 1), mode='reflect').unsqueeze(0)
    gx = F.conv2d(Xp, kx.expand(C, 1, 3, 3), groups=C)
    gy = F.conv2d(Xp, ky.expand(C, 1, 3, 3), groups=C)
    stencil_ok = valid2d & torch.roll(valid2d, 1, dims=0) & torch.roll(valid2d, -1, dims=0) & torch.roll(valid2d, 1, dims=1) & torch.roll(valid2d, -1, dims=1)
    gx = gx * stencil_ok[None, None]
    gy = gy * stencil_ok[None, None]
    gfeat_fine = torch.sqrt(torch.clamp(gx * gx + gy * gy, min=0.0))[0]
    grad_feat_by_level, grad_by_level = ({}, {})
    combine = str(cfg.get('policy', {}).get('combine', 'l2')).lower()
    raw_w = cfg.get('policy', {}).get('refine_weights', None)
    if raw_w is not None:
        w = torch.as_tensor(raw_w, device=dev, dtype=torch.float32)
        if w.numel() != C:
            w = torch.cat([w, w.new_ones(C - w.numel())]) if w.numel() < C else w[:C]
    else:
        w = None

    def _combine(stack):
        if combine == 'max':
            return stack.max(dim=0).values
        if combine in ('sum', 'weighted_sum'):
            return (stack if w is None else w[:, None, None] * stack).sum(dim=0)
        if w is None:
            return torch.sqrt((stack ** 2).sum(dim=0))
        return torch.sqrt(((w[:, None, None] * stack) ** 2).sum(dim=0))
    grad_feat_by_level[Lmax] = gfeat_fine.permute(1, 2, 0).contiguous()
    grad_by_level[Lmax] = _combine(gfeat_fine)
    for L in range(Lmax - 1, -1, -1):
        scale = rr ** (Lmax - L)
        pooled = F.max_pool2d(gfeat_fine, kernel_size=scale, stride=scale)
        grad_feat_by_level[L] = pooled.permute(1, 2, 0).contiguous()
        grad_by_level[L] = _combine(pooled)
    return {'grad_by_level': grad_by_level, 'grad_feat_by_level': grad_feat_by_level}

@torch.no_grad()
def predict_masks_hierarchical_from_gt_gradients(batch: dict, cfg: dict, H: int, W: int, dx_unused: float, dy_unused: float, *, device=None, debug_out: dict | None=None, timing_out: dict | None=None) -> dict[int, torch.Tensor]:
    t_all_t0 = time.perf_counter()
    timing: Dict[str, float] = {'predict_grad_raster_s': 0.0, 'predict_combine_levels_s': 0.0, 'predict_pool_up_s': 0.0, 'predict_prev_masks_s': 0.0, 'predict_threshold_hysteresis_s': 0.0, 'predict_dilation_s': 0.0, 'predict_normalize_masks_s': 0.0, 'predict_debug_out_s': 0.0, 'predict_legacy_policy_s': 0.0}
    dev = device or next((v.device for v in batch.values() if torch.is_tensor(v)), torch.device('cpu'))
    pol = cfg.get('policy', {})
    rr = _get_refine_ratio(cfg)
    tau_by_level = pol.get('tau_by_level', None)
    tau_low_def = float(pol.get('tau_low', 0.02))
    tau_high_def = float(pol.get('tau_high', 0.03))
    mode = str(pol.get('hysteresis_mode', 'absolute')).lower()
    pct_low_default = float(pol.get('percentile_low', 75.0))
    pct_high_default = float(pol.get('percentile_high', 90.0))
    pct_by_level = pol.get('percentiles_by_level', {})
    pct_mode = 'auto'
    if isinstance(pct_by_level, dict):
        pct_mode = str(pct_by_level.get('selection', 'auto')).strip().lower()
    if pct_mode not in ('auto', 'global', 'per_level', 'per-level', 'level', 'levels'):
        raise ValueError('policy.percentiles_by_level.selection must be one of: auto, global, per_level')
    use_global_pct = pct_mode == 'global'
    use_level_pct = pct_mode in ('per_level', 'per-level', 'level', 'levels')
    dbg_thr = bool(pol.get('debug_thresholds', False))

    def _pct_level_entry(L: int):
        if not isinstance(pct_by_level, dict):
            return None
        if L in pct_by_level:
            return pct_by_level[L]
        s = str(L)
        if s in pct_by_level:
            return pct_by_level[s]
        return None

    def _taus(L: int) -> tuple[float, float]:
        if isinstance(tau_by_level, dict) and L in tau_by_level:
            tl = tau_by_level[L]
            if isinstance(tl, dict):
                return (float(tl.get('low', tau_low_def)), float(tl.get('high', tau_high_def)))
            else:
                return (tau_low_def, float(tl))
        return (tau_low_def, tau_high_def)
    xmin, xmax, ymin, ymax = cfg['data']['bbox']
    dx0 = (xmax - xmin) / float(W)
    dy0 = (ymax - ymin) / float(H)
    t_grad_t0 = time.perf_counter()
    grads = compute_hierarchical_gradients_from_gt_raster(batch, cfg, H, W, dx0, dy0, feature_idx=None, feature_names=None)
    timing['predict_grad_raster_s'] = float(time.perf_counter() - t_grad_t0)
    t_combine_t0 = time.perf_counter()
    combine = str(cfg.get('policy', {}).get('combine', 'l2')).lower()
    _anyL = next(iter(grads['grad_feat_by_level'].keys()))
    Fdim = grads['grad_feat_by_level'][_anyL].shape[-1]
    use_cols = _resolve_channel_indices(cfg, Fdim)
    raw_w = cfg.get('policy', {}).get('refine_weights', None)
    if raw_w is not None:
        w = torch.as_tensor(raw_w, dtype=torch.float32)
        if w.numel() != use_cols.numel():
            if w.numel() < use_cols.numel():
                w = torch.cat([w, w.new_ones(use_cols.numel() - w.numel())])
            else:
                w = w[:use_cols.numel()]
    else:
        w = None
    G = {L: _combine_level(torch.nan_to_num(grads['grad_feat_by_level'][L], nan=0.0), use_cols=use_cols, combine=combine, w=w) for L in grads['grad_feat_by_level']}
    timing['predict_combine_levels_s'] = float(time.perf_counter() - t_combine_t0)
    t_pool_t0 = time.perf_counter()
    pooled_up = {L: G[L].clone() for L in G}
    maxL = max(G.keys())
    for L in range(maxL - 1, -1, -1):
        for Lc in range(L + 1, maxL + 1):
            scale = rr ** (Lc - L)
            pooled = torch.nn.functional.max_pool2d(G[Lc][None, None], kernel_size=scale, stride=scale)[0, 0]
            pooled_up[L] = torch.maximum(pooled_up[L], pooled)
    timing['predict_pool_up_s'] = float(time.perf_counter() - t_pool_t0)
    t_prev_masks_t0 = time.perf_counter()
    prev_refine_by_level: Dict[int, torch.Tensor] = {}
    level_t_raw = batch.get('prev_level_t', batch.get('level_t', None))
    ij_t_raw = batch.get('prev_ij_t', batch.get('ij_t', None))
    if level_t_raw is not None and ij_t_raw is not None:
        try:
            lv_prev = torch.as_tensor(level_t_raw, device=dev).view(-1).long()
            ij_prev = torch.as_tensor(ij_t_raw, device=dev)
            if ij_prev.ndim >= 3 and ij_prev.shape[0] == 1:
                ij_prev = ij_prev.squeeze(0)
            if ij_prev.ndim == 2 and ij_prev.shape[0] == 2 and (ij_prev.shape[1] != 2):
                ij_prev = ij_prev.transpose(0, 1)
            if ij_prev.ndim > 2 and ij_prev.shape[-1] == 2:
                ij_prev = ij_prev.reshape(-1, 2)
            if ij_prev.ndim == 2 and ij_prev.shape[1] == 2 and (ij_prev.shape[0] == lv_prev.numel()):
                ij_prev = ij_prev.to(torch.long)
                for L in range(1, maxL + 1):
                    h_p = int(H * rr ** (L - 1))
                    w_p = int(W * rr ** (L - 1))
                    prev_mask_L = torch.zeros((h_p, w_p), dtype=torch.bool, device=dev)
                    sel = lv_prev >= int(L)
                    if bool(sel.any()):
                        rel_lv = (lv_prev[sel] - int(L - 1)).to(torch.long)
                        parent_flat = _parents_from_level_ij(rel_lv, ij_prev[sel], h_p, w_p, refine_ratio=rr).view(-1).long()
                        if parent_flat.numel() > 0:
                            parent_flat = parent_flat.clamp_(0, h_p * w_p - 1)
                            prev_mask_L.view(-1)[parent_flat] = True
                    prev_refine_by_level[L] = prev_mask_L
            elif dbg_thr:
                print('[THR] warning: level_t/ij_t shape mismatch; falling back to L1 mask_t hysteresis only.')
        except Exception as e:
            if dbg_thr:
                print(f'[THR] warning: failed to build per-level previous masks ({e}); using fallback.')
    timing['predict_prev_masks_s'] = float(time.perf_counter() - t_prev_masks_t0)
    t_thr_t0 = time.perf_counter()
    dilation_s = 0.0
    masks_by_level = {}
    for L in range(1, maxL + 1):
        parent = L - 1
        Gparent = pooled_up[parent].to(dev)
        if mode == 'percentile':
            if use_global_pct:
                p_low, p_high = (pct_low_default, pct_high_default)
            else:
                pL = _pct_level_entry(L)
                if use_level_pct and pL is None:
                    p_low, p_high = (pct_low_default, pct_high_default)
                elif isinstance(pL, dict):
                    p_low = float(pL.get('low', pct_low_default))
                    p_high = float(pL.get('high', pct_high_default))
                else:
                    p_low, p_high = (pct_low_default, pct_high_default)
            p_low = max(0.0, min(100.0, p_low))
            p_high = max(0.0, min(100.0, p_high))
            if p_low > p_high:
                p_low, p_high = (p_high, p_low)
            finite = torch.isfinite(Gparent)
            if finite.any():
                thr_low = torch.quantile(Gparent[finite], p_low / 100.0)
                thr_high = torch.quantile(Gparent[finite], p_high / 100.0)
            else:
                thr_low = torch.tensor(float('inf'), device=Gparent.device)
                thr_high = torch.tensor(float('inf'), device=Gparent.device)
        else:
            tau_low, tau_high = _taus(L)
            thr_low = torch.as_tensor(tau_low, device=Gparent.device, dtype=Gparent.dtype)
            thr_high = torch.as_tensor(tau_high, device=Gparent.device, dtype=Gparent.dtype)
        if dbg_thr:
            finite = torch.isfinite(Gparent)
            gmin = Gparent[finite].min().item() if finite.any() else float('nan')
            gmax = Gparent[finite].max().item() if finite.any() else float('nan')
            print(f'[THR] L{L} mode={mode} thr_low={float(thr_low):.3e} thr_high={float(thr_high):.3e} min={gmin:.3e} max={gmax:.3e}')
        h_p, w_p = (int(Gparent.shape[0]), int(Gparent.shape[1]))
        prev_mask = prev_refine_by_level.get(L, None)
        if prev_mask is None and L == 1:
            prev_flat_raw = batch.get('prev_mask_t', batch.get('mask_t', None))
            if prev_flat_raw is None:
                prev_flat = None
            else:
                prev_flat = torch.as_tensor(prev_flat_raw, device=dev)
        if prev_mask is None and L == 1 and (prev_flat is not None):
            H0 = int(cfg['data'].get('H', H))
            W0 = int(cfg['data'].get('W', W))
            if H0 * W0 == prev_flat.numel():
                prev2d = prev_flat.view(H0, W0).float()
            else:
                side = int(round(prev_flat.numel() ** 0.5))
                if side * side == prev_flat.numel():
                    prev2d = prev_flat.view(side, side).float()
                else:
                    prev2d = torch.zeros((h_p, w_p), device=dev, dtype=torch.float32)
            prev_mask = prev2d.bool()
        if prev_mask is not None:
            if prev_mask.shape != (h_p, w_p):
                prev_mask = torch.nn.functional.interpolate(prev_mask.float()[None, None], size=(h_p, w_p), mode='nearest')[0, 0].bool()
            keep = prev_mask & (Gparent > thr_low)
            newr = ~prev_mask & (Gparent > thr_high)
            M = keep | newr
        else:
            M = Gparent > thr_high
        if L > 1:
            allow = torch.nn.functional.interpolate(masks_by_level[L - 1].float()[None, None], scale_factor=float(rr), mode='nearest')[0, 0].bool()
            h, w = Gparent.shape
            M = M & allow[:h, :w]
        pol_cfg = cfg.get('policy', {}) or {}
        dxL_here = (xmax - xmin) / float(W * rr ** (L - 1))
        n_cells = None
        raw_cells = pol_cfg.get(f'dilate_cells_L{L}', None)
        if raw_cells is None and L == maxL:
            raw_cells = pol_cfg.get('dilate_cells_lmax', None)
        if raw_cells is not None:
            try:
                n_cells = float(raw_cells)
            except Exception:
                n_cells = None
        if n_cells is not None:
            r_phys = float(max(0.0, n_cells) * dxL_here)
        else:
            r_phys = float(pol_cfg.get(f'dilate_phys_L{L}', 0.0))
        if r_phys > 0:
            t_dilate_t0 = time.perf_counter()
            r_cells = max(1, int(round(r_phys / dxL_here)))
            k = 2 * r_cells + 1
            M = torch.nn.functional.max_pool2d(M.float()[None, None], kernel_size=k, stride=1, padding=r_cells)[0, 0].bool()
            dilation_s += float(time.perf_counter() - t_dilate_t0)
        masks_by_level[L] = M
    thr_total = float(time.perf_counter() - t_thr_t0)
    timing['predict_dilation_s'] = float(dilation_s)
    timing['predict_threshold_hysteresis_s'] = max(0.0, thr_total - float(dilation_s))
    t_norm_t0 = time.perf_counter()
    norm_masks_by_level = {}
    for L, M in masks_by_level.items():
        M = M.to(torch.bool)
        expected_h = H * rr ** (L - 1)
        expected_w = W * rr ** (L - 1)
        h, w = (int(M.shape[0]), int(M.shape[1]))
        if (h, w) == (expected_h, expected_w):
            norm_masks_by_level[L] = M
            continue
        if (h, w) == (expected_h * rr, expected_w * rr):
            Mp = torch.nn.functional.max_pool2d(M.float()[None, None], kernel_size=rr, stride=rr)[0, 0].to(torch.bool)
            norm_masks_by_level[L] = Mp
            continue
        Mp = torch.nn.functional.interpolate(M.float()[None, None], size=(expected_h, expected_w), mode='nearest')[0, 0].to(torch.bool)
        norm_masks_by_level[L] = Mp
    masks_by_level = norm_masks_by_level
    timing['predict_normalize_masks_s'] = float(time.perf_counter() - t_norm_t0)
    t_dbg_out_t0 = time.perf_counter()
    if debug_out is not None:
        debug_out['G'] = {L: v.detach().clone() for L, v in G.items()}
        debug_out['pooled_up'] = {L: v.detach().clone() for L, v in pooled_up.items()}
    timing['predict_debug_out_s'] = float(time.perf_counter() - t_dbg_out_t0)
    timing['predict_legacy_policy_s'] = float(time.perf_counter() - t_all_t0)
    if isinstance(timing_out, dict):
        timing_out.clear()
        timing_out.update(timing)
    return masks_by_level

def coarse_aggregate_from_dynamic(X: torch.Tensor, parents: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Aggregate node-wise features X (N,D) to the coarse parent grid (H*W,D)
    using 'parents' indices in [0..H*W-1].

    X:        [N, D]
    parents:  [N]  (long)
    returns:  [H*W, D]
    """
    assert X.dim() == 2, f'X must be [N,D], got {tuple(X.shape)}'
    N, D = X.shape
    parents = parents.view(-1).long()
    assert parents.numel() == N, f'parents length ({parents.numel()}) must equal N ({N})'
    out = X.new_zeros((H * W, D))
    cnt = X.new_zeros((H * W, 1))
    out.index_add_(0, parents, X)
    ones = X.new_ones((N, 1))
    cnt.index_add_(0, parents, ones)
    cnt.clamp_min_(1.0)
    out = out / cnt
    return out
