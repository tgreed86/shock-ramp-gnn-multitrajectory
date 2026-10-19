# Retained runtime and state kernels; see PORTING.md.
from __future__ import annotations
from typing import Any
from typing import Dict
import torch.nn.functional as F
from typing import List
from typing import Optional
from collections import OrderedDict
from typing import Tuple
from mesh_geometry import _get_wedge_clip_level_lookup
from mesh_geometry import _lookup_level_masks_on_device
from geometry import apply_idw_map
from mesh_geometry import build_amr_face_adjacency_edges
from geometry import build_idw_map
from policy import coarse_aggregate_from_dynamic
import physics as dec
import h5py
import hashlib
import numpy as np
import os
import re
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

def _apply_momentum_sigma_mode(sigma: torch.Tensor, *, mode: str, momentum_x_idx: int, momentum_y_idx: int) -> torch.Tensor:
    if mode == 'independent':
        return sigma
    if mode != 'shared_rms':
        raise ValueError(f'Unsupported momentum sigma mode: {mode!r}')
    max_idx = max(int(momentum_x_idx), int(momentum_y_idx))
    if sigma.numel() <= max_idx:
        raise ValueError(f'Cannot apply shared momentum sigma: feature dimension is too small for momentum indices mx={momentum_x_idx}, my={momentum_y_idx}, numel={sigma.numel()}.')
    out = sigma.clone()
    sig_x = out[int(momentum_x_idx)]
    sig_y = out[int(momentum_y_idx)]
    sig_shared = torch.sqrt(0.5 * (sig_x * sig_x + sig_y * sig_y)).clamp_min(1e-12)
    out[int(momentum_x_idx)] = sig_shared
    out[int(momentum_y_idx)] = sig_shared
    return out

@torch.no_grad()
def _coarse_grad_mag(field_coarse: torch.Tensor, H: int, W: int, dx: float, dy: float, p: float=1.0) -> torch.Tensor:
    """Centered‑difference gradient magnitude on the coarse H×W grid.
    field_coarse: (H*W,) → returns (H*W,) magnitudes (optionally raised to power p).
    """
    f = field_coarse.view(H, W)
    fx = torch.zeros_like(f)
    fy = torch.zeros_like(f)
    if W > 1:
        fx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * dx)
        fx[:, 0] = (f[:, 1] - f[:, 0]) / dx
        fx[:, -1] = (f[:, -1] - f[:, -2]) / dx
    if H > 1:
        fy[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2.0 * dy)
        fy[0, :] = (f[1, :] - f[0, :]) / dy
        fy[-1, :] = (f[-1, :] - f[-2, :]) / dy
    g = torch.sqrt(fx * fx + fy * fy).reshape(-1)
    if p != 1.0:
        g = g ** p
    return g

def _resolve_channel_indices(cfg: Dict[str, Any]) -> List[int]:
    chmap = cfg.get('channels', {})
    names = cfg.get('policy', {}).get('refine_channels') or cfg.get('features', {}).get('supervised')
    idxs: List[int] = []
    if names:
        for nm in names:
            if isinstance(nm, int):
                idxs.append(int(nm))
            elif nm in chmap:
                idxs.append(int(chmap[nm]))
    if idxs:
        return idxs
    if cfg.get('features', {}).get('use_columns'):
        return list(cfg['features']['use_columns'])
    if cfg.get('data', {}).get('feature_idx'):
        return list(cfg['data']['feature_idx'])
    return [0, 1, 2]

def _dt_hat_feature_column(dt_hat: torch.Tensor | float, *, n_nodes: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return dt_hat as a per-node column of shape (N, 1)."""
    if torch.is_tensor(dt_hat):
        dt = dt_hat.to(device=device, dtype=dtype)
    else:
        dt = torch.tensor(float(dt_hat), device=device, dtype=dtype)
    if dt.numel() == 1:
        return dt.view(1, 1).expand(n_nodes, 1)
    if dt.dim() == 1 and dt.numel() == n_nodes:
        return dt.view(n_nodes, 1)
    if dt.dim() == 2 and tuple(dt.shape) == (n_nodes, 1):
        return dt
    raise RuntimeError(f'Could not broadcast dt_hat to node feature column. Expected scalar or (N,) / (N,1), got shape={tuple(dt.shape)} with N={n_nodes}.')

def _to_scalar_float(v: Any) -> float | None:
    if v is None:
        return None
    if torch.is_tensor(v):
        if v.numel() != 1:
            return None
        return float(v.detach().cpu().item())
    try:
        return float(v)
    except Exception:
        return None

def _extract_angle_from_mapping(d: Dict[str, Any]) -> float | None:
    if not isinstance(d, dict):
        return None
    angle_keys = ('ramp_angle_deg', 'ramp_angle', 'angle_deg', 'angle', 'theta_deg', 'theta')
    for k in angle_keys:
        if k in d:
            val = _to_scalar_float(d.get(k))
            if val is not None:
                return float(val)
    for mk in ('meta', 'metadata', 'attrs', 'info'):
        mv = d.get(mk, None)
        if isinstance(mv, dict):
            val = _extract_angle_from_mapping(mv)
            if val is not None:
                return float(val)
    return None

def _extract_angle_from_data_obj(data_obj: Any) -> float | None:
    """
    Try to read a ramp-angle-like scalar from a loaded .pt payload.
    Returns degrees when found.
    """
    if isinstance(data_obj, dict):
        val = _extract_angle_from_mapping(data_obj)
        if val is not None:
            return float(val)
        for k in ('snapshots', 'steps', 'time_steps', 'sequence', 'data_list'):
            if k in data_obj and isinstance(data_obj[k], list) and (len(data_obj[k]) > 0):
                return _extract_angle_from_data_obj(data_obj[k][0])
        return None
    if isinstance(data_obj, (list, tuple)):
        if len(data_obj) == 0:
            return None
        return _extract_angle_from_data_obj(data_obj[0])
    for k in ('ramp_angle_deg', 'ramp_angle', 'angle_deg', 'angle', 'theta_deg', 'theta'):
        if hasattr(data_obj, k):
            val = _to_scalar_float(getattr(data_obj, k))
            if val is not None:
                return float(val)
    if hasattr(data_obj, 'meta'):
        meta = getattr(data_obj, 'meta')
        if isinstance(meta, dict):
            return _extract_angle_from_mapping(meta)
    return None

def _extract_angle_from_path(path_like: str | None) -> float | None:
    if not path_like:
        return None
    s = os.path.basename(str(path_like))
    patterns = ('(?:^|[_\\-])ramp[-_]?angle[-_]?(-?\\d+(?:\\.\\d+)?)', '(?:^|[_\\-])angle[-_]?(-?\\d+(?:\\.\\d+)?)')
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m is not None:
            try:
                return float(m.group(1))
            except Exception:
                continue
    return None

def _extract_xy_from_snapshot_like(snapshot: Any) -> torch.Tensor | None:
    """
    Return centers (N,2) from a snapshot-like object when available.
    """
    keys = ('pos', 'xy', 'centers', 'pos_phys')
    if isinstance(snapshot, dict):
        for k in keys:
            if k in snapshot:
                v = snapshot.get(k)
                t = torch.as_tensor(v) if v is not None else None
                if torch.is_tensor(t) and t.ndim == 2 and (t.size(1) >= 2) and (t.numel() > 0):
                    return t[:, :2].to(torch.float32)
        return None
    for k in keys:
        if hasattr(snapshot, k):
            v = getattr(snapshot, k)
            t = torch.as_tensor(v) if v is not None else None
            if torch.is_tensor(t) and t.ndim == 2 and (t.size(1) >= 2) and (t.numel() > 0):
                return t[:, :2].to(torch.float32)
    return None

def _fit_ramp_line_from_centers(centers_xy: torch.Tensor, *, low_y_quantile: float=0.12, min_points: int=32) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fit a line from near-bottom boundary points:
      - point p0 on line (2,)
      - unit normal n (2,), oriented so median signed distance over all points is positive
    """
    if not torch.is_tensor(centers_xy) or centers_xy.ndim != 2 or centers_xy.size(1) < 2:
        raise RuntimeError(f'Cannot fit ramp line: expected centers (N,2), got {(None if not torch.is_tensor(centers_xy) else tuple(centers_xy.shape))}.')
    pts = centers_xy[:, :2].detach().to(device='cpu', dtype=torch.float64)
    n_all = int(pts.size(0))
    if n_all < 4:
        raise RuntimeError(f'Cannot fit ramp line: need >=4 centers, got {n_all}.')
    q = float(low_y_quantile)
    q = min(max(q, 0.01), 0.49)
    y = pts[:, 1]
    y_thr = torch.quantile(y, q)
    sel = y <= y_thr
    if int(sel.sum().item()) < int(max(4, min_points)):
        k = int(min(max(4, min_points), n_all))
        idx = torch.argsort(y)[:k]
        fit_pts = pts.index_select(0, idx)
    else:
        fit_pts = pts[sel]
    p0 = fit_pts.mean(dim=0)
    centered = fit_pts - p0
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    tang = vh[0]
    tang = tang / tang.norm().clamp_min(1e-12)
    n = torch.stack([-tang[1], tang[0]], dim=0)
    n = n / n.norm().clamp_min(1e-12)
    signed_all = (pts - p0.view(1, 2)) @ n.view(2, 1)
    med = torch.median(signed_all.view(-1))
    if float(med.item()) < 0.0:
        n = -n
    return (p0.to(torch.float32), n.to(torch.float32))

def _build_ramp_feature_context(cfg: Dict[str, Any], *, data_obj: Any, pt_path: str | None, mesh_spec_path: str | None) -> Dict[str, Any]:
    build_cfg = cfg.get('features', {}).get('build', {}) or {}
    include_ramp_angle = bool(build_cfg.get('include_ramp_angle', False))
    include_signed_dist = bool(build_cfg.get('include_signed_distance_to_ramp', False))
    ctx: Dict[str, Any] = {'include_ramp_angle': include_ramp_angle, 'include_signed_distance_to_ramp': include_signed_dist}
    if not (include_ramp_angle or include_signed_dist):
        return ctx
    angle_deg = _to_scalar_float(build_cfg.get('ramp_angle_deg', build_cfg.get('ramp_angle_degrees', None)))
    if angle_deg is None:
        angle_deg = _extract_angle_from_data_obj(data_obj)
    if angle_deg is None:
        angle_deg = _extract_angle_from_path(pt_path)
    if angle_deg is None:
        angle_deg = _extract_angle_from_path(mesh_spec_path)
    if angle_deg is not None:
        angle_deg = float(angle_deg)
        ctx['angle_deg'] = angle_deg
    if include_signed_dist:
        first_snapshot = None
        if isinstance(data_obj, list) and len(data_obj) > 0:
            first_snapshot = data_obj[0]
        elif isinstance(data_obj, dict):
            for k in ('snapshots', 'steps', 'time_steps', 'sequence', 'data_list'):
                vv = data_obj.get(k, None)
                if isinstance(vv, list) and len(vv) > 0:
                    first_snapshot = vv[0]
                    break
            if first_snapshot is None:
                first_snapshot = data_obj
        else:
            first_snapshot = data_obj
        xy0 = _extract_xy_from_snapshot_like(first_snapshot)
        if xy0 is None:
            raise RuntimeError('features.build.include_signed_distance_to_ramp=true but could not find centers (pos/xy/centers) in loaded data to fit ramp line.')
        q = float(build_cfg.get('signed_distance_low_y_quantile', 0.12))
        mpts = int(build_cfg.get('signed_distance_min_points', 32))
        p0, n = _fit_ramp_line_from_centers(xy0, low_y_quantile=q, min_points=mpts)
        ctx['distance_point'] = p0
        ctx['distance_normal'] = n
        normalize_dist = bool(build_cfg.get('signed_distance_normalize', True))
        dist_scale = 1.0
        if normalize_dist:
            bbox = cfg.get('data', {}).get('bbox', None)
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x0, x1, y0, y1 = map(float, bbox)
                dist_scale = float(np.hypot(x1 - x0, y1 - y0))
            else:
                xy_cpu = xy0.detach().to(torch.float64).cpu()
                x0 = float(xy_cpu[:, 0].min().item())
                x1 = float(xy_cpu[:, 0].max().item())
                y0 = float(xy_cpu[:, 1].min().item())
                y1 = float(xy_cpu[:, 1].max().item())
                dist_scale = float(np.hypot(x1 - x0, y1 - y0))
            if dist_scale <= 0.0:
                dist_scale = 1.0
        ctx['distance_scale'] = float(dist_scale)
    if include_ramp_angle:
        units = str(build_cfg.get('ramp_angle_units', 'radians')).strip().lower()
        if units not in ('degrees', 'degree', 'deg', 'radians', 'radian', 'rad'):
            raise ValueError('features.build.ramp_angle_units must be one of {radians, degrees} (aliases allowed).')
        if angle_deg is None:
            raise RuntimeError('features.build.include_ramp_angle=true but no ramp angle could be inferred. Set features.build.ramp_angle_deg explicitly or ensure the data/paths include angle metadata.')
        angle_feature = float(angle_deg)
        if units in ('radians', 'radian', 'rad'):
            angle_feature = float(np.deg2rad(angle_feature))
        ctx['angle_feature_value'] = float(angle_feature)
        ctx['angle_units'] = 'radians' if units in ('radians', 'radian', 'rad') else 'degrees'
    return ctx

def _ramp_feature_source_id_for_t(ramp_feature_ctx: Dict[str, Any] | None, step_t_abs: int | None) -> int | None:
    if ramp_feature_ctx is None or step_t_abs is None:
        return None
    src_by_t = ramp_feature_ctx.get('source_id_by_t', None)
    if src_by_t is None:
        return None
    try:
        ti = int(step_t_abs)
        if isinstance(src_by_t, (list, tuple)):
            if 0 <= ti < len(src_by_t):
                return int(src_by_t[ti])
            return None
        if isinstance(src_by_t, dict):
            if ti in src_by_t:
                return int(src_by_t[ti])
            ks = str(ti)
            return int(src_by_t[ks]) if ks in src_by_t else None
    except Exception:
        return None
    return None

def _build_X(feat, centers, levels, cfg, dt_hat=None, ramp_feature_ctx: Dict[str, Any] | None=None, step_t_abs: int | None=None):
    """
    Build node features for FeatureNet:
      [ physics_feat ] (+ [x,y] if use_pos) (+ [level] if use_level)
      (+ [dt_hat_fixed] if features.build.use_dt_hat_fixed)

    feat:    (N, F) on whatever device we want to run the model on (cpu/cuda/mps)
    centers: (N, 2) (typically from precompute; may be on cpu)
    levels:  (N,) or (N,1)
    """
    build_cfg = cfg.get('features', {}).get('build', {})
    use_pos = build_cfg.get('use_pos', True)
    use_level = build_cfg.get('use_level', True)
    use_dt_hat_fixed = bool(build_cfg.get('use_dt_hat_fixed', False))
    include_ramp_angle = bool(build_cfg.get('include_ramp_angle', False))
    include_signed_dist = bool(build_cfg.get('include_signed_distance_to_ramp', False))
    dev = feat.device
    Xs = [feat]
    if use_pos:
        Xs.append(centers.to(dev))
    if use_level:
        lvl = levels
        if lvl.dim() == 1:
            lvl = lvl.unsqueeze(-1)
        Xs.append(lvl.to(dev).to(feat.dtype))
    if use_dt_hat_fixed:
        if dt_hat is None:
            raise RuntimeError('features.build.use_dt_hat_fixed=true but dt_hat was not provided to _build_X.')
        Xs.append(_dt_hat_feature_column(dt_hat, n_nodes=int(feat.size(0)), device=dev, dtype=feat.dtype))
    if include_ramp_angle:
        if ramp_feature_ctx is None:
            raise RuntimeError('features.build.include_ramp_angle=true but ramp_feature_ctx was not provided.')
        aval = None
        sid = _ramp_feature_source_id_for_t(ramp_feature_ctx, step_t_abs)
        by_source = ramp_feature_ctx.get('angle_feature_by_source', None)
        if sid is not None and isinstance(by_source, dict):
            aval = by_source.get(int(sid), None)
            if aval is None:
                aval = by_source.get(str(int(sid)), None)
        if aval is None:
            by_t = ramp_feature_ctx.get('angle_feature_by_t', None)
            if isinstance(by_t, (list, tuple)) and step_t_abs is not None:
                ti = int(step_t_abs)
                if 0 <= ti < len(by_t):
                    aval = by_t[ti]
            elif isinstance(by_t, dict) and step_t_abs is not None:
                ti = int(step_t_abs)
                aval = by_t.get(ti, by_t.get(str(ti), None))
        if aval is None:
            aval = ramp_feature_ctx.get('angle_feature_value', None)
        if aval is None:
            raise RuntimeError('features.build.include_ramp_angle=true but angle_feature_value is missing. Set features.build.ramp_angle_deg or provide angle metadata in data/path.')
        a = torch.tensor(float(aval), device=dev, dtype=feat.dtype).view(1, 1)
        Xs.append(a.expand(int(feat.size(0)), 1))
    if include_signed_dist:
        if centers is None or not torch.is_tensor(centers) or centers.ndim != 2 or (centers.size(1) < 2):
            raise RuntimeError('features.build.include_signed_distance_to_ramp=true requires centers tensor with shape (N,2).')
        if ramp_feature_ctx is None:
            ramp_feature_ctx = {}
        p0 = ramp_feature_ctx.get('distance_point', None)
        n = ramp_feature_ctx.get('distance_normal', None)
        dscale = float(ramp_feature_ctx.get('distance_scale', 1.0))
        sid = _ramp_feature_source_id_for_t(ramp_feature_ctx, step_t_abs)
        by_source = ramp_feature_ctx.get('distance_by_source', None)
        if sid is not None and isinstance(by_source, dict):
            src_pack = by_source.get(int(sid), by_source.get(str(int(sid)), None))
            if isinstance(src_pack, dict):
                p0 = src_pack.get('point', p0)
                n = src_pack.get('normal', n)
                dscale = float(src_pack.get('scale', dscale))
        if p0 is None or n is None:
            p0, n = _fit_ramp_line_from_centers(centers.detach().to(device='cpu', dtype=torch.float32), low_y_quantile=float(build_cfg.get('signed_distance_low_y_quantile', 0.12)), min_points=int(build_cfg.get('signed_distance_min_points', 32)))
        p0_t = torch.as_tensor(p0, dtype=feat.dtype, device=dev).view(1, 2)
        n_t = torch.as_tensor(n, dtype=feat.dtype, device=dev).view(2, 1)
        signed = (centers[:, :2].to(device=dev, dtype=feat.dtype) - p0_t) @ n_t
        signed = signed.view(-1, 1)
        if dscale <= 0.0:
            dscale = 1.0
        if bool(build_cfg.get('signed_distance_normalize', True)):
            signed = signed / float(dscale)
        Xs.append(signed)
    return torch.cat(Xs, dim=-1)

def _get_bbox(cfg: Dict[str, Any]) -> Tuple[float, float, float, float]:
    if cfg.get('data', {}).get('bbox'):
        xmin, xmax, ymin, ymax = cfg['data']['bbox']
    else:
        dom = cfg.get('domain', {})
        xmin = float(dom.get('xmin', 0.0))
        xmax = float(dom.get('xmax', 1.0))
        ymin = float(dom.get('ymin', 0.0))
        ymax = float(dom.get('ymax', 1.0))
    return (xmin, xmax, ymin, ymax)

def _select_idw_backend(*, src_n: int, requested_chunk: int, out_device: torch.device) -> tuple[torch.device, int]:
    """
    Choose a safer device/chunk for IDW cdist blocks.
    - On MPS, run IDW on CPU to avoid large MPS cdist allocations.
    - Cap chunk so chunk*src_n pairwise matrix stays bounded.
    """
    out_dev = torch.device(out_device)
    idw_dev = torch.device('cpu') if out_dev.type == 'mps' else out_dev
    if src_n <= 0:
        return (idw_dev, max(1, int(requested_chunk)))
    max_pair_elems = 64000000 if idw_dev.type == 'cpu' else 32000000
    cap_chunk = max(1, int(max_pair_elems // int(src_n)))
    eff_chunk = max(1, min(int(requested_chunk), cap_chunk))
    return (idw_dev, eff_chunk)

def _cell_level_ij_from_centers(*, centers: torch.Tensor, levels: torch.Tensor, H: int, W: int, bbox: Tuple[float, float, float, float], refine_ratio: int, index_mode: str='round') -> tuple[torch.Tensor, torch.Tensor]:
    """
    Recover integer (row, col) indices of each cell on its own level grid from centers.
    """
    if centers.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=centers.device)
        return (empty, empty)
    dev = centers.device
    lv = levels.view(-1).long()
    max_l = int(torch.clamp(lv.max(), min=0).item())
    rr = int(refine_ratio)
    rr_pows = torch.tensor([rr ** i for i in range(max_l + 1)], device=dev, dtype=torch.float32)
    ww_l = (float(W) * rr_pows).index_select(0, lv.clamp(0, max_l))
    hh_l = (float(H) * rr_pows).index_select(0, lv.clamp(0, max_l))
    xmin, xmax, ymin, ymax = [float(v) for v in bbox]
    xs = max(float(xmax - xmin), 1e-12)
    ys = max(float(ymax - ymin), 1e-12)
    mode = str(index_mode).strip().lower()
    if mode == 'round':
        col_f = (centers[:, 0].to(torch.float32) - xmin) * ww_l / xs - 0.5
        row_f = (centers[:, 1].to(torch.float32) - ymin) * hh_l / ys - 0.5
        col = torch.round(col_f).to(torch.long)
        row = torch.round(row_f).to(torch.long)
    elif mode == 'floor':
        col_f = (centers[:, 0].to(torch.float32) - xmin) * ww_l / xs
        row_f = (centers[:, 1].to(torch.float32) - ymin) * hh_l / ys
        col = torch.floor(col_f).to(torch.long)
        row = torch.floor(row_f).to(torch.long)
    else:
        raise ValueError(f"Unsupported index_mode={index_mode!r}; expected 'round' or 'floor'.")
    ww_i = ww_l.to(torch.long)
    hh_i = hh_l.to(torch.long)
    col = torch.minimum(torch.maximum(col, torch.zeros_like(col)), ww_i - 1)
    row = torch.minimum(torch.maximum(row, torch.zeros_like(row)), hh_i - 1)
    return (row, col)

def _cell_keys_level_ij(*, levels: torch.Tensor, row: torch.Tensor, col: torch.Tensor, H: int, W: int, refine_ratio: int) -> torch.Tensor:
    """
    Build a unique int64 key for (level,row,col), with level-specific row/col.
    """
    if levels.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=levels.device)
    lv = levels.view(-1).long()
    rr = int(refine_ratio)
    max_l = int(torch.clamp(lv.max(), min=0).item())
    h_max = int(H * rr ** max_l)
    w_max = int(W * rr ** max_l)
    h_max = max(1, h_max)
    w_max = max(1, w_max)
    key = (lv * int(h_max) + row.view(-1).long()) * int(w_max) + col.view(-1).long()
    return key.to(torch.long)

class _RuntimeMultiResGtLookup:
    """
    H5-backed GT lookup on runtime meshes using precomputed uniform-mesh caches.

    The lookup key is (level, row, col), where row/col are reconstructed from centers.
    """

    def __init__(self, *, level_files: Dict[int, str], H: int, W: int, bbox: Tuple[float, float, float, float], refine_ratio: int, max_level: int, max_cached_feature_steps: int=16):
        if h5py is None:
            raise ImportError('Multi-resolution GT lookup requires h5py, but import failed.')
        if not level_files:
            raise ValueError('Runtime multi-resolution lookup was enabled, but no level files were provided.')
        self.H = int(H)
        self.W = int(W)
        self.bbox = tuple((float(v) for v in bbox))
        self.refine_ratio = int(refine_ratio)
        self.max_level = int(max(0, max_level))
        self._key_h = int(self.H * self.refine_ratio ** self.max_level)
        self._key_w = int(self.W * self.refine_ratio ** self.max_level)
        self._h5_by_file_level: Dict[int, Any] = {}
        self._file_level_index: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}
        self._source_file_for_level: Dict[int, int] = {}
        self._feature_cache: 'OrderedDict[Tuple[int, int, str], torch.Tensor]' = OrderedDict()
        self._max_cached_feature_steps = max(1, int(max_cached_feature_steps))
        self.level_files = dict(sorted(((int(k), str(v)) for k, v in level_files.items())))
        for file_level, path in self.level_files.items():
            f = h5py.File(path, 'r')
            self._h5_by_file_level[file_level] = f
            g_static = self._select_static_group(f)
            if g_static is None:
                raise RuntimeError(f'H5 file has no /static or /tXXXXX groups: {path}')
            if 'pred_centers' not in g_static or 'pred_levels' not in g_static:
                raise RuntimeError(f'H5 file is missing static geometry datasets in {path} (need pred_centers and pred_levels).')
            centers = np.asarray(g_static['pred_centers'][...], dtype=np.float32)
            levels = np.asarray(g_static['pred_levels'][...], dtype=np.int64)
            self._file_level_index[file_level] = self._build_level_index(centers, levels)
        for lv in range(self.max_level + 1):
            src_file = None
            if lv in self._file_level_index and lv in self._file_level_index[lv]:
                src_file = lv
            else:
                for file_level in sorted(self._file_level_index.keys()):
                    if lv in self._file_level_index[file_level]:
                        src_file = file_level
                        break
            if src_file is not None:
                self._source_file_for_level[lv] = src_file

    @staticmethod
    def _select_static_group(f: Any):
        if 'static' in f:
            return f['static']
        if 't00001' in f:
            return f['t00001']
        t_groups = sorted([k for k in f.keys() if str(k).startswith('t')])
        if not t_groups:
            return None
        return f[t_groups[0]]

    def close(self):
        for f in self._h5_by_file_level.values():
            try:
                f.close()
            except Exception:
                pass
        self._h5_by_file_level.clear()
        self._feature_cache.clear()

    def _make_keys(self, levels: torch.Tensor, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        lv = levels.view(-1).long()
        rr = row.view(-1).long()
        cc = col.view(-1).long()
        return (lv * int(self._key_h) + rr) * int(self._key_w) + cc

    def _build_level_index(self, centers: np.ndarray, levels: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
        centers_t = torch.from_numpy(centers.astype(np.float32, copy=False))
        levels_t = torch.from_numpy(levels.astype(np.int64, copy=False))
        row_t, col_t = _cell_level_ij_from_centers(centers=centers_t, levels=levels_t, H=self.H, W=self.W, bbox=self.bbox, refine_ratio=self.refine_ratio, index_mode='floor')
        keys = self._make_keys(levels_t, row_t, col_t).cpu().numpy().astype(np.int64, copy=False)
        idx_all = np.arange(keys.shape[0], dtype=np.int64)
        out: Dict[int, Dict[str, np.ndarray]] = {}
        for lv in np.unique(levels.astype(np.int64, copy=False)):
            lv_i = int(lv)
            mask = levels == lv_i
            k_l = keys[mask]
            i_l = idx_all[mask]
            order = np.argsort(k_l)
            out[lv_i] = {'keys_sorted': k_l[order], 'idx_sorted': i_l[order]}
        return out

    def _cache_get(self, key: Tuple[int, int, str]) -> Optional[torch.Tensor]:
        v = self._feature_cache.get(key, None)
        if v is not None:
            self._feature_cache.move_to_end(key)
        return v

    def _cache_put(self, key: Tuple[int, int, str], value: torch.Tensor):
        self._feature_cache[key] = value
        self._feature_cache.move_to_end(key)
        while len(self._feature_cache) > self._max_cached_feature_steps:
            self._feature_cache.popitem(last=False)

    def _read_feature(self, *, file_level: int, t_dst: int, dataset_name: str) -> torch.Tensor:
        ck = (int(file_level), int(t_dst), str(dataset_name))
        cached = self._cache_get(ck)
        if cached is not None:
            return cached
        f = self._h5_by_file_level[int(file_level)]
        gname = f't{int(t_dst):05d}'
        if gname not in f:
            raise KeyError(f"Missing timestep group '{gname}' in file-level {file_level}.")
        g = f[gname]
        if dataset_name not in g:
            raise KeyError(f"Missing dataset '{dataset_name}' in group '{gname}' (file-level {file_level}).")
        arr = np.asarray(g[dataset_name][...], dtype=np.float32)
        ten = torch.from_numpy(arr)
        self._cache_put(ck, ten)
        return ten

    @torch.no_grad()
    def lookup(self, *, t_dst: int, dataset_name: str, pred_centers: torch.Tensor, pred_levels: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor, Dict[str, Any]]:
        lv_t = pred_levels.view(-1).long()
        N = int(lv_t.numel())
        matched_cpu = torch.zeros((N,), dtype=torch.bool, device='cpu')
        stats: Dict[str, Any] = {'requested': N, 'matched': 0, 'missing': N, 'by_level': {}}
        if N == 0:
            return (None, matched_cpu.to(device=pred_centers.device), stats)
        row_t, col_t = _cell_level_ij_from_centers(centers=pred_centers, levels=lv_t, H=self.H, W=self.W, bbox=self.bbox, refine_ratio=self.refine_ratio, index_mode='floor')
        key_np = self._make_keys(lv_t, row_t, col_t).detach().cpu().numpy().astype(np.int64, copy=False)
        lvl_np = lv_t.detach().cpu().numpy().astype(np.int64, copy=False)
        out_cpu: Optional[torch.Tensor] = None
        for lv in sorted(np.unique(lvl_np).tolist()):
            lv_i = int(lv)
            dst_pos = np.nonzero(lvl_np == lv_i)[0]
            if dst_pos.size == 0:
                continue
            src_file_level = self._source_file_for_level.get(lv_i, None)
            if src_file_level is None:
                stats['by_level'][lv_i] = {'requested': int(dst_pos.size), 'matched': 0, 'source_file_level': None}
                continue
            src_level_map = self._file_level_index.get(src_file_level, {}).get(lv_i, None)
            if src_level_map is None:
                stats['by_level'][lv_i] = {'requested': int(dst_pos.size), 'matched': 0, 'source_file_level': int(src_file_level)}
                continue
            keys_sorted = src_level_map['keys_sorted']
            idx_sorted = src_level_map['idx_sorted']
            q_keys = key_np[dst_pos]
            pos = np.searchsorted(keys_sorted, q_keys, side='left')
            valid = pos < keys_sorted.shape[0]
            if np.any(valid):
                eq = np.zeros_like(valid, dtype=np.bool_)
                eq[valid] = keys_sorted[pos[valid]] == q_keys[valid]
                valid = eq
            n_match = int(np.count_nonzero(valid))
            stats['by_level'][lv_i] = {'requested': int(dst_pos.size), 'matched': n_match, 'source_file_level': int(src_file_level)}
            if n_match <= 0:
                continue
            feat_cpu = self._read_feature(file_level=src_file_level, t_dst=int(t_dst), dataset_name=str(dataset_name))
            if out_cpu is None:
                out_cpu = torch.empty((N, int(feat_cpu.shape[1])), dtype=torch.float32, device='cpu')
            src_idx_np = idx_sorted[pos[valid]].astype(np.int64, copy=False)
            dst_idx_np = dst_pos[valid].astype(np.int64, copy=False)
            src_idx_t = torch.from_numpy(src_idx_np).to(dtype=torch.long)
            vals = feat_cpu.index_select(0, src_idx_t)
            dst_idx_t = torch.from_numpy(dst_idx_np).to(dtype=torch.long)
            out_cpu.index_copy_(0, dst_idx_t, vals.to(dtype=torch.float32, device='cpu'))
            matched_cpu[dst_idx_t] = True
        matched_n = int(matched_cpu.sum().item())
        stats['matched'] = matched_n
        stats['missing'] = int(N - matched_n)
        out = None if out_cpu is None else out_cpu.to(device=pred_centers.device, dtype=torch.float32)
        matched = matched_cpu.to(device=pred_centers.device)
        return (out, matched, stats)

def _runtime_coarse_parent_counts(*, parents: torch.Tensor, H: int, W: int) -> torch.Tensor:
    p = torch.as_tensor(parents, dtype=torch.long).view(-1)
    cnt = torch.zeros(int(H) * int(W), dtype=torch.float32, device=p.device)
    if p.numel() == 0:
        return cnt
    p = p.clamp(0, int(H) * int(W) - 1)
    ones = torch.ones((int(p.numel()),), dtype=torch.float32, device=p.device)
    cnt.index_add_(0, p, ones)
    return cnt

def _runtime_build_coarse_rc_grid(H: int, W: int, device: torch.device | None=None) -> torch.Tensor:
    dev = device if device is not None else torch.device('cpu')
    rr = torch.arange(int(H), dtype=torch.float32, device=dev)
    cc = torch.arange(int(W), dtype=torch.float32, device=dev)
    gy, gx = torch.meshgrid(rr, cc, indexing='ij')
    return torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=1).contiguous()

def _runtime_fill_empty_interior_coarse_nearest(*, coarse_chw: torch.Tensor, parents: torch.Tensor, H: int, W: int, physical_parent_mask: torch.Tensor, rc_grid: torch.Tensor | None=None, chunk: int=1024) -> torch.Tensor:
    coarse = torch.as_tensor(coarse_chw, dtype=torch.float32)
    if coarse.ndim != 3:
        raise ValueError(f'coarse_chw must be (C,H,W), got {tuple(coarse.shape)}')
    C, HH, WW = (int(coarse.shape[0]), int(coarse.shape[1]), int(coarse.shape[2]))
    if HH != int(H) or WW != int(W):
        raise ValueError(f'coarse_chw spatial shape mismatch: got {(HH, WW)} expected {(int(H), int(W))}')
    phys = torch.as_tensor(physical_parent_mask, dtype=torch.bool, device=coarse.device).view(-1)
    if int(phys.numel()) != int(H) * int(W):
        raise ValueError(f'physical_parent_mask must have {int(H) * int(W)} entries, got {int(phys.numel())}')
    counts = _runtime_coarse_parent_counts(parents=parents, H=int(H), W=int(W)).to(device=coarse.device)
    fill_flat = phys & (counts <= 0.0)
    valid_flat = phys & (counts > 0.0)
    if not bool(fill_flat.any()) or not bool(valid_flat.any()):
        return coarse
    valid_idx = torch.nonzero(valid_flat, as_tuple=False).view(-1)
    fill_idx = torch.nonzero(fill_flat, as_tuple=False).view(-1)
    if valid_idx.numel() == 0 or fill_idx.numel() == 0:
        return coarse
    if rc_grid is None:
        rc = _runtime_build_coarse_rc_grid(int(H), int(W), device=coarse.device)
    else:
        rc = torch.as_tensor(rc_grid, dtype=torch.float32, device=coarse.device)
        if rc.ndim != 2 or int(rc.shape[1]) != 2 or int(rc.shape[0]) != int(H) * int(W):
            raise ValueError(f'rc_grid must be (H*W,2), got {tuple(rc.shape)}')
    valid_rc = rc.index_select(0, valid_idx)
    fill_rc = rc.index_select(0, fill_idx)
    flat = coarse.view(C, int(H) * int(W))
    out = flat.clone()
    chunk = max(1, int(chunk))
    for s in range(0, int(fill_idx.numel()), chunk):
        e = min(s + chunk, int(fill_idx.numel()))
        q_idx = fill_idx[s:e]
        q_rc = fill_rc[s:e]
        d = q_rc[:, None, :] - valid_rc[None, :, :]
        d2 = (d * d).sum(dim=-1)
        nn = torch.argmin(d2, dim=1)
        src_idx = valid_idx.index_select(0, nn)
        out[:, q_idx] = out[:, src_idx]
    return out.view(C, int(H), int(W)).contiguous()
_RUNTIME_WEDGE_CONSTRAINTS_CACHE: Dict[str, Dict[str, Any]] = {}

def _runtime_lookup_mask_values_by_level(mask_by_level: Dict[int, torch.Tensor], *, levels: torch.Tensor, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
    lv = levels.view(-1).long()
    rv = row.view(-1).long()
    cv = col.view(-1).long()
    out = torch.zeros(lv.shape[0], dtype=torch.bool, device=lv.device)
    if lv.numel() == 0:
        return out
    lmin = int(lv.min().item())
    lmax = int(lv.max().item())
    for Lint in range(lmin, lmax + 1):
        if Lint not in mask_by_level:
            continue
        sel = lv == Lint
        if not bool(sel.any()):
            continue
        M = mask_by_level[Lint]
        out[sel] = M[rv[sel], cv[sel]]
    return out

def _runtime_wedge_constraints_cache_key(*, wedge_path, cfg: Dict[str, Any], H: int, W: int, device: torch.device) -> str:
    verts = np.asarray(wedge_path.vertices, dtype=np.float64)
    vhash = hashlib.sha1(verts.tobytes()).hexdigest()[:20]
    bbox = _get_bbox(cfg)
    xmin, xmax, ymin, ymax = [float(v) for v in bbox]
    rr = _get_refine_ratio(cfg)
    Lmax = int((cfg.get('policy', {}) or {}).get('max_level', 3))
    dev = torch.device(device)
    dkey = f'{dev.type}:{(dev.index if dev.index is not None else -1)}'
    return f'v={vhash}|H={int(H)}|W={int(W)}|Lmax={int(Lmax)}|rr={int(rr)}|bbox={xmin:.16g},{xmax:.16g},{ymin:.16g},{ymax:.16g}|dev={dkey}'

def _get_runtime_wedge_constraints(*, cfg: Dict[str, Any], H: int, W: int, device: torch.device, wedge_path) -> Dict[str, Any]:
    key = _runtime_wedge_constraints_cache_key(wedge_path=wedge_path, cfg=cfg, H=H, W=W, device=device)
    got = _RUNTIME_WEDGE_CONSTRAINTS_CACHE.get(key, None)
    if got is not None:
        return got
    pol = cfg.get('policy', {}) or {}
    Lmax = int(pol.get('max_level', 3))
    rr = _get_refine_ratio(cfg)
    bbox = _get_bbox(cfg)
    xspan = float(bbox[1] - bbox[0])
    yspan = float(bbox[3] - bbox[2])
    radius = float(pol.get('wedge_clip_radius', 1e-09 * max(xspan, yspan)))
    dev = torch.device(device)
    lookup = _get_wedge_clip_level_lookup(wedge_path=wedge_path, H=H, W=W, Lmax=Lmax, refine_ratio=rr, bbox=bbox, radius=radius)
    full_by_level, intersect_by_level = _lookup_level_masks_on_device(lookup, dev)
    leaf_full: Dict[int, torch.Tensor] = {}
    parent_intersect: Dict[int, torch.Tensor] = {}
    parent_boundary: Dict[int, torch.Tensor] = {}
    for l in range(0, Lmax + 1):
        leaf_full[l] = full_by_level[l].to(device=dev, dtype=torch.bool)
    for L in range(1, Lmax + 1):
        l = L - 1
        inter = intersect_by_level[l].to(device=dev, dtype=torch.bool)
        full = full_by_level[l].to(device=dev, dtype=torch.bool)
        parent_intersect[L] = inter
        parent_boundary[L] = inter & ~full
    out = {'max_level': int(Lmax), 'refine_ratio': int(rr), 'bbox': tuple((float(v) for v in bbox)), 'leaf_full': leaf_full, 'parent_intersect': parent_intersect, 'parent_boundary': parent_boundary}
    _RUNTIME_WEDGE_CONSTRAINTS_CACHE[key] = out
    print(f'[RUNTIME-WEDGE] built static constraints (H={int(H)} W={int(W)} Lmax={int(Lmax)} rr={int(rr)})', flush=True)
    return out

def _runtime_apply_wedge_constraints_to_masks(*, masks_by_level: Dict[int, torch.Tensor], constraints: Dict[str, Any]) -> Dict[int, torch.Tensor]:
    rr = int(constraints['refine_ratio'])
    Lmax = int(constraints['max_level'])
    out: Dict[int, torch.Tensor] = {}
    for L in range(1, Lmax + 1):
        inter = constraints['parent_intersect'][L]
        boundary = constraints['parent_boundary'][L]
        m = masks_by_level.get(L, None)
        if m is None:
            m = torch.zeros_like(inter, dtype=torch.bool)
        else:
            m = m.to(device=inter.device, dtype=torch.bool)
        if tuple(m.shape) != tuple(inter.shape):
            raise RuntimeError(f'Runtime wedge constraint shape mismatch at L={L}: mask={tuple(m.shape)} expected={tuple(inter.shape)}')
        m = m & inter | boundary
        if L > 1:
            allow = F.interpolate(out[L - 1].float().unsqueeze(0).unsqueeze(0), scale_factor=float(rr), mode='nearest')[0, 0].to(torch.bool)
            m = m & allow[:m.shape[0], :m.shape[1]]
        out[L] = m
    return out

def _runtime_filter_mesh_with_wedge_constraints(*, pred_centers: torch.Tensor, pred_levels: torch.Tensor, parent_flat: torch.Tensor, cfg: Dict[str, Any], H: int, W: int, device: torch.device, constraints: Dict[str, Any], timing_out: Dict[str, float] | None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = torch.device(device)
    timing = {'lookup_classify_s': 0.0, 'edge_build_s': 0.0}
    rr = int(constraints['refine_ratio'])
    bbox = tuple(constraints['bbox'])
    leaf_full: Dict[int, torch.Tensor] = constraints['leaf_full']
    t_cls0 = time.perf_counter()
    row, col = _cell_level_ij_from_centers(centers=pred_centers, levels=pred_levels, H=H, W=W, bbox=bbox, refine_ratio=rr)
    lvl = pred_levels.view(-1).long()
    keep = _runtime_lookup_mask_values_by_level(leaf_full, levels=lvl, row=row, col=col)
    pred_centers = pred_centers[keep]
    pred_levels = pred_levels[keep]
    parent_flat = parent_flat[keep]
    timing['lookup_classify_s'] = float(time.perf_counter() - t_cls0)
    if pred_centers.numel() == 0:
        raise RuntimeError('Runtime wedge constrained mesh became empty after full-cell filter.')
    t_edge0 = time.perf_counter()
    edge_method = str(cfg.get('edges', {}).get('method', 'amr_local_knn')).lower()
    pred_ei = build_amr_face_adjacency_edges(pred_centers, pred_levels, H, W, bbox=bbox, return_edge_attr=False, refine_ratio=rr,
        backend=cfg.get('train', {}).get('runtime_mesh', {}).get('adjacency_backend', 'auto')).to(dev, dtype=torch.long)
    if pred_ei.numel() == 0 or int(pred_ei.shape[1]) == 0:
        raise RuntimeError('Empty edge_index after runtime wedge-constrained filtering.')
    timing['edge_build_s'] = float(time.perf_counter() - t_edge0)
    mask_pred_parent = _mask_from_parent_indices(parent_flat, H, W, dev)
    if isinstance(timing_out, dict):
        timing_out.clear()
        timing_out.update(timing)
    return (pred_centers, pred_levels, parent_flat, pred_ei, mask_pred_parent)

def _runtime_parents_from_level_ij(*, levels: torch.Tensor, row: torch.Tensor, col: torch.Tensor, H: int, W: int, refine_ratio: int) -> torch.Tensor:
    if levels.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=levels.device)
    lv = levels.view(-1).long()
    rr = int(refine_ratio)
    max_l = int(torch.clamp(lv.max(), min=0).item())
    rr_pows = torch.tensor([rr ** i for i in range(max_l + 1)], device=levels.device, dtype=torch.long)
    scale = rr_pows.index_select(0, lv.clamp(0, max_l))
    i0 = torch.div(col.view(-1).long(), scale, rounding_mode='floor')
    j0 = torch.div(row.view(-1).long(), scale, rounding_mode='floor')
    return (j0 * int(W) + i0).to(torch.long)

def _map_pred_to_next_pred(pred_centers_src, feats_src, levels_src, parents_src, pred_centers_dst, levels_dst, parents_dst, mask_pred_dst, H, W, knn_k=8, chunk=8192, *, bbox: Tuple[float, float, float, float], refine_ratio: int=2, knn_backend: str='exact', knn_backend_kwargs: Dict[str, Any] | None=None):
    """
    Map predicted features from mesh k -> k+1.

    - First: copy coarse cells where parent index is unchanged between src and dst.
    - Then: run IDW *only* for the remaining nodes.

    All tensors are moved to the device of feats_src.
    """
    dev = feats_src.device
    idw_dev, idw_chunk = _select_idw_backend(src_n=int(pred_centers_src.shape[0]), requested_chunk=int(chunk), out_device=dev)
    pred_centers_src = pred_centers_src.to(dev)
    pred_centers_dst = pred_centers_dst.to(dev)
    levels_src = levels_src.to(dev)
    levels_dst = levels_dst.to(dev)
    parents_src = parents_src.to(dev)
    parents_dst = parents_dst.to(dev)
    mask_pred_dst = mask_pred_dst.to(dev)
    feats_src = feats_src.to(dev)
    N_dst, F = (pred_centers_dst.shape[0], feats_src.shape[1])
    out = feats_src.new_empty((N_dst, F), device=dev)
    src_is_coarse = levels_src == 0
    dst_is_coarse = levels_dst == 0
    parent_src = parents_src.clamp(0, H * W - 1)
    parent_dst = parents_dst.clamp(0, H * W - 1)
    src_coarse_idx = torch.nonzero(src_is_coarse, as_tuple=False).view(-1)
    dst_coarse_idx = torch.nonzero(dst_is_coarse, as_tuple=False).view(-1)
    need_idw = torch.ones(N_dst, dtype=torch.bool, device=dev)
    if src_coarse_idx.numel() > 0 and dst_coarse_idx.numel() > 0:
        lookup = -torch.ones(H * W, dtype=torch.long, device=dev)
        lookup[parent_src[src_coarse_idx]] = src_coarse_idx
        src_match_for_dst = lookup[parent_dst[dst_coarse_idx]]
        has_match = src_match_for_dst >= 0
        if has_match.any():
            dst_keep = dst_coarse_idx[has_match]
            src_keep = src_match_for_dst[has_match]
            out[dst_keep] = feats_src[src_keep]
            need_idw[dst_keep] = False
    if pred_centers_src.numel() > 0 and pred_centers_dst.numel() > 0:
        src_row, src_col = _cell_level_ij_from_centers(centers=pred_centers_src, levels=levels_src, H=H, W=W, bbox=bbox, refine_ratio=int(refine_ratio))
        dst_row, dst_col = _cell_level_ij_from_centers(centers=pred_centers_dst, levels=levels_dst, H=H, W=W, bbox=bbox, refine_ratio=int(refine_ratio))
        src_key = _cell_keys_level_ij(levels=levels_src, row=src_row, col=src_col, H=H, W=W, refine_ratio=int(refine_ratio))
        dst_key = _cell_keys_level_ij(levels=levels_dst, row=dst_row, col=dst_col, H=H, W=W, refine_ratio=int(refine_ratio))
        src_order = torch.argsort(src_key)
        src_key_sorted = src_key[src_order]
        src_idx_sorted = src_order
        dst_need_idx = need_idw.nonzero(as_tuple=True)[0]
        if dst_need_idx.numel() > 0:
            dst_need_key = dst_key[dst_need_idx]
            pos = torch.searchsorted(src_key_sorted, dst_need_key)
            valid = (pos >= 0) & (pos < src_key_sorted.numel())
            if valid.any():
                eq = torch.zeros_like(valid, dtype=torch.bool)
                eq[valid] = src_key_sorted[pos[valid]] == dst_need_key[valid]
                if eq.any():
                    dst_keep = dst_need_idx[eq]
                    src_keep = src_idx_sorted[pos[eq]]
                    out[dst_keep] = feats_src[src_keep]
                    need_idw[dst_keep] = False
    q_idx = need_idw.nonzero(as_tuple=True)[0]
    if q_idx.numel() > 0:
        backend_kwargs = dict(knn_backend_kwargs or {})
        q_pts = pred_centers_dst[q_idx].to(idw_dev, dtype=torch.float32)
        src_pts = pred_centers_src.to(idw_dev, dtype=torch.float32)
        src_feats = feats_src.to(idw_dev)
        idx_map, w_map = build_idw_map(q_pts, src_pts, k=knn_k, chunk=idw_chunk, backend=knn_backend, **backend_kwargs)
        vals = apply_idw_map(idx_map, w_map, src_feats).to(dev)
        out[q_idx] = vals
    return out

def _mask_from_parent_indices(parents: torch.Tensor, H: int, W: int, device: torch.device) -> torch.Tensor:
    m = torch.zeros(H * W, dtype=torch.bool, device=device)
    if parents is not None and torch.is_tensor(parents) and (parents.numel() > 0):
        p = parents.view(-1).long().clamp_(0, H * W - 1)
        m[p] = True
    return m.view(H, W)

@torch.no_grad()
def _runtime_predict_masks_from_fast_gradients(*, centers_t: torch.Tensor, level_t: torch.Tensor, feat_policy: torch.Tensor, parents_t: torch.Tensor, mask_t_parent: torch.Tensor, cfg: Dict[str, Any], H: int, W: int, dx: float, dy: float, device: torch.device, physical_parent_mask: torch.Tensor | None=None, coarse_rc_grid: torch.Tensor | None=None, use_prev_state_masks: bool=True, debug_out: Dict[str, Any] | None=None, timing_out: Dict[str, float] | None=None) -> Dict[int, torch.Tensor]:
    """
    Fast gradient backend for runtime mesh prediction.
    - Aggregate dynamic node features to coarse HxW parent grid.
    - Compute per-channel coarse-grid gradient magnitudes.
    - Combine channels and apply hierarchical refine masks with parent gating.

    This keeps the existing "gradient" backend unchanged and provides a fast
    alternative for timing/quality comparison.
    """
    dev = torch.device(device)
    timing: Dict[str, float] = {'predict_coarse_agg_s': 0.0, 'predict_fill_empty_s': 0.0, 'predict_grad_score_s': 0.0, 'predict_prev_masks_s': 0.0, 'predict_hierarchy_s': 0.0, 'predict_threshold_hysteresis_s': 0.0, 'predict_dilation_s': 0.0, 'predict_pool_up_s': 0.0}
    pol = cfg.get('policy', {}) or {}
    rt_cfg = cfg.get('train', {}).get('runtime_mesh', {}) or {}
    gf_cfg = rt_cfg.get('gradient_fast', {}) or {}
    if not isinstance(gf_cfg, dict):
        gf_cfg = {}
    rr = _get_refine_ratio(cfg)
    Lmax = int(pol.get('max_level', cfg.get('data', {}).get('L_max', 3)))
    p_pow = float(pol.get('p', 1.0))
    combine = str(pol.get('combine', 'l2')).strip().lower()
    weights = pol.get('refine_weights', None)
    tau_by_level = pol.get('tau_by_level', None)
    tau_low_def = float(pol.get('tau_low', 0.02))
    tau_high_def = float(pol.get('tau_high', 0.03))
    mode = str(pol.get('hysteresis_mode', 'absolute')).strip().lower()
    if mode not in ('absolute', 'percentile'):
        raise ValueError(f"policy.hysteresis_mode must be 'absolute' or 'percentile', got {mode!r}")
    pct_low_default = float(pol.get('percentile_low', 75.0))
    pct_high_default = float(pol.get('percentile_high', 90.0))
    pct_by_level = pol.get('percentiles_by_level', {})
    if not isinstance(pct_by_level, dict):
        pct_by_level = {}
    pct_mode = str(pct_by_level.get('selection', 'auto')).strip().lower()
    if pct_mode not in ('auto', 'global', 'per_level', 'per-level', 'level', 'levels'):
        raise ValueError('policy.percentiles_by_level.selection must be one of: auto, global, per_level')
    use_global_pct = pct_mode == 'global'
    use_level_pct = pct_mode in ('per_level', 'per-level', 'level', 'levels')

    def _pct_level_entry(L: int):
        if L in pct_by_level:
            return pct_by_level[L]
        s = str(L)
        if s in pct_by_level:
            return pct_by_level[s]
        return None

    def _taus(L: int) -> tuple[float, float]:
        if isinstance(tau_by_level, dict):
            ent = tau_by_level.get(L, tau_by_level.get(str(L), None))
            if isinstance(ent, dict):
                return (float(ent.get('low', tau_low_def)), float(ent.get('high', tau_high_def)))
            if ent is not None:
                return (tau_low_def, float(ent))
        return (tau_low_def, tau_high_def)

    def _first_present(src: Dict[str, Any], keys: list[str]):
        if not isinstance(src, dict):
            return None
        for k in keys:
            if k in src:
                return src.get(k)
        return None

    def _resolve_dilate_cells_for_level(L: int) -> float | None:
        raw_cells = _first_present(gf_cfg, [f'dilate_cells_L{L}', f'dilate_cells_l{L}', f'dilate_cells_level{L}'])
        if raw_cells is None:
            raw_cells = _first_present(pol, [f'dilate_cells_L{L}', f'dilate_cells_l{L}', f'dilate_cells_level{L}'])
        if raw_cells is None and L == Lmax:
            raw_cells = _first_present(gf_cfg, ['dilate_cells_lmax', 'dilate_cells_Lmax', 'dilate_cells_max'])
        if raw_cells is None and L == Lmax:
            raw_cells = _first_present(pol, ['dilate_cells_lmax', 'dilate_cells_Lmax', 'dilate_cells_max'])
        if raw_cells is None:
            return None
        try:
            return float(raw_cells)
        except Exception:
            return None

    def _resolve_dilate_phys_for_level(L: int) -> float:
        raw_phys = _first_present(gf_cfg, [f'dilate_phys_L{L}', f'dilate_phys_l{L}', f'dilate_phys_level{L}'])
        if raw_phys is None:
            raw_phys = _first_present(pol, [f'dilate_phys_L{L}', f'dilate_phys_l{L}', f'dilate_phys_level{L}'])
        try:
            return float(raw_phys) if raw_phys is not None else 0.0
        except Exception:
            return 0.0
    raw_lmax_cells = _first_present(gf_cfg, ['dilate_cells_lmax', 'dilate_cells_Lmax', 'dilate_cells_max'])
    if raw_lmax_cells is None:
        raw_lmax_cells = _first_present(pol, ['dilate_cells_lmax', 'dilate_cells_Lmax', 'dilate_cells_max'])
    try:
        lmax_cells_val = float(raw_lmax_cells) if raw_lmax_cells is not None else None
    except Exception:
        lmax_cells_val = None
    has_lmax_dilation_cfg = bool(lmax_cells_val is not None and lmax_cells_val > 0.0)
    propagate_lmax_to_parents = bool(gf_cfg.get('dilate_cells_lmax_propagate_to_parents', pol.get('dilate_cells_lmax_propagate_to_parents', True)))
    t_coarse_t0 = time.perf_counter()
    coarse = coarse_aggregate_from_dynamic(feat_policy, parents_t, H, W).to(dev, dtype=torch.float32)
    timing['predict_coarse_agg_s'] = float(time.perf_counter() - t_coarse_t0)
    coarse_hwf = coarse.view(H, W, -1).contiguous()
    if bool(gf_cfg.get('fill_empty_interior', True)) and physical_parent_mask is not None:
        t_fill_t0 = time.perf_counter()
        fill_chunk = int(gf_cfg.get('fill_empty_chunk', 1024))
        if fill_chunk < 1:
            fill_chunk = 1
        coarse_chw = coarse_hwf.permute(2, 0, 1).contiguous()
        coarse_chw = _runtime_fill_empty_interior_coarse_nearest(coarse_chw=coarse_chw, parents=parents_t, H=H, W=W, physical_parent_mask=torch.as_tensor(physical_parent_mask, dtype=torch.bool, device=dev), rc_grid=coarse_rc_grid, chunk=fill_chunk)
        coarse_hwf = coarse_chw.permute(1, 2, 0).contiguous()
        timing['predict_fill_empty_s'] = float(time.perf_counter() - t_fill_t0)
    coarse = coarse_hwf.view(H * W, -1).contiguous()
    t_gradscore_t0 = time.perf_counter()
    Fdim = int(coarse.shape[1])
    idxs = [int(c) for c in _resolve_channel_indices(cfg) if 0 <= int(c) < Fdim]
    if len(idxs) == 0:
        idxs = list(range(Fdim))
    if len(idxs) == 0:
        idxs = [0]
    score_list: List[torch.Tensor] = []
    for c in idxs:
        score_list.append(_coarse_grad_mag(coarse[:, c], H, W, dx, dy, p=p_pow))
    S = torch.stack(score_list, dim=0)
    if weights is not None:
        w = torch.as_tensor(weights, device=S.device, dtype=S.dtype)
        if w.numel() != S.size(0):
            if w.numel() < S.size(0):
                w = torch.cat([w, w.new_ones(S.size(0) - w.numel())])
            else:
                w = w[:S.size(0)]
    else:
        w = torch.ones(S.size(0), device=S.device, dtype=S.dtype)
    if combine == 'max':
        s_ref = S.max(dim=0).values
    elif combine in ('sum', 'weighted_sum'):
        s_ref = (w.view(-1, 1) * S).sum(dim=0)
    else:
        s_ref = torch.sqrt(((w.view(-1, 1) * S) ** 2).sum(dim=0))
    g_base = torch.nan_to_num(s_ref.view(H, W), nan=0.0, posinf=0.0, neginf=0.0).to(dev)
    timing['predict_grad_score_s'] = float(time.perf_counter() - t_gradscore_t0)
    xmin, xmax, ymin, ymax = _get_bbox(cfg)
    masks_by_level: Dict[int, torch.Tensor] = {}
    t_prev_t0 = time.perf_counter()
    prev_refine_by_level: Dict[int, torch.Tensor] = {}
    if bool(use_prev_state_masks):
        try:
            lv_prev = level_t.to(dev, dtype=torch.long).view(-1)
            centers_prev = centers_t.to(dev, dtype=torch.float32)
            if lv_prev.numel() == int(centers_prev.shape[0]) and lv_prev.numel() > 0:
                row_prev, col_prev = _cell_level_ij_from_centers(centers=centers_prev, levels=lv_prev, H=H, W=W, bbox=(xmin, xmax, ymin, ymax), refine_ratio=rr)
                for L in range(1, Lmax + 1):
                    h_p = int(H * rr ** (L - 1))
                    w_p = int(W * rr ** (L - 1))
                    prev_mask_L = torch.zeros((h_p, w_p), dtype=torch.bool, device=dev)
                    sel = lv_prev >= int(L)
                    if bool(sel.any()):
                        exp = (lv_prev[sel] - int(L - 1)).to(torch.long)
                        max_e = int(torch.clamp(exp.max(), min=0).item())
                        pow_tbl = torch.tensor([rr ** i for i in range(max_e + 1)], device=dev, dtype=torch.long)
                        scale = pow_tbl.index_select(0, exp.clamp(0, max_e))
                        prow = torch.div(row_prev[sel], scale, rounding_mode='floor')
                        pcol = torch.div(col_prev[sel], scale, rounding_mode='floor')
                        prow = prow.clamp_(0, h_p - 1)
                        pcol = pcol.clamp_(0, w_p - 1)
                        parent_flat = (prow * int(w_p) + pcol).view(-1).long()
                        if parent_flat.numel() > 0:
                            prev_mask_L.view(-1)[parent_flat] = True
                    prev_refine_by_level[L] = prev_mask_L
        except Exception:
            prev_refine_by_level = {}
    timing['predict_prev_masks_s'] = float(time.perf_counter() - t_prev_t0)
    t_hier_t0 = time.perf_counter()
    dilation_s = 0.0
    for L in range(1, Lmax + 1):
        h_p = H * rr ** (L - 1)
        w_p = W * rr ** (L - 1)
        if (h_p, w_p) == (H, W):
            g_parent = g_base
        else:
            g_parent = F.interpolate(g_base[None, None], size=(h_p, w_p), mode='bilinear', align_corners=False)[0, 0]
        g_parent = torch.nan_to_num(g_parent, nan=0.0, posinf=0.0, neginf=0.0)
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
            finite = torch.isfinite(g_parent)
            if finite.any():
                thr_low = torch.quantile(g_parent[finite], p_low / 100.0)
                thr_high = torch.quantile(g_parent[finite], p_high / 100.0)
            else:
                thr_low = torch.as_tensor(float('inf'), device=dev, dtype=g_parent.dtype)
                thr_high = torch.as_tensor(float('inf'), device=dev, dtype=g_parent.dtype)
        else:
            tau_low, tau_high = _taus(L)
            thr_low = torch.as_tensor(tau_low, device=dev, dtype=g_parent.dtype)
            thr_high = torch.as_tensor(tau_high, device=dev, dtype=g_parent.dtype)
        prev_mask = prev_refine_by_level.get(L, None)
        if prev_mask is None and L == 1:
            prev2d = mask_t_parent.to(dev, dtype=torch.float32).view(H, W)
            prev_mask = prev2d.bool()
        if prev_mask is not None:
            if prev_mask.shape != (h_p, w_p):
                prev_mask = F.interpolate(prev_mask.float()[None, None], size=(h_p, w_p), mode='nearest')[0, 0].bool()
            keep = prev_mask & (g_parent > thr_low)
            newr = ~prev_mask & (g_parent > thr_high)
            M = keep | newr
        else:
            M = g_parent > thr_high
        if L > 1:
            allow = F.interpolate(masks_by_level[L - 1].float()[None, None], scale_factor=float(rr), mode='nearest')[0, 0].bool()
            M = M & allow[:h_p, :w_p]
        dxL_here = (xmax - xmin) / float(W * rr ** (L - 1))
        n_cells = _resolve_dilate_cells_for_level(L)
        if n_cells is not None:
            r_phys = float(max(0.0, n_cells) * dxL_here)
        else:
            r_phys = _resolve_dilate_phys_for_level(L)
        if r_phys > 0:
            t_dilate_t0 = time.perf_counter()
            r_cells = max(1, int(round(r_phys / max(dxL_here, 1e-12))))
            k = 2 * r_cells + 1
            M = F.max_pool2d(M.float()[None, None], kernel_size=k, stride=1, padding=r_cells)[0, 0].bool()
            dilation_s += float(time.perf_counter() - t_dilate_t0)
        masks_by_level[L] = M
    thr_total = float(time.perf_counter() - t_hier_t0)
    if has_lmax_dilation_cfg and propagate_lmax_to_parents:
        t_pool_t0 = time.perf_counter()
        for L in range(int(Lmax), 1, -1):
            child = masks_by_level.get(L, None)
            parent = masks_by_level.get(L - 1, None)
            if child is None:
                continue
            child = child.to(dev, dtype=torch.bool)
            req_parent = F.max_pool2d(child.float()[None, None], kernel_size=int(rr), stride=int(rr))[0, 0].bool()
            if parent is None:
                parent = torch.zeros_like(req_parent, dtype=torch.bool, device=dev)
            else:
                parent = parent.to(dev, dtype=torch.bool)
                if tuple(parent.shape) != tuple(req_parent.shape):
                    parent = F.interpolate(parent.float()[None, None], size=tuple(req_parent.shape), mode='nearest')[0, 0].bool()
            masks_by_level[L - 1] = parent | req_parent
        timing['predict_pool_up_s'] = float(time.perf_counter() - t_pool_t0)
    timing['predict_hierarchy_s'] = thr_total
    timing['predict_dilation_s'] = float(dilation_s)
    timing['predict_threshold_hysteresis_s'] = max(0.0, thr_total - float(dilation_s))
    if isinstance(timing_out, dict):
        timing_out.clear()
        timing_out.update(timing)
    if isinstance(debug_out, dict):
        debug_out.clear()
        debug_out['backend'] = 'gradient_fast'
        debug_out['selected_idx'] = [int(i) for i in idxs]
        debug_out['coarse_hwf'] = coarse_hwf.detach().to('cpu', dtype=torch.float32).clone()
        debug_out['g_base_hxw'] = g_base.detach().to('cpu', dtype=torch.float32).clone()
    return {L: masks_by_level[L].to(device=device, dtype=torch.bool) for L in masks_by_level.keys()}

def _sanitize_float_tensor(t: torch.Tensor | None, *, clip_abs: float=0.0, fill_value: float=0.0, nonneg: bool=False) -> torch.Tensor | None:
    """
    Make floating tensors safe for downstream ops:
      - replace NaN/Inf
      - optional magnitude clipping
    """
    if t is None or not torch.is_tensor(t):
        return t
    if not t.dtype.is_floating_point:
        return t
    c = float(clip_abs)
    if c > 0.0:
        t = torch.nan_to_num(t, nan=fill_value, posinf=c, neginf=-c)
        if nonneg:
            t = t.clamp(min=0.0, max=c)
        else:
            t = t.clamp(min=-c, max=c)
    else:
        t = torch.nan_to_num(t, nan=fill_value, posinf=fill_value, neginf=fill_value)
        if nonneg:
            t = t.clamp_min(0.0)
    return t

def _sanitize_ops_term(t: torch.Tensor | None, cfg: dict) -> torch.Tensor | None:
    """
    Guard DEC/MLS operator terms against catastrophic values.
    """
    loss = cfg.get('loss', {}) or {}
    clip_abs = float(loss.get('ops_term_clip', 0.0))
    return _sanitize_float_tensor(t, clip_abs=clip_abs, fill_value=0.0, nonneg=False)

def _sanitize_parc_extra_tensor(t: torch.Tensor | None, cfg: dict, *, post_adapter: bool=False) -> torch.Tensor | None:
    """
    Keep PARC feature block bounded even when adapter is disabled.
    """
    loss = cfg.get('loss', {}) or {}
    if post_adapter:
        clip_abs = float(loss.get('parc_feat_clip_post', loss.get('parc_feat_clip_pre', 0.0)))
    else:
        clip_abs = float(loss.get('parc_feat_clip_pre', 0.0))
    return _sanitize_float_tensor(t, clip_abs=clip_abs, fill_value=0.0, nonneg=False)

def _apply_rate_guardrails(y_rate: torch.Tensor, dt_hat: torch.Tensor | float, cfg: dict) -> torch.Tensor:
    """
    Optional clipping in model-rate space to limit rollout explosions.
    """
    train_cfg = cfg.get('train', {}) or {}
    rate_clip = float(train_cfg.get('rate_clip_norm', 0.0))
    delta_clip = float(train_cfg.get('delta_clip_norm', 0.0))
    y = y_rate
    if rate_clip > 0.0:
        y = y.clamp(min=-rate_clip, max=rate_clip)
    if delta_clip > 0.0:
        if torch.is_tensor(dt_hat):
            dt = dt_hat.to(device=y.device, dtype=y.dtype)
        else:
            dt = torch.tensor(float(dt_hat), device=y.device, dtype=y.dtype)
        dt_safe = dt.abs().clamp_min(1e-12)
        delta = (y * dt).clamp(min=-delta_clip, max=delta_clip)
        y = delta / dt_safe
    return y

def sanitize_state_for_ops(x_abs: torch.Tensor, cfg: dict, rho_floor=1e-06, E_floor=1e-06):
    loss = cfg.get('loss', {}) or {}
    u_clip = float(loss.get('u_clip', 1000.0))
    rho_max = float(loss.get('ops_rho_max', loss.get('state_rho_max', 0.0)))
    E_max = float(loss.get('ops_E_max', loss.get('state_E_max', 0.0)))
    m_max = float(loss.get('ops_m_max', loss.get('state_m_max', 0.0)))
    idx = dec.infer_feature_indices(cfg, x_abs.size(1))
    rho = x_abs[:, idx['rho']].clamp_min(rho_floor)
    E = x_abs[:, idx['E']].clamp_min(E_floor)
    if rho_max > 0.0:
        rho = rho.clamp_max(rho_max)
    if E_max > 0.0:
        E = E.clamp_max(E_max)
    mx = x_abs[:, idx['mx']]
    my = x_abs[:, idx['my']]
    if u_clip > 0:
        mx = mx.clamp(-u_clip * rho, u_clip * rho)
        my = my.clamp(-u_clip * rho, u_clip * rho)
    if m_max > 0.0:
        mx = mx.clamp(min=-m_max, max=m_max)
        my = my.clamp(min=-m_max, max=m_max)
    rho = _sanitize_float_tensor(rho, clip_abs=rho_max, fill_value=rho_floor, nonneg=True)
    E = _sanitize_float_tensor(E, clip_abs=E_max, fill_value=E_floor, nonneg=True)
    mx = _sanitize_float_tensor(mx, clip_abs=m_max, fill_value=0.0, nonneg=False)
    my = _sanitize_float_tensor(my, clip_abs=m_max, fill_value=0.0, nonneg=False)
    cols = [x_abs[:, j] for j in range(x_abs.size(1))]
    cols[idx['rho']] = rho
    cols[idx['E']] = E
    cols[idx['mx']] = mx
    cols[idx['my']] = my
    return torch.stack(cols, dim=1)

def _enforce_physical_state(x_abs: torch.Tensor, cfg: dict, rho_floor: float=1e-06, E_floor: float=1e-06):
    """
    Clamp rho and E using the same feature-index mapping as DEC/PARC.
    This avoids assuming channel order is [rho, mx, my, E].
    """
    if x_abs is None:
        return x_abs
    if x_abs.ndim != 2:
        raise ValueError(f'_enforce_physical_state expects [N,F], got {tuple(x_abs.shape)}')
    Fdim = int(x_abs.size(1))
    idx = dec.infer_feature_indices(cfg, Fdim)
    loss = cfg.get('loss', {}) or {}
    u_clip = float(loss.get('u_clip', 1000.0))
    rho_max = float(loss.get('state_rho_max', 0.0))
    E_max = float(loss.get('state_E_max', 0.0))
    m_max = float(loss.get('state_m_max', 0.0))
    cols = [x_abs[:, j] for j in range(Fdim)]
    rho = cols[idx['rho']].clamp_min(rho_floor)
    E = cols[idx['E']].clamp_min(E_floor)
    if rho_max > 0.0:
        rho = rho.clamp_max(rho_max)
    if E_max > 0.0:
        E = E.clamp_max(E_max)
    mx = cols[idx['mx']]
    my = cols[idx['my']]
    if u_clip > 0.0:
        mx = mx.clamp(min=-u_clip * rho, max=u_clip * rho)
        my = my.clamp(min=-u_clip * rho, max=u_clip * rho)
    if m_max > 0.0:
        mx = mx.clamp(min=-m_max, max=m_max)
        my = my.clamp(min=-m_max, max=m_max)
    cols[idx['rho']] = _sanitize_float_tensor(rho, clip_abs=rho_max, fill_value=rho_floor, nonneg=True)
    cols[idx['E']] = _sanitize_float_tensor(E, clip_abs=E_max, fill_value=E_floor, nonneg=True)
    cols[idx['mx']] = _sanitize_float_tensor(mx, clip_abs=m_max, fill_value=0.0, nonneg=False)
    cols[idx['my']] = _sanitize_float_tensor(my, clip_abs=m_max, fill_value=0.0, nonneg=False)
    return torch.stack(cols, dim=1)
