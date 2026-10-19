# Focused numerical kernels retained from Physics-Aware; see PORTING.md.
from __future__ import annotations

import numpy as np
import torch

def amr_composite_to_finest_grid(centers: torch.Tensor, levels: torch.Tensor, values: torch.Tensor, H: int, W: int, bbox=(0.0, 1.0, 0.0, 1.0), refine_ratio: int=2):
    """
    Composite an AMR field onto the *finest* implied grid
    (H*refine_ratio^Lmax, W*refine_ratio^Lmax),
    with 'fine overwriting coarse'.

    Returns:
      img_flat:   (HH*WW, F)   values on finest grid
      valid_flat: (HH*WW,)     bool mask: True where at least one AMR cell wrote
      HH, WW
    """
    rr = int(refine_ratio)
    if rr < 2:
        raise ValueError(f'refine_ratio must be >=2, got {refine_ratio}')
    if isinstance(centers, np.ndarray):
        centers = torch.from_numpy(centers)
    if isinstance(levels, np.ndarray):
        levels = torch.from_numpy(levels)
    if isinstance(values, np.ndarray):
        values = torch.from_numpy(values)
    centers = centers.to(torch.float32)
    levels = levels.to(torch.int64).view(-1)
    vals = values if values.dim() == 2 else values.view(-1, 1)
    xmin, xmax, ymin, ymax = bbox
    dx = (xmax - xmin) / float(W)
    dy = (ymax - ymin) / float(H)
    Lmax = int(levels.max().item()) if levels.numel() > 0 else 0
    scale = rr ** Lmax
    HH, WW = (H * scale, W * scale)
    N = centers.size(0)
    F = vals.size(1)
    x_units = (centers[:, 0] - xmin) / dx
    y_units = (centers[:, 1] - ymin) / dy
    colf = torch.floor(x_units * scale).clamp(0, WW - 1).to(torch.int64)
    rowf = torch.floor(y_units * scale).clamp(0, HH - 1).to(torch.int64)
    order = torch.argsort(levels)
    rowf = rowf[order]
    colf = colf[order]
    levO = levels[order]
    valO = vals[order]
    img = torch.zeros((HH, WW, F), dtype=vals.dtype, device=vals.device)
    valid = torch.zeros((HH, WW), dtype=torch.bool, device=vals.device)
    for i in range(N):
        l = int(levO[i])
        block = rr ** (Lmax - l)
        r = int(rowf[i])
        c = int(colf[i])
        r0 = r // block * block
        c0 = c // block * block
        r1 = min(r0 + block, HH)
        c1 = min(c0 + block, WW)
        img[r0:r1, c0:c1, :] = valO[i].view(1, 1, F)
        valid[r0:r1, c0:c1] = True
    return (img.view(HH * WW, F), valid.view(HH * WW), HH, WW)
