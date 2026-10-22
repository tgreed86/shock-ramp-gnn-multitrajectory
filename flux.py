"""Finite-volume geometry and conservation-preserving FluxGraphNet guardrails."""
from dataclasses import dataclass

import torch

from mesh_geometry import amr_cell_wh_area_from_levels


BOUNDARY_SIDES = ('left', 'right', 'bottom', 'top')


@torch.no_grad()
def load_boundary_segments(spec_path, bbox):
    """Rescale labeled physical sides exactly like the runtime wedge polygon.

    Real mesh specs have curved left/right/top sides, so testing distance to
    the rectangular bbox would incorrectly classify most open faces as walls.
    Minimal polygon-only specs use each segment's dominant outward normal.
    """
    spec = torch.load(spec_path, map_location='cpu', weights_only=False)
    outer = torch.as_tensor(spec['outer_polygon_ccw'], dtype=torch.float32)
    lo, hi = outer.amin(0), outer.amax(0)
    x0, x1, y0, y1 = bbox
    offset, span = outer.new_tensor([x0, y0]), outer.new_tensor([x1-x0, y1-y0])
    starts, ends, sides = [], [], []
    if all(side in spec for side in BOUNDARY_SIDES):
        for side, name in enumerate(BOUNDARY_SIDES):
            points = (torch.as_tensor(spec[name], dtype=outer.dtype) - lo) / (hi-lo) * span + offset
            starts.append(points[:-1])
            ends.append(points[1:])
            sides.append(torch.full((len(points)-1,), side, dtype=torch.long))
    else:
        points = (outer - lo) / (hi-lo) * span + offset
        if not torch.equal(points[0], points[-1]):
            points = torch.cat([points, points[:1]])
        delta = points[1:] - points[:-1]
        normal = torch.stack([delta[:, 1], -delta[:, 0]], dim=1)
        side = torch.where(normal[:, 0].abs() > normal[:, 1].abs(),
                           (normal[:, 0] > 0).long(), 2 + (normal[:, 1] > 0).long())
        starts, ends, sides = [points[:-1]], [points[1:]], [side]
    a, b, side = torch.cat(starts), torch.cat(ends), torch.cat(sides)
    keep = (a != b).any(1)
    return a[keep], b[keep], side[keep]


@torch.no_grad()
def nearest_boundary_side(points, segments):
    a, b, side = (tensor.to(device=points.device) for tensor in segments)
    a, b = a.to(points.dtype), b.to(points.dtype)
    vector = b - a
    squared_length = vector.square().sum(1)
    result = torch.empty(len(points), device=points.device, dtype=torch.long)
    for start in range(0, len(points), 1024):
        relative = points[start:start+1024, None] - a[None]
        t = ((relative * vector).sum(-1) / squared_length).clamp(0, 1)
        distance2 = (relative - t[:, :, None]*vector).square().sum(-1)
        result[start:start+1024] = side[distance2.argmin(1)]
    return result


@dataclass
class FluxGeometry:
    area: torch.Tensor
    forward_edges: torch.Tensor
    reverse_edges: torch.Tensor
    face_length: torch.Tensor
    boundary_owner: torch.Tensor
    boundary_normal: torch.Tensor
    boundary_length: torch.Tensor
    boundary_open: torch.Tensor


@torch.no_grad()
def build_flux_geometry(centers, levels, edges, edge_attr, *, dx0, dy0, bbox,
                        refine_ratio=2, open_boundary_sides=('left', 'right', 'top'), boundary_segments=None):
    """Pair directed faces and find uncovered portions of each cell side.

    The runtime supplies complete, bidirected face adjacency. Subtracting every
    shared subface length handles coarse-fine interfaces without mistaking low
    node degree for an exterior boundary. Remaining faces on configured bbox
    sides are open; the bottom/ramp staircase and other closed sides are walls.
    With mesh-spec segments, nearest physical side replaces bbox classification.
    """
    n = len(centers)
    w, h, area, hx, hy = amr_cell_wh_area_from_levels(
        levels, dx0=dx0, dy0=dy0, refine_ratio=refine_ratio)
    src, dst = edges
    keys, order = torch.sort(src * n + dst)
    if (torch.any(src == dst) or torch.any(keys[1:] == keys[:-1])
            or edge_attr.shape != (edges.shape[1], 5)):
        raise ValueError('FluxGraphNet requires unique, non-self, bidirected face edges with [E,5] geometry')
    forward = ((edge_attr[:, 0] > 0) | (edge_attr[:, 1] > 0)).nonzero(as_tuple=True)[0]
    reverse_keys = dst[forward] * n + src[forward]
    pos = torch.searchsorted(keys, reverse_keys)
    if len(forward) * 2 != len(src) or torch.any(pos >= len(keys)):
        raise ValueError('FluxGraphNet requires a reverse edge for every interior face')
    reverse = order[pos]
    if (not torch.equal(keys[pos], reverse_keys)
            or not torch.allclose(edge_attr[forward, :2], -edge_attr[reverse, :2])
            or not torch.allclose(edge_attr[forward, 2], edge_attr[reverse, 2])):
        raise ValueError('FluxGraphNet reverse face geometry is inconsistent')

    # Side order: left, right, bottom, top; normals point out of each cell.
    side = torch.where(edge_attr[:, 0] != 0, (edge_attr[:, 0] > 0).long(),
                       2 + (edge_attr[:, 1] > 0).long())
    covered = centers.new_zeros(n * 4)
    covered.index_add_(0, src * 4 + side, edge_attr[:, 2].to(centers))
    full_length = torch.stack([h, h, w, w], dim=1).to(centers)
    exposed = full_length - covered.view(n, 4)
    tolerance = full_length * 1e-5
    if torch.any(exposed < -tolerance):
        raise ValueError('Interior faces cover more than a complete cell side')
    owner, side = (exposed > tolerance).nonzero(as_tuple=True)
    normals = centers.new_tensor([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])
    x0, x1, y0, y1 = bbox
    distance = torch.stack([centers[:, 0] - hx - x0, centers[:, 0] + hx - x1,
                            centers[:, 1] - hy - y0, centers[:, 1] + hy - y1], dim=1)
    tol = 8 * torch.finfo(centers.dtype).eps * max(abs(v) for v in (*bbox, dx0, dy0))
    on_box = distance.abs() <= tol
    allowed = torch.tensor([name in open_boundary_sides for name in BOUNDARY_SIDES], device=centers.device)
    boundary_open = on_box[owner, side] & allowed[side]
    if boundary_segments is not None:
        half_size = torch.stack([hx, hy], dim=1).to(centers)
        midpoint = centers[owner] + normals[side] * half_size[owner]
        boundary_open = allowed[nearest_boundary_side(midpoint, boundary_segments)]
    return FluxGeometry(area, forward, reverse, edge_attr[forward, 2], owner,
                        normals[side], exposed[owner, side], boundary_open)


def limit_flux_rate(rate, state, sigma, dt_hat, cfg):
    """Scale the entire update by one scalar; never clip individual cells.

    The shared scale preserves cancellation of every internal transfer and
    scales exterior exchange by the same amount. Linear state constraints
    implement the existing rho/E floors, optional maxima, and velocity bounds.
    Detach the limiter coefficient so active bounds do not erase flux gradients.
    """
    train, loss = cfg['train'], cfg['loss']
    with torch.no_grad():
        alpha = rate.new_ones(())
        for key, change in [('rate_clip_norm', rate), ('delta_clip_norm', rate * dt_hat)]:
            bound = float(train.get(key, 0.))
            if bound > 0:
                alpha = torch.minimum(alpha, bound / change.abs().amax().clamp_min(torch.finfo(rate.dtype).tiny))
        delta = rate * dt_hat * sigma
        slack = [state[:, 0] - 1e-6, state[:, 3] - 1e-6]
        change = [delta[:, 0], delta[:, 3]]
        magnitude = [state[:, 0].abs() + 1e-6, state[:, 3].abs() + 1e-6]
        for key, channel in [('state_rho_max', 0), ('state_E_max', 3)]:
            bound = float(loss.get(key, 0.))
            if bound > 0:
                slack.append(bound - state[:, channel])
                change.append(-delta[:, channel])
                magnitude.append(bound + state[:, channel].abs())
        for channel in (1, 2):
            for sign in (-1, 1):
                bound = float(loss.get('state_m_max', 0.))
                if bound > 0:
                    slack.append(bound + sign * state[:, channel])
                    change.append(sign * delta[:, channel])
                    magnitude.append(bound + state[:, channel].abs())
                velocity = float(loss.get('u_clip', 1000.))
                if velocity > 0:
                    slack.append(velocity * state[:, 0] + sign * state[:, channel])
                    change.append(velocity * delta[:, 0] + sign * delta[:, channel])
                    magnitude.append(velocity * state[:, 0].abs() + state[:, channel].abs())
        slack, change = torch.stack(slack), torch.stack(change)
        roundoff = 16 * torch.finfo(state.dtype).eps * torch.stack(magnitude)
        if not torch.isfinite(rate).all() or not torch.isfinite(state).all():
            raise FloatingPointError('FluxGraphNet requires finite states and flux rates')
        if torch.any(slack < -roundoff):
            raise ValueError(f'FluxGraphNet input state violates physical bounds (minimum slack={slack.min().item():.8g}); '
                             'conservative limiting requires an admissible state')
        safe_slack = (slack - roundoff).clamp_min(0)
        admissible = torch.where(change < 0, safe_slack / (-change).clamp_min(torch.finfo(rate.dtype).tiny),
                                 torch.full_like(change, float('inf')))
        alpha = torch.minimum(alpha, admissible.amin()).clamp(0., 1.)
        # Stay just inside active constraints despite floating-point roundoff.
        alpha = torch.where(alpha < 1, alpha * (1 - 8 * torch.finfo(rate.dtype).eps), alpha)
    return rate * alpha


@torch.no_grad()
def leaf_owners(centers, levels, *, H, W, bbox, max_level, refine_ratio=2):
    """Map a fixed finest lattice to its containing dyadic leaf (-1 outside)."""
    x0, x1, y0, y1 = bbox
    owners = torch.full((H, W), -1, dtype=torch.long, device=centers.device)
    for level in range(max_level + 1):
        if level:
            owners = owners.repeat_interleave(refine_ratio, 0).repeat_interleave(refine_ratio, 1)
        ids = (levels == level).nonzero(as_tuple=True)[0]
        col = ((centers[ids, 0] - x0) / ((x1-x0) / owners.shape[1])).floor().long()
        row = ((centers[ids, 1] - y0) / ((y1-y0) / owners.shape[0])).floor().long()
        owners[row, col] = ids
    return owners.reshape(-1)


def conservative_remap(features, source_owners, target_owners, target_count):
    """Piecewise-constant prolongation and area-weighted restriction.

    Each finest tile has equal area. The domain must be identical on both
    meshes, as ensured by the runtime's forced refinement at the wedge wall.
    """
    inside = source_owners >= 0
    if not torch.equal(inside, target_owners >= 0):
        raise ValueError('Conservative remapping requires identical source and target domains')
    src, dst = source_owners[inside], target_owners[inside]
    # Aggregate overlap weights first to avoid storing a differentiable field
    # at every finest tile in every step of an autoregressive window.
    pairs, counts = torch.unique(src * target_count + dst, return_counts=True)
    src, dst = pairs // target_count, pairs % target_count
    weights = counts.to(features.dtype)
    total = features.new_zeros(target_count)
    total.index_add_(0, dst, weights)
    if torch.any(total == 0):
        raise ValueError('Conservative remapping found an uncovered target cell')
    out = features.new_zeros((target_count, features.shape[1]))
    out.index_add_(0, dst, features[src] * (weights / total[dst])[:, None])
    return out
