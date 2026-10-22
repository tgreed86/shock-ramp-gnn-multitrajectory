"""Display geometry: dyadic model cells and mapped native AMR quadrilaterals."""
from pathlib import Path

import h5py
import numpy as np
import torch


CORNERS = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])


def array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def rectangular_quads(centers, levels, cfg):
    x0, x1, y0, y1 = cfg['data']['bbox']
    size = np.array([(x1-x0)/cfg['data']['W'], (y1-y0)/cfg['data']['H']])
    widths = size[None, :] / cfg['policy']['refine_ratio'] ** array(levels).astype(float)[:, None]
    return array(centers)[:, None, :] + widths[:, None, :] * CORNERS[None, :, :]


def native_map_path(directory, snapshot):
    iteration = int(snapshot['t'])
    return Path(directory) / f'DMR.plot.{iteration:06d}.2d.map.hdf5'


def mapped_quads(snapshot, path):
    """Read map nodes and select cells by the .pt's exact (level, i, j) ordering.

    No plotfile field values are used: the original-GT panels always use .pt x.
    Coordinates are transformed to the same normalized coordinates as .pt pos.
    """
    indices, levels = array(snapshot['ij']), array(snapshot['level']).reshape(-1)
    vertices = np.empty((len(levels), 4, 2), dtype=np.float64)
    found = np.zeros(len(levels), dtype=np.int8)
    with h5py.File(path, 'r') as f:
        for level in np.unique(levels):
            group = f[f'level_{level}']
            if not np.isclose(float(group.attrs['time']), float(snapshot['time']), rtol=1e-6, atol=1e-9):
                raise ValueError(f'Native map time does not match .pt snapshot: {path}')
            if int(group['data_attributes'].attrs['comps']) != 2:
                raise ValueError(f'Expected two coordinate components: {path}')
            boxes = group['boxes'][...]
            offsets = group['data:offsets=0'][...]
            data = group['data:datatype=0']
            if len(offsets) != len(boxes)+1 or offsets[0] != 0 or offsets[-1] != data.size:
                raise ValueError(f'Invalid native map box offsets: {path}')
            selected = np.flatnonzero(levels == level)
            ij = indices[selected]
            for b, box in enumerate(boxes):
                if boxes.dtype.names:
                    i0, j0, i1, j1 = (int(box[key]) for key in ('lo_i', 'lo_j', 'hi_i', 'hi_j'))
                else:
                    i0, j0, i1, j1 = map(int, box)
                ids = np.flatnonzero((ij[:, 0] >= i0) & (ij[:, 0] <= i1)
                                    & (ij[:, 1] >= j0) & (ij[:, 1] <= j1))
                if not len(ids):
                    continue
                nx, ny = i1-i0+2, j1-j0+2
                raw = data[int(offsets[b]):int(offsets[b+1])]
                if raw.size != 2*nx*ny:
                    raise ValueError(f'Unsupported native map ghost-cell layout: {path}')
                xy = raw.reshape(2, ny, nx).transpose(1, 2, 0)
                i, j = ij[ids, 0]-i0, ij[ids, 1]-j0
                vertices[selected[ids]] = np.stack([xy[j, i], xy[j, i+1], xy[j+1, i+1], xy[j+1, i]], axis=1)
                found[selected[ids]] += 1
    if not np.all(found == 1) or not np.isfinite(vertices).all():
        raise ValueError(f'Native map must match every .pt cell exactly once: {path}')
    if not np.allclose(vertices.mean(axis=1), array(snapshot['pos_phys'])[:, :2], rtol=1e-5, atol=5e-7):
        raise ValueError(f'Native map cell centers do not match .pt: {path}')
    physical = array(snapshot['bbox_phys'])
    normalized = array(snapshot['bbox'])
    origin, extent = physical[[0, 2]], physical[[1, 3]]-physical[[0, 2]]
    target_origin, target_extent = normalized[[0, 2]], normalized[[1, 3]]-normalized[[0, 2]]
    if np.any(extent <= 0) or np.any(target_extent <= 0):
        raise ValueError('Native coordinate bounds must have positive extent')
    vertices = (vertices-origin) / extent * target_extent + target_origin
    if not np.allclose(vertices.mean(axis=1), array(snapshot['pos'])[:, :2], rtol=1e-5, atol=3e-6):
        raise ValueError('Native pos and pos_phys normalization is inconsistent')
    return vertices


class NativeGeometry:
    def __init__(self, native, indices, *, map_dir=None):
        self.native = native
        self.map_dir = Path(map_dir) if map_dir is not None else native.path.parent/'plot'
        for index in indices:
            path = native_map_path(self.map_dir, native.series[index])
            if not path.is_file():
                raise FileNotFoundError(f'Exact native mesh requires {path}. Supply --native-map-dir '
                                        'with the simulation .2d.map.hdf5 files.')

    def quads(self, index):
        snapshot = self.native.series[index]
        return mapped_quads(snapshot, native_map_path(self.map_dir, snapshot))

    @property
    def label(self):
        return 'Original GT (native AMR)'
