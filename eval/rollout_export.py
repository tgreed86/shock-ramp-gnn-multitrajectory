"""Lossless rollout fields and aligned, masked grids for offline error metrics."""
import json

import h5py
import numpy as np

from eval.rollout_geometry import array, rectangular_quads


FEATURE_NAMES = ['density', 'x_momentum', 'y_momentum', 'energy']


class ComparisonGrid:
    """Sample containing-cell values at fixed finest-grid centers, without IDW."""

    def __init__(self, cfg):
        scale = cfg['policy']['refine_ratio'] ** cfg['policy']['max_level']
        self.height, self.width = cfg['data']['H']*scale, cfg['data']['W']*scale
        self.bbox = np.asarray(cfg['data']['bbox'], dtype=float)
        x0, x1, y0, y1 = self.bbox
        self.dx, self.dy = (x1-x0)/self.width, (y1-y0)/self.height
        self.x = x0 + (np.arange(self.width)+.5)*self.dx
        self.y = y0 + (np.arange(self.height)+.5)*self.dy

    def owners(self, vertices, levels):
        """Rasterize quadrilaterals by point containment, including curved-map cells.

        Candidate points come from each cell's bounding box. Ray crossings work
        for convex or concave simple quads. Shared-edge ties favor finer levels,
        then the later original cell index; outside-domain points remain -1.
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        levels = np.asarray(levels).reshape(-1)
        if vertices.shape != (len(levels), 4, 2) or not np.isfinite(vertices).all():
            raise ValueError('Invalid cell vertices for comparison-grid sampling')
        order = np.argsort(levels, kind='stable')
        verts = vertices[order]
        low, high = verts.min(axis=1), verts.max(axis=1)
        origin = self.bbox[[0, 2]]
        spacing = np.array([self.dx, self.dy])
        shape = np.array([self.width, self.height])
        lo = np.maximum(np.ceil((low-origin)/spacing-.5-1e-9).astype(np.int64), 0)
        hi = np.minimum(np.floor((high-origin)/spacing-.5+1e-9).astype(np.int64), shape-1)
        sizes = np.maximum(hi-lo+1, 0)
        counts = sizes.prod(axis=1)
        offsets = np.concatenate(([0], np.cumsum(counts)))
        owners = np.full(self.height*self.width, -1, dtype=np.int64)
        start = 0
        # Bound temporary candidate arrays independently of the number of cells.
        while start < len(verts):
            end = max(start+1, int(np.searchsorted(offsets, offsets[start]+500_000, side='right')-1))
            end = min(end, len(verts))
            n = int(offsets[end]-offsets[start])
            if n:
                ranks = np.repeat(np.arange(start, end), counts[start:end])
                local = np.arange(n) - np.repeat(offsets[start:end]-offsets[start], counts[start:end])
                col = lo[ranks, 0] + local % sizes[ranks, 0]
                row = lo[ranks, 1] + local // sizes[ranks, 0]
                x, y = self.x[col], self.y[row]
                inside, boundary = np.zeros(n, bool), np.zeros(n, bool)
                tolerance = max(self.dx, self.dy)*1e-9
                for edge in range(4):
                    a, b = verts[ranks, edge], verts[ranks, (edge+1) % 4]
                    vx, vy = b[:, 0]-a[:, 0], b[:, 1]-a[:, 1]
                    crosses = (a[:, 1] > y) != (b[:, 1] > y)
                    cut = a[:, 0] + vx*(y-a[:, 1])/np.where(vy != 0, vy, 1.)
                    inside ^= crosses & (x < cut)
                    cross = vx*(y-a[:, 1])-vy*(x-a[:, 0])
                    boundary |= ((np.abs(cross) <= tolerance*np.hypot(vx, vy))
                                 & (x >= np.minimum(a[:, 0], b[:, 0])-tolerance)
                                 & (x <= np.maximum(a[:, 0], b[:, 0])+tolerance)
                                 & (y >= np.minimum(a[:, 1], b[:, 1])-tolerance)
                                 & (y <= np.maximum(a[:, 1], b[:, 1])+tolerance))
                keep = inside | boundary
                np.maximum.at(owners, row[keep]*self.width+col[keep], ranks[keep])
            start = end
        valid = owners >= 0
        owners[valid] = order[owners[valid]]
        return owners.reshape(self.height, self.width)

    def sample(self, owners, values):
        valid = owners >= 0
        image = np.full((self.height, self.width, 4), np.nan, dtype=np.float32)
        image[valid] = values[owners[valid]]
        return image, valid


def compressed(group, name, data):
    return group.create_dataset(name, data=data, compression='gzip', compression_opts=4, shuffle=True)


class RolloutExport:
    """Stream all timesteps to HDF5; retain raw cells as well as comparison grids."""

    def __init__(self, path, cfg, source, uniform_centers, uniform_levels, mu, sigma, dt_ref, start, steps):
        self.cfg, self.source, self.start, self.steps = cfg, source, start, steps
        self.grid = ComparisonGrid(cfg)
        self.uniform_owners = self.grid.owners(rectangular_quads(uniform_centers, uniform_levels, cfg),
                                               array(uniform_levels))
        self.file = h5py.File(path, 'w')
        self.count = 0
        self.ranges = {key: np.array([[np.inf, -np.inf]]*4) for key in ('uniform_gt', 'native_gt')}
        f = self.file
        f.attrs['format'] = 'shock-ramp-rollout-v1'
        f.attrs['complete'] = False
        f.attrs['values'] = 'Unnormalized conservative fields; predictions include training state guardrails'
        f.attrs['grid_sampling'] = 'Containing-cell value at finest-grid center; no interpolation or extrapolation'
        f.attrs['grid_axes'] = 'step, y, x, feature; y increases from bottom to top'
        f.attrs['mask_rule'] = 'Use prediction_valid & reference_valid; invalid pixels are NaN'
        f.attrs['ssim_note'] = ('Use a fixed per-feature data_range; exclude SSIM windows touching invalid pixels. '
                                'Reference ranges span all exported steps; zero range needs an explicit convention.')
        f.create_dataset('feature_names', data=FEATURE_NAMES, dtype=h5py.string_dtype('utf-8'))
        f['rollout_step'] = np.arange(steps+1)
        f['snapshot_index'] = np.arange(start, start+steps+1)
        times = array(source.times[start:start+steps+1])
        f['time'] = times
        f['elapsed_time'] = times-times[0]
        f['dt'] = np.concatenate(([0.], np.diff(times)))
        f['is_initial'] = np.arange(steps+1) == 0
        f['normalization/mu'] = array(mu)
        f['normalization/sigma'] = array(sigma)
        f['normalization/dt_ref'] = dt_ref
        f.create_dataset('config_json', data=json.dumps(cfg, allow_nan=False), dtype=h5py.string_dtype('utf-8'))
        compressed(f, 'uniform_mesh/centers', array(uniform_centers))
        compressed(f, 'uniform_mesh/levels', array(uniform_levels))
        f['grid/x'] = self.grid.x
        f['grid/y'] = self.grid.y
        f['grid/bbox'] = self.grid.bbox
        f['grid'].attrs['cell_area_model_coordinates'] = self.grid.dx*self.grid.dy
        chunks = (1, min(64, self.grid.height), min(64, self.grid.width))
        for name in ('prediction', 'uniform_gt', 'native_gt'):
            f.create_dataset(f'grid/{name}', shape=(steps+1, self.grid.height, self.grid.width, 4),
                             dtype='f4', chunks=(*chunks, 4), compression='gzip', compression_opts=4,
                             shuffle=True, fillvalue=np.nan)
            f.create_dataset(f'grid/{name}_valid', shape=(steps+1, self.grid.height, self.grid.width),
                             dtype='bool', chunks=chunks, compression='gzip', compression_opts=4)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                if self.count != self.steps+1:
                    raise ValueError('Incomplete rollout export')
                for name, bounds in self.ranges.items():
                    bounds[~np.isfinite(bounds)] = np.nan
                    self.file[f'reference_ranges/{name}/min'] = bounds[:, 0]
                    self.file[f'reference_ranges/{name}/max'] = bounds[:, 1]
                    self.file[f'reference_ranges/{name}/data_range'] = bounds[:, 1]-bounds[:, 0]
                self.file.attrs['complete'] = True
        finally:
            self.file.close()

    def append(self, index, centers, levels, predicted, uniform, native_centers, native_levels,
               original, native_vertices, native_snapshot):
        step = index-self.start
        if step != self.count:
            raise ValueError('Rollout export must include every step in order')
        group = self.file.create_group(f'snapshots/{index:06d}')
        group.attrs['rollout_step'] = step
        group.attrs['snapshot_index'] = index
        group.attrs['time'] = float(self.source.times[index])
        for name, values in {
                'prediction/centers': centers, 'prediction/levels': levels, 'prediction/values': predicted,
                'uniform_gt/values': uniform, 'native_gt/centers': native_centers,
                'native_gt/levels': native_levels, 'native_gt/values': original,
                'native_gt/vertices': native_vertices}.items():
            compressed(group, name, values)
        for name in ('bbox', 'bbox_phys', 'ij', 't'):
            if name in native_snapshot:
                group[f'native_gt/{name}'] = array(native_snapshot[name])
        owners = {
            'prediction': self.grid.owners(rectangular_quads(centers, levels, self.cfg), levels),
            'uniform_gt': self.uniform_owners,
            'native_gt': self.grid.owners(native_vertices, native_levels),
        }
        for name, values in zip(owners, (predicted, uniform, original)):
            image, valid = self.grid.sample(owners[name], values)
            self.file[f'grid/{name}'][step] = image
            self.file[f'grid/{name}_valid'][step] = valid
            if name in self.ranges and valid.any():
                self.ranges[name][:, 0] = np.minimum(self.ranges[name][:, 0], image[valid].min(axis=0))
                self.ranges[name][:, 1] = np.maximum(self.ranges[name][:, 1], image[valid].max(axis=0))
        self.count += 1
