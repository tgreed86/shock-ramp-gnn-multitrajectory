"""Byte-bounded GT feature caches and reusable CPU/CUDA cell lookup plans."""
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch

from kernels import _RuntimeMultiResGtLookup


class TensorLRU:
    """One budget shared by every trajectory; oversized tensors are not retained."""

    def __init__(self, max_bytes):
        self.max_bytes = int(max_bytes)
        self.bytes = 0
        self.entries = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key):
        value = self.entries.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
            self.entries.move_to_end(key)
        return value

    def put(self, key, value):
        old = self.entries.pop(key, None)
        if old is not None:
            self.bytes -= old.numel() * old.element_size()
        size = value.numel() * value.element_size()
        if size > self.max_bytes:
            return
        while self.bytes + size > self.max_bytes:
            _, old = self.entries.popitem(last=False)
            self.bytes -= old.numel() * old.element_size()
        self.entries[key] = value
        self.bytes += size

    def clear(self):
        self.entries.clear()
        self.bytes = 0


@dataclass
class LookupPlan:
    owner: object
    device: torch.device
    size: int
    groups: list
    missing: torch.Tensor


class MultiResGtLookup(_RuntimeMultiResGtLookup):
    """Reuse the validated static indices, but never round-trip CUDA queries."""

    def __init__(self, *, cpu_cache, device_cache, **kwargs):
        super().__init__(**kwargs)
        self.cpu_cache, self.device_cache = cpu_cache, device_cache
        # Stable across reader eviction, distinct across trajectories and cache families.
        self.namespace = tuple(self.level_files.items())
        self.plan_token = object()
        self.dense_tables = {}
        self.disk_reads = 0
        self.feature_dim = next(iter(self._h5_by_file_level.values()))['t00001/feat_t_on_pred'].shape[1]

    def close(self):
        super().close()
        self.dense_tables.clear()

    def _feature(self, file_level, local_t, device):
        key = (self.namespace, int(file_level), int(local_t))
        device_key = (str(device), key)
        if device.type == 'cuda':
            cached = self.device_cache.get(device_key)
            if cached is not None:
                return cached
        value = self.cpu_cache.get(key)
        if value is None:
            # Adjacent HDF5 groups duplicate physical timesteps. Read one canonical copy.
            group, dataset = ((1, 'feat_t_on_pred') if local_t == 0
                              else (local_t, 'feat_tp1_on_pred'))
            value = torch.from_numpy(np.asarray(
                self._h5_by_file_level[file_level][f't{group:05d}'][dataset][...], dtype=np.float32))
            self.disk_reads += 1
            self.cpu_cache.put(key, value)
        if device.type == 'cuda':
            value = value.to(device)
            self.device_cache.put(device_key, value)
        return value

    def _read_feature(self, *, file_level, t_dst, dataset_name):
        # Also make the reference CPU search path use canonical, byte-bounded features.
        if dataset_name not in ('feat_t_on_pred', 'feat_tp1_on_pred'):
            raise ValueError(f'Unsupported GT feature dataset: {dataset_name}')
        local_t = t_dst - (dataset_name == 'feat_t_on_pred')
        return self._feature(file_level, local_t, torch.device('cpu'))

    def _table(self, level, device):
        key = (level, str(device))
        if key not in self.dense_tables:
            file_level = self._source_file_for_level[level]
            index = self._file_level_index[file_level][level]
            width, height = self.W * self.refine_ratio**level, self.H * self.refine_ratio**level
            table = np.full(height * width, -1, dtype=np.int64)
            keys = index['keys_sorted']
            col = keys % self._key_w
            row = (keys // self._key_w) % self._key_h
            table[row * width + col] = index['idx_sorted']
            self.dense_tables[key] = torch.from_numpy(table).to(device)
        return self.dense_tables[key]

    @torch.no_grad()
    def prepare(self, centers, levels, *, device):
        """Geometry only: run once per mesh/reader, not once per timestep."""
        device = torch.device(device)
        centers, levels = centers.to(device), levels.view(-1).to(device=device, dtype=torch.long)
        matched = torch.zeros(len(levels), dtype=torch.bool, device=device)
        groups = []
        xmin, xmax, ymin, ymax = self.bbox
        for level, file_level in self._source_file_for_level.items():
            dst = (levels == level).nonzero(as_tuple=True)[0]
            if not dst.numel():
                continue
            width, height = self.W * self.refine_ratio**level, self.H * self.refine_ratio**level
            xy = centers[dst].float()
            # Same floor/clamp convention as the reference lookup, including edge cells.
            col = ((xy[:, 0] - xmin) * width / (xmax - xmin)).floor().long().clamp(0, width - 1)
            row = ((xy[:, 1] - ymin) * height / (ymax - ymin)).floor().long().clamp(0, height - 1)
            src = self._table(level, device)[row * width + col]
            valid = src >= 0
            dst, src = dst[valid], src[valid]
            if dst.numel():
                matched[dst] = True
                groups.append((file_level, dst, src))
        return LookupPlan(self.plan_token, device, len(levels), groups,
                          (~matched).nonzero(as_tuple=True)[0])

    @torch.no_grad()
    def gather(self, local_t, plan, *, out_device):
        if plan.owner is not self.plan_token:
            raise ValueError('Lookup plan belongs to a different cache reader')
        if local_t < 0:
            raise IndexError('GT local timestep must be nonnegative')
        out = torch.zeros((plan.size, self.feature_dim), dtype=torch.float32, device=plan.device)
        for file_level, dst, src in plan.groups:
            values = self._feature(file_level, local_t, plan.device)
            out.index_copy_(0, dst, values.index_select(0, src))
        return out.to(out_device)
