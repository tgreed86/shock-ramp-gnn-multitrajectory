#!/usr/bin/env python3
"""One-time physical timestamp export; cache-only training never runs this tool."""
import argparse
import copy
import json
from pathlib import Path

import h5py
import torch

from data import discover, uniform_cache_files
from train import load_config


def export_times(cfg, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite timestamp metadata: {output}')
    cfg = copy.deepcopy(cfg)
    # Export all available trajectories so later exclusions remain configurable.
    cfg['data']['excluded_trajectories'] = []
    root = Path(cfg['data']['root']).resolve()
    cache_root = Path(cfg['train']['runtime_mesh']['multires_gt_lookup']['root'])
    records = {}
    for path in discover(cfg):
        series = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        times = [float(s.get('time', s.get('t'))) for s in series]
        values = torch.tensor(times, dtype=torch.float64)
        if len(times) < 2 or not torch.isfinite(values).all() or not (values.diff() > 0).all():
            raise ValueError(f'Invalid physical timestamps: {path}')
        size = path.stat().st_size
        relative = str(path.relative_to(root))
        for cache in uniform_cache_files(cache_root, path.stem, cfg['policy']['max_level']).values():
            with h5py.File(cache, 'r') as f:
                provenance = json.loads(f['meta'].attrs['uniform_gt_provenance'])
                if (int(f['meta'].attrs['T']) != len(times) or provenance['source_size'] != size
                        or Path(provenance['source_path']).stem != path.stem
                        or Path(*Path(provenance['source_path']).parts[-3:]) != Path(relative)):
                    raise ValueError(f'Timestamp source does not match cache provenance: {cache}')
        records[path.stem] = {'source_relative_path': relative, 'source_size': size, 'times': times}
        print(f'[TIMES] {path.stem}: {len(times)} physical timestamps', flush=True)
        del series
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing an existing dataset description.
    with output.open('x') as stream:
        json.dump({'schema': 1, 'trajectories': records}, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')),
                        help='Native-data config used only for this one-time export')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(f'[TIMES] Saved {export_times(load_config(args.config), args.output)}')
