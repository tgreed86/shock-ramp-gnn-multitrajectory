#!/usr/bin/env python3
"""Train one GNN across included shock-ramp trajectories with runtime remeshing."""
import argparse
import json
import os
from pathlib import Path
import random
import warnings

import numpy as np
import torch

from data import load_trajectories, normalization, split_manifest, split_windows
from engine import build_model, run_epoch
from runtime import Runtime


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    for section, key in [('data', 'root'), ('mesh', 'starting_mesh_path'), ('train', 'save_dir')]:
        value = Path(os.path.expandvars(cfg[section][key])).expanduser()
        cfg[section][key] = str(value.resolve() if value.is_absolute() else (path.parent/value).resolve())
    lookup = cfg['train']['runtime_mesh']['multires_gt_lookup']
    value = Path(os.path.expandvars(lookup['root'])).expanduser()
    lookup['root'] = str(value.resolve() if value.is_absolute() else (path.parent/value).resolve())
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    progress_enabled = cfg['debug'].get('progress_bar', True)
    progress_interval = cfg['debug'].get('progress_bar_mininterval', 5.0)
    if not isinstance(progress_enabled, bool):
        raise ValueError('debug.progress_bar must be true or false')
    if (isinstance(progress_interval, bool) or not isinstance(progress_interval, (int, float))
            or not np.isfinite(progress_interval) or progress_interval < 0):
        raise ValueError('debug.progress_bar_mininterval must be a finite nonnegative number of seconds')
    rt = cfg['train']['runtime_mesh']
    rt.setdefault('adjacency_backend', 'auto')
    rt.setdefault('reuse_unchanged_mesh', True)
    lookup = rt['multires_gt_lookup']
    lookup.setdefault('backend', 'auto')
    lookup.setdefault('cpu_cache_mb', 2048)
    lookup.setdefault('cuda_cache_mb', 512)
    if 'max_cached_feature_steps' in lookup:
        warnings.warn('max_cached_feature_steps is obsolete and ignored; use global cpu_cache_mb/cuda_cache_mb budgets',
                      FutureWarning, stacklevel=2)
    if rt['adjacency_backend'] not in ('auto', 'cpu', 'tensor'):
        raise ValueError('adjacency_backend must be auto, cpu, or tensor')
    if not isinstance(rt['reuse_unchanged_mesh'], bool):
        raise ValueError('reuse_unchanged_mesh must be true or false')
    if lookup['backend'] not in ('auto', 'cpu', 'cuda'):
        raise ValueError('multires_gt_lookup.backend must be auto, cpu, or cuda')
    for key in ('cpu_cache_mb', 'cuda_cache_mb'):
        value = lookup[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be a finite nonnegative MiB budget')
    removed = {'precomp_cache_path', 'precomp_force_recompute', 'precompute_t_start', 'precompute_t_end',
               'mesh_mode', 'batch_size'} & cfg['train'].keys()
    removed |= {'cnn', 'warm_start_from_precompute', 'mesh_plot', 'step_log',
                'detach_policy_input', 'gradient_step0_parent_mapping_mode'} & rt.keys()
    removed |= {'chunk', 'diagnostics', 'speed'} & cfg.keys()
    removed |= {'use_attention', 'attention', 'conv_type'} & cfg['model'].keys()
    removed |= {key for key in cfg['loss'] if key.startswith('mls_') or key in
                ('parc_use_adapter', 'laplacian_weight', 'temporal_weight', 'parc_inputs')}
    if removed:
        raise ValueError(f'Unsupported legacy options in this focused project: {sorted(removed)}')
    if set(cfg['data']) - {'root', 'excluded_trajectories', 'H', 'W', 'bbox', 'window_size', 'stride'}:
        raise ValueError('Use data.root and data.excluded_trajectories; legacy data options are not supported')
    if not rt['enabled'] or rt['domain_mode'] != 'wedge_lookup':
        raise ValueError('This runtime-only project requires runtime_mesh.enabled=true and domain_mode=wedge_lookup')
    if rt['policy_backend'] not in ('gradient_fast', 'gradient'):
        raise ValueError('Supported mesh predictors: gradient_fast, gradient')
    if cfg['model']['predict_type'] != 'rate' or cfg['train']['time_integrator'] != 'euler':
        raise ValueError('This project supports rate prediction with Euler integration')
    if cfg['model']['name'] != 'FeatureNet' or cfg['loss']['physics_backend'] != 'dec':
        raise ValueError('This project supports FeatureNet and DEC/PARC')
    if cfg['edges']['method'] != 'face':
        raise ValueError('DEC training requires face adjacency')
    if cfg['features']['use_columns'] != [0, 1, 2, 3]:
        raise ValueError('Expected conservative-state columns [rho, mx, my, E] = [0,1,2,3]')
    if cfg['features']['normalization']['momentum_sigma_mode'] not in ('independent', 'shared_rms'):
        raise ValueError('momentum_sigma_mode must be independent or shared_rms')
    if cfg['features']['build']['ramp_angle_deg'] is not None:
        raise ValueError('Multi-angle training infers ramp angle per trajectory; keep ramp_angle_deg=null')
    if cfg['model']['layers'] < 2 or cfg['model']['hidden'] < 1:
        raise ValueError('FeatureNet requires layers>=2 and hidden>=1')
    if rt['idw']['backend'] not in ('exact', 'kdtree'):
        raise ValueError('Supported remapping backends: exact, kdtree')
    if cfg['data']['window_size'] < 2:
        raise ValueError('window_size must be at least 2')
    for name, value in [('stride', cfg['data']['stride']), ('H', cfg['data']['H']), ('W', cfg['data']['W']),
                        ('update_every_steps', rt['update_every_steps']), ('idw.k', rt['idw']['k']),
                        ('idw.chunk', rt['idw']['chunk']), ('epochs', cfg['train']['epochs']),
                        ('max_open_trajectories', rt['multires_gt_lookup']['max_open_trajectories']),
                        ('validation_every_epochs', cfg['train']['validation_every_epochs'])]:
        if not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if cfg['policy']['refine_ratio'] != 2 or cfg['policy']['max_level'] < 1:
        raise ValueError('Expected dyadic refinement with max_level >= 1')
    x0, x1, y0, y1 = cfg['data']['bbox']
    if not (x1 > x0 and y1 > y0):
        raise ValueError('bbox must have positive width and height')
    a, b = cfg['split']['train'], cfg['split']['val']
    if not 0 < a <= 1 or not 0 <= b < 1 or a + b > 1:
        raise ValueError('Invalid train/validation window fractions')


def device_for(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu or --device mps for local tests')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable; use --device cpu')
    return torch.device(name)


def numerical_runtime_config(runtime):
    """Cache sizes/backends may change on resume without changing the experiment."""
    result = {key: value for key, value in runtime.items()
              if key not in ('adjacency_backend', 'reuse_unchanged_mesh', 'multires_gt_lookup')}
    result['multires_gt_lookup'] = {key: value for key, value in runtime['multires_gt_lookup'].items()
        if key not in ('backend', 'cpu_cache_mb', 'cuda_cache_mb', 'max_cached_feature_steps', 'max_open_trajectories')}
    return result


def save_checkpoint(path, payload):
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def rng_state():
    return {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def main(args):
    cfg = load_config(args.config)
    if args.device:
        cfg['device'] = args.device
    if args.out_dir:
        cfg['train']['save_dir'] = str(Path(args.out_dir).resolve())
    if args.epochs is not None:
        cfg['train']['epochs'] = args.epochs
    validate_config(cfg)
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError('--max-batches must be positive')
    torch.set_num_threads(cfg['cpu_threads'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
    sources = load_trajectories(cfg)
    splits = split_windows(sources, cfg)
    manifest = split_manifest(sources, splits, cfg)
    print('[SPLIT]', {s: len(w) for s, w in splits.items()}, 'excluded=', cfg['data']['excluded_trajectories'], flush=True)
    if args.dry_run:
        print('[DRY-RUN] source/cache/spec checks passed; no training or output files written')
        return
    device = device_for(cfg['device'])
    out = Path(cfg['train']['save_dir'])
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f'Output directory is not empty: {out}; use a new --out-dir or --resume')
    out.mkdir(parents=True, exist_ok=True)
    resume = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if resume is not None:
        if resume['format'] != 'shock-ramp-multitrajectory-v1' or resume['manifest'] != manifest:
            raise ValueError('Resume checkpoint belongs to a different trajectory/window split')
        for key in ('features', 'model', 'loss', 'policy'):
            if resume['config'][key] != cfg[key]:
                raise ValueError(f'Resume {key} configuration differs from checkpoint')
        old_runtime = numerical_runtime_config(resume['config']['train']['runtime_mesh'])
        new_runtime = numerical_runtime_config(cfg['train']['runtime_mesh'])
        if old_runtime != new_runtime or resume['config']['scheduler'] != cfg['scheduler']:
            raise ValueError('Resume runtime/scheduler configuration differs from checkpoint')
    dt_ref = float(torch.cat([s.times.diff() for s in sources]).median())
    mu, sigma = ((resume['mu'], resume['sigma']) if resume else normalization(sources, splits['train'], cfg))
    mu, sigma = mu.to(device), sigma.to(device)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    sched_cfg = cfg['scheduler']
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=sched_cfg['factor'],
                patience=sched_cfg['patience'], min_lr=sched_cfg['min_lr']) if sched_cfg['use'] else None
    best, start, history = float('inf'), 1, []
    if resume:
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        if scheduler and resume['scheduler'] is not None:
            scheduler.load_state_dict(resume['scheduler'])
        best, start, history = resume['best_val'], resume['epoch']+1, resume['history']
        dt_ref = resume['dt_ref']
        restore_rng(resume['rng'])
    cfg['smoke_max_batches'] = args.max_batches
    (out/'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    (out/'split_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    runtime = Runtime(cfg, device)
    print(f'[RUNTIME] gt_lookup={runtime.lookup_device if runtime.lookup_cfg["enabled"] else "disabled"} '
          f'adjacency={runtime.adjacency_backend} reuse_unchanged_mesh={runtime.rt["reuse_unchanged_mesh"]} '
          f'feature_cache_cpu={runtime.cpu_features.max_bytes/1024**2:g}MiB '
          f'feature_cache_cuda={runtime.cuda_features.max_bytes/1024**2:g}MiB', flush=True)
    try:
        for epoch in range(start, cfg['train']['epochs']+1):
            tr = run_epoch(model, splits['train'], sources, cfg, runtime, mu, sigma, dt_ref,
                           optimizer=optimizer, epoch=epoch, max_batches=args.max_batches)
            val = None
            if splits['val'] and (epoch % cfg['train']['validation_every_epochs'] == 0 or epoch == cfg['train']['epochs']):
                val = run_epoch(model, splits['val'], sources, cfg, runtime, mu, sigma, dt_ref,
                                epoch=epoch, split='val', max_batches=args.max_batches)
            if scheduler and val:
                scheduler.step(val['loss'])
            improved = val is not None and val['loss'] < best
            if improved:
                best = val['loss']
            history.append({'epoch': epoch, 'train': tr, 'val': val, 'lr': optimizer.param_groups[0]['lr']})
            payload = {'format': 'shock-ramp-multitrajectory-v1', 'model': model.state_dict(),
                       'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict() if scheduler else None,
                       'epoch': epoch, 'best_val': best, 'mu': mu.cpu(), 'sigma': sigma.cpu(), 'dt_ref': dt_ref,
                       'config': cfg, 'manifest': manifest, 'history': history, 'rng': rng_state()}
            save_checkpoint(out/'last_model.pt', payload)
            if improved:
                save_checkpoint(out/'best_model.pt', payload)
            (out/'history.json').write_text(json.dumps(history, indent=2) + '\n')
            print(f'[EPOCH] {epoch} train={tr["loss"]:.6g} val={val["loss"] if val else None} '
                  f'train_wall={tr["seconds"]:.1f}s', flush=True)
        # Retain the original final-model, random-window test behavior. Withheld trajectories are never loaded here.
        test = run_epoch(model, splits['test'], sources, cfg, runtime, mu, sigma, dt_ref,
                         split='test', max_batches=args.max_batches) if splits['test'] else None
        (out/'test_metrics.json').write_text(json.dumps(test, indent=2) + '\n')
        print('[TEST]', test, flush=True)
    finally:
        runtime.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--device')
    parser.add_argument('--out-dir')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--max-batches', type=int, help='Smoke-test cap per split, not for production comparisons')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', help='New-project last_model.pt checkpoint')
    main(parser.parse_args())
