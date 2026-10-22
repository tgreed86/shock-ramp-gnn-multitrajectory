#!/usr/bin/env python3
"""Write four field GIFs, a mesh comparison GIF, and numerical rollout_data.h5."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

# Allow `python eval/plot_rollout.py ...` from any working directory.
PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import numpy as np
import h5py
from matplotlib import colormaps

from eval.rollout import load_inputs
from eval.rollout_geometry import NativeGeometry
from eval.rollout_render import FEATURES, prepare_frames, render_gifs
from runtime import Runtime


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--cache-dir', required=True, type=Path,
                    help='One p-*_angle-* cache family, flat or containing uniform_meshes/')
    ap.add_argument('--native-pt', required=True, type=Path, help='Corresponding original AMR trajectory')
    ap.add_argument('--mesh-specs', type=Path, help='Override angle-specific mesh_specs directory')
    ap.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:N, or mps')
    ap.add_argument('--start', type=int, default=0, help='Initial snapshot index (default: 0)')
    ap.add_argument('--steps', type=int, help='Number of autoregressive transitions (default: remaining trajectory)')
    ap.add_argument('--frame-stride', type=int, default=1,
                    help='Display every Nth snapshot; ALL intermediate model steps still run')
    ap.add_argument('--output-dir', type=Path,
                    help='Output root (default: <checkpoint directory>/rollout_plots); '
                         'results go in <output-dir>/<trajectory>/')
    ap.add_argument('--fps', type=float, default=10, help='Constant playback FPS; physical time is labeled')
    ap.add_argument('--dpi', type=int, default=100)
    ap.add_argument('--cmap', default='viridis')
    ap.add_argument('--linewidth', type=float, default=.12)
    ap.add_argument('--min-edge-pixels', type=float, default=2,
                    help='Hide unresolved cell edges in mesh panels; 0 draws every edge')
    ap.add_argument('--xlim', type=float, nargs=2, help='Zoom x limits in model coordinates')
    ap.add_argument('--ylim', type=float, nargs=2, help='Zoom y limits in model coordinates')
    ap.add_argument('--native-map-dir', type=Path,
                    help='Directory of .2d.map.hdf5 files (auto-detects <native-pt parent>/plot)')
    ap.add_argument('--overwrite', action='store_true', help='Replace existing GIFs, rollout.json, and rollout_data.h5')
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if (args.start < 0 or (args.steps is not None and args.steps < 1) or args.frame_stride < 1
            or args.dpi < 1):
        ap.error('Use nonnegative start and positive steps/stride/DPI')
    if (not 0 < args.fps <= 100 or not np.isfinite(args.linewidth) or args.linewidth < 0
            or not np.isfinite(args.min_edge_pixels) or args.min_edge_pixels < 0):
        ap.error('Use 0 < FPS <= 100 and finite nonnegative edge widths/pixel thresholds')
    for limits in (args.xlim, args.ylim):
        if limits and (not np.isfinite(limits).all() or limits[0] >= limits[1]):
            ap.error('Axis limits must be finite and increasing')
    for name in ('checkpoint', 'cache_dir', 'native_pt', 'output_dir', 'mesh_specs', 'native_map_dir'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output_dir is None:
        args.output_dir = args.checkpoint.parent/'rollout_plots'
    try:
        colormaps[args.cmap]
        cfg, model, source, native, mu, sigma, dt_ref, checkpoint = load_inputs(
            args.checkpoint, args.cache_dir, args.native_pt, device=args.device, mesh_specs=args.mesh_specs)
        steps = args.steps if args.steps is not None else source.num_steps-args.start-1
        if steps < 1 or args.start+steps >= source.num_steps:
            raise ValueError('Requested rollout exceeds the trajectory or has no prediction steps')
        indices = list(range(args.start, args.start+steps+1, args.frame_stride))
        if indices[-1] != args.start+steps:
            indices.append(args.start+steps)
        geometry = NativeGeometry(native, range(args.start, args.start+steps+1), map_dir=args.native_map_dir)
        output = args.output_dir/source.name
        filenames = [f'{name}.gif' for name, _ in FEATURES] + ['mesh.gif', 'rollout.json', 'rollout_data.h5']
        if not args.overwrite and any((output/name).exists() for name in filenames):
            raise FileExistsError(f'Rollout output already exists in {output}; use --overwrite or a new --output-dir')
        output.mkdir(parents=True, exist_ok=True)
        runtime = Runtime(cfg, mu.device)
        print(f'[ROLLOUT] model={cfg["model"]["name"]}, data_mode={source.backend}, '
              f'start={args.start}, steps={steps}, frames={len(indices)}, geometry=exact mapped cells', flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix='.rollout-', dir=output) as temporary:
                limits, lengths = prepare_frames(temporary, indices, model, source, native, cfg, runtime,
                                                mu, sigma, dt_ref, geometry, start=args.start, steps=steps)
                duration = render_gifs(temporary, indices, source, native, cfg, runtime, geometry, limits, lengths,
                                      fps=args.fps, dpi=args.dpi, cmap=args.cmap, linewidth=args.linewidth,
                                      min_edge_pixels=args.min_edge_pixels, xlim=args.xlim, ylim=args.ylim)
                metadata = {
                    'checkpoint': str(args.checkpoint), 'checkpoint_epoch': checkpoint['epoch'],
                    'model': cfg['model'], 'data_source': source.backend, 'trajectory': source.name,
                    'cache_dir': str(source.cache_files[0].parent), 'native_pt': str(native.path),
                    'mesh_spec': str(source.spec), 'device': str(mu.device),
                    'trajectory_in_training_sources': any(s['name'] == source.name for s in checkpoint['manifest']['sources']),
                    'start': args.start, 'prediction_steps': steps, 'frame_indices': indices,
                    'data_export': 'rollout_data.h5', 'data_export_format': 'shock-ramp-rollout-v1',
                    'exported_steps': list(range(steps+1)),
                    'exported_snapshot_indices': list(range(args.start, args.start+steps+1)),
                    'frame_times': [float(source.times[t]) for t in indices], 'dt_ref': dt_ref,
                    'mu': mu.cpu().tolist(), 'sigma': sigma.cpu().tolist(),
                    'native_geometry': 'exact mapped cells',
                    'native_map_dir': str(geometry.map_dir),
                    'color_limits': {name: limits[i].tolist() for i, (name, _) in enumerate(FEATURES)},
                    'frame_duration_seconds': duration, 'display_coordinates': 'model / normalized native coordinates',
                    'xlim': args.xlim, 'ylim': args.ylim,
                    'initialization': 'training mode: initial mesh from Lmax cache for uniform_cache, native GT for native; '
                                      'initial features from configured GT lookup on that mesh',
                    'evolution': 'one continuous autoregressive rollout, physical dt, checkpoint dt_ref/normalization, '
                                 'training remesh cadence and guardrails; no GT resets or future GT inputs',
                    'mesh_timing': 'frame t+1 shows the output on the mesh chosen from state t',
                }
                (Path(temporary)/'rollout.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
                with h5py.File(Path(temporary)/'rollout_data.h5', 'r+') as data:
                    data.create_dataset('metadata_json', data=json.dumps(metadata, allow_nan=False),
                                        dtype=h5py.string_dtype('utf-8'))
                for name in filenames:
                    (Path(temporary)/name).replace(output/name)
        finally:
            runtime.close()
        print(f'[SAVED] {output}', flush=True)
        return output
    except (OSError, ValueError, KeyError, FloatingPointError, RuntimeError) as exc:
        ap.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
