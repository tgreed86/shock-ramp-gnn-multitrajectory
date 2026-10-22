"""Bounded-memory GIF rendering with fixed axes and shared field color scales."""
from contextlib import ExitStack
from pathlib import Path

import imageio.v2 as imageio
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, Normalize
import numpy as np

from eval.rollout import iter_rollout
from eval.rollout_export import RolloutExport
from eval.rollout_geometry import array, rectangular_quads


FEATURES = [('density', 'Density'), ('x_momentum', 'X momentum'),
            ('y_momentum', 'Y momentum'), ('energy', 'Energy density')]


def update_edge_lengths(lengths, quads, levels):
    shortest = np.linalg.norm(quads - np.roll(quads, 1, axis=1), axis=2).min(axis=1)
    for level in np.unique(levels):
        value = float(shortest[levels == level].min())
        lengths[int(level)] = min(lengths.get(int(level), float('inf')), value)


def prepare_frames(folder, indices, model, source, native, cfg, runtime, mu, sigma,
                   dt_ref, geometry, *, start, steps):
    """Export every physical step once; GIF color ranges use displayed frames."""
    selected = set(indices)
    limits = np.array([[np.inf, -np.inf]] * 4)
    edge_lengths = [{}, {}]
    reader = runtime.lookup(source)
    uc, _, ul = reader.uniform_snapshot(cfg['policy']['max_level'], start, 'cpu')
    with RolloutExport(Path(folder)/'rollout_data.h5', cfg, source, uc, ul, mu, sigma, dt_ref, start, steps) as export:
        for index, mesh, state in iter_rollout(model, source, cfg, runtime, mu, sigma, dt_ref, start=start, steps=steps):
            centers, levels, predicted = array(mesh.centers), array(mesh.levels), array(state)
            _, uniform, _ = reader.uniform_snapshot(cfg['policy']['max_level'], index, 'cpu')
            nc, original, nl = native.snapshot(index, cfg, 'cpu')
            uniform, nc, original, nl = array(uniform), array(nc), array(original), array(nl)
            if nc.shape != (len(nl), 2) or original.shape != (len(nl), 4):
                raise ValueError(f'Invalid native snapshot geometry/fields at {index}')
            for values in (predicted, uniform, original):
                if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
                    raise ValueError(f'Invalid or non-finite fields at snapshot {index}')
                if index in selected:
                    limits[:, 0] = np.minimum(limits[:, 0], values.min(axis=0))
                    limits[:, 1] = np.maximum(limits[:, 1], values.max(axis=0))
            vertices = geometry.quads(index)
            export.append(index, centers, levels, predicted, uniform, nc, nl, original, vertices, native.series[index])
            if index in selected:
                update_edge_lengths(edge_lengths[0], rectangular_quads(centers, levels, cfg), levels)
                update_edge_lengths(edge_lengths[1], vertices, nl)
            print(f'[ROLLOUT] snapshot {index}/{start+steps}, t={float(source.times[index]):.6g}, '
                  f'predicted cells={len(levels):,}', flush=True)
    for bounds in limits:
        if bounds[0] == bounds[1]:
            pad = max(abs(bounds[0])*.01, 1e-12)
            bounds += [-pad, pad]
    return limits, edge_lengths


def make_axes(count, cfg, dpi, xlim, ylim):
    fig, axes = plt.subplots(1, count, figsize=(5.1*count, 4.7), dpi=dpi, squeeze=False)
    axes = list(axes[0])
    x0, x1, y0, y1 = cfg['data']['bbox']
    for ax in axes:
        ax.set_xlim(xlim or (x0, x1))
        ax.set_ylim(ylim or (y0, y1))
        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor('#f3f4f6')
        ax.set_xlabel('x (model coordinates)')
    axes[0].set_ylabel('y (model coordinates)')
    fig.subplots_adjust(left=.055, right=.92, bottom=.14, top=.81, wspace=.18)
    return fig, axes


def ensure_infinite_loop(path):
    # GIF-PIL encodes loop=0 as 65535; patch the GIF extension to true infinite looping.
    marker = b'\x21\xff\x0bNETSCAPE2.0\x03\x01'
    with Path(path).open('r+b') as stream:
        header = stream.read(4096)
        offset = header.find(marker)
        if offset < 0:
            raise ValueError(f'Missing GIF loop extension: {path}')
        stream.seek(offset + len(marker))
        stream.write(b'\x00\x00')


def render_gifs(folder, indices, source, native, cfg, runtime, geometry, limits, edge_lengths,
                *, fps, dpi, cmap, linewidth, min_edge_pixels, xlim, ylim):
    folder = Path(folder)
    duration = max(1, round(100/fps))/100  # GIF durations have 10 ms precision.
    field_fig, field_axes = make_axes(3, cfg, dpi, xlim, ylim)
    mesh_fig, mesh_axes = make_axes(2, cfg, dpi, xlim, ylim)
    field_collections = [PolyCollection([], edgecolors='none', antialiaseds=False) for _ in field_axes]
    mesh_collections = []
    try:
        for ax, coll in zip(field_axes, field_collections):
            ax.add_collection(coll)
            coll.set_cmap(cmap)
        cax = field_fig.add_axes([.94, .2, .012, .54])
        field_bar = field_fig.colorbar(field_collections[0], cax=cax)
        for ax, lengths, label in zip(mesh_axes, edge_lengths, ['Adaptive level (ratio 2)', 'Native AMR level']):
            highest = max(lengths)
            coll = PolyCollection([], cmap=plt.get_cmap('viridis', highest+1),
                                  norm=BoundaryNorm(np.arange(highest+2)-.5, highest+1),
                                  edgecolors='#20252b', antialiaseds=False)
            ax.add_collection(coll)
            mesh_collections.append(coll)
            bar = mesh_fig.colorbar(coll, ax=ax, fraction=.04, pad=.025, ticks=range(highest+1))
            bar.set_label(label)
        mesh_fig.canvas.draw()
        norms = [Normalize(low, high) for low, high in limits]
        names = [name for name, _ in FEATURES] + ['mesh']
        with ExitStack() as stack:
            data = stack.enter_context(h5py.File(folder/'rollout_data.h5', 'r'))
            uc, ul = data['uniform_mesh/centers'][...], data['uniform_mesh/levels'][...]
            writers = {name: stack.enter_context(imageio.get_writer(
                str(folder/f'{name}.gif'), format='GIF-PIL', mode='I',
                duration=duration, loop=0, subrectangles=False)) for name in names}
            for number, index in enumerate(indices, 1):
                saved = data[f'snapshots/{index:06d}']
                pc, pl, pv = (saved[f'prediction/{key}'][...] for key in ('centers', 'levels', 'values'))
                uv, nv, nl = saved['uniform_gt/values'][...], saved['native_gt/values'][...], saved['native_gt/levels'][...]
                quads = [rectangular_quads(pc, pl, cfg), rectangular_quads(uc, ul, cfg), saved['native_gt/vertices'][...]]
                stamp = f'{source.name} | snapshot {index} | t = {float(source.times[index]):.6g}'
                titles = [f'Model prediction ({len(pl):,} cells)',
                          f'Uniform L{cfg["policy"]["max_level"]} GT ({len(ul):,} cells)',
                          f'{geometry.label} ({len(nl):,} cells)']
                if number == 1:
                    titles[0] = f'Initial GT on predicted mesh ({len(pl):,} cells)'
                for ax, coll, vertices, title in zip(field_axes, field_collections, quads, titles):
                    coll.set_verts(vertices)
                    ax.set_title(title, fontsize=10)
                for channel, (name, label) in enumerate(FEATURES):
                    for coll, values in zip(field_collections, (pv, uv, nv)):
                        coll.set_array(values[:, channel])
                        coll.set_norm(norms[channel])
                    field_bar.update_normal(field_collections[0])
                    field_bar.set_label(label)
                    field_fig.suptitle(f'{label} — {stamp}', fontsize=12, y=.96)
                    field_fig.canvas.draw()
                    writers[name].append_data(np.asarray(field_fig.canvas.buffer_rgba())[..., :3].copy())
                for ax, coll, vertices, levels, lengths, title in zip(
                        mesh_axes, mesh_collections, (quads[0], quads[2]), (pl, nl), edge_lengths,
                        ('Predicted adaptive mesh', geometry.label)):
                    coll.set_verts(vertices)
                    coll.set_array(levels)
                    # One decision per level across all frames prevents moving alias contours.
                    pixels = ax.bbox.width / (ax.get_xlim()[1]-ax.get_xlim()[0])
                    widths = np.zeros(len(levels))
                    for level, size in lengths.items():
                        if min_edge_pixels == 0 or size*pixels >= min_edge_pixels:
                            widths[levels == level] = linewidth
                    coll.set_linewidth(widths)
                    ax.set_title(f'{title}\n{len(levels):,} cells', fontsize=10)
                mesh_fig.suptitle(f'Mesh — {stamp}', fontsize=12, y=.96)
                mesh_fig.canvas.draw()
                writers['mesh'].append_data(np.asarray(mesh_fig.canvas.buffer_rgba())[..., :3].copy())
                print(f'[GIF] frame {number}/{len(indices)}', flush=True)
        for name in names:
            ensure_infinite_loop(folder/f'{name}.gif')
    finally:
        plt.close(field_fig)
        plt.close(mesh_fig)
    return duration
