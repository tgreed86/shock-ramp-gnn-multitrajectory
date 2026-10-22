#!/usr/bin/env python3
"""Plot training/validation loss from a run directory or its history.json file."""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # Save figures on clusters without a display.
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def load_losses(path):
    history = json.loads(path.read_text())
    if not isinstance(history, list) or not history:
        raise ValueError('History must be a nonempty list of epoch records')
    series = {'train': [], 'val': []}
    seen = set()
    for record in history:
        if not isinstance(record, dict):
            raise ValueError('Each history record must be an object')
        epoch = record.get('epoch')
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise ValueError(f'Invalid epoch: {epoch!r}')
        if epoch in seen:
            raise ValueError(f'Duplicate epoch: {epoch}')
        seen.add(epoch)
        for split in series:
            metrics = record.get(split)
            if metrics is None:
                continue
            if not isinstance(metrics, dict):
                raise ValueError(f'Epoch {epoch}: {split} metrics must be an object or null')
            loss = metrics.get('loss')
            if loss is None:
                continue
            if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss):
                raise ValueError(f'Epoch {epoch}: {split} loss must be a finite number')
            series[split].append((epoch, loss))
    if not any(series.values()):
        raise ValueError('History contains no training or validation loss values')
    return {split: sorted(points) for split, points in series.items()}


def plot_history(history_path, output_path, *, log_y=False, title=None):
    series = load_losses(history_path)
    if log_y and any(loss <= 0 for points in series.values() for _, loss in points):
        raise ValueError('--log-y requires strictly positive loss values; use the default linear scale')
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    try:
        for split, label, color in [('train', 'Training', '#2563eb'), ('val', 'Validation', '#d97706')]:
            points = series[split]
            if points:
                epochs, losses = zip(*points)
                ax.plot(epochs, losses, label=label, color=color, linewidth=1.8,
                        marker='o', markersize=3.5)
        if log_y:
            ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (mean per rollout step)')
        ax.set_title(title if title is not None else f'Training history — {history_path.parent.name}')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=.25)
        ax.legend(frameon=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    finally:
        plt.close(fig)
    return series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('history', type=Path, help='Run directory or path to history.json')
    parser.add_argument('-o', '--output', type=Path,
                        help='Figure path (PNG, PDF, SVG, etc.); default: loss_history.png beside history.json')
    parser.add_argument('--log-y', action='store_true', help='Use a logarithmic loss axis')
    parser.add_argument('--title', help='Override the plot title')
    args = parser.parse_args()
    history_path = args.history.expanduser().resolve()
    if history_path.is_dir():
        history_path /= 'history.json'
    output_path = (args.output.expanduser().resolve() if args.output
                   else history_path.parent/'loss_history.png')
    if output_path == history_path:
        parser.error('The output path must differ from the input history file')
    try:
        series = plot_history(history_path, output_path, log_y=args.log_y, title=args.title)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f'Saved {output_path} '
          f'({len(series["train"])} training points, {len(series["val"])} validation points)')


if __name__ == '__main__':
    main()
