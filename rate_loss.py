"""Fixed, training-only scaling of the supervised normalized-rate MSE."""
from collections import Counter
import math

import torch
import torch.nn.functional as F


def loss_config_defaults(loss):
    return {'rate_loss_scaling': 'none', 'rate_loss_scaling_epsilon': 1e-8,
            'flux_raw_rate_weight': 0., 'flux_supervision_weight': 0., **loss}


def validate_rate_loss_config(loss):
    weight = loss.get('flux_raw_rate_weight', 0.)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError('loss.flux_raw_rate_weight must be finite and nonnegative')
    mode = loss.get('rate_loss_scaling', 'none')
    if mode not in ('none', 'training_rms'):
        raise ValueError('loss.rate_loss_scaling must be none or training_rms')
    epsilon = loss.get('rate_loss_scaling_epsilon', 1e-8)
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('loss.rate_loss_scaling_epsilon must be finite and positive')
    if mode != 'none' and loss['use_huber']:
        raise ValueError('training_rms rate loss scaling requires loss.use_huber=false (MSE)')


@torch.no_grad()
def estimate_rate_loss_scaling(sources, windows, cfg, runtime, sigma, dt_ref):
    """Average cellwise squared GT rates per transition, then over training steps.

    Each unique transition uses a fresh mesh predicted from its GT initial state.
    Repeated transitions in overlapping training windows retain their multiplicity.
    Both endpoints are sampled onto that same mesh through the usual GT path.
    """
    if cfg['loss'].get('rate_loss_scaling', 'none') == 'none':
        return None
    validate_rate_loss_config(cfg['loss'])
    counts = Counter((sid, t + j) for sid, t in windows
                     for j in range(cfg['data']['window_size'] - 1))
    if not counts:
        raise ValueError('Rate loss scaling requires training transitions')
    # Aggregate on CPU in float64, including when training on MPS (no float64).
    scale = sigma.detach().to(device='cpu', dtype=torch.float64)
    if not torch.isfinite(scale).all() or not (scale > 0).all() or not math.isfinite(dt_ref) or dt_ref <= 0:
        raise ValueError('Rate loss scaling requires positive finite sigma and dt_ref')
    mean_square = torch.zeros_like(scale)
    total = sum(counts.values())
    for index, ((sid, t), repeats) in enumerate(sorted(counts.items()), 1):
        source = sources[sid]
        centers, features, levels = runtime.initial_snapshot(source, t)
        mesh = runtime.predict_mesh(source, centers, features, levels, 0)
        current, _ = runtime.gt(source, t, mesh, initial=True)
        following, _ = runtime.gt(source, t + 1, mesh)
        dt = float(source.times[t + 1] - source.times[t])
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError(f'Invalid rate-scaling timestep: {source.name} t={t}')
        rate = (following.to('cpu', torch.float64) - current.to('cpu', torch.float64)) / scale / (dt / dt_ref)
        squared = rate.square().mean(0)
        if not torch.isfinite(squared).all():
            raise ValueError(f'Non-finite rate-scaling targets: {source.name} t={t}')
        mean_square += squared * (repeats / total)
        if index == 1 or index % 100 == 0 or index == len(counts):
            print(f'[RATE-SCALING] GT transitions {index}/{len(counts)}', flush=True)
    epsilon = cfg['loss'].get('rate_loss_scaling_epsilon', 1e-8)
    # Normalize before reciprocating to avoid overflow for tiny positive epsilon.
    denominator = mean_square + epsilon
    weights = denominator.min() / denominator
    weights = (weights / weights.mean()).float()
    if not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError('Rate loss weights are not representable as positive finite float32 values')
    return {'mode': 'training_rms', 'epsilon': epsilon,
            'target_rate_mean_square': mean_square.cpu(), 'weights': weights.cpu(),
            'training_steps': total, 'unique_transitions': len(counts),
            'estimator': 'gt-initial-runtime-mesh-v1'}


def checkpoint_rate_loss_scaling(checkpoint, cfg):
    """Never silently refit missing statistics on resume or use validation data."""
    state = checkpoint.get('rate_loss_scaling')
    if cfg['loss'].get('rate_loss_scaling', 'none') == 'none':
        if state is not None:
            raise ValueError('Checkpoint has rate loss scaling but configuration disables it')
        return None
    if not isinstance(state, dict) or state.get('mode') != 'training_rms':
        raise ValueError('Checkpoint is missing training_rms rate loss scaling statistics')
    if state.get('epsilon') != cfg['loss'].get('rate_loss_scaling_epsilon', 1e-8):
        raise ValueError('Checkpoint rate loss scaling epsilon differs from configuration')
    channels = len(cfg['features']['use_columns'])
    for name in ('weights', 'target_rate_mean_square'):
        value = state.get(name)
        if (not isinstance(value, torch.Tensor) or value.shape != (channels,)
                or not torch.isfinite(value).all()
                or not (value > 0 if name == 'weights' else value >= 0).all()):
            raise ValueError(f'Invalid checkpoint rate loss scaling {name}')
    denominator = state['target_rate_mean_square'].double() + state['epsilon']
    expected = denominator.min() / denominator
    expected /= expected.mean()
    if not torch.allclose(state['weights'].double(), expected, rtol=1e-5, atol=0):
        raise ValueError('Checkpoint rate loss weights disagree with saved target statistics')
    return state


def supervised_rate_loss(rate, target, cfg, weights=None):
    if cfg.get('rate_loss_scaling', 'none') == 'training_rms':
        if weights is None:
            raise ValueError('training_rms requires fixed training-derived rate loss weights')
        return ((rate - target).square().mean(0) * weights).mean()
    return (F.huber_loss(rate, target, delta=cfg['huber_delta']) if cfg['use_huber']
            else F.mse_loss(rate, target))
