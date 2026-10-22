"""One autoregressive runtime-mesh path shared by training and evaluation."""
import time
import sys
from contextlib import nullcontext, redirect_stdout

import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile

import kernels as k
import physics as dec
from rate_loss import supervised_rate_loss
from flux import limit_flux_rate, limit_face_transports
from model import FeatureNet, FluxGraphNet, MeshGraphNet, resolve_ramp_boundary_channels


def build_model(cfg):
    build, mc = cfg['features']['build'], cfg['model']
    channels = len(cfg['features']['use_columns'])
    inputs = channels + 2 * build['use_pos'] + build['use_level'] + build['use_dt_hat_fixed']
    inputs += build['include_ramp_angle'] + build['include_signed_distance_to_ramp']
    if cfg['loss']['parc']:
        inputs += dec.parc_extra_in_channels(cfg, channels)
    if mc['name'] == 'FeatureNet':
        kwargs = {key: mc[key] for key in ('hidden', 'layers', 'dropout', 'use_skip', 'skip_type',
                  'use_layernorm', 'layernorm_eps', 'activation', 'activation_negative_slope', 'activation_elu_alpha')}
        return FeatureNet(in_channels=inputs, out_channels=channels, **kwargs)
    if mc['name'] in ('MeshGraphNet', 'FluxGraphNet'):
        kwargs = {key: mc[key] for key in ('hidden', 'message_passing_steps', 'mlp_layers', 'dropout',
                  'layernorm_eps', 'activation', 'activation_negative_slope', 'activation_elu_alpha') if key in mc}
        x0, x1, y0, y1 = cfg['data']['bbox']
        scale = max((x1-x0)/cfg['data']['W'], (y1-y0)/cfg['data']['H'])
        model_type = MeshGraphNet
        if mc['name'] == 'FluxGraphNet':
            model_type = FluxGraphNet
            kwargs.update({key: mc[key] for key in ('use_open_boundary_source', 'use_ramp_boundary_source',
                'open_boundary_source_channels') if key in mc})
            kwargs['ramp_boundary_source_channels'] = resolve_ramp_boundary_channels(mc)
        return model_type(in_channels=inputs, out_channels=channels, length_scale=scale, **kwargs)
    raise ValueError(f'Unsupported model.name: {mc["name"]}')


def model_step(model, features, target, mesh, source, cfg, runtime, mu, sigma, dt, dt_ref, step,
               *, rate_loss_weights=None, diagnostics=None):
    """Advance one step; target=None runs the identical predictor without a loss."""
    norm = (features - mu) / sigma
    dt_phys = features.new_tensor(dt)
    dt_ref_tensor = features.new_tensor(dt_ref)
    dt_hat = dt_phys / dt_ref_tensor
    loss_cfg = cfg['loss']
    adv = diff = area = phy = None
    need_phy = loss_cfg['parc'] or loss_cfg['blend_weight'] != 0 or loss_cfg['residual_weight'] != 0
    if need_phy:
        adv_active = loss_cfg['parc_include_adv'] and (step == 0 or loss_cfg['adv_all_steps']) and loss_cfg['adv_weight'] != 0
        diff_active = loss_cfg['parc_include_diff'] and loss_cfg['diff_weight'] != 0
        with torch.no_grad():
            state = k.sanitize_state_for_ops(features, cfg)
            adv, diff, area = dec.dec_advdiff_terms_abs(state, mesh.edges, mesh.edge_attr, mesh.levels,
                dx0=runtime.dx, dy0=runtime.dy, cfg=cfg, compute_adv=adv_active, compute_diff=diff_active)
            adv, diff = k._sanitize_ops_term(adv, cfg), k._sanitize_ops_term(diff, cfg)
            phy = features.new_zeros(features.shape)
            if adv is not None:
                phy = phy + loss_cfg['adv_weight'] * adv
            if diff is not None:
                phy = phy + loss_cfg['diff_weight'] * diff
            phy = k._sanitize_ops_term(phy, cfg)
    inputs = k._build_X(norm, mesh.centers, mesh.levels, cfg, dt_hat=dt_hat, ramp_feature_ctx=source.ramp)
    if loss_cfg['parc']:
        zeros = torch.zeros_like(features)
        extra = dec.parc_terms_to_node_inputs(adv if adv is not None else zeros,
                diff if diff is not None else zeros, dt_phys=dt_phys, dt_ref=dt_ref_tensor,
                sigma=sigma, predict_type='rate', cfg=cfg, dtype=inputs.dtype, detach=True)
        inputs = torch.cat([inputs, k._sanitize_parc_extra_tensor(extra, cfg)], dim=1)
    flux_model = cfg['model']['name'] == 'FluxGraphNet'
    face_limiter = flux_model and cfg['train'].get('flux_limiter', 'global') == 'local'
    if flux_model:
        geometry = runtime.flux_geometry(mesh)
        if face_limiter:
            rate, transport, exterior = model.predict_rate_and_flux(inputs, mesh.edges, mesh.centers, geometry)
        else:
            rate = model(inputs, mesh.edges, mesh.centers, geometry)
    elif cfg['model']['name'] == 'MeshGraphNet':
        rate = model(inputs, mesh.edges, mesh.centers)
    else:
        rate = model(inputs, mesh.edges)
    if loss_cfg['blend_weight'] != 0:
        baseline = dec.physics_to_model_units(phy, dt_phys=dt_phys, dt_ref=dt_ref_tensor,
                                              sigma=sigma, predict_type='rate')
        rate = rate + loss_cfg['blend_weight'] * baseline
    if flux_model:
        raw_rate = rate
        if face_limiter:
            rate, _, _, scales = limit_face_transports(transport, exterior, mesh.edges, geometry,
                features, sigma, dt_hat, model.length_scale, cfg)
        else:
            rate = limit_flux_rate(rate, features, sigma, dt_hat, cfg)
        if diagnostics is not None:
            diagnostics['raw_rate_rms'] = raw_rate.detach().square().mean().sqrt()
            diagnostics['applied_rate_rms'] = rate.detach().square().mean().sqrt()
            if face_limiter:
                diagnostics['limited_face_fraction'] = ((scales.detach() < 1).float().mean()
                    if scales.numel() else rate.new_zeros(()))
        predicted = features + rate * dt_hat * sigma
    else:
        rate = k._apply_rate_guardrails(rate, dt_hat, cfg)
        predicted = (norm + rate * dt_hat) * sigma + mu
    if target is None:
        return None, predicted
    norm_target = (target - mu) / sigma
    target_rate = (norm_target - norm) / dt_hat.clamp_min(1e-12)
    loss = supervised_rate_loss(rate, target_rate, loss_cfg, rate_loss_weights)
    raw_weight = loss_cfg.get('flux_raw_rate_weight', 0.) if flux_model else 0.
    if raw_weight > 0:
        raw_loss = supervised_rate_loss(raw_rate, target_rate, loss_cfg, rate_loss_weights)
        if diagnostics is not None:
            diagnostics['raw_rate_loss'] = raw_loss.detach()
            diagnostics['applied_rate_loss'] = loss.detach()
        loss = loss + raw_weight * raw_loss
    if loss_cfg['residual_weight'] > 0:
        selected = sorted(set(dec.parc_select_feature_indices_adv(cfg, features.shape[1]) +
                              dec.parc_select_feature_indices_diff(cfg, features.shape[1])))
        mask = features.new_zeros(features.shape[1])
        mask[selected] = 1
        loss = loss + loss_cfg['residual_weight'] * dec.physics_residual_loss_delta(
            y_pred_abs=predicted, x_in_abs=features, dt_phys=dt_phys, r_phy_abs=phy,
            area=area, sigma=sigma, channel_mask=mask)
    return loss, predicted


def carry_state(predicted, cfg):
    # FluxGraphNet already limits the whole update to the physical bounds.
    # Per-node clipping here would inject unaccounted sources into the interior.
    return predicted if cfg['model']['name'] == 'FluxGraphNet' else k._enforce_physical_state(predicted, cfg)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def run_epoch(model, windows, sources, cfg, runtime, mu, sigma, dt_ref, *, optimizer=None,
              epoch=0, split='train', max_batches=None, rate_loss_weights=None):
    training = optimizer is not None
    model.train(training)
    order = np.random.default_rng(cfg['seed'] + epoch).permutation(len(windows))
    if max_batches is not None:
        order = order[:max_batches]
    totals = {'loss': 0., 'mae': 0., 'steps': 0, 'batches': 0, 'missing': 0, 'rebuilds': 0}
    by_source = {}
    start_epoch = time.perf_counter()
    timed_sync = cfg['debug']['sync_runtime_timing']

    def now():
        if timed_sync:
            sync(runtime.device)
        return time.perf_counter()

    show_progress = cfg['debug'].get('progress_bar', True) and len(order) > 0
    description = f'Epoch {epoch} | {split}' if epoch > 0 else split
    with tqdm(total=len(order), desc=description, unit='batch', file=sys.stdout,
              disable=not show_progress, dynamic_ncols=True, ascii=True,
              mininterval=cfg['debug'].get('progress_bar_mininterval', 5.0)) as progress, \
         (redirect_stdout(DummyTqdmFile(sys.stdout)) if show_progress else nullcontext()), \
         torch.set_grad_enabled(training):
        for bi, wi in enumerate(order):
            sid, t0 = windows[wi]
            source = sources[sid]
            start = now()
            elapsed = dict(mesh=0., remap=0., model=0., backward=0.)
            centers, state, levels = runtime.initial_snapshot(source, t0)
            mesh = None
            losses, maes, flux_diagnostics = [], [], []
            missing, rebuilds = 0, 0
            if training:
                optimizer.zero_grad(set_to_none=True)
            for step in range(cfg['data']['window_size']-1):
                old_mesh = mesh
                rebuild = step % cfg['train']['runtime_mesh']['update_every_steps'] == 0
                if rebuild:
                    mark = now()
                    mesh = runtime.predict_mesh(source, centers, state, levels, step, previous_mesh=old_mesh)
                    elapsed['mesh'] += now() - mark
                    rebuilds += 1
                mark = now()
                if step == 0:
                    state, misses = runtime.gt(source, t0, mesh, initial=True)
                    missing += misses
                elif rebuild:
                    state = runtime.remap(old_mesh, state, mesh)
                target, misses = runtime.gt(source, t0 + step + 1, mesh)
                missing += misses
                elapsed['remap'] += now() - mark
                mark = now()
                dt = float(source.times[t0+step+1] - source.times[t0+step])
                diagnostics = {}
                loss, predicted = model_step(model, state, target, mesh, source, cfg, runtime,
                                             mu, sigma, dt, dt_ref, step, rate_loss_weights=rate_loss_weights,
                                             diagnostics=diagnostics)
                flux_diagnostics.append(diagnostics)
                losses.append(loss)
                maes.append((predicted.detach()-target).abs().mean())
                state = carry_state(predicted, cfg)
                centers, levels = mesh.centers, mesh.levels
                elapsed['model'] += now() - mark
            loss_sum = torch.stack(losses).sum()
            if not torch.isfinite(loss_sum):
                raise FloatingPointError(f'{source.name} t0={t0}: non-finite loss')
            if training:
                mark = now()
                # Preserve the baseline's sum across the unrolled window for backward.
                loss_sum.backward()
                clip = cfg['train']['grad_clip']
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip > 0 else float('inf'))
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f'{source.name} t0={t0}: non-finite gradients')
                optimizer.step()
                elapsed['backward'] += now() - mark
            batch_loss = float(loss_sum.detach())
            batch_mae = float(torch.stack(maes).sum())
            steps = len(losses)
            totals['loss'] += batch_loss
            totals['mae'] += batch_mae
            totals['steps'] += steps
            totals['batches'] += 1
            if flux_diagnostics[0]:
                keys = list(flux_diagnostics[0])
                values = torch.stack([torch.stack([d[key] for key in keys])
                                      for d in flux_diagnostics]).sum(0).tolist()
                for key, value in zip(keys, values):
                    totals.setdefault('flux', {}).setdefault(key, 0.)
                    totals['flux'][key] += value
            totals['missing'] += missing
            totals['rebuilds'] += rebuilds
            rec = by_source.setdefault(source.name, {'loss': 0., 'mae': 0., 'steps': 0})
            rec['loss'] += batch_loss
            rec['mae'] += batch_mae
            rec['steps'] += steps
            every = cfg['debug']['progress_every_batches']
            if every and ((bi + 1) % every == 0 or bi == 0):
                seconds = now() - start
                parts = ' '.join(f'{key}={value:.3f}s' for key, value in elapsed.items())
                print(f'[BATCH] {split} {bi+1}/{len(order)} source={source.name} t0={t0} '
                      f'wall={seconds:.3f}s {parts} rebuilds={rebuilds} missing={missing}', flush=True)
            # Use already-materialized metrics; the bar adds no device synchronization.
            if show_progress:
                progress.set_postfix(loss=f'{totals["loss"]/totals["steps"]:.5g}',
                                     mae=f'{totals["mae"]/totals["steps"]:.5g}', refresh=False)
                progress.update(1)
    sync(runtime.device)
    totals['seconds'] = time.perf_counter() - start_epoch
    for record in [totals, *by_source.values()]:
        n = record['steps']
        record['loss'] = record['loss']/n if n else None
        record['mae'] = record['mae']/n if n else None
    totals['by_source'] = by_source
    if 'flux' in totals:
        totals['flux'] = {key: value / totals['steps'] for key, value in totals['flux'].items()}
    return totals
