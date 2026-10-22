"""Training-only rate statistics, weighted gradients, and checkpoint lifecycle."""
import argparse
import copy
from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rate_loss import (estimate_rate_loss_scaling, supervised_rate_loss,
                       checkpoint_rate_loss_scaling, validate_rate_loss_config)
from train import main
import test_cache_only
import test_runtime


class RateLossTests(unittest.TestCase):
    def test_training_transitions_dt_sigma_multiplicity_and_cell_weighting(self):
        # Two overlapping windows visit transitions 0,1 and 1,2. The last
        # transition and the other source are deliberately forbidden (held out).
        cfg = {'data': {'window_size': 3}, 'loss': {'use_huber': False,
               'rate_loss_scaling': 'training_rms', 'rate_loss_scaling_epsilon': 1e-8}}
        source = SimpleNamespace(name='train', times=torch.tensor([0., 1., 3., 4., 5.]))
        sigma = torch.tensor([2., 4., 1., 1.])
        delta = torch.tensor([[2., 8., 0., 1.], [8., 8., 0., 4.], [6., 4., 0., 3.]])
        seen = []

        class FakeRuntime:
            def initial_snapshot(self, s, t):
                assert s is source and t in (0, 1, 2)
                seen.append(t)
                return t, None, None

            def predict_mesh(self, s, t, features, levels, step):
                assert step == 0
                return t

            def gt(self, s, t, mesh, *, initial=False):
                assert t == mesh + (not initial)
                # Different cell counts must not alter transition weights.
                x = torch.zeros(mesh + 1, 4) if initial else delta[mesh].repeat(mesh + 1, 1)
                return x, 0

        state = estimate_rate_loss_scaling([source, object()], [(0, 0), (0, 1)],
                                           cfg, FakeRuntime(), sigma, 2.)
        rates = delta.double() / sigma / torch.tensor([.5, 1., .5])[:, None]
        expected = (rates[0].square() + 2*rates[1].square() + rates[2].square()) / 4
        torch.testing.assert_close(state['target_rate_mean_square'], expected)
        inverse = 1 / (expected + 1e-8)
        torch.testing.assert_close(state['weights'], (inverse / inverse.mean()).float())
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(state['training_steps'], 4)
        self.assertEqual(state['unique_transitions'], 3)
        self.assertTrue((state['weights'] > 0).all())  # Includes a zero-rate feature.
        self.assertAlmostEqual(float(state['weights'].mean()), 1.)

    def test_weighted_objective_and_gradients_and_legacy_mse_huber(self):
        pred = torch.tensor([[1., 2., 3., 4.], [-2., 0., 1., 3.]], requires_grad=True)
        target = torch.zeros_like(pred)
        weights = torch.tensor([.2, .4, 1., 2.4])
        cfg = {'use_huber': False, 'rate_loss_scaling': 'training_rms'}
        actual = supervised_rate_loss(pred, target, cfg, weights)
        torch.testing.assert_close(actual, (pred.square()*weights).mean())
        actual.backward()
        torch.testing.assert_close(pred.grad, 2*pred.detach()*weights / pred.numel())
        with self.assertRaisesRegex(ValueError, 'fixed training-derived'):
            supervised_rate_loss(pred, target, cfg)
        for huber in (False, True):
            cfg = {'use_huber': huber, 'huber_delta': .2}
            expected = F.huber_loss(pred, target, delta=.2) if huber else F.mse_loss(pred, target)
            torch.testing.assert_close(supervised_rate_loss(pred, target, cfg), expected, rtol=0, atol=0)

    def test_invalid_options_and_checkpoint_statistics(self):
        base = {'use_huber': False, 'rate_loss_scaling': 'training_rms'}
        for changes in ({'rate_loss_scaling': True}, {'rate_loss_scaling': 'typo'},
                        {'use_huber': True}, *({'rate_loss_scaling_epsilon': x}
                          for x in (0, -1, float('nan'), float('inf'), True))):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_rate_loss_config({**base, **changes})
        cfg = {'loss': base, 'features': {'use_columns': [0, 1, 2, 3]}}
        with self.assertRaisesRegex(ValueError, 'missing'):
            checkpoint_rate_loss_scaling({}, cfg)
        state = {'mode': 'training_rms', 'epsilon': 1e-8, 'weights': torch.ones(4),
                 'target_rate_mean_square': torch.ones(4)}
        self.assertIs(checkpoint_rate_loss_scaling({'rate_loss_scaling': state}, cfg), state)
        for key, value in [('weights', torch.zeros(4)), ('weights', torch.ones(3)),
                           ('weights', torch.tensor([1., 1., 1., 2.])),
                           ('target_rate_mean_square', torch.full((4,), float('nan')))]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                checkpoint_rate_loss_scaling({'rate_loss_scaling': {**state, key: value}}, cfg)
        self.assertIsNone(checkpoint_rate_loss_scaling({}, {'loss': {'use_huber': False}}))


class RateLossTrainingTests(unittest.TestCase):
    setUp = test_cache_only.CacheOnlyTests.setUp
    tearDown = test_runtime.RuntimeTests.tearDown
    values = test_runtime.RuntimeTests.values

    def run_training(self, cfg, folder, *, resume=None, epochs=1):
        path = self.root/'rate_config.json'
        path.write_text(json.dumps(cfg))
        args = argparse.Namespace(config=str(path), device='cpu', epochs=epochs, out_dir=str(folder),
                                  max_batches=1, dry_run=False, resume=resume)
        with redirect_stdout(io.StringIO()):
            main(args)
        return torch.load(folder/'last_model.pt', weights_only=False)

    def test_all_models_weighted_train_val_test_and_resume_without_refitting(self):
        import engine
        for name in ('FeatureNet', 'MeshGraphNet', 'FluxGraphNet'):
            with self.subTest(model=name):
                cfg = copy.deepcopy(self.cfg)
                cfg['model'].update(name=name, message_passing_steps=1)
                cfg['loss'].update(rate_loss_scaling='training_rms', use_huber=False)
                cfg['split'].update(train=.6, val=.2, test=.2)
                folder = self.root/name
                original = engine.run_epoch
                splits = []

                def capture(*args, **kwargs):
                    self.assertIsNotNone(kwargs.get('rate_loss_weights'))
                    splits.append(kwargs.get('split', 'train'))
                    return original(*args, **kwargs)

                with patch('train.run_epoch', side_effect=capture):
                    payload = self.run_training(cfg, folder)
                self.assertEqual(splits, ['train', 'val', 'test'])
                saved = payload['rate_loss_scaling']
                self.assertEqual(saved['training_steps'], len(payload['manifest']['windows']['train'])*2)
                self.assertTrue(torch.isfinite(torch.tensor(payload['history'][0]['train']['loss'])))
                # Change only total epochs. Resume must restore, never estimate.
                with patch('train.estimate_rate_loss_scaling', side_effect=AssertionError('Refitted on resume')):
                    resumed = self.run_training(cfg, folder, resume=str(folder/'last_model.pt'), epochs=2)
                for key in ('weights', 'target_rate_mean_square'):
                    torch.testing.assert_close(resumed['rate_loss_scaling'][key], saved[key], rtol=0, atol=0)
                self.assertEqual(resumed['epoch'], 2)

    def test_legacy_checkpoint_resume_and_disabled_mode_does_not_read_statistics(self):
        folder = self.root/'legacy'
        cfg = copy.deepcopy(self.cfg)
        cfg['loss'].pop('rate_loss_scaling')
        cfg['loss'].pop('rate_loss_scaling_epsilon')
        with patch('rate_loss.Counter', side_effect=AssertionError('Statistics read with scaling disabled')):
            payload = self.run_training(cfg, folder)
        payload.pop('rate_loss_scaling')
        payload['config']['loss'].pop('rate_loss_scaling')
        payload['config']['loss'].pop('rate_loss_scaling_epsilon')
        torch.save(payload, folder/'legacy.pt')
        resumed = self.run_training(cfg, folder, resume=str(folder/'legacy.pt'), epochs=2)
        self.assertEqual(resumed['epoch'], 2)
        self.assertIsNone(resumed['rate_loss_scaling'])

    def test_native_gt_without_cache(self):
        cfg = copy.deepcopy(self.cfg)
        (self.root/'unavailable_native_data').rename(self.root/'data')
        cfg['data'].update(source='native', root=str(self.root/'data'))
        cfg['loss'].update(rate_loss_scaling='training_rms', use_huber=False)
        cfg['train']['runtime_mesh']['multires_gt_lookup']['enabled'] = False
        payload = self.run_training(cfg, self.root/'native')
        self.assertTrue((payload['rate_loss_scaling']['weights'] > 0).all())
