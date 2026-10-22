"""Architecture behavior, geometry routing, and checkpoint compatibility."""
import argparse
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest

import torch
from torch import nn

import test_runtime
from engine import build_model
from model import FeatureNet, MeshGraphNet, MeshGraphNetBlock
from train import load_config, main, validate_config


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(29)
        self.cfg = load_config(Path(__file__).resolve().parents[1]/'config.json')
        self.nodes = torch.randn(5, 6)
        self.centers = torch.tensor([[0., 0.], [1., 0.], [1., 2.], [3., 2.], [5., 4.]])
        # Unequal incoming degree and an isolated node test sum aggregation.
        self.edges = torch.tensor([[0, 1, 1, 2, 3, 1], [1, 0, 2, 1, 1, 3]])

    def net(self):
        return MeshGraphNet(6, 4, hidden=8, message_passing_steps=3, mlp_layers=2, dropout=0.)

    def test_feature_net_factory_preserves_parameters_and_forward(self):
        for skips in ('none', 'block', 'input', 'both'):
            for layernorm in (False, True):
                mc = self.cfg['model']
                mc.update(hidden=8, use_skip=skips != 'none', skip_type='block' if skips == 'none' else skips,
                          use_layernorm=layernorm, dropout=.2)
                torch.manual_seed(4)
                built = build_model(self.cfg)
                self.assertIsInstance(built, FeatureNet)
                torch.manual_seed(4)
                original = FeatureNet(built.convs[0].in_channels, 4,
                    **{key: value for key, value in mc.items() if key not in ('name', 'predict_type')})
                self.assertEqual(set(original.state_dict()), set(built.state_dict()))
                x = torch.randn(5, built.convs[0].in_channels)
                torch.manual_seed(9)
                expected = original(x, self.edges)
                torch.manual_seed(9)
                torch.testing.assert_close(built(x, self.edges), expected, rtol=0, atol=0)

    def test_geometry_direction_scale_and_translation(self):
        net = self.net().eval()
        net.length_scale = 2.
        expected = torch.tensor([[-.5, 0., .5], [.5, 0., .5], [0., -1., 1.],
                                 [0., 1., 1.], [1., 1., 2**.5], [-1., -1., 2**.5]])
        torch.testing.assert_close(net.edge_features(self.centers, self.edges), expected)
        output = net(self.nodes, self.edges, self.centers)
        torch.testing.assert_close(net(self.nodes, self.edges, self.centers + 32), output)
        self.assertFalse(torch.allclose(net(self.nodes, self.edges, self.centers * 2), output))

    def test_node_and_edge_permutations_preserve_predictions(self):
        net = self.net().eval()
        output = net(self.nodes, self.edges, self.centers)
        order = torch.tensor([3, 0, 4, 2, 1])
        inverse = torch.argsort(order)
        relabeled = inverse[self.edges][:, torch.tensor([5, 2, 0, 4, 1, 3])]
        torch.testing.assert_close(net(self.nodes[order], relabeled, self.centers[order]),
                                   output[order], rtol=1e-5, atol=1e-6)

    def test_processor_sums_new_incoming_messages_then_adds_residuals(self):
        block = MeshGraphNetBlock(1, lambda n_in, n_out: nn.Linear(n_in, n_out, bias=False))
        with torch.no_grad():
            block.edge_mlp.weight.copy_(torch.tensor([[1., 2., 3.]]))
            block.node_mlp.weight.copy_(torch.tensor([[.5, 2.]]))
        nodes = torch.tensor([[1.], [2.], [4.], [8.]])
        edges = torch.tensor([[10.], [20.], [30.]])
        # Messages are 35, 68, 94, incoming to nodes 1, 1, 0 respectively.
        new_nodes, new_edges = block(nodes, edges, torch.tensor([[0, 2, 1], [1, 1, 0]]))
        torch.testing.assert_close(new_edges, torch.tensor([[45.], [88.], [124.]]))
        torch.testing.assert_close(new_nodes, torch.tensor([[189.5], [209.], [6.], [12.]]))

    def test_gradients_and_empty_edges(self):
        net = self.net()
        x = self.nodes.clone().requires_grad_()
        output = net(x, self.edges, self.centers)
        self.assertEqual(output.shape, (5, 4))
        output.square().mean().backward()
        for parameter in [x, *net.parameters()]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        for name in ('edge_encoder', 'processor.0.edge_mlp', 'processor.0.node_mlp', 'decoder'):
            self.assertGreater(sum(float(p.grad.abs().sum()) for n, p in net.named_parameters()
                                   if n.startswith(name)), 0.)
        self.assertTrue(torch.isfinite(net(x, torch.empty(2, 0, dtype=torch.long), self.centers)).all())

    def test_config_name_switch_defaults_and_minimal_meshgraphnet(self):
        self.cfg['model']['name'] = 'MeshGraphNet'
        validate_config(self.cfg)
        net = build_model(self.cfg)
        self.assertIsInstance(net, MeshGraphNet)
        self.assertEqual(len(net.processor), 15)
        self.cfg['model'] = {'name': 'MeshGraphNet', 'hidden': 8, 'predict_type': 'rate',
                             'message_passing_steps': 2, 'mlp_layers': 1}
        for parc in (True, False):
            self.cfg['loss']['parc'] = parc
            validate_config(self.cfg)
            net = build_model(self.cfg)
            inputs = torch.randn(5, net.node_encoder[0].in_features)
            self.assertEqual(net(inputs, self.edges, self.centers).shape, (5, 4))
            self.assertEqual(net.length_scale, 1/64)

    def test_invalid_model_options(self):
        self.cfg['model']['name'] = 'unsupported'
        with self.assertRaisesRegex(ValueError, 'model.name'):
            validate_config(self.cfg)
        self.cfg['model']['name'] = 'MeshGraphNet'
        for key in ('hidden', 'message_passing_steps', 'mlp_layers'):
            for value in (0, -1, True, 1.5, 'two'):
                cfg = copy.deepcopy(self.cfg)
                cfg['model'][key] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    validate_config(cfg)
        for key, values in [('dropout', [-.1, 1., float('nan')]),
                            ('layernorm_eps', [0., -1., float('inf')])]:
            for value in values:
                cfg = copy.deepcopy(self.cfg)
                cfg['model'][key] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    validate_config(cfg)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_forward_backward_matches_cpu(self):
        cpu, gpu = self.net(), self.net().cuda()
        gpu.load_state_dict(cpu.state_dict())
        a = cpu(self.nodes, self.edges, self.centers)
        b = gpu(self.nodes.cuda(), self.edges.cuda(), self.centers.cuda())
        torch.testing.assert_close(a, b.cpu(), rtol=2e-4, atol=1e-5)
        a.square().mean().backward()
        b.square().mean().backward()
        for p, q in zip(cpu.parameters(), gpu.parameters()):
            torch.testing.assert_close(p.grad, q.grad.cpu(), rtol=1e-3, atol=1e-5)


class MeshGraphNetTrainingTests(unittest.TestCase):
    tearDown = test_runtime.RuntimeTests.tearDown
    values = test_runtime.RuntimeTests.values

    def setUp(self):
        test_runtime.RuntimeTests.setUp(self)
        self.cfg['model'].update(name='MeshGraphNet', message_passing_steps=2, mlp_layers=2)

    def exercise_training(self, device):
        for predictor in ('gradient_fast', 'gradient'):
            test_runtime.RuntimeTests.exercise_training(self, device, predictor)

    def test_native_autoregressive_training_both_predictors(self):
        self.exercise_training('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_training(self):
        self.exercise_training('cuda')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS is not available')
    def test_mps_training(self):
        self.exercise_training('mps')

    def test_resume_matches_uninterrupted_training_and_rejects_architecture_change(self):
        self.cfg['model']['dropout'] = .15
        config = self.root/'config.json'
        config.write_text(json.dumps(self.cfg))
        args = argparse.Namespace(config=str(config), device='cpu', epochs=2,
                                  out_dir=str(self.root/'continuous'), max_batches=1, dry_run=False, resume=None)
        with redirect_stdout(io.StringIO()):
            main(args)
        expected = torch.load(self.root/'continuous/last_model.pt', weights_only=False)
        args.epochs, args.out_dir = 1, str(self.root/'resumed')
        with redirect_stdout(io.StringIO()):
            main(args)
        args.epochs, args.resume = 2, str(self.root/'resumed/last_model.pt')
        with redirect_stdout(io.StringIO()):
            main(args)
        actual = torch.load(args.resume, weights_only=False)
        for key, tensor in expected['model'].items():
            torch.testing.assert_close(tensor, actual['model'][key], rtol=0, atol=0)
        for key, value in [('name', 'FeatureNet'), ('message_passing_steps', 3)]:
            cfg = copy.deepcopy(self.cfg)
            cfg['model'][key] = value
            config.write_text(json.dumps(cfg))
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'model configuration differs'):
                main(args)


if __name__ == '__main__':
    unittest.main()
