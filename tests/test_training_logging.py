"""Epoch-only logging keeps summaries and saved artifacts without batch output."""
import argparse
from contextlib import redirect_stdout
import io
import json
import unittest

import test_runtime
from train import main, validate_config


class TrainingLoggingTests(unittest.TestCase):
    setUp = test_runtime.RuntimeTests.setUp
    tearDown = test_runtime.RuntimeTests.tearDown
    values = test_runtime.RuntimeTests.values

    def arguments(self, name, **overrides):
        config = self.root/'config.json'
        config.write_text(json.dumps(self.cfg))
        return argparse.Namespace(**dict(dict(config=str(config), device='cpu', epochs=2,
            out_dir=str(self.root/name), max_batches=1, dry_run=False, resume=None,
            epoch_only=False), **overrides))

    def test_epoch_only_overrides_bars_batches_and_preserves_artifacts(self):
        self.cfg['debug'].update(epoch_only=True, progress_bar=True, progress_every_batches=1)
        self.cfg['loss']['rate_loss_scaling'] = 'training_rms'
        output = io.StringIO()
        args = self.arguments('quiet')
        with redirect_stdout(output):
            main(args)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        for epoch, line in enumerate(lines, 1):
            self.assertTrue(line.startswith(f'[EPOCH] {epoch} train='))
            self.assertIn('val=', line)
        folder = self.root/'quiet'
        self.assertEqual(len(json.loads((folder/'history.json').read_text())), 2)
        self.assertIsNotNone(json.loads((folder/'test_metrics.json').read_text()))
        self.assertTrue((folder/'last_model.pt').is_file())
        self.assertTrue((folder/'best_model.pt').is_file())

    def test_cli_override_and_detailed_logging_on_resume(self):
        self.cfg['debug'].update(epoch_only=False, progress_bar=True, progress_every_batches=1)
        args = self.arguments('override', epochs=1, epoch_only=True)
        output = io.StringIO()
        with redirect_stdout(output):
            main(args)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        args.epoch_only = False
        args.epochs = 2
        args.resume = str(self.root/'override/last_model.pt')
        output = io.StringIO()
        with redirect_stdout(output):
            main(args)
        self.assertIn('[BATCH]', output.getvalue())
        self.assertIn('[EPOCH] 2', output.getvalue())
        self.assertIn('[TEST]', output.getvalue())

    def test_dry_run_completion_and_invalid_setting(self):
        self.cfg['debug']['epoch_only'] = True
        output = io.StringIO()
        with redirect_stdout(output):
            main(self.arguments('dry', dry_run=True))
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertTrue(output.getvalue().startswith('[DRY-RUN]'))
        self.assertFalse((self.root/'dry').exists())
        self.cfg['debug']['epoch_only'] = 'true'
        with self.assertRaisesRegex(ValueError, 'debug.epoch_only'):
            validate_config(self.cfg)


if __name__ == '__main__':
    unittest.main()
