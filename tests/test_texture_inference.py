"""CLI integration without checkpoint download or real model inference."""
import tempfile
from pathlib import Path
import unittest
from unittest import mock
import numpy as np
import soundfile as sf
import torch
import inference


class IntegrationTests(unittest.TestCase):
    def test_optional_postprocessing_and_default_bypass(self):
        t = np.arange(4410)/44100
        x = np.array([.4*np.sin(2*np.pi*220*t) + .02*(.1+(np.mod(t,.08)<.008))*np.sin(2*np.pi*4000*t)], dtype='float32')
        from texture_softening import soften_texture
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, dest, checkpoint = root/'in.wav', root/'out.wav', root/'model.pt'
            checkpoint.touch()
            sf.write(source, x.T, 44100, subtype='FLOAT')
            with mock.patch.object(inference.look2hear.models.BaseModel, 'from_pretrain', return_value=torch.nn.Identity()):
                with mock.patch('texture_softening.soften_texture', wraps=soften_texture) as process:
                    inference.run_inference(source, dest, checkpoint, 'cpu')
                    process.assert_not_called()
                    y, _ = sf.read(dest, dtype='float32', always_2d=True)
                    np.testing.assert_array_equal(y.T, x)
                    inference.run_inference(source, dest, checkpoint, 'cpu', texture_reduction_db=2)
                    process.assert_called_once()
                    y, _ = sf.read(dest, dtype='float32', always_2d=True)
                    expected, _ = soften_texture(x)
                    np.testing.assert_array_equal(y.T, expected)
            original, _ = sf.read(source, dtype='float32', always_2d=True)
            np.testing.assert_array_equal(original.T, x)

    def test_invalid_cap_rejected_before_model_load(self):
        for cap in (-1, 4, np.nan, np.inf):
            with self.assertRaises(ValueError):
                inference.run_inference('unused.wav', 'out.wav', 'unused.pt', texture_reduction_db=cap)

    def test_missing_scipy_fails_before_audio_checkpoint_or_model(self):
        missing = ModuleNotFoundError("No module named scipy", name="scipy")
        with mock.patch.object(inference.importlib, 'import_module', side_effect=missing), \
             mock.patch.object(inference, 'load_audio') as audio, \
             mock.patch.object(inference, 'resolve_checkpoint') as checkpoint, \
             mock.patch.object(inference.look2hear.models.BaseModel, 'from_pretrain') as model:
            with self.assertRaisesRegex(RuntimeError, 'requires SciPy'):
                inference.run_inference('in.wav', 'out.wav', 'model.pt', 'cpu', texture_reduction_db=2)
            audio.assert_not_called()
            checkpoint.assert_not_called()
            model.assert_not_called()

    def test_short_clip_rejected_before_checkpoint_but_disabled_still_works(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, dest = root/'short.wav', root/'out.wav'
            x = np.zeros((1024, 1), dtype='float32')
            sf.write(source, x, 44100, subtype='FLOAT')
            with mock.patch.object(inference, 'resolve_checkpoint') as checkpoint, \
                 mock.patch.object(inference.look2hear.models.BaseModel, 'from_pretrain', return_value=torch.nn.Identity()) as model:
                with self.assertRaisesRegex(ValueError, 'at least 2048'):
                    inference.run_inference(source, dest, 'model.pt', 'cpu', texture_reduction_db=2)
                checkpoint.assert_not_called()
                model.assert_not_called()
                self.assertFalse(dest.exists())
                with mock.patch.object(inference, 'load_texture_processor') as optional:
                    inference.run_inference(source, dest, 'model.pt', 'cpu')
                    optional.assert_not_called()
                y, _ = sf.read(dest, dtype='float32', always_2d=True)
                np.testing.assert_array_equal(y, x)

    def test_unrelated_import_errors_are_not_misreported(self):
        missing = ModuleNotFoundError('missing other', name='other')
        with mock.patch.object(inference.importlib, 'import_module', side_effect=missing):
            with self.assertRaises(ModuleNotFoundError):
                inference.load_texture_processor()
