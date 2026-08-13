import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import soundfile as sf
import torch

import inference


class IdentityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.input_device = None

    def forward(self, audio):
        self.input_device = audio.device
        return audio


class DeviceSelectionTests(unittest.TestCase):
    @mock.patch.object(torch.backends.mps, "is_available", return_value=True)
    @mock.patch.object(torch.cuda, "is_available", return_value=True)
    def test_auto_prefers_cuda_over_mps(self, cuda_available, mps_available):
        self.assertEqual(inference.select_device("auto"), torch.device("cuda"))

    @mock.patch.object(torch.backends.mps, "is_available", return_value=True)
    @mock.patch.object(torch.cuda, "is_available", return_value=False)
    def test_auto_uses_mps_when_cuda_is_unavailable(
        self, cuda_available, mps_available
    ):
        self.assertEqual(inference.select_device("auto"), torch.device("mps"))

    @mock.patch.object(torch.backends.mps, "is_available", return_value=False)
    @mock.patch.object(torch.cuda, "is_available", return_value=False)
    def test_auto_falls_back_to_cpu(self, cuda_available, mps_available):
        self.assertEqual(inference.select_device("auto"), torch.device("cpu"))

    @mock.patch.object(torch.cuda, "is_available", return_value=False)
    def test_explicit_unavailable_cuda_fails(self, cuda_available):
        with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
            inference.select_device("cuda")

    @mock.patch.object(torch.backends.mps, "is_available", return_value=False)
    def test_explicit_unavailable_mps_fails(self, mps_available):
        with self.assertRaisesRegex(RuntimeError, "MPS was requested"):
            inference.select_device("mps")


class InferenceTests(unittest.TestCase):
    def test_cpu_inference_preserves_input_and_audio_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.wav"
            output_path = root / "output.wav"
            checkpoint_path = root / "checkpoint.bin"
            checkpoint_path.touch()

            samples = np.linspace(-0.5, 0.5, inference.SAMPLE_RATE)
            stereo = np.stack((samples, -samples), axis=1).astype(np.float32)
            sf.write(input_path, stereo, inference.SAMPLE_RATE, subtype="FLOAT")
            input_digest = hashlib.sha256(input_path.read_bytes()).hexdigest()

            model = IdentityModel()
            with mock.patch.object(
                inference.look2hear.models.BaseModel,
                "from_pretrain",
                return_value=model,
            ):
                device = inference.run_inference(
                    input_path,
                    output_path,
                    checkpoint_path,
                    requested_device="cpu",
                )

            output, sample_rate = sf.read(
                output_path, dtype="float32", always_2d=True
            )
            self.assertEqual(device, torch.device("cpu"))
            self.assertEqual(model.input_device, torch.device("cpu"))
            self.assertEqual(sample_rate, inference.SAMPLE_RATE)
            self.assertEqual(output.shape, stereo.shape)
            self.assertTrue(np.isfinite(output).all())
            self.assertEqual(
                hashlib.sha256(input_path.read_bytes()).hexdigest(), input_digest
            )

    def test_refuses_to_overwrite_input(self):
        path = Path("same.wav")
        with self.assertRaisesRegex(ValueError, "must be different"):
            inference.run_inference(path, path, "checkpoint.bin", "cpu")


if __name__ == "__main__":
    unittest.main()
