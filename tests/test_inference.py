import hashlib
import os
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
        self.input_devices = []
        self.input_lengths = []

    def forward(self, audio):
        self.input_device = audio.device
        self.input_devices.append(audio.device)
        self.input_lengths.append(audio.shape[-1])
        return audio


class ShortOutputModel(IdentityModel):
    def forward(self, audio):
        super().forward(audio)
        return audio[..., :-1]


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
    def test_official_checkpoint_downloads_from_trusted_repository(self):
        with mock.patch.object(
            inference,
            "hf_hub_download",
            return_value="/cache/pytorch_model.bin",
        ) as download:
            path = inference.resolve_checkpoint(inference.OFFICIAL_CHECKPOINT)

        self.assertEqual(path, Path("/cache/pytorch_model.bin"))
        download.assert_called_once_with(
            repo_id=inference.OFFICIAL_CHECKPOINT,
            filename="pytorch_model.bin",
        )

    def test_existing_local_checkpoint_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            checkpoint.touch()
            self.assertEqual(inference.resolve_checkpoint(checkpoint), checkpoint)

    def test_arbitrary_hugging_face_repository_is_rejected(self):
        for reference in ("someone/model", "someone/model.v2"):
            with self.subTest(reference=reference), mock.patch.object(
                inference, "hf_hub_download"
            ) as download:
                with self.assertRaisesRegex(ValueError, "Unsupported checkpoint"):
                    inference.resolve_checkpoint(reference)
                download.assert_not_called()

    def test_missing_local_checkpoint_has_clear_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "Local checkpoint not found"):
            inference.resolve_checkpoint("models/missing.pth")

    def test_checkpoint_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(IsADirectoryError, "must be a file"):
                inference.resolve_checkpoint(directory)

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

    def test_chunked_cpu_inference_preserves_samples_and_limits_chunk_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.wav"
            output_path = root / "output.wav"
            checkpoint_path = root / "checkpoint.bin"
            checkpoint_path.touch()

            sample_count = int(inference.SAMPLE_RATE * 1.8)
            samples = np.linspace(-0.5, 0.5, sample_count, dtype=np.float32)
            stereo = np.stack((samples, -samples), axis=1)
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
                    chunk_seconds=0.5,
                    overlap_seconds=0.1,
                )

            output, sample_rate = sf.read(
                output_path, dtype="float32", always_2d=True
            )
            self.assertEqual(device, torch.device("cpu"))
            self.assertEqual(sample_rate, inference.SAMPLE_RATE)
            self.assertEqual(output.shape, stereo.shape)
            np.testing.assert_allclose(output, stereo, rtol=0, atol=1e-6)
            self.assertGreater(len(model.input_lengths), 1)
            self.assertEqual(
                set(model.input_lengths), {int(inference.SAMPLE_RATE * 0.5)}
            )
            self.assertEqual(set(model.input_devices), {torch.device("cpu")})
            self.assertEqual(
                hashlib.sha256(input_path.read_bytes()).hexdigest(), input_digest
            )

    def test_short_input_uses_original_full_file_path(self):
        audio = torch.zeros(1, 2, inference.SAMPLE_RATE // 4)
        model = IdentityModel()

        output = inference.run_model(
            model,
            audio,
            torch.device("cpu"),
            chunk_samples=inference.SAMPLE_RATE,
            overlap_samples=inference.SAMPLE_RATE // 4,
        )

        self.assertEqual(model.input_lengths, [audio.shape[-1]])
        self.assertTrue(torch.equal(output, audio))

    def test_chunk_settings_reject_invalid_durations(self):
        invalid_settings = (
            (0, 0, "positive finite"),
            (float("inf"), 0, "positive finite"),
            (1, -0.1, "non-negative finite"),
            (1, float("nan"), "non-negative finite"),
            (1, 0.6, "half the chunk"),
        )
        for chunk_seconds, overlap_seconds, message in invalid_settings:
            with self.subTest(
                chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds
            ), self.assertRaisesRegex(ValueError, message):
                inference.resolve_chunking(chunk_seconds, overlap_seconds)

    def test_chunking_rejects_short_model_output(self):
        audio = torch.zeros(1, 2, 100)
        with self.assertRaisesRegex(RuntimeError, "shorter"):
            inference.run_model(
                ShortOutputModel(),
                audio,
                torch.device("cpu"),
                chunk_samples=60,
                overlap_samples=10,
            )

    def test_refuses_to_overwrite_input(self):
        path = Path("same.wav")
        with self.assertRaisesRegex(ValueError, "must be different"):
            inference.run_inference(path, path, "checkpoint.bin", "cpu")

    def test_wrong_sample_rate_fails_before_download_or_device_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.wav"
            sf.write(input_path, np.zeros((100, 2), dtype=np.float32), 48_000)

            with mock.patch.object(
                inference, "hf_hub_download"
            ) as download, mock.patch.object(torch.Tensor, "to") as transfer:
                with self.assertRaisesRegex(ValueError, "expects 44100 Hz"):
                    inference.run_inference(
                        input_path,
                        root / "output.wav",
                        inference.OFFICIAL_CHECKPOINT,
                        "cpu",
                    )
                download.assert_not_called()
                transfer.assert_not_called()

    def test_refuses_hard_link_output_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.wav"
            output_path = root / "output.wav"
            sf.write(
                input_path,
                np.zeros((100, 2), dtype=np.float32),
                inference.SAMPLE_RATE,
            )
            try:
                os.link(input_path, output_path)
            except OSError as error:
                self.skipTest(f"hard links are unavailable: {error}")

            with self.assertRaisesRegex(ValueError, "same file"):
                inference.run_inference(
                    input_path, output_path, "checkpoint.bin", "cpu"
                )


if __name__ == "__main__":
    unittest.main()
