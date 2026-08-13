import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download
import numpy as np
import soundfile as sf
import torch

import look2hear.models


SAMPLE_RATE = 44_100
DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def select_device(requested):
    """Resolve an inference device, preferring CUDA, then MPS, then CPU."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(requested)


def resolve_checkpoint(reference):
    """Resolve a local checkpoint path or download one from Hugging Face."""
    path = Path(reference).expanduser()
    if path.is_file():
        return path
    if path.suffix:
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return Path(
        hf_hub_download(repo_id=reference, filename="pytorch_model.bin")
    )


def load_audio(file_path, device):
    audio, sample_rate = sf.read(
        file_path, dtype="float32", always_2d=True
    )
    # SoundFile uses [samples, channels]; Apollo uses [batch, channels, samples].
    audio = torch.from_numpy(np.ascontiguousarray(audio.T)).unsqueeze(0)
    return audio.to(device), sample_rate


def save_audio(file_path, audio, sample_rate):
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio.detach().squeeze(0).to("cpu").numpy().T
    sf.write(output_path, audio, sample_rate, subtype="FLOAT")


def run_inference(input_wav, output_wav, checkpoint, requested_device="auto"):
    input_path = Path(input_wav)
    output_path = Path(output_wav)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")

    device = select_device(requested_device)
    checkpoint_path = resolve_checkpoint(checkpoint)
    test_data, sample_rate = load_audio(input_path, device)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Apollo expects {SAMPLE_RATE} Hz audio, got {sample_rate} Hz."
        )

    model = look2hear.models.BaseModel.from_pretrain(
        str(checkpoint_path),
        sr=SAMPLE_RATE,
        win=20,
        feature_dim=256,
        layer=6,
    ).to(device).eval()
    with torch.inference_mode():
        output = model(test_data)
    save_audio(output_path, output, sample_rate)
    return device


def main():
    parser = argparse.ArgumentParser(description="Audio Inference Script")
    parser.add_argument(
        "--in_wav", type=Path, required=True, help="Path to input WAV file"
    )
    parser.add_argument(
        "--out_wav", type=Path, required=True, help="Path to output WAV file"
    )
    parser.add_argument(
        "--checkpoint",
        default="JusperLee/Apollo",
        help="Local checkpoint path or Hugging Face repository ID",
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="Inference device (auto prefers CUDA, then MPS, then CPU)",
    )
    args = parser.parse_args()

    device = run_inference(
        args.in_wav, args.out_wav, args.checkpoint, args.device
    )
    print(f"Inference completed on {device}: {args.out_wav}")


if __name__ == "__main__":
    main()
