import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download
import numpy as np
import soundfile as sf
import torch

import look2hear.models


SAMPLE_RATE = 44_100
DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")
OFFICIAL_CHECKPOINT = "JusperLee/Apollo"
LOCAL_CHECKPOINT_SUFFIXES = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}


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
    """Resolve the official remote checkpoint or an existing local file."""
    reference = str(reference)
    if reference == OFFICIAL_CHECKPOINT:
        return Path(
            hf_hub_download(
                repo_id=OFFICIAL_CHECKPOINT, filename="pytorch_model.bin"
            )
        )

    path = Path(reference).expanduser()
    if path.is_file():
        return path
    if path.exists():
        raise IsADirectoryError(f"Checkpoint must be a file, not a directory: {path}")
    if (
        path.suffix.lower() in LOCAL_CHECKPOINT_SUFFIXES
        or path.is_absolute()
        or reference.startswith(("./", "../", "~/"))
    ):
        raise FileNotFoundError(f"Local checkpoint not found: {path}")
    raise ValueError(
        "Unsupported checkpoint source. Use JusperLee/Apollo or an existing "
        "local checkpoint file."
    )


def load_audio(file_path):
    audio, sample_rate = sf.read(
        file_path, dtype="float32", always_2d=True
    )
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Apollo expects {SAMPLE_RATE} Hz audio, got {sample_rate} Hz."
        )
    if audio.shape[0] == 0 or audio.shape[1] == 0:
        raise ValueError("Input audio must contain at least one sample and channel.")
    if not np.isfinite(audio).all():
        raise ValueError("Input audio contains NaN or infinite values.")
    # SoundFile uses [samples, channels]; Apollo uses [batch, channels, samples].
    audio = torch.from_numpy(np.ascontiguousarray(audio.T)).unsqueeze(0)
    return audio, sample_rate


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
    if output_path.exists() and input_path.samefile(output_path):
        raise ValueError("Input and output paths must not reference the same file.")

    device = select_device(requested_device)
    test_data, sample_rate = load_audio(input_path)
    checkpoint_path = resolve_checkpoint(checkpoint)
    test_data = test_data.to(device)

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
        default=OFFICIAL_CHECKPOINT,
        help="JusperLee/Apollo or an existing local checkpoint file",
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
