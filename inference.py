import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

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


def resolve_chunking(chunk_seconds, overlap_seconds):
    """Convert optional chunk settings to samples and validate crossfades."""
    if chunk_seconds is None:
        return None, 0
    if not np.isfinite(chunk_seconds) or chunk_seconds <= 0:
        raise ValueError("Chunk duration must be a positive finite number.")
    if not np.isfinite(overlap_seconds) or overlap_seconds < 0:
        raise ValueError("Overlap duration must be a non-negative finite number.")

    chunk_samples = int(round(chunk_seconds * SAMPLE_RATE))
    overlap_samples = int(round(overlap_seconds * SAMPLE_RATE))
    if chunk_samples < 1:
        raise ValueError("Chunk duration is shorter than one sample.")
    if overlap_samples * 2 > chunk_samples:
        raise ValueError("Overlap duration must not exceed half the chunk duration.")
    return chunk_samples, overlap_samples


def chunk_starts(total_samples, chunk_samples, overlap_samples):
    """Return starts that cover the signal without a redundant final chunk."""
    hop_samples = chunk_samples - overlap_samples
    starts = [0]
    while starts[-1] + chunk_samples < total_samples:
        starts.append(starts[-1] + hop_samples)
    return starts


def crossfade_weights(length, overlap_samples, fade_in, fade_out, dtype):
    """Create linear edge weights for normalized overlap-add."""
    weights = torch.ones(length, dtype=dtype)
    fade_samples = min(overlap_samples, length)
    if fade_samples:
        ramp = torch.linspace(0.0, 1.0, fade_samples, dtype=dtype)
        if fade_in:
            weights[:fade_samples] = ramp
        if fade_out:
            weights[-fade_samples:] = torch.flip(ramp, dims=(0,))
    return weights.view(1, 1, -1)


def run_model(model, audio, device, chunk_samples=None, overlap_samples=0):
    """Run full-file or normalized overlap-add chunked inference."""
    total_samples = audio.shape[-1]
    if chunk_samples is None or total_samples <= chunk_samples:
        return model(audio.to(device)).detach().to("cpu")

    output_sum = torch.zeros_like(audio, device="cpu")
    weight_sum = torch.zeros((1, 1, total_samples), dtype=audio.dtype)
    starts = chunk_starts(total_samples, chunk_samples, overlap_samples)

    for index, start in enumerate(starts):
        end = min(start + chunk_samples, total_samples)
        valid_samples = end - start
        chunk = audio[..., start:end]
        if valid_samples < chunk_samples:
            chunk = F.pad(chunk, (0, chunk_samples - valid_samples))

        chunk_output = model(chunk.to(device)).detach().to("cpu")
        if chunk_output.ndim != 3 or chunk_output.shape[:2] != audio.shape[:2]:
            raise RuntimeError(
                "Chunk output must preserve the input batch and channel dimensions."
            )
        if chunk_output.shape[-1] < valid_samples:
            raise RuntimeError(
                "Chunk output is shorter than the corresponding input segment."
            )
        chunk_output = chunk_output[..., :valid_samples]

        weights = crossfade_weights(
            valid_samples,
            overlap_samples,
            fade_in=index > 0,
            fade_out=index < len(starts) - 1,
            dtype=audio.dtype,
        )
        output_sum[..., start:end] += chunk_output * weights
        weight_sum[..., start:end] += weights

    if torch.any(weight_sum <= 0):
        raise RuntimeError("Chunk overlap-add left uncovered output samples.")
    return output_sum / weight_sum


def run_inference(
    input_wav,
    output_wav,
    checkpoint,
    requested_device="auto",
    chunk_seconds=None,
    overlap_seconds=1.0,
):
    input_path = Path(input_wav)
    output_path = Path(output_wav)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")
    if output_path.exists() and input_path.samefile(output_path):
        raise ValueError("Input and output paths must not reference the same file.")

    chunk_samples, overlap_samples = resolve_chunking(
        chunk_seconds, overlap_seconds
    )
    device = select_device(requested_device)
    test_data, sample_rate = load_audio(input_path)
    checkpoint_path = resolve_checkpoint(checkpoint)

    model = look2hear.models.BaseModel.from_pretrain(
        str(checkpoint_path),
        sr=SAMPLE_RATE,
        win=20,
        feature_dim=256,
        layer=6,
    ).to(device).eval()
    with torch.inference_mode():
        output = run_model(
            model,
            test_data,
            device,
            chunk_samples=chunk_samples,
            overlap_samples=overlap_samples,
        )
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
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=None,
        help="Optional chunk duration for long audio; disabled by default",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=1.0,
        help="Chunk crossfade duration (default: 1.0; at most half a chunk)",
    )
    args = parser.parse_args()

    device = run_inference(
        args.in_wav,
        args.out_wav,
        args.checkpoint,
        args.device,
        args.chunk_seconds,
        args.overlap_seconds,
    )
    print(f"Inference completed on {device}: {args.out_wav}")


if __name__ == "__main__":
    main()
