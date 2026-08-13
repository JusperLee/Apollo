# Apple Silicon macOS inference

The upstream `look2hear.yml` records a Linux CUDA baseline with
`torch==2.0.0+cu118` and `torchaudio==2.0.1+cu118`; it cannot be installed
unchanged on macOS arm64. The MPS smoke test for this change used native arm64
Python 3.10, PyTorch 2.11.0, and TorchAudio 2.11.0. The inference entry point
uses SoundFile for WAV I/O so that it works with both the older baseline and
newer TorchAudio releases whose I/O depends on TorchCodec.

Create and activate an isolated environment, then install the macOS runtime
requirements:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-macos-arm64.txt
```

Run the repository's public six-second test WAV on MPS:

```bash
python inference.py \
  --device=mps \
  --checkpoint=models/pytorch_model.bin \
  --in_wav=asserts/input_wav.wav \
  --out_wav=apollo_mps_smoke.wav
```

If `--checkpoint` is omitted, inference downloads the official
`JusperLee/Apollo` `pytorch_model.bin` from Hugging Face. Custom checkpoints
must be existing local files; arbitrary remote repositories are rejected
because the upstream loader uses PyTorch deserialization. Only use local
checkpoints from sources you trust. The official checkpoint used for local
validation had this SHA-256 digest (recorded for reproducibility, not enforced
as a gate so that future official updates remain possible):

```text
99d9af7f1ff20e63c393035513a655392818d66b4d7fc23d658175c1f15e8d76
```

## Local validation reference

The public six-second WAV was tested on an Apple M3 with 24 GB unified memory
and macOS 27.0. These are reference measurements, not performance guarantees:

| Device | End-to-end time | Maximum RSS | Peak memory footprint |
| --- | ---: | ---: | ---: |
| MPS | 5.78 s | 412.0 MiB | 4.83 GiB |
| CPU | 316.46 s | 3.76 GiB | 3.27 GiB |

Both outputs were stereo, 44.1 kHz, six seconds long, and contained no NaN or
infinite values. Comparing the float32 CPU and MPS WAV outputs gave a maximum
absolute difference of `0.000249214470387`, RMSE of `2.02574076096e-05`, and
correlation of `0.999999991468`. The input SHA-256 remained unchanged:

```text
3c9a053913c3016b493ddc0b92e21a4e682e8fb5f872dafc945154155bddf771
```

## Device behavior

- `auto` preserves CUDA as the first choice, then selects MPS, then CPU.
- `cuda`, `mps`, and `cpu` explicitly select one backend and fail clearly if
  the requested accelerator is unavailable.
- The model and input are moved to the same device. Output is detached and
  moved to CPU before it is written.
- Apollo validates sample rate, basic shape, and finite samples on CPU before
  downloading the checkpoint or moving the input to an accelerator.
- Apollo expects 44.1 kHz audio. Input and output paths must not identify the
  same file, including through a hard link.

## Known limitations

- MPS validation covered the public stereo, 44.1 kHz, six-second test WAV on
  PyTorch 2.11.0. MPS operator support can vary with PyTorch and macOS versions.
- PyTorch 2.11.0 emitted non-fatal STFT/iSTFT output-resize deprecation warnings
  during the MPS run.
- CUDA remains the first `auto` choice and follows the original inference
  behavior, but CUDA was not hardware-tested on the Apple Silicon validation
  machine.
- Training and dataset evaluation still contain CUDA-specific configuration;
  this change only makes the public inference entry point device-agnostic.
- There is no long-audio chunking or crossfade in this change. Long inputs may
  require substantially more memory.
- A successful technical smoke test is not a listening-quality review and does
  not prove that information missing from a lossy source was truly recovered.
