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

Full-file inference is the default. To bound accelerator memory for longer
audio, opt in to chunking and an overlap crossfade:

```bash
python inference.py \
  --device=mps \
  --checkpoint=models/pytorch_model.bin \
  --in_wav=long_input.wav \
  --out_wav=long_output.wav \
  --chunk-seconds=6 \
  --overlap-seconds=1 \
  --chunk-batch-size=2
```

`--chunk-batch-size` controls how many chunks are transferred and processed in
one model forward pass. The default of `1` minimizes accelerator memory; larger
values keep memory bounded by a fixed batch but may trade additional memory for
throughput. The last chunk is padded for inference and cropped back to its
original size; normalized linear crossfades preserve the exact input frame
count. The overlap may be zero but must not exceed half the chunk duration.
Inputs no longer than the requested chunk use the original full-file path.

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

## Chunked validation reference

A separate MPS check concatenated the same public fixture twice to create a
12-second input, then compared full-file inference with 6-second chunks and a
1-second overlap. The derived test input SHA-256 was
`3e4a91f95975054bc092637176bb60bcb4bbd1469b8f8e92c14eb6e4ad0431c2`;
the repository fixture itself was not modified.

| Mode | End-to-end time | Peak memory footprint |
| --- | ---: | ---: |
| Full file | 10.22 s | 10.36 GiB |
| 6 s chunks, 1 s overlap | 12.59 s | 5.39 GiB |

The chunked run reduced the measured peak memory footprint by 48.0%. Both
outputs were float32 PCM, stereo, 44.1 kHz, exactly 12 seconds long, and finite.
Compared with full-file output, the chunked output had maximum absolute
difference `0.7703997492790222`, RMSE `0.027596083317049432`, and correlation
`0.9835697382563144`. At the two tested chunk starts, output sample differences
from the full-file result were at most `3.3527612686157227e-08`. These objective
checks do not establish that the two versions are perceptually equivalent.

The same public 12-second input was also used to compare chunk batch sizes on
MPS. In this Apple M3 run, batching did not improve MPS performance and should
therefore remain an explicit memory/throughput choice rather than a universal
speed claim:

| Chunk batch size | End-to-end time | Peak memory footprint |
| ---: | ---: | ---: |
| 1 | 11.76 s | 4.65 GiB |
| 2 | 12.28 s | 10.48 GiB |

Both outputs were float32 PCM, stereo, 44.1 kHz, exactly 12 seconds long, and
finite. Batch sizes 1 and 2 differed by at most `1.04308128356934e-06`, with
RMSE `3.84001256540259e-08` and correlation `0.999999999999966`. CUDA may have
different throughput characteristics, but it was not available for local
hardware validation.

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
- Chunking reduces accelerator memory demand but also limits model context and
  can produce a different result from full-file inference. Crossfades reduce
  boundary discontinuities but do not replace listening tests for seam or
  musical-consistency artifacts.
- Larger chunk batches can reduce the number of model forward passes but use
  more accelerator memory. They are not guaranteed to improve every backend;
  the tested MPS configuration was slower with batch size 2.
- A successful technical smoke test is not a listening-quality review and does
  not prove that information missing from a lossy source was truly recovered.
