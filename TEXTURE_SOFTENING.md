# Optional experimental texture softening

Apollo remains by Kai Li and Yi Luo / Look2Hear. This Neuracoust contribution
is adapted from Sound Build-up post-processing; the repository CC BY-SA 4.0
license and upstream attribution apply. It changes neither weights nor training.

```sh
python inference.py --in_wav=input.wav --out_wav=output.wav --texture-reduction-db=2
```

The option defaults to **0 (off)** and accepts 0–3 dB. When enabled, a
stereo-linked NumPy/SciPy processor runs once on the existing Apollo prediction,
after either full-file or chunked inference; it adds no model forward passes.
Install SciPy in the inference environment (`python -m pip install scipy`).
Enabled processing checks SciPy before audio/checkpoint loading and rejects
inputs shorter than 2048 frames before checkpoint/model loading. With the
option disabled, SciPy is not required and the existing short-input path remains.

The processor attenuates short positive spectral excursions against a roughly
40 ms temporal median, tapering between 1.2 and 14 kHz with a broad attack gate.
The cap bounds STFT gain reduction, not every time-domain sample difference.
This is a heuristic, not an artifact/vocal detector: wanted detail, modulation,
and transients can be suppressed. Keep this experimental and opt-in.

A full-output STFT makes memory **O(duration)** even with chunked model
inference. Historical model-excluded probes measured 388.74 MiB tracked peak /
648.77 MiB process RSS for 30 s, and 2328.58 / 3042.25 MiB for a 180 s repeated
Apollo prediction. The latter is an allocation probe, not a full-song listening
test; these host-dependent figures are not new measurements of this review copy.

Run `python -m unittest discover -s tests -v` for DSP and CLI regression tests.
The CLI tests mock model loading; they do not validate a checkpoint or listening
quality. [Existing texture-only audio bundle](https://github.com/hemanpapa-coder/Apollo/releases/tag/texture-evaluation-20261016)
and [fixed detailed evidence](https://github.com/hemanpapa-coder/Apollo/blob/f160a8a1e267626da2136fc20a487eef99f8cf28/evaluation/texture-20261016/REPORT.md)
provide eight aligned 15 s excerpts, raw 0/1/2/3 dB WAVs, loudness-matched blind
copies, commands and provenance. Sources are lossy proxies, not lossless clean
references; controlled listening and maintainer evaluation remain outstanding.
No universal quality improvement is claimed. The full enhancement chain is
outside this feature's scope and remains separate fork experiment material.
