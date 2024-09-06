# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as tf
import librosa.filters as filters

from typing import Optional, Tuple
from distutils.version import LooseVersion

EPSILON = float(np.finfo(np.float32).eps)
TORCH_VERSION = th.__version__

if TORCH_VERSION >= LooseVersion("1.7"):
    from torch.fft import fft as fft_func
else:
    pass


def export_jit(transform: nn.Module) -> nn.Module:
    """
    Export transform module for inference
    """
    export_out = [module for module in transform if module.exportable()]
    return nn.Sequential(*export_out)


def init_window(wnd: str, frame_len: int, device: th.device = "cpu") -> th.Tensor:
    """
    Return window coefficient
    Args:
        wnd: window name
        frame_len: length of the frame
    """

    def sqrthann(frame_len, periodic=True):
        return th.hann_window(frame_len, periodic=periodic) ** 0.5

    if wnd not in ["bartlett", "hann", "hamm", "blackman", "rect", "sqrthann"]:
        raise RuntimeError(f"Unknown window type: {wnd}")

    wnd_tpl = {
        "sqrthann": sqrthann,
        "hann": th.hann_window,
        "hamm": th.hamming_window,
        "blackman": th.blackman_window,
        "bartlett": th.bartlett_window,
        "rect": th.ones,
    }
    if wnd != "rect":
        # match with librosa
        c = wnd_tpl[wnd](frame_len, periodic=True)
    else:
        c = wnd_tpl[wnd](frame_len)
    return c.to(device)


def init_kernel(
    frame_len: int,
    frame_hop: int,
    window: th.Tensor,
    round_pow_of_two: bool = True,
    normalized: bool = False,
    inverse: bool = False,
    mode: str = "librosa",
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Return STFT kernels
    Args:
        frame_len: length of the frame
        frame_hop: hop size between frames
        window: window tensor
        round_pow_of_two: if true, choose round(#power_of_two) as the FFT size
        normalized: return normalized DFT matrix
        inverse: return iDFT matrix
        mode: framing mode (librosa or kaldi)
    """
    if mode not in ["librosa", "kaldi"]:
        raise ValueError(f"Unsupported mode: {mode}")
    # FFT size: B
    if round_pow_of_two or mode == "kaldi":
        fft_size = 2 ** math.ceil(math.log2(frame_len))
    else:
        fft_size = frame_len
    # center padding window if needed
    if mode == "librosa" and fft_size != frame_len:
        lpad = (fft_size - frame_len) // 2
        window = tf.pad(window, (lpad, fft_size - frame_len - lpad))
    if normalized:
        # make K^H * K = I
        S = fft_size ** 0.5
    else:
        S = 1
    # W x B x 2
    if TORCH_VERSION >= LooseVersion("1.7"):
        K = fft_func(th.eye(fft_size) / S, dim=-1)
        K = th.stack([K.real, K.imag], dim=-1)
    else:
        I = th.stack([th.eye(fft_size), th.zeros(fft_size, fft_size)], dim=-1)
        K = th.fft(I / S, 1)
    if mode == "kaldi":
        K = K[:frame_len]
    if inverse and not normalized:
        # to make K^H * K = I
        K = K / fft_size
    # 2 x B x W
    K = th.transpose(K, 0, 2)
    # 2B x 1 x W
    K = th.reshape(K, (fft_size * 2, 1, K.shape[-1]))
    return K.to(window.device), window


def mel_filter(
    frame_len: int,
    round_pow_of_two: bool = True,
    num_bins: Optional[int] = None,
    sr: int = 16000,
    num_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    norm: bool = False,
) -> th.Tensor:
    """
    Return mel filter coefficients
    Args:
        frame_len: length of the frame
        round_pow_of_two: if true, choose round(#power_of_two) as the FFT size
        num_bins: number of the frequency bins produced by STFT
        num_mels: number of the mel bands
        fmin: lowest frequency (in Hz)
        fmax: highest frequency (in Hz)
        norm: normalize the mel filter coefficients
    """
    # FFT points
    if num_bins is None:
        N = 2 ** math.ceil(math.log2(frame_len)) if round_pow_of_two else frame_len
    else:
        N = (num_bins - 1) * 2
    # fmin & fmax
    freq_upper = sr // 2
    if fmax is None:
        fmax = freq_upper
    else:
        fmax = min(fmax + freq_upper if fmax < 0 else fmax, freq_upper)
    fmin = max(0, fmin)
    # mel filter coefficients
    mel = filters.mel(
        sr,
        N,
        n_mels=num_mels,
        fmax=fmax,
        fmin=fmin,
        htk=True,
        norm="slaney" if norm else None,
    )
    # num_mels x (N // 2 + 1)
    return th.tensor(mel, dtype=th.float32)


def speed_perturb_filter(
    src_sr: int, dst_sr: int, cutoff_ratio: float = 0.95, num_zeros: int = 64
) -> th.Tensor:
    """
    Return speed perturb filters, reference:
        https://github.com/danpovey/filtering/blob/master/lilfilter/resampler.py
    Args:
        src_sr: sample rate of the source signal
        dst_sr: sample rate of the target signal
    Return:
        weight (Tensor): coefficients of the filter
    """
    if src_sr == dst_sr:
        raise ValueError(f"src_sr should not be equal to dst_sr: {src_sr}/{dst_sr}")
    gcd = math.gcd(src_sr, dst_sr)
    src_sr = src_sr // gcd
    dst_sr = dst_sr // gcd
    if src_sr == 1 or dst_sr == 1:
        raise ValueError("do not support integer downsample/upsample")
    zeros_per_block = min(src_sr, dst_sr) * cutoff_ratio
    padding = 1 + int(num_zeros / zeros_per_block)
    # dst_sr x src_sr x K
    times = (
        np.arange(dst_sr)[:, None, None] / float(dst_sr)
        - np.arange(src_sr)[None, :, None] / float(src_sr)
        - np.arange(2 * padding + 1)[None, None, :]
        + padding
    )
    window = np.heaviside(1 - np.abs(times / padding), 0.0) * (
        0.5 + 0.5 * np.cos(times / padding * math.pi)
    )
    weight = np.sinc(times * zeros_per_block) * window * zeros_per_block / float(src_sr)
    return th.tensor(weight, dtype=th.float32)


def splice_feature(
    feats: th.Tensor, lctx: int = 1, rctx: int = 1, op: str = "cat"
) -> th.Tensor:
    """
    Splice feature
    Args:
        feats (Tensor): N x ... x T x F, original feature
        lctx: left context
        rctx: right context
        op: operator on feature context
    Return:
        splice (Tensor): feature with context padded
    """
    if lctx + rctx == 0:
        return feats
    if op not in ["cat", "stack"]:
        raise ValueError(f"Unknown op for feature splicing: {op}")
    # [N x ... x T x F, ...]
    ctx = []
    T = feats.shape[-2]
    for c in range(-lctx, rctx + 1):
        idx = th.arange(c, c + T, device=feats.device, dtype=th.int64)
        idx = th.clamp(idx, min=0, max=T - 1)
        ctx.append(th.index_select(feats, -2, idx))
    if op == "cat":
        # N x ... x T x FD
        splice = th.cat(ctx, -1)
    else:
        # N x ... x T x F x D
        splice = th.stack(ctx, -1)
    return splice


def _forward_stft(
    wav: th.Tensor,
    kernel: th.Tensor,
    window: th.Tensor,
    return_polar: bool = False,
    pre_emphasis: float = 0,
    frame_hop: int = 256,
    onesided: bool = False,
    center: bool = False,
    eps: float = EPSILON,
) -> th.Tensor:
    """
    STFT function implemented by conv1d (not efficient, but we don't care during training)
    Args:
        wav (Tensor): N x (C) x S
        kernel (Tensor): STFT transform kernels, from init_kernel(...)
        return_polar: return [magnitude; phase] Tensor or [real; imag] Tensor
        pre_emphasis: factor of preemphasis
        frame_hop: frame hop size in number samples
        onesided: return half FFT bins
        center: if true, we assumed to have centered frames
    Return:
        transform (Tensor): STFT transform results
    """
    wav_dim = wav.dim()
    if wav_dim not in [2, 3]:
        raise RuntimeError(f"STFT expect 2D/3D tensor, but got {wav_dim:d}D")
    # if N x S, reshape N x 1 x S
    # else: reshape NC x 1 x S
    N, S = wav.shape[0], wav.shape[-1]
    wav = wav.view(-1, 1, S)
    # NC x 1 x S+2P
    if center:
        pad = kernel.shape[-1] // 2
        # NOTE: match with librosa
        wav = tf.pad(wav, (pad, pad), mode="reflect")
    # STFT
    kernel = kernel * window
    if pre_emphasis > 0:
        # NC x W x T
        frames = tf.unfold(
            wav[:, None], (1, kernel.shape[-1]), stride=frame_hop, padding=0
        )
        # follow Kaldi's Preemphasize
        frames[:, 1:] = frames[:, 1:] - pre_emphasis * frames[:, :-1]
        frames[:, 0] *= 1 - pre_emphasis
        # 1 x 2B x W, NC x W x T,  NC x 2B x T
        packed = th.matmul(kernel[:, 0][None, ...], frames)
    else:
        packed = tf.conv1d(wav, kernel, stride=frame_hop, padding=0)
    # NC x 2B x T => N x C x 2B x T
    if wav_dim == 3:
        packed = packed.view(N, -1, packed.shape[-2], packed.shape[-1])
    # N x (C) x B x T
    real, imag = th.chunk(packed, 2, dim=-2)
    # N x (C) x B/2+1 x T
    if onesided:
        num_bins = kernel.shape[0] // 4 + 1
        real = real[..., :num_bins, :]
        imag = imag[..., :num_bins, :]
    if return_polar:
        mag = (real ** 2 + imag ** 2 + eps) ** 0.5
        pha = th.atan2(imag, real)
        return th.stack([mag, pha], dim=-1)
    else:
        return th.stack([real, imag], dim=-1)


def _inverse_stft(
    transform: th.Tensor,
    kernel: th.Tensor,
    window: th.Tensor,
    return_polar: bool = False,
    frame_hop: int = 256,
    onesided: bool = False,
    center: bool = False,
    eps: float = EPSILON,
) -> th.Tensor:
    """
    iSTFT function implemented by conv1d
    Args:
        transform (Tensor): STFT transform results
        kernel (Tensor): STFT transform kernels, from init_kernel(...)
        return_polar (bool): keep same with the one in _forward_stft
        frame_hop: frame hop size in number samples
        onesided: return half FFT bins
        center: used in _forward_stft
    Return:
        wav (Tensor), N x S
    """
    # (N) x F x T x 2
    transform_dim = transform.dim()
    # if F x T x 2, reshape 1 x F x T x 2
    if transform_dim == 3:
        transform = th.unsqueeze(transform, 0)
    if transform_dim != 4:
        raise RuntimeError(f"Expect 4D tensor, but got {transform_dim}D")

    if return_polar:
        real = transform[..., 0] * th.cos(transform[..., 1])
        imag = transform[..., 0] * th.sin(transform[..., 1])
    else:
        real, imag = transform[..., 0], transform[..., 1]

    if onesided:
        # [self.num_bins - 2, ..., 1]
        reverse = range(kernel.shape[0] // 4 - 1, 0, -1)
        # extend matrix: N x B x T
        real = th.cat([real, real[:, reverse]], 1)
        imag = th.cat([imag, -imag[:, reverse]], 1)
    # pack: N x 2B x T
    packed = th.cat([real, imag], dim=1)
    # N x 1 x T
    wav = tf.conv_transpose1d(packed, kernel * window, stride=frame_hop, padding=0)
    # normalized audio samples
    # refer: https://github.com/pytorch/audio/blob/2ebbbf511fb1e6c47b59fd32ad7e66023fa0dff1/torchaudio/functional.py#L171
    num_frames = packed.shape[-1]
    win_length = window.shape[0]
    # W x T
    win = th.repeat_interleave(window[..., None] ** 2, num_frames, dim=-1)
    # Do OLA on windows
    # v1)
    I = th.eye(win_length, device=win.device)[:, None]
    denorm = tf.conv_transpose1d(win[None, ...], I, stride=frame_hop, padding=0)
    # v2)
    # num_samples = (num_frames - 1) * frame_hop + win_length
    # denorm = tf.fold(win[None, ...], (num_samples, 1), (win_length, 1),
    #                  stride=frame_hop)[..., 0]
    if center:
        pad = kernel.shape[-1] // 2
        wav = wav[..., pad:-pad]
        denorm = denorm[..., pad:-pad]
    wav = wav / (denorm + eps)
    # N x S
    return wav.squeeze(1)


def _pytorch_stft(
    wav: th.Tensor,
    frame_len: int,
    frame_hop: int,
    n_fft: int = 512,
    return_polar: bool = False,
    window: str = "sqrthann",
    normalized: bool = False,
    onesided: bool = True,
    center: bool = False,
    eps: float = EPSILON,
) -> th.Tensor:
    """
    Wrapper of PyTorch STFT function
    Args:
        wav (Tensor): source audio signal
        frame_len: length of the frame
        frame_hop: hop size between frames
        n_fft: number of the FFT size
        return_polar: return the results in polar coordinate
        window: window tensor
        center: same definition with the parameter in librosa.stft
        normalized: use normalized DFT kernel
        onesided: output onesided STFT
    Return:
        transform (Tensor), STFT transform results
    """
    if TORCH_VERSION < LooseVersion("1.7"):
        raise RuntimeError("Can not use this function as TORCH_VERSION < 1.7")
    wav_dim = wav.dim()
    if wav_dim not in [2, 3]:
        raise RuntimeError(f"STFT expect 2D/3D tensor, but got {wav_dim:d}D")
    # if N x C x S, reshape NC x S
    wav = wav.view(-1, wav.shape[-1])
    # STFT: N x F x T x 2
    stft = th.stft(
        wav,
        n_fft,
        hop_length=frame_hop,
        win_length=window.shape[-1],
        window=window,
        center=center,
        normalized=normalized,
        onesided=onesided,
        return_complex=False,
    )
    if wav_dim == 3:
        N, F, T, _ = stft.shape
        stft = stft.view(N, -1, F, T, 2)
    # N x (C) x F x T x 2
    if not return_polar:
        return stft
    # N x (C) x F x T
    real, imag = stft[..., 0], stft[..., 1]
    mag = (real ** 2 + imag ** 2 + eps) ** 0.5
    pha = th.atan2(imag, real)
    return th.stack([mag, pha], dim=-1)


def _pytorch_istft(
    transform: th.Tensor,
    frame_len: int,
    frame_hop: int,
    window: th.Tensor,
    n_fft: int = 512,
    return_polar: bool = False,
    normalized: bool = False,
    onesided: bool = True,
    center: bool = False,
    eps: float = EPSILON,
) -> th.Tensor:
    """
    Wrapper of PyTorch iSTFT function
    Args:
        transform (Tensor): results of STFT
        frame_len: length of the frame
        frame_hop: hop size between frames
        window: window tensor
        n_fft: number of the FFT size
        return_polar: keep same with _pytorch_stft
        center: same definition with the parameter in librosa.stft
        normalized: use normalized DFT kernel
        onesided: output onesided STFT
    Return:
        wav (Tensor): synthetic audio
    """
    if TORCH_VERSION < LooseVersion("1.7"):
        raise RuntimeError("Can not use this function as TORCH_VERSION < 1.7")

    transform_dim = transform.dim()
    # if F x T x 2, reshape 1 x F x T x 2
    if transform_dim == 3:
        transform = th.unsqueeze(transform, 0)
    if transform_dim != 4:
        raise RuntimeError(f"Expect 4D tensor, but got {transform_dim}D")

    if return_polar:
        real = transform[..., 0] * th.cos(transform[..., 1])
        imag = transform[..., 0] * th.sin(transform[..., 1])
        transform = th.stack([real, imag], -1)
    # stft is a complex tensor of PyTorch
    stft = th.view_as_complex(transform)
    # (N) x S
    wav = th.istft(
        stft,
        n_fft,
        hop_length=frame_hop,
        win_length=window.shape[-1],
        window=window,
        center=center,
        normalized=normalized,
        onesided=onesided,
        return_complex=False,
    )
    return wav


def forward_stft(
    wav: th.Tensor,
    frame_len: int,
    frame_hop: int,
    window: str = "sqrthann",
    round_pow_of_two: bool = True,
    return_polar: bool = False,
    pre_emphasis: float = 0,
    normalized: bool = False,
    onesided: bool = True,
    center: bool = False,
    mode: str = "librosa",
    eps: float = EPSILON,
) -> th.Tensor:
    """
    STFT function implementation, equals to STFT layer
    Args:
        wav: source audio signal
        frame_len: length of the frame
        frame_hop: hop size between frames
        return_polar: return [magnitude; phase] Tensor or [real; imag] Tensor
        window: window name
        center: center flag (similar with that in librosa.stft)
        round_pow_of_two: if true, choose round(#power_of_two) as the FFT size
        pre_emphasis: factor of preemphasis
        normalized: use normalized DFT kernel
        onesided: output onesided STFT
        inverse: using iDFT kernel (for iSTFT)
        mode: STFT mode, "kaldi" or "librosa" or "torch"
    Return:
        transform: results of STFT
    """
    window = init_window(window, frame_len, device=wav.device)
    if mode == "torch":
        n_fft = 2 ** math.ceil(math.log2(frame_len)) if round_pow_of_two else frame_len
        return _pytorch_stft(
            wav,
            frame_len,
            frame_hop,
            n_fft=n_fft,
            return_polar=return_polar,
            window=window,
            normalized=normalized,
            onesided=onesided,
            center=center,
            eps=eps,
        )
    else:
        kernel, window = init_kernel(
            frame_len,
            frame_hop,
            window=window,
            round_pow_of_two=round_pow_of_two,
            normalized=normalized,
            inverse=False,
            mode=mode,
        )
        return _forward_stft(
            wav,
            kernel,
            window,
            return_polar=return_polar,
            frame_hop=frame_hop,
            pre_emphasis=pre_emphasis,
            onesided=onesided,
            center=center,
            eps=eps,
        )


def inverse_stft(
    transform: th.Tensor,
    frame_len: int,
    frame_hop: int,
    return_polar: bool = False,
    window: str = "sqrthann",
    round_pow_of_two: bool = True,
    normalized: bool = False,
    onesided: bool = True,
    center: bool = False,
    mode: str = "librosa",
    eps: float = EPSILON,
) -> th.Tensor:
    """
    iSTFT function implementation, equals to iSTFT layer
    Args:
        transform: results of STFT
        frame_len: length of the frame
        frame_hop: hop size between frames
        return_polar: keep same with function forward_stft(...)
        window: window name
        center: center flag (similar with that in librosa.stft)
        round_pow_of_two: if true, choose round(#power_of_two) as the FFT size
        normalized: use normalized DFT kernel
        onesided: output onesided STFT
        mode: STFT mode, "kaldi" or "librosa" or "torch"
    Return:
        wav: synthetic signals
    """
    window = init_window(window, frame_len, device=transform.device)
    if mode == "torch":
        n_fft = 2 ** math.ceil(math.log2(frame_len)) if round_pow_of_two else frame_len
        return _pytorch_istft(
            transform,
            frame_len,
            frame_hop,
            n_fft=n_fft,
            return_polar=return_polar,
            window=window,
            normalized=normalized,
            onesided=onesided,
            center=center,
            eps=eps,
        )
    else:
        kernel, window = init_kernel(
            frame_len,
            frame_hop,
            window,
            round_pow_of_two=round_pow_of_two,
            normalized=normalized,
            inverse=True,
            mode=mode,
        )
        return _inverse_stft(
            transform,
            kernel,
            window,
            return_polar=return_polar,
            frame_hop=frame_hop,
            onesided=onesided,
            center=center,
            eps=eps,
        )


class STFTBase(nn.Module):
    """
    Base layer for (i)STFT
    Args:
        frame_len: length of the frame
        frame_hop: hop size between frames
        window: window name
        center: center flag (similar with that in librosa.stft)
        round_pow_of_two: if true, choose round(#power_of_two) as the FFT size
        normalized: use normalized DFT kernel
        pre_emphasis: factor of preemphasis
        mode: STFT mode, "kaldi" or "librosa" or "torch"
        onesided: output onesided STFT
        inverse: using iDFT kernel (for iSTFT)
    """

    def __init__(
        self,
        frame_len: int,
        frame_hop: int,
        window: str = "sqrthann",
        round_pow_of_two: bool = True,
        normalized: bool = False,
        pre_emphasis: float = 0,
        onesided: bool = True,
        inverse: bool = False,
        center: bool = False,
        mode: str = "librosa",
    ) -> None:
        super(STFTBase, self).__init__()
        if mode != "torch":
            K, w = init_kernel(
                frame_len,
                frame_hop,
                init_window(window, frame_len),
                round_pow_of_two=round_pow_of_two,
                normalized=normalized,
                inverse=inverse,
                mode=mode,
            )
            self.K = nn.Parameter(K, requires_grad=False)
            self.w = nn.Parameter(w, requires_grad=False)
            self.num_bins = self.K.shape[0] // 4 + 1
            self.pre_emphasis = pre_emphasis
            self.win_length = self.K.shape[2]
        else:
            self.K = None
            w = init_window(window, frame_len)
            self.w = nn.Parameter(w, requires_grad=False)
            fft_size = (
                2 ** math.ceil(math.log2(frame_len)) if round_pow_of_two else frame_len
            )
            self.num_bins = fft_size // 2 + 1
            self.pre_emphasis = 0
            self.win_length = fft_size
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.window = window
        self.normalized = normalized
        self.onesided = onesided
        self.center = center
        self.mode = mode

    def num_frames(self, wav_len: th.Tensor) -> th.Tensor:
        """
        Compute number of the frames
        """
        assert th.sum(wav_len <= self.win_length) == 0
        if self.center:
            wav_len += self.win_length
        return (
            th.div(wav_len - self.win_length, self.frame_hop, rounding_mode="trunc") + 1
        )

    def extra_repr(self) -> str:
        str_repr = (
            f"num_bins={self.num_bins}, win_length={self.win_length}, "
            + f"stride={self.frame_hop}, window={self.window}, "
            + f"center={self.center}, mode={self.mode}"
        )
        if not self.onesided:
            str_repr += f", onesided={self.onesided}"
        if self.pre_emphasis > 0:
            str_repr += f", pre_emphasis={self.pre_emphasis}"
        if self.normalized:
            str_repr += f", normalized={self.normalized}"
        return str_repr


class STFT(STFTBase):
    """
    Short-time Fourier Transform as a Layer
    """

    def __init__(self, *args, **kwargs):
        super(STFT, self).__init__(*args, inverse=False, **kwargs)

    def forward(
        self, wav: th.Tensor, return_polar: bool = False, eps: float = EPSILON
    ) -> th.Tensor:
        """
        Accept (single or multiple channel) raw waveform and output magnitude and phase
        Args
            wav (Tensor) input signal, N x (C) x S
        Return
            transform (Tensor), N x (C) x F x T x 2
        """
        if self.mode == "torch":
            return _pytorch_stft(
                wav,
                self.frame_len,
                self.frame_hop,
                n_fft=(self.num_bins - 1) * 2,
                return_polar=return_polar,
                window=self.w,
                normalized=self.normalized,
                onesided=self.onesided,
                center=self.center,
                eps=eps,
            )
        else:
            return _forward_stft(
                wav,
                self.K,
                self.w,
                return_polar=return_polar,
                frame_hop=self.frame_hop,
                pre_emphasis=self.pre_emphasis,
                onesided=self.onesided,
                center=self.center,
                eps=eps,
            )


class iSTFT(STFTBase):
    """
    Inverse Short-time Fourier Transform as a Layer
    """

    def __init__(self, *args, **kwargs):
        super(iSTFT, self).__init__(*args, inverse=True, **kwargs)

    def forward(
        self, transform: th.Tensor, return_polar: bool = False, eps: float = EPSILON
    ) -> th.Tensor:
        """
        Accept phase & magnitude and output raw waveform
        Args
            transform (Tensor): STFT output, N x F x T x 2
        Return
            s (Tensor): N x S
        """
        if self.mode == "torch":
            return _pytorch_istft(
                transform,
                self.frame_len,
                self.frame_hop,
                n_fft=(self.num_bins - 1) * 2,
                return_polar=return_polar,
                window=self.w,
                normalized=self.normalized,
                onesided=self.onesided,
                center=self.center,
                eps=eps,
            )
        else:
            return _inverse_stft(
                transform,
                self.K,
                self.w,
                return_polar=return_polar,
                frame_hop=self.frame_hop,
                onesided=self.onesided,
                center=self.center,
                eps=eps,
            )
