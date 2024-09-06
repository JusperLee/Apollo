###
# Author: Kai Li
# Date: 2021-06-18 16:32:50
# LastEditors: Kai Li
# LastEditTime: 2021-06-19 01:02:04
###
import os
import warnings
import torch
import numpy as np
import soundfile as sf


def get_device(tensor_or_module, default=None):
    if hasattr(tensor_or_module, "device"):
        return tensor_or_module.device
    elif hasattr(tensor_or_module, "parameters"):
        return next(tensor_or_module.parameters()).device
    elif default is None:
        raise TypeError(
            f"Don't know how to get device of {type(tensor_or_module)} object"
        )
    else:
        return torch.device(default)


class Separator:
    def forward_wav(self, wav, **kwargs):
        raise NotImplementedError

    def sample_rate(self):
        raise NotImplementedError


def separate(model, wav, **kwargs):
    if isinstance(wav, np.ndarray):
        return numpy_separate(model, wav, **kwargs)
    elif isinstance(wav, torch.Tensor):
        return torch_separate(model, wav, **kwargs)
    else:
        raise ValueError(
            f"Only support filenames, numpy arrays and torch tensors, received {type(wav)}"
        )


@torch.no_grad()
def torch_separate(model: Separator, wav: torch.Tensor, **kwargs) -> torch.Tensor:
    """Core logic of `separate`."""
    if model.in_channels is not None and wav.shape[-2] != model.in_channels:
        raise RuntimeError(
            f"Model supports {model.in_channels}-channel inputs but found audio with {wav.shape[-2]} channels."
            f"Please match the number of channels."
        )
    # Handle device placement
    input_device = get_device(wav, default="cpu")
    model_device = get_device(model, default="cpu")
    wav = wav.to(model_device)
    # Forward
    separate_func = getattr(model, "forward_wav", model)
    out_wavs = separate_func(wav, **kwargs)

    # FIXME: for now this is the best we can do.
    out_wavs *= wav.abs().sum() / (out_wavs.abs().sum())

    # Back to input device (and numpy if necessary)
    out_wavs = out_wavs.to(input_device)
    return out_wavs


def numpy_separate(model: Separator, wav: np.ndarray, **kwargs) -> np.ndarray:
    """Numpy interface to `separate`."""
    wav = torch.from_numpy(wav)
    out_wavs = torch_separate(model, wav, **kwargs)
    out_wavs = out_wavs.data.numpy()
    return out_wavs


def wav_chunk_inference(model, mixture_tensor, sr=16000, target_length=12.0, hop_length=4.0, batch_size=10, n_tracks=3):
    """
    Input:
        mixture_tensor: Tensor, [nch, input_length]
        
    Output:
        all_target_tensor: Tensor, [nch, n_track, input_length]    
    """
    batch_mixture = mixture_tensor

    # split data into segments
    batch_length = batch_mixture.shape[-1]

    session = int(sr * target_length)
    target = int(sr * target_length)
    ignore = (session - target) // 2
    hop = int(sr * hop_length)
    tr_ratio = target_length / hop_length
    if ignore > 0:
        zero_pad = torch.zeros(batch_mixture.shape[0], batch_mixture.shape[1], ignore).type(batch_mixture.type()).to(batch_mixture.device)
        batch_mixture_pad = torch.cat([zero_pad, batch_mixture, zero_pad], -1)
    else:
        batch_mixture_pad = batch_mixture
    if target - hop > 0:
        hop_pad = torch.zeros(batch_mixture.shape[0], batch_mixture.shape[1], target-hop).type(batch_mixture.type()).to(batch_mixture.device)
        batch_mixture_pad = torch.cat([hop_pad, batch_mixture_pad, hop_pad], -1)

    skip_idx = ignore + target - hop
    zero_pad = torch.zeros(batch_mixture.shape[0], batch_mixture.shape[1], session).type(batch_mixture.type()).to(batch_mixture.device)
    num_session = (batch_mixture_pad.shape[-1] - session) // hop + 2
    all_target = torch.zeros(batch_mixture_pad.shape[0], n_tracks, batch_mixture_pad.shape[1], batch_mixture_pad.shape[2]).to(batch_mixture_pad.device)
    all_input = []
    all_segment_length = []

    for i in range(num_session):
        this_input = batch_mixture_pad[:,:,i*hop:i*hop+session]
        segment_length = this_input.shape[-1]
        if segment_length < session:
            this_input = torch.cat([this_input, zero_pad[:,:,:session-segment_length]], -1)
        all_input.append(this_input)
        all_segment_length.append(segment_length)

    all_input = torch.cat(all_input, 0)
    num_batch = num_session // batch_size
    if num_session % batch_size > 0:
        num_batch += 1
    
    for i in range(num_batch):

        this_input = all_input[i*batch_size:(i+1)*batch_size]
        actual_batch_size = this_input.shape[0]
        with torch.no_grad():
            est_target = model(this_input)
            # print(est_target.shape)
        for j in range(actual_batch_size):
            this_est_target = est_target[j,:,:,:all_segment_length[i*batch_size+j]][:,:,ignore:ignore+target].unsqueeze(0)
            all_target[:,:,:,ignore+(i*batch_size+j)*hop:ignore+(i*batch_size+j)*hop+target] += this_est_target

    all_target = all_target[:,:,:,skip_idx:skip_idx+batch_length].contiguous() / tr_ratio

    return all_target.squeeze(0)