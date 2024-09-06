###
# Author: Kai Li
# Date: 2021-06-18 17:29:21
# LastEditors: Kai Li
# LastEditTime: 2021-06-21 23:52:52
###

import torch
import torch.nn as nn


def pad_x_to_y(x, y, axis: int = -1):
    if axis != -1:
        raise NotImplementedError
    inp_len = y.shape[axis]
    output_len = x.shape[axis]
    return nn.functional.pad(x, [0, inp_len - output_len])


def shape_reconstructed(reconstructed, size):
    if len(size) == 1:
        return reconstructed.squeeze(0)
    return reconstructed


def tensors_to_device(tensors, device):
    """Transfer tensor, dict or list of tensors to device.

    Args:
        tensors (:class:`torch.Tensor`): May be a single, a list or a
            dictionary of tensors.
        device (:class: `torch.device`): the device where to place the tensors.

    Returns:
        Union [:class:`torch.Tensor`, list, tuple, dict]:
            Same as input but transferred to device.
            Goes through lists and dicts and transfers the torch.Tensor to
            device. Leaves the rest untouched.
    """
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device)
    elif isinstance(tensors, (list, tuple)):
        return [tensors_to_device(tens, device) for tens in tensors]
    elif isinstance(tensors, dict):
        for key in tensors.keys():
            tensors[key] = tensors_to_device(tensors[key], device)
        return tensors
    else:
        return tensors
