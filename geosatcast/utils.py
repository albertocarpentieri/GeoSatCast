import pickle as pkl
import numpy as np
import yaml
import torch
import torch.nn as nn


def activation(act_type="swish"):
    act_dict = {"swish": nn.SiLU(),
                "gelu": nn.GELU(),
                "relu": nn.ReLU(),
                "tanh": nn.Tanh()}
    if act_type:
        if act_type in act_dict:
            return act_dict[act_type]
        else:
            raise NotImplementedError(act_type)
    elif not act_type:
        return nn.Identity()

# normalization
class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
def normalization(channels, norm_type="group", num_groups=32):
    if norm_type == "batch3d":
        return nn.BatchNorm3d(channels)
    if norm_type == "batch2d":
        return nn.BatchNorm2d(channels)
    elif norm_type == "group":
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    elif norm_type == "group32":
        return GroupNorm32()
    elif norm_type == 'full_group':
        return nn.GroupNorm(num_groups=1, num_channels=channels)
    elif norm_type == 'layer':
        return nn.LayerNorm(num_channels=channels)
    elif (not norm_type) or (norm_type.lower() == 'none'):
        return nn.Identity()
    else:
        raise NotImplementedError(norm_type)


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)

# pooling layer
def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

# convolution
def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

# VAE utils
def kl_from_standard_normal(mean, log_var):
    kl = 0.5 * (log_var.exp() + mean.square() - 1.0 - log_var)
    return kl.mean()


def sample_from_standard_normal(mean, log_var, num=None):
    std = (0.5 * log_var).exp()
    shape = mean.shape
    if num is not None:
        # expand channel 1 to create several samples
        shape = shape[:1] + (num,) + shape[1:]
        mean = mean[:, None, ...]
        std = std[:, None, ...]
    return mean + std * torch.randn(shape, device=mean.device)