import torch.nn as nn
from geosatcast.utils import activation, normalization


class ResBlock3D(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            resample=None,
            resample_factor=(1, 1, 1), 
            kernel_size=(1, 3, 3),
            act='swish', 
            norm='group', 
            upsampling_mode='nearest',
            **kwargs
    ):
        super().__init__(**kwargs)
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        padding = tuple(k // 2 for k in kernel_size)
        if resample == "down":
            self.resample = nn.AvgPool3d(resample_factor, ceil_mode=True)
            self.conv1 = nn.Conv3d(in_channels, out_channels,
                                   kernel_size=kernel_size, stride=resample_factor,
                                   padding=padding)
            self.conv2 = nn.Conv3d(out_channels, out_channels,
                                   kernel_size=kernel_size, 
                                   padding=padding)
        elif resample == "up":
            self.resample = nn.Upsample(
                scale_factor=resample_factor, mode=upsampling_mode)
            self.conv1 = nn.ConvTranspose3d(in_channels, out_channels,
                                            kernel_size=kernel_size, padding=padding)
            output_padding = tuple(
                2 * p + s - k for (p, s, k) in zip(padding, resample_factor, kernel_size)
            )
            self.conv2 = nn.ConvTranspose3d(out_channels, out_channels,
                                            kernel_size=kernel_size, stride=resample_factor,
                                            padding=padding, output_padding=output_padding)
        else:
            self.resample = nn.Identity()
            self.conv1 = nn.Conv3d(in_channels, out_channels,
                                   kernel_size=kernel_size, padding=padding)
            self.conv2 = nn.Conv3d(out_channels, out_channels,
                                   kernel_size=kernel_size, padding=padding)
        
        if isinstance(act, str):
            act = (act, act)
        self.act1 = activation(act_type=act[0])
        self.act2 = activation(act_type=act[1])

        self.norm1 = normalization(in_channels, norm_type=norm)
        self.norm2 = normalization(out_channels, norm_type=norm)
    
    def forward(self, x):
        x_in = self.resample(self.proj(x))
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x + x_in
