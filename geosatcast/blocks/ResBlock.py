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
            act='gelu', 
            norm='group', 
            upsampling_mode='nearest',
            **kwargs
    ):
        super().__init__(**kwargs)
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding_mode="reflect") if in_channels != out_channels else nn.Identity()
        padding = tuple(k // 2 for k in kernel_size)
        if resample == "down":
            self.resample = nn.AvgPool3d(resample_factor, ceil_mode=True)
            self.conv1 = nn.Conv3d(in_channels, out_channels,
                                   kernel_size=kernel_size, stride=resample_factor,
                                   padding=padding, padding_mode="reflect")
            self.conv2 = nn.Conv3d(out_channels, out_channels,
                                   kernel_size=kernel_size, 
                                   padding=padding, padding_mode="reflect")
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
                                   kernel_size=kernel_size, padding=padding, padding_mode="reflect")
            self.conv2 = nn.Conv3d(out_channels, out_channels,
                                   kernel_size=kernel_size, padding=padding, padding_mode="reflect")
        
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.zeros_(self.conv2.bias)
        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')  # Use 'relu' for GELU
        
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
