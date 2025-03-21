import torch
import torch.nn as nn
import numpy as np
from geosatcast.blocks.ResBlock import ResBlock3D
from geosatcast.utils import sample_from_standard_normal, kl_from_standard_normal
from geosatcast.utils import activation, normalization, conv_nd

class Encoder(nn.Module):
    def __init__(
        self, 
        in_dim=1, 
        levels=2, 
        min_ch=64,
        max_ch=64, 
        extra_resblock_levels=[], 
        downsampling_mode='resblock', 
        norm=None,
        init="he",
        kernel_sizes=[(1,3,3), (1,3,3)],
        resample_factors=[(1,2,2), (1,2,2)],
        channels=None):
        
        super().__init__()
        
        if channels is None:
            self.max_ch = max_ch
            self.min_ch = min_ch
            channels = np.hstack((in_dim, np.arange(1, (levels + 1)) * min_ch))
            channels[channels > max_ch] = max_ch
            channels[-1] = max_ch
        else:
            self.max_ch = channels[-1]
            self.min_ch = channels[0]
            channels = np.hstack((in_dim, np.array(channels)))
        
        sequence = []
        res_block_fun = ResBlock3D

        for i in range(levels):
            kernel_size = kernel_sizes[i]
            resample_factor = resample_factors[i]
            in_channels = int(channels[i])
            out_channels = int(channels[i + 1])

            if i in extra_resblock_levels:
                sequence.append(res_block_fun(in_channels, out_channels, resample=None, kernel_size=(1,3,3), norm=norm, init=init))
                in_channels = out_channels

            if downsampling_mode == 'resblock':
                sequence.append(res_block_fun(in_channels, out_channels, resample='down', kernel_size=kernel_size, resample_factor=resample_factor, norm=norm, init=init))
            
            elif downsampling_mode == 'stride':
                conv_layer = conv_nd(3, in_channels, out_channels, kernel_size=resample_factor, stride=resample_factor, padding_mode="reflect")
                torch.nn.init.zeros_(conv_layer.bias)
                if init == "he":
                    torch.nn.init.kaiming_normal_(conv_layer.weight, mode='fan_in', nonlinearity='relu')  # Use 'relu' for GELU
                elif init == "xavier":
                    torch.nn.init.xavier_uniform_(conv_layer.weight)
                sequence.append(conv_layer)
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(
        self, 
        in_dim=1, 
        out_dim=1, 
        levels=2, 
        min_ch=64, 
        max_ch=64, 
        extra_resblock_levels=[], 
        upsampling_mode='stride', 
        norm=None,
        init="he",
        kernel_size=(1,3,3),
        resample_factor=(1,2,2)):
        
        super().__init__()
        
        self.max_ch = max_ch
        self.min_ch = min_ch
        channels = np.hstack((in_dim, np.arange(1, (levels + 1)) * min_ch))
        channels[channels > max_ch] = max_ch
        channels[0] = out_dim
        channels[-1] = max_ch

        sequence = []
        stride_conv = nn.ConvTranspose3d
        res_block_fun = ResBlock3D

        for i in reversed(range(levels)):
            in_channels = int(channels[i + 1])
            out_channels = int(channels[i])

            if upsampling_mode == 'stride':
                if i in extra_resblock_levels:
                    conv_layer = stride_conv(in_channels, in_channels, kernel_size=resample_factor, stride=resample_factor)
                else:
                    conv_layer = stride_conv(in_channels, out_channels, kernel_size=resample_factor, stride=resample_factor)
                torch.nn.init.zeros_(conv_layer.bias)
                if init == "he":
                    torch.nn.init.kaiming_normal_(conv_layer.weight, mode='fan_in', nonlinearity='relu')  # Use 'relu' for GELU
                elif init == "xavier":
                    torch.nn.init.xavier_uniform_(conv_layer.weight)
                sequence.append(conv_layer)

            elif upsampling_mode == 'resblock':
                if i in extra_resblock_levels:
                    sequence.append(res_block_fun(in_channels, in_channels, resample='up', kernel_size=kernel_size, resample_factor=resample_factor, norm=norm, init=init))
                else:
                    sequence.append(res_block_fun(in_channels, out_channels, resample='up', kernel_size=kernel_size, resample_factor=resample_factor, norm=norm, init=init))
            if i in extra_resblock_levels:
                sequence.append(res_block_fun(in_channels, out_channels, resample=None, kernel_size=kernel_size, norm=norm, init=init))

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class VAE(nn.Module):
    def __init__(self, encoder, decoder, hidden_width, encoded_channels):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.to_mean = nn.Conv3d(encoded_channels, hidden_width, kernel_size=1)
        self.to_var = nn.Conv3d(encoded_channels, hidden_width, kernel_size=1)

    def encode(self, x):
        x = self.encoder(x)
        x = self.to_mean(x)
        log_var = self.to_var(x)
        return x, log_var

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, sample_posterior=True):
        mean, log_var = self.encode(x)
        if sample_posterior:
            z = sample_from_standard_normal(mean, log_var)
            z = self.decode(z)
        else:
            z = self.decode(mean)
        return z, mean, log_var

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))