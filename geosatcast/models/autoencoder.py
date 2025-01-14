import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
import numpy as np
from geosatcast.blocks.ResBlock import ResBlock3D
from geosatcast.utils import sample_from_standard_normal, kl_from_standard_normal
from geosatcast.utils import activation, normalization, conv_nd

class Encoder(nn.Sequential):
    def __init__(
            self,
            in_dim=1, 
            levels=2, 
            min_ch=64, 
            max_ch=64, 
            extra_resblock_levels=[],
            downsampling_mode='resblock',
            norm=None,
            ):
        self.max_ch = max_ch 
        self.min_ch = min_ch
        
        sequence = []
        print(levels, in_dim)
        channels = np.hstack((in_dim, np.arange(1, (levels + 1)) * min_ch))
        channels[channels > max_ch] = max_ch
        channels[-1] = max_ch
        
        res_block_fun = ResBlock3D
        kernel_size = (1, 3, 3)
        resample_factor = (1, 2, 2)

        for i in range(levels):
            in_channels = int(channels[i])
            out_channels = int(channels[i + 1])
            
            if i in extra_resblock_levels:
                extra_block = res_block_fun(
                    in_channels, out_channels,
                    resample=None, 
                    kernel_size=kernel_size, 
                    resample_factor=None,
                    norm=norm)
                in_channels = out_channels
                sequence.append(extra_block)
            
            if downsampling_mode == 'resblock':
                downsample = res_block_fun(
                    in_channels, out_channels,
                    resample='down', 
                    kernel_size=kernel_size, 
                    resample_factor=resample_factor,
                    norm=None)
            
            elif downsampling_mode == 'stride':
                downsample = conv_nd(3, 
                                     in_channels, 
                                     out_channels,
                                     kernel_size=resample_factor, 
                                     stride=resample_factor)
            sequence.append(downsample)

        super().__init__(*sequence)

class Decoder(nn.Sequential):
    def __init__(
            self, 
            in_dim=1, 
            out_dim=1, 
            levels=2, 
            min_ch=64, 
            max_ch=64, 
            extra_resblock_levels=[],
            upsampling_mode='stride',
            norm=None):
        
        self.max_ch = max_ch 
        self.min_ch = min_ch
        self.out_dim = out_dim
        sequence = []
        channels = np.hstack((in_dim, np.arange(1, (levels + 1)) * min_ch))
        channels[channels > max_ch] = max_ch
        channels[0] = out_dim
        channels[-1] = max_ch
        
        res_block_fun = ResBlock3D
        kernel_size = (1, 3, 3)
        resample_factor = (1, 2, 2)
        stride_conv = nn.ConvTranspose3d

        for i in reversed(list(range(levels))):
            in_channels = int(channels[i + 1])
            out_channels = int(channels[i])
            
            if upsampling_mode == 'stride':
                upsample = stride_conv(in_channels, in_channels,
                                       kernel_size=resample_factor, 
                                       stride=resample_factor)
            
            elif upsampling_mode == 'resblock':
                upsample = res_block_fun(
                    in_channels, 
                    in_channels,
                    resample='up', 
                    kernel_size=kernel_size, 
                    resample_factor=resample_factor,
                    norm=None,
                    upsampling_mode=upsampling_mode)
            sequence.append(upsample)
            
            if i in extra_resblock_levels:
                extra_block = res_block_fun(
                    in_channels, out_channels,
                    resample=None, 
                    kernel_size=kernel_size, 
                    resample_factor=None,
                    norm=norm)
                sequence.append(extra_block)
        super().__init__(*sequence)

class VAE(LightningModule):
    def __init__(self,
                 encoder,
                 decoder,
                 lr,
                 kl_weight,
                 encoded_channels,
                 hidden_width,
                 opt_patience,
                 **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=['encoder', 'decoder'])
        self.lr = lr
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_width = hidden_width
        self.opt_patience = opt_patience
        self.encoded_channels = encoded_channels

        self.to_var = nn.Conv3d(encoded_channels, 
                                hidden_width,
                                kernel_size=1)
        self.kl_weight = kl_weight

    def encode(self, x):
        x = self.encoder(x)
        log_var = self.to_var(x)
        return x, log_var

    def decode(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x, sample_posterior=True):
        (z, log_var) = self.encode(x)
        if sample_posterior:
            z = sample_from_standard_normal(z, log_var)
        dec = self.decode(z)
        return dec, z, log_var

    def _loss(self, x):
        (y_pred, mean, log_var) = self.forward(x)
        rec_loss = (x - y_pred).abs().mean()
        kl_loss = kl_from_standard_normal(mean, log_var)
        total_loss = (1 - self.kl_weight) * rec_loss + self.kl_weight * kl_loss
        return total_loss, rec_loss, kl_loss

    def training_step(self, batch, batch_idx):
        x, _, _, _ = batch
        loss, rec_loss, kl_loss = self._loss(x)
        log_params = {"on_step": True, "on_epoch": True, "prog_bar": True, "sync_dist": False}
        self.log('train_loss', loss, **log_params)
        self.log('train_rec_loss', rec_loss, **log_params)
        self.log('train_kl_loss', kl_loss, **log_params)
        return loss

    @torch.no_grad()
    def val_test_step(self, batch, batch_idx, split="val"):
        x, _, _, _ = batch
        (total_loss, rec_loss, kl_loss) = self._loss(x.detach())
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True, "sync_dist": True}
        self.log(f"{split}_loss", total_loss, **log_params)
        self.log(f"{split}_rec_loss", rec_loss.mean(), **log_params)
        self.log(f"{split}_kl_loss", kl_loss, **log_params)

    def validation_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                      betas=(0.5, 0.9), weight_decay=1e-3)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.opt_patience, factor=0.25, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": "val_rec_loss",
                "frequency": 1,
            },
        }