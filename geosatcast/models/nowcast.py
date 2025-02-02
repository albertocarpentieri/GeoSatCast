import torch
from torch import nn
import torch.nn.functional as F
from geosatcast.utils import avg_pool_nd, conv_nd
from geosatcast.blocks.NAT import NATBlock2D
from geosatcast.blocks.AFNO import AFNOBlock2D
from geosatcast.blocks.ResBlock import ResBlock3D


class AFNOCastLatent(nn.Module):
    def __init__(
            self,
            embed_dim=256,
            forecast_depth=0,
            num_blocks=8,
            mlp_ratio=4,
            norm='group',
            layer_scale="auto",
            **kwargs
        ):
        super().__init__()

        self.embed_dim = embed_dim
        self.forecast_depth = forecast_depth
        
        if layer_scale == "auto":
            layer_scale=.5/(forecast_depth)
        elif layer_scale == "none":
            layer_scale = None
        self.forecast = nn.Sequential(
            *(AFNOBlock2D(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio, 
                    num_blocks=num_blocks,
                    norm=norm,
                    channel_first=False,
                    layer_scale=layer_scale
                )
            for _ in range(forecast_depth))
            )

    def forward(self, x):
        x = x.squeeze(2).permute(0,2,3,1)
        x = torch.utils.checkpoint.checkpoint_sequential(self.forecast, self.forecast_depth, x, use_reentrant=False)
        return x.permute(0,3,1,2).unsqueeze(2)


class NATCastLatent(nn.Module):
    def __init__(
            self,
            embed_dim=256,
            forecast_depth=0,
            num_blocks=8,
            mlp_ratio=1,
            kernel_size=3,
            norm="none",
            layer_scale="none"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.forecast_depth = forecast_depth

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size for _ in range(forecast_depth)]
        
        if layer_scale == "auto":
            layer_scale=.5/(forecast_depth)
        elif layer_scale == "none":
            layer_scale = None
        self.forecast = nn.Sequential(
                *(NATBlock2D(
                        dim=embed_dim,
                        mlp_ratio=mlp_ratio, 
                        num_blocks=num_blocks,
                        norm=norm,
                        kernel_size=kernel_size[i],
                        layer_scale=layer_scale
                    )
                for i in range(forecast_depth))
                )
    
    def forward(self, x):
        x = x.squeeze(2).permute(0,2,3,1)
        x = torch.utils.checkpoint.checkpoint_sequential(self.forecast, self.forecast_depth, x, use_reentrant=False)
        return x.permute(0,3,1,2).unsqueeze(2)


class AFNONAT(nn.Module):
    def __init__(
            self,
            embed_dim=256,
            afno_num_blocks=8,
            nat_num_blocks=8,
            mlp_ratio=1,
            kernel_size=3,
            nat_norm="none",
            afno_norm="none",
            layer_scale=1,
            mode="sequential"
        ):
        super().__init__()
        self.mode = mode
        self.afno = AFNOBlock2D(
            dim=embed_dim,
            mlp_ratio=mlp_ratio, 
            num_blocks=afno_num_blocks,
            norm=afno_norm,
            channel_first=False,
            layer_scale=layer_scale
        )
        self.nat = NATBlock2D(
            dim=embed_dim,
            mlp_ratio=mlp_ratio, 
            num_blocks=nat_num_blocks,
            norm=nat_norm,
            kernel_size=kernel_size,
            layer_scale=layer_scale
        )
        if self.mode == "parallel":
            self.afno_alpha = nn.Parameter(.5 * torch.ones(embed_dim, dtype=torch.float32), requires_grad=True)
            self.nat_alpha = nn.Parameter(.5 * torch.ones(embed_dim, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        if self.mode == "sequential":
            return self.nat(self.afno(x))
        elif self.mode == "parallel":
            nat_out = self.nat(x)
            # In-place scaling and addition
            nat_out.mul_(self.nat_alpha)  # In-place scaling
            nat_out.add_(self.afno(x).mul(self.afno_alpha))  # In-place addition
            return nat_out
        else:
            raise Exception("Mode needs to be 'sequential' or 'parallel'")


class AFNONATCastLatent(nn.Module):
    def __init__(
            self,
            embed_dim=256,
            forecast_depth=1,
            afno_num_blocks=8,
            nat_num_blocks=8,
            mlp_ratio=1,
            kernel_size=3,
            nat_norm="none",
            afno_norm="none",
            layer_scale="none",
            mode="sequential",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.forecast_depth = forecast_depth
        self.mode = mode
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size for _ in range(forecast_depth)]
        
        if layer_scale == "auto" and mode == "sequential":
            layer_scale = .5 / (forecast_depth * 2)
        elif layer_scale == "auto" and mode == "parallel":
            layer_scale = .5 / forecast_depth
        elif layer_scale == "none":
            layer_scale = None

        forecast = []
        for i in range(forecast_depth):
            forecast.append(AFNONAT(
                embed_dim=embed_dim,
                afno_num_blocks=afno_num_blocks,
                nat_num_blocks=nat_num_blocks,
                mlp_ratio=mlp_ratio,
                kernel_size=kernel_size[i],
                nat_norm=nat_norm,
                afno_norm=afno_norm,
                layer_scale=layer_scale,
                mode=mode
            ))
        
        self.forecast = nn.Sequential(*forecast)

    def forward(self, x):
        x = x.squeeze(2).permute(0,2,3,1)
        x = torch.utils.checkpoint.checkpoint_sequential(self.forecast, self.forecast_depth, x, use_reentrant=False)
        return x.permute(0,3,1,2).unsqueeze(2)

class DummyLatent(nn.Module):
    def __init__(self):
        super().__init__()
        self.forecast = nn.Identity()
    def forward(self, x):
        return self.forecast(x)

class Nowcaster(nn.Module):
    def __init__(
            self,
            latent_model,
            encoder,
            decoder,
            in_steps,
    ):
        super().__init__()

        self.latent_model = latent_model
        self.encoder = encoder
        self.decoder = decoder
        self.in_steps = in_steps
    
    def forward(self, x, inv, n_steps=1):
        n_steps = min((n_steps, inv.shape[2]-self.in_steps+1))
        yhat = torch.empty((*x.shape[:2], n_steps, *x.shape[3:]), dtype=x.dtype, device=x.device)
        for i in range(n_steps):
            z = torch.cat((x, inv[:,:,i:i+self.in_steps]), dim=1)
            z = self.encoder(z)
            z = self.latent_model(z)
            z = self.decoder(z)
            yhat[:,:,i:i+1] = z
            if i < n_steps-1:
                x = torch.concat((x[:,:,-1:], z), dim=2)
        return yhat

class UNet(nn.Module):
    def __init__(
            self,
            latent_model,
            encoder,
            decoder,
            in_steps,
    ):
        super().__init__()

        self.latent_model = latent_model
        self.encoder = encoder
        self.decoder = decoder
        self.in_steps = in_steps
    
    def forward(self, x, inv, n_steps=1):
        n_steps = min((n_steps, inv.shape[2]-self.in_steps+1))
        yhat = torch.empty((*x.shape[:2], n_steps, *x.shape[3:]), dtype=x.dtype, device=x.device)
        for i in range(n_steps):
            z = torch.cat((x, inv[:,:,i:i+self.in_steps]), dim=1)
            z = self.encoder(z)
            z = self.latent_model(z)
            z = self.decoder(z)
            yhat[:,:,i:i+1] = z
            if i < n_steps-1:
                x = torch.concat((x[:,:,-1:], z), dim=2)
        return yhat


if __name__ == "__main__":
    from geosatcast.models.autoencoder import Encoder, Decoder, AutoEncoder 
    import torch 

    encoder = Encoder(
        in_dim=14, 
        levels=2, 
        min_ch=128,
        max_ch=512, 
        extra_resblock_levels=[0,1], 
        downsampling_mode='stride', 
        norm=None,
        kernel_sizes=[(2,3,3), (1,3,3)],
        resample_factors=[(2,2,2), (1,2,2)])

    decoder = Decoder(
        in_dim=512, 
        out_dim=11, 
        levels=2, 
        min_ch=128, 
        max_ch=512, 
        extra_resblock_levels=[0,1], 
        upsampling_mode='stride', 
        norm=None,
        kernel_size=(1,3,3),
        resample_factor=(1,2,2)
    )

    autoencoder = AutoEncoder(encoder, decoder).to("cuda")
    x = torch.randn((1,11,2,256,256)).to("cuda")
    inv = torch.randn((1,3,66,256,256)).to("cuda")
    
    
    latent_model = AFNOCastLatent(
            embed_dim=512,
            forecast_depth=1,
            num_blocks=8,
            mlp_ratio=4,
            norm='none',
            layer_scale="auto").to("cuda")
    latent_model = AFNONATCastLatent(
        embed_dim=512,
        forecast_depth=1,
    ).to("cuda")
    nowcaster = Nowcaster(
        latent_model,
        autoencoder,
        in_steps=2).to("cuda")
    
    yhat = nowcaster(x, inv, n_steps=64)
    print(yhat.shape)

