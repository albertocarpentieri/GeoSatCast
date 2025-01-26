import torch
from torch import nn
import torch.nn.functional as F
from geosatcast.utils import avg_pool_nd, conv_nd
from geosatcast.blocks.NAT import NATBlock2D
from geosatcast.blocks.AFNO import AFNOBlock2D


class AFNOCastLatent(nn.Module):
    def __init__(
            self,
            hidden_dim,
            in_steps=2,
            embed_dim=256,
            embed_dim_out=128,
            forecast_depth=0,
            num_blocks=8,
            mlp_ratio=4,
            norm='group',
            **kwargs
        ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.embed_dim_out = embed_dim_out
        self.in_steps = in_steps
        self.forecast_depth = forecast_depth
        
        # compresses the time dimension to embed the input
        self.proj = conv_nd(
                 3,
                 hidden_dim, 
                 embed_dim,
                 kernel_size=(self.in_steps,1,1),
                 stride=(self.in_steps,1,1))
        
        self.reproj = conv_nd(
                3,
                embed_dim, 
                embed_dim_out,
                kernel_size=1,
                stride=1
                )
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

    def forecast_step(self, x, inv):
        x = torch.cat((x, inv), dim=1)
        x = self.proj(x)
        x = x.squeeze(2).permute(0,2,3,1)
        x = torch.utils.checkpoint.checkpoint_sequential(self.forecast, self.forecast_depth, x, use_reentrant=False)
        x = self.reproj(x.permute(0,3,1,2).unsqueeze(2))
        return x
    
    def forward(self, x, inv, n_steps=1):
        n_steps = min((n_steps, inv.shape[2]-self.in_steps+1))
        yhat = torch.empty((*x.shape[:2], n_steps, *x.shape[3:]), dtype=x.dtype, device=x.device)
        for i in range(n_steps):
            yhat_ = self.forecast_step(x, inv[:,:,i:i+self.in_steps])
            yhat[:,:,i:i+1] = yhat_
            if i < n_steps-1:
                x = torch.concat((x[:,:,-1:], yhat_), dim=2)
        return yhat


class NATCastLatent(nn.Module):
    def __init__(
            self,
            hidden_dim,
            in_steps=2,
            embed_dim=256,
            embed_dim_out=128,
            forecast_depth=0,
            num_blocks=8,
            mlp_ratio=1,
            kernel_size=3,
            norm="none",
            layer_scale="none"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.embed_dim_out = embed_dim_out
        self.in_steps = in_steps
        self.forecast_depth = forecast_depth

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size for _ in range(forecast_depth)]
        
        # compresses the time dimension to embed the input
        self.proj = conv_nd(
                    3,
                    hidden_dim, 
                    embed_dim,
                    kernel_size=(self.in_steps,1,1),
                    stride=(self.in_steps,1,1))
        
        self.reproj = conv_nd(
                3,
                embed_dim, 
                embed_dim_out,
                kernel_size=1,
                stride=1
                )
        
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
    
    def forecast_step(self, x, inv):
        x = torch.cat((x, inv), dim=1)
        x = self.proj(x).squeeze(2).permute(0,2,3,1)
        x = torch.utils.checkpoint.checkpoint_sequential(self.forecast, self.forecast_depth, x, use_reentrant=False)
        x = self.reproj(x.permute(0,3,1,2).unsqueeze(2))
        return x
    
    def forward(self, x, inv, n_steps=1):
        n_steps = min((n_steps, inv.shape[2]-self.in_steps+1))
        yhat = torch.empty((*x.shape[:2], n_steps, *x.shape[3:]), dtype=x.dtype, device=x.device)
        for i in range(n_steps):
            yhat_ = self.forecast_step(x, inv[:,:,i:i+self.in_steps])
            yhat[:,:,i:i+1] = yhat_
            if i < n_steps-1:
                x = torch.concat((x[:,:,-1:], yhat_), dim=2)
        return yhat


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
            self.afno_alpha = nn.Parameter(.5 * torch.ones(embed_dim), requires_grad=True)
            self.nat_alpha = nn.Parameter(.5 * torch.ones(embed_dim), requires_grad=True)

    def forward(self, x):
        if self.mode == "sequential":
            return self.nat(self.afno(x))
        elif self.mode == "parallel":
            return self.nat_alpha * self.nat(x) + self.afno_alpha * self.afno(x)
        else:
            raise Exception("Mode needs to be 'sequential' or 'parallel'")


class AFNONATCastLatent(nn.Module):
    def __init__(
            self,
            hidden_dim,
            in_steps=2,
            embed_dim=256,
            embed_dim_out=128,
            forecast_depth=1,
            afno_num_blocks=8,
            nat_num_blocks=8,
            mlp_ratio=1,
            kernel_size=3,
            nat_norm="none",
            afno_norm="none",
            layer_scale="none",
            mode="sequential"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.embed_dim_out = embed_dim_out
        self.in_steps = in_steps
        self.forecast_depth = forecast_depth
        self.mode = mode
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size for _ in range(forecast_depth)]
        
        # compresses the time dimension to embed the input
        self.proj = conv_nd(
                    3,
                    hidden_dim, 
                    embed_dim,
                    kernel_size=(self.in_steps,1,1),
                    stride=(self.in_steps,1,1))
        
        self.reproj = conv_nd(
                3,
                embed_dim, 
                embed_dim_out,
                kernel_size=1,
                stride=1
                )
        
        if layer_scale == "auto":
            layer_scale=.5/(forecast_depth * 2)
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

    def forecast_step(self, x, inv):
        x = torch.cat((x, inv), dim=1)
        x = self.proj(x).squeeze(2).permute(0,2,3,1)
        x = torch.utils.checkpoint.checkpoint_sequential(self.forecast, self.forecast_depth, x, use_reentrant=False)
        x = self.reproj(x.permute(0,3,1,2).unsqueeze(2))
        return x
    
    def forward(self, x, inv, n_steps=1):
        n_steps = min((n_steps, inv.shape[2]-self.in_steps+1))
        yhat = torch.empty((*x.shape[:2], n_steps, *x.shape[3:]), dtype=x.dtype, device=x.device)
        for i in range(n_steps):
            yhat_ = self.forecast_step(x, inv[:,:,i:i+self.in_steps])
            yhat[:,:,i:i+1] = yhat_
            if i < n_steps-1:
                x = torch.concat((x[:,:,-1:], yhat_), dim=2)
        return yhat

class Nowcaster(nn.Module):
    def __init__(
            self,
            latent_model,
            encoder,
            decoder,
            inv_encoder
    ):
        super().__init__()

        self.latent_model = latent_model
        self.encoder = encoder
        self.decoder = decoder
        self.inv_encoder = inv_encoder
        for param in self.autoencoder.parameters():
            param.requires_grad = False
    
    def latent_forward(self, x, inv, n_steps=48):
        x = self.encoder(x)
        # encode invariants
        inv = self.inv_encoder(inv)
        x = self.latent_model(x, inv, n_steps)
        return x 
    
    def forward(self, x, inv, n_steps):
        return self.decoder(self.latent_forward(x, inv, n_steps))


if __name__ == "__main__":
    pass
