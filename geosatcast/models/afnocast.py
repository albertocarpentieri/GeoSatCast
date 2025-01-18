import torch
from torch import nn
from geosatcast.blocks.AFNO import AFNOBlock3d
from geosatcast.blocks.ResBlock import ResBlock3D
from geosatcast.models.autoencoder import VAE, Encoder
import torch.nn.functional as F
from geosatcast.utils import avg_pool_nd, conv_nd


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
        self.proj =  conv_nd(
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

        self.forecast = nn.Sequential(
            *(AFNOBlock3d(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio, 
                    num_blocks=num_blocks,
                    norm=norm
                )
            for _ in range(forecast_depth))
            )

    def forecast_step(self, x, inv):
        x = torch.cat((x, inv), dim=1)
        x = self.proj(x)
        x = torch.utils.checkpoint.checkpoint_sequential(self.forecast, self.forecast_depth, x, use_reentrant=True)
        x = self.reproj(x)
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


class AFNOCast(nn.Module):
    def __init__(
            self,
            afnocast_latent,
            vae,
            inv_encoder
        ):
        super().__init__()

        self.afnocast_latent = afnocast_latent
        self.vae = vae
        self.inv_encoder = inv_encoder
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def latent_forward(self, x, inv, n_steps=48):
        x, _ = self.vae.encode(x)
        # encode invariants
        inv = self.inv_encoder(inv)
        x = self.afnocast_latent(x, inv, n_steps)
        return x 
    
    def forward(self, x, inv, n_steps):
        return self.vae.decode(self.latent_forward(x, inv, n_steps))
        

if __name__ == "__main__":
    pass