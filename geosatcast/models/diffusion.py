import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from natten import NeighborhoodAttention3D

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule from:
    https://openreview.net/pdf?id=-NEXDKk8gZ
    or "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal).
    
    timesteps: number of diffusion steps
    s: small offset to avoid singularities
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # normalize to 1 at t=0
    # Now alphas_cumprod[t] = alpha_bar at step t
    # betas[t] = 1 - alpha_bar[t] / alpha_bar[t-1]
    
    betas = []
    for i in range(1, len(alphas_cumprod)):
        alpha_bar_t   = alphas_cumprod[i]
        alpha_bar_t_1 = alphas_cumprod[i-1]
        betas.append(1 - alpha_bar_t / alpha_bar_t_1)
    betas = torch.stack(betas)  # shape [timesteps]
    return betas


class FiLM3D(nn.Module):
    """
    For each scale, we do a global pooling of the condition features
    to generate gamma, beta, which we apply to the main features.

    You can also do more advanced 3D FiLM if you want separate scale/shift
    per (T, H, W) location, but that can be large in memory. 
    Here, we do a simple global pooling -> MLP -> FiLM.
    """
    def __init__(self, cond_in_ch, main_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))  # reduce [B, cond_in_ch, T, H, W] to [B, cond_in_ch, 1,1,1]
        self.mlp = nn.Sequential(
            nn.Linear(cond_in_ch, main_ch*2),
            nn.ReLU(inplace=True),
            nn.Linear(main_ch*2, main_ch*2)
        )
    
    def forward(self, main_feats, cond_feats):
        """
        main_feats: [B, main_ch, T, H, W]
        cond_feats: [B, cond_in_ch, T, H, W], same resolution, but we'll just pool it.
        """
        B, Cc, T, H, W = cond_feats.shape
        # Global pool condition
        cond_pooled = self.pool(cond_feats).view(B, Cc)  # [B, Cc]
        
        # MLP => [B, main_ch*2]
        scale_shift = self.mlp(cond_pooled)  # [B, 2*main_ch]
        
        mainC = main_feats.shape[1]
        gamma, beta = scale_shift[:, :mainC], scale_shift[:, mainC:]
        
        # Reshape to [B, mainC, 1,1,1]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        return main_feats * (1 + gamma) + beta

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        skip = self.skip(x)
        x = self.norm1(self.conv1(x))
        x = F.relu(x, inplace=True)
        x = self.norm2(self.conv2(x))
        x = x + skip
        return F.relu(x, inplace=True)

class NeighborhoodAttentionBlock3D(nn.Module):
    def __init__(self, channels, kernel_size=7, num_heads=4):
        super().__init__()
        if not NAT_AVAILABLE:
            raise RuntimeError("NATTEN not installed.")
        
        self.norm = nn.GroupNorm(8, channels)
        self.natten = NeighborhoodAttention3D(
            dim=channels, 
            kernel_size=kernel_size,
            num_heads=num_heads,
            dilation=1,
            bias=True
        )
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.natten(x)
        x = self.proj(x)
        return x + skip


def downsample_3d(in_ch, out_ch, kernel_size=(2,2,2)):
    return nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=kernel_size)

def upsample_3d(in_ch, out_ch, kernel_size=(2,2,2)):
    return nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size, stride=kernel_size)

class VideoDiffusionUNetFiLM(nn.Module):
    """
    3D U-Net with:
     - Downsample first in the down path
     - NAT or ResBlock
     - FiLM injection
     - In the up path, do block first, FiLM, then upsample, then skip concat, etc.
    """
    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        num_scales=3,
        nat_scales=(1,2),   # which levels use NAT
        nat_kernel=7,
        nat_heads=4,
        cond_in_ch=16,      # condition channels
        time_emb_dim=256
    ):
        super().__init__()
        self.num_scales = num_scales
        self.nat_scales = set(nat_scales)
        
        # Time embedding (if needed)
        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        self.in_proj = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # ---------- Down path ----------
        self.downs = nn.ModuleList()
        ch = base_channels
        for scale_idx in range(num_scales):
            out_ch = ch * 2
            down = downsample_3d(ch, out_ch)  # downsample first
            if scale_idx in self.nat_scales:
                block = NeighborhoodAttentionBlock3D(out_ch, nat_kernel, nat_heads)
            else:
                block = ResBlock3D(out_ch, out_ch)
            
            # FiLM injection
            film = FiLM3D(cond_in_ch, out_ch)
            
            self.downs.append(nn.ModuleList([
                down,  # 1) downsample
                block, # 2) block
                film   # 3) FiLM
            ]))
            ch = out_ch
        
        # ---------- Middle ----------
        # We can do one NAT block or ResBlock in the middle
        if num_scales in self.nat_scales:
            self.mid_block = NeighborhoodAttentionBlock3D(ch, nat_kernel, nat_heads)
        else:
            self.mid_block = ResBlock3D(ch, ch)
        
        # ---------- Up path ----------
        self.ups = nn.ModuleList()
        for scale_idx in reversed(range(num_scales)):
            in_ch = ch
            out_ch = ch // 2
            
            if scale_idx in self.nat_scales:
                block = NeighborhoodAttentionBlock3D(in_ch, nat_kernel, nat_heads)
            else:
                block = ResBlock3D(in_ch, in_ch)
            
            film = FiLM3D(cond_in_ch, in_ch)
            up = upsample_3d(in_ch, out_ch)
            
            # second block after skip (optional)
            post_skip = ResBlock3D(out_ch*2, out_ch)
            
            self.ups.append(nn.ModuleList([
                block,   # 1) block
                film,    # 2) FiLM
                up,      # 3) upsample
                post_skip # 4) fuse skip
            ]))
            
            ch = out_ch
        
        self.out_proj = nn.Conv3d(ch, in_channels, kernel_size=3, padding=1)
    
    def time_embedding(self, t):
        device = t.device
        half_dim = self.time_embed[0].in_features // 2
        freqs = torch.exp(-math.log(10000.0)*torch.arange(0, half_dim, device=device)/half_dim)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.time_embed(emb)
    
    def forward(self, x, t, cond_feats):
        """
        x: [B, in_channels, T, H, W]
        cond_feats: list of condition feats at each scale:
          cond_feats[0] => largest scale
          cond_feats[1] => half scale
          ...
          cond_feats[num_scales] => deepest
        We assume they match the shape after each down or up step.
        """
        _ = self.time_embedding(t)  # Not explicitly used in blocks, but can be added if desired.
        
        h = self.in_proj(x)
        skip_list = []
        
        # ---- Down path ----
        for scale_idx, (down, block, film) in enumerate(self.downs):
            # downsample first
            h = down(h)
            # block
            h = block(h)
            # FiLM
            h = film(h, cond_feats[scale_idx+1])  # cond_feats index offset
            skip_list.append(h)
        
        # ---- Middle ----
        h = self.mid_block(h)
        
        # ---- Up path ----
        for scale_idx, (block, film, up, post_skip) in enumerate(self.ups):
            # block first
            h = block(h)
            # FiLM
            real_idx = self.num_scales - scale_idx  # e.g. if scale_idx=0 => real_idx=3
            h = film(h, cond_feats[real_idx])       # match shape
            # upsample
            h = up(h)
            
            # skip connection
            skip = skip_list.pop()
            # concat
            h = torch.cat([h, skip], dim=1)
            
            # second block
            h = post_skip(h)
        
        out = self.out_proj(h)
        return out


class VideoDiffusionModelFiLM:
    """
    Diffusion wrapper that:
      - uses a 3D U-Net with FiLM (local attention only)
      - uses a ConditionEncoder3D or something that yields cond_feats
      - Cosine schedule
      - Residual modeling => x_diff = x_gt - x_forecast
    """
    def __init__(
        self,
        unet: nn.Module,
        cond_encoder: nn.Module,
        timesteps=200,
        schedule='cosine'
    ):
        super().__init__()
        self.unet = unet
        self.cond_encoder = cond_encoder
        self.timesteps = timesteps
        
        # Beta schedule
        if schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps)
        else:
            # fallback or implement other schedules
            self.betas = torch.linspace(1e-4, 2e-2, timesteps)
        
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1,0), value=1.0)

    def q_sample(self, x0_diff, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0_diff)
        alpha_t = self.alpha_cumprod[t].reshape(-1,1,1,1,1)
        return torch.sqrt(alpha_t)*x0_diff + torch.sqrt(1-alpha_t)*noise
    
    def p_losses(self, x_gt, x_forecast, t, cond_maps):
        """
        x_gt: [B, C, T, H, W]
        x_forecast: [B, C, T, H, W]
        cond_maps: [B, 2, T, H, W] or whatever the ConditionEncoder expects
        """
        # residual
        x_diff0 = x_gt - x_forecast
        
        # forward diffuse
        noise = torch.randn_like(x_diff0)
        x_diff_t = self.q_sample(x_diff0, t, noise=noise)
        
        # condition feats
        cond_feats = self.cond_encoder(cond_maps)  # list of feats
        # predict noise
        noise_pred = self.unet(x_diff_t, t, cond_feats)
        
        return F.mse_loss(noise_pred, noise)
    
    @torch.no_grad()
    def p_sample(self, x_diff, t, cond_feats):
        betas_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alpha_cumprod[t]
        alpha_cumprod_prev_t = self.alpha_cumprod_prev[t]
        
        noise_pred = self.unet(x_diff, t, cond_feats)
        
        x_pred = (1./torch.sqrt(alpha_t))*(x_diff - (1-alpha_t)/torch.sqrt(1-alpha_cumprod_t)*noise_pred)
        
        if t > 0:
            sigma_t = torch.sqrt((1-alpha_cumprod_prev_t)/(1-alpha_cumprod_t)*betas_t)
            noise = torch.randn_like(x_diff)
            return x_pred + sigma_t*noise
        else:
            return x_pred
    
    @torch.no_grad()
    def sample(self, x_forecast, cond_maps, frames_shape):
        """
        x_forecast: [B, C, T, H, W], we add the predicted residual to it at the end.
        frames_shape: same as x_forecast shape => (B, C, T, H, W)

        returns: x_final = x_forecast + predicted_residual
        """
        device = x_forecast.device
        x_diff = torch.randn(frames_shape, device=device)
        
        cond_feats = self.cond_encoder(cond_maps)
        
        for i in reversed(range(self.timesteps)):
            t = torch.tensor([i]*frames_shape[0], device=device, dtype=torch.long)
            x_diff = self.p_sample(x_diff, t, cond_feats)
        
        x_final = x_forecast + x_diff
        return x_final, x_diff
    
    def training_step(self, x_gt, x_forecast, cond_maps, optimizer):
        b = x_gt.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=x_gt.device).long()
        loss = self.p_losses(x_gt, x_forecast, t, cond_maps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss
