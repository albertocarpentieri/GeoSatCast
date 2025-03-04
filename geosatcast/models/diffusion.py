import torch
import torch.nn.functional as F
import torch.nn as nn
import math


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

class VideoDiffusionModel:
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
    
    def p_losses(self, x_gt, x, xinv, t):
        """
        x_gt: [B, C, T, H, W]
        x_forecast: [B, C, T, H, W]
        cond_maps: [B, 2, T, H, W] or whatever the ConditionEncoder expects
        """
        # residual
        x_forecast, cond_feats = self.cond_encoder(x, xinv, n_steps=x_gt.shape[2])
        x_diff0 = x_gt - x_forecast
        
        # forward diffuse
        noise = torch.randn_like(x_diff0)
        x_diff_t = self.q_sample(x_diff0, t, noise=noise)
        
        # predict noise
        noise_pred = self.unet(x_diff_t, t, cond_maps)
        
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

