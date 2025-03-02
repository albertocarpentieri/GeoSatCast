import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from geosatcast.utils import normalization, conv_nd
import torch.utils.checkpoint as checkpoint

class AFNO2D(nn.Module):
    def __init__(
            self, hidden_size, num_blocks=8, sparsity_threshold=0.01,
            hard_thresholding_fraction=1, hidden_size_factor=1, res_mult=1,
            channel_first=True
    ):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.channel_first = channel_first
        self.res_mult = res_mult

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x, modes_range=None):
        bias = x
        if self.channel_first:
            x = x.permute(0, 2, 3, 1).contiguous()
        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        x = torch.fft.rfftn(x, dim=(1, 2), norm="ortho")
        x = x.view(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor],
                              device=x.device)
        o1_imag = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor],
                              device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = H // 2 + 1
        if modes_range:
            mode_min, mode_max = modes_range[0], modes_range[1]
        else:    
            mode_min, mode_max = 0, int(total_modes * self.hard_thresholding_fraction)

        # if mode_min > 0:
        #     x[:, :, total_modes - mode_min:total_modes + mode_min, :mode_min] = 0
        o1_real[:, :, total_modes - mode_max:total_modes + mode_max, :mode_max] = F.relu(
            torch.einsum('...bi,bio->...bo',
                        x[:, :, total_modes - mode_max:total_modes + mode_max, :mode_max].real, self.w1[0]) -
            torch.einsum('...bi,bio->...bo',
                        x[:, :, total_modes - mode_max:total_modes + mode_max, :mode_max].imag, self.w1[1]) +
            self.b1[0]
        )

        o1_imag[:, :, total_modes - mode_max:total_modes + mode_max, :mode_max] = F.relu(
            torch.einsum('...bi,bio->...bo',
                        x[:, :, total_modes - mode_max:total_modes + mode_max, :mode_max].imag, self.w1[0]) +
            torch.einsum('...bi,bio->...bo',
                        x[:, :, total_modes - mode_max:total_modes + mode_max, :mode_max].real, self.w1[1]) +
            self.b1[1]
        )

        o2_real[:, :, total_modes - mode_max:total_modes + mode_max, :mode_max] = (
                torch.einsum('...bi,bio->...bo',
                            o1_real[:, :, total_modes - mode_max:total_modes + mode_max, :mode_max],
                            self.w2[0]) -
                torch.einsum('...bi,bio->...bo',
                            o1_imag[:, :, total_modes - mode_max:total_modes + mode_max, :mode_max],
                            self.w2[1]) +
                self.b2[0]
        )

        o2_imag[:, :, total_modes - mode_max:total_modes + mode_max, :mode_max] = (
                torch.einsum('...bi,bio->...bo',
                            o1_imag[:, :, total_modes - mode_max:total_modes + mode_max, :mode_max],
                            self.w2[0]) +
                torch.einsum('...bi,bio->...bo',
                            o1_real[:, :, total_modes - mode_max:total_modes + mode_max, :mode_max],
                            self.w2[1]) +
                self.b2[1]
        )
        x = torch.stack([o2_real, o2_imag], dim=-1)

        if mode_min > 0:
            x[:,:, total_modes - mode_min:total_modes + mode_min] = 0
            x[:,:,:,:mode_min] = 0

        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.view(B, H, W // 2 + 1, C)
        x = torch.fft.irfftn(x, s=(H*self.res_mult, W*self.res_mult), dim=(1, 2), norm="ortho")
        x = x.type(dtype)
        if self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.res_mult>1:
            return x
        else:
            return x + bias


class Mlp(nn.Module):
    def __init__(
            self,
            in_features, hidden_features=None, out_features=None,
            act_layer=nn.GELU, drop=0.0, channel_first=True
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if channel_first:
            self.fc1 = conv_nd(2, in_features, hidden_features, 1)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        
        if channel_first:
            self.fc2 = conv_nd(2, hidden_features, out_features, 1)
        else:
            self.fc2 = nn.Linear(hidden_features, out_features)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNOBlock2D(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            drop=0.,
            act_layer=nn.GELU,
            norm='layer',
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
            mlp_out_features=None,
            afno_res_mult=1,
            channel_first=True,
            layer_scale=None

    ):
        super().__init__()
        self.norm = norm
        self.afno_res_mult = afno_res_mult
        self.norm1 = normalization(dim, norm)
        self.filter = AFNO2D(
            dim, 
            num_blocks, 
            sparsity_threshold,
            hard_thresholding_fraction, 
            res_mult=afno_res_mult,
            channel_first=channel_first)
        self.norm2 = normalization(dim, norm)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, 
            out_features=mlp_out_features,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer, 
            drop=drop,
            channel_first=channel_first
        )
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
    
    def forward(self, x, modes_range=None):
        if not self.layer_scale:
            residual = x
            x = self.norm1(x)
            x = self.filter(x, modes_range=modes_range)
            if self.afno_res_mult > 1:
                residual = F.interpolate(residual, x.shape[2:])
            if self.double_skip:
                x = x + residual
                residual = x

            x = self.norm2(x)
            x = self.mlp(x)
            x = x + residual
            return x
        
        residual = x
        x = self.filter(self.norm1(x), modes_range=modes_range)
        if self.afno_res_mult > 1:
            residual = F.interpolate(residual, x.shape[2:])
        x = self.gamma1 * x + residual
        # residual = x        
        x = self.gamma2 * self.mlp(self.norm2(x)) + x
        return x
        

#############################
# Helper: Sinusoidal Frequency Embedding
#############################
def build_sinusoidal_freq_embedding(H, W_half, d_embed, base=10000.0, device='cpu'):
    """
    Creates a 2D sinusoidal embedding for frequency indices:
      - For the x-axis: positions = [0, 1, ..., H-1] with shape (H,1)
      - For the y-axis: positions = [0, 1, ..., W_half-1] with shape (W_half,1)
    Returns a tensor of shape [H, W_half, d_embed].
    """
    k_x = torch.arange(H, device=device).unsqueeze(1).float()  # (H,1)
    k_y = torch.arange(W_half, device=device).unsqueeze(1).float()  # (W_half,1)
    
    d_embed_half = d_embed // 2
    div_term_x = torch.exp(torch.arange(0, d_embed_half, 2, device=device).float() * (-math.log(base) / d_embed_half))
    div_term_y = torch.exp(torch.arange(0, d_embed_half, 2, device=device).float() * (-math.log(base) / d_embed_half))
    
    freq_embed_x = torch.zeros(H, d_embed_half, device=device)
    freq_embed_y = torch.zeros(W_half, d_embed_half, device=device)
    
    # For x-axis: unsqueeze div_term_x so multiplication broadcasts
    freq_embed_x[:, 0::2] = torch.sin(k_x * div_term_x.unsqueeze(0))
    freq_embed_x[:, 1::2] = torch.cos(k_x * div_term_x.unsqueeze(0))
    # For y-axis:
    freq_embed_y[:, 0::2] = torch.sin(k_y * div_term_y.unsqueeze(0))
    freq_embed_y[:, 1::2] = torch.cos(k_y * div_term_y.unsqueeze(0))
    
    freq_embed_x2d = freq_embed_x.unsqueeze(1).expand(-1, W_half, -1)  # (H, W_half, d_embed_half)
    freq_embed_y2d = freq_embed_y.unsqueeze(0).expand(H, -1, -1)         # (H, W_half, d_embed_half)
    return torch.cat([freq_embed_x2d, freq_embed_y2d], dim=-1)           # (H, W_half, d_embed)

#############################
# Helper: Wavelength Mask Builder
#############################
def build_wavelength_mask(H, W_half, wavelength_crop_x, wavelength_crop_y, domain_size, device='cpu'):
    """
    Builds a boolean mask (of shape [H, W_half]) selecting FFT indices whose
    corresponding wavelengths (λ = domain_size / frequency) fall within the given ranges.
    
    For the x-axis (unshifted):
      - For kx ≤ H//2, frequency = kx; for kx > H//2, frequency = H - kx.
      - Wavelength λₓ = domain_size / frequency (with λₓ = ∞ if frequency==0).
    For the y-axis:
      - Wavelength λᵧ = domain_size / ky (with λᵧ = ∞ if ky==0).
      
    The mask is True if:
      wavelength_crop_x[0] ≤ λₓ ≤ wavelength_crop_x[1] and
      wavelength_crop_y[0] ≤ λᵧ ≤ wavelength_crop_y[1].
    """
    mask = torch.zeros(H, W_half, dtype=torch.bool, device=device)
    for kx in range(H):
        freq_x = kx if kx <= H // 2 else H - kx
        lambda_x = float('inf') if freq_x == 0 else domain_size / freq_x
        for ky in range(W_half):
            freq_y = ky  # ky is nonnegative in rfft
            lambda_y = float('inf') if freq_y == 0 else domain_size / freq_y
            if (wavelength_crop_x[0] <= lambda_x <= wavelength_crop_x[1]) and \
               (wavelength_crop_y[0] <= lambda_y <= wavelength_crop_y[1]):
                mask[kx, ky] = True
    return mask

#############################
# AdaFNO2D Module
#############################
class AdaFNO2D(nn.Module):
    def __init__(
        self,
        hidden_size,            # internal FFT-domain channel count (must be divisible by num_blocks)
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1,
        hidden_size_factor=1,
        res_mult=1,
        channel_first=True,
        domain_size=64,         # training domain size (assumed square)
        # Wavelength crop parameters as tuples: (min_λ, max_λ)
        wavelength_crop_x=(8, 64),
        wavelength_crop_y=(8, 64),
        base=10000.0
    ):
        """
        AdaFNO2D that applies AFNO filters only to FFT coefficients whose corresponding
        wavelengths (computed using domain_size) fall within the user-specified ranges.
        
        Args:
          hidden_size: Internal channel dimension (should be >= input_channels).
          num_blocks: Must divide hidden_size.
          domain_size: The training domain size (assumed square).
          wavelength_crop_x: Tuple (min_λₓ, max_λₓ) for x-axis wavelengths.
          wavelength_crop_y: Tuple (min_λᵧ, max_λᵧ) for y-axis wavelengths.
          freq_emb_dim: Dimension for the sinusoidal frequency embedding.
          emb_scale: Initial weight for frequency embedding addition.
        """
        super().__init__()
        assert hidden_size % num_blocks == 0, "hidden_size must be divisible by num_blocks"
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks

        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.channel_first = channel_first
        self.res_mult = res_mult

        self.domain_size = domain_size
        self.wavelength_crop_x = wavelength_crop_x
        self.wavelength_crop_y = wavelength_crop_y


        # Frequency embedding parameters.
        self.base = base
        self.freq_emb_alpha = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

        # Precompute the wavelength-based mask for the expected training domain.
        H = domain_size    # assume training input has height = domain_size
        W_half = domain_size // 2 + 1
        wavelength_mask = build_wavelength_mask(H, W_half, wavelength_crop_x, wavelength_crop_y, domain_size, device="cpu")
        self.register_buffer("wavelength_mask", wavelength_mask)

        # Define linear layers per block.
        self.first_r = nn.ModuleList()
        self.first_i = nn.ModuleList()
        self.second_r = nn.ModuleList()
        self.second_i = nn.ModuleList()
        for _ in range(num_blocks):
            self.first_r.append(nn.Linear(self.block_size, self.block_size * hidden_size_factor, bias=True))
            self.first_i.append(nn.Linear(self.block_size, self.block_size * hidden_size_factor, bias=True))
            self.second_r.append(nn.Linear(self.block_size * hidden_size_factor, self.block_size, bias=True))
            self.second_i.append(nn.Linear(self.block_size * hidden_size_factor, self.block_size, bias=True))
        

    def _get_sinusoidal_embedding(self, B, H, W_half, device):
        freq_embed = build_sinusoidal_freq_embedding(H, W_half, self.hidden_size, base=self.base, device=device)
        return freq_embed.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W_half, freq_emb_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
          Input x: shape (B, C, H, W) if channel_first, else (B, H, W, C).
          Output: Same spatial shape as input, in channel-first format if channel_first.
        """
        if self.channel_first:
            # Project input from input_channels to hidden_size.
            x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, hidden_size]

        B, H, W, _ = x.shape
        x = x.float()
        device = x.device

        # 1) Compute FFT along spatial dimensions.
        x_fft = torch.fft.rfftn(x, dim=(1, 2), norm='ortho')  # [B, H, W_half, hidden_size]

        # 2) Optionally add sinusoidal frequency embedding.
        W_half = x_fft.shape[2]
        freq_emb = self._get_sinusoidal_embedding(B, H, W_half, device)  # [B, H, W_half, freq_emb_dim]
        x_fft = x_fft + self.freq_emb_alpha * freq_emb

        # 3) Reshape FFT result to separate blocks: [B, H, W_half, num_blocks, block_size]
        x_fft = x_fft.view(B, H, x_fft.shape[2], self.num_blocks, self.block_size)

        # Prepare buffers for the two-stage transform.
        o1_real = torch.zeros((B, H, x_fft.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor), device=device)
        o1_imag = torch.zeros_like(o1_real)
        o2_real = torch.zeros((B, H, x_fft.shape[2], self.num_blocks, self.block_size), device=device)
        o2_imag = torch.zeros_like(o2_real)

        # Use the wavelength-based mask.
        # The mask has shape [H, W_half]. We flatten spatial dimensions.
        if H == self.domain_size:
            mask = self.wavelength_mask  # [H, W_half]
        else:
            mask = build_wavelength_mask(H, W_half, self.wavelength_crop_x, self.wavelength_crop_y, H, device=device)
        mask_flat = mask.view(-1)    # [H*W_half]

        # Process each block (flattening over B, H, and W_half).
        for b in range(self.num_blocks):
            # Extract block b: shape [B, H, W_half, block_size]
            x_b = x_fft[:, :, :, b, :]
            x_b_real = x_b.real  # [B, H, W_half, block_size]
            x_b_imag = x_b.imag  # [B, H, W_half, block_size]
            # Flatten spatial dims together (B * H * W_half, block_size)
            x_b_real_flat = x_b_real.view(B, H * x_fft.shape[2], self.block_size)
            x_b_imag_flat = x_b_imag.view(B, H * x_fft.shape[2], self.block_size)

            # Select masked coefficients.
            x_b_real_masked = x_b_real_flat[:, mask_flat]
            x_b_imag_masked = x_b_imag_flat[:, mask_flat]

            # First linear transform.
            out_r1 = self.first_r[b](x_b_real_masked) - self.first_i[b](x_b_imag_masked)
            out_i1 = self.first_r[b](x_b_imag_masked) + self.first_i[b](x_b_real_masked)

            # Prepare full output tensor for block b.
            o1_block_real = torch.zeros(B, H * x_fft.shape[2], self.block_size * self.hidden_size_factor, device=device, dtype=out_r1.dtype)
            o1_block_imag = torch.zeros_like(o1_block_real)
            o1_block_real[:, mask_flat] = out_r1
            o1_block_imag[:, mask_flat] = out_i1
            o1_block_real = o1_block_real.view(B, H, x_fft.shape[2], self.block_size * self.hidden_size_factor)
            o1_block_imag = o1_block_imag.view(B, H, x_fft.shape[2], self.block_size * self.hidden_size_factor)
            o1_real[:, :, :, b, :] = o1_block_real
            o1_imag[:, :, :, b, :] = o1_block_imag

            # Second linear transform.
            o1_block_real_flat = o1_block_real.reshape(B, H * x_fft.shape[2], self.block_size * self.hidden_size_factor)
            o1_block_imag_flat = o1_block_imag.reshape(B, H * x_fft.shape[2], self.block_size * self.hidden_size_factor)
            o1_real_masked = o1_block_real_flat[:, mask_flat]
            o1_imag_masked = o1_block_imag_flat[:, mask_flat]
            
            out_r2 = self.second_r[b](o1_real_masked) - self.second_i[b](o1_imag_masked)
            out_i2 = self.second_r[b](o1_imag_masked) + self.second_i[b](o1_real_masked)
            
            o2_block_real = torch.zeros(B, H * x_fft.shape[2], self.block_size, device=device, dtype=out_r2.dtype)
            o2_block_imag = torch.zeros_like(o2_block_real)
            o2_block_real[:, mask_flat] = out_r2
            o2_block_imag[:, mask_flat] = out_i2
            o2_block_real = o2_block_real.view(B, H, x_fft.shape[2], self.block_size)
            o2_block_imag = o2_block_imag.view(B, H, x_fft.shape[2], self.block_size)
            o2_real[:, :, :, b, :] = o2_block_real
            o2_imag[:, :, :, b, :] = o2_block_imag

        # 4) Combine real and imaginary parts and apply softshrink.
        x_c = torch.stack([o2_real, o2_imag], dim=-1)  # [B, H, W_half, num_blocks, block_size, 2]
        x_c = F.softshrink(x_c, lambd=self.sparsity_threshold)
        x_c = torch.view_as_complex(x_c)  # [B, H, W_half, num_blocks, block_size]
        # Merge block dimensions to recover [B, H, W_half, hidden_size]
        x_c = x_c.view(B, H, x_fft.shape[2], self.hidden_size)

        # 5) Inverse FFT to return to spatial domain.
        new_H = int(H * self.res_mult)
        new_W = int(W * self.res_mult)
        x_out = torch.fft.irfftn(x_c, s=(new_H, new_W), dim=(1,2), norm='ortho')
        x_out = x_out.to(x.dtype)
        if self.channel_first:
            x_out = x_out.permute(0, 3, 1, 2).contiguous()
        return x_out


class AdaFNOBlock2D(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            drop=0.,
            act_layer=nn.GELU,
            norm='layer',
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
            mlp_out_features=None,
            afno_res_mult=1,
            channel_first=True,
            layer_scale=None,
            domain_size=64,
            wavelength_crop=(8, 64)
    ):
        super().__init__()
        self.norm = norm
        self.afno_res_mult = afno_res_mult
        self.norm1 = normalization(dim, norm)
        self.filter = AdaFNO2D(
            dim, 
            num_blocks, 
            sparsity_threshold,
            hard_thresholding_fraction, 
            res_mult=afno_res_mult,
            channel_first=channel_first,
            domain_size=domain_size,
            wavelength_crop_x=wavelength_crop,
            wavelength_crop_y=wavelength_crop)

        self.norm2 = normalization(dim, norm)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, 
            out_features=mlp_out_features,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer, 
            drop=drop,
            channel_first=channel_first
        )
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
    
    def forward(self, x):
        if not self.layer_scale:
            residual = x
            x = self.norm1(x)
            x = self.filter(x)
            if self.afno_res_mult > 1:
                residual = F.interpolate(residual, x.shape[2:])
            if self.double_skip:
                x = x + residual
                residual = x

            x = self.norm2(x)
            x = self.mlp(x)
            x = x + residual
            return x
        
        residual = x
        x = self.filter(self.norm1(x))
        if self.afno_res_mult > 1:
            residual = F.interpolate(residual, x.shape[2:])
        x = self.gamma1 * x + residual
        # residual = x        
        x = self.gamma2 * self.mlp(self.norm2(x)) + x
        return x

if __name__ == "__main__":
    #############################
# Simple Test Example
#############################
    def test_adfno2d():
        B, C, H, W = 2, 32, 64, 64
        x = torch.randn(B, C, H, W)
        # Here, we set the wavelength crop such that we only process wavelengths between 8 and 64 pixels
        model = AdaFNO2D(
            hidden_size=32,
            num_blocks=1,
            wavelength_crop_x=(0, torch.inf),
            wavelength_crop_y=(0, torch.inf),
            channel_first=True,
            domain_size=64
        )
        with torch.no_grad():
            out = model(x)
        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {out.shape}")
        print("Test successful: Forward pass ran without errors.")

    test_adfno2d()
