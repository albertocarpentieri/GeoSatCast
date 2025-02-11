import torch
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

    def forward(self, x):
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
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, :, total_modes - kept_modes:total_modes + kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo',
                         x[:, :, total_modes - kept_modes:total_modes + kept_modes, :kept_modes].real, self.w1[0]) -
            torch.einsum('...bi,bio->...bo',
                         x[:, :, total_modes - kept_modes:total_modes + kept_modes, :kept_modes].imag, self.w1[1]) +
            self.b1[0]
        )

        o1_imag[:, :, total_modes - kept_modes:total_modes + kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo',
                         x[:, :, total_modes - kept_modes:total_modes + kept_modes, :kept_modes].imag, self.w1[0]) +
            torch.einsum('...bi,bio->...bo',
                         x[:, :, total_modes - kept_modes:total_modes + kept_modes, :kept_modes].real, self.w1[1]) +
            self.b1[1]
        )

        o2_real[:, :, total_modes - kept_modes:total_modes + kept_modes, :kept_modes] = (
                torch.einsum('...bi,bio->...bo',
                             o1_real[:, :, total_modes - kept_modes:total_modes + kept_modes, :kept_modes],
                             self.w2[0]) -
                torch.einsum('...bi,bio->...bo',
                             o1_imag[:, :, total_modes - kept_modes:total_modes + kept_modes, :kept_modes],
                             self.w2[1]) +
                self.b2[0]
        )

        o2_imag[:, :, total_modes - kept_modes:total_modes + kept_modes, :kept_modes] = (
                torch.einsum('...bi,bio->...bo',
                             o1_imag[:, :, total_modes - kept_modes:total_modes + kept_modes, :kept_modes],
                             self.w2[0]) +
                torch.einsum('...bi,bio->...bo',
                             o1_real[:, :, total_modes - kept_modes:total_modes + kept_modes, :kept_modes],
                             self.w2[1]) +
                self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
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
        