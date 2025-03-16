from typing import Optional, Union, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from geosatcast.utils import normalization, conv_nd, activation
from geosatcast.blocks.RoPe import Rope2DMixed, SphericalRoPE

# ----------------------------------------------------------------------------
# NATTEN imports (no is_fna_enabled usage)
# ----------------------------------------------------------------------------
import natten
from natten.utils import check_all_args
from natten.functional import na2d_qk, na2d_av, na3d_qk, na3d_av

# ----------------------------------------------------------------------------
# 2) Nat2D: self/cross + optional distance-based bias
# ----------------------------------------------------------------------------

class LearnedAbsPosConcatEmbed(nn.Module):
    """
    Computes an absolute positional embedding from 2D coordinates.
    For each head, it learns a frequency vector and computes sine and cosine
    features from the coordinates. The four resulting features are concatenated and
    then passed through a linear layer to produce an embedding of dimension pos_dim.
    The output shape is [B, num_heads, H, W, pos_dim] so that it can be concatenated
    to the query and key.
    """
    def __init__(self, num_heads: int, pos_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = pos_dim
        self.freq_x = nn.Parameter(torch.randn(num_heads, pos_dim))
        self.freq_y = nn.Parameter(torch.randn(num_heads, pos_dim))
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: Tensor of shape [B, H, W, 2], where the last dim are (x, y) coordinates.
        Returns:
            pos_emb: Tensor of shape [B, num_heads, H, W, pos_dim]
        """
        B, H, W, _ = coords.shape
        coords = coords.unsqueeze(3).expand(B, H, W, self.num_heads, 2)
        x = coords[..., 0].unsqueeze(-1) * self.freq_x 
        y = coords[..., 1].unsqueeze(-1) * self.freq_y      
        pos_emb = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=-1)
        # pos_emb = self.linear(pos_features)
        pos_emb = pos_emb.permute(0, 3, 1, 2, 4)
        return pos_emb

class NonlinearPosBiasMLP(nn.Module):
    """
    Maps an (offset_y, offset_x) to a single scalar bias:
        offset: [N, 2] --> [N, 1]
    where N = (2*kH - 1)*(2*kW - 1).
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.mlp = Mlp(in_features=2, hidden_features=hidden_dim, out_features=1)

    def forward(self, offset_grid: torch.Tensor) -> torch.Tensor:
        """
        offset_grid shape => [H, W, 2]  (H = 2*kH - 1, W = 2*kW - 1)
        Returns shape => [H, W]
        """
        H, W, _ = offset_grid.shape
        flat_inp = offset_grid.view(-1, 2)
        flat_out = self.mlp(flat_inp)
        return flat_out.view(H, W)


class SinusoidalAbsPosEmbed(nn.Module):
    """
    Given a 2D coordinate (x, y), produces a sinusoidal positional embedding.
    The embedding is computed as the concatenation of:
      sin(x * freq_x), cos(x * freq_x), sin(y * freq_y), cos(y * freq_y)
    where freq_x and freq_y are learnable frequency parameters.
    For an output embedding of dimension `dim`, we require that dim % 4 == 0.
    """
    def __init__(self, dim: int):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("dim must be divisible by 4 for SinusoidalAbsPosEmbed")
        self.dim = dim
        self.freq_x = nn.Parameter(torch.randn(dim // 4))
        self.freq_y = nn.Parameter(torch.randn(dim // 4))
    
    def forward(self, coords: Tensor) -> Tensor:
        """
        coords: [B, H, W, 2] tensor where:
            coords[..., 0] is the x coordinate (distance along East-West),
            coords[..., 1] is the y coordinate (distance along North-South).
        Returns:
            pos_emb: [B, H, W, dim] sinusoidal positional embedding.
        """
        # Separate x and y coordinates and unsqueeze for multiplication.
        x = coords[..., 0].unsqueeze(-1)  
        y = coords[..., 1].unsqueeze(-1)  
        sin_x = torch.sin(x * self.freq_x)  
        cos_x = torch.cos(x * self.freq_x)  
        sin_y = torch.sin(y * self.freq_y)  
        cos_y = torch.cos(y * self.freq_y)
        pos_emb = torch.cat([sin_x, cos_x, sin_y, cos_y], dim=-1)
        return pos_emb

class Nat2D(nn.Module):
    """
    Neighborhood Attention 2D for self-attention or cross-attention (cross=True).

    If rel_dist_bias=True, we define one nn.Parameter (rel_dist_scale) for lat/lon,
    then retrieve partial offsets from compute_relative_distance_map(...) => (2, H, W, k).
    We'll compute (offset_x * scale)^2 + (offset_y * scale)^2 as a distance, then subtract it 
    from the attention logits (or add a negative distance).
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: Union[int, Tuple[int,int]],
        dilation: Union[int, Tuple[int,int]] = 1,
        is_causal: Union[bool, Tuple[bool,bool]] = False,
        cross: bool = False,
        qkv_bias: bool = True,
        qv_bias: bool = True,
        k_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        emb_method: str = "rope", # from ["rope", "corrected_rope", "spherical_rope", "rel_dist_bias", "rel_pos_bias", "nonlinear_dist_bias" "none"]
        resolution: Tuple[float, float] = [1,1] 
    ):
        super().__init__()
        kernel_size_, dilation_, is_causal_ = check_all_args(2, kernel_size, dilation, is_causal)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or (self.head_dim**-0.5)
        self.kernel_size = kernel_size_
        self.dilation = dilation_
        self.is_causal = is_causal_
        self.cross = cross
        self.emb_method = emb_method
        self.resolution = resolution

        # QKV for self-attn or QV+K for cross-attn
        if not self.cross:
            self.qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
            torch.nn.init.xavier_uniform_(self.qkv.weight)
            torch.nn.init.zeros_(self.qkv.bias)
        else:
            self.qv = nn.Linear(dim, 2*dim, bias=qv_bias)
            self.k = nn.Linear(dim, dim, bias=k_bias)
            torch.nn.init.xavier_uniform_(self.qv.weight)
            torch.nn.init.zeros_(self.qv.bias)
            torch.nn.init.xavier_uniform_(self.k.weight)
            torch.nn.init.zeros_(self.k.bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # relative position bias
        self.rpb = None
        if self.emb_method == "rel_pos_bias":
            self.rpb = nn.Parameter(
                torch.zeros(
                    num_heads,
                    (2*self.kernel_size[0]-1),
                    (2*self.kernel_size[1]-1),
                )
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)

        # distance-based bias => single param
        self.register_buffer("offset", None)
        if self.emb_method == "rel_dist_bias":
            self.rpb = nn.Parameter(torch.zeros(num_heads, 2))
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
            kH, kW = self.kernel_size
            offset = torch.zeros(
                num_heads,
                2,
                (2*kH - 1),
                (2*kW - 1),
            )
            for h in range(num_heads):
                yvals = torch.abs(torch.arange(-kH + 1, kH))
                xvals = torch.abs(torch.arange(-kW + 1, kW))
                mesh_y = yvals.unsqueeze(1).expand(kH*2 - 1, kW*2 - 1)
                mesh_x = xvals.unsqueeze(0).expand(kH*2 - 1, kW*2 - 1)
                offset[h, 0] = mesh_y  
                offset[h, 1] = mesh_x
            self.register_buffer("offset", offset)
        
        if self.emb_method == "nonlinear_dist_bias":
            kH, kW = self.kernel_size
            yvals = torch.arange(-kH+1, kH) * self.resolution[0]
            xvals = torch.arange(-kW+1, kW) * self.resolution[1]
            grid_y = yvals[:, None].expand(yvals.size(0), xvals.size(0))  # [2*kH-1, 2*kW-1]
            grid_x = xvals[None, :].expand(yvals.size(0), xvals.size(0))  # same shape
            offset_grid = torch.stack([grid_y, grid_x], dim=-1)  # [2*kH-1, 2*kW-1, 2]
            self.nlpe = NonlinearPosBiasMLP(hidden_dim=64)
            self.register_buffer("offset_grid", offset_grid.type(self.nlpe.mlp.fc1.weight.dtype))
        
        if self.emb_method == "abs_emb":
            self.abs_embed = SinusoidalAbsPosEmbed(dim)
        
        # Modified absolute embedding branch:
        if self.emb_method == "concat_abs_emb":
            self.abs_embed = LearnedAbsPosConcatEmbed(num_heads=self.num_heads, pos_dim=32)

        self.rope = None
        if self.emb_method == "rope":
            self.rope = Rope2DMixed(num_heads, self.head_dim)
        
        if self.emb_method == "corrected_rope":
            self.rope = Rope2DMixed(num_heads, self.head_dim, spherical_correction=True)
        
        if self.emb_method == "spherical_rope":
            self.rope = SphericalRoPE(num_heads, self.head_dim)

    def forward(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
        coords: Optional[Tensor] = None,
    ) -> Tensor:
        """
        x => [B,H,W,C]
        y => if cross=True => second input. If None => self-attn.
        """
        if x.dim() != 4:
            raise ValueError("Nat2D expects [B,H,W,C] rank-4 input.")
        B, H, W, C = x.shape

        # Apply absolute sinusoidal positional embedding if requested.
        if self.emb_method == "abs_emb" and coords is not None:
            lat = coords[..., 0]  
            lon = coords[..., 1]
            pos_emb = self.abs_embed(coords)
            x = x + pos_emb
            if self.cross and y is not None:
                y = y + pos_emb


        if not self.cross:
            # self-attn
            qkv = self.qkv(x).reshape(B,H,W,3,self.num_heads,self.head_dim)
            qkv = qkv.permute(3,0,4,1,2,5)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            if y is None:
                raise ValueError("Nat2D with cross=True requires a second tensor y.")
            if y.shape != x.shape:
                raise ValueError("For cross-attn in Nat2D, x.shape must match y.shape.")
            qv = self.qv(x).reshape(B,H,W,2,self.num_heads,self.head_dim)
            qv = qv.permute(3,0,4,1,2,5)
            q, v = qv[0], qv[1]
            k = self.k(y).reshape(B,H,W,self.num_heads,self.head_dim).permute(0,3,1,2,4)

        q = q * self.scale

        rpb = self.rpb
        if self.emb_method == "rel_dist_bias":
            scale = rpb.view(self.num_heads, 2, 1, 1)  # => [num_heads,2,1,1]
            dist = (self.offset * scale).pow(2).sum(dim=1)  # => [num_heads,H,W]
            rpb = - dist  # negative distance

        elif self.emb_method == "nonlinear_dist_bias":
            rpb = self.nlpe(self.offset_grid)
            rpb = rpb.unsqueeze(0).expand(self.num_heads, -1, -1)
        
        elif self.emb_method in ["rope", "corrected_rope", "spherical_rope"]:
            q, k = self.rope(q, k, coords)        
        
        # For "abs_emb", compute the learned absolute positional embedding and concatenate it to q and k.
        if self.emb_method == "concat_abs_emb":
            pos_emb = self.abs_embed(coords)  
            q = torch.cat([q, pos_emb], dim=-1)
            k = torch.cat([k, pos_emb], dim=-1)

        attn = na2d_qk(
            q, k,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            is_causal=self.is_causal,
            rpb=rpb,
        ) 

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = na2d_av(
            attn, v,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            is_causal=self.is_causal,
        )
        out = out.permute(0,2,3,1,4).reshape(B,H,W,C)
        out = self.proj_drop(self.proj(out))
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act="gelu", drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = activation(act)
        self.fc2 = nn.Linear(hidden_features, out_features)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class NATBlock2D(nn.Module):
    """
    Unifies self- and cross-attention in 2D. 
    If cross=True => pass (x, y) to forward. If cross=False => just x => self-attn.
    If rel_dist_bias=True => single param scale in the embedded Nat2D.
    """
    def __init__(
        self,
        dim,
        num_blocks,
        kernel_size=7,
        dilation=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        act="gelu",
        norm=None,
        layer_scale=None,
        cross=False,
        emb_method="rel_pos_bias",
        resolution=1.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.cross = cross

        self.norm1 = normalization(dim, norm)
        # Build Nat2D (self or cross) with optional distance bias
        self.attn = Nat2D(
            dim=dim,
            num_heads=num_blocks,
            kernel_size=kernel_size,
            dilation=dilation or 1,
            is_causal=False,
            cross=cross,
            qkv_bias=qkv_bias,
            qv_bias=qkv_bias,
            k_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            emb_method=emb_method,
            resolution=resolution,
        )

        self.norm2 = normalization(dim, norm)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act=act,
            drop=drop,
        )

        self.layer_scale = False
        if layer_scale is not None and isinstance(layer_scale, (int, float)):
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim))
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(
        self, 
        x: Tensor, 
        y: Optional[Tensor] = None, 
        coords: Optional[Tensor] = None) -> Tensor:
        """
        If cross=True => pass (x, y). If cross=False => pass x only => self-attn.
        """
        shortcut = x
        x = self.norm1(x)
        # pass x,y to self.attn if cross => else y=None
        x_attn = self.attn(x, y=y, coords=coords)

        if not self.layer_scale:
            x = shortcut + x_attn
            x = x + self.mlp(self.norm2(x))
        else:
            x = shortcut + self.gamma1 * x_attn
            x = x + self.gamma2 * self.mlp(self.norm2(x))
        return x

# ----------------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2D self-attention
    b, h, w, c = 2, 32, 32, 64
    x2d = torch.randn(b, h, w, c, device=device)
    block2d = NATBlock2D(
        dim=c,
        num_blocks=4,
        kernel_size=(7,7),
        cross=False,          # self-attn
        rel_dist_bias=True,   # add distance-based offset
        resolution=1.0
    ).to(device)
    out2d = block2d(x2d)
    print("NATBlock2D self-attn shape:", out2d.shape)

    # 2D cross-attn
    x2d_q = torch.randn(b, h, w, c, device=device)
    x2d_kv = torch.randn(b, h, w, c, device=device)
    block2d_cross = NATBlock2D(
        dim=c,
        num_blocks=4,
        kernel_size=(5,5),
        cross=True,           # cross-attn
        rel_dist_bias=True
    ).to(device)
    out2d_cross = block2d_cross(x2d_q, y=x2d_kv)
    print("NATBlock2D cross-attn shape:", out2d_cross.shape)
