from typing import Optional, Union, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from geosatcast.utils import normalization, conv_nd, activation
from geosatcast.blocks.RoPe import SphericalRoPE

# ----------------------------------------------------------------------------
# NATTEN imports (no is_fna_enabled usage)
# ----------------------------------------------------------------------------
import natten
from natten.utils import check_all_args
from natten.functional import na2d_qk, na2d_av, na3d_qk, na3d_av


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
        use_rope: bool = False,
        cross: bool = False,
        qkv_bias: bool = True,
        qv_bias: bool = True,
        k_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
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

        # QKV for self-attn or QV+K for cross-attn
        if not self.cross:
            self.qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
        else:
            self.qv = nn.Linear(dim, 2*dim, bias=qv_bias)
            self.k = nn.Linear(dim, dim, bias=k_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.use_rope = use_rope
        if self.use_rope:
            self.spherical_rope = SphericalRoPE(num_heads, self.head_dim)
        else:
            self.spherical_rope = None
        

    def forward(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
        grid: Optional[Tensor] = None,
    ) -> Tensor:
        """
        x => [B,H,W,C]
        y => if cross=True => second input. If None => self-attn.
        """
        if x.dim() != 4:
            raise ValueError("Nat2D expects [B,H,W,C] rank-4 input.")
        B, H, W, C = x.shape

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

        if self.use_rope:
            q, k = self.rope2d(q, k, grid)
        
        attn = na2d_qk(
            q, k,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            is_causal=self.is_causal,
            rpb=None,
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

# ----------------------------------------------------------------------------
# 4) Mlp block
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# 5) NATBlock2D / NATBlock3D
#    - Each can do self or cross, via cross=True/False
#    - We unify old cross-block logic here
# ----------------------------------------------------------------------------

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
        use_rope=False,
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
            use_rope=use_rope,
        )
        # weight init
        if not cross:
            nn.init.zeros_(self.attn.qkv.bias)
            nn.init.xavier_uniform_(self.attn.qkv.weight)
        else:
            nn.init.zeros_(self.attn.qv.bias)
            nn.init.zeros_(self.attn.k.bias)
            nn.init.xavier_uniform_(self.attn.qv.weight)
            nn.init.xavier_uniform_(self.attn.k.weight)
        nn.init.zeros_(self.attn.proj.bias)
        nn.init.xavier_uniform_(self.attn.proj.weight)
        

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

    def forward(self, x: Tensor, y: Optional[Tensor] = None, grid: Optional[Tensor] = None) -> Tensor:
        """
        If cross=True => pass (x, y). If cross=False => pass x only => self-attn.
        """
        shortcut = x
        x = self.norm1(x)
        # pass x,y to self.attn if cross => else y=None
        x_attn = self.attn(x, y=y, grid=grid)

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