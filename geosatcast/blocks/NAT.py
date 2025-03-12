from typing import Optional, Union, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from geosatcast.utils import normalization, conv_nd, activation
from geosatcast.blocks.RoPe import Rope2DMixed

# ----------------------------------------------------------------------------
# NATTEN imports (no is_fna_enabled usage)
# ----------------------------------------------------------------------------
import natten
from natten.utils import check_all_args
from natten.functional import na2d_qk, na2d_av, na3d_qk, na3d_av

# ----------------------------------------------------------------------------
# 2) Nat2D: self/cross + optional distance-based bias
# ----------------------------------------------------------------------------

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
        # Flatten [H*W, 2]
        flat_inp = offset_grid.view(-1, 2)
        # MLP => [H*W, 1]
        flat_out = self.mlp(flat_inp)
        # Reshape back => [H, W]
        return flat_out.view(H, W)


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
        rel_pos_bias: bool = False,
        rel_dist_bias: bool = False,
        resolution: Union[float, Tuple[float,float]] = 1.0,
        use_rope: bool = False,
        use_nl_rdb: bool = False,
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
        self.rel_dist_bias = rel_dist_bias
        if isinstance(resolution, float):
            resolution = [resolution, resolution]
        self.resolution = resolution

        # QKV for self-attn or QV+K for cross-attn
        if not self.cross:
            self.qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
        else:
            self.qv = nn.Linear(dim, 2*dim, bias=qv_bias)
            self.k = nn.Linear(dim, dim, bias=k_bias)

        # relative position bias
        self.rpb = None
        if rel_pos_bias:
            self.rpb = nn.Parameter(
                torch.zeros(
                    num_heads,
                    (2*self.kernel_size[0]-1),
                    (2*self.kernel_size[1]-1),
                )
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # distance-based bias => single param
        if self.rel_dist_bias:
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
                # 1D coords:
                yvals = torch.abs(torch.arange(-kH + 1, kH)) * self.resolution[0]
                xvals = torch.abs(torch.arange(-kW + 1, kW)) * self.resolution[1]
                mesh_y = yvals.unsqueeze(1).expand(kH*2 - 1, kW*2 - 1)
                mesh_x = xvals.unsqueeze(0).expand(kH*2 - 1, kW*2 - 1)
                offset[h, 0] = mesh_y  # lat dimension
                offset[h, 1] = mesh_x  # lon dimension

            # Make it a buffer, so it moves with .to(device) etc.
            self.register_buffer("offset", offset)
        
        self.use_rope = use_rope
        if self.use_rope:
            self.rope2d = Rope2DMixed(num_heads, self.head_dim)
        else:
            self.rope2d = None
        
        self.nonlinear_rpb = use_nl_rdb
        if self.nonlinear_rpb:
            kH, kW = self.kernel_size
            # Build an offset mesh => shape [2*kH - 1, 2*kW - 1, 2]
            #   offset_grid[y, x, :] = (offset_y, offset_x)
            yvals = torch.arange(-kH+1, kH) * self.resolution[0]
            xvals = torch.arange(-kW+1, kW) * self.resolution[1]
            grid_y = yvals[:, None].expand(yvals.size(0), xvals.size(0))  # [2*kH-1, 2*kW-1]
            grid_x = xvals[None, :].expand(yvals.size(0), xvals.size(0))  # same shape
            offset_grid = torch.stack([grid_y, grid_x], dim=-1)  # [2*kH-1, 2*kW-1, 2]
            self.nlpe = NonlinearPosBiasMLP(hidden_dim=64)
            self.register_buffer("offset_grid", offset_grid.type(self.nlpe.mlp.fc1.weight.dtype))
            

    def forward(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
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

        rpb = self.rpb
        if self.rel_dist_bias:
            scale = rpb.view(self.num_heads, 2, 1, 1)  # => [num_heads,2,1,1]
            dist = (self.offset * scale).pow(2).sum(dim=1)  # => [num_heads,H,W]
            rpb = - dist  # negative distance

        if self.use_rope:
            # q, k => [B,heads,H,W,head_dim]
            q, k = self.rope2d(q, k, H, W, self.resolution)
        
        if self.nonlinear_rpb:
            # shape => [2*kH-1, 2*kW-1]
            rpb_2d = self.nlpe(self.offset_grid)
            # broadcast => [num_heads, 2*kH-1, 2*kW-1]
            rpb = rpb_2d.unsqueeze(0).expand(self.num_heads, -1, -1)
        
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

# ----------------------------------------------------------------------------
# 3) Nat3D: self/cross + distance-based bias (time + lat/lon)
# ----------------------------------------------------------------------------
class Nat3D(nn.Module):
    """
    Neighborhood Attention 3D. If rel_dist_bias=True, we define two parameters:
      rel_dist_scale (for lat/lon) and rel_time_scale (for time).
    We'll retrieve partial coords => shape (3,D,H,W,kProd), combine them, and subtract.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: Union[int, Tuple[int,int,int]],
        dilation: Union[int, Tuple[int,int,int]] = 1,
        is_causal: Union[bool, Tuple[bool,bool,bool]] = False,
        rel_pos_bias: bool = False,
        rel_dist_bias: bool = False,
        resolution: Union[float, Tuple[float,float,float]] = 1.0,
        cross: bool = False,
        qkv_bias: bool = True,
        qv_bias: bool = True,
        k_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        kernel_size_, dilation_, is_causal_ = check_all_args(3, kernel_size, dilation, is_causal)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or (self.head_dim**-0.5)
        self.kernel_size = kernel_size_
        self.dilation = dilation_
        self.is_causal = is_causal_
        self.cross = cross
        self.rel_dist_bias = rel_dist_bias
        if isinstance(resolution, float):
            self.resolution = [resolution, resolution, resolution]
        self.resolution = resolution

        if not self.cross:
            self.qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
        else:
            self.qv = nn.Linear(dim, 2*dim, bias=qv_bias)
            self.k = nn.Linear(dim, dim, bias=k_bias)

        self.rpb = None
        if rel_pos_bias:
            self.rpb = nn.Parameter(
                torch.zeros(
                    num_heads,
                    (2*self.kernel_size[0]-1),
                    (2*self.kernel_size[1]-1),
                    (2*self.kernel_size[2]-1),
                )
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # distance-based bias => single param
        if self.rel_dist_bias:
            self.rpb = nn.Parameter(torch.ones(num_heads, 3))
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)

            kT, kH, kW = self.kernel_size
            offset = torch.zeros(
                num_heads,
                3,
                (2*kT - 1),
                (2*kH - 1),
                (2*kW - 1),
            )
            for h in range(num_heads):
                # 1D coords:
                tvals = torch.arange(-kT + 1, kT) * self.resolution[0]
                yvals = torch.arange(-kH + 1, kH) * self.resolution[0]
                xvals = torch.arange(-kW + 1, kW) * self.resolution[1]
                mesh_t = tvals.unsqueeze(1).expand(kT*2-1, kH*2 - 1, kW*2 - 1)
                mesh_y = yvals.unsqueeze(1).expand(kT*2-1, kH*2 - 1, kW*2 - 1)
                mesh_x = xvals.unsqueeze(0).expand(kT*2-1, kH*2 - 1, kW*2 - 1)
                offset[h, 0] = mesh_t  # lat dimension
                offset[h, 1] = mesh_y  # lat dimension
                offset[h, 2] = mesh_x  # lon dimension

            # Make it a buffer, so it moves with .to(device) etc.
            self.register_buffer("offset", offset)

    def forward(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
    ) -> Tensor:
        """
        x => [B,D,H,W,C]
        y => if cross=True => second input. Else None => self-attn.
        """
        if x.dim() != 5:
            raise ValueError("Nat3D expects [B,D,H,W,C] rank-5 input.")
        B, D, H, W, C = x.shape

        if not self.cross:
            qkv = self.qkv(x).reshape(B,D,H,W,3,self.num_heads,self.head_dim)
            qkv = qkv.permute(4,0,5,1,2,3,6)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            if y is None:
                raise ValueError("Nat3D with cross=True requires a second tensor y.")
            if y.shape != x.shape:
                raise ValueError("For cross-attn in Nat3D, x.shape must match y.shape.")
            qv = self.qv(x).reshape(B,D,H,W,2,self.num_heads,self.head_dim)
            qv = qv.permute(4,0,5,1,2,3,6)
            q, v = qv[0], qv[1]
            k = self.k(y).reshape(B,D,H,W,self.num_heads,self.head_dim)
            k = k.permute(0,4,1,2,3,5)

        q = q * self.scale

        rpb = self.rpb
        if self.rel_dist_bias:
            scale = rpb.view(self.num_heads, 2, 1, 1)  # => [num_heads,2,1,1]
            dist = (self.offset * scale).pow(2).sum(dim=1)  # => [num_heads,H,W]
            rpb = - dist  # negative distance
        
        attn = na3d_qk(
            q, k,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            is_causal=self.is_causal,
            rpb=rpb,
        )  # => [B,nHeads,D,H,W,kProd]
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = na3d_av(
            attn, v,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            is_causal=self.is_causal,
        )
        out = out.permute(0,2,3,4,1,5).reshape(B,D,H,W,C)
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
        rpb=True,
        cross=False,
        rel_dist_bias=False,
        use_rope=False,
        use_nl_rdb=False,
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
            rel_pos_bias=rpb,
            rel_dist_bias=rel_dist_bias,
            resolution=resolution,
            cross=cross,
            qkv_bias=qkv_bias,
            qv_bias=qkv_bias,
            k_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_rope=use_rope,
            use_nl_rdb=use_nl_rdb
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
        if self.attn.rpb is not None and not self.attn.rel_dist_bias and not use_nl_rdb:
            nn.init.zeros_(self.attn.rpb)

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

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        If cross=True => pass (x, y). If cross=False => pass x only => self-attn.
        """
        shortcut = x
        x = self.norm1(x)
        # pass x,y to self.attn if cross => else y=None
        x_attn = self.attn(x, y=y)

        if not self.layer_scale:
            x = shortcut + x_attn
            x = x + self.mlp(self.norm2(x))
        else:
            x = shortcut + self.gamma1 * x_attn
            x = x + self.gamma2 * self.mlp(self.norm2(x))
        return x


class NATBlock3D(nn.Module):
    """
    Unifies self- and cross-attention in 3D.
    If cross=True => pass (x, y). If cross=False => pass x only => self-attn.
    If rel_dist_bias=True => separate time, space scale params in Nat3D.
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
        rpb=True,
        cross=False,
        rel_dist_bias=False,
        resolution=1.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.cross = cross

        self.norm1 = normalization(dim, norm)
        self.attn = Nat3D(
            dim=dim,
            num_heads=num_blocks,
            kernel_size=kernel_size,
            dilation=dilation or 1,
            is_causal=False,
            rel_pos_bias=rpb,
            rel_dist_bias=rel_dist_bias,
            resolution=resolution,
            cross=cross,
            qkv_bias=qkv_bias,
            qv_bias=qkv_bias,
            k_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
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
        if self.attn.rpb is not None and not self.attn.rel_dist_bias:
            nn.init.zeros_(self.attn.rpb)

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

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        If cross=True => pass (x,y). If cross=False => pass x => self-attn.
        """
        shortcut = x
        x = self.norm1(x)
        x_attn = self.attn(x, y=y)

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

    # 3D self-attn
    b, d, h, w, c3 = 2, 8, 16, 16, 32
    x3d = torch.randn(b, d, h, w, c3, device=device)
    block3d = NATBlock3D(
        dim=c3,
        num_blocks=2,
        kernel_size=(3,3,3),
        cross=False,
        rel_dist_bias=True,  # separate time vs. space scale
        resolution=(1.0,1.0,1.0)
    ).to(device)
    out3d = block3d(x3d)
    print("NATBlock3D self-attn shape:", out3d.shape)
