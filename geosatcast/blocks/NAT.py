from typing import Optional
import torch 
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from geosatcast.utils import normalization, conv_nd, activation


import natten
from natten import NeighborhoodAttention3D as Nat3D
from natten import NeighborhoodAttention2D as Nat2D
from natten.context import is_fna_enabled
from natten.utils import check_all_args, log
from natten.functional import na3d, na3d_av, na3d_qk, na2d, na2d_av, na2d_qk
from natten.types import CausalArg3DTypeOrDed, Dimension3DTypeOrDed, CausalArg2DTypeOrDed, Dimension2DTypeOrDed
is_natten_post_017 = hasattr(natten, "context")



class CrossNat2D(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: Dimension2DTypeOrDed,
        dilation: Dimension2DTypeOrDed = 1,
        is_causal: CausalArg2DTypeOrDed = False,
        rel_pos_bias: bool = False,
        qv_bias: bool = True,
        k_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        kernel_size_, dilation_, is_causal_ = check_all_args(
            2, kernel_size, dilation, is_causal
        )
        assert len(kernel_size_) == len(dilation_) == len(is_causal_) == 2
        if any(is_causal_) and rel_pos_bias:
            raise NotImplementedError(
                "Causal neighborhood attention is undefined with positional biases."
                "Please consider disabling positional biases, or open an issue."
            )

        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_size = kernel_size_
        self.dilation = dilation_
        self.is_causal = is_causal_

        self.qv = nn.Linear(dim, dim * 2, bias=qv_bias)
        self.k = nn.Linear(dim, dim, bias=k_bias)
        if rel_pos_bias:
            self.rpb = nn.Parameter(
                torch.zeros(
                    num_heads,
                    (2 * self.kernel_size[0] - 1),
                    (2 * self.kernel_size[1] - 1),
                )
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(self.attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.dim() != 4 or y.dim() != 4:
            raise ValueError(
                f"CrossNeighborhoodAttention2D expected a rank-4 input tensor; got x dim {x.dim()=} and y dim {y.dim()=}."
            )

        B, H, W, C = x.shape

        if is_fna_enabled():
            if self.attn_drop_rate > 0:
                logger.error(
                    "You're using fused neighborhood attention, and passed in a "
                    "non-zero attention dropout rate. This implementation does "
                    "support attention dropout yet, which means dropout is NOT being applied "
                    "to your attention weights."
                )

            qv = (
                self.qkv(x)
                .reshape(B, H, W, 3, self.num_heads, self.head_dim)
                .permute(3, 0, 1, 2, 4, 5)
            )
            q, v = qv[0], qv[1]

            k = self.k(y).reshape(B, H, W, self.num_heads, self.head_dim)

            x = na2d(
                q,
                k,
                v,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
                rpb=self.rpb,
                scale=self.scale,
            )
            x = x.reshape(B, H, W, C)

        else:
            qv = (
                self.qkv(x)
                .reshape(B, H, W, 3, self.num_heads, self.head_dim)
                .permute(3, 0, 1, 2, 4, 5)
            )
            q, v = qv[0], qv[1]
            k = self.k(y).reshape(B, H, W, self.num_heads, self.head_dim)
            q = q * self.scale
            attn = na2d_qk(
                q,
                k,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
                rpb=self.rpb,
            )
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = na2d_av(
                attn,
                v,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
            )
            x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)

        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, "
            + f"dilation={self.dilation}, "
            + f"is_causal={self.is_causal}, "
            + f"has_bias={self.rpb is not None}"
        )


class CrossNat3D(nn.Module):
    """
    Neighborhood Attention 3D Module
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: Dimension3DTypeOrDed,
        dilation: Dimension3DTypeOrDed = 1,
        is_causal: CausalArg3DTypeOrDed = False,
        rel_pos_bias: bool = False,
        qv_bias: bool = True,
        k_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        kernel_size_, dilation_, is_causal_ = check_all_args(
            3, kernel_size, dilation, is_causal
        )
        assert len(kernel_size_) == len(dilation_) == len(is_causal_) == 3
        if any(is_causal_) and rel_pos_bias:
            raise NotImplementedError(
                "Causal neighborhood attention is undefined with positional biases."
                "Please consider disabling positional biases, or open an issue."
            )

        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_size = kernel_size_
        self.dilation = dilation_
        self.is_causal = is_causal_

        self.qv = nn.Linear(dim, dim * 2, bias=qv_bias)
        self.k = nn.Linear(dim, dim, bias=k_bias)
        if rel_pos_bias:
            self.rpb = nn.Parameter(
                torch.zeros(
                    num_heads,
                    (2 * self.kernel_size[0] - 1),
                    (2 * self.kernel_size[1] - 1),
                    (2 * self.kernel_size[2] - 1),
                )
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(self.attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"NeighborhoodAttention2D expected a rank-5 input tensor; got {x.dim()=}."
            )

        B, T, H, W, C = x.shape

        if x.shape != y.shape:
            raise ValueError(
                f"x.shape musth match y.shape."
            )

        if is_fna_enabled():
            if self.attn_drop_rate > 0:
                logger.error(
                    "You're using fused neighborhood attention, and passed in a "
                    "non-zero attention dropout rate. This implementation does "
                    "support attention dropout yet, which means dropout is NOT being applied "
                    "to your attention weights."
                )

            qv = (
                self.qv(x)
                .reshape(B, T, H, W, 2, self.num_heads, self.head_dim)
                .permute(4, 0, 1, 2, 3, 5, 6)
            )
            q, v = qv[0], qv[1]
            k = self.k(y).reshape(B, T, H, W, self.num_heads, self.head_dim)
            x = na3d(
                q,
                k,
                v,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
                rpb=self.rpb,
                scale=self.scale,
            )
            x = x.reshape(B, T, H, W, C)

        else:
            qv = (
                self.qv(x)
                .reshape(B, T, H, W, 2, self.num_heads, self.head_dim)
                .permute(4, 0, 5, 1, 2, 3, 6)
            )
            q, v = qv[0], qv[1]
            k = self.k(y).reshape(B, T, H, W, self.num_heads, self.head_dim).permute(0, 4, 1, 2, 3, 5)
            q = q * self.scale
            attn = na3d_qk(
                q,
                k,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
                rpb=self.rpb,
            )
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = na3d_av(
                attn,
                v,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
            )
            x = x.permute(0, 2, 3, 4, 1, 5).reshape(B, T, H, W, C)

        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, "
            + f"dilation={self.dilation}, "
            + f"is_causal={self.is_causal}, "
            + f"has_bias={self.rpb is not None}"
        )



class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act="gelu",
        drop=0.0,
    ):
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


class NATBlock3D(nn.Module):
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
        drop_path=0.0,
        act="gelu",
        norm=None,
        layer_scale=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.mlp_ratio = mlp_ratio

        self.norm1 = normalization(dim, norm)
        extra_args = {"rel_pos_bias": True} if is_natten_post_017 else {"bias": True}
        self.attn = Nat3D(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_blocks,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            **extra_args,
        )
        torch.nn.init.zeros_(self.attn.proj.bias)
        torch.nn.init.zeros_(self.attn.qkv.bias)
        torch.nn.init.zeros_(self.attn.rpb)
        torch.nn.init.xavier_uniform_(self.attn.proj.weight)
        torch.nn.init.xavier_uniform_(self.attn.qkv.weight)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = normalization(dim, norm)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act=act,
            drop=drop,
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
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x

class NATBlock2D(nn.Module):
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
    ):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.mlp_ratio = mlp_ratio

        self.norm1 = normalization(dim, norm)
        extra_args = {"rel_pos_bias": True} if is_natten_post_017 else {"bias": True}
        self.attn = Nat2D(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_blocks,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            **extra_args,
        )
        torch.nn.init.zeros_(self.attn.proj.bias)
        torch.nn.init.zeros_(self.attn.qkv.bias)
        torch.nn.init.xavier_uniform_(self.attn.proj.weight)
        torch.nn.init.xavier_uniform_(self.attn.qkv.weight)

        self.norm2 = normalization(dim, norm)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act=act,
            drop=drop,
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
        
        elif layer_scale == "uniform":
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                torch.rand(dim) * 1e-3, requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                torch.rand(dim) * 1e-3, requires_grad=True
            )

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.attn(self.norm1(x))
            x = shortcut + x
            x = x + self.mlp(self.norm2(x))
            return x
        shortcut = x
        x = self.attn(self.norm1(x))
        x = shortcut + self.gamma1 * x
        x = x + self.gamma2 * self.mlp(self.norm2(x))
        return x

class CrossNATBlock3D(nn.Module):
    def __init__(
        self,
        dim,
        num_blocks,
        kernel_size=7,
        dilation=None,
        mlp_ratio=4.0,
        qv_bias=True,
        k_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act="gelu",
        norm=None,
        layer_scale=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.mlp_ratio = mlp_ratio

        self.norm1 = normalization(dim, norm)
        extra_args = {"rel_pos_bias": True} if is_natten_post_017 else {"bias": True}
        self.attn = CrossNat3D(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_blocks,
            qv_bias=qv_bias,
            k_bias=k_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            **extra_args,
        )
        torch.nn.init.zeros_(self.attn.proj.bias)
        if qv_bias:
            torch.nn.init.zeros_(self.attn.qv.bias)
        if k_bias:
            torch.nn.init.zeros_(self.attn.k.bias)
        torch.nn.init.xavier_uniform_(self.attn.proj.weight)
        torch.nn.init.xavier_uniform_(self.attn.qv.weight)
        torch.nn.init.xavier_uniform_(self.attn.k.weight)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = normalization(dim, norm)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act=act,
            drop=drop,
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

    def forward(self, x, y):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x, y)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, y)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x

class CrossNATBlock2D(nn.Module):
    def __init__(
        self,
        dim,
        num_blocks,
        kernel_size=7,
        dilation=None,
        mlp_ratio=4.0,
        qv_bias=True,
        k_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        act="gelu",
        norm=None,
        layer_scale=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.mlp_ratio = mlp_ratio

        self.norm1 = normalization(dim, norm)
        extra_args = {"rel_pos_bias": True} if is_natten_post_017 else {"bias": True}
        self.attn = CrossNat2D(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_blocks,
            qv_bias=qv_bias,
            k_bias=k_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            **extra_args,
        )
        torch.nn.init.zeros_(self.attn.proj.bias)
        if qv_bias:
            torch.nn.init.zeros_(self.attn.qv.bias)
        if k_bias:
            torch.nn.init.zeros_(self.attn.k.bias)
        torch.nn.init.xavier_uniform_(self.attn.proj.weight)
        torch.nn.init.xavier_uniform_(self.attn.qv.weight)
        torch.nn.init.xavier_uniform_(self.attn.k.weight)

        self.norm2 = normalization(dim, norm)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act=act,
            drop=drop,
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
        
        elif layer_scale == "uniform":
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                torch.rand(dim) * 1e-3, requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                torch.rand(dim) * 1e-3, requires_grad=True
            )

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.attn(self.norm1(x), y)
            x = shortcut + x
            x = x + self.mlp(self.norm2(x))
            return x
        shortcut = x
        x = self.attn(self.norm1(x), y)
        x = shortcut + self.gamma1 * x
        x = x + self.gamma2 * self.mlp(self.norm2(x))
        return x



if __name__ == "__main__":
    nat_block = NATBlock3D(512, 1, (3,3,3))
    x = torch.randn((1,8,64,64,512))
    x = nat_block(x)
    print(x.shape)
