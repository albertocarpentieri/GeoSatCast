import torch 
import torch.nn as nn
import torch.nn.functional as F
import natten
from natten import NeighborhoodAttention3D as Nat3D
from natten import NeighborhoodAttention2D as Nat2D

from geosatcast.utils import normalization, conv_nd, activation
is_natten_post_017 = hasattr(natten, "context")

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
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        
        self.act = activation(act)
        
        self.fc2 = nn.Linear(hidden_features, out_features)
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
        num_heads,
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
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = normalization(dim, norm)
        extra_args = {"rel_pos_bias": True} if is_natten_post_017 else {"bias": True}
        self.attn = Nat3D(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
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


if __name__ == "__main__":
    nat_block = NATBlock2D(512, 1, (3,3))
    x = torch.randn((1,64,64,512))
    x = nat_block(x)
    print(x.shape)
