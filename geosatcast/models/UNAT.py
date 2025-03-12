import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math 

from geosatcast.utils import avg_pool_nd, conv_nd
from geosatcast.blocks.ResBlock import ResBlock3D
from geosatcast.blocks.NAT import NATBlock2D, NATBlock3D


################################################################################
# The UNet with NAT: Configurable Depth, Channels, Skip Connections
################################################################################


class DownBlock(nn.Module):
    """
    A 'downsample' block that reduces spatial resolution (and possibly temporal).
    Internally:
      1) Applies a stride=2 convolution (or 3D conv if nat_dim=3).
      2) Applies NAT-based attention (NATBlock2D or NATBlock3D).
    Returns:
      A tensor with reduced spatial/temporal dimensions.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride=(2,2),
        kernel_size=3,
        nat_dim: int =2,
        norm=None,
        layer_scale=0,
        mlp_ratio=4,
        num_blocks=2,
        depth=1,
        rpb=True,
        rel_dist_bias=False,
        resolution=1.0,
        use_rope=False,
        use_nl_rdb=False
    ):
        super().__init__()
        self.downsample = conv_nd(3, in_ch, out_ch, kernel_size=stride, stride=stride, padding=0, padding_mode="reflect")
        torch.nn.init.xavier_uniform_(self.downsample.weight)
        torch.nn.init.zeros_(self.downsample.bias)
        self.nat_dim = nat_dim
        if depth > 0:
            if nat_dim == 2:
                self.nat_blocks = nn.Sequential(
                    *[NATBlock2D(
                    dim=out_ch,
                    mlp_ratio=mlp_ratio, 
                    num_blocks=num_blocks,
                    norm=norm,
                    kernel_size=kernel_size,
                    layer_scale=layer_scale,
                    rpb=rpb,
                    rel_dist_bias=rel_dist_bias,
                    resolution=resolution,
                    use_rope=use_rope,
                    use_nl_rdb=use_nl_rdb,
                    cross=False,) for _ in range(depth)])
            elif nat_dim == 3:
                self.nat_blocks = nn.Sequential(
                    *[NATBlock3D(
                    dim=out_ch,
                    mlp_ratio=mlp_ratio, 
                    num_blocks=num_blocks,
                    norm=norm,
                    kernel_size=kernel_size,
                    layer_scale=layer_scale,
                    rpb=rpb,
                    rel_dist_bias=rel_dist_bias,
                    resolution=resolution,
                    use_rope=use_rope,
                    cross=False) for _ in range(depth)])
        else:
            self.nat_blocks = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        if self.nat_blocks:
            if self.nat_dim == 2:
                x = x.squeeze(2).permute(0,2,3,1)
            else:
                x = x.permute(0,2,3,4,1)
            for blk in self.nat_blocks: 
                x = checkpoint(blk, x, use_reentrant=False)
            if self.nat_dim == 2:
                x = x.permute(0,3,1,2).unsqueeze(2)
            else:
                x = x.permute(0,4,1,2,3)
        return x


class UpBlock(nn.Module):
    """
    An 'upsample' block that increases spatial resolution.
    Internally:
      1) Uses a transposed convolution (or 3D) to upsample.
      2) Merges a skip connection (via addition or concatenation).
      3) Optionally applies NAT-based attention.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size=(5,5),
        stride=(1,2,2),
        norm=None,
        layer_scale=0,
        mlp_ratio=4,
        num_blocks=2,
        skip_type='add',
        depth=1,
        rpb=True,
        rel_dist_bias=False,
        resolution=1.0,
        use_rope=False,
        use_nl_rdb=False
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=stride, stride=stride, padding=0)
        torch.nn.init.xavier_uniform_(self.upsample.weight)
        torch.nn.init.zeros_(self.upsample.bias)
        if depth>0:
            self.nat_blocks = nn.Sequential(
                *[NATBlock2D(
                dim=out_ch,
                mlp_ratio=mlp_ratio, 
                num_blocks=num_blocks,
                norm=norm,
                kernel_size=kernel_size,
                layer_scale=layer_scale,
                rpb=rpb,
                rel_dist_bias=rel_dist_bias,
                resolution=resolution,
                use_rope=use_rope,
                use_nl_rdb=use_nl_rdb,
                cross=False) for _ in range(depth)])
        else:
            self.nat_blocks = None

        
        self.skip_type = skip_type # (for 'add' or 'concat')
        if self.skip_type == "layer_scale":
            self.skip_scale = nn.Parameter(torch.rand((1,out_ch,1,1,1)) * 1e-3, requires_grad=True)
            

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.upsample(x) 
        if skip is not None:
            if self.skip_type == 'add':
                x = x + skip
            elif self.skip_type == 'layer_scale':
                x = x + self.skip_scale * skip
            elif self.skip_type == 'concat':  # 'concat'
                x = torch.cat([x, skip], dim=1)
        x = x.squeeze(2).permute(0,2,3,1)
        if self.nat_blocks:
            for blk in self.nat_blocks: 
                x = checkpoint(blk, x, use_reentrant=False)
        return x.permute(0,3,1,2).unsqueeze(2)

class UNAT(nn.Module):
    """
    A flexible UNet using your NAT-based DownBlock / UpBlock. 
    Features:
      - Arbitrary depth (number of down / up stages).
      - Skip connections between down and up stages.
      - Configurable channel progression (either manual or auto-generated).
      - NAT dimension (2D or 3D) determined by 'nat_dim'.
      - Optional final convolution to map to desired out_channels.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_channels: list = None,
        up_channels: list = None,
        down_strides: list = [(2,2)],  
        down_block_depths: list = [1], 
        down_kernel_sizes: list = [(3,3)],    # or (2,2,2) if 3D
        up_strides: list = [(2,2)], 
        up_block_depths: list = [1],
        up_kernel_sizes: list = [(3,3)],  
        norm=None,
        layer_scale=0,
        mlp_ratio=4,
        num_blocks=2,
        skip_type='add',
        skip_down_levels=[],
        skip_up_levels=[],
        in_steps=2,    # or 'concat'
        final_conv=True,
        rpb=True,
        rel_dist_bias=False,
        resolution=1.0,
        use_rope=False,
        use_nl_rdb=False
    ):
        """
        Args:
            in_channels:    Number of channels in the input.
            out_channels:   Number of channels in the final output.
            depth:          Number of Down->Up stages (3 => 3 down, 3 up).
            down_channels:  List of channels for each down block.
            up_channels:    List of channels for each up block.
            stride:         Convolution / transpose stride (halves/doubles).
            kernel_size:    NATBlock kernel size (and also conv kernel in blocks).
            norm, layer_scale, mlp_ratio:
                Additional NATBlock parameters.
            num_blocks:     How many NAT layers in each block.
            skip_type:      'add' or 'concat' for merging skip connections.
        """
        super().__init__()

        self.skip_down_levels = skip_down_levels
        self.skip_up_levels = skip_up_levels
        self.skip_type = skip_type
        self.in_steps = in_steps

        ######################
        # 1) Determine channels
        ######################
        if up_channels is None:
            # Often reversed, e.g. [1024, 512, 256]
            up_channels = list(reversed(down_channels))

        self.down_channels = down_channels
        self.down_depth = len(down_channels)
        self.up_channels = up_channels
        self.up_depth = len(up_channels)
        ######################
        # 2) Build Down Blocks
        ######################
        self.down_blocks = nn.ModuleList()
        prev_ch = in_channels
        for i in range(self.down_depth):
            block_in_ch = prev_ch
            block_out_ch = down_channels[i]

            down_blk = DownBlock(
                in_ch=block_in_ch,
                out_ch=block_out_ch,
                nat_dim=len(down_kernel_sizes[i]),
                stride=down_strides[i],
                kernel_size=down_kernel_sizes[i],
                norm=norm,
                layer_scale=layer_scale,
                mlp_ratio=mlp_ratio,
                num_blocks=num_blocks,
                depth=down_block_depths[i],
                rpb=rpb,
                rel_dist_bias=rel_dist_bias,
                resolution=resolution,
                use_rope=use_rope,
                use_nl_rdb=use_nl_rdb
            )
            self.down_blocks.append(down_blk)
            prev_ch = block_out_ch

        out_ch = block_out_ch
        self.up_blocks = nn.ModuleList()
        for i in range(self.up_depth):
            # We'll typically feed up_blocks[i] the output from the previous stage
            # plus a skip from down_blocks[depth - i - 1].
            # The in/out channels can be tuned as needed.
            in_ch = up_channels[i]
            
            
            if i < len(up_channels) - 1:
                out_ch = up_channels[i+1]
            else:
                if final_conv:
                    out_ch = down_channels[0]
                else:
                    out_ch = out_channels
                
            
            if i not in skip_up_levels:
                s = None 
            else:
                s = skip_type

            up_blk = UpBlock(
                in_ch=in_ch,
                out_ch=out_ch,
                stride=up_strides[i],
                kernel_size=up_kernel_sizes[i],
                norm=norm,
                layer_scale=layer_scale,
                mlp_ratio=mlp_ratio,
                num_blocks=num_blocks,
                skip_type=s,
                depth=up_block_depths[i],
                rpb=rpb,
                rel_dist_bias=rel_dist_bias,
                resolution=resolution,
                use_rope=use_rope,
                use_nl_rdb=use_nl_rdb
            )
            self.up_blocks.append(up_blk)

        if final_conv:
            self.final_conv = nn.Conv3d(
                out_ch,
                out_channels,
                kernel_size=1
            )
            torch.nn.init.xavier_uniform_(self.final_conv.weight)
            torch.nn.init.zeros_(self.final_conv.bias)
        else:
            self.final_conv = nn.Identity()
        

    def one_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UNet.
        Args:
            x: (B, in_channels, H, W) if nat_dim=2
               (B, in_channels, T, H, W) if nat_dim=3
        Returns:
            (B, out_channels, H, W) or (B, out_channels, T, H, W)
        """
        skips = []
        
        for i in range(self.down_depth):
            x = self.down_blocks[i](x)
            
            if i in self.skip_down_levels:
                skips.append(x)

        for i in range(self.up_depth):
            if i in self.skip_up_levels:
                x = self.up_blocks[i](x, skips.pop())
            else:
                x = self.up_blocks[i](x)

        x = self.final_conv(x)

        return x

    def forward(self, x: torch.Tensor, inv: torch.Tensor, n_steps=1):
        n_steps = min((n_steps, inv.shape[2]-self.in_steps+1))
        yhat = torch.empty((*x.shape[:2], n_steps, *x.shape[3:]), dtype=x.dtype, device=x.device)
        for i in range(n_steps):
            z = torch.cat((x, inv[:,:,i:i+self.in_steps]), dim=1)            
            z = self.one_step_forward(z)
            yhat[:,:,i:i+1] = z
            if i < n_steps-1:
                x = torch.concat((x[:,:,-1:], z), dim=2)
        return yhat


# ################################################################################
# # The UNet with NAT for Video Diffusion: Configurable Depth, Channels, Skip Connections
# ################################################################################

# class CondEncoder(nn.Module):
#     def __init__(
#         self,
#         unat,
#         levels,
#         in_channels,
#         out_channels,
#         strides,
#     ):
#         super().__init__()
#         self.unat = unat 
#         for param in self.unat.parameters():
#             param.requires_grad = False
            
#         self.levels = levels
#         layers = []
#         sequential = False
#         for i in range(len(strides)):
#             stride = strides[i]
#             if stride != 1:
#                 sequential = True
#                 linear = conv_nd(3,in_channels[i],out_channels[i],kernel_size=stride,stride=stride)
#                 torch.nn.init.zeros_(linear.bias)
#                 torch.nn.init.xavier_uniform_(linear.weight)
#                 layers.append(linear)
#         if sequential:
#             self.layers = nn.Sequential(*layers)
#         else:
#             self.layers = nn.Identity()

#     @torch.no_grad()
#     def one_step_forward(self, x):
#         skips = []
#         out = []
        
#         for i in range(self.unat.down_depth):
#             x = self.unat.down_blocks[i](x)
            
#             if i in self.unat.skip_down_levels:
#                 skips.append(x)
        
#         if 0 in self.levels:
#             out.append(x)

#         for i in range(self.unat.up_depth):
#             if i in self.unat.skip_up_levels:
#                 x = self.unat.up_blocks[i](x, skips.pop())
#             else:
#                 x = self.unat.up_blocks[i](x)
            
#             if i+1 in self.levels:
#                 out.append(x)
#         return out, x
    
#     @torch.no_grad()
#     def cond_forward(self, x: torch.Tensor, inv: torch.Tensor, n_steps=1):
#         n_steps = min((n_steps, inv.shape[2]-self.unat.in_steps+1))
#         yhat = torch.empty((*x.shape[:2], n_steps, *x.shape[3:]), dtype=x.dtype, device=x.device)
#         cond = {}
#         for i in range(n_steps):
#             z = torch.cat((x, inv[:,:,i:i+self.unat.in_steps]), dim=1)            
#             out, z = self.one_step_forward(z)
#             yhat[:,:,i:i+1] = z
#             for z_ in out:
#                 s = z_.shape[3:]
#                 if s in cond:
#                     cond[s].append(z_)
#                 else:
#                     cond[s] = [z_]
            
#             if i < n_steps-1:
#                 x = torch.concat((x[:,:,:-1], z), dim=2)
#         return yhat, cond
    
#     def forward(self, x: torch.Tensor, inv: torch.Tensor, n_steps=1):
#         yhat, cond = self.cond_forward(x, inv, n_steps)
#         for i, k in enumerate(cond.keys()):
#             cond[k] = self.layers[i](torch.cat(cond[k], dim=2)).permute(0,2,3,4,1)
#         return yhat, cond


# class DownViBlock3D(nn.Module):
#     """
#     A 'downsample' block that reduces spatial resolution (and possibly temporal).
#     Internally:
#       1) Applies a stride=2 convolution (or 3D conv if nat_dim=3).
#       2) Applies NAT-based attention (NATBlock2D or NATBlock3D).
#     Returns:
#       A tensor with reduced spatial/temporal dimensions.
#     """
#     def __init__(
#         self,
#         in_ch: int,
#         out_ch: int,
#         cross: bool,
#         stride=(2,2),
#         kernel_size=3,
#         norm=None,
#         layer_scale=0,
#         mlp_ratio=4,
#         num_blocks=2,
#         depth=1,
#         emb_dim=256
#     ):
#         super().__init__()
#         self.downsample = conv_nd(3, in_ch, out_ch, kernel_size=stride, stride=stride, padding=0, padding_mode="reflect")
#         torch.nn.init.xavier_uniform_(self.downsample.weight)
#         torch.nn.init.zeros_(self.downsample.bias)
#         self.nat_blocks = None
#         self.cross_nat_blocks = None
#         self.depth = depth

        

#         if depth > 0:
#             self.emb_film = nn.Linear(emb_dim, 2 * out_ch)
#             nat_blocks = []
#             for i in range(depth):
#                 nat_blocks.append(
#                     NATBlock3D(
#                         dim=out_ch,
#                         mlp_ratio=mlp_ratio, 
#                         num_blocks=num_blocks,
#                         norm=norm,
#                         kernel_size=kernel_size,
#                         layer_scale=layer_scale
#                     )
#                 )
#             self.nat_blocks = nn.Sequential(*nat_blocks)
#             if cross:
#                 cross_nat_blocks = []
#                 for i in range(depth):
#                     cross_nat_blocks.append(
#                         CrossNATBlock3D(
#                             dim=out_ch,
#                             mlp_ratio=mlp_ratio, 
#                             num_blocks=num_blocks,
#                             norm=norm,
#                             kernel_size=kernel_size,
#                             layer_scale=layer_scale
#                         )
#                     )
#                 self.cross_nat_blocks = nn.Sequential(*cross_nat_blocks)

#     def forward(self, x: torch.Tensor, c: dict = None, t: torch.Tensor = None) -> torch.Tensor:
        
#         x = self.downsample(x) 
#         if self.nat_blocks:
#             x = x.permute(0,2,3,4,1)
#             scale, shift = torch.chunk(self.emb_film(t), 2, dim=-1)
#             x = x * (1 + scale) + shift
#             for i in range(self.depth): 
#                 if self.cross_nat_blocks:
#                     x = checkpoint(self.cross_nat_blocks[i], x, c[x.shape[2:4]], use_reentrant=False)        
#                 x = checkpoint(self.nat_blocks[i], x, use_reentrant=False)
#             x = x.permute(0,4,1,2,3)
#         return x


# class UpViBlock3D(nn.Module):
#     """
#     An 'upsample' block that increases spatial resolution.
#     Internally:
#       1) Uses a transposed convolution (or 3D) to upsample.
#       2) Merges a skip connection (via addition or concatenation).
#       3) Optionally applies NAT-based attention.
#     """
#     def __init__(
#         self,
#         in_ch: int,
#         out_ch: int,
#         cross: bool,
#         kernel_size=(5,5),
#         stride=(1,2,2),
#         norm=None,
#         layer_scale=0,
#         mlp_ratio=4,
#         num_blocks=2,
#         skip_type='add',
#         depth=1,
#         emb_dim=256
#     ):
#         super().__init__()
#         self.upsample = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=stride, stride=stride, padding=0)
#         torch.nn.init.xavier_uniform_(self.upsample.weight)
#         torch.nn.init.zeros_(self.upsample.bias)
#         self.nat_blocks = None
#         self.cross_nat_blocks = None
#         self.depth = depth
        

#         if depth > 0:
#             self.emb_film = nn.Linear(emb_dim, 2 * out_ch)
#             nat_blocks = []
#             for i in range(depth):
#                 nat_blocks.append(
#                     NATBlock3D(
#                         dim=out_ch,
#                         mlp_ratio=mlp_ratio, 
#                         num_blocks=num_blocks,
#                         norm=norm,
#                         kernel_size=kernel_size,
#                         layer_scale=layer_scale
#                     )
#                 )
#             self.nat_blocks = nn.Sequential(*nat_blocks)
#             if cross:
#                 cross_nat_blocks = []
#                 for i in range(depth):
#                     cross_nat_blocks.append(
#                         CrossNATBlock3D(
#                             dim=out_ch,
#                             mlp_ratio=mlp_ratio, 
#                             num_blocks=num_blocks,
#                             norm=norm,
#                             kernel_size=kernel_size,
#                             layer_scale=layer_scale
#                         )
#                     )
#                 self.cross_nat_blocks = nn.Sequential(*cross_nat_blocks)

        
#         self.skip_type = skip_type # (for 'add' or 'concat')
#         if self.skip_type == "layer_scale":
#             self.skip_scale = nn.Parameter(torch.rand((1,out_ch,1,1,1)) * 1e-3, requires_grad=True)
            

#     def forward(self, x: torch.Tensor, skip: torch.Tensor = None, c: dict = None, t: torch.Tensor = None) -> torch.Tensor:
#         x = self.upsample(x)
#         if skip is not None:
#             if self.skip_type == 'add':
#                 x = x + skip
#             elif self.skip_type == 'layer_scale':
#                 x = x + self.skip_scale * skip
#             elif self.skip_type == 'concat':  # 'concat'
#                 x = torch.cat([x, skip], dim=1)
        
#         if self.nat_blocks:
#             x = x.permute(0,2,3,4,1)
#             scale, shift = torch.chunk(self.emb_film(t), 2, dim=-1)
#             x = x * (1 + scale) + shift
#             for i in range(self.depth): 
#                 if self.cross_nat_blocks:
#                     x = checkpoint(self.cross_nat_blocks[i], x, c[x.shape[2:4]], use_reentrant=False)        
#                 x = checkpoint(self.nat_blocks[i], x, use_reentrant=False)
#             x = x.permute(0,4,1,2,3)
#         return x


# class ViDiffUNAT(nn.Module):
#     """
#     A flexible UNet using your NAT-based DownBlock / UpBlock. 
#     Features:
#       - Arbitrary depth (number of down / up stages).
#       - Skip connections between down and up stages.
#       - Configurable channel progression (either manual or auto-generated).
#       - NAT dimension (2D or 3D) determined by 'nat_dim'.
#       - Optional final convolution to map to desired out_channels.
#     """

#     def __init__(
#         self,
#         in_channels: int = 3,
#         out_channels: int = 3,
#         down_channels: list = None,
#         up_channels: list = None,
#         down_strides: list = [(2,2)],  
#         down_block_depths: list = [1], 
#         down_block_cross: list = [True],
#         down_kernel_sizes: list = [(3,3)],    # or (2,2,2) if 3D
#         up_strides: list = [(2,2)], 
#         up_block_depths: list = [1],
#         up_block_cross: list = [True],
#         up_kernel_sizes: list = [(3,3)],  
#         norm=None,
#         layer_scale=0,
#         mlp_ratio=4,
#         num_blocks=2,
#         skip_type='add',
#         skip_down_levels=[],
#         skip_up_levels=[],
#         emb_dim=256,
#     ):
#         """
#         Args:
#             in_channels:    Number of channels in the input.
#             out_channels:   Number of channels in the final output.
#             depth:          Number of Down->Up stages (3 => 3 down, 3 up).
#             down_channels:  List of channels for each down block.
#             up_channels:    List of channels for each up block.
#             stride:         Convolution / transpose stride (halves/doubles).
#             kernel_size:    NATBlock kernel size (and also conv kernel in blocks).
#             norm, layer_scale, mlp_ratio:
#                 Additional NATBlock parameters.
#             num_blocks:     How many NAT layers in each block.
#             skip_type:      'add' or 'concat' for merging skip connections.
#         """
#         super().__init__()

#         self.skip_down_levels = skip_down_levels
#         self.skip_up_levels = skip_up_levels
#         self.skip_type = skip_type

#         ######################
#         # 1) Determine channels
#         ######################
#         if up_channels is None:
#             # Often reversed, e.g. [1024, 512, 256]
#             up_channels = list(reversed(down_channels))

#         self.down_channels = down_channels
#         self.down_depth = len(down_channels)
#         self.up_channels = up_channels
#         self.up_depth = len(up_channels)
#         ######################
#         # 2) Build Down Blocks
#         ######################
#         self.down_blocks = nn.ModuleList()
#         prev_ch = in_channels
#         for i in range(self.down_depth):
#             block_in_ch = prev_ch
#             block_out_ch = down_channels[i]
#             cross = down_block_cross[i]
#             down_blk = DownViBlock3D(
#                 in_ch=block_in_ch,
#                 out_ch=block_out_ch,
#                 cross=cross,
#                 stride=down_strides[i],
#                 kernel_size=down_kernel_sizes[i],
#                 norm=norm,
#                 layer_scale=layer_scale,
#                 mlp_ratio=mlp_ratio,
#                 num_blocks=num_blocks,
#                 depth=down_block_depths[i],
#                 emb_dim=emb_dim
#             )
#             self.down_blocks.append(down_blk)
#             prev_ch = block_out_ch


#         self.up_blocks = nn.ModuleList()
#         for i in range(self.up_depth):
#             # We'll typically feed up_blocks[i] the output from the previous stage
#             # plus a skip from down_blocks[depth - i - 1].
#             # The in/out channels can be tuned as needed.
#             in_ch = up_channels[i]
#             if i < len(up_channels) - 1:
#                 out_ch = up_channels[i+1]
#                 s = skip_type
#             else:
#                 out_ch = out_channels
#                 s = None
            
#             if i not in skip_up_levels:
#                 s = None 
#             else:
#                 s = skip_type

#             up_blk = UpViBlock3D(
#                 in_ch=in_ch,
#                 out_ch=out_ch,
#                 cross=up_block_cross[i],
#                 stride=up_strides[i],
#                 kernel_size=up_kernel_sizes[i],
#                 norm=norm,
#                 layer_scale=layer_scale,
#                 mlp_ratio=mlp_ratio,
#                 num_blocks=num_blocks,
#                 skip_type=s,
#                 depth=up_block_depths[i],
#                 emb_dim=emb_dim
#             )
#             self.up_blocks.append(up_blk)

#         self.time_embed = nn.Sequential(
#             nn.Linear(emb_dim, emb_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(emb_dim, emb_dim),
#         )

#     def time_embedding(self, t):
#         device = t.device
#         half_dim = self.time_embed[0].in_features // 2
#         freqs = torch.exp(-math.log(10000.0)*torch.arange(0, half_dim, device=device)/half_dim)
#         args = t[:, None].float() * freqs[None]
#         emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
#         return self.time_embed(emb).view(t.shape[0], 1, 1, 1, self.time_embed[0].in_features)

#     def forward(self, x: torch.Tensor, t: torch.Tensor = None, c: dict = None) -> torch.Tensor:
#         t = self.time_embedding(t)
#         skips = []
#         for i in range(self.down_depth):
#             x = self.down_blocks[i](x, c, t)
#             if i in self.skip_down_levels:
#                 skips.append(x)

#         for i in range(self.up_depth):
#             if i in self.skip_up_levels:
#                 x = self.up_blocks[i](x, skips.pop(), c, t)
#             else:
#                 x = self.up_blocks[i](x, None, c, t)
#         return x
        

# if __name__ == "__main__":
#     unat = UNAT(
#         in_channels=4,
#         out_channels=3,
#         down_channels=[128,512,1024],
#         up_channels=[1024,512],
#         down_strides=[(2,1,1),(1,2,2),(1,2,2)],  
#         down_block_depths=[0,4,8], 
#         down_kernel_sizes=[(5,5),(5,5),(5,5)],    # or (2,2,2) if 3D
#         up_strides=[(1,2,2),(1,2,2)], 
#         up_block_depths=[4,0],
#         up_kernel_sizes=[5,5,5],  
#         norm=None,
#         layer_scale=0,
#         mlp_ratio=4,
#         num_blocks=1,
#         skip_type='add',
#         skip_down_levels=[1],
#         skip_up_levels=[0],
#         in_steps=2,    # or 'concat'
#         final_conv=False).to("cuda")
    
#     condencoder = CondEncoder(unat,
#         levels=[0,1],
#         in_channels=[1024,512],
#         out_channels=[1024,512],
#         strides=[[2,1,1],[2,1,1]]).to("cuda")
    
#     x = torch.randn((1,3,2,64,64)).to("cuda")
#     inv = torch.randn((1,1,20,64,64)).to("cuda")

#     yhat, cond = condencoder(x, inv, n_steps=18)
    
#     for k in cond:
#         print(k, cond[k].shape, cond[k].device)

#     vidiffunat = ViDiffUNAT(
#         in_channels=3,
#         out_channels=3,
#         down_channels=[128,512,1024],
#         up_channels=[1024,512],
#         down_strides=[(2,1,1),(1,2,2),(1,2,2)], 
#         down_block_depths=[0,4,8], 
#         down_block_cross=[False,True,True],
#         down_kernel_sizes=[(5,5,5),(5,5,5),(5,5,5)],  # or (2,2,2) if 3D
#         up_strides=[(1,2,2),(1,2,2)], 
#         up_block_depths=[4,0],
#         up_block_cross=[True,True,False],
#         up_kernel_sizes=[5,5],
#         norm=None,
#         layer_scale=0,
#         mlp_ratio=4,
#         num_blocks=1,
#         skip_type='layer_scale',
#         skip_down_levels=[1],
#         skip_up_levels=[0],
#         emb_dim=256,
#     ).to("cuda")

#     noisy_x = torch.randn((1,3,18,64,64)).to("cuda")
#     t = torch.randn((1)).to("cuda")
#     denoised_x = vidiffunat(noisy_x, t, cond)
#     print(x.mean(), x.std())
#     print(noisy_x.mean(), noisy_x.std())
#     print(denoised_x.mean(), denoised_x.std())

#     def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(count_parameters(vidiffunat))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    import time
    # 3) Build a small model
    model = UNAT(
        in_channels=4,
        out_channels=3,
        down_channels=[128,512],
        up_channels=[512],
        down_strides=[(2,1,1),(1,2,2)],  
        down_block_depths=[0,4], 
        down_kernel_sizes=[(5,5),(5,5)],    # or (2,2,2) if 3D
        up_strides=[(1,2,2)], 
        up_block_depths=[4,],
        up_kernel_sizes=[5],  
        norm=None,
        layer_scale=0.5,
        mlp_ratio=4,
        num_blocks=1,
        skip_type='add',
        skip_down_levels=[],
        skip_up_levels=[],
        in_steps=2,
        rel_dist_bias=False,
        use_rope=False,
        rpb=False,
        resolution=[1,1]).to(device)

    # 4) Create random input & target
    B, H, W, C = 2, 64, 64, 16
    x = torch.randn(B,3,2,H,W, device=device)
    inv = torch.randn(B,1,3,H,W, device=device)
    # We'll treat out as [B,H,W,C], do a simple MSE to a random target
    target = torch.randn(B,3,1,H,W, device=device)

    # 5) Simple training step: define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 6) Forward pass
    start = time.time()
    out = model(x, inv)
    print(time.time() - start)
    loss = F.mse_loss(out, target)

    # 7) Backward pass
    optimizer.zero_grad()
    loss.backward()

    # 8) Check the gradient for rel_dist_scale
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f'Gradients/{name}/{param.grad.detach().cpu().numpy().mean()}')

    # 9) (Optional) step
    optimizer.step()

    # print final parameter
    if hasattr(model, "rpb"):
        print("rel_dist_scale after step:", model.rpb.item())