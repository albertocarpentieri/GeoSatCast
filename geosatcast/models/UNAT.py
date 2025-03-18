import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.init import trunc_normal_
import math 

from geosatcast.utils import avg_pool_nd, conv_nd
from geosatcast.blocks.NAT import NATBlock2D


################################################################################
# The UNet with NAT: Configurable Depth, Channels, Skip Connections
################################################################################

def _downsample_grid(grid: torch.Tensor, stride=(2,2)) -> torch.Tensor:
    """
    Downsample the grid (B, H, W, 2) by the given stride, 
    e.g. using average pooling on the last two dims.
    """
    if len(stride) == 3:
        stride = stride[1:]
    b, h, w, c = grid.shape
    grid_2d = grid.permute(0, 3, 1, 2)  # (B, 2, H, W)
    grid_down = F.avg_pool2d(grid_2d, kernel_size=stride, stride=stride)  
    # Now permute back to (B, H', W', 2)
    grid_down = grid_down.permute(0, 2, 3, 1)
    return grid_down

class DownBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride=(2,2),
        kernel_size=3,
        norm=None,
        layer_scale=0,
        mlp_ratio=4,
        num_blocks=2,
        depth=1,
        emb_method="rel_pos_bias",
        resolution=1.0,
        downsample_type: str = "conv",  # "conv" or "avgpool"
    ):
        super().__init__()
        self.downsample_type = downsample_type
        self.stride = stride
        if self.downsample_type == "conv":
            # Original: use a stride=2 (or stride=stride) conv
            self.downsample = conv_nd(
                dims=3, 
                in_channels=in_ch, 
                out_channels=out_ch, 
                kernel_size=stride, 
                stride=stride, 
                padding=0,
                padding_mode="reflect",
            )
            nn.init.xavier_uniform_(self.downsample.weight)
            nn.init.zeros_(self.downsample.bias)
            self.channel_proj = None

        else:  # "avgpool"
            # We'll do a 1×1 conv (only if channels differ) + average pooling
            self.downsample = None
            if in_ch != out_ch:
                self.channel_proj = conv_nd(
                    dims=3,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    padding_mode="reflect"
                )
                nn.init.xavier_uniform_(self.channel_proj.weight)
                nn.init.zeros_(self.channel_proj.bias)
            else:
                self.channel_proj = nn.Identity()

        # NAT block(s) as before:
        if depth > 0:
            self.nat_blocks = nn.Sequential(
                *[
                    NATBlock2D(
                        dim=out_ch,
                        mlp_ratio=mlp_ratio, 
                        num_blocks=num_blocks,
                        norm=norm,
                        kernel_size=kernel_size,
                        layer_scale=layer_scale,
                        emb_method=emb_method,
                        resolution=resolution,
                        cross=False
                    )
                    for _ in range(depth)
                ]
            )
        else:
            self.nat_blocks = None

    def forward(self, x: torch.Tensor, grid: torch.Tensor = None) -> torch.Tensor:
        if self.downsample_type == "conv":
            x = self.downsample(x)
        else:
            x = self.channel_proj(x)
            x = F.avg_pool3d(x, kernel_size=self.stride, stride=self.stride)
        
        if grid is not None:
            grid = _downsample_grid(grid, stride=self.stride)
        
        # NAT blocks in 2D style
        if self.nat_blocks:
            x = x.squeeze(2).permute(0, 2, 3, 1)   # (B, H, W, C)
            for blk in self.nat_blocks:
                x = checkpoint(blk, x, None, grid, use_reentrant=False)
            x = x.permute(0, 3, 1, 2).unsqueeze(2)
        return x, grid


class UpBlock(nn.Module):
    """
    An 'upsample' block that increases spatial resolution
    either by transposed convolution or interpolation.
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
        resolution=1.0,
        emb_method="spherical_rope",
        upsample_type: str = "transposed",  # "transposed" or "interp"
        interp_mode: str = "nearest", 
    ):
        super().__init__()
        self.upsample_type = upsample_type
        self.interp_mode = interp_mode
        self.skip_type = skip_type

        if self.upsample_type == "transposed":
            self.upsample = nn.ConvTranspose3d(
                in_ch, out_ch, 
                kernel_size=stride, 
                stride=stride, 
                padding=0
            )
            nn.init.xavier_uniform_(self.upsample.weight)
            nn.init.zeros_(self.upsample.bias)
            self.channel_proj = None
            self.scale_factor = None
        else:
            # Interpolation-based upsampling (nearest or trilinear, etc.)
            self.upsample = None
            # 1×1 conv to adjust channels
            if in_ch != out_ch:
                self.channel_proj = nn.Conv3d(in_ch, out_ch, kernel_size=1)
                nn.init.xavier_uniform_(self.channel_proj.weight)
                # nn.init.zeros_(self.channel_proj.bias)
            else:
                self.channel_proj = nn.Identity()
            self.scale_factor = stride

        if depth > 0:
            self.nat_blocks = nn.Sequential(
                *[
                    NATBlock2D(
                        dim=out_ch,
                        mlp_ratio=mlp_ratio, 
                        num_blocks=num_blocks,
                        norm=norm,
                        kernel_size=kernel_size,
                        layer_scale=layer_scale,
                        emb_method=emb_method,
                        resolution=resolution,
                        cross=False
                    )
                    for _ in range(depth)
                ]
            )
        else:
            self.nat_blocks = None

        if self.skip_type == "layer_scale":
            self.skip_scale = nn.Parameter(
                torch.rand((1, out_ch, 1, 1, 1)),
                requires_grad=True
            )
            trunc_normal_(self.skip_scale, std=0.02, mean=0.0, a=-2.0, b=2.0)
            

    def forward(self, x: torch.Tensor, grid: torch.Tensor = None, skip: torch.Tensor = None) -> torch.Tensor:
        if self.upsample_type == "transposed":
            x = self.upsample(x)
        else:
            x = self.channel_proj(x)
            x = F.interpolate(
                x, 
                scale_factor=self.scale_factor, 
                mode=self.interp_mode,
                # align_corners only relevant if "linear" modes:
                align_corners=False if "linear" in self.interp_mode else None
            )
        
        # Merge skip
        if skip is not None:
            if self.skip_type == 'add':
                x = x + skip
            elif self.skip_type == 'layer_scale':
                x = x + self.skip_scale * skip
            elif self.skip_type == 'concat':
                x = torch.cat([x, skip], dim=1)

        # NAT blocks
        if self.nat_blocks:
            x = x.squeeze(2).permute(0, 2, 3, 1)
            for blk in self.nat_blocks: 
                x = checkpoint(blk, x, None, grid, use_reentrant=False)
            x = x.permute(0, 3, 1, 2).unsqueeze(2)
        return x

class UNAT(nn.Module):
    """
    A flexible UNet using your NAT-based DownBlock / UpBlock. 
    Features:
      - Arbitrary depth (number of down / up stages).
      - Skip connections between down and up stages.
      - Configurable channel progression (either manual or auto-generated).
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
        resolution=1.0,
        emb_method="rel_pos_bias",
        downsample_type="conv",   # can be str or list[str]
        upsample_type="transposed",  # can be str or list[str]
        interp_mode="nearest", 
    ):
        super().__init__()

        self.skip_down_levels = skip_down_levels
        self.skip_up_levels = skip_up_levels
        self.skip_type = skip_type
        self.in_steps = in_steps

        # Just in case user gave single strings, make them lists of correct length:
        if isinstance(downsample_type, str):
            downsample_type = [downsample_type]*len(down_channels)
        assert len(downsample_type) == len(down_channels)

        if isinstance(upsample_type, str):
            upsample_type = [upsample_type]*len(up_channels)
        assert len(upsample_type) == len(up_channels)

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
                stride=down_strides[i],
                kernel_size=down_kernel_sizes[i],
                norm=norm,
                layer_scale=layer_scale,
                mlp_ratio=mlp_ratio,
                num_blocks=num_blocks,
                depth=down_block_depths[i],
                emb_method=emb_method,
                resolution=resolution,
                downsample_type=downsample_type[i]
            )
            self.down_blocks.append(down_blk)
            prev_ch = block_out_ch

        in_ch = down_channels[-1]
        self.up_blocks = nn.ModuleList()
        for i in range(self.up_depth):
            out_ch = up_channels[i]
                
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
                resolution=resolution,
                emb_method=emb_method,
                upsample_type=upsample_type[i],
                interp_mode=interp_mode
            )
            self.up_blocks.append(up_blk)
            in_ch = out_ch

        if final_conv:
            self.final_conv = nn.Conv3d(
                in_ch,
                out_channels,
                kernel_size=1
            )
            torch.nn.init.xavier_uniform_(self.final_conv.weight)
            # torch.nn.init.zeros_(self.final_conv.bias)
        else:
            self.final_conv = nn.Identity()
        

    def one_step_forward(self, x: torch.Tensor, grid: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the UNet for a single step (given x plus 'in_steps' from inv).
        We'll:
          - Downsample x while also downsampling grid, stage by stage.
          - Then upsample x using the stored grids (unchanged).
        """
        skips = []
        if grid is not None:
            grid_scales = [grid] 
        
        for i in range(self.down_depth):
            x, grid = self.down_blocks[i](x, grid)
            if grid is not None:
                grid_scales.append(grid)
            if i in self.skip_down_levels:
                skips.append(x)

        for i in range(self.up_depth):
            if i in self.skip_up_levels:
                skip_x = skips.pop()  # from the end
            else:
                skip_x = None
            if grid is not None:
                grid = grid_scales[self.down_depth - i - 1]
            x = self.up_blocks[i](x, grid, skip_x)

        # final conv
        x = self.final_conv(x)
        return x

    def forward(
        self, 
        x: torch.Tensor, 
        inv: torch.Tensor, 
        grid: torch.Tensor = None, 
        n_steps=1):
        """
        Example forward that runs multiple 'one_step_forward' calls
        in a sliding manner across 'inv'.
        """
        n_steps = min((n_steps, inv.shape[2] - self.in_steps + 1))
        yhat = torch.empty((*x.shape[:2], n_steps, *x.shape[3:]), 
                           dtype=x.dtype, device=x.device)
        for i in range(n_steps):
            z = torch.cat((x, inv[:,:,i:i+self.in_steps]), dim=1)
            z = self.one_step_forward(z, grid)  # run UNet
            yhat[:,:,i:i+1] = z
            if i < n_steps-1:
                # shift for next iteration
                x = torch.cat((x[:,:,-1:], z), dim=2)
        return yhat

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    import time
    # 3) Build a small model
    model = UNAT(
        in_channels=4,
        out_channels=3,
        down_channels=[192,768],
        up_channels=[192],
        down_strides=[(2,1,1),(1,2,2)],  
        down_block_depths=[4,4], 
        down_kernel_sizes=[(5,5),(5,5)],    # or (2,2,2) if 3D
        up_strides=[(1,2,2)], 
        up_block_depths=[4],
        up_kernel_sizes=[(5,5)],  
        norm=None,
        layer_scale="normal",
        mlp_ratio=4,
        num_blocks=1,
        skip_type='layer_scale',
        skip_down_levels=[0],
        skip_up_levels=[0],
        in_steps=2,
        resolution=1.0,
        emb_method="spherical_rope",
        downsample_type="avgpool",   # can be str or list[str]
        upsample_type="interp",  # can be str or list[str]
        interp_mode="nearest").to(device)

    # 4) Create random input & target
    B, H, W, C = 2, 64, 64, 16
    x = torch.randn(B,3,2,H,W, device=device)
    inv = torch.randn(B,1,3,H,W, device=device)
    grid = torch.randn(B,H,W,2, device=device)
    # We'll treat out as [B,H,W,C], do a simple MSE to a random target
    target = torch.randn(B,3,1,H,W, device=device)

    # 5) Simple training step: define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 6) Forward pass
    start = time.time()
    out = model(x, inv, grid)
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