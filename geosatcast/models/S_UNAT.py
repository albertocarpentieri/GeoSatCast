import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math 

from geosatcast.utils import avg_pool_nd, conv_nd
from geosatcast.blocks.ResBlock import ResBlock3D
from geosatcast.blocks.S_NAT import NATBlock2D, NATBlock3D


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
        norm=None,
        layer_scale=0,
        mlp_ratio=4,
        num_blocks=2,
        depth=1,
        resolution=1.0,
    ):
        super().__init__()
        self.downsample = conv_nd(3, in_ch, out_ch, kernel_size=stride, stride=stride, padding=0, padding_mode="reflect")
        torch.nn.init.xavier_uniform_(self.downsample.weight)
        torch.nn.init.zeros_(self.downsample.bias)
        if depth > 0:
            self.nat_blocks = nn.Sequential(
                *[NATBlock2D(
                dim=out_ch,
                mlp_ratio=mlp_ratio, 
                num_blocks=num_blocks,
                norm=norm,
                kernel_size=kernel_size,
                layer_scale=layer_scale,
                use_rope=use_rope,
                cross=False,) for _ in range(depth)])
        else:
            self.nat_blocks = None

    def forward(self, x: torch.Tensor, grid: torch.Tensor = None) -> torch.Tensor:
        x = self.downsample(x)
        if self.nat_blocks:
            x = x.squeeze(2).permute(0,2,3,1)
            for blk in self.nat_blocks: 
                x = checkpoint(blk, x, None, grid, use_reentrant=False)
            
            x = x.permute(0,3,1,2).unsqueeze(2)
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
        use_rope=False,
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
                use_rope=use_rope,
                cross=False) for _ in range(depth)])
        else:
            self.nat_blocks = None

        
        self.skip_type = skip_type # (for 'add' or 'concat')
        if self.skip_type == "layer_scale":
            self.skip_scale = nn.Parameter(torch.rand((1,out_ch,1,1,1)) * 1e-3, requires_grad=True)
            

    def forward(self, x: torch.Tensor, grid: torch.Tensor = None, skip: torch.Tensor = None) -> torch.Tensor:
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
                x = checkpoint(blk, x, None, grid, use_reentrant=False)
        return x.permute(0,3,1,2).unsqueeze(2)

class S_UNAT(nn.Module):
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
        use_rope=False,
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
                stride=down_strides[i],
                kernel_size=down_kernel_sizes[i],
                norm=norm,
                layer_scale=layer_scale,
                mlp_ratio=mlp_ratio,
                num_blocks=num_blocks,
                depth=down_block_depths[i],
                use_rope=use_rope,
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
                use_rope=use_rope,
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
        

    def one_step_forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
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
            x = self.down_blocks[i](x, grid)
            
            if i in self.skip_down_levels:
                skips.append(x)

        for i in range(self.up_depth):
            if i in self.skip_up_levels:
                x = self.up_blocks[i](x, grid, skips.pop())
            else:
                x = self.up_blocks[i](x, grid)

        x = self.final_conv(x)

        return x

    def forward(self, x: torch.Tensor, inv: torch.Tensor, n_steps=1):
        n_steps = min((n_steps, inv.shape[2]-self.in_steps+1))
        inv, grid = inv[:,:-2], inv[:,-2:,0].permute(0,2,3,1)
        yhat = torch.empty((*x.shape[:2], n_steps, *x.shape[3:]), dtype=x.dtype, device=x.device)
        for i in range(n_steps):
            z = torch.cat((x, inv[:,:,i:i+self.in_steps]), dim=1)            
            z = self.one_step_forward(z, grid)
            yhat[:,:,i:i+1] = z
            if i < n_steps-1:
                x = torch.concat((x[:,:,-1:], z), dim=2)
        return yhat

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    import time
    # 3) Build a small model
    model = S_UNAT(
        in_channels=4,
        out_channels=3,
        down_channels=[128],
        up_channels=[],
        down_strides=[(2,1,1)],  
        down_block_depths=[0], 
        down_kernel_sizes=[(5,5)],    # or (2,2,2) if 3D
        up_strides=[], 
        up_block_depths=[],
        up_kernel_sizes=[],  
        norm=None,
        layer_scale=0.5,
        mlp_ratio=4,
        num_blocks=1,
        skip_type='add',
        skip_down_levels=[],
        skip_up_levels=[],
        in_steps=2,
        use_rope=True,).to(device)

    # 4) Create random input & target
    B, H, W, C = 2, 64, 64, 16
    x = torch.randn(B,3,2,H,W, device=device)
    inv = torch.randn(B,3,3,H,W, device=device)
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