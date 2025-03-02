import torch
from torch import nn
import torch.nn.functional as F
from geosatcast.utils import avg_pool_nd, conv_nd
from geosatcast.blocks.NAT import NATBlock2D, NATBlock3D
from geosatcast.blocks.AFNO import AFNOBlock2D, AdaFNOBlock2D
from geosatcast.blocks.ResBlock import ResBlock3D


class AFNOCastLatent(nn.Module):
    def __init__(
            self,
            embed_dim=256,
            forecast_depth=0,
            num_blocks=8,
            mlp_ratio=4,
            norm='group',
            layer_scale="auto",
            **kwargs
        ):
        super().__init__()

        self.embed_dim = embed_dim
        self.forecast_depth = forecast_depth
        
        if layer_scale == "auto":
            layer_scale=.5/(forecast_depth)
        elif layer_scale == "none":
            layer_scale = None
        self.forecast = nn.Sequential(
            *(AFNOBlock2D(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio, 
                    num_blocks=num_blocks,
                    norm=norm,
                    channel_first=False,
                    layer_scale=layer_scale
                )
            for _ in range(forecast_depth))
            )

    def forward(self, x, modes_range=None):
        x = x.squeeze(2).permute(0,2,3,1)
        for block in self.forecast:
            x = torch.utils.checkpoint.checkpoint(block, x, modes_range, use_reentrant=False)
        # x = torch.utils.checkpoint.checkpoint_sequential(self.forecast, self.forecast_depth, x, use_reentrant=False)
        return x.permute(0,3,1,2).unsqueeze(2)

class AdaFNOCastLatent(nn.Module):
    def __init__(
            self,
            embed_dim=256,
            forecast_depth=0,
            num_blocks=8,
            mlp_ratio=4,
            norm='group',
            layer_scale="auto",
            domain_size=64,
            wavelength_crop='all',
            **kwargs
        ):
        super().__init__()

        self.embed_dim = embed_dim
        self.forecast_depth = forecast_depth
        
        if layer_scale == "auto":
            layer_scale=.5/(forecast_depth)
        elif layer_scale == "none":
            layer_scale = None
        if wavelength_crop == "all":
            wavelength_crop = (0, torch.inf)
        
        self.forecast = nn.Sequential(
            *(AdaFNOBlock2D(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio, 
                    num_blocks=num_blocks,
                    norm=norm,
                    channel_first=False,
                    layer_scale=layer_scale,
                    domain_size=domain_size,
                    wavelength_crop=wavelength_crop
                )
            for _ in range(forecast_depth))
            )

    def forward(self, x):
        x = x.squeeze(2).permute(0,2,3,1)
        for block in self.forecast:
            x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        # x = torch.utils.checkpoint.checkpoint_sequential(self.forecast, self.forecast_depth, x, use_reentrant=False)
        return x.permute(0,3,1,2).unsqueeze(2)


class NATCastLatent(nn.Module):
    def __init__(
            self,
            embed_dim=256,
            forecast_depth=0,
            num_blocks=8,
            mlp_ratio=1,
            kernel_size=3,
            norm="none",
            layer_scale="none"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.forecast_depth = forecast_depth

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size for _ in range(forecast_depth)]
        
        if layer_scale == "auto":
            layer_scale=.5/(forecast_depth)
        elif layer_scale == "none":
            layer_scale = None
        self.forecast = nn.Sequential(
                *(NATBlock2D(
                        dim=embed_dim,
                        mlp_ratio=mlp_ratio, 
                        num_blocks=num_blocks,
                        norm=norm,
                        kernel_size=kernel_size[i],
                        layer_scale=layer_scale
                    )
                for i in range(forecast_depth))
                )
    
    def forward(self, x):
        x = x.squeeze(2).permute(0,2,3,1)
        x = torch.utils.checkpoint.checkpoint_sequential(self.forecast, self.forecast_depth, x, use_reentrant=False)
        return x.permute(0,3,1,2).unsqueeze(2)


class AFNONAT(nn.Module):
    def __init__(
            self,
            embed_dim=256,
            afno_num_blocks=8,
            nat_num_blocks=8,
            mlp_ratio=1,
            kernel_size=3,
            nat_norm="none",
            afno_norm="none",
            layer_scale=1,
            mode="sequential"
        ):
        super().__init__()
        self.mode = mode
        self.afno = AFNOBlock2D(
            dim=embed_dim,
            mlp_ratio=mlp_ratio, 
            num_blocks=afno_num_blocks,
            norm=afno_norm,
            channel_first=False,
            layer_scale=layer_scale
        )
        self.nat = NATBlock2D(
            dim=embed_dim,
            mlp_ratio=mlp_ratio, 
            num_blocks=nat_num_blocks,
            norm=nat_norm,
            kernel_size=kernel_size,
            layer_scale=layer_scale
        )
        if self.mode == "parallel":
            self.afno_alpha = nn.Parameter(.5 * torch.ones(embed_dim, dtype=torch.float32), requires_grad=True)
            self.nat_alpha = nn.Parameter(.5 * torch.ones(embed_dim, dtype=torch.float32), requires_grad=True)

    def forward(self, x, modes_range=None):
        if self.mode == "sequential":
            return self.nat(self.afno(x, modes_range=modes_range))
        elif self.mode == "parallel":
            nat_out = self.nat(x)
            # In-place scaling and addition
            nat_out.mul_(self.nat_alpha)  # In-place scaling
            nat_out.add_(self.afno(x,modes_range=modes_range).mul(self.afno_alpha))  # In-place addition
            return nat_out
        else:
            raise Exception("Mode needs to be 'sequential' or 'parallel'")


class AFNONATCastLatent(nn.Module):
    def __init__(
            self,
            embed_dim=256,
            forecast_depth=1,
            afno_num_blocks=8,
            nat_num_blocks=8,
            mlp_ratio=1,
            kernel_size=3,
            nat_norm="none",
            afno_norm="none",
            layer_scale="none",
            mode="sequential",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.forecast_depth = forecast_depth
        self.mode = mode
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size for _ in range(forecast_depth)]
        
        if layer_scale == "auto" and mode == "sequential":
            layer_scale = .5 / (forecast_depth * 2)
        elif layer_scale == "auto" and mode == "parallel":
            layer_scale = .5 / forecast_depth
        elif layer_scale == "none":
            layer_scale = None

        forecast = []
        for i in range(forecast_depth):
            forecast.append(AFNONAT(
                embed_dim=embed_dim,
                afno_num_blocks=afno_num_blocks,
                nat_num_blocks=nat_num_blocks,
                mlp_ratio=mlp_ratio,
                kernel_size=kernel_size[i],
                nat_norm=nat_norm,
                afno_norm=afno_norm,
                layer_scale=layer_scale,
                mode=mode
            ))
        
        self.forecast = nn.Sequential(*forecast)

    def forward(self, x, modes_range=None):
        x = x.squeeze(2).permute(0,2,3,1)
        for block in self.forecast:
            x = torch.utils.checkpoint.checkpoint(block, x, modes_range, use_reentrant=False)
        # x = torch.utils.checkpoint.checkpoint_sequential(self.forecast, self.forecast_depth, x, use_reentrant=False)
        return x.permute(0,3,1,2).unsqueeze(2)

class DummyLatent(nn.Module):
    def __init__(self):
        super().__init__()
        self.forecast = nn.Identity()
    def forward(self, x):
        return self.forecast(x)

class Nowcaster(nn.Module):
    def __init__(
            self,
            latent_model,
            encoder,
            decoder,
            in_steps,
    ):
        super().__init__()

        self.latent_model = latent_model
        self.encoder = encoder
        self.decoder = decoder
        self.in_steps = in_steps
    
    def forward(self, x, inv, n_steps=1):
        n_steps = min((n_steps, inv.shape[2]-self.in_steps+1))
        yhat = torch.empty((*x.shape[:2], n_steps, *x.shape[3:]), dtype=x.dtype, device=x.device)
        for i in range(n_steps):
            z = torch.cat((x, inv[:,:,i:i+self.in_steps]), dim=1)
            z = self.encoder(z)
            z = self.latent_model(z)
            z = self.decoder(z)
            yhat[:,:,i:i+1] = z
            if i < n_steps-1:
                x = torch.concat((x[:,:,-1:], z), dim=2)
        return yhat

class NowcasterLAR(nn.Module):
    def __init__(
            self,
            latent_model,
            encoder,
            inv_encoder,
            decoder,
            in_steps,
    ):
        super().__init__()

        self.latent_model = latent_model
        self.encoder = encoder
        self.inv_encoder = inv_encoder
        self.decoder = decoder
        self.in_steps = in_steps
    
    def forward(self, x, inv, n_steps=1):
        n_steps = min((n_steps, inv.shape[2]-self.in_steps+1))
        yhat = torch.empty((*x.shape[:2], n_steps, *x.shape[3:]), dtype=x.dtype, device=x.device)
        x = self.encoder(x)
        for i in range(n_steps):
            zinv = self.inv_encoder(inv[:,:,i:i+self.in_steps])
            z = torch.cat((x, zinv), dim=1)
            z = self.latent_model(z)
            yhat[:,:,i:i+1] = self.decoder(z)
            if i < n_steps-1:
                x = torch.concat((x[:,:,-1:], z), dim=2)
        return yhat




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
        depth=1
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
                    layer_scale=layer_scale
                    ) for _ in range(depth)])
            elif nat_dim == 3:
                self.nat_blocks = nn.Sequential(
                    *[NATBlock3D(
                    dim=out_ch,
                    mlp_ratio=mlp_ratio, 
                    num_blocks=num_blocks,
                    norm=norm,
                    kernel_size=kernel_size,
                    layer_scale=layer_scale
                    ) for _ in range(depth)])
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
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
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
        depth=1
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
                layer_scale=layer_scale
                ) for _ in range(depth)])
        else:
            self.nat_blocks = None

        
        self.skip_type = skip_type # (for 'add' or 'concat')
        if self.skip_type == "layer_scale":
            self.skip_scale = nn.Parameter(torch.rand((1,out_ch,1,1,1)) * 1e-3, requires_grad=True)
            

    def forward(self, x: torch.Tensor, skips: dict = None) -> torch.Tensor:
        x = self.upsample(x) 
        if skips is not None:
            skip = skips[x.shape[-2:]]
            if self.skip_type == 'add':
                x = x + skip
            elif self.skip_type == 'layer_scale':
                x = x + self.skip_scale * skip
            elif self.skip_type == 'concat':  # 'concat'
                x = torch.cat([x, skip], dim=1)
        x = x.squeeze(2).permute(0,2,3,1)
        if self.nat_blocks:
            for blk in self.nat_blocks: 
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
        return x.permute(0,3,1,2).unsqueeze(2)


################################################################################
# The UNet with NAT: Configurable Depth, Channels, Skip Connections
################################################################################

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
        final_conv=True
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
                depth=down_block_depths[i]
            )
            self.down_blocks.append(down_blk)
            prev_ch = block_out_ch


        self.up_blocks = nn.ModuleList()
        for i in range(self.up_depth):
            # We'll typically feed up_blocks[i] the output from the previous stage
            # plus a skip from down_blocks[depth - i - 1].
            # The in/out channels can be tuned as needed.
            in_ch = up_channels[i]
            
            
            if i < len(up_channels) - 1:
                out_ch = up_channels[i+1]
                s = skip_type
            else:
                if final_conv:
                    out_ch = 128
                else:
                    out_ch = out_channels
                s = None
            
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
                depth=up_block_depths[i]
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
        skips = {}
        for i in range(self.down_depth):
            x = self.down_blocks[i](x)
            if i in self.skip_down_levels:
                skips[x.shape[-2:]] = x

        for i in range(self.up_depth):
            if i in self.skip_up_levels:
                x = self.up_blocks[i](x, skips)
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
                x = torch.concat((x[:,:,:-1], z), dim=2)
        return yhat

################################################################################
# Example usage
################################################################################

if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    # Suppose we have a 2D input of shape (B=2, C=3, H=128, W=128)
    x = torch.randn(1, 11, 2, 512, 512)
    inv = torch.randn(1, 3, 2, 512, 512)
    model = UNAT(
        in_channels = 14,
        out_channels = 11,
        down_channels = [128, 256, 512],
        up_channels = [512, 256],
        down_strides = [(2,1,1),(1,2,2),(1,2,2)],  
        down_block_depths = [0,1,2], 
        down_kernel_sizes = [(5,5), (5,5), (5,5)],   # or (2,2,2) if 3D
        up_strides = [(1,2,2), (1,2,2)], 
        up_block_depths = [1, 0],
        up_kernel_sizes = [(5,5), (5,5)],  
        norm=None,
        layer_scale="add",
        mlp_ratio=4,
        num_blocks=1,
        skip_type=None,
        skip_down_levels=[],
        skip_up_levels=[],
        in_steps=2,    # or 'concat'
        final_conv=False
    )
    print(model)
    def count_parameters(model):
        """Count the number of trainable parameters in a PyTorch model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(count_parameters(model))
    
    flop_analyzer = FlopCountAnalysis(model, (x, inv, 1))
    print(flop_count_table(flop_analyzer))
    total_flops = flop_analyzer.total()
    print(f"Total FLOPs per forward pass: {total_flops / 1e9:.2f} GFLOPs")

    # Forward pass
    out = model(x, inv)
    print("Input shape :", x.shape)
    print("Output shape:", out.shape)
    # Should be (2, 3, 128, 128) if the up blocks exactly invert the downsamplings.