from geosatcast.models.UNAT import UNAT 
import torch

model = UNAT(
        in_channels=14,
        out_channels=11,
        down_channels=[192, 1152],
        down_gated=False,
        down_num_blocks=[8, 8],
        down_mlp_ratio=4,
        down_strides=[[2,1,1],[1,4,4]],
        down_block_depths=[24, 24], 
        down_kernel_sizes=[(9,9), (9,9)],
        up_channels=[192],  
        up_gated=False,
        up_num_blocks=8,
        up_mlp_ratio=4,
        up_strides=[[1,4,4]],
        up_block_depths=[24],
        up_kernel_sizes=[(9,9)],
        norm=None,
        layer_scale=0.01,
        skip_type='inv_crossnat',
        skip_down_levels=[0],
        skip_up_levels=[0],
        in_steps=2,    # or 'concat'
        final_conv=True,
        resolution=1.0,
        emb_method=None,
        downsample_type=["timecrossnat", "avgpool"],   # can be str or list[str]
        upsample_type="interp",  # can be str or list[str]
        interp_mode="nearest").to("cuda")

x = torch.randn((1,11,2,128,128)).to("cuda")
inv = torch.randn((1,3,2,128,128)).to("cuda")
grid = torch.randn((1,128,128,2)).to("cuda")
x = model(x,inv,grid)