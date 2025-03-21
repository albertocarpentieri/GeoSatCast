from geosatcast.train.distribute_training import load_unatcast, load_predrnn, load_predformer
from geosatcast.models.predformer import PredFormer_Model
from fvcore.nn import FlopCountAnalysis, flop_count_table
from thop import profile
import torch
import psutil
import time

def count_parameters(model):
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

unatcast = load_unatcast("Checkpoints/UNATCast-large/UNATCast-large-ks13-nb1-dd4_4_4-ud4_4-spherical_rope-192_768_768-L1/UNATCast-large-ks13-nb1-dd4_4_4-ud4_4-spherical_rope-192_768_768-L1_0.pt").to("cuda")
print("unatcast:", count_parameters(unatcast))
predrnn = load_predrnn("/capstor/scratch/cscs/acarpent/Checkpoints/predrnn/predrnn-inv-s2-fd_4-nh_64/predrnn-inv-s2-fd_4-nh_64_0.pt").to("cuda")
print("predrnn:", count_parameters(predrnn))
# predformer = load_predformer("/capstor/scratch/cscs/acarpent/Checkpoints/predformer/predformer/predformer_0.pt").to("cuda")
# print("predformer:", count_parameters(predformer))

model_config = {
        'patch_size': 8,
        'pre_seq': 2,
        'dim': 1024,
        'in_channels': 14,
        'out_channels': 11,
        'heads': 8,
        'dim_head': 32,
        'dropout': 0.1,
        'attn_dropout': 0.1,
        'drop_path': 0.25,
        'scale_dim': 2,
        'Ndepth': 2,
}
predformer = PredFormer_Model(model_config, in_steps=2).to("cuda")
print("predformer:", count_parameters(predformer))

x = torch.randn(1,11,2,768,768).to("cuda")
inv = torch.randn(1,3,18,768,768).to("cuda")
grid = torch.randn(1,768,768,2).to("cuda")
with torch.no_grad():
    torch.cuda.reset_peak_memory_stats("cuda")
    start_time = time.time()
    _ = predrnn(x, inv, 16)
    end_time = time.time()
    flops = FlopCountAnalysis(predrnn, (x, inv, 16))
    flops_thop, params_thop = profile(predrnn, inputs=(x, inv, 16))
    flops = flops.total() / 1e9
    peak_memory = torch.cuda.max_memory_allocated("cuda") / 1024**2  # in MB
    final_memory = torch.cuda.memory_allocated("cuda") / 1024**2     # in MB
    print(f"Total GFLOPs (predrnn): {flops:.2f} GFLOPs, {peak_memory}, {final_memory}, {end_time-start_time}")
    print(f"Total Thop GFLOPs (predrnn): {flops_thop / 1e9:.2f} GFLOPs")

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats("cuda")
    start_time = time.time()
    _ = unatcast(x, inv, grid, 16)
    end_time = time.time()
    flops = FlopCountAnalysis(unatcast, (x, inv, grid, 16))
    flops_thop, params_thop = profile(unatcast, inputs=(x, inv, grid, 16))
    flops = flops.total() / 1e9
    peak_memory = torch.cuda.max_memory_allocated("cuda") / 1024**2  # in MB
    final_memory = torch.cuda.memory_allocated("cuda") / 1024**2     # in MB
    print(f"Total GFLOPs (unat): {flops:.2f} GFLOPs, {peak_memory}, {final_memory}, {end_time-start_time}")
    print(f"Total Thop GFLOPs (predrnn): {flops_thop / 1e9:.2f} GFLOPs")

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats("cuda")
    start_time = time.time()
    _ = predformer(x, inv, 16)
    end_time = time.time()
    flops = FlopCountAnalysis(predformer, (x, inv, 16))
    flops_thop, params_thop = profile(predformer, inputs=(x, inv, 16))
    flops = flops.total() / 1e9
    peak_memory = torch.cuda.max_memory_allocated("cuda") / 1024**2  # in MB
    final_memory = torch.cuda.memory_allocated("cuda") / 1024**2     # in MB
    print(f"Total GFLOPs (predformer): {flops:.2f} GFLOPs, {peak_memory}, {final_memory}, {end_time-start_time}")
    print(f"Total Thop GFLOPs (predrnn): {flops_thop / 1e9:.2f} GFLOPs")



