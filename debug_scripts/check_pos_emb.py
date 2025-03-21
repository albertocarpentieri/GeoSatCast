import torch 
import matplotlib.pyplot as plt
from geosatcast.blocks.RoPe import Rope2DMixed, SphericalRoPE

def compute_attention_map(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """
    Compute dot-product attention scores between a single query (shape [d])
    and a set of keys (shape [H, W, d]). Returns a tensor of shape [H, W].
    """
    # Broadcast query over keys and compute dot product along last dim.
    attn = (keys * query.view(1, 1, -1)).sum(dim=-1)
    return attn

def plot_attention_maps(raw_attn: torch.Tensor, rope_attn: torch.Tensor, title_prefix: str, ax_row):
    """
    Plot the softmaxed raw attention, rope-transformed attention,
    and their difference.
    """
    # Apply softmax (flatten, then reshape)
    raw_attn_soft = torch.softmax(raw_attn.view(-1), dim=0).view(raw_attn.shape)
    rope_attn_soft = torch.softmax(rope_attn.view(-1), dim=0).view(rope_attn.shape)
    
    im0 = ax_row[0].imshow(raw_attn_soft.detach().numpy(), cmap="viridis")
    ax_row[0].set_title(f"{title_prefix} Raw Attention")
    plt.colorbar(im0, ax=ax_row[0])
    
    im1 = ax_row[1].imshow(rope_attn_soft.detach().numpy(), cmap="viridis")
    ax_row[1].set_title(f"{title_prefix} RoPE Attention")
    plt.colorbar(im1, ax=ax_row[1])
    
    im2 = ax_row[2].imshow((rope_attn_soft - raw_attn_soft).detach().numpy(), cmap="bwr", vmin=-1e-3,vmax=1e-3)
    ax_row[2].set_title(f"{title_prefix} Difference")
    plt.colorbar(im2, ax=ax_row[2])

def main():
    # Use an even-sized patch, here 8x8.
    patch_H, patch_W = 21, 21
    # Define a grid of lat-lon values. For demonstration, let lat go from 15 to 15+patch_H*0.05,
    # and lon go from -15 to -15+patch_W*0.05.
    lat = torch.linspace(15, 15 + (patch_H-1)*0.05, patch_H)
    lon = torch.linspace(-15, -15 + (patch_W-1)*0.05, patch_W)
    grid = torch.stack(torch.meshgrid(lat, lon, indexing="ij"), dim=-1)  # [patch_H, patch_W, 2]
    grid = grid.unsqueeze(0)  # Add batch dimension: [1, patch_H, patch_W, 2]
    
    # Define a center index for our even-sized patch.
    center_i, center_j = 10, 10  # For an 8x8 patch, center can be defined as (3,3)
    
    ###########################################
    # Test with Rope2DMixed
    d_head_rope = 6
    q = torch.ones(1, 1, patch_H, patch_W, d_head_rope) / 2
    k = torch.randn(1, 1, patch_H, patch_W, d_head_rope)
    


    # Compute raw attention (before rope) for comparison:
    q_raw_center = q[0, 0, center_i, center_j]  # shape [d_head]
    raw_attn_rope = compute_attention_map(q_raw_center, k[0, 0])  # [patch_H, patch_W]
    
    # Apply Rope2DMixed
    rope2d = Rope2DMixed(1, d_head_rope)
    rope_q, rope_k = rope2d(q, k, grid)
    q_rope_center = rope_q[0, 0, center_i, center_j]
    rope_attn_rope = compute_attention_map(q_rope_center, k[0, 0])
    
    ###########################################
    # Compute raw attention (before spherical rope)
    q_raw_center_sph = q[0, 0, center_i, center_j]
    raw_attn_sph = compute_attention_map(q_raw_center_sph, k[0, 0])
    
    # Apply SphericalRoPE
    sph_rope = SphericalRoPE(1, d_head_rope)
    sph_q, sph_k = sph_rope(q, k, grid)
    q_sph_center = sph_q[0, 0, center_i, center_j]
    rope_attn_sph = compute_attention_map(q_sph_center, k[0, 0])
    
    ###########################################
    # Plotting results for both methods.
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    plot_attention_maps(raw_attn_rope, rope_attn_rope, "Rope2DMixed", axes[0])
    plot_attention_maps(raw_attn_sph, rope_attn_sph, "SphericalRoPE", axes[1])
    
    plt.tight_layout()
    plt.savefig("/capstor/scratch/cscs/acarpent/attention_weights_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()