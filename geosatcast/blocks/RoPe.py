import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

class Rope2DMixed(nn.Module):
    """
    Implements the "Mixed-Frequency 2D RoPE" from:
      "Rotary Position Embedding for Vision Transformer" (Heo et al., 2024)
    
    For each head h and each channel t in [0..(d_head//2 - 1)],
    we learn two parameters: (theta_x[h,t], theta_y[h,t]), so that
      phase(r,c,t,h) = (r * resolution_y) * theta_x[h,t]
                     + (c * resolution_x) * theta_y[h,t]
    Then we convert that to cos/sin for an even-odd interleaving in the
    standard RoPE style.

    We also add a simple caching mechanism so that if the input shape
    (H,W) doesn't change, we skip recomputing the expansions.
    """
    def __init__(self, num_heads: int, d_head: int):
        """
        Args:
          num_heads: number of attention heads
          d_head: per-head dimension. Must be an even integer so we
                  can do the standard (even, odd) interleaving.
        """
        super().__init__()
        if d_head % 2 != 0:
            raise ValueError("For Mixed-Frequency RoPE, d_head must be even.")
        self.num_heads = num_heads
        self.d_head = d_head
        half = d_head // 2

        # Learnable frequencies: shape => [num_heads, half, 2]
        #   freq[h, t, 0] = theta_x
        #   freq[h, t, 1] = theta_y
        # (the paper sometimes uses one param per layer, but we can store them here.)
        self.freq = nn.Parameter(torch.zeros(num_heads, half, 2))
        # Typical small init, so they can learn the best scale:
        nn.init.normal_(self.freq, mean=0.0, std=0.02)

        # We'll keep a dict for caching the expansions:
        #   _cache[(H, W, resolution_y, resolution_x)] = (cos_map, sin_map)
        # Because we might feed different resolutions or shapes. 
        # cos_map, sin_map => shape [num_heads, half, H, W]
        self._cache: Dict[Tuple[int,int,float,float], Tuple[torch.Tensor,torch.Tensor]] = {}

    def forward(
        self,
        q: torch.Tensor,  # [B, heads, H, W, d_head]
        k: torch.Tensor,  # [B, heads, H, W, d_head]
        H: int,
        W: int,
        resolution: Tuple[float, float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Mixed-Frequency 2D RoPE to (q, k).

        Inputs:
          q,k => [B, heads, H, W, d_head]
          H,W => actual image/spatial size
          resolution => (res_y, res_x) multiplied by row/col index

        Returns:
          q_out, k_out => the same shape, but with RoPE-applied channels.
        """
        device = q.device
        dtype = q.dtype
        B, heads, H_, W_, d = q.shape
        assert H_ == H and W_ == W
        assert heads == self.num_heads and d == self.d_head

        # 1) Retrieve or build the cos_map, sin_map from cache
        res_y, res_x = resolution
        # cache_key = (H, W, float(res_y), float(res_x))
        # if cache_key in self._cache:
        #     cos_map, sin_map = self._cache[cache_key]
        #     # they might be on a different device if we transferred the model
        #     # so re-locate them if needed:
        #     if cos_map.device != device:
        #         cos_map = cos_map.to(device=device, dtype=dtype)
        #         sin_map = sin_map.to(device=device, dtype=dtype)
        #         self._cache[cache_key] = (cos_map, sin_map)
        # else:
        #     # Build them new
        #     with torch.no_grad():
        #         # If we already have cos_map in the cache, skip building it
        #         # else build it, all in no_grad block:
        #         cos_map, sin_map = self._build_phase_maps(H, W, resolution)
        cos_map, sin_map = self._build_phase_maps(H, W, res_y, res_x, device, dtype)
            # self._cache[cache_key] = (cos_map, sin_map)

        # cos_map, sin_map => [heads, half, H, W]
        # We'll apply them to the "first half" of q,k channels in the usual
        # “even-odd” rearrangement:
        #   x_even' = x_even * cos - x_odd * sin
        #   x_odd'  = x_odd  * cos + x_even * sin
        #
        # So we first split q => q_row = q[..., :half], q_col = q[..., half:]
        # But for "mixed" approach, we do the standard 1D rope formula on the entire "first half".
        # Then the second half is unaffected by these angles.  Actually, for "Mixed" from the paper,
        # we typically apply these expansions across the entire dimension. But the standard approach
        # is that the freq dimension is half the channels. Let's implement the classic even-odd trick
        # across the entire d_head dimension.

        # We'll do it all at once by chunking even/odd indices of the last dimension:
        # shape => [B, heads, H, W, d_head]
        x_even_q = q[..., 0::2]
        x_odd_q  = q[..., 1::2]
        x_even_k = k[..., 0::2]
        x_odd_k  = k[..., 1::2]

        # cos_map, sin_map => broadcast to [B, heads, H, W, half]
        # but we only have shape [heads, half, H, W]. 
        # We'll reorder them to [heads, H, W, half] so we can broadcast
        # along the batch dimension easily:
        cos_map_reshaped = cos_map.permute(0,2,3,1)  # => [heads, H, W, half]
        sin_map_reshaped = sin_map.permute(0,2,3,1)  # => [heads, H, W, half]

        # expand to [B, heads, H, W, half]
        cos_map_reshaped = cos_map_reshaped.unsqueeze(0).expand(B, -1, -1, -1, -1)
        sin_map_reshaped = sin_map_reshaped.unsqueeze(0).expand(B, -1, -1, -1, -1)

        # Now apply the standard rope formula on pairs (x_even, x_odd):
        x_even_q2 = x_even_q * cos_map_reshaped - x_odd_q * sin_map_reshaped
        x_odd_q2  = x_odd_q  * cos_map_reshaped + x_even_q * sin_map_reshaped

        x_even_k2 = x_even_k * cos_map_reshaped - x_odd_k * sin_map_reshaped
        x_odd_k2  = x_odd_k  * cos_map_reshaped + x_even_k * sin_map_reshaped

        # Re-interleave them back along last dimension
        # shape => [B, heads, H, W, d_head]
        q_out = torch.zeros_like(q)
        k_out = torch.zeros_like(k)
        q_out[..., 0::2] = x_even_q2
        q_out[..., 1::2] = x_odd_q2
        k_out[..., 0::2] = x_even_k2
        k_out[..., 1::2] = x_odd_k2

        return q_out, k_out

    def _build_phase_maps(
        self,
        H: int,
        W: int,
        res_y: float,
        res_x: float,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create [heads, half, H, W] cos_map, sin_map for
        the 'mixed freq' approach. We'll do:
          phase[h,t,(r,c)] = r*res_y * freq[h,t,0] + c*res_x * freq[h,t,1]
        then cos,sin of that.
        """
        # shape => [num_heads, half, 2]
        freq = self.freq.to(device=device, dtype=dtype)

        # row coords => shape [H], col coords => shape [W]
        # multiply by resolution
        rr = torch.arange(H, device=device, dtype=dtype) * res_y  # [H]
        cc = torch.arange(W, device=device, dtype=dtype) * res_x  # [W]

        # We'll build the expansions as:
        # phase[h, t, r, c] = rr[r]*freq[h,t,0] + cc[c]*freq[h,t,1]
        # then we take sin/cos over that 4D array
        # We'll do it in a broadcast-friendly way.

        # shape => [num_heads, half, H, W]
        # We'll do a small nested approach for clarity:
        # We'll do freq[h,t,0] * rr[r], plus freq[h,t,1] * cc[c].
        # The naive approach is a double for-loop, but let's vectorize:
        #   freq[h,t,0] => shape [num_heads, half] => we want [num_heads, half, H, W]
        #   rr[r] => shape [H], cc[c] => shape [W].
        # We'll expand them carefully.
        # freq_x = freq[..., 0] => shape [num_heads, half]
        # freq_y = freq[..., 1] => shape [num_heads, half]
        freq_x = freq[..., 0].unsqueeze(-1).unsqueeze(-1)  # [num_heads, half, 1, 1]
        freq_y = freq[..., 1].unsqueeze(-1).unsqueeze(-1)

        rr_2d = rr.view(1,1,H,1)         # => [1,1,H,1]
        cc_2d = cc.view(1,1,1,W)         # => [1,1,1,W]

        phase_x = freq_x * rr_2d         # => [num_heads, half, H, W]
        phase_y = freq_y * cc_2d         # => [num_heads, half, H, W]
        phase = phase_x + phase_y        # => [num_heads, half, H, W]

        cos_map = torch.cos(phase)
        sin_map = torch.sin(phase)
        return cos_map, sin_map




class SphericalRoPE(nn.Module):
    """
    Implements Spherical RoPE for geotokens.
    Assumes d_head is a multiple of 3, so that each head’s embedding is split
    into 3-dimensional blocks. For each head and block, two learnable frequencies
    (for longitude and latitude) are used to compute phase angles.

    For a geotoken at position pos with shape [B, H, W, 2] (where pos[...,0]=latitude
    and pos[...,1]=longitude in radians), we compute:
      A = longitude * freq[..., 0]
      B = latitude  * freq[..., 1]

    Then, for each block we construct the rotation matrix:
          [ cos(A)      -cos(B)*sin(A)    sin(B)*sin(A) ]
          [ sin(A)       cos(B)*cos(A)   -sin(B)*cos(A) ]
          [ 0            sin(B)           cos(B)       ]
    
    This 3x3 rotation is applied to each block of 3 channels of the input embeddings.
    """
    def __init__(self, num_heads: int, d_head: int):
        super().__init__()
        if d_head % 3 != 0:
            raise ValueError("For Spherical RoPE, d_head must be a multiple of 3.")
        self.num_heads = num_heads
        self.d_head = d_head
        self.num_blocks = d_head // 3  # each block has 3 channels

        # Learnable frequencies: shape [num_heads, num_blocks, 2]
        # freq[..., 0] used for longitude, freq[..., 1] used for latitude.
        self.freq = nn.Parameter(torch.zeros(num_heads, self.num_blocks, 2))
        nn.init.normal_(self.freq, mean=0.0, std=0.02)

    def forward(
        self,
        q: torch.Tensor,   # shape: [B, heads, H, W, d_head]
        k: torch.Tensor,   # shape: [B, heads, H, W, d_head]
        pos: torch.Tensor  # shape: [B, H, W, 2] (pos[...,0]=lat, pos[...,1]=lon in radians)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = q.device
        dtype = q.dtype
        B, heads, H, W, d = q.shape
        assert heads == self.num_heads and d == self.d_head

        # Ensure pos is on the right device and dtype. Now pos: [B, H, W, 2]
        pos = pos.to(device=device, dtype=dtype)
        # Extract latitude and longitude from pos:
        lat = pos[..., 0]  # shape: [B, H, W]
        lon = pos[..., 1]  # shape: [B, H, W]

        # Expand learned frequencies: shape becomes [heads, num_blocks, 1, 1, 2]
        freq = self.freq.unsqueeze(2).unsqueeze(3)  # [heads, num_blocks, 1, 1, 2]

        # We need to broadcast pos so that we can compute phase for each sample, head, and block.
        # Expand lat and lon to shape [B, 1, H, W] so they broadcast with freq.
        lat_exp = lat.unsqueeze(1)  # [B, 1, H, W]
        lon_exp = lon.unsqueeze(1)  # [B, 1, H, W]

        # Compute phase angles A and B:
        # A = lon * freq[..., 0]
        # B = lat * freq[..., 1]
        # Resulting shape: [B, heads, num_blocks, H, W]
        A = lon_exp.unsqueeze(2) * freq[..., 0]
        B = lat_exp.unsqueeze(2) * freq[..., 1]

        # Compute trigonometric functions for rotation matrix elements.
        r11 = torch.cos(A)                    # [B, heads, num_blocks, H, W]
        r12 = -torch.cos(B) * torch.sin(A)
        r13 = torch.sin(B) * torch.sin(A)
        r21 = torch.sin(A)
        r22 = torch.cos(B) * torch.cos(A)
        r23 = -torch.sin(B) * torch.cos(A)
        r31 = torch.zeros_like(A)
        r32 = torch.sin(B)
        r33 = torch.cos(B)

        # Stack rows to form each 3x3 rotation matrix.
        # Each row: shape [B, heads, num_blocks, H, W, 3]
        row1 = torch.stack([r11, r12, r13], dim=-1)
        row2 = torch.stack([r21, r22, r23], dim=-1)
        row3 = torch.stack([r31, r32, r33], dim=-1)
        # Final rotation matrices: shape [B, heads, num_blocks, H, W, 3, 3]
        rot_matrices = torch.stack([row1, row2, row3], dim=-2)

        # Reshape q and k into blocks of 3 channels.
        # From [B, heads, H, W, d_head] to [B, heads, H, W, num_blocks, 3]
        q_blocks = q.view(B, heads, H, W, self.num_blocks, 3)
        k_blocks = k.view(B, heads, H, W, self.num_blocks, 3)

        # Apply the rotation matrices to each block.
        # We perform a batch matrix multiplication on the last two dims.
        # q_blocks.unsqueeze(-2): shape [B, heads, H, W, num_blocks, 1, 3]
        # rot_matrices: shape [B, heads, num_blocks, H, W, 3, 3]
        # But note: our rot_matrices is currently ordered as [B, heads, num_blocks, H, W, 3, 3].
        # We need to align the H, W and num_blocks dims. Let's permute rot_matrices to [B, heads, H, W, num_blocks, 3, 3].
        rot = rot_matrices.permute(0, 1, 3, 4, 2, 5, 6)
        # Now rot: [B, heads, H, W, num_blocks, 3, 3]
        q_rot = torch.matmul(q_blocks.unsqueeze(-2), rot).squeeze(-2)
        k_rot = torch.matmul(k_blocks.unsqueeze(-2), rot).squeeze(-2)

        # Reshape back to original shape [B, heads, H, W, d_head]
        q_out = q_rot.view(B, heads, H, W, d)
        k_out = k_rot.view(B, heads, H, W, d)
        return q_out, k_out