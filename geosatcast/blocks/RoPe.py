import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

class Rope2DMixed(nn.Module):
    """
    Implements the "Mixed-Frequency 2D RoPE" from:
      "Rotary Position Embedding for Vision Transformer" (Heo et al., 2024)
    
    For each head h and each channel t in [0..(d_head//2 - 1)],
    Then we convert that to cos/sin for an even-odd interleaving in the
    standard RoPE style.

    We also add a simple caching mechanism so that if the input shape
    (H,W) doesn't change, we skip recomputing the expansions.
    """
    def __init__(self, num_heads: int, d_head: int, spherical_correction: bool = False, scale: float = 1.0):
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
        self.freq = nn.Parameter(torch.zeros(1, num_heads, half, 2))
        self.spherical_correction = spherical_correction
        self.scale = scale
        # Typical small init, so they can learn the best scale:
        nn.init.normal_(self.freq, mean=0.0, std=0.02)
        self._cache: Dict[Tuple[int,int,float,float], Tuple[torch.Tensor,torch.Tensor]] = {}

    def forward(
        self,
        q: torch.Tensor,  # [B, heads, H, W, d_head]
        k: torch.Tensor,  # [B, heads, H, W, d_head]
        coords: torch.Tensor,
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
        B, heads, H, W, d = q.shape
        
        assert heads == self.num_heads and d == self.d_head

        lat, lon = coords[...,0], coords[...,1]
        if self.spherical_correction:
            lon = lon * torch.cos(lat)
        lat = lat * self.scale 
        lon = lon * self.scale
        
        cos_map, sin_map = self._build_phase_maps(lat, lon, device, dtype)
        
        x_even_q = q[..., 0::2]
        x_odd_q  = q[..., 1::2]
        x_even_k = k[..., 0::2]
        x_odd_k  = k[..., 1::2]

        cos_map_reshaped = cos_map.permute(0,1,3,4,2)  # => [B, heads, H, W, half]
        sin_map_reshaped = sin_map.permute(0,1,3,4,2)  # => [B, heads, H, W, half]
        # # expand to [B, heads, H, W, half]
        # cos_map_reshaped = cos_map_reshaped.unsqueeze(0).expand(B, -1, -1, -1, -1)
        # sin_map_reshaped = sin_map_reshaped.unsqueeze(0).expand(B, -1, -1, -1, -1)

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
        lat, 
        lon,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create [heads, half, H, W] cos_map, sin_map for
        the 'mixed freq' approach. We'll do:
          phase[h,t,(r,c)] = r*res_y * freq[h,t,0] + c*res_x * freq[h,t,1]
        then cos,sin of that.
        """
        B, H, W = lat.shape
        freq = self.freq.to(device=device, dtype=dtype)

        freq_x = freq[..., 0].unsqueeze(-1).unsqueeze(-1)  
        freq_y = freq[..., 1].unsqueeze(-1).unsqueeze(-1)

        lon = lon.view(B,1,1,H,W)         
        lat = lat.view(B,1,1,H,W)         

        phase_x = freq_x * lon         
        phase_y = freq_y * lat         
        phase = phase_x + phase_y  # phase mixing

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
        self.freq = nn.Parameter(torch.zeros(num_heads, self.num_blocks, 2))
        nn.init.normal_(self.freq, mean=0.0, std=0.02)

    def forward(
        self,
        q: torch.Tensor,   
        k: torch.Tensor,   
        coords: torch.Tensor 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = q.device
        dtype = q.dtype
        B, heads, H, W, d = q.shape
        assert heads == self.num_heads and d == self.d_head

        # Ensure pos is on the right device and dtype. Now pos: [B, H, W, 2]
        coords = coords.to(device=device, dtype=dtype)
        # Extract latitude and longitude from pos:
        lat = coords[..., 0]  # shape: [B, H, W]
        lon = coords[..., 1]  # shape: [B, H, W]

        # Expand learned frequencies: shape becomes [heads, num_blocks, 1, 1, 2]
        freq = self.freq.unsqueeze(2).unsqueeze(3)  # [heads, num_blocks, 1, 1, 2]

        lat = lat.unsqueeze(1).unsqueeze(2) * freq[..., 1]  # [B, 1, H, W]
        lon = lon.unsqueeze(1).unsqueeze(2) * freq[..., 0]  # [B, 1, H, W]

        row1 = torch.stack(
          [
            torch.cos(lon), 
            -torch.cos(lat) * torch.sin(lon), 
            torch.sin(lat) * torch.sin(lon)
            ], dim=-1)
        row2 = torch.stack(
          [
            torch.sin(lon), 
            torch.cos(lat) * torch.cos(lon), 
            -torch.sin(lat) * torch.cos(lon)
            ], dim=-1)
        row3 = torch.stack(
          [torch.zeros_like(lon), 
          torch.sin(lat), 
          torch.cos(lat)
          ], dim=-1)

        # Final rotation matrices: shape [B, heads, num_blocks, H, W, 3, 3]
        rot_matrices = torch.stack([row1, row2, row3], dim=-2)

        q = q.view(B, heads, H, W, self.num_blocks, 3)
        k = k.view(B, heads, H, W, self.num_blocks, 3)

        rot = rot_matrices.permute(0, 1, 3, 4, 2, 5, 6)

        q = torch.matmul(q.unsqueeze(-2), rot).squeeze(-2)
        k = torch.matmul(k.unsqueeze(-2), rot).squeeze(-2)

        # Reshape back to original shape [B, heads, H, W, d_head]
        q = q.view(B, heads, H, W, d)
        k = k.view(B, heads, H, W, d)
        return q, k

class TemporalRoPE(nn.Module):
    """
    Implements a simple 1D RoPE for temporal positions.
    Assumes the time dimension is treated independently.
    """
    def __init__(self, d_model: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model for temporal RoPE must be even.")
        self.d_model = d_model
        half = d_model // 2
        # Learnable frequencies for time dimension: shape [half]
        self.freq = nn.Parameter(torch.zeros(half))
        nn.init.normal_(self.freq, mean=0.0, std=0.02)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B, T] tensor of time positions (e.g. in seconds or radians for cyclic time)
        Returns a positional embedding of shape [B, T, d_model]
        """
        B, T = t.shape
        # Expand time positions: [B, T, 1]
        t_exp = t.unsqueeze(-1)  # [B, T, 1]
        # Compute phase: [B, T, half]
        phase = t_exp * self.freq.unsqueeze(0).unsqueeze(0)  # broadcasting
        cos_phase = torch.cos(phase)  # [B, T, half]
        sin_phase = torch.sin(phase)  # [B, T, half]
        # Interleave to form a d_model vector per time step:
        # This mimics the standard RoPE even-odd interleaving.
        emb = torch.zeros(B, T, self.d_model, device=t.device, dtype=t.dtype)
        emb[..., 0::2] = cos_phase
        emb[..., 1::2] = sin_phase
        return emb

class SpatioTemporalRoPE(nn.Module):
    """
    Factorized Spatiotemporal RoPE.
    Applies Spherical RoPE for spatial dimensions (H, W) and a separate temporal RoPE
    for the time dimension. The final q and k embeddings are adjusted by the sum of the
    two positional encodings.
    
    Expects:
      - q, k: [B, T, heads, H, W, d_head]
      - spatial_pos: [B, H, W, 2] (lat, lon in radians)
      - time_pos: [B, T] (time positions, e.g. seconds or normalized phase)
    """
    def __init__(self, num_heads: int, d_head: int, d_time: Optional[int] = None):
        super().__init__()
        # For spatial RoPE, d_head must be a multiple of 3.
        if d_head % 3 != 0:
            raise ValueError("For Spherical RoPE, d_head must be a multiple of 3.")
        self.spatial_rope = SphericalRoPE(num_heads, d_head)
        # For temporal RoPE, we can choose to use the same d_head or a smaller one.
        # Here, we choose d_time equal to d_head for simplicity (you can adjust as needed).
        if d_time is None:
            d_time = d_head
        if d_time % 2 != 0:
            raise ValueError("d_time for Temporal RoPE must be even.")
        self.temporal_rope = TemporalRoPE(d_time)
        # Project temporal encoding to d_head dimension if necessary.
        if d_time != d_head:
            self.temp_proj = nn.Linear(d_time, d_head)
        else:
            self.temp_proj = nn.Identity()
    
    def forward(
        self,
        q: torch.Tensor,  # [B, T, heads, H, W, d_head]
        k: torch.Tensor,  # [B, T, heads, H, W, d_head]
        spatial_pos: torch.Tensor,  # [B, H, W, 2]
        time_pos: torch.Tensor      # [B, T]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, heads, H, W, d_head = q.shape
        
        # Compute spatial encoding per sample (applied independently per time slice)
        # We reshape time and batch to combine them: [B*T, heads, H, W, d_head]
        q_reshaped = q.view(B*T, heads, H, W, d_head)
        k_reshaped = k.view(B*T, heads, H, W, d_head)
        # For spatial positions, we assume the same grid per sample in the batch.
        spatial_encoded_q, spatial_encoded_k = self.spatial_rope(q_reshaped, k_reshaped, spatial_pos.shape[1], spatial_pos.shape[2], 
                                                                (1.0, 1.0))  # adjust resolution as needed
        # Reshape back: [B, T, heads, H, W, d_head]
        spatial_encoded_q = spatial_encoded_q.view(B, T, heads, H, W, d_head)
        spatial_encoded_k = spatial_encoded_k.view(B, T, heads, H, W, d_head)
        
        # Compute temporal encoding: [B, T, d_time]
        temp_enc = self.temporal_rope(time_pos)  # [B, T, d_time]
        temp_enc = self.temp_proj(temp_enc)      # [B, T, d_head]
        # Expand temporal encoding to shape [B, T, 1, 1, 1, d_head] to broadcast over heads, H, W.
        temp_enc = temp_enc.view(B, T, 1, 1, 1, d_head)
        
        # Combine: here we add the spatial and temporal positional adjustments.
        # You could also choose a different combination (like concatenation and a linear projection)
        q_out = spatial_encoded_q + temp_enc
        k_out = spatial_encoded_k + temp_enc
        return q_out, k_out


if __name__ == "__main__":
    lon = torch.arange(-5, 5, 0.05)
    lat = torch.arange(-5, 5, 0.05)
    coords = torch.stack(torch.meshgrid(lat, lon), dim=-1)
    q = k = torch.ones((1,1,200,200,6))
    print(lon.shape, lat.shape)

    rope = Rope2DMixed(1,6)

    q, k = rope(q, k, coords)
    print(q, k )