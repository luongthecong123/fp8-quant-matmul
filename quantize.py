import torch

# https://gist.github.com/malfet/7874d96b99670c3da83cbb779ab770c6
def to_float8(x, dtype=torch.float8_e4m3fnuz):
    finfo = torch.finfo(dtype)
    # Calculate the scale as dtype max divided by absmax
    scale = finfo.max / x.abs().max().clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    return x_scl_sat.to(dtype), scale.float().reciprocal()

def to_float8_block_scale(x, dtype=torch.float8_e4m3fnuz):
    finfo = torch.finfo(dtype)
    M, K = x.shape
    
    # Calculate number of blocks
    # M_blocks = (M + 127) // 128
    # K_blocks = (K + 127) // 128
    M_blocks = M // 128
    K_blocks = K // 128    
    # Reshape to blocks: (M_blocks, K_blocks, 128, 128)
    x_blocks = x.view(M_blocks, 128, K_blocks, 128).transpose(1, 2)
    
    # Calculate scale per block (vectorized)
    block_absmax = x_blocks.abs().amax(dim=(-2, -1)).clamp(min=1e-12)
    block_scales = (finfo.max / block_absmax).unsqueeze(-1).unsqueeze(-1)
    
    # Scale and clamp all blocks at once
    x_scaled_blocks = (x_blocks * block_scales).clamp(min=finfo.min, max=finfo.max)
    
    # Reshape back
    x_scaled = x_scaled_blocks.transpose(1, 2).view(M, K).to(dtype)
    
    # Return inverse scales
    scales_inv = block_scales.squeeze(-1).squeeze(-1).float().reciprocal()
    
    return x_scaled, scales_inv.contiguous()

def to_float8_rowwise_scale(x, dtype=torch.float8_e4m3fnuz):
    finfo = torch.finfo(dtype)
    M, K = x.shape
    
    # K must be divisible by 128
    assert K % 128 == 0, f"K ({K}) must be divisible by 128"
    
    K_blocks = K // 128
    
    # Reshape to (M, K_blocks, 128) - group columns into blocks of 128
    x_grouped = x.view(M, K_blocks, 128)
    
    # Calculate scale per row-block: max absolute value across the 128 columns
    row_block_absmax = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)  # Shape: (M, K_blocks, 1)
    row_block_scales = (finfo.max / row_block_absmax)  # Shape: (M, K_blocks, 1)
    
    # Scale and clamp each row-block
    x_scaled_grouped = (x_grouped * row_block_scales).clamp(min=finfo.min, max=finfo.max)
    
    # Reshape back to original shape
    x_scaled = x_scaled_grouped.view(M, K).to(dtype)
    
    # Return inverse scales with shape (M, K//128)
    x_inv_s = row_block_scales.squeeze(-1).float().reciprocal()  # Shape: (M, K_blocks)
    
    return x_scaled, x_inv_s