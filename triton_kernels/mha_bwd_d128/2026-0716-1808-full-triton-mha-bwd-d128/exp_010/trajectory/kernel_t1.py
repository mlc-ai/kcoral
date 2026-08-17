import torch
import triton
import triton.language as tl


@triton.jit
def load_tile(ptr, bh, row_start, S):
    """
    Load a [16, 128] bf16 tile from global memory.
    
    Args:
        ptr: Base pointer to the [B, H, S, d] tensor
        bh: Batch * Head index
        row_start: Starting row index in S
        S: Sequence length
    
    Returns:
        [16, 128] block tensor of bf16 values
    """
    row_offs = row_start + tl.arange(0, 16)
    col_offs = tl.arange(0, 128)
    # Construct the linear offset. 
    # Memory layout for a contiguous [B, H, S, d] tensor is ((bh * S + s) * d + f).
    ptrs = ptr + (bh * S + row_offs[:, None]) * 128 + col_offs[None, :]
    
    # Mask out-of-bounds rows. Out-of-bounds elements load as 0.0 (valid identity for our logic).
    mask = (row_offs[:, None] < S)
    return tl.load(ptrs, mask=mask, other=0.0)


@triton.jit
def store_tile(ptr, bh, row_start, value, S):
    """
    Store a [16, 128] bf16 tile to global memory.
    
    Args:
        ptr: Base pointer to the [B, H, S, d] tensor
        bh: Batch * Head index
        row_start: Starting row index in S
        value: [16, 128] block tensor of bf16 values
        S: Sequence length
    """
    row_offs = row_start + tl.arange(0, 16)
    col_offs = tl.arange(0, 128)
    ptrs = ptr + (bh * S + row_offs[:, None]) * 128 + col_offs[None, :]
    
    mask = (row_offs[:, None] < S)
    tl.store(ptrs, value, mask=mask)


@triton.jit
def dKdV_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
                S, scale):
    """
    FlashAttention backward dKdV kernel.
    
    Iterates over query blocks to compute gradients for a specific KV block.
    """
    j = tl.program_id(0)
    bh = tl.program_id(1)
    off_j = j * 16
    
    k_tile = load_tile(K_ptr, bh, off_j, S)
    v_tile = load_tile(V_ptr, bh, off_j, S)
    
    acc_dk = tl.zeros((16, 128), tl.float32)
    acc_dv = tl.zeros((16, 128), tl.float32)
    
    num_blocks = triton.cdiv(S, 16)
    
    for i in range(num_blocks):
        off_i = i * 16
        
        q_tile = load_tile(Q_ptr, bh, off_i, S)
        do_tile = load_tile(dO_ptr, bh, off_i, S)
        o_tile = load_tile(O_ptr, bh, off_i, S)
        
        # Compute scaled row sum component per query block row
        d_i = tl.sum(do_tile * o_tile, axis=1)
        
        # Use 0.0 padding for safely bounding mathematical sequences outside S bounds limits
        l_i = tl.load(L_ptr + bh * S + off_i + tl.arange(0, 16), mask=(off_i + tl.arange(0, 16)) < S, other=0.0)
        
        acc_s = tl.dot(q_tile, k_tile.T)
        acc_dp = tl.dot(do_tile, v_tile.T)
        
        s_tmp = acc_s * scale
        
        # Extrapolate dynamically shifted P bounds mapping relative to block i 
        p_tmp = tl.exp(s_tmp - l_i[:, None])
        ds_tmp = p_tmp * (acc_dp - d_i[:, None]) * scale
        
        acc_dv = tl.dot(p_tmp.T, do_tile, acc_dv)
        acc_dk = tl.dot(ds_tmp.T, q_tile, acc_dk)
        
    store_tile(dK_ptr, bh, off_j, acc_dk.to(tl.bfloat16), S)
    store_tile(dV_ptr, bh, off_j, acc_dv.to(tl.bfloat16), S)


@triton.jit
def dQ_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
              S, scale):
    """
    FlashAttention backward dQ kernel.
    
    Iterates over KV blocks to compute gradients for a specific Q block.
    """
    i = tl.program_id(0)
    bh = tl.program_id(1)
    off_i = i * 16
    
    q_tile = load_tile(Q_ptr, bh, off_i, S)
    do_tile = load_tile(dO_ptr, bh, off_i, S)
    o_tile = load_tile(O_ptr, bh, off_i, S)
    
    d_i = tl.sum(do_tile * o_tile, axis=1)
    
    # Use 0.0 padding for safely bounding mathematical sequences outside S bounds limits
    l_i = tl.load(L_ptr + bh * S + off_i + tl.arange(0, 16), mask=(off_i + tl.arange(0, 16)) < S, other=0.0)
    
    acc_dq = tl.zeros((16, 128), tl.float32)
    
    num_blocks = triton.cdiv(S, 16)
    
    for j in range(num_blocks):
        off_j = j * 16
        
        k_tile = load_tile(K_ptr, bh, off_j, S)
        v_tile = load_tile(V_ptr, bh, off_j, S)
        
        acc_s = tl.dot(q_tile, k_tile.T)
        acc_dp = tl.dot(do_tile, v_tile.T)
        
        s_tmp = acc_s * scale
        
        # Extrapolate dynamically shifted P bounds mapping relative to block i 
        p_tmp = tl.exp(s_tmp - l_i[:, None])
        ds_tmp = p_tmp * (acc_dp - d_i[:, None]) * scale
        
        acc_dq = tl.dot(ds_tmp, k_tile, acc_dq)
        
    store_tile(dQ_ptr, bh, off_i, acc_dq.to(tl.bfloat16), S)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Computes the backward pass of multi-head attention natively on CUDA."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / (128 ** 0.5)
    
    # Utilize massive parallelism mapped directly across independent sequence blocks 
    grid = (triton.cdiv(S, 16), B * H)
    
    # Dispatch independent cascades utilizing distinct CTAs mapping 
    dKdV_kernel[grid](Q, K, V, O, dO, L, dK, dV, S, scale, num_warps=4, num_stages=2)
    dQ_kernel[grid](Q, K, V, O, dO, L, dQ, S, scale, num_warps=4, num_stages=2)