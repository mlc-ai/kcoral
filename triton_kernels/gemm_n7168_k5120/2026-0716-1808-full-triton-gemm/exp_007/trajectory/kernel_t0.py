import torch
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(A_ptr, B_ptr, C_ptr, M, N, K, 
                stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, 
                BLOCK_K_inner: tl.constexpr, NUM_K_TILES: tl.constexpr, GROUP_M: tl.constexpr):
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    tile_id = tl.program_id(0)
    num_tiles_per_group = GROUP_M * num_pid_n
    group_id = tile_id // num_tiles_per_group
    first_m = group_id * GROUP_M
    m_range = min(num_pid_m - first_m, GROUP_M)
    
    idx = tile_id % num_tiles_per_group
    m_tile = first_m + (idx % m_range)
    n_tile = idx // m_range
    
    m_offsets = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    for k_chunk in range(2):
        for k_tile in range(NUM_K_TILES):
            k_abs = k_chunk * 2560 + k_tile * BLOCK_K_inner + tl.arange(0, BLOCK_K_inner)
            
            mask_a = (m_offsets[:, None] < M) & (k_abs[None, :] < 5120)
            a = tl.load(A_ptr + m_offsets[:, None] * stride_am + k_abs[None, :] * stride_ak, 
                        mask=mask_a, other=0.0)
            
            mask_b = (n_offsets[None, :] < 7168) & (k_abs[:, None] < 5120)
            b = tl.load(B_ptr + n_offsets[None, :] * stride_bn + k_abs[:, None] * stride_bk, 
                        mask=mask_b, other=0.0)
            
            b_transposed = b.T
            acc = tl.dot(a, b_transposed, acc)
            
    out_ptr = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    mask_c = (m_offsets[:, None] < M) & (n_offsets[None, :] < 7168)
    
    tl.store(out_ptr, acc.to(tl.bfloat16), mask=mask_c)


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(C.device)
    
    M, K = A.shape
    N, _ = B.shape
    
    assert K == 5120
    assert N == 7168
    
    num_pid_m = triton.cdiv(M, 256)
    num_pid_n = triton.cdiv(N, 64)
    num_programs = num_pid_m * num_pid_n
    
    grid = (num_programs,)
    gemm_kernel[grid](
        A, B, C,
        M, N, K,
        5120, 1, 5120, 1, 7168, 1,
        BLOCK_M=256, BLOCK_N=64, BLOCK_K_inner=256, NUM_K_TILES=10, GROUP_M=8,
        num_warps=4, num_stages=1, maxnreg=128
    )