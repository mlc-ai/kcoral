import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    a_desc,
    b_desc,
    temp_buf_ptr,
    num_m_tiles: tl.constexpr,
    num_n_tiles: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 3D Grid maps innermost dimensions first: n_tile varies fastest, then m_tile, then k_tile
    k_tile = tl.program_id(0)
    m_tile = tl.program_id(1)
    n_tile = tl.program_id(2)
    
    offset_k = k_tile * 1024 # Using constexpr 1024 for shape matching
    
    a = a_desc.load([m_tile * BLOCK_M, offset_k])
    b = b_desc.load([n_tile * BLOCK_N, offset_k])
    
    acc = tl.dot(a, b.T)
    
    offset = k_tile * (num_m_tiles * num_n_tiles * BLOCK_M * BLOCK_N) + \
             m_tile * (num_n_tiles * BLOCK_M * BLOCK_N) + \
             n_tile * (BLOCK_M * BLOCK_N)
             
    out_ptr = temp_buf_ptr + offset
    tl.store(out_ptr + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :], acc)


@triton.jit
def _accumulate_kernel(
    temp_buf_ptr,
    c_ptr,
    M,
    N,
    num_k_tiles: tl.constexpr,
    num_n_tiles: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
):
    m_tile = tl.program_id(0)
    n_tile = tl.program_id(1)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_tile in range(num_k_tiles):
        offset = k_tile * (num_k_tiles * num_n_tiles * BLOCK_M * BLOCK_N) + \
                 m_tile * (num_n_tiles * BLOCK_M * BLOCK_N) + \
                 n_tile * (BLOCK_M * BLOCK_N)
        
        in_ptr = temp_buf_ptr + offset
        tile = tl.load(in_ptr + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :])
        acc += tile
        
    c_val = acc.to(tl.bfloat16)
    
    tile_row_idx = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    tile_col_idx = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask = (tile_row_idx[:, None] < M) & (tile_col_idx[None, :] < N)
    tl.store(c_ptr + tile_row_idx[:, None] * stride_cm + tile_col_idx[None, :] * stride_cn, c_val, mask=mask)


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    with torch.no_grad():
        torch.cuda.set_device(A.device)
        
        if A.numel() == 0:
            return

        M, K = A.shape
        N, K_b = B.shape
        assert K == K_b, f"K dimension mismatch: {K} vs {K_b}"
        
        BLOCK_M = 256
        BLOCK_N = 512
        BLOCK_K = 1024
        
        num_k_tiles = triton.cdiv(K, BLOCK_K)
        num_m_tiles = triton.cdiv(M, BLOCK_M)
        num_n_tiles = triton.cdiv(N, BLOCK_N)
        
        temp_buf = torch.empty((num_k_tiles, num_m_tiles, num_n_tiles, BLOCK_M, BLOCK_N), 
                                device=A.device, dtype=torch.float32)
        
        a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
        b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
        
        grid_gemm = (num_k_tiles, num_m_tiles, num_n_tiles)
        _gemm_kernel[grid_gemm](
            a_desc, b_desc, temp_buf,
            num_m_tiles, num_n_tiles,
            BLOCK_M, BLOCK_N,
            num_warps=8,
            num_stages=2,
        )
        
        grid_accum = (num_m_tiles, num_n_tiles)
        _accumulate_kernel[grid_accum](
            temp_buf, C, M, N,
            num_k_tiles, num_n_tiles,
            BLOCK_M, BLOCK_N,
            stride_cm=N, stride_cn=1,
            num_warps=8,
            num_stages=2,
        )