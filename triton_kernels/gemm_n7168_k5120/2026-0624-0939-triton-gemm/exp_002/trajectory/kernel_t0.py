import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    for k_offset in range(0, K, BLOCK_K):
        # Load A tile [BLOCK_M, BLOCK_K]
        a_ptr = A + (m_start + offs_m)[:, None] * stride_am + (k_offset + offs_k)[None, :] * stride_ak
        a_tile = tl.load(a_ptr, mask=(m_start + offs_m)[:, None] < M, other=0.0)
        
        # Load B tile [BLOCK_N, BLOCK_K] -> transpose to get B^T [BLOCK_K, BLOCK_N]
        b_ptr = B + (n_start + offs_n)[:, None] * stride_bk + (k_offset + offs_k)[None, :] * stride_bn
        b_tile = tl.load(b_ptr, mask=(n_start + offs_n)[:, None] < N, other=0.0)
        
        acc = tl.dot(a_tile, b_tile.T, acc)
        
    # Convert to output dtype and store C tile [BLOCK_M, BLOCK_N]
    c_ptr = C + (m_start + offs_m)[:, None] * stride_cm + (n_start + offs_n)[None, :] * stride_cn
    out = acc.to(tl.bfloat16)
    tl.store(c_ptr, out, mask=(m_start + offs_m)[:, None] < M)


def run(A, B, C):
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bk, stride_bn = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    grid = (num_pid_m, num_pid_n)
    
    _gemm_kernel[grid](
        A, B, C, M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )