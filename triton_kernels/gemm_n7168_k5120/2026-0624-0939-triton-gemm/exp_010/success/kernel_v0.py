import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m_base = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    for k_step in range(tl.cdiv(K, BLOCK_K)):
        k_idx = k_step * BLOCK_K + tl.arange(0, BLOCK_K)
        
        a_ptr = A + offs_m_base[:, None] * stride_am + k_idx[None, :] * stride_ak
        a_tile = tl.load(a_ptr, mask=(offs_m_base[:, None] < M) & (k_idx[None, :] < K), other=0.0)
        
        b_ptr = B + offs_n_base[:, None] * stride_bk + k_idx[None, :] * stride_bn
        b_tile = tl.load(b_ptr, mask=(offs_n_base[:, None] < N) & (k_idx[None, :] < K), other=0.0)
        
        acc = tl.dot(a_tile, b_tile.T, acc)
    
    out_ptr = C + offs_m_base[:, None] * stride_cm + offs_n_base[None, :] * stride_cn
    tl.store(out_ptr, acc.to(tl.bfloat16), mask=offs_m_base[:, None] < M)


def run(A, B, C):
    """Compute C = A @ B.T into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    
    M, K = A.shape
    N = B.shape[0]
    
    assert K == 5120
    assert N == 7168
    
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    _gemm_kernel[grid](
        A, B, C, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )