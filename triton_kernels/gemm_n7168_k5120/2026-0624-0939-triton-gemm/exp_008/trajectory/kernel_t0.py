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
    BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_idx in range(0, K, BLOCK_K):
        k = k_idx + tl.arange(0, BLOCK_K)
        
        a_ptr = A + offs_m[:, None] * stride_am + k[None, :] * stride_ak
        a_tile = tl.load(a_ptr, mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        
        b_ptr = B + offs_n[:, None] * stride_bn + k[None, :] * stride_bk
        b_tile = tl.load(b_ptr, mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0.0)
        
        acc = tl.dot(a_tile, b_tile.T, acc)

    c_ptr = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptr, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run(A, B, C):
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    K = A.shape[1]
    N = C.shape[1]
    
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    
    _gemm_kernel[grid](
        A, 
        B, 
        C, 
        M, 
        N, 
        K,
        stride_am=K, 
        stride_ak=1,
        stride_bk=1, 
        stride_bn=K,
        stride_cm=N, 
        stride_cn=1,
        BLOCK_M=128, 
        BLOCK_N=128, 
        BLOCK_K=64,
        num_warps=4, 
        num_stages=3
    )