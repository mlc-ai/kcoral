import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A, B, C, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_iters = K // BLOCK_K
    for k_idx in range(num_k_iters):
        k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        
        a_ptr = A + offs_m[:, None] * K + k[None, :]
        a_tile = tl.load(a_ptr, mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        
        b_ptr = B + offs_n[:, None] * K + k[None, :]
        b_tile = tl.load(b_ptr, mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0.0)
        
        acc = tl.dot(a_tile, b_tile.T, acc)

    c_ptr = C + offs_m[:, None] * N + offs_n[None, :]
    out = acc.to(tl.bfloat16)
    tl.store(c_ptr, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run(A, B, C):
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    K = A.shape[1]
    N = C.shape[1]
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    _gemm_kernel[grid](
        A, B, C, M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2
    )