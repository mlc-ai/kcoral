import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A, B, C,
    M, N, K,
    stride_a_row, stride_b_row, stride_c_row,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Tiled GEMM kernel computing C = A @ B.T."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    num_k_tiles = K // BLOCK_K
    
    for k0 in range(num_k_tiles):
        k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        
        a_ptr = A + offs_m[:, None] * stride_a_row + k[None, :]
        a_tile = tl.load(
            a_ptr,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0
        )
        
        b_ptr = B + offs_n[:, None] * stride_b_row + k[None, :]
        b_tile = tl.load(
            b_ptr,
            mask=(offs_n[:, None] < N) & (k[None, :] < K),
            other=0.0
        )
        
        acc = tl.dot(a_tile, b_tile.T, acc)

    c_ptr = C + offs_m[:, None] * stride_c_row + offs_n[None, :]
    out = acc.to(tl.bfloat16)
    tl.store(c_ptr, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run(A, B, C):
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _gemm_kernel[grid](
        A, B, C, M, N, K, 
        stride_a_row=K, stride_b_row=K, stride_c_row=N,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        num_warps=4, num_stages=2
    )