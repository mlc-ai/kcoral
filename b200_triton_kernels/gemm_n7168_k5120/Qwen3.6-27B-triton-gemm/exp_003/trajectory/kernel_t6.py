import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Larger tiles reduce launch overhead per element computed
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bm,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tiled GEMM: C[M,N] = A[M,K] @ B[N,K].T
    
    Both A and B row-major contiguous. B logically transposed.
    Load A as [BLOCK_M, BLOCK_K], load B as [BLOCK_K, BLOCK_N].
    Accumulate in fp32, convert to bf16 at store.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Prefetch A pointers (will be reused across all K iterations)
    A_row_ptrs = A_ptr + offs_m[:, None] * stride_am

    # We need to iterate over K dimension in blocks
    # For efficient loading of B in columnar fashion, use block K increments
    for k_block_start in range(0, tl.cdiv(K, BLOCK_K)):
        curr_k = k_block_start * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = curr_k < K

        # Load A tile [BLOCK_M, BLOCK_K]: row-contiguous layout
        a = tl.load(
            A_row_ptrs + curr_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )

        # Load B tile [BLOCK_K, BLOCK_N]: B stored row-major [N, K]
        # Access pattern: B[k, n] -- column-wise access for each k, row-wise for n
        b = tl.load(
            B_ptr + curr_k[:, None] * stride_bk + offs_n[None, :] * stride_bm,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0,
        )

        acc = tl.dot(a, b, acc=acc)

    # Store result
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def run(A, B, C):
    """Compute C = A @ B.T into preallocated output C."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )