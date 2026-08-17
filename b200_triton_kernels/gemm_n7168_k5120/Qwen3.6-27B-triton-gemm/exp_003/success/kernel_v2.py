import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
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
    
    Physical layouts: A[M,K], B[N,K] both row-major contiguous.
    Logical operation: C[m,n] = sum_k A[m,k] * B[n,k]
    
    We load A as [BLOCK_M, BLOCK_K] and B as [BLOCK_K, BLOCK_N] directly
    (without transpose) so tl.dot produces [BLOCK_M, BLOCK_N].
    Accumulate in fp32, convert to bf16 only on store.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        # Current K offsets
        curr_k = offs_k

        # Load A tile [BLOCK_M, BLOCK_K]: A[m, k]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + curr_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (curr_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile [BLOCK_K, BLOCK_N]: B[n, k] -- but we want layout [BLOCK_K, BLOCK_N]
        # B[n, k] uses stride_bk for k (dim 1, stride=1) and stride_bm for n (dim 0, stride=K)
        b_ptrs = B_ptr + curr_k[:, None] * stride_bk + offs_n[None, :] * stride_bm
        b_mask = (curr_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc=acc)

        # Advance K
        offs_k += BLOCK_K

    # Store result
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def run(A, B, C):
    """Compute C = A @ B.T into preallocated output C.
    
    Args:
        A: [M, K] bfloat16 input
        B: [N, K] bfloat16 input (logically transposed in operation)
        C: [M, N] bfloat16 preallocated output
    """
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