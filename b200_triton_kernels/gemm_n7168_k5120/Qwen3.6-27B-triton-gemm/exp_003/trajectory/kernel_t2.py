import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
                      num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64},
                      num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
                      num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128},
                      num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128},
                      num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128},
                      num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
                      num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
                      num_warps=8, num_stages=4),
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
    
    Both A and B are stored row-major. B is logically transposed: we compute
    C[m,n] = sum_k A[m,k] * B[n,k]. So B tile layout in dot is [K, BLOCK_N].
    Accumulate in fp32, convert to bf16 only at store.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets into matrices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for loading A tiles [BLOCK_M, BLOCK_K]
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # Pointers for loading B tiles [BLOCK_N, BLOCK_K] 
    b_ptrs = B_ptr + offs_n[:, None] * stride_bm + offs_k[None, :] * stride_bk

    # Masks for boundary checking
    m_mask = (offs_m[:, None] < M)
    n_mask = (offs_n[:, None] < N) & (offs_n[:, None] < N)
    k_mask = offs_k[None, :] < K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Number of K tiles - K=5120, so with BLOCK_K=64 → 80 steps, BLOCK_K=128 → 40 steps
    num_k_tiles = tl.cdiv(K, BLOCK_K)

    for k_idx in range(num_k_tiles):
        k_offset = k_idx * BLOCK_K + offs_k
        actual_k_mask = k_offset < K

        # Load A tile [BLOCK_M, BLOCK_K]
        a_tile = tl.load(
            A_ptr + offs_m[:, None] * stride_am + k_offset[None, :] * stride_ak,
            mask=m_mask & actual_k_mask[None, :],
            other=0.0,
        )
        # Load B tile [BLOCK_N, BLOCK_K], then transpose for dot → [BLOCK_K, BLOCK_N]
        b_tile = tl.load(
            B_ptr + k_offset[:, None] * stride_bk + offs_n[None, :] * stride_bm,
            mask=actual_k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a_tile, b_tile, acc=acc)

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

    grid = (
        triton.cdiv(M, 128),
        triton.cdiv(N, 128),
    )

    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),   # stride_am, stride_ak
        B.stride(0), B.stride(1),   # stride_bm, stride_bk
        C.stride(0), C.stride(1),   # stride_cm, stride_cn
    )