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
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute C[M,N] = A[M,K] @ B[N,K].T into C using tiled matrix multiplication.
    
    Memory layout:
    - A is row-major [M, K]
    - B is physically stored as [N, K]; we need B.T -> [K, N] logically
    - C is row-major [M, N]
    
    Each program computes a BLOCK_M x BLOCK_N tile of C by iterating over BLOCK_K tiles of K.
    Accumulation is done in FP32; final store converts to BF16.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Base offsets for this program's output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Precompute pointer bases (in element units)
    a_ptrs_base = A + offs_m[:, None] * stride_am   # [BLOCK_M, 1]
    b_ptrs_base = B + offs_n[None, :] * stride_bn    # [1, BLOCK_N]
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    # Output mask: ensures we don't write out-of-bounds
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    mask_out = mask_m & mask_n

    # Accumulate in FP32 for numerical accuracy
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    # Iterate over K dimension
    for k_step in range(tl.cdiv(K, BLOCK_K)):
        k_offsets = k_step * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k_a = k_offsets[None, :] < K
        mask_k_b = k_offsets[:, None] < K

        # Construct full pointers: A[BLOCK_M, BLOCK_K]
        a_ptrs = a_ptrs_base + k_offsets[None, :] * stride_ak
        a_tile = tl.load(a_ptrs, mask=mask_m & mask_k_a, other=0.0)

        # Construct full pointers: B[BLOCK_N, BLOCK_K]
        # B is [N, K] row-major, so index is [n, k]
        b_ptrs = b_ptrs_base + k_offsets[:, None] * stride_bk
        b_tile = tl.load(b_ptrs, mask=mask_k_b & mask_n, other=0.0)

        # tl.dot(a_tile [BLOCK_M, BLOCK_K], b_tile.T [BLOCK_K, BLOCK_N]) -> [BLOCK_M, BLOCK_N]
        acc = tl.dot(a_tile, b_tile.T, acc)

    # Convert to BF16 and store
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_out)


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C.
    
    A: [M, K=5120] bfloat16
    B: [N=7168, K=5120] bfloat16
    C: [M, N=7168] bfloat16 (preallocated destination)
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N = B.shape[0]

    # Tuned tile configuration for Hopper (SM90/SM90a)
    # - Large BLOCK_M/BLOCK_N maximizes FLOPs per tile and amortizes overhead
    # - BLOCK_K=128 balances register pressure vs loop iterations (5120/128=40 iters)
    # - 8 warps saturates WGMMA units on Hopper
    # - num_stages=3 hides memory latency with software pipelining
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128

    # Grid: one program per output tile
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=3,
    )