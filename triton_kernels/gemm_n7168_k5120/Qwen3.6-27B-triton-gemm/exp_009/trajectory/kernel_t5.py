import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


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
    Optimized GEMM kernel for Hopper: C[M,N] = A[M,K] @ B[N,K].T
    
    Uses standard tiled approach with FP32 accumulation and BF16 I/O.
    B is physically stored [N, K]; we transpose in the dot call.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute global row/column offsets for this program's tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Build base pointers for this tile
    a_ptrs = A + offs_m[:, None] * stride_am   # [BLOCK_M, 1]
    b_ptrs = B + offs_n[None, :] * stride_bn    # [1, BLOCK_N]

    # Output pointer array
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    # Boundary masks
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Pointer hints for load vectorization
    offs_k = tl.arange(0, BLOCK_K)

    num_k_steps = tl.cdiv(K, BLOCK_K)
    for k_idx in range(num_k_steps):
        k_base = k_idx * BLOCK_K
        k_offs = k_base + offs_k

        # A tile pointers: [BLOCK_M, BLOCK_K]
        a_tile_ptrs = a_ptrs + k_offs[None, :] * stride_ak
        a_mask = mask_m & (k_offs[None, :] < K)
        a_tile = tl.load(a_tile_ptrs, mask=a_mask, other=0.0)

        # B tile pointers: [1, BLOCK_N] * [BLOCK_K, 1] -> [BLOCK_K, BLOCK_N] after broadcast
        # But we need [BLOCK_N, BLOCK_K] to match physical layout, then .T in dot
        b_tile_ptrs = k_offs[:, None] * stride_bk + b_ptrs.T.T
        # Simpler: construct [BLOCK_N, BLOCK_K] directly
        b_tile_ptrs = b_ptrs.T.T + k_offs[:, None] * stride_bk
        # Actually the cleanest way:
        # b_ptrs shape [1, BLOCK_N], k_offs[:, None] shape [BLOCK_K, 1]
        # b_tile_ptrs = k_offs[:, None] * stride_bk + b_ptrs.broadcast([BLOCK_K, BLOCK_N])

        # Let me do this explicitly and correctly
        b_tile_ptrs = (
            k_offs[:, None] * stride_bk          # [BLOCK_K, 1]
            + offs_n[None, :] * stride_bn         # broadcast to [1, BLOCK_N] -> [BLOCK_K, BLOCK_N]
        )
        # This gives us [BLOCK_K, BLOCK_N] indices into B[N,K]
        # But B strides are [stride_bn(N dim), stride_bk(K dim)]
        # So ptr = n*stride_bn + k*stride_bk is correct for B[n,k]
        b_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        b_tile = tl.load(b_tile_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Now dot: a_tile[BLOCK_M, BLOCK_K] x b_tile[BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc = tl.dot(a_tile, b_tile, acc=acc)

    # Store result
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_m & mask_n)


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C.
    
    A: [M, K=5120] bfloat16
    B: [N=7168, K=5120] bfloat16 (physical layout)
    C: [M, N=7168] bfloat16 (preallocated destination)
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N = B.shape[0]

    # Aggressive tiling for Hopper throughput
    # BLOCK_K=128 -> only 40 iterations of inner loop
    # Larger tiles amortize address computation overhead
    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 128

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