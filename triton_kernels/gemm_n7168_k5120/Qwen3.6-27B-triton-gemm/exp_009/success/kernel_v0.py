import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tiled GEMM: C[M,N] = A[M,K] @ B[K,N], where B is physically stored as [N,K]."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N

    # Accumulate in FP32 for numerical accuracy
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    # Iterate over K dimension in BLOCK_K-sized tiles
    num_k_steps = tl.cdiv(K, BLOCK_K)
    for step in range(num_k_steps):
        k_offset = step * BLOCK_K
        # Load tiles via TMA (Hopper); out-of-bounds elements are zero-padded
        a_tile = a_desc.load([offset_m, k_offset])       # [BLOCK_M, BLOCK_K]
        b_tile = b_desc.load([offset_n, k_offset])       # [BLOCK_N, BLOCK_K]
        # b_tile.T makes [BLOCK_K, BLOCK_N], so dot produces [BLOCK_M, BLOCK_N]
        acc = tl.dot(a_tile, b_tile.T, acc=acc)

    # Store result converted back to bfloat16; out-of-bounds writes are ignored
    c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C.

    A: [M, K] bfloat16
    B: [N, K] bfloat16  (physical layout; logical operand is B.T -> [K, N])
    C: [M, N] bfloat16  (preallocated output, destination-passing)
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N = B.shape[0]

    # Tile configuration tuned for Hopper tensor cores
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    # Create host-side tensor descriptors — lowers to TMA on Hopper
    # A desc: reads [BLOCK_M, BLOCK_K] tiles from [M, K] layout
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    # B desc: reads [BLOCK_N, BLOCK_K] tiles from [N, K] physical layout
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    # C desc: writes [BLOCK_M, BLOCK_N] tiles to [M, N] layout
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])

    # Launch grid: one program per output tile
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _gemm_kernel[grid](
        a_desc, b_desc, c_desc, M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )