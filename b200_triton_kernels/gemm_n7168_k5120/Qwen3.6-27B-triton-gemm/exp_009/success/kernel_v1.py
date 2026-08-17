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
    """
    Compute C[M,N] = A[M,K] @ B[K,N], where B is physically stored as [N,K].
    
    Uses Hopper tensor descriptors for TMA-backed loads/stores.
    Accumulation is done in FP32; final store converts to BF16.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets in the output space
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N

    # Accumulate in FP32 for numerical accuracy
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    # Iterate over K dimension in BLOCK_K-sized tiles
    num_k_steps = tl.cdiv(K, BLOCK_K)
    for k_idx in range(num_k_steps):
        k_offset = k_idx * BLOCK_K
        # Load A tile: [BLOCK_M, BLOCK_K] from physical [M, K] layout
        a_tile = a_desc.load([offset_m, k_offset])
        # Load B tile: [BLOCK_N, BLOCK_K] from physical [N, K] layout
        b_tile = b_desc.load([offset_n, k_offset])
        # b_tile.T -> [BLOCK_K, BLOCK_N]; dot yields [BLOCK_M, BLOCK_N]
        acc = tl.dot(a_tile, b_tile.T, acc=acc)

    # Store converted to BF16; descriptor ignores OOB writes
    c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C.
    
    A: [M, K=5120] bfloat16
    B: [N=7168, K=5120] bfloat16 (physical storage layout)
    C: [M, N=7168] bfloat16 (preallocated destination)
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N = B.shape[0]

    # Large tiles for maximum throughput on Hopper
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128

    # Host-side tensor descriptors lower to TMA on Hopper
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])

    # Grid: one program per output tile
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _gemm_kernel[grid](
        a_desc, b_desc, c_desc,
        M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=3,
    )