import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    A_desc,
    B_desc,
    C_desc,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # L2-aware grouped tile mapping: programs in the same group share
    # B-tile L2 residency by iterating multiple M tiles per N position.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_steps = tl.cdiv(K, BLOCK_K)
    for k in range(num_k_steps):
        a_tile = A_desc.load([off_m, k * BLOCK_K])
        b_tile = B_desc.load([off_n, k * BLOCK_K])
        acc = tl.dot(a_tile, b_tile.T, acc=acc)

    C_desc.store([off_m, off_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute C = A @ B.T into preallocated output tensor C.

    Uses Hopper tensor descriptors (TMA) for async memory transfers,
    FP32 accumulation, and L2-aware grouped tile ordering.
    
    Tile sizing rationale for N=7168, K=5120:
      - BLOCK_K=128 → K=5120 requires exactly 40 iterations
      - BLOCK_N=128 → N=7168 requires exactly 56 tiles along N
      - BLOCK_M=128 → M divides evenly for typical QKV batch sizes
      - num_warps=8, num_stages=3 for WGMMA pipeline depth
    """
    torch.cuda.set_device(A.device)

    M, K = A.shape
    N = B.shape[0]

    # Tile configuration optimized for this specific GEMM shape
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    GROUP_M = 4

    # Build host tensor descriptors. These have correct alignment:
    #   BF16 has 2-byte element size; strides multiplied by 2 give byte strides.
    #   A: [M, K] stride [K,1] -> bytes [5120*2, 1*2] = [10240, 2] (10240%16==0) ✓
    #   B: [N, K] stride [K,1] -> bytes [5120*2, 1*2] = [10240, 2] (10240%16==0) ✓
    #   C: [M, N] stride [N,1] -> bytes [7168*2, 1*2] = [14336, 2] (14336%16==0) ✓
    A_desc = TensorDescriptor.from_tensor(A, block_shape=[BLOCK_M, BLOCK_K])
    B_desc = TensorDescriptor.from_tensor(B, block_shape=[BLOCK_N, BLOCK_K])
    C_desc = TensorDescriptor.from_tensor(C, block_shape=[BLOCK_M, BLOCK_N])

    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    grid = (num_pid_m * num_pid_n,)

    _gemm_kernel[grid](
        A_desc, B_desc, C_desc,
        M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_warps=8,
        num_stages=3,
    )