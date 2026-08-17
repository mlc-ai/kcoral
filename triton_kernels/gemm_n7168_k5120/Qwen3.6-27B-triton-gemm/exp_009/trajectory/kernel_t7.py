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
    Tiled GEMM using host-side tensor descriptors on Hopper.
    
    Computes C[M,N] = A[M,K] @ B[N,K].T into preallocated C.
    Accumulation uses FP32; output is cast to BF16.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    num_k_steps = tl.cdiv(K, BLOCK_K)
    for ki in range(num_k_steps):
        offset_k = ki * BLOCK_K
        # Load [BLOCK_M, BLOCK_K] from A's physical [M, K] layout
        a_tile = a_desc.load([offset_m, offset_k])
        # Load [BLOCK_N, BLOCK_K] from B's physical [N, K] layout
        b_tile = b_desc.load([offset_n, offset_k])
        # dot(a[BLOCK_M,BLOCK_K], b.T[BLOCK_K,BLOCK_N]) -> [BLOCK_M,BLOCK_N]
        acc = tl.dot(a_tile, b_tile.T, acc)

    c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C.
    
    A: [M, K=5120] bfloat16
    B: [N=7168, K=5120] bfloat16 (physical layout)
    C: [M, N=7168] bfloat16 (preallocated destination)
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N = B.shape[0]

    # Tile configuration optimized for Hopper throughput:
    # - BLOCK_M=128 gives good register utilization per program
    # - BLOCK_N=256 doubles output coverage per program while keeping WGMMA happy
    # - BLOCK_K=128 means only 40 iterations (5120/128=40), reducing loop overhead
    # - 8 warps saturates WGMMA units on SM90
    # - 3 stages hides memory latency
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 128

    # Create host-side tensor descriptors — lower to TMA loads/stores on Hopper
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