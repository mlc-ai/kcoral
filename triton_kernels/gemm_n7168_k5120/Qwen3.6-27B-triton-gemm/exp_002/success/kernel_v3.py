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
):
    """Blocked GEMM: C = A @ B.T using host-side tensor descriptors for TMA."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N

    num_k_iters = tl.cdiv(K, BLOCK_K)
    for k in range(0, num_k_iters):
        offset_k = k * BLOCK_K

        # Load A tile [BLOCK_M, BLOCK_K]
        a_tile = A_desc.load([offset_m, offset_k])
        
        # Load B tile [BLOCK_N, BLOCK_K], then transpose to [BLOCK_K, BLOCK_N]
        b_tile = B_desc.load([offset_n, offset_k])

        acc = tl.dot(a_tile, b_tile.T, acc)

    # Store result
    C_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """
    Compute C = A @ B.T into preallocated output C.
    
    A: [M, K] bfloat16
    B: [N, K] bfloat16
    C: [M, N] bfloat16 (output)
    """
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = B.shape[1]

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    # Create host-side tensor descriptors for TMA access
    # A is [M, K] row-major
    A_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    
    # B is [N, K] row-major
    B_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    
    # C is [M, N] row-major
    C_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    _gemm_kernel[grid](
        A_desc,
        B_desc,
        C_desc,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=4,
    )