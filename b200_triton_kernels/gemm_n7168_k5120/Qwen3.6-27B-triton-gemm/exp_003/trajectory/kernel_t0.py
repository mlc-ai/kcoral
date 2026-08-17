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
    """Tiled GEMM kernel: C[M,N] = A[M,K] @ B[K,N], using Hopper TMA descriptors.
    
    Physical layout: B is stored [N, K]; we load [BLOCK_N, BLOCK_K] and .T for dot.
    Accumulates in fp32, converts to bf16 only on store.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        offset_k = k_tile * BLOCK_K
        # Load [BLOCK_M, BLOCK_K] tile of A and [BLOCK_N, BLOCK_K] tile of B
        # Descriptor padding supplies zeros for out-of-bounds regions
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        # b is [BLOCK_N, BLOCK_K], transpose to [BLOCK_K, BLOCK_N] for dot
        acc = tl.dot(a, b.T, acc=acc)

    # Store result; descriptor ignores out-of-bounds elements automatically
    c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C.
    
    Args:
        A: [M, K] bfloat16 input matrix
        B: [N, K] bfloat16 input matrix (logically transposed in the operation)
        C: [M, N] bfloat16 preallocated output tensor
    """
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]

    # Tile sizes chosen for Hopper WGMMA compatibility:
    #   BLOCK_M=64, BLOCK_N=64 fit a comfortable fp32 accumulator in registers
    #   BLOCK_K=64 divides both N=7168 and K=5120 evenly (no partial K tiles)
    #   num_warps=8 enables warp-group based Hopper tensor core path
    #   num_stages=3 pipelines descriptor loads across the K reduction loop
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    # Build tensor descriptors for TMA-backed memory access on Hopper
    a_desc = TensorDescriptor.from_tensor(A, block_shape=[BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, block_shape=[BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, block_shape=[BLOCK_M, BLOCK_N])

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _gemm_kernel[grid](
        a_desc, b_desc, c_desc,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=3,
    )