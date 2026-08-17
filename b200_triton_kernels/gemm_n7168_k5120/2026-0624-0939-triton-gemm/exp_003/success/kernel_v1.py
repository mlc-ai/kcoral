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
    Standard tiled GEMM using Host Tensor Descriptors.
    Computes C = A @ B.T where A is [M, K], B is [N, K], and C is [M, N].
    Each program handles exactly one (BLOCK_M x BLOCK_N) tile of the output.
    """
    # Identify the output tile this program instance is responsible for
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    # Iterate over the K dimension, accumulating the dot product
    num_k_tiles = K // BLOCK_K
    for k_step in range(num_k_tiles):
        offset_k = k_step * BLOCK_K
        
        # Load tiles directly using TMA-backed descriptors
        a = a_desc.load([offset_m, offset_k])      # Shape: [BLOCK_M, BLOCK_K]
        b = b_desc.load([offset_n, offset_k])      # Shape: [BLOCK_N, BLOCK_K]
        
        # Compute partial dot product. b.T converts [BLOCK_N, BLOCK_K] to [BLOCK_K, BLOCK_N]
        acc = tl.dot(a, b.T, acc)
    
    # Convert the float32 accumulator to the target bfloat16 format and store the result
    c = acc.to(tl.bfloat16)
    c_desc.store([offset_m, offset_n], c)


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = 7168
    K = 5120
    
    # Utilize standard 128x128x128 tiles to naturally align with K=5120 and N=7168
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    # Create host-side tensor descriptors enabling TMA on Hopper
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    grid = (num_pid_m, num_pid_n)
    
    _gemm_kernel[grid](
        a_desc, b_desc, c_desc,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )