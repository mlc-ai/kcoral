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
    Utilizes a standard 2D execution grid where each SM resolves precisely one output 
    tile to maintain high occupancy and robust L2 cache sharing.
    """
    # Explicitly map the 2D execution grid onto the logical tiling coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_k_tiles = K // BLOCK_K
    
    # Iterate over the K dimension. Staged to overlap TMA load and WGMMA instructions.
    for k_step in range(num_k_tiles):
        offset_k = k_step * BLOCK_K
        
        # Direct TMA load matching physical [Rows, K] layout
        a = a_desc.load([offset_m, offset_k])      # Shape: [BLOCK_M, BLOCK_K]
        
        # Direct TMA load matching physical [Cols, K] layout
        b = b_desc.load([offset_n, offset_k])      # Shape: [BLOCK_N, BLOCK_K]
        
        # Accumulate the matrix multiplication fragment natively via Tensor Cores
        acc = tl.dot(a, b.T, acc)
    
    # Convert the FP32 accumulation into the required bfloat16 format
    c = acc.to(tl.bfloat16)
    
    # Persist the result fragment to global memory via TMA store
    c_desc.store([offset_m, offset_n], c)


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = 7168
    K = 5120
    
    # 128x128x128 maximizes instruction density while perfectly dividing K=5120 and N=7168
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128  
    
    # Pre-construct TMA descriptors with explicit zero-padding boundaries
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
        num_warps=8, num_stages=3,
    )