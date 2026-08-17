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
    NUM_SMS: tl.constexpr,
):
    """
    Persistent-scheduled tiled GEMM using Host Tensor Descriptors.
    Computes C = A @ B.T where A is [M, K], B is [N, K], and C is [M, N].
    Uses double-buffered K iterations to maximize dot density.
    """
    # Setup persistent execution boundaries
    num_pid_m = M // BLOCK_M
    num_pid_n = N // BLOCK_N
    num_tiles = num_pid_m * num_pid_n
    
    start_pid = tl.program_id(0)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    # Iterate across the complete tiling domain strictly ordered to boost L2 caching
    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        num_k_steps = K // BLOCK_K
        for k_step in range(num_k_steps):
            offset_k = k_step * BLOCK_K
            
            # Load contiguous blocks directly into locally tracked layouts
            a = a_desc.load([offset_m, offset_k])      # Shape: [BLOCK_M, BLOCK_K]
            b = b_desc.load([offset_n, offset_k])      # Shape: [BLOCK_N, BLOCK_K]
            
            # Accumulate the dot product. 
            # b.T efficiently transforms layout [BLOCK_N, BLOCK_K] -> [BLOCK_K, BLOCK_N]
            acc = tl.dot(a, b.T, acc)
        
        # Convert accumulated FP32 output to bfloat16 and persist via TMA store
        c = acc.to(tl.bfloat16)
        c_desc.store([offset_m, offset_n], c)


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = 7168
    K = 5120
    
    # Target large contiguous blocks enabling massive WGMMA instructions
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 512  
    
    # Configure TMA descriptors pointing to physically contiguous tensor blocks
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    # Schedule a persistent cluster sized perfectly for the Hopper SM count
    NUM_SMS = 132
    grid = (min(NUM_SMS, num_tiles),)
    
    _gemm_kernel[grid](
        a_desc, b_desc, c_desc,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_SMS=NUM_SMS,
        num_warps=8, num_stages=2,
    )