import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def gemm_kernel(
    A_desc, B_desc, C_desc, M, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Optimized Persistent GEMM utilizing Hardware TMA descriptors and WGMMA.
    1. Uses a single dynamic dimension for PITCH linear mapping across the static 5120 hidden states.
    2. Pipelines K dimension loads using a hardware TMA intrinsic pipeline.
    3. Computes C = A @ B.T chunked effectively over the underlying feature space.
    """
    
    # Initialize offsets for a cyclic persistent schedule mapped optimally over Hopper SM resources
    start_pid = tl.program_id(0)
    tile_stride = tl.num_programs(0)
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(7168, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    for tile_id in range(start_pid, num_tiles, tile_stride):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # Iterating linearly and unrolled across chunks of the static 5120 hidden feature space
        for k in tl.static_range(0, 40, 1):
            a = A_desc.load([offset_m, k * BLOCK_K])
            b = B_desc.load([offset_n, k * BLOCK_K])
            
            # Fundamental WGMMA reduction step accumulating over K blocks in native FP32 precision internally
            acc = tl.dot(a, b.T, acc)
        
        # Utilize native Hardware TMA capabilities to commit the final aggregated outputs directly to HBM
        C_desc.store([offset_m, offset_n], acc.to(tl.bfloat16), boundary_check=(M,))


def run(A, B, C):
    """
    Destination-passing wrapper for the GEMM. 
    Properly initializes TMA descriptors mapping the exact physical topologies of A, B, and C.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 1024, 128
    
    A_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    B_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    C_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    NUM_SMS = 132
    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(7168, BLOCK_N)
    grid = (min(NUM_SMS, num_tiles),)
    
    gemm_kernel[grid](
        A_desc, B_desc, C_desc, M,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_ctas=1, num_stages=3
    )