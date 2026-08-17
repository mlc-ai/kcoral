import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def gemm_kernel(
    A_desc, B_desc, C_desc, M, N, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Optimized Persistent GEMM utilizing Hardware TMA descriptors and WGMMA.
    1. Allocates static shared memory and applies a swizzle-compatible flatten layout.
    2. Pipelines K dimension loads using a dual-buffer strategy.
    3. Computes C = A @ B.T chunked over the 5120 hidden state dimension.
    """
    
    # Create a contiguous block of memory for our double-buffer pipeline.
    # We flatten dimensions 1 and 2 to establish the intrinsic swizzling patterns 
    # required for efficient Hopper TMA descriptor loads.
    extern_shared_storage = tlnand.extern_static_shared_array(
        elem_ty=tl.bfloat16, shape=(2, max(BLOCK_M, BLOCK_N), BLOCK_K)
    )
    shared_storage = extern_shared_storage.flatten(1, 2)
    shared_a = shared_storage[:, :BLOCK_M * BLOCK_K, :]
    shared_b = shared_storage[:, BLOCK_M * BLOCK_K:, :]
    
    # Initialize offsets for a cyclic persistent schedule mapped optimally over Hopper SM resources
    start_pid = tl.program_id(0)
    tile_stride = tl.num_programs(0)
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    idx = 0
    next_idx = 1
    
    # Prime the pump: load the initial k=0 segment directly into buffer 0
    A_desc.load_into(shared_a[0, :, :], [0, 0])
    B_desc.load_into(shared_b[0, :, :], [0, 0])
    
    for tile_id in range(start_pid, num_tiles, tile_stride):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # Iterating linearly across chunks of the static 5120 hidden feature space
        for k in range(0, 5120, BLOCK_K):
            # Software pipelining: preload the subsequent stage's data while processing the current stage
            if k + BLOCK_K < 5120:
                A_desc.load_into(shared_a[next_idx, :, :], [offset_m, k + BLOCK_K])
                B_desc.load_into(shared_b[next_idx, :, :], [offset_n, k + BLOCK_K])
            
            tlnand.sync_threads()
            
            a = shared_a[idx, :, :]
            b = shared_b[idx, :, :]
            
            # Fundamental WGMMA reduction step accumulating over K blocks in native FP32 precision internally
            acc = tl.dot(a, b.T, acc)
            tlnand.sync_threads()
            
            idx, next_idx = next_idx, idx
        
        # Utilize native Hardware TMA capabilities to commit the final aggregated outputs directly to HBM
        C_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """
    Destination-passing wrapper for the GEMM. 
    Properly initializes TMA descriptors mapping the exact physical topologies of A, B, and C.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 256
    
    A_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    B_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    C_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    NUM_SMS = 132
    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(NUM_SMS, num_tiles),)
    
    gemm_kernel[grid](
        A_desc, B_desc, C_desc, M, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_ctas=1, num_stages=3
    )