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
    N_out,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Implements C = A @ B^T using Hopper TMA descriptors and persistent scheduling.
    Accumulates in FP32 and converts to BF16 at the store phase.
    """
    start_pid = tl.program_id(0)
    tile_stride = tl.num_programs(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N_out, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    for tile_id in range(start_pid, num_tiles, tile_stride):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            acc = tl.dot(a, b.T, acc)
            
        if offset_m + BLOCK_M <= M and offset_n + BLOCK_N <= N_out:
            c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into the preallocated tensor C."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N_out = B.shape[0]
    K = B.shape[1]
    
    block_m = 128
    block_n = 1024
    block_k = 1024
    
    a_desc = TensorDescriptor.from_tensor(A, [block_m, block_k])
    b_desc = TensorDescriptor.from_tensor(B, [block_n, block_k])
    c_desc = TensorDescriptor.from_tensor(C, [block_m, block_n])
    
    num_pid_m = triton.cdiv(M, block_m)
    num_pid_n = triton.cdiv(N_out, block_n)
    num_tiles = num_pid_m * num_pid_n
    
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    grid = (min(num_sms, num_tiles),)
    
    _gemm_kernel[grid](
        a_desc, 
        b_desc, 
        c_desc, 
        M, N_out, K,
        BLOCK_M=block_m, 
        BLOCK_N=block_n, 
        BLOCK_K=block_k,
        num_warps=8, 
        num_stages=2
    )