import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel_desc(
    A_desc, 
    B_desc, 
    C_desc, 
    M, 
    N, 
    K, 
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr
):
    start_tile = tl.program_id(0)
    tile_stride = tl.num_programs(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = N // BLOCK_N  
    num_tiles = num_pid_m * num_pid_n
    
    for tile_id in range(start_tile, num_tiles, tile_stride):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        num_k_steps = K // BLOCK_K
        
        a_buf0 = A_desc.load([offset_m, 0])
        a_buf1 = A_desc.load([offset_m, BLOCK_K])
        
        for k_step in range(num_k_steps):
            idx = k_step % 2
            
            if idx == 0:
                a = a_buf0
                if k_step + 2 < num_k_steps:
                    a_buf0 = A_desc.load([offset_m, (k_step + 2) * BLOCK_K])
            else:
                a = a_buf1
                if k_step + 2 < num_k_steps:
                    a_buf1 = A_desc.load([offset_m, (k_step + 2) * BLOCK_K])
            
            offset_k = k_step * BLOCK_K
            b = B_desc.load([offset_n, offset_k])
            acc = tl.dot(a, b.T, acc)
        
        C_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    block_m = 128
    block_n = 128
    block_k = 64
    
    A_desc = TensorDescriptor.from_tensor(A, [block_m, block_k])
    B_desc = TensorDescriptor.from_tensor(B, [block_n, block_k])
    C_desc = TensorDescriptor.from_tensor(C, [block_m, block_n])
    
    num_pid_m = triton.cdiv(M, block_m)
    num_pid_n = N // block_n  
    num_tiles = num_pid_m * num_pid_n
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    grid = (min(num_sms, num_tiles),)
    
    _gemm_kernel_desc[grid](
        A_desc,
        B_desc,
        C_desc,
        M,
        N,
        K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )