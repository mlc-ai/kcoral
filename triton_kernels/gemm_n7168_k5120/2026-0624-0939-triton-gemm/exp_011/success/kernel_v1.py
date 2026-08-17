import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel_desc(
    A_desc, 
    B_left_desc, 
    B_right_desc, 
    C_left_desc, 
    C_right_desc, 
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
        
        acc_left = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        acc_right = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        num_k = K // 64
        for k in range(num_k):
            offset_k = k * 64
            
            a = A_desc.load([offset_m, offset_k])
            b_l = B_left_desc.load([offset_n, offset_k])
            b_r = B_right_desc.load([offset_n, offset_k])
            
            acc_left = tl.dot(a, b_l.T, acc_left)
            acc_right = tl.dot(a, b_r.T, acc_right)
        
        C_left_desc.store([offset_m, offset_n], acc_left.to(tl.bfloat16))
        C_right_desc.store([offset_m, offset_n], acc_right.to(tl.bfloat16))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    block_m = 64
    block_n = 64
    block_k = 64
    
    B_left = B.narrow(0, 0, N // 2)
    B_right = B.narrow(0, N // 2, N // 2)
    C_left = C.narrow(1, 0, N // 2)
    C_right = C.narrow(1, N // 2, N // 2)
    
    A_desc = TensorDescriptor.from_tensor(A, [block_m, block_k])
    B_left_desc = TensorDescriptor.from_tensor(B_left, [block_n, block_k])
    B_right_desc = TensorDescriptor.from_tensor(B_right, [block_n, block_k])
    C_left_desc = TensorDescriptor.from_tensor(C_left, [block_m, block_n])
    C_right_desc = TensorDescriptor.from_tensor(C_right, [block_m, block_n])
    
    num_pid_m = triton.cdiv(M, block_m)
    num_pid_n = (N // 2) // block_n  
    num_tiles = num_pid_m * num_pid_n
    num_sms = 132
    grid = (min(num_sms, num_tiles),)
    
    _gemm_kernel_desc[grid](
        A_desc,
        B_left_desc,
        B_right_desc,
        C_left_desc,
        C_right_desc,
        M,
        N,
        K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=16,
        num_stages=4,
    )