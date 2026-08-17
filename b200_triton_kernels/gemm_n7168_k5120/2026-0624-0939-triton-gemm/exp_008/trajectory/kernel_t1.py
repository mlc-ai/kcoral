import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel_persistent(
    a_desc, 
    b_desc, 
    c_desc, 
    M, 
    N, 
    K,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr
):
    start_pid = tl.program_id(0)
    tile_stride = tl.num_programs(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    for tile_id in range(start_pid, num_tiles, tile_stride):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        num_k_tiles = tl.cdiv(K, BLOCK_K)
        for k_idx in range(num_k_tiles):
            offset_k = k_idx * BLOCK_K
            
            a_tile = a_desc.load([offset_m, offset_k])
            b_tile = b_desc.load([offset_k, offset_n])
            
            acc = tl.dot(a_tile, b_tile, acc)
        
        c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    K = A.shape[1]
    N = C.shape[1]
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 256
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    B_t = B.T.contiguous()
    b_desc = TensorDescriptor.from_tensor(B_t, [BLOCK_K, BLOCK_N])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count
    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(NUM_SMS, num_tiles),)
    
    _gemm_kernel_persistent[grid](
        a_desc, b_desc, c_desc, M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_SMS=NUM_SMS,
        num_warps=4, num_stages=3
    )