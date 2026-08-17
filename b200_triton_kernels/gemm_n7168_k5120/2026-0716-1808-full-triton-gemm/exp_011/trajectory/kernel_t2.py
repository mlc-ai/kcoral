import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def gemm_kernel(A_desc, B_desc, C_desc, M, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    extern_shared_storage = tlnand.extern_static_shared_array(
        elem_ty=tl.bfloat16, shape=(2, max(BLOCK_M, BLOCK_N), BLOCK_K)
    )
    shared_storage = extern_shared_storage.reshape((2, BLOCK_M, BLOCK_N, BLOCK_K))
    shared_a = shared_storage[:, 0, :, :]
    shared_b = shared_storage[:, 1, :, :]
    
    start_pid = tl.program_id(0)
    tile_stride = tl.num_programs(0)
    
    num_pid_m = M // BLOCK_M
    num_pid_n = 7168 // BLOCK_N
    num_tiles = num_pid_m * num_pid_n
    
    for tile_id in range(start_pid, num_tiles, tile_stride):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        idx = 0
        next_idx = 1
        
        A_desc.load_into(shared_a[next_idx, :, :], [offset_m, 0])
        B_desc.load_into(shared_b[next_idx, :, :], [offset_n, 0])
        
        for k in range(0, 5120, BLOCK_K):
            if k + BLOCK_K < 5120:
                A_desc.load_into(shared_a[next_idx, :, :], [offset_m, k + BLOCK_K])
                B_desc.load_into(shared_b[next_idx, :, :], [offset_n, k + BLOCK_K])
            
            tlnand.sync_threads()
            
            a = shared_a[idx, :, :]
            b = shared_b[idx, :, :]
            
            acc = tl.dot(a, b.T, acc)
            tlnand.sync_threads()
            
            idx, next_idx = next_idx, idx
        
        C_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 128
    
    A_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    B_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    C_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    NUM_SMS = 132
    num_tiles = (M // BLOCK_M) * (7168 // BLOCK_N)
    grid = (min(NUM_SMS, num_tiles),)
    
    gemm_kernel[grid](
        A_desc, B_desc, C_desc, M,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_ctas=1, num_stages=2
    )