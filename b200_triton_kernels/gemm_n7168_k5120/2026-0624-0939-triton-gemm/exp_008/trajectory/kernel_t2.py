import torch
import triton
import triton.language as tl


def set_allocator():
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)

set_allocator()


@triton.autotune(
    configs=[
        triton.Config({"WARP_SPECIALIZE": True, "acc_to_bf16": False}, num_warps=8, num_stages=4),
        triton.Config({"WARP_SPECIALIZE": False, "acc_to_bf16": False}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_kernel_persistent(
    a_ptr, 
    b_ptr, 
    c_ptr, 
    M, 
    N, 
    K,
    NUM_SMS,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr, 
    WARP_SPECIALIZE: tl.constexpr,
    acc_to_bf16: tl.constexpr,
):
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero")
    
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K], padding_option="zero")
        
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N])

    start_pid = tl.program_id(0)
    tile_stride = tl.num_programs(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    for tile_id in tl.range(start_pid, num_tiles, tile_stride, flatten=False, warp_specialize=WARP_SPECIALIZE):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        num_k_steps = tl.cdiv(K, BLOCK_K)
        for k_step in range(num_k_steps):
            offset_k = k_step * BLOCK_K
            
            a_tile = a_desc.load([offset_m, offset_k])
            b_tile = b_desc.load([offset_n, offset_k])
            
            if acc_to_bf16:
                acc = acc.to(tl.bfloat16)
                
            acc = tl.dot(a_tile, b_tile.T, acc)
            
            if acc_to_bf16:
                acc = acc.to(tl.float32)
        
        c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    K = A.shape[1]
    N = C.shape[1]
    
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 128
    
    NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count
    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(NUM_SMS, num_tiles),)
    
    _gemm_kernel_persistent[grid](
        A, B, C, M, N, K, NUM_SMS,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )