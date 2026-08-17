import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_tile_coordinates(
    tile_id,
    num_pid_m,
    num_pid_n,
    GROUP_M: tl.constexpr,
):
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    return pid_m, pid_n


@triton.jit
def _gemm_kernel(
    A, B, C,
    M, N, K,
    NUM_SMS,
    GROUP_M: tl.constexpr,
):
    a_desc = tl.make_tensor_descriptor(
        A, shape=[M, K], strides=[K, 1], 
        block_shape=[64, 64], padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        B, shape=[N, K], strides=[K, 1], 
        block_shape=[64, 64], padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        C, shape=[M, N], strides=[N, 1], 
        block_shape=[64, 64], padding_option="zero"
    )

    start_pid = tl.program_id(0)
    num_pid_m = cdiv(M, 64)
    num_pid_n = cdiv(N, 64)
    num_tiles = num_pid_m * num_pid_n
    
    if start_pid >= num_tiles:
        return
        
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False, warp_specialize=True):
        if tile_id >= num_tiles:
            return
            
        pid_m, pid_n = _grouped_tile_coordinates(
            tile_id, num_pid_m, num_pid_n, GROUP_M
        )
        
        offset_m = pid_m * 64
        offset_n = pid_n * 64
        
        acc = tl.zeros((64, 64), tl.float32)
        
        num_k_tiles = cdiv(K, 64)
        for k_idx in range(num_k_tiles):
            offset_k = k_idx * 64
            a_tile = a_desc.load([offset_m, offset_k])
            b_tile = b_desc.load([offset_n, offset_k])
            acc = tl.dot(a_tile, b_tile.T, acc)
        
        c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    torch.cuda.set_device(A.device)
    
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    assert M > 0 and N > 0 and K > 0, "Invalid dimensions"
    assert (A.data_ptr() % 16 == 0) and (B.data_ptr() % 16 == 0) and (C.data_ptr() % 16 == 0), "Tensors must be 16-byte aligned"
    
    NUM_SMS = 132
    total_tiles = triton.cdiv(M, 64) * triton.cdiv(N, 64)
    grid = (min(NUM_SMS, total_tiles),)
    
    _gemm_kernel[grid](
        A.data_ptr(), B.data_ptr(), C.data_ptr(),
        M, N, K,
        NUM_SMS,
        GROUP_M=8,
        num_warps=8,
        num_stages=3
    )