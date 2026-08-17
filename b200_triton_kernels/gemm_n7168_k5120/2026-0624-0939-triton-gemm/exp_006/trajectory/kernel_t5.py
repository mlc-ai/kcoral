import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


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
    a_desc, b_desc, c_desc,
    M, N, K,
    NUM_SMS: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = M // 128
    num_pid_n = N // 256
    num_tiles = num_pid_m * num_pid_n
    
    if start_pid >= num_tiles:
        return
        
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        if tile_id >= num_tiles:
            break
            
        pid_m, pid_n = _grouped_tile_coordinates(
            tile_id, num_pid_m, num_pid_n, GROUP_M
        )
        
        offset_m = pid_m * 128
        offset_n = pid_n * 256
        
        acc = tl.zeros((128, 256), tl.float32)
        
        for k_idx in range(K // 128):
            offset_k = k_idx * 128
            a_tile = a_desc.load([offset_m, offset_k])
            b_tile = b_desc.load([offset_n, offset_k])
            acc = tl.dot(a_tile, b_tile.T, acc)
        
        c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    assert M % 128 == 0, f"M must be divisible by 128, got {M}"
    assert N % 256 == 0, f"N must be divisible by 256, got {N}"
    assert K % 128 == 0, f"K must be divisible by 128, got {K}"
    
    a_desc = TensorDescriptor.from_tensor(A, [128, 128])
    b_desc = TensorDescriptor.from_tensor(B, [256, 128])
    c_desc = TensorDescriptor.from_tensor(C, [128, 256])
    
    NUM_SMS = 132
    total_tiles = (M // 128) * (N // 256)
    grid = (min(NUM_SMS, total_tiles),)
    
    _gemm_kernel[grid](
        a_desc, b_desc, c_desc,
        M, N, K,
        NUM_SMS=132,
        GROUP_M=8,
        num_warps=8,
        num_stages=3
    )