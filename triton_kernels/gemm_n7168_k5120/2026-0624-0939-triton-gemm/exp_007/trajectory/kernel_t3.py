import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.heuristics(values={"NUM_SMS": lambda args: 132})
@triton.jit
def _persistent_gemm_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_tile = tl.program_id(0)
    
    num_pid_m = triton.cdiv(M, 128)
    num_pid_n = N // 128
    num_tiles = num_pid_m * num_pid_n
    num_k_steps = K // 128
    
    for tile_id in range(start_tile, num_tiles, NUM_SMS):
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        
        pid_in_group = tile_id % num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m
        
        offset_m = pid_m * 128
        offset_n = pid_n * 128
        
        acc = tl.zeros((128, 128), tl.float32)
        
        for k_step in range(num_k_steps):
            offset_k = k_step * 128
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            acc = tl.dot(a, b.T, acc)
        
        c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into the preallocated output tensor ``C``."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    a_desc = TensorDescriptor.from_tensor(A, [128, 128])
    b_desc = TensorDescriptor.from_tensor(B, [128, 128])
    c_desc = TensorDescriptor.from_tensor(C, [128, 128])
    
    num_pid_m = triton.cdiv(M, 128)
    num_pid_n = N // 128
    num_tiles = num_pid_m * num_pid_n
    
    grid = (min(132, num_tiles),)
    
    _persistent_gemm_kernel[grid](
        a_desc, b_desc, c_desc, M, N, K,
        GROUP_M=4,
        num_warps=8, num_stages=3
    )