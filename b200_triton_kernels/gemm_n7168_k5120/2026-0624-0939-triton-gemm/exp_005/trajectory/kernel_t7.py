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
    if num_pid_m <= 0 or num_pid_n <= 0:
        return -1, -1
        
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    
    if first_pid_m >= num_pid_m:
        return -1, -1
        
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    if group_size_m <= 0:
        return -1, -1

    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    return pid_m, pid_n


@triton.jit
def _gemm_kernel(
    a_desc,
    b_desc,
    c_desc,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    acc_0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_2 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_3 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False):
        pid_m, pid_n = _grouped_tile_coordinates(
            tile_id,
            num_pid_m,
            num_pid_n,
            GROUP_M,
        )
        
        if pid_m == -1 or pid_n == -1:
            continue
        
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        k = 0
        a = a_desc.load([offset_m, k])
        b_0 = b_desc.load([k, offset_n])
        b_1 = b_desc.load([k + 512, offset_n])
        b_2 = b_desc.load([k + 2*512, offset_n])
        b_3 = b_desc.load([k + 3*512, offset_n])
        
        num_k_steps = K // (2*512)
        
        for step in range(num_k_steps):
            next_k = k + 2*512
            
            a_next = a_desc.load([offset_m, next_k])
            b_next_0 = b_desc.load([next_k, offset_n])
            b_next_1 = b_desc.load([next_k + 512, offset_n])
            
            if step < num_k_steps - 1:
                b_next_2 = b_desc.load([next_k + 2*512, offset_n])
                b_next_3 = b_desc.load([next_k + 3*512, offset_n])
                
            acc_0 += tl.dot(a, b_0, acc_0)
            acc_1 += tl.dot(a, b_1, acc_1)
            
            if step < num_k_steps - 1:
                acc_2 += tl.dot(a, b_2, acc_2)
                acc_3 += tl.dot(a, b_3, acc_3)
                
            a = a_next
            b_0 = b_next_0
            b_1 = b_next_1
            if step < num_k_steps - 1:
                b_2 = b_next_2
                b_3 = b_next_3
            k = next_k
            
        acc = acc_0 + acc_1 + acc_2 + acc_3
        c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    BLOCK_M = 256
    BLOCK_N = 1024
    BLOCK_K = 512
    GROUP_M = 2
    
    NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(NUM_SMS, total_tiles),)
    
    _gemm_kernel[grid](
        a_desc, b_desc, c_desc, M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        NUM_SMS=NUM_SMS,
        num_warps=8,
        num_stages=4,
        num_ctas=1,
    )