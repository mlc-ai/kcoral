import torch
import triton
import triton.language as tl


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.heuristics(values={"NUM_SMS": lambda args: 132})
@triton.jit
def _persistent_gemm_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr,
):
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero")
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[1, K],
        block_shape=[BLOCK_K, BLOCK_N], padding_option="zero")
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N], padding_option="zero")
    
    start_tile = tl.program_id(0)
    num_pid_m = M // BLOCK_M
    num_pid_n = N // BLOCK_N
    num_tiles = num_pid_m * num_pid_n
    num_k_steps = K // BLOCK_K
    
    a_buf_0 = tl.zeros((BLOCK_M, BLOCK_K), tl.bfloat16)
    a_buf_1 = tl.zeros((BLOCK_M, BLOCK_K), tl.bfloat16)
    
    for tile_id in range(start_tile, num_tiles, NUM_SMS):
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        
        pid_in_group = tile_id % num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m
        
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc_top = tl.zeros((128, BLOCK_N), tl.float32)
        acc_bot = tl.zeros((128, BLOCK_N), tl.float32)
        
        if num_k_steps > 0:
            a_buf_0 = a_desc.load([offset_m, 0])
        if num_k_steps > 1:
            a_buf_1 = a_desc.load([offset_m, BLOCK_K])
        
        for k_step in tl.range(0, num_k_steps, 1, num_stages=4):
            if k_step % 2 == 0:
                a_curr = a_buf_0
                if k_step + 2 < num_k_steps:
                    a_next = a_desc.load([offset_m, (k_step + 2) * BLOCK_K])
                    a_buf_0 = a_next
            else:
                a_curr = a_buf_1
                if k_step + 2 < num_k_steps:
                    a_next = a_desc.load([offset_m, (k_step + 2) * BLOCK_K])
                    a_buf_1 = a_next
            
            b = b_desc.load([k_step * BLOCK_K, offset_n])
            
            a_top = a_curr[0:128, :]
            a_bot = a_curr[128:256, :]
            
            acc_top = tl.dot(a_top, b, acc_top)
            acc_bot = tl.dot(a_bot, b, acc_bot)
        
        c_top = acc_top.to(tl.bfloat16)
        c_bot = acc_bot.to(tl.bfloat16)
        c_top = tl.reshape(c_top, (BLOCK_M, BLOCK_N))
        c_bot = tl.reshape(c_bot, (BLOCK_M, BLOCK_N))
        
        c_desc.store([offset_m, offset_n], c_top)
        c_desc.store([offset_m + 128, offset_n], c_bot)


def run(A, B, C):
    """Compute ``C = A @ B.T`` into the preallocated output tensor ``C``."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    BLOCK_M = 256
    BLOCK_N = 128
    BLOCK_K = 256
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    grid = (min(132, num_tiles),)
    
    _persistent_gemm_kernel[grid](
        A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=8,
        num_warps=8, num_stages=4
    )