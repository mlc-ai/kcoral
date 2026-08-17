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
    A_ptr, B_ptr, c_desc,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_A_m, stride_A_k, stride_B_n, stride_B_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = _grouped_tile_coordinates(
        tile_id,
        num_pid_m,
        num_pid_n,
        GROUP_M,
    )
    
    if pid_m == -1 or pid_n == -1:
        return
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    a_buf = tl.shared("[2, 256, 128]", dtype=tl.float32)
    b_buf = tl.shared("[2, 512, 128]", dtype=tl.float32)
    ctrl = tl.shared("[1]", dtype=tl.int32, initial_value=0)
    
    acc = tl.zeros((256, 512), tl.float32)
    
    num_k_steps = K // (2 * 128)
    
    k = 0
    a_ptrs = A_ptr + offset_m * stride_A_m + k * stride_A_k + \
             tl.arange(0, 256)[:, None] * stride_A_m + \
             tl.arange(0, 256)[None, :] * stride_A_k
    mask_a = (offset_m + tl.arange(0, 256)[:, None]) < M
    tl.async_pg_copy(a_ptrs[..., 0:128], a_buf[0, :, 0:128], mask=mask_a)
    tl.async_pg_copy(a_ptrs[..., 128:256], a_buf[0, :, 128:256], mask=mask_a)
    
    b_ptrs = B_ptr + offset_n * stride_B_n + k * stride_B_k + \
             tl.arange(0, 512)[:, None] * stride_B_n + \
             tl.arange(0, 128)[None, :] * stride_B_k
    mask_b = (offset_n + tl.arange(0, 512)[:, None]) < N
    tl.async_pg_copy(b_ptrs[..., 0:128], b_buf[0, :, 0:128], mask=mask_b)
    tl.async_pg_copy(b_ptrs[..., 128:256], b_buf[0, :, 128:256], mask=mask_b)
    tl.commit()
    
    for step in range(num_k_steps):
        next_k = (step + 1) * 2 * 128
        next_ctrl = (ctrl[0] + 1) % 2
        
        if next_k + 128 <= K:
            a_ptrs_next = A_ptr + offset_m * stride_A_m + next_k * stride_A_k + \
                          tl.arange(0, 256)[:, None] * stride_A_m + \
                          tl.arange(0, 256)[None, :] * stride_A_k
            tl.async_pg_copy(a_ptrs_next[..., 0:128], a_buf[next_ctrl, :, 0:128], mask=mask_a)
            tl.async_pg_copy(a_ptrs_next[..., 128:256], a_buf[next_ctrl, :, 128:256], mask=mask_a)
            
            b_ptrs_next = B_ptr + offset_n * stride_B_n + next_k * stride_B_k + \
                          tl.arange(0, 512)[:, None] * stride_B_n + \
                          tl.arange(0, 128)[None, :] * stride_B_k
            tl.async_pg_copy(b_ptrs_next[..., 0:128], b_buf[next_ctrl, :, 0:128], mask=mask_b)
            tl.async_pg_copy(b_ptrs_next[..., 128:256], b_buf[next_ctrl, :, 128:256], mask=mask_b)
        
        tl.commit()
        
        acc = tl.dot(a_buf[ctrl[0], :, :], b_buf[ctrl[0], :, :], acc)
        
        ctrl[0] = next_ctrl
        k = next_k
    
    c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    BLOCK_M = 256
    BLOCK_N = 512
    BLOCK_K = 128
    GROUP_M = 2
    
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    stride_A_m = A.stride(0)
    stride_A_k = A.stride(1)
    stride_B_n = B.stride(0)
    stride_B_k = B.stride(1)
    
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (total_tiles,)
    
    _gemm_kernel[grid](
        A, B, c_desc, M, N, K,
        stride_A_m, stride_A_k, stride_B_n, stride_B_k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_warps=8,
        num_stages=4,
    )