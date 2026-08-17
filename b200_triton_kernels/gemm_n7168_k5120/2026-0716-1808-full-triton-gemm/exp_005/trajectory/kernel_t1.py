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
    a_desc,
    b_desc,
    c_ptr,
    M,
    N,
    K,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m, pid_n = _grouped_tile_coordinates(
        tile_id, num_pid_m, num_pid_n, GROUP_M
    )
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k_iter in range(tl.cdiv(K, BLOCK_K)):
        k_step = k_iter * BLOCK_K
        
        a = a_desc.load([offset_m, k_step])
        b = b_desc.load([offset_n, k_step])
        
        acc = tl.dot(a, b.T, acc)

    mask = (offset_m + tl.arange(0, BLOCK_M)[:, None] < M) & \
           (offset_n + tl.arange(0, BLOCK_N)[None, :] < N)
           
    c_val = acc.to(tl.bfloat16)
    tl.store(c_ptr + (offset_m + tl.arange(0, BLOCK_M)[:, None]) * stride_cm + 
                  (offset_n + tl.arange(0, BLOCK_N)[None, :]) * stride_cn, 
              c_val, mask=mask, eviction_policy="evict_first")


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    
    if A.numel() == 0:
        return

    M, K = A.shape
    N, K_b = B.shape
    
    a_desc = TensorDescriptor.from_tensor(A, [256, 2048])
    b_desc = TensorDescriptor.from_tensor(B, [2048, 2048])
    
    grid = (triton.cdiv(M, 256) * triton.cdiv(N, 2048),)
    
    _gemm_kernel[grid](
        a_desc, b_desc, C,
        M, N, K,
        stride_cm=N, stride_cn=1,
        BLOCK_M=256,
        BLOCK_N=2048,
        BLOCK_K=2048,
        GROUP_M=8,
        num_warps=8,
        num_stages=4,
        maxnreg=128,
    )