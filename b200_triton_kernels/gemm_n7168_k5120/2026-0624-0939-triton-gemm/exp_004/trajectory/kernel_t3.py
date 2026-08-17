import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _persistent_gemm_kernel(
    a_desc, b_desc, C_ptr,
    M, N, K,
    stride_Cm, stride_Cn,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False):
        # Explicitly resolve our (BLOCK_M, BLOCK_N) destination tile
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_in_group = tile_id % num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m
        
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        # Iterate over the contracted K dimension resolving TMA descriptors directly.
        num_k_tiles = tl.cdiv(K, BLOCK_K)
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            # Mask out of bounds elements safely participating in the reduction
            row = tl.arange(0, BLOCK_M)[:, None]
            col_a = tl.arange(0, BLOCK_K)[None, :]
            mask_a = (offset_m + row) < M
            a = a * mask_a
            
            col_b = tl.arange(0, BLOCK_K)[:, None]
            mask_b = (offset_k + col_b) < K
            b = b * mask_b
            
            acc = tl.dot(a, b.T, acc)
            
        out = acc.to(tl.bfloat16)
        
        row = tl.arange(0, BLOCK_M)
        col = tl.arange(0, BLOCK_N)
        ptrs = C_ptr + (offset_m + row)[:, None] * stride_Cm + (offset_n + col)[None, :] * stride_Cn
        
        mask = ((offset_m + row)[:, None] < M) & ((offset_n + col)[None, :] < N)
        tl.store(ptrs, out, mask=mask)


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    
    M, K = A.shape
    N = B.shape[0]
    
    block_m = 64
    block_n = 128
    block_k = 64
    
    NUM_SMS = 132
    GROUP_M = 8
    
    num_pid_m = triton.cdiv(M, block_m)
    num_pid_n = triton.cdiv(N, block_n)
    num_tiles = num_pid_m * num_pid_n
    
    grid = (min(NUM_SMS, num_tiles),)
    
    a_desc = TensorDescriptor.from_tensor(A, [block_m, block_k])
    b_desc = TensorDescriptor.from_tensor(B, [block_n, block_k])
    
    _persistent_gemm_kernel[grid](
        a_desc, b_desc, C, M, N, K,
        N, 1,  
        NUM_SMS=NUM_SMS,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        GROUP_M=GROUP_M,
        num_warps=8,
        num_stages=4,
    )