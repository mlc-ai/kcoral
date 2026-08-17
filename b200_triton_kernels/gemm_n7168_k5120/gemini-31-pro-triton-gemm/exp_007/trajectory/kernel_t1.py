import torch
import triton
import triton.language as tl

def get_configs():
    """
    Return a list of configurations for autotuning.
    These tile sizes and warp counts are tuned for Hopper WGMMA execution.
    """
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=5),
    ]

@triton.autotune(
    configs=get_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_pointer_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Robust pointer-based GEMM kernel computing C = A @ B.T.
    Avoids restrictive TMA host descriptor bugs while mapping to Hopper WGMMA properly.
    """
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2-Aware Grouped Tile Ordering
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Output boundary masks
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Calculate initial pointers
    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    
    # Initialize FP32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for k0 in range(num_k_tiles):
        k = k0 * BLOCK_K + offs_k
        mask_k = k < K
        
        # 2D masks for bounds-safe loading
        a_mask = mask_m[:, None] & mask_k[None, :]
        b_mask = mask_n[:, None] & mask_k[None, :]
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # B is logically loaded as [BLOCK_N, BLOCK_K]. 
        # Passing b.T performs a [BLOCK_K, BLOCK_N] dot operation implicitly triggering optimal WGMMA.
        acc = tl.dot(a, b.T, acc)
        
        # Advance pointers to next K block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Writeback
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=c_mask)


def run(A, B, C):
    """
    Public entry point to compute C = A @ B.T into a preallocated CUDA tensor.
    N (7168) and K (5120) are fixed, M varies.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    if M == 0 or N == 0 or K == 0:
        return
        
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    
    _gemm_pointer_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )