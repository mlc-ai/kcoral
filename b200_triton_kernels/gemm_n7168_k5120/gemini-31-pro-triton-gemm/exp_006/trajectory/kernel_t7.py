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
    """L2-aware mapping that keeps output tiles in localized blocks to reuse L2 resident operands."""
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    return pid_m, pid_n


@triton.autotune(
    configs=[
        # Thread Block Clusters (Hopper feature) usually provide the best L2 cache reuse 
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4, num_ctas=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4, num_ctas=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4, num_ctas=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4, num_ctas=4),

        # High arithmetic intensity shapes fallback
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        
        # Deep pipelines for smaller blocks (preventing shared mem overallocation)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        
        # Wider K
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=["M"],
)
@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    M_DIVISIBLE: tl.constexpr,
):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m, pid_n = _grouped_tile_coordinates(tile_id, num_pid_m, num_pid_n, GROUP_M)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # N=7168 and K=5120 perfectly divide our config block sizes, avoiding inner loop bound-checking
    if M_DIVISIBLE:
        # Fast path without ANY masking guarantees peak software pipelining capability
        for _ in range(0, K // BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            # Logically transpose the loaded `[BLOCK_N, BLOCK_K]` slice to fit WGMMA hardware mapping
            acc = tl.dot(a, b.T, acc)
            
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
            
        c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))
        
    else:
        # Standard fallback path with robust M masking
        mask_m = offs_m < M
        mask_a = mask_m[:, None]
        for _ in range(0, K // BLOCK_K):
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)
            b = tl.load(b_ptrs) 
            acc = tl.dot(a, b.T, acc)
            
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
            
        c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask_m[:, None])


def run(A, B, C):
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    
    # Adhere directly to problem context constraints
    N = 7168
    K = 5120
    
    # Max block-M supported is 256. Using mod isolates inner-loop behavior ahead-of-time safely
    m_divisible = (M % 256 == 0)
    
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    
    _gemm_kernel[grid](
        A, B, C,
        M,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        N=N, K=K,
        M_DIVISIBLE=m_divisible
    )