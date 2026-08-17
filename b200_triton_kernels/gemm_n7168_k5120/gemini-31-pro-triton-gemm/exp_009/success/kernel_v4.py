import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # Large configs to maximize WGMMA Tensor Core throughput
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
        
        # 128 K-tiles for extreme math density
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
        
        # Deep pipelines for smaller blocks ensuring absolute memory latency hiding
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=8, num_stages=6),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=4, num_stages=6),
        
        # High occupancy profiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=4, num_stages=5),
    ],
    key=['M'],
)
@triton.jit
def _gemm_pointer_optimized(
    a_ptr, b_ptr, c_ptr,
    M,
    stride_am, stride_bn, stride_cm,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Grid Swizzle for L2 cache reuse
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m_actual = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m_actual)
    pid_n = (pid % num_pid_in_group) // group_size_m_actual
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # A is row-major physically. We load it straight.
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :])
    
    # B is [N, K] physically, therefore contiguous along K (stride_bk = 1).
    # We construct our tile block as [BLOCK_K, BLOCK_N], assigning B's fast K dim to our fast dim 0.
    # This fully coalesces memory reads while perfectly depositing a column-major matrix into SMEM.
    # Hopper WGMMA natively executes row-major(A) @ col-major(B) achieving immense bandwidth!
    b_ptrs = b_ptr + (offs_k[:, None] + offs_n[None, :] * stride_bn)
    
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :])
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Only M requires a boundary check
    mask_m = offs_m[:, None] < M
    
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=mask_m, other=0.0)
        
        # N=7168 & K=5120 are perfectly divisible by all configured BLOCKS. Safely omitting masks
        # completely side-steps bounds checking instructions yielding maximum software pipeline throughput.
        b = tl.load(b_ptrs)
        
        acc = tl.dot(a, b, acc)
        
        # Because we arranged both pointers to move along the contiguous K axis as their 
        # innermost layout, advancing elements is reduced down to flat scalar adds!
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
        
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_m)


def run(A, B, C):
    """
    Highly Optimized GEMM C = A @ B.T.
    A: [M, 5120]
    B: [7168, 5120]
    C: [M, 7168]
    """
    if C.numel() == 0:
        return

    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = 7168
    K = 5120
    
    def grid_fn(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    _gemm_pointer_optimized[grid_fn](
        A, B, C,
        M,
        A.stride(0),
        B.stride(0),
        C.stride(0),
        N=N, K=K
    )