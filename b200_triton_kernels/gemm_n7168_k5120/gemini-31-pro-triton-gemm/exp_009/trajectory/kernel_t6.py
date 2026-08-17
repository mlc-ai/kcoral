import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # High compute density with L2 Cache Clustering (num_ctas=2, 4)
        # Thread Block Clusters significantly boost performance for large GEMMs by sharing L2.
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4, num_ctas=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4, num_ctas=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4, num_ctas=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4, num_ctas=4),
        
        # Max K-dimension configs (Only possible without transpose overhead)
        # 128x256x128 with 2 stages uses ~196KB SMEM, fitting exactly in Hopper's 227KB limit.
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=2, num_ctas=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=2, num_ctas=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=2, num_ctas=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=2, num_ctas=4),
        
        # Balanced 128x128 tiles
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3, num_ctas=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3, num_ctas=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=4, num_stages=5, num_ctas=2),
        
        # Non-clustered fallbacks
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4, num_ctas=1),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4, num_ctas=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=2, num_ctas=1),
    ],
    key=['M'],
)
@triton.jit
def _gemm_kernel(
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
    
    # L2 Cache Grid Swizzle
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m_actual = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m_actual)
    pid_n = (pid % num_pid_in_group) // group_size_m_actual
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :])
    
    # [OPTIMIZATION] 
    # B is physically [N, K] (stride_bn, 1). We load it into SMEM transposed as [BLOCK_K, BLOCK_N].
    # This directly forces a column-major layout into shared memory which is perfectly contiguous for reads.
    # Hopper WGMMA Hardware natively executes Row-Major(A) @ Col-Major(B) achieving peak dense FLOPS
    # and strictly avoids the BLOCK_K=64 limitation required when dynamically transposing via `b.T`.
    b_ptrs = b_ptr + (offs_k[:, None] + offs_n[None, :] * stride_bn)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Only A and C require mask on M. 
    # B requires no mask because N=7168 and K=5120 are strictly perfectly divisible.
    mask_m = offs_m[:, None] < M
    
    for _ in range(0, K // BLOCK_K):
        a = tl.load(a_ptrs, mask=mask_m, other=0.0)
        b = tl.load(b_ptrs)
        
        # WGMMA executes natively over a @ b without transposing registers.
        acc = tl.dot(a, b, acc)
        
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
        
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :])
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask_m)


def run(A, B, C):
    """
    GEMM C = A @ B.T.
    Optimal for Hopper SM90 via Thread Block Clusters (num_ctas) and direct col-major B loads.
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

    _gemm_kernel[grid_fn](
        A, B, C,
        M,
        A.stride(0),
        B.stride(0),
        C.stride(0),
        N=N, K=K
    )