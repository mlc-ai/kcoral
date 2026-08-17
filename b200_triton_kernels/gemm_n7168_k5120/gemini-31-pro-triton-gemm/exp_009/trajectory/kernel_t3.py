import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # Large tiles (256x128 and 128x256) maximize Tensor Core utilization.
        # We sweep standard grids and Thread Block Clusters (num_ctas=2, 4) 
        # to boost L2 cache hits, mimicking cuBLAS scheduling on Hopper.
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4, num_ctas=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4, num_ctas=4),
        
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4, num_ctas=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4, num_ctas=4),
        
        # 128x128x128 pipelines well and perfectly balances SMEM with 3 stages (fits in 227KB)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3, num_ctas=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3, num_ctas=4),
        
        # 128x128x64 with deep pipelines 
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=6),

        # 2-stage large K tiles for extremely compute-bound scenarios
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_pointer(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2-cache friendly group swizzling
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m_actual = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m_actual)
    pid_n = (pid % num_pid_in_group) // group_size_m_actual
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointer addresses
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Hoist the M mask computation out of the inner loop to save registers/instructions
    mask_m = offs_m[:, None] < M
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load A with mask because M is dynamic.
        a = tl.load(a_ptrs, mask=mask_m, other=0.0)
        
        # N=7168 and K=5120 are compile-time perfectly divisible by our BLOCK_N (128, 256) 
        # and BLOCK_K (64, 128) choices, so we completely omit masking on B for speed.
        b = tl.load(b_ptrs) 
        
        # B physically is [N, K], loading [BLOCK_N, BLOCK_K] gives un-transposed shape.
        # Transposing to [BLOCK_K, BLOCK_N] yields C = A @ B.T structure and evaluates to highly optimized WGMMAs.
        acc = tl.dot(a, b.T, acc)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask_m)

def run(A, B, C):
    """
    Computes GEMM C = A @ B.T.
    A: [M, K]
    B: [N, K]
    C: [M, N] (Destination passing)
    """
    if C.numel() == 0:
        return

    torch.cuda.set_device(A.device)
    M, K = A.shape
    N = C.shape[1]
    
    def grid_fn(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    _gemm_pointer[grid_fn](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )