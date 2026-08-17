import torch
import triton
import triton.language as tl

@triton.heuristics({
    'EVEN_M': lambda args: args['M'] % args['BLOCK_M'] == 0,
})
@triton.autotune(
    configs=[
        # 128x256 block configurations (High arithmetic intensity)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        
        # 256x128 block configurations
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        
        # 128x128 block configurations
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=4),
        
        # 64x256 and 256x64 fallback configurations
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M']
)
@triton.jit
def _gemm_kernel(
    A, B, C,
    M, 
    stride_am, stride_bn, stride_cm,
    # Injecting inner strides as constexpr integers perfectly informs the compiler of statically contiguous 
    # memory loads, guaranteeing optimal Hopper LDSM instruction generation without masking ambiguity.
    STRIDE_AK: tl.constexpr, STRIDE_BK: tl.constexpr, STRIDE_CN: tl.constexpr,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2 Cache locality swizzling (Groups computation into blocks to maximize L2 hit rates)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    GROUP_M_ACTUAL = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % GROUP_M_ACTUAL)
    pid_n = (pid % num_pid_in_group) // GROUP_M_ACTUAL

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * STRIDE_AK)
    b_ptrs = B + (offs_bn[:, None] * stride_bn + offs_k[None, :] * STRIDE_BK)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K (5120) and N (7168) dimensions are perfectly divisible by all tuned BLOCK sizes.
    # Therefore, we conditionally eliminate the M mask altogether via EVEN_M heuristic to hit peak TFLOPS.
    if EVEN_M:
        for _ in range(0, K // BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs) 
            
            # Leveraging .T here perfectly bridges physical memory layouts to Hopper's WGMMA expected fast-path layout.
            acc = tl.dot(a, b.T, acc)
            
            a_ptrs += BLOCK_K * STRIDE_AK
            b_ptrs += BLOCK_K * STRIDE_BK
    else:
        mask_m = offs_am[:, None] < M
        for _ in range(0, K // BLOCK_K):
            a = tl.load(a_ptrs, mask=mask_m, other=0.0)
            b = tl.load(b_ptrs) 
            
            acc = tl.dot(a, b.T, acc)
            
            a_ptrs += BLOCK_K * STRIDE_AK
            b_ptrs += BLOCK_K * STRIDE_BK
            
    acc = acc.to(C.dtype.element_ty)
    
    c_ptrs = C + (offs_am[:, None] * stride_cm + offs_bn[None, :] * STRIDE_CN)
    
    if EVEN_M:
        tl.store(c_ptrs, acc)
    else:
        mask_m = offs_am[:, None] < M
        tl.store(c_ptrs, acc, mask=mask_m)

def run(A, B, C):
    """
    Computes general matrix multiplication C = A @ B.T heavily optimized for standard Hopper architecture.
    Inputs:
      A: [M, K] bfloat16
      B: [N, K] bfloat16
    Outputs:
      C: [M, N] (Preallocated) bfloat16
    """
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = 7168
    K = 5120
    
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
        
    _gemm_kernel[grid](
        A, B, C,
        M, 
        A.stride(0), B.stride(0), C.stride(0),
        STRIDE_AK=A.stride(1), STRIDE_BK=B.stride(1), STRIDE_CN=C.stride(1),
        N=N, K=K,
    )