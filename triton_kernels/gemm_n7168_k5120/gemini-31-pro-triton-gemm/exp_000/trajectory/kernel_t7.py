import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # BLOCK_K = 128 (Maximum arithmetic intensity)
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=2, num_warps=8),

        # BLOCK_K = 64 (Better memory latency hiding via pipelining with more stages)
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        
        # Fallbacks for specific smaller shapes mapped evenly across warps
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=4),
    ],
    key=['M']
)
@triton.jit
def _gemm_kernel(
    A, B, C,
    M, 
    stride_am, stride_bn, stride_cm,
    N: tl.constexpr, 
    K: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2 Cache-optimized grouped swizzling algorithm
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    GROUP_M_ACTUAL = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % GROUP_M_ACTUAL)
    pid_n = (pid % num_pid_in_group) // GROUP_M_ACTUAL

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Intentionally utilizing `1` statically guarantees to the Triton compiler that the 
    # inner dimensions are memory-contiguous, perfectly resolving Hopper LDSM unrolling.
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * 1)
    b_ptrs = B + (offs_bn[:, None] * stride_bn + offs_k[None, :] * 1)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # N=7168 and K=5120 are perfectly divisible by all tuned BLOCK_N & BLOCK_K factors.
    # Therefore, dynamic conditional branching entirely removes bounding logic costs for perfectly mapped M shapes.
    if M % BLOCK_M == 0:
        for _ in range(0, K, BLOCK_K):
            # No masking necessary; pure async throughput
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            
            # WGMMA prefers col-major layout for right operand conceptually; `b.T` pushes layout effectively
            acc = tl.dot(a, b.T, acc)
            
            a_ptrs += BLOCK_K * 1
            b_ptrs += BLOCK_K * 1
    else:
        mask_m = offs_am[:, None] < M
        for _ in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=mask_m, other=0.0)
            b = tl.load(b_ptrs)
            
            acc = tl.dot(a, b.T, acc)
            
            a_ptrs += BLOCK_K * 1
            b_ptrs += BLOCK_K * 1
            
    acc = acc.to(C.dtype.element_ty)
    c_ptrs = C + (offs_am[:, None] * stride_cm + offs_bn[None, :] * 1)
    
    if M % BLOCK_M == 0:
        tl.store(c_ptrs, acc)
    else:
        mask_m = offs_am[:, None] < M
        tl.store(c_ptrs, acc, mask=mask_m)

def run(A, B, C):
    """
    Computes general matrix multiplication C = A @ B.T finely tuned for H100 Hopper constraints.
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
        # A standard non-persistent launch leverages cuBLAS-like round-robin hardware dispatch precisely
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
        
    _gemm_kernel[grid](
        A, B, C,
        M, 
        A.stride(0), B.stride(0), C.stride(0),
        N=N, K=K,
    )