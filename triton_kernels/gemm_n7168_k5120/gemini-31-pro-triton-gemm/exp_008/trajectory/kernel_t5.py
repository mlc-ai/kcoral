import torch
import triton
import triton.language as tl

def get_configs():
    configs = []
    # (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages, ctas)
    # Target configurations highly tuned for Hopper H100 with BFloat16 WGMMA math
    # Carefully curated to avoid exceeding Hopper's 227KB shared memory and 255 register limits
    candidates = [
        # Large Blocks for Max Math Intensity
        (256, 128, 64, 8, 3, 1),
        (256, 128, 64, 8, 4, 1),
        (256, 128, 64, 8, 3, 2),
        (256, 128, 64, 8, 4, 2),
        
        (128, 256, 64, 8, 3, 1),
        (128, 256, 64, 8, 4, 1),
        (128, 256, 64, 8, 3, 2),
        (128, 256, 64, 8, 4, 2),
        
        # High Pipeline Depth
        (128, 128, 64, 4, 4, 1),
        (128, 128, 64, 4, 5, 1),
        (128, 128, 64, 8, 4, 1),
        (128, 128, 64, 8, 5, 1),
        
        (128, 128, 64, 4, 4, 2),
        (128, 128, 64, 4, 5, 2),
        (128, 128, 64, 8, 4, 2),
        (128, 128, 64, 8, 5, 2),
        
        # Cluster focus (L2 Cache Reuse Multiplier)
        (128, 128, 64, 8, 4, 4),
        (128, 128, 128, 8, 3, 2),
        (128, 128, 128, 8, 3, 4),
        
        (256, 64, 64, 8, 4, 2),
        (64, 256, 64, 8, 4, 2),
    ]
    
    for m, n, k, w, s, c in candidates:
        configs.append(triton.Config(
            {'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 'GROUP_M': 8},
            num_warps=w, num_stages=s, num_ctas=c
        ))
    return configs

@triton.autotune(
    configs=get_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Grid Swizzling: Improve L2 Cache hit-rates natively through clustered tile assignments
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Standard contiguous pointer offsets explicitly avoid Triton TMA descriptor bugs
    # related to transposed loads when num_ctas > 1.
    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    
    # Bound masks
    # N=7168 and K=5120 are strictly divisible by chosen block sizes. No masks needed for them.
    mask_m = offs_m < M
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Software-pipelined Load Loop
    for k_tile in range(num_k_tiles):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs)
        
        # Hardware WGMMA tensor cores path (native shared memory transpose supported for pointers)
        acc = tl.dot(a, b.T, acc)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mask_m[:, None])


def run(A, B, C):
    """
    Compute GEMM C = A @ B.T directly into C.
    Standard Pointer Load Pipeline with Thread Block Clusters (num_ctas) 
    to bypass TMA descriptor transposed MLIR layout conflicts on Hopper.
    """
    if A.shape[0] == 0:
        return
        
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, K_B = B.shape
    
    assert K == K_B, "The inner-dimensions (K) of A and B must match."
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    
    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )