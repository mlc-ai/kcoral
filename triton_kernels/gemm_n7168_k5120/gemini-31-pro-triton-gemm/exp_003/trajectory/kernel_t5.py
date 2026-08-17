import torch
import triton
import triton.language as tl

def get_configs():
    configs = []
    # Exhaustive tuning space optimized for NVIDIA Hopper SM90 architecture
    # Focuses on high warp counts, optimal WGMMMA tile sizes, and high L2 reuse.
    for block_m, block_n, block_k, warps, stages, ctas in [
        # Large tiles for massive throughput (prioritizing 128x256 and 256x128)
        (256, 128, 128, 8, 3, 1),
        (128, 256, 128, 8, 3, 1),
        (256, 128, 64,  8, 4, 1),
        (128, 256, 64,  8, 4, 1),
        
        # Balanced 128x128 tiles with varying pipeline depths and clustering
        (128, 128, 128, 8, 3, 1),
        (128, 128, 128, 8, 4, 1),
        (128, 128, 128, 8, 3, 2), # CTA clustering for L2 locality
        (128, 128, 64,  4, 5, 1),
        (128, 128, 64,  8, 5, 1),
        
        # Skewed/narrower tiles which can hit resonance with specific SM bounds
        (64, 128, 128,  4, 5, 1),
        (128, 64, 128,  4, 5, 1),
        (64, 256, 64,   4, 5, 1),
        (256, 64, 64,   4, 5, 1),
        
        # Deep K-tiles (heavy k-reduction per step)
        (64, 128, 256,  4, 2, 1),
        (128, 64, 256,  4, 2, 1),
    ]:
        configs.append(triton.Config(
            {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': 8},
            num_stages=stages, num_warps=warps, num_ctas=ctas
        ))
    return configs


@triton.autotune(
    configs=get_configs(),
    key=['M'],
)
@triton.jit
def _gemm_pointer_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = N // BLOCK_N  # Exact, as N=7168 is perfectly divisible by all BLOCK_N candidates
    
    # L2 Cache Swizzling for spatial locality
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointer computation
    A_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    
    # Accumulate precision optimally in float32 for bf16 dots
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    num_k_steps = K // BLOCK_K  # Exact
    
    # Dynamic uniformity check: entirely bypass mask evaluations if M nicely divides blocks.
    if M % BLOCK_M == 0:
        for _ in range(num_k_steps):
            # Hardware vectorized loads implicitly derived via stride analysis
            a = tl.load(A_ptrs)
            b = tl.load(B_ptrs)
            # Physical layout translation avoids manual shm transposes: tl.dot translates b.T 
            acc = tl.dot(a, b.T, acc)
            
            A_ptrs += BLOCK_K * stride_ak
            B_ptrs += BLOCK_K * stride_bk
    else:
        a_mask = offs_am[:, None] < M
        for _ in range(num_k_steps):
            a = tl.load(A_ptrs, mask=a_mask, other=0.0)
            b = tl.load(B_ptrs)
            acc = tl.dot(a, b.T, acc)
            
            A_ptrs += BLOCK_K * stride_ak
            B_ptrs += BLOCK_K * stride_bk
            
    # Conversion back to bfloat16 upon complete reduction
    acc = acc.to(tl.bfloat16)
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    
    # Matching conditional write-back logic
    if M % BLOCK_M == 0:
        tl.store(C_ptrs, acc)
    else:
        c_mask = offs_cm[:, None] < M
        tl.store(C_ptrs, acc, mask=c_mask)


def run(A, B, C):
    """
    Computes general matrix multiply C = A @ B.T where:
    A: [M, K]
    B: [N, K]
    C: [M, N]
    Where N is exactly 7168 and K is exactly 5120.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = 7168
    K = 5120
    
    # Number of thread blocks across spatial grid dimensions
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * (N // META['BLOCK_N']),)
    
    _gemm_pointer_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )