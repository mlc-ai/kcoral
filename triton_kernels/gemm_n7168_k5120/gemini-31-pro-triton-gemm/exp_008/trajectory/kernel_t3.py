import torch
import triton
import triton.language as tl

def get_configs():
    configs = []
    # Exhaustive hardware-aligned candidates for optimal memory constraints and cluster dimensions.
    # Carefully curated to avoid H100 shared memory (227KB) and register (255/thread) limits.
    valid_combinations = [
        # (block_m, block_n, block_k, warps, stages, ctas)
        
        # 256x128x64 (128 regs, 49KB/stage)
        (256, 128, 64, 8, 3, 1),
        (256, 128, 64, 8, 4, 1),
        (256, 128, 64, 8, 3, 2),
        (256, 128, 64, 8, 4, 2),
        
        # 128x256x64 (128 regs, 49KB/stage)
        (128, 256, 64, 8, 3, 1),
        (128, 256, 64, 8, 4, 1),
        (128, 256, 64, 8, 3, 2),
        (128, 256, 64, 8, 4, 2),
        
        # 128x128x128 (64 regs, 65.5KB/stage)
        (128, 128, 128, 8, 3, 1),
        (128, 128, 128, 8, 3, 2),
        
        # 128x256x128 (128 regs, 98KB/stage) -> restricted to 2 stages
        (128, 256, 128, 8, 2, 1),
        (128, 256, 128, 8, 2, 2),
        
        # 256x128x128 (128 regs, 98KB/stage) -> restricted to 2 stages
        (256, 128, 128, 8, 2, 1),
        (256, 128, 128, 8, 2, 2),
        
        # 128x128x64 (64-128 regs, 32KB/stage)
        (128, 128, 64, 4, 4, 1),
        (128, 128, 64, 4, 4, 2),
        (128, 128, 64, 8, 4, 1),
        (128, 128, 64, 8, 4, 2),
        
        # Aggressive Cluster sizes for high L2 Cache Reuse
        (128, 128, 64, 8, 4, 4),
        (128, 128, 128, 8, 3, 4),
    ]
    for block_m, block_n, block_k, warps, stages, ctas in valid_combinations:
        configs.append(triton.Config(
            {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': 8},
            num_warps=warps, 
            num_stages=stages, 
            num_ctas=ctas
        ))
    return configs


@triton.autotune(
    configs=get_configs(),
    key=['M'],
)
@triton.jit
def _gemm_kernel(
    A, B, C,
    M,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # N and K are strictly bound constants from the problem context.
    N: tl.constexpr = 7168
    K: tl.constexpr = 5120

    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Grid Swizzling: Improve L2 Reuse natively through clustered mapping layout.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    
    # Math trick: Construct b_ptrs with shape (BLOCK_K, BLOCK_N) effectively fetching it transposed.
    # This prevents the H100 MLIR legalization bug caused by `#ttg.memdesc transposed = true`.
    b_ptrs = B + (offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    # N (7168) and K (5120) are mathematically perfectly divisible by all chosen BLOCK_N & BLOCK_K respectively.
    # Bounds checking is aggressively stripped from the inner pipeline!
    mask_m = offs_am < M
    
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Software-pipelined load loop. Triton lowers to CP.ASYNC or WGMMA/TMA arrays seamlessly
    for k_tile in range(num_k_tiles):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs)
        
        # WGMMA tensor core dot without `.T` invocation since B was layout-loaded inverted
        acc = tl.dot(a, b, acc)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        
    c_ptrs = C + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mask_m[:, None])


def run(A, B, C):
    """
    Compute GEMM C = A @ B.T directly into C.
    Hardware-Scheduled highly optimized implementation tailored for Hopper.
    """
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(7168, META['BLOCK_N']), )
    
    _gemm_kernel[grid](
        A, B, C,
        M,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )