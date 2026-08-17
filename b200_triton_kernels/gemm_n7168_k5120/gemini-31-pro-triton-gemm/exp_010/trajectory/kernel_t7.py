import torch
import triton
import triton.language as tl

def get_configs():
    configs = []
    # Exhaustive sweep over max-throughput Hopper configurations.
    # Tile size structures specifically saturating WGMMA usage and hiding memory latencies.
    for block_m, block_n, block_k, num_warps, num_stages in [
        # Max-throughput heavy tile configs
        (256, 128, 128, 8, 3),
        (128, 256, 128, 8, 3),
        (256, 128, 64, 8, 3),
        (256, 128, 64, 8, 4),
        (128, 256, 64, 8, 3),
        (128, 256, 64, 8, 4),
        (128, 128, 128, 8, 3),
        (128, 128, 128, 8, 4),
        
        # Balanced configs focusing on SM occupancy and tail latency distribution
        (128, 128, 64, 4, 3),
        (128, 128, 64, 4, 4),
        (128, 128, 64, 8, 4),
        (128, 128, 128, 4, 3),
        
        # Finer granularity for odd edge shapes
        (128, 64, 64, 4, 4),
        (64, 128, 64, 4, 4),
        (64, 256, 64, 4, 4),
    ]:
        # Filter against maximum allowable threads/registers limit (255 registers per thread Hopper)
        acc_size = block_m * block_n
        threads = num_warps * 32
        if acc_size / threads > 128:
            continue
            
        # Hard check for 227 KiB Hopper usable shared memory max per CTA limits
        a_size = block_m * block_k * 2 
        b_size = block_n * block_k * 2
        total_smem = (a_size + b_size) * num_stages
        
        if total_smem <= 227 * 1024:
            for group_m in [8]:
                # num_ctas=1 specifically ensures MLIR avoids the WGMMA b.T transposed lowering bug in clustered TMA
                configs.append(triton.Config(
                    {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': group_m},
                    num_warps=num_warps, num_stages=num_stages, num_ctas=1
                ))
    return configs

@triton.autotune(
    configs=get_configs(),
    key=['M']
)
@triton.jit
def _gemm_pointer(
    A, B, C,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Hierarchical Grid Swizzling to Maximize L2 Cache Reuse 
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Affine pointer formulation natively lowers to TMA on Hopper architectures
    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Constrained N (7168) and K (5120) perfectly guarantee divisibility over typical block boundaries. 
    # M is dynamic. We use a runtime fast-path if it aligns flawlessly.
    if M % BLOCK_M == 0:
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            
            # Dot is strictly A @ B.T. b is loaded as [BLOCK_N, BLOCK_K] matching B[N, K] layouts precisely
            acc = tl.dot(a, b.T, acc)
            
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
            
        c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        tl.store(c_ptrs, acc.to(C.dtype.element_ty))
    else:
        mask_m = offs_m[:, None] < M
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=mask_m, other=0.0)
            b = tl.load(b_ptrs)
            
            acc = tl.dot(a, b.T, acc)
            
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
            
        c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mask_m)


def run(A, B, C):
    """
    Computes general matrix multiply C = A @ B.T directly into C output pointer.
    
    Inputs:
    A: [M, K] bfloat16 tensor
    B: [N, K] bfloat16 tensor
    C: [M, N] preallocated bfloat16 output tensor
    """
    torch.cuda.set_device(A.device)
    
    M = A.size(0)
    # Supplied structurally cleanly per exact constant constraint parameters
    N = 7168 
    K = 5120

    def grid_fn(meta):
        # Exact non-persistent grid eliminates tail latency over mismatched boundaries
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    _gemm_pointer[grid_fn](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )