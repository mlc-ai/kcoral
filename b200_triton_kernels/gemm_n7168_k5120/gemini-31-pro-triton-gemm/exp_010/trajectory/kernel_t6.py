import torch
import triton
import triton.language as tl

def get_configs():
    configs = []
    # Exhaustive sweep over max-throughput Hopper configurations.
    # We aggressively tune block sizes, pipeline stages, and thread cluster sizes.
    for block_m, block_n, block_k, num_warps, num_stages in [
        # Max-throughput large tile configs for heavy workloads
        (256, 128, 128, 8, 3),
        (128, 256, 128, 8, 3),
        (256, 128, 64, 8, 3),
        (256, 128, 64, 8, 4),
        (128, 256, 64, 8, 3),
        (128, 256, 64, 8, 4),
        (128, 128, 128, 8, 3),
        (128, 128, 128, 8, 4),
        
        # Balanced configs focusing on SM occupancy
        (128, 128, 64, 4, 3),
        (128, 128, 64, 4, 4),
        (128, 128, 64, 8, 4),
        
        # Smaller block chunks optimizing dynamic load-balancing (tail latency prevention)
        (64, 256, 64, 4, 3),
        (256, 64, 64, 4, 3),
        (64, 128, 128, 4, 3),
        (128, 64, 128, 4, 3),
    ]:
        acc_size = block_m * block_n
        threads = num_warps * 32
        
        # Max 255 registers per thread Hopper limit check
        if acc_size / threads > 128:
            continue
            
        a_size = block_m * block_k * 2 
        b_size = block_n * block_k * 2
        total_smem = (a_size + b_size) * num_stages
        
        # Guard against maximum usable shared memory limit on Hopper (227 KiB)
        if total_smem <= 227 * 1024:
            for group_m in [8]:
                # Enable Thread Block Clusters (num_ctas > 1) for extreme TMA & L2 efficiency
                for num_ctas in [1, 2, 4]:
                    configs.append(triton.Config(
                        {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': group_m},
                        num_warps=num_warps, num_stages=num_stages, num_ctas=num_ctas
                    ))
    return configs

@triton.autotune(
    configs=get_configs(),
    key=['M']
)
@triton.jit
def _gemm_pointer(
    a_ptr, b_ptr, c_ptr,
    M,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    
    # 1. Hierarchical Grid Swizzling to Maximize L2 Cache Reuse 
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

    # 2. Structure Affine Pointer Blocks
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    
    # B is physically [N, K]. Stride BK is 1, stride BN is K.
    # We address B structurally as [BLOCK_K, BLOCK_N]. 
    # Because stride_bk = 1, its dim 0 is contiguous in global memory.
    # When Triton loads this, it evaluates cleanly as a column-major tensor perfectly natively aligning 
    # to WGMMA operand B (preventing the transposed=True MLIR compilation bugs associated with b.T in clusters).
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 3. Main Product Loop
    # We apply a hardware fast-path evaluating uniform execution when shapes natively divide the blocks.
    if M % BLOCK_M == 0:
        for _ in range(0, K, BLOCK_K):
            # Drop constraint masks entirely for N and K since 7168 and 5120 divide seamlessly
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            
            # Multiply loaded [BLOCK_M, BLOCK_K] with [BLOCK_K, BLOCK_N] computing exactly C = A @ B.T
            acc = tl.dot(a, b, acc)
            
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
            
        c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))
    else:
        mask_m = offs_m[:, None] < M
        for _ in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=mask_m, other=0.0)
            b = tl.load(b_ptrs)
            
            acc = tl.dot(a, b, acc)
            
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
            
        c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask_m)


def run(A, B, C):
    """
    Computes general matrix multiply (GEMM) C = A @ B.T.
    
    Inputs:
    A: [M, K] bfloat16 tensor
    B: [N, K] bfloat16 tensor
    C: [M, N] preallocated definition bfloat16 output tensor
    """
    torch.cuda.set_device(A.device)
    
    M = A.size(0)
    # Passed safely as strict compile-time geometry given by constraints
    N = 7168 
    K = 5120

    def grid_fn(meta):
        # Perfect dynamic grid bounds ensuring hardware manages any potential straggler/tail effects
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    _gemm_pointer[grid_fn](
        A, B, C,
        M,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        N=N, K=K
    )