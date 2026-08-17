import torch
import triton
import triton.language as tl

def get_configs():
    configs = []
    # Exhaustive tuning space optimized for NVIDIA Hopper SM90 architecture.
    # By avoiding runtime mask checks for well-behaved dimensions, we allow the compiler
    # to heavily pipeline these tiles utilizing Hopper's asynchronous memory operations natively.
    for block_m, block_n, block_k, warps, stages, ctas in [
        # Huge tiles for massive dense throughput
        (256, 128, 128, 8, 2, 1),
        (128, 256, 128, 8, 2, 1),
        (256, 128, 64,  8, 4, 1),
        (256, 128, 64,  8, 3, 2),
        (128, 256, 64,  8, 4, 1),
        (128, 256, 64,  8, 3, 2),
        
        # Balanced 128x128 tiles with varying pipeline depths and CTA clustering
        (128, 128, 128, 8, 3, 1),
        (128, 128, 128, 8, 3, 2),
        (128, 128, 64,  8, 5, 1),
        (128, 128, 64,  4, 5, 1),
        
        # Narrow tiles which can hit resource resonance on specific grid sizes
        (64, 256, 64,   4, 5, 1),
        (256, 64, 64,   4, 5, 1),
        (64, 128, 128,  4, 4, 1),
        (128, 64, 128,  4, 4, 1),
    ]:
        # Enforce Hopper H100 shared memory limits (~227 KiB usable per SM per block)
        shm_size = (block_m * block_k * 2 + block_n * block_k * 2) * stages
        if shm_size > 225000:
            continue
            
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
    EVEN_M: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = N // BLOCK_N  
    
    # L2 Cache Swizzling for spatial locality
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    
    # Safely compute group size boundary without tripping Python min() in MLIR tracing
    diff = num_pid_m - first_pid_m
    group_size_m = tl.where(diff < GROUP_M, diff, GROUP_M)
    
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointer computation leveraging contiguous access properties
    A_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    
    # Accumulate precision optimally in float32 for bf16 tensor cores
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    num_k_steps = K // BLOCK_K 
    
    if EVEN_M:
        # Fully optimized unrolled loop lacking mask conditions
        for _ in range(num_k_steps):
            a = tl.load(A_ptrs)
            b = tl.load(B_ptrs)
            
            # Physical memory layout translated properly into dot operands natively:
            # B is [N, K], transposing it gives [K, N] logically matching dot structure.
            acc = tl.dot(a, b.T, acc)
            
            A_ptrs += BLOCK_K * stride_ak
            B_ptrs += BLOCK_K * stride_bk
    else:
        # Safe fallback loop with masking enforced primarily against dynamic M boundaries
        a_mask = offs_am[:, None] < M
        for _ in range(num_k_steps):
            a = tl.load(A_ptrs, mask=a_mask, other=0.0)
            b = tl.load(B_ptrs)
            
            acc = tl.dot(a, b.T, acc)
            
            A_ptrs += BLOCK_K * stride_ak
            B_ptrs += BLOCK_K * stride_bk
            
    # Conversion back to output storage type
    acc = acc.to(tl.bfloat16)
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    
    # Matching conditional write-back logic
    if EVEN_M:
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
    
    # Allows the compiler to construct a mask-free version by knowing that M is 
    # perfectly divisible by the maximum BLOCK_M parameter in the autotuning space.
    EVEN_M = (M % 256 == 0)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * (N // META['BLOCK_N']),)
    
    _gemm_pointer_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        EVEN_M=EVEN_M,
    )