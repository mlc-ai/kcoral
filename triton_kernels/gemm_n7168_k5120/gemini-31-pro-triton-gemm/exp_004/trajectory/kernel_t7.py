import torch
import triton
import triton.language as tl

def get_autotune_config():
    configs = []
    # Broad configuration space thoroughly targeting Hopper WGMMA registers and L2 characteristics.
    shapes = [
        # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
        (256, 128, 64, 8),
        (128, 256, 64, 8),
        (128, 128, 128, 8),
        (128, 128, 64, 4),
        (256, 64, 64, 4),
        (64, 256, 64, 4),
        (128, 64, 128, 4),
        (64, 128, 128, 4),
    ]
    
    for m, n, k, w in shapes:
        # bfloat16 demands 2 bytes per element. 
        bytes_per_stage = (m * k + n * k) * 2
        # Keep total staged tiles safely beneath Hopper's ~228KB Shared Memory limit per SM module.
        max_stages = min(5, 225000 // bytes_per_stage)
        
        for stages in range(2, max_stages + 1):
            for group_m in [8, 16]:
                # num_ctas activates threadblock clustering natively for massive L2 throughput
                for ctas in [1, 2, 4]:
                    configs.append(triton.Config(
                        {'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 'GROUP_M': group_m},
                        num_stages=stages, num_warps=w, num_ctas=ctas
                    ))
    return configs

@triton.autotune(
    configs=get_autotune_config(),
    key=['M'],
)
@triton.jit
def _gemm_kernel(
    A, B, C,
    M,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    M_DIVISIBLE: tl.constexpr
):
    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2-Aware Output Tile Ordering mapping
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    
    # Low-precision floating point accumulation should execute via FP32 format natively.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Because M_DIVISIBLE evaluates strictly at compile-time (tl.constexpr), Triton guarantees 
    # the unmasked fast-path maintains uninterrupted software pipeline loops.
    if M_DIVISIBLE:
        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            a_tile = tl.load(a_ptrs)
            b_tile = tl.load(b_ptrs)
            
            # Submitting transposed layout right-operand (b_tile.T) properly informs Triton
            # to target peak hardware WGMMA formatting pathways.
            acc = tl.dot(a_tile, b_tile.T, acc)
            
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
            
        c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        tl.store(c_ptrs, acc.to(tl.bfloat16))
        
    else:
        a_mask = offs_m[:, None] < M
        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b_tile = tl.load(b_ptrs)
            
            acc = tl.dot(a_tile, b_tile.T, acc)
            
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
            
        c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        c_mask = offs_m[:, None] < M
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)

def run(A, B, C):
    """
    General matrix multiply C = A @ B.T.
    A: [M, K]
    B: [N, K]
    C: [M, N]
    Both N (7168) and K (5120) strictly maintained as specified compile-time constant shapes.
    """
    if A.shape[0] == 0:
        return
        
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    
    # Extract structural dimensions provided from problem definition
    N = 7168
    K = 5120
    
    # Because our maximum BLOCK_M sweep tests up to 256, if M is perfectly divisible by 256,
    # it is mathematically proven to be perfectly divisible by all tested BLOCK_M options (64, 128, 256).
    m_divisible = (M % 256 == 0)
    
    def grid_fn(META):
        num_pid_m = triton.cdiv(M, META['BLOCK_M'])
        num_pid_n = triton.cdiv(N, META['BLOCK_N'])
        return (num_pid_m * num_pid_n, )
    
    _gemm_kernel[grid_fn](
        A, B, C,
        M,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        N=N, K=K,
        M_DIVISIBLE=m_divisible
    )