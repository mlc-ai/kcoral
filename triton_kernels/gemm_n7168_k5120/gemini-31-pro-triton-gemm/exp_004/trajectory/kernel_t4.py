import torch
import triton
import triton.language as tl

def get_autotune_config():
    configs = []
    # Comprehensive autotuning for SM90 mapping path optimized for WGMMA
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m, block_n, block_k, num_warps in [
            (256, 128, 128, 8),
            (128, 256, 128, 8),
            (256, 128, 64, 8),
            (128, 256, 64, 8),
            (128, 128, 128, 8),
            (128, 128, 128, 4),
            (128, 128, 64, 8),
            (128, 128, 64, 4),
            (256, 64, 128, 8),
            (64, 256, 128, 8),
            (256, 64, 64, 4),
            (64, 256, 64, 4),
            (128, 64, 128, 4),
            (64, 128, 128, 4),
        ]:
            # Shared memory capacity check (~228KB usable per SM on Hopper)
            # 2 bytes per element for bfloat16
            bytes_per_stage = (block_m * block_k + block_n * block_k) * 2
            if bytes_per_stage * num_stages > 225000:
                continue
                
            # L2 Cache-aware layout combinations
            for group_m in [1, 4, 8]:
                configs.append(triton.Config(
                    {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': group_m},
                    num_stages=num_stages, num_warps=num_warps
                ))
    return configs

@triton.autotune(
    configs=get_autotune_config(),
    key=['M'],
)
@triton.jit
def _gemm_kernel(
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
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    if GROUP_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        
        pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
        pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    num_k_steps = K // BLOCK_K
    
    # Fast path: Completely unmasked loads inner loop when M is perfectly divisible by BLOCK_M
    # This evaluates to a uniform branch efficiently compiled down into direct 128-bit vector loads without predicates
    if M % BLOCK_M == 0:
        for k0 in range(num_k_steps):
            a_tile = tl.load(a_ptrs)
            b_tile = tl.load(b_ptrs)
            
            # Submitting transposed `b_tile` explicitly informs Triton logic of native Right-Operand SM90 formatting
            acc = tl.dot(a_tile, b_tile.T, acc)
            
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    else:
        a_mask = offs_m[:, None] < M
        for k0 in range(num_k_steps):
            a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b_tile = tl.load(b_ptrs)
            
            acc = tl.dot(a_tile, b_tile.T, acc)
            
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
            
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    
    if M % BLOCK_M == 0:
        tl.store(c_ptrs, acc.to(tl.bfloat16))
    else:
        c_mask = offs_m[:, None] < M
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


def run(A, B, C):
    """
    General matrix multiply (GEMM) C = A @ B.T.
    A: [M, K]
    B: [N, K]
    C: [M, N]
    Captured from Qwen3 14B qkv_proj. Both N=7168 and K=5120 are tightly constrained logic bounds.
    """
    if A.shape[0] == 0:
        return
        
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    
    N = 7168
    K = 5120
    
    def grid_fn(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    
    _gemm_kernel[grid_fn](
        A, B, C,
        M,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        N=N, K=K
    )