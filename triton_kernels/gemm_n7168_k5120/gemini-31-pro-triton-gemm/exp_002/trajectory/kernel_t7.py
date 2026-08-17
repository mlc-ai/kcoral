import torch
import triton
import triton.language as tl

def get_pointer_configs():
    configs = []
    # Exhaustively search optimal tiles for an 8192x7168x5120 GEMM
    for block_m, block_n, block_k in [
        (128, 256, 64),
        (256, 128, 64),
        (128, 128, 64),
        (128, 256, 128),
        (256, 128, 128),
        (128, 128, 128)
    ]:
        for num_stages in [3, 4, 5]:
            for num_warps in [4, 8]:
                # Prune invalid combinations to save shared memory and compile time
                if num_warps == 4 and block_m * block_n >= 32768: 
                    continue
                if num_stages == 5 and block_k == 128: 
                    continue 
                
                # Exploit Hopper's Thread Block Clusters for L2 Cache locality sharing
                for num_ctas in [1, 2, 4]:
                    configs.append(triton.Config(
                        {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': 8},
                        num_stages=num_stages, num_warps=num_warps, num_ctas=num_ctas
                    ))
    return configs

@triton.autotune(
    configs=get_pointer_configs(),
    key=['M', 'N', 'K']
)
@triton.jit
def _pointer_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2-optimizing group swizzle layout
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Only M is dynamic, mask is only applied to the A operand.
    mask_m = offs_am < M
    
    # K is exactly 5120 and all configs have a BLOCK_K that gracefully divides 5120.
    num_k_tiles = K // BLOCK_K
    
    for k in tl.range(0, num_k_tiles):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        # N and K boundaries are guaranteed safe, stripping masking overhead entirely
        b = tl.load(b_ptrs) 
        
        # Hardware mathematically transposes b implicitly utilizing optimized shared memory fast paths
        acc = tl.dot(a, b.T, acc)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask_m[:, None])

def run(A, B, C):
    """
    Computes C = A @ B.T structurally bypassing TMA limits using an exhaustively tuned pointer implementation.
      A is of shape [M, 5120]
      B is of shape [7168, 5120]
      C is of shape [M, 7168]
    """
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    
    N = 7168
    K = 5120
    
    def grid_fn(META):
        num_pid_m = triton.cdiv(M, META['BLOCK_M'])
        num_pid_n = N // META['BLOCK_N']
        return (num_pid_m * num_pid_n,)
        
    _pointer_gemm_kernel[grid_fn](
        A, B, C,
        M,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        N=N, K=K
    )