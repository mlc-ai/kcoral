import torch
import triton
import triton.language as tl

def get_configs():
    configs = []
    for block_m, block_n, block_k, num_warps, num_stages in [
        (128, 256, 128, 8, 3),
        (256, 128, 128, 8, 3),
        (128, 256, 64, 8, 3),
        (256, 128, 64, 8, 3),
        (128, 128, 128, 8, 3),
        (128, 128, 64, 4, 4),
        (64, 128, 64, 4, 4),
        (128, 64, 64, 4, 4),
        (64, 64, 64, 4, 4),
    ]:
        configs.append(
            triton.Config(
                {
                    "BLOCK_M": block_m,
                    "BLOCK_N": block_n,
                    "BLOCK_K": block_k,
                    "GROUP_M": 8,
                },
                num_warps=num_warps,
                num_stages=num_stages,
            )
        )
    return configs

@triton.autotune(
    configs=get_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    # L2 cache swizzling for better data reuse
    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(grid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Logically A is [M, K]
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # Physically B is [N, K], we load [BLOCK_N, BLOCK_K] tile to ensure coalesced access
    b_ptrs = b_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k = k0 * BLOCK_K + offs_k
        
        # Masking guards against partial tiles in M, N, and K dimensions
        a_mask = (offs_m[:, None] < M) & (k[None, :] < K)
        b_mask = (offs_n[:, None] < N) & (k[None, :] < K)
        
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # We pass B as transposed to match physical layout & Blackwell tensor core optimizations
        acc = tl.dot(a_tile, b_tile.T, acc)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)

def run(A, B, C):
    """
    Computes a general matrix multiply C = A @ B.T where:
    A is logically [M, K]
    B is physically [N, K]
    C is logically [M, N]
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, K_b = B.shape
    
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    
    gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )