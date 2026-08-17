import torch
import triton
import triton.language as tl

def get_configs():
    configs = []
    # Exhaustive tuning space optimized for Blackwell TMEM/TMA path
    for ws in [True, False]:
        for stages in [3, 4, 5]:
            for block_m, block_n, block_k, warps in [
                (128, 256, 128, 8),
                (256, 128, 128, 8),
                (128, 128, 128, 4),
                (128, 128, 128, 8),
                (256, 64, 128, 8),
                (64, 256, 128, 8),
                (128, 64, 128, 4),
                (64, 128, 128, 4),
                (256, 128, 64, 8),
                (128, 256, 64, 8),
                (128, 128, 64, 4),
            ]:
                configs.append(triton.Config(
                    {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': 8, 'WARP_SPECIALIZE': ws},
                    num_warps=warps, num_stages=stages
                ))
    return configs

@triton.autotune(
    configs=get_configs(),
    key=["M"]
)
@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, WARP_SPECIALIZE: tl.constexpr,
):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    
    a_mask = offs_am[:, None] < M
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    for k0 in tl.range(0, k_tiles, warp_specialize=WARP_SPECIALIZE):
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptrs)
        
        acc = tl.dot(a_tile, b_tile.T, acc)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_mask = offs_am[:, None] < M
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)

def run(A, B, C):
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    if M == 0 or N == 0 or K == 0:
        return
        
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    
    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )