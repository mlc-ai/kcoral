import torch
import triton
import triton.language as tl


# To use device-created descriptors for TMA, Triton requires an allocator setup.
def _descriptor_allocator(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_descriptor_allocator)


@triton.autotune(
    configs=[
        # Non-warp-specialized baselines
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "WARP_SPECIALIZE": False, "LOOP_STAGES": 3}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 4, "WARP_SPECIALIZE": False, "LOOP_STAGES": 3}, num_warps=4, num_stages=3),
        # Blackwell-only Warp Specialized targets
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "WARP_SPECIALIZE": True, "LOOP_STAGES": 4}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "WARP_SPECIALIZE": True, "LOOP_STAGES": 4}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"]
)
@triton.jit
def _gemm_tma_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    # L2-Aware Grouped tile mapping 
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
    
    # Base spatial offsets for our current Block
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    
    # TMA descriptors for A and B
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],         # Logically N x K storage, perfectly maps to B
        strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero"
    )
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Main K-loop utilizing TMA-backed async loads and Blackwell automatic warp specialization
    for k0 in tl.range(0, k_tiles, num_stages=LOOP_STAGES, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([offs_m, k0 * BLOCK_K])
        b = b_desc.load([offs_n, k0 * BLOCK_K])
        # By logical contract, dot accepts [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N] via transpose
        acc = tl.dot(a, b.T, acc)
        
    # Epilogue standard pointer store logic guarantees exact masking over variable partial boundaries (ex. variable M)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


def run(A, B, C):
    """
    Computes a General Matrix Multiply operation C = A @ B.T.
    Matches Qwen-3 projection workloads dynamically scaled constraints.
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    if M == 0 or N == 0 or K == 0:
        return
        
    # Standard linear single program 1D launch; grouped coordinates logic acts internally 
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    
    _gemm_tma_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )