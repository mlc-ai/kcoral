import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # 8 warps for Hopper WGMMA, large rectangular tiles for max arithmetic intensity
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=4),
        
        # 4 warps, deep pipelines for latency hiding
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=5),
        
        # Standard balanced configs
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2-Aware Grouped Tile Ordering
    # Changes output tile computation order so programs launched near one another 
    # reuse the same operand from L2 cache.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # A is logically [M, K], physically [M, K]
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    
    # B is physically [N, K]. 
    # For A @ B.T, we need B.T which is logically [K, N].
    # Rather than loading it as [BLOCK_N, BLOCK_K] and transposing (which can conflict 
    # with TMA-backed layout expectations on Hopper for certain shapes), we directly 
    # construct pointers for a [BLOCK_K, BLOCK_N] tile to feed standard tl.dot seamlessly.
    # The row index in B.T is `offs_k` (column index in B).
    # The col index in B.T is `offs_n` (row index in B).
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulate in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for k_idx in range(num_k_tiles):
        # M is variable, so A needs masking on the M dimension.
        # N=7168 and K=5120 are known to be exact multiples of all proposed block sizes,
        # so we can omit bounds checking for B, saving instructions in the inner loop.
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptrs)
        
        acc = tl.dot(a, b, acc)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    
    # Store block - M is variable, N is constant and perfectly divisible by BLOCK_N
    c_mask = offs_m[:, None] < M
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)

def run(A, B, C):
    """
    Compute C = A @ B.T into a preallocated CUDA tensor using standard Triton.
    
    A: [M, K]
    B: [N, K]
    C: [M, N]
    """
    torch.cuda.set_device(A.device)
    if C.numel() == 0:
        return
        
    M, K = A.shape
    N, _ = B.shape
    
    def grid_fn(META):
        num_pid_m = triton.cdiv(M, META['BLOCK_M'])
        num_pid_n = triton.cdiv(N, META['BLOCK_N'])
        return (num_pid_m * num_pid_n,)

    _gemm_kernel[grid_fn](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )