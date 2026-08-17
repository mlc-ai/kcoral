import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # Maximize MACs per cycle with large tiles (48KB/stage) - Max 4 stages to fit SM90 shared memory limits
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3, num_ctas=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4, num_ctas=2),
        
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3, num_ctas=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4, num_ctas=2),

        # High arithmetic intensity (64KB/stage) - Max 3 stages
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=3, num_ctas=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=3, num_ctas=4),

        # Standard blocks (32KB/stage) - Deep pipelines for latency hiding
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4, num_ctas=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=5, num_ctas=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4, num_ctas=1),
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
    # Modifies the output tile computation order so programs launched near each other 
    # reuse the same operand residing in the L2 cache, massively boosting cache hit rate.
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
    # For A @ B.T, we need B logically transposed to [K, N].
    # We explicitly construct pointers for a [BLOCK_K, BLOCK_N] tile.
    # Since stride_bk == 1, the fastest changing dimension in memory is mapped to the rows (offs_k).
    # Triton recognizes this strided load and optimally resolves it into a fast column-major 
    # shared memory layout, which natively matches Hopper WGMMA FP16/BF16 layout requirements 
    # without needing `.T` transposition instructions or padded TMA descriptors.
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulate in FP32 for precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for k_idx in range(num_k_tiles):
        # M is marked as "var", so A must be masked on the M dimension.
        # N=7168 and K=5120 are known exact multiples of all proposed block sizes,
        # meaning bounds checking for B can be completely omitted to save pipeline instructions.
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptrs) 
        
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    
    c_mask = offs_m[:, None] < M
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def run(A, B, C):
    """
    Compute C = A @ B.T into a preallocated CUDA tensor.
    
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