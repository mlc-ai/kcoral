import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_tile_coordinates(
    tile_id,
    num_pid_m,
    num_pid_n,
    GROUP_M: tl.constexpr,
):
    """L2-aware mapping that keeps output tiles in localized blocks to reuse L2 resident operands."""
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    return pid_m, pid_n


@triton.autotune(
    configs=[
        # High arithmetic intensity shapes 
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        
        # Standard tile sizes with pipeline depth variations
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        
        # L2 grouped locality (clustered) options 
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4, num_ctas=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4, num_ctas=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4, num_ctas=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4, num_ctas=2),
    ],
    key=["M"],
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
    GROUP_M: tl.constexpr,
):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m, pid_n = _grouped_tile_coordinates(tile_id, num_pid_m, num_pid_n, GROUP_M)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # A block logically [BLOCK_M, BLOCK_K]
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    
    # -------------------------------------------------------------------------
    # Workaround for TMA `transposed = true` compilation failure with WGMMA:
    # Instead of requesting B as [BLOCK_N, BLOCK_K] and later writing `b.T` in `tl.dot`,
    # we explicitly ask for it physically as [BLOCK_K, BLOCK_N] from memory since
    # B has physical layout [N, K]. This causes it to become natively column-major.
    # WGMMA expects exactly this operand mapping, eliminating the transposition entirely!
    # -------------------------------------------------------------------------
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    mask_a = offs_m[:, None] < M
    
    # tl.range requests advanced software pipelining (Hopper async pipeline passes)
    for k_idx in tl.range(0, K // BLOCK_K):
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        
        # N=7168 and K=5120 are tightly divisible by 64/128/256 config parameters.
        # This completely omits inner-loop masking checks for B which increases execution throughput.
        b = tl.load(b_ptrs)
        
        # Accrue purely with native operand layouts
        acc = tl.dot(a, b, acc)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask_c = offs_m[:, None] < M
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask_c)


def run(A, B, C):
    torch.cuda.set_device(A.device)
    
    # Runtime variable bound constraints
    M = A.shape[0]
    
    # Constants specified by prompt contract constraints
    N = 7168
    K = 5120
    
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    
    _gemm_kernel[grid](
        A, B, C,
        M,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        N=N, K=K
    )