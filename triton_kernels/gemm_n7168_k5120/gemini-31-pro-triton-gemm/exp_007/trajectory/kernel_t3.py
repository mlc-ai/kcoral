import torch
import triton
import triton.language as tl

# Set Triton's allocator to enable device-created TensorDescriptors.
# This infrastructure storage is required for `tl.make_tensor_descriptor`.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    """
    Exhaustive search space for Hopper SM90 TMA+WGMMA standard grid.
    Configurations are bounded by the 228KB shared memory limit per CTA.
    """
    configs = []
    combinations = [
        # Massive tiles, fewer stages (Memory footprint: ~192KB)
        (256, 128, 128, 8, 2),
        (128, 256, 128, 8, 2),
        
        # High compute density tiles (Memory footprint: ~192KB)
        (256, 128, 64,  8, 4),
        (128, 256, 64,  8, 4),
        
        # Balanced tiles (Memory footprint: ~192KB)
        (128, 128, 128, 8, 3),
        (128, 128, 128, 4, 3),
        
        # Deep pipeline for maximum latency hiding (Memory footprint: ~160KB - ~192KB)
        (128, 128, 64,  8, 5),
        (128, 128, 64,  8, 6),
        (128, 128, 64,  4, 5),
        (128, 128, 64,  4, 6),
        
        # Skinny tiles
        (64,  256, 64,  8, 4),
        (256, 64,  64,  8, 4),
        (64,  128, 128, 4, 3),
        (128, 64,  128, 4, 3),
        
        # Small tile fallback
        (64,  64,  128, 4, 5),
    ]
    
    for block_m, block_n, block_k, warps, stages in combinations:
        for group_m in [8, 4]:
            configs.append(triton.Config(
                {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': group_m},
                num_warps=warps, num_stages=stages
            ))
    return configs

@triton.autotune(
    configs=get_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_tma_full_grid_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    """
    Standard full-grid GEMM kernel leveraging TMA descriptors for reads and WGMMA for compute.
    C store uses vectorized pointer writes for maximal boundary safety without extra TMA overhead.
    """
    dtype = C_ptr.dtype.element_ty
    
    # TMA descriptor creation handles bounds automatically
    a_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[N, K],
        strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero"
    )

    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2-Aware Grouped Tile Ordering
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    # Accumulate natively in high precision FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for k_tile in tl.range(0, num_k_tiles):
        offset_k = k_tile * BLOCK_K
        
        # Async TMA hardware fetches
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        
        # B physical layout is [N, K]. Providing logical transpose directly feeds native Hopper WGMMA non-transposed fast path
        acc = tl.dot(a, b.T, acc)

    # Standard pointer-based write back for explicit vectorization control
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    offs_n = offset_n + tl.arange(0, BLOCK_N)
    
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    
    # Strictly mask output coordinates (although M, N are usually multiples of BLOCK for this shape)
    mask_m = offs_m < M
    mask_n = offs_n < N
    c_mask = mask_m[:, None] & mask_n[None, :]
    
    tl.store(c_ptrs, acc.to(dtype), mask=c_mask)

def run(A, B, C):
    """
    Destination-passing execute boundary. Computes C = A @ B.T without allocations.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    if M == 0 or N == 0 or K == 0:
        return
        
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    
    _gemm_tma_full_grid_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )