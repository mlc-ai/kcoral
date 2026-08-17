import torch
import triton
import triton.language as tl

# Standard Blackwell allocator for device-created tensor descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_autotune_config():
    configs = []
    # Test across both specialized and non-specialized warp layouts 
    for ws in [True, False]:
        # Sweep standard and aggressive stage/warp configs, ensuring warps are a power of 2
        for stages in [3, 4, 5, 6]:
            for warps in [4, 8]:
                for block_m, block_n, block_k in [
                    (256, 128, 128),
                    (128, 256, 128),
                    (256, 128, 64),
                    (128, 256, 64),
                    (128, 128, 128),
                    (128, 128, 64),
                    (256, 64, 128),
                    (64, 256, 128),
                    (256, 256, 64),
                ]:
                    # Limit to ~227 KiB shared memory per block (Blackwell SM100 limits)
                    bytes_per_stage = (block_m * block_k + block_n * block_k) * 2
                    if bytes_per_stage * stages <= 220 * 1024:
                        configs.append(triton.Config(
                            {
                                'BLOCK_M': block_m, 
                                'BLOCK_N': block_n, 
                                'BLOCK_K': block_k, 
                                'GROUP_M': 8, 
                                'WARP_SPECIALIZE': ws
                            },
                            num_stages=stages, 
                            num_warps=warps
                        ))
    return configs

@triton.autotune(
    configs=get_autotune_config(),
    key=['M'],
)
@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr
):
    # Blackwell TMA tensor descriptors natively handle padding on boundary loads.
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )

    # Calculate grid position and L2 cache swizzling
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = num_pid_m - first_pid_m
    if group_size_m > GROUP_M:
        group_size_m = GROUP_M
        
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # N and K are fully known at compile-time for this problem definition
    k_tiles = K // BLOCK_K
    
    # Core loop utilizing automatic warp specialization (when configured)
    # Allows asynchronous overlapping of TMA fetched descriptors and TC MMA math operations.
    for k0 in tl.range(0, k_tiles, warp_specialize=WARP_SPECIALIZE):
        # TMA descriptor loads take scalar coordinates as offsets directly
        a = a_desc.load([offs_m, k0 * BLOCK_K])
        b = b_desc.load([offs_n, k0 * BLOCK_K])
        
        # Matrix B is logically (N, K). By loading a block of shape (BLOCK_N, BLOCK_K)
        # we maintain inner-dimension contiguity. b.T correctly presents it as (BLOCK_K, BLOCK_N).
        acc = tl.dot(a, b.T, acc)
        
    c = acc.to(tl.bfloat16)
    
    # Store epilogue via masked standard pointers to ensure compatibility and performance
    offs_cm = offs_m + tl.arange(0, BLOCK_M)
    offs_cn = offs_n + tl.arange(0, BLOCK_N)
    
    # Fast row-major memory writes
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    tl.store(c_ptrs, c, mask=c_mask)

def run(A, B, C):
    """
    Compute C = A @ B.T
    
    Args:
        A: Tensor of shape (M, K)
        B: Tensor of shape (N, K)
        C: Preallocated destination tensor of shape (M, N)
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    # Merge block shapes for flat 1D grid launch
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    # Pass N and K as kwargs for tl.constexpr evaluation
    gemm_kernel[grid](
        A, B, C,
        M,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        N=N, K=K
    )