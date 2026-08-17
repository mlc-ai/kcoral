import torch
import triton
import triton.language as tl

# Set Triton's allocator to enable device-created TensorDescriptors.
# This serves as infrastructure storage for device-side descriptor creation,
# exactly matching the Hopper TMA contract for Warp-Specialized persistent loops.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    """
    Extensive set of configs specifically tuned for Hopper SM90 TMA/WGMMA limits.
    Filters out configurations that exceed the ~228KB CTA shared memory limit.
    """
    configs = []
    
    # Base shapes to test (BLOCK_M, BLOCK_N, BLOCK_K, stages)
    shapes = [
        # Large tiles for max Tensor Core utilization
        (128, 256, 64, 3),
        (256, 128, 64, 3),
        (128, 128, 128, 3),
        
        # Deep pipeline for excellent latency hiding
        (128, 128, 64, 4),
        (128, 128, 64, 5),
        (128, 128, 64, 6),
        
        # Skinny tiles
        (64, 256, 64, 4),
        (256, 64, 64, 4),
        (64, 128, 128, 3),
        (128, 64, 128, 3),
        (64, 128, 128, 4),
        (128, 64, 128, 4),
        
        # Safe fallback
        (128, 128, 64, 3),
    ]
    
    for m, n, k, s in shapes:
        # Approximate shared memory usage for TMA (A, B, and C staging)
        # Hopper H100 CTA shared memory limit is ~228 KiB (233,472 bytes)
        shmem = 2 * (m * k + n * k) * s + (m * n * 2)
        if shmem > 233000:
            continue
            
        for group in [8]:
            for warp_spec in [True, False]:
                for w in [4, 8]:
                    configs.append(triton.Config(
                        {'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 
                         'GROUP_M': group, 'WARP_SPECIALIZE': warp_spec},
                        num_warps=w, num_stages=s
                    ))
    return configs

@triton.autotune(
    configs=get_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_tma_persistent_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_bn, stride_cm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr, WARP_SPECIALIZE: tl.constexpr
):
    """
    Standard Hopper warp-specialized descriptor loop.
    Maps an entire GEMM over exactly NUM_SMS persistent thread blocks.
    TMA loads directly from HBM to SMEM, driving Hopper WGMMA via dot(a, b.T).
    """
    dtype = C_ptr.dtype.element_ty
    
    a_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[M, K],
        strides=[stride_am, 1],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[N, K],
        strides=[stride_bn, 1],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[M, N],
        strides=[stride_cm, 1],
        block_shape=[BLOCK_M, BLOCK_N]
    )

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE
    ):
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
        
        # Accumulate natively on Tensor Cores in FP32
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            # Since physically B is [N, K], b.T generates a physical column-major tile
            # which seamlessly triggers the fastest Hopper WGMMA path.
            acc = tl.dot(a, b.T, acc)
            
        c_desc.store([offset_m, offset_n], acc.to(dtype))


def run(A, B, C):
    """
    Destination-passing execution entry point.
    Computes C = A @ B.T robustly across all sizes.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    if M == 0 or N == 0 or K == 0:
        return
        
    # Get exact physical SM count for persistent grid dimensioning
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    def grid_fn(META):
        num_tiles = triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"])
        # Cap grid at SMs precisely as specified by the Hopper recipe
        return (min(num_sms, num_tiles),)
    
    _gemm_tma_persistent_kernel[grid_fn](
        A, B, C,
        M, N, K,
        A.stride(0), B.stride(0), C.stride(0),
        NUM_SMS=num_sms
    )