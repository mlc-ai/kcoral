import torch
import triton
import triton.language as tl

# Install Triton's descriptor allocator before the first launch.
# This infrastructure storage is required for device-side tl.make_tensor_descriptor.
def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

if not hasattr(triton, "_alloc_setup_done"):
    triton.set_allocator(_alloc_fn)
    triton._alloc_setup_done = True

def get_autotune_configs():
    configs = []
    # Sweeping the most dominant Hopper SM90 WGMMA combinations.
    # To hit ~95%+ of theoretical peak (beating cuBLAS by 1.2x), we must tune
    # across deep pipelines, Thread Block Clusters (num_ctas), and Warp Specialization.
    for ws in [True, False]:
        for ctas in [1, 2, 4]:
            for m, n, k, w, s in [
                # Maximum dense throughput tiles
                (256, 128, 64, 8, 3),
                (128, 256, 64, 8, 3),
                (256, 128, 64, 8, 4),
                (128, 256, 64, 8, 4),
                
                # Balanced tiles
                (128, 128, 64, 8, 3),
                (128, 128, 64, 4, 3),
                (128, 128, 64, 8, 4),
                (128, 128, 64, 4, 4),
                (128, 128, 64, 8, 5),
            ]:
                # Validate against Hopper H100 ~228 KiB usable shared memory limit
                shm_req = (m * k + n * k) * 2 * s
                if shm_req <= 228 * 1024:
                    configs.append(triton.Config(
                        {'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 'WARP_SPECIALIZE': ws},
                        num_stages=s, num_warps=w, num_ctas=ctas
                    ))
    return configs


@triton.autotune(
    configs=get_autotune_configs(),
    key=['M'],
)
@triton.jit
def _tma_persistent_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M,
    stride_am, stride_bn, stride_cm,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty

    # Create TMA descriptors on the device natively.
    # Since inner dimensions are contiguous based on requirements, inner strides are explicitly 1.
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, 1],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero"
    )
    # B is physically [N, K]. We load it as [BLOCK_N, BLOCK_K] chunks and logcially
    # transpose it during `tl.dot(a, b.T)` to optimally feed Hopper's tensor cores.
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[stride_bn, 1],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[stride_cm, 1],
        block_shape=[BLOCK_M, BLOCK_N]
    )

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_k_tiles = tl.cdiv(K, BLOCK_K)

    GROUP_M: tl.constexpr = 8
    num_pid_in_group = GROUP_M * num_pid_n

    # Hardware warp-specialized persistent loop scheduler
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE,
    ):
        # L2 cache swizzling
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        
        diff = num_pid_m - first_pid_m
        group_size_m = tl.where(diff < GROUP_M, diff, GROUP_M)
        
        local_id = tile_id % num_pid_in_group
        pid_m = first_pid_m + (local_id % group_size_m)
        pid_n = local_id // group_size_m
        
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        # K-loop driven entirely by TMA ops allowing aggressive async prefetching
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            acc = tl.dot(a, b.T, acc)

        # Safe boundaries TMA store
        c_desc.store([offset_m, offset_n], acc.to(dtype))


def run(A, B, C):
    """
    Computes C = A @ B.T where:
    A: [M, K]
    B: [N, K]
    C: [M, N]
    Where N is exactly 7168 and K is exactly 5120.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = 7168
    K = 5120
    
    # Restrict total grid dispatch to maximum available physical SM streaming map
    NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N'])), )
    
    _tma_persistent_gemm_kernel[grid](
        A, B, C,
        M,
        A.stride(0), B.stride(0), C.stride(0),
        N, K,
        NUM_SMS=NUM_SMS,
    )