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
    # Exhaustive tuning space incorporating Hopper warp-specialization and standard software pipelining
    for ws in [True, False]:
        for num_stages in [3, 4, 5]:
            for block_m, block_n, block_k in [
                (128, 256, 64),
                (256, 128, 64),
                (128, 128, 128),
                (64, 256, 128),
                (256, 64, 128),
                (128, 64, 128),
                (64, 128, 128),
                (128, 128, 64),
                (64, 64, 128),
                (64, 64, 256),
                (128, 256, 128),
                (256, 128, 128),
            ]:
                # Ensure shared memory fits within H100 limits (~228 KiB usable)
                a_size = block_m * block_k * 2
                b_size = block_n * block_k * 2
                total_shm = (a_size + b_size) * num_stages
                if total_shm > 220 * 1024:
                    continue
                
                # Dedicated TMA and Math warps effectively utilize resources when chunking is adequate
                warps = 8 if block_m * block_n >= 128 * 128 else 4
                
                configs.append(triton.Config(
                    {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': 8, 'WARP_SPECIALIZE': ws},
                    num_stages=num_stages, num_warps=warps, num_ctas=1
                ))
                
                # Further expand config space for prime candidates with thread block clusters
                if block_m == 128 and block_n == 128 and num_stages in [3, 4]:
                    configs.append(triton.Config(
                        {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': 8, 'WARP_SPECIALIZE': ws},
                        num_stages=num_stages, num_warps=warps, num_ctas=2
                    ))
    return configs

@triton.autotune(
    configs=get_autotune_configs(),
    key=['M'],
)
@triton.jit
def _descriptor_persistent_matmul(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty

    # Create TMA descriptors on the device to feed the specialized loop natively
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero"
    )
    # B is physically [N, K]. Fetch it natively in row-major chunks, then logically transpose
    # during dot accumulation to provide column-major chunks for the SM90 hardware
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_k_tiles = tl.cdiv(K, BLOCK_K)

    num_pid_in_group = GROUP_M * num_pid_n

    # Hardware warp-specialized persistent loop pipeline pattern
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE,
    ):
        # L2 Cache Swizzle grouped by M
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        
        diff = num_pid_m - first_pid_m
        group_size_m = tl.minimum(diff, GROUP_M)
        
        local_id = tile_id % num_pid_in_group
        pid_m = first_pid_m + (local_id % group_size_m)
        pid_n = local_id // group_size_m
        
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        # Extended range floating point accumulation
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            
            # Sync-abstracted TMA Loads with Zero-Padding
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            # Compute partial GEMM: (BLOCK_M x BLOCK_K) @ (BLOCK_K x BLOCK_N)
            acc = tl.dot(a, b.T, acc)

        # Standard bounds-checked TMA Store formatting safely
        c_desc.store([offset_m, offset_n], acc.to(dtype))

def run(A, B, C):
    """
    Computes C = A @ B.T
    A: [M, K]
    B: [N, K]
    C: [M, N]
    where N is exactly 7168 and K is exactly 5120.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = 7168
    K = 5120
    
    # Query streaming multiprocessors dynamically to cap persistent scheduling efficiently
    NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    # Restrict total grid dispatch to maximum available physical SM processors execution map
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N'])), )
    
    _descriptor_persistent_matmul[grid](
        A, B, C,
        M,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        N, K,
        NUM_SMS=NUM_SMS,
    )