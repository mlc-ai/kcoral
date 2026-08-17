import torch
import triton
import triton.language as tl

# Provide infrastructure storage for device-created TMA descriptors.
# Safely allocated onto the global cuda context memory pool.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    # Evaluate configuration payload against the physical limits of the target hardware
    if not torch.cuda.is_available():
        return []
        
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    
    configs = []
    
    # Tuple mapping: (BLOCK_M, BLOCK_N, BLOCK_K, min_warps_required)
    shapes = [
        # Extreme arithmetic intensity (requires dense register files via 8 warps)
        (128, 256, 64, 8),
        (256, 128, 64, 8),
        # Balanced profiles mapped for multi-CTA occupancy per SM
        (128, 128, 128, 4),
        (128, 128, 64, 4),
        (64, 128, 64, 4),
        (128, 64, 64, 4),
    ]
    
    for m, n, k, min_warps in shapes:
        for num_stages in [3, 4, 5]:
            for num_warps in [4, 8]:
                
                # Exclude configs that lack sufficient threads for large accumulator spaces
                if num_warps < min_warps:
                    continue
                
                # Calculate required shared memory per pipeline stage (2 bytes per bfloat16)
                shm_size = (m * k + n * k) * 2 * num_stages
                
                # Hopper SMs provide exactly 227 KB of usable shared memory max per CTA
                if shm_size <= 227000:
                    # Dynamically extract how many CTAs can concurrently fit perfectly onto a single SM
                    ctas_per_sm = 227000 // shm_size
                    # Limit to at most 4 CTAs per SM to avoid excessive register pressure
                    ctas_per_sm = min(ctas_per_sm, 4) 
                    grid_size = num_sms * ctas_per_sm
                    
                    for ws in [True, False]:
                        configs.append(triton.Config(
                            {
                                'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k,
                                'GRID_SIZE': grid_size, 'WARP_SPECIALIZE': ws, 'GROUP_M': 8
                            },
                            num_stages=num_stages,
                            num_warps=num_warps
                        ))
    return configs


@triton.autotune(
    configs=get_configs(),
    key=['M', 'N', 'K']
)
@triton.jit
def _gemm_descriptor_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_bn, stride_cm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    
    # 1. Device-created Descriptors
    # Implements Hopper's optimal path natively handling zero-padding of out-of-bounds bounds.
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, 1],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero"
    )
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
        block_shape=[BLOCK_M, BLOCK_N],
    )

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_k_tiles = tl.cdiv(K, BLOCK_K)

    # 2. Strict Persistent Matmul Loop Structure
    # Strides traversing perfectly distributed waves mapped specifically to SM occupancy densities.
    # Utilizing `flatten=False` and a constexpr step satisfies Triton 3.7+ Warp-Specialization standards.
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        GRID_SIZE,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE,
    ):
        
        # 3. L2 Cache Locality Swizzling
        # Re-maps linear progression into dense blocks so parallel adjacent SMs natively request intersecting B chunks.
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        rem_m = num_pid_m - first_pid_m
        
        group_size_m = tl.minimum(rem_m, GROUP_M)
        inner_id = tile_id % num_pid_in_group
        
        pid_m = first_pid_m + (inner_id % group_size_m)
        pid_n = inner_id // group_size_m

        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        # 4. WGMMA Mathematical Inner Loop
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            # WGMMA seamlessly reinterprets dynamically allocated B chunks natively via `.T`.
            acc = tl.dot(a, b.T, acc)

        c_desc.store([offset_m, offset_n], acc.to(dtype))


def run(A, B, C):
    """
    Computes a generalized hardware-accelerated GEMM C = A @ B.T.
    Output is populated natively into the preallocated destination tensor `C`.
    """
    M, K = A.shape
    N, _ = B.shape

    if M == 0 or N == 0 or K == 0:
        return

    torch.cuda.set_device(A.device)
    
    # Bounded safely dynamically limiting unnecessary warp scheduler overhead processing tails
    grid = lambda META: (min(META['GRID_SIZE'], triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N'])), )

    _gemm_descriptor_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), B.stride(0), C.stride(0)
    )