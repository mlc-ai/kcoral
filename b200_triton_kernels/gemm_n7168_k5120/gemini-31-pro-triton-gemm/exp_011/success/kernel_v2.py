import torch
import triton
import triton.language as tl

# Provide infrastructure storage for device-created TMA descriptors.
# Safely allocated onto the global cuda context memory pool.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # 1. 256-width configs - Extremely high arithmetic intensity
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'WARP_SPECIALIZE': True, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'WARP_SPECIALIZE': True, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'WARP_SPECIALIZE': False, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'WARP_SPECIALIZE': False, 'GROUP_M': 8}, num_stages=4, num_warps=8),

        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'WARP_SPECIALIZE': True, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'WARP_SPECIALIZE': True, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'WARP_SPECIALIZE': False, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'WARP_SPECIALIZE': False, 'GROUP_M': 8}, num_stages=3, num_warps=8),

        # 2. Deep K-dimension pipelines (BLOCK_K=128 for rapid reductions)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'WARP_SPECIALIZE': True, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'WARP_SPECIALIZE': False, 'GROUP_M': 8}, num_stages=3, num_warps=8),

        # 3. Balanced configurations maximizing Stage masking
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'WARP_SPECIALIZE': True, 'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'WARP_SPECIALIZE': True, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'WARP_SPECIALIZE': False, 'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'WARP_SPECIALIZE': False, 'GROUP_M': 8}, num_stages=4, num_warps=8),

        # 4. Fallback configs (useful if cluster occupancy hits fragmentation bounds)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'WARP_SPECIALIZE': True, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'WARP_SPECIALIZE': False, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def _gemm_descriptor_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_bn, stride_cm,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    
    # 1. Device-created Descriptors
    # Enables hardware-backed Tensor Memory Accelerator (TMA) pathways seamlessly 
    # taking advantage of out-of-bounds pad zeroing automatically.
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

    # 2. Persistent Scheduling Loop
    # Assumes contiguous cycle distribution; leaps identically by available SM volume
    # establishing a bounded wave structure processing independent tiles.
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE,
    ):
        # 3. L2 Cache Swizzle Grouping
        # Re-maps tiles into compact groups so adjacent SMs frequently request overlapping datasets.
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

        # 4. Main Matrix Multiplication Pathway
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            # Leveraging physical `[BLOCK_N, BLOCK_K]` layouts from DRAM transposed smoothly via `.T`  
            # matching optimal Hopper tensor core matrix orientations seamlessly.
            acc = tl.dot(a, b.T, acc, out_dtype=tl.float32)

        c_desc.store([offset_m, offset_n], acc.to(dtype))

def run(A, B, C):
    """
    Computes generalized matrix multiplication C = A @ B.T writing directly into preallocated `C`.
    """
    M, K = A.shape
    N, _ = B.shape

    if M == 0 or N == 0 or K == 0:
        return

    torch.cuda.set_device(A.device)
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    # Cap launched execution scope to available streaming multiprocessors perfectly
    # avoiding warp scheduler thrashing on persistent pipelines.
    grid = lambda META: (min(num_sms, triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N'])), )

    _gemm_descriptor_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), B.stride(0), C.stride(0),
        NUM_SMS=num_sms
    )