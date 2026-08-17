import torch
import triton
import triton.language as tl

# Install Triton's descriptor allocator to support device-created TMA descriptors.
# We map standard "cuda" device requests to the current active device context.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


def get_configs():
    configs = []
    # Exhaustive hardware-aligned tuning space for Hopper SM90 Tensor Cores.
    # We strictly bound shared memory below 227KB and accumulator registers below 128 per thread 
    # to avoid destructive register spilling while maximizing math intensity.
    valid_combinations = [
        # (block_m, block_n, block_k, warps, stages, group_m)
        
        # High Math Intensity (128 registers per thread, ~192KB SMEM)
        (256, 128, 64, 8, 4, 8),
        (256, 128, 64, 8, 4, 16),
        (256, 128, 64, 8, 3, 8),
        
        (128, 256, 64, 8, 4, 8),
        (128, 256, 64, 8, 4, 16),
        (128, 256, 64, 8, 3, 8),
        
        # Balanced (64 registers per thread, highly pipelined up to 6 stages)
        (128, 128, 64, 8, 6, 8),
        (128, 128, 64, 8, 5, 8),
        (128, 128, 64, 8, 4, 8),
        (128, 128, 64, 8, 4, 16),
        (128, 128, 64, 4, 4, 8),
        
        # K-Heavy execution (Maximizes TMA width)
        (256, 128, 128, 8, 2, 8),
        (128, 256, 128, 8, 2, 8),
        (128, 128, 128, 8, 3, 8),
        (128, 128, 128, 8, 3, 16),
    ]
    
    for m, n, k, w, s, gm in valid_combinations:
        # Hopper Native Warp-Specialized Pipelines (Best Performance Path)
        configs.append(triton.Config(
            {'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 'GROUP_M': gm, 'WARP_SPECIALIZE': True},
            num_warps=w, num_stages=s
        ))
        # Standard software-pipelined baseline
        configs.append(triton.Config(
            {'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 'GROUP_M': gm, 'WARP_SPECIALIZE': False},
            num_warps=w, num_stages=s
        ))
        
    return configs


@triton.autotune(
    configs=get_configs(),
    key=['M'],
)
@triton.jit
def _descriptor_persistent_matmul(
    a_ptr, b_ptr, c_ptr,
    M,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    # Lock mathematically constant dimensions natively into the hardware compilation path
    N: tl.constexpr = 7168
    K: tl.constexpr = 5120
    dtype = c_ptr.dtype.element_ty
    
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero",
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero",
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

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE,
    ):
        # L2-Optimized Swizzle schedule targeting Hopper's 50MB Cache
        # M-Fast Traversal locks B into the L2 working set tightly.
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        tile_id_in_group = tile_id % num_pid_in_group
        
        pid_m = first_pid_m + (tile_id_in_group % group_size_m)
        pid_n = (tile_id_in_group // group_size_m)

        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        # Full FP32 accumulation footprint
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            
            # Pipelined hardware TMA loads (Zero masked)
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            # Hardware WGMMA tensor core path mapped onto natively transposed load layouts
            acc = tl.dot(a, b.T, acc)

        # Precision downcast and native TMA store directly to VRAM bounds
        c_desc.store([offset_m, offset_n], acc.to(dtype))


def run(A, B, C):
    """
    Execute GEMM C = A @ B.T purely into the preallocated destination tensor.
    Leverages natively compiled constants and heavily optimized Hopper TMA hardware queues.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    # Cap the execution to perfectly saturate SM slots sequentially maximizing caching.
    grid = lambda META: (min(num_sms, triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(7168, META['BLOCK_N'])), )
    
    _descriptor_persistent_matmul[grid](
        A, B, C,
        M,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        NUM_SMS=num_sms,
    )