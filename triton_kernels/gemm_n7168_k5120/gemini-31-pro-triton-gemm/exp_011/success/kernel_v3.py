import torch
import triton
import triton.language as tl

# Provide infrastructure storage for device-created TMA descriptors.
# Safely allocated onto the global cuda context memory pool.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    configs = []
    # Curated configurations targeting a mix of high arithmetic intensity (256x128) 
    # and maximum SM CTA occupancy (128x128 fits 2 CTAs per SM).
    tile_shapes = [
        (256, 128, 64),
        (128, 256, 64),
        (128, 128, 128),
        (128, 128, 64),
        (64, 128, 128),
        (128, 64, 128),
        (64, 128, 64),
        (128, 64, 64),
    ]
    for block_m, block_n, block_k in tile_shapes:
        for num_stages in [3, 4, 5]:
            for num_warps in [4, 8]:
                # Hopper SMs provide 227KB of usable shared memory max per CTA
                # TMA utilizes shared memory heavily to pipeline block loads (2 bytes per bfloat16)
                shm_size = (block_m * block_k + block_n * block_k) * 2 * num_stages
                
                if shm_size <= 227000:
                    configs.append(triton.Config(
                        {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': 8},
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
    GROUP_M: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    
    # 1. Device-created Descriptors
    # Harnessing TMA completely avoids index arithmetic overhead and cleanly pads zero-bounds.
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

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # 2. L2 Cache Locality Grouping
    # Re-maps tiles into compact groups so adjacent SMs frequently request overlapping datasets.
    if GROUP_M > 0:
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        rem_m = num_pid_m - first_pid_m
        
        # Determine valid grid boundary size dynamically
        group_size_m = tl.minimum(rem_m, GROUP_M)
        inner_id = pid % num_pid_in_group
        
        pid_m = first_pid_m + (inner_id % group_size_m)
        pid_n = inner_id // group_size_m
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    num_k_tiles = tl.cdiv(K, BLOCK_K)

    # 3. Main WGMMA Mathematical Reduction Loop
    for k_tile in range(num_k_tiles):
        offset_k = k_tile * BLOCK_K
        
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        
        # Leveraging physical `[BLOCK_N, BLOCK_K]` layouts from DRAM transposed smoothly via `.T`  
        # matching optimal Hopper tensor core matrix orientations seamlessly.
        acc = tl.dot(a, b.T, acc, out_dtype=tl.float32)

    # 4. Result Finalization via TMA
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
    
    # Full grid launch. Distributes freely allowing Hopper CTA hardware scheduler 
    # to perfectly weave overlapping executions across physical SMs if capacity suffices.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )

    _gemm_descriptor_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), B.stride(0), C.stride(0)
    )