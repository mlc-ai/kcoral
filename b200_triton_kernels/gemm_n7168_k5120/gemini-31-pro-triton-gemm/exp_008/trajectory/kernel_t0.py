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
    # Retain WARP_SPECIALIZE=False as the correctness and performance baseline
    for block_m, block_n, block_k, warps, stages in [
        (128, 128, 64, 4, 3),
        (128, 256, 64, 8, 3),
        (256, 128, 64, 8, 3),
        (128, 128, 128, 8, 3),
        (128, 128, 64, 8, 4),
        (64, 128, 64, 4, 4),
    ]:
        configs.append(triton.Config(
            {
                'BLOCK_M': block_m, 
                'BLOCK_N': block_n, 
                'BLOCK_K': block_k, 
                'GROUP_M': 8, 
                'WARP_SPECIALIZE': False
            },
            num_warps=warps, 
            num_stages=stages
        ))
    
    # Test WARP_SPECIALIZE=True configuration space
    for block_m, block_n, block_k, warps, stages in [
        (128, 128, 64, 8, 3),
        (128, 256, 64, 8, 3),
        (256, 128, 64, 8, 3),
        (128, 128, 128, 8, 3),
    ]:
        configs.append(triton.Config(
            {
                'BLOCK_M': block_m, 
                'BLOCK_N': block_n, 
                'BLOCK_K': block_k, 
                'GROUP_M': 8, 
                'WARP_SPECIALIZE': True
            },
            num_warps=warps, 
            num_stages=stages
        ))
    return configs


@triton.autotune(
    configs=get_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _descriptor_persistent_matmul(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
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
    dtype = c_ptr.dtype.element_ty
    
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero",
    )
    # B is physically stored as [N, K], yielding a descriptor that loads [BLOCK_N, BLOCK_K] tiles
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
        # L2-Optimized Swizzle schedule mapping flat tile index to local geometry
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        tile_id_in_group = tile_id % num_pid_in_group
        
        pid_m = first_pid_m + (tile_id_in_group % group_size_m)
        pid_n = (tile_id_in_group // group_size_m)

        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            
            # Pipelined Hopper TMA descriptor loads
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            # WGMMA execution path over loaded descriptors
            acc = tl.dot(a, b.T, acc)

        # Downcast precision back to output target explicitly and perform asynchronous TMA write
        c_desc.store([offset_m, offset_n], acc.to(dtype))


def run(A, B, C):
    """
    Execute the matrix multiplication operation efficiently on Hopper hardware.
    Computes C = A @ B.T where A is of shape [M, K] and B is of shape [N, K].
    Outputs directly into the destination tensor C.
    """
    torch.cuda.set_device(A.device)
    
    M, K = A.shape
    N, K_B = B.shape
    assert K == K_B, "The inner-dimensions (K) of A and B must match."
    
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    # Cap the persistent execution grid to the available SMs, scaling minimally strictly where needed
    grid = lambda META: (min(num_sms, triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N'])), )
    
    _descriptor_persistent_matmul[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        NUM_SMS=num_sms,
    )