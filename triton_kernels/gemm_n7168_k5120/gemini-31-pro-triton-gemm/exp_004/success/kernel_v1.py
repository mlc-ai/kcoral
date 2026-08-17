import torch
import triton
import triton.language as tl

# Install Triton's descriptor allocator to support device-created descriptors for TMA
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_autotune_config():
    configs = []
    # Broad configuration search for Hopper SM90 TMA + WGMMA paths.
    for ws in [True, False]:
        for num_stages in [3, 4, 5]:
            for block_m, block_n, block_k, num_warps in [
                (128, 256, 64, 8),
                (256, 128, 64, 8),
                (128, 128, 128, 8),
                (128, 128, 64, 4),
                (64, 128, 64, 4),
            ]:
                configs.append(triton.Config({
                    'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 
                    'GROUP_M': 8, 'WARP_SPECIALIZE': ws
                }, num_stages=num_stages, num_warps=num_warps))
    return configs

@triton.autotune(
    configs=get_autotune_config(),
    key=['M'],
)
@triton.jit
def _descriptor_persistent_matmul(
    a_ptr, b_ptr, c_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    
    # Create TMA descriptors for Hopper natively on device
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero"
    )
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
        block_shape=[BLOCK_M, BLOCK_N]
    )

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_k_tiles = tl.cdiv(K, BLOCK_K)

    # Persistent matmul loop with L2-aware grouping and warp specialization
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE,
    ):
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        
        pid_in_group = tile_id % num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m

        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            
            # WGMMA memory operations via TMA descriptors
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            # b.T creates natively right-operand compatible layout mapping on SM90
            acc = tl.dot(a, b.T, acc)

        c_desc.store([offset_m, offset_n], acc.to(dtype))

def run(A, B, C):
    """
    General matrix multiply C = A @ B.T.
    A: [M, K]
    B: [N, K]
    C: [M, N]
    Both M, N, K mapping with N=7168 and K=5120.
    """
    if A.shape[0] == 0:
        return
        
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    
    # Constants from the problem definition
    N = 7168
    K = 5120
    
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    def grid_fn(META):
        num_pid_m = triton.cdiv(M, META['BLOCK_M'])
        num_pid_n = triton.cdiv(N, META['BLOCK_N'])
        num_tiles = num_pid_m * num_pid_n
        
        # Cap the grid at max parallel SM execution instances (persistent grid)
        return (min(num_sms, num_tiles), )
    
    _descriptor_persistent_matmul[grid_fn](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        NUM_SMS=num_sms
    )