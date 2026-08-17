import torch
import triton
import triton.language as tl

# Set Triton's allocator to enable device-created TensorDescriptors.
# This is infrastructure storage for descriptors, not an output allocation.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    """
    Autotune configurations mapped for Hopper WGMMA and TMA.
    Configurations are carefully sized to fit within the ~228KB CTA shared memory limit.
    """
    return [
        # Warp specialized paths (Highly recommended for Hopper TMA/WGMMA)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_warps=4, num_stages=4),
        
        # Fallbacks (Without warp specialization)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPECIALIZE': False}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8, 'WARP_SPECIALIZE': False}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8, 'WARP_SPECIALIZE': False}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8, 'WARP_SPECIALIZE': False}, num_warps=4, num_stages=4),
    ]

@triton.autotune(
    configs=get_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_tma_persistent_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr, WARP_SPECIALIZE: tl.constexpr
):
    """
    Restricted Hopper warp-specialized descriptor loop.
    Leverages TMA and implicitly activates WGMMA instructions by passing `b.T`.
    """
    dtype = C_ptr.dtype.element_ty
    
    a_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[N, K],
        strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N]
    )

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Persistent loop utilizing hardware multithreading effectively across limited SMs
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
        
        # Accumulation heavily relies on FP32 precision internally
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            # Since physically B is [N, K], loading and performing the transposed dot handles Layout constraints naturally
            acc = tl.dot(a, b.T, acc)
            
        # TMA naturally ignores out-of-bounds writes without software masks
        c_desc.store([offset_m, offset_n], acc.to(dtype))

def run(A, B, C):
    """
    Public destination-passing execution function.
    Solves C = A @ B.T.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    if M == 0 or N == 0 or K == 0:
        return
        
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    # Grid configuration capped persistently by active SMS
    def grid_fn(META):
        num_tiles = triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"])
        return (min(num_sms, num_tiles),)
    
    _gemm_tma_persistent_kernel[grid_fn](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        NUM_SMS=num_sms
    )