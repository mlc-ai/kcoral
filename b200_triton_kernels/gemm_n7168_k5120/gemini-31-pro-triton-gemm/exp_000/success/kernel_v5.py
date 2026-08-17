import torch
import triton
import triton.language as tl

# Set Triton allocator for device-side tensor descriptor creation.
# This infrastructure storage is necessary to utilize Hopper TMA lowerings natively.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # 1. WARP_SPECIALIZE=False 
        # Often out-performs True because of perfectly balanced warps and pure software pipelining
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': False}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': False}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPECIALIZE': False}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPECIALIZE': False}, num_stages=2, num_warps=8),
        
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPECIALIZE': False}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': False}, num_stages=5, num_warps=4),
        
        # 2. WARP_SPECIALIZE=True 
        # Hopper officially recommended path (enables producer/consumer warp split for TMA loads vs WGMMAs)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_stages=2, num_warps=8),
        
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_stages=4, num_warps=4),
    ],
    key=['M']
)
@triton.jit
def _tma_persistent_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M,
    stride_am, stride_bn, stride_cm,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    
    # Descriptors: By passing 1 directly, we statically guarantee to the compiler that inner strides are 1.
    # This matches PyTorch dense tensors exactly and safely ensures pure vectorized loads. 
    # TMA natively bounds-checks logically dynamically-sized leading edges.
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, 1], block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[stride_bn, 1], block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[stride_cm, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    num_pid_in_group = GROUP_M * num_pid_n

    for tile_id in tl.range(
        tl.program_id(0), 
        num_tiles, 
        NUM_SMS, 
        flatten=False, 
        warp_specialize=WARP_SPECIALIZE
    ):
        # L2 Cache SWIZZLING: Maps consecutive tiles to same B blocks
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        GROUP_M_ACTUAL = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % GROUP_M_ACTUAL)
        pid_n = (tile_id % num_pid_in_group) // GROUP_M_ACTUAL

        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            # WGMMA FP16/BF16 -> FP32 accumulation.
            # `b` is physically stored in [BLOCK_N, BLOCK_K]. 
            # Supplying `b.T` forces dot lowering seamlessly into optimal fast-paths mapped natively.
            acc = tl.dot(a, b.T, acc)

        c_desc.store([offset_m, offset_n], acc.to(dtype))


def run(A, B, C):
    """
    Computes general matrix multiplication C = A @ B.T thoroughly mapping out TMA configurations.
    Inputs:
      A: [M, K] bfloat16
      B: [N, K] bfloat16
    Outputs:
      C: [M, N] (Preallocated) bfloat16
    """
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = 7168
    K = 5120
    
    # Retrieve SM count to scale persistent grids perfectly bounding to actual SKUs optimally
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    def grid(META):
        num_pid_m = triton.cdiv(M, META['BLOCK_M'])
        num_pid_n = triton.cdiv(N, META['BLOCK_N'])
        num_tiles = num_pid_m * num_pid_n
        return (min(num_sms, num_tiles),)
        
    _tma_persistent_gemm_kernel[grid](
        A, B, C,
        M, 
        A.stride(0), B.stride(0), C.stride(0),
        N=N, K=K,
        NUM_SMS=num_sms,
    )