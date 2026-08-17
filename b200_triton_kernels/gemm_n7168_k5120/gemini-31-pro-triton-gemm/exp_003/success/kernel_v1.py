import torch
import triton
import triton.language as tl

# Install Triton's descriptor allocator before the first launch.
# This is required for device-side tl.make_tensor_descriptor.
def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

if not hasattr(triton, "_alloc_setup_done"):
    triton.set_allocator(_alloc_fn)
    triton._alloc_setup_done = True

@triton.autotune(
    configs=[
        # Warp-specialized configs (typically use 8 warps for better throughput on Hopper)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_stages=4, num_warps=8),
        
        # Standard software-pipelined baselines
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': False}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': False}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPECIALIZE': False}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': False}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': False}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': False}, num_stages=4, num_warps=4),
    ],
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

    # Create TMA descriptors on the device for Hoppers
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero"
    )
    # B is physically [N, K], we'll load [BLOCK_N, BLOCK_K] blocks
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

    num_pid_in_group = GROUP_M * num_pid_n

    # Persistent loop stepped by NUM_SMS with optional compiler warp-specialization
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE,
    ):
        # L2 Cache Swizzle mapping
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

        # K-loop driven by TMA loads
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            # B block is [BLOCK_N, BLOCK_K], so we transpose it here logically
            acc = tl.dot(a, b.T, acc)

        # TMA store ignores out-of-bounds, no mask needed
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
    
    # Use real SM count for the persistent grid bound
    NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    # Cap grid size to NUM_SMS for the persistent scheduling
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