import torch
import triton
import triton.language as tl

# Set allocator for device-created descriptors as required by Triton's TMA implementation
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # --- Warp-specialized persistent configs (Device TMA + WGMMA) ---
        # Large tiles optimally sized to consume exactly 128 registers per thread for the accumulator
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_stages=4, num_warps=8),
        
        # Medium tiles balanced for deep pipelining
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_stages=5, num_warps=4),
        
        # Clustered configurations for hardware multicast and L2 locality
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_stages=3, num_warps=8, num_ctas=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': True}, num_stages=3, num_warps=8, num_ctas=2),
        
        # --- Baselines without warp specialization ---
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': False}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': False}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': False}, num_stages=4, num_warps=4),
    ],
    key=['M']
)
@triton.jit
def _hopper_gemm(
    a_ptr, b_ptr, c_ptr,
    M,
    stride_am, stride_bn, stride_cm,
    N: tl.constexpr, K: tl.constexpr, NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    
    # TMA descriptors natively map to Hopper WGMMA acceleration without runtime layout conversions
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
    num_pid_n = N // BLOCK_N
    num_k_tiles = K // BLOCK_K
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n
    
    start_pid = tl.program_id(0)
    
    # Persistent loop strictly bounded by the number of active SMs
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False, warp_specialize=WARP_SPECIALIZE):
        
        # Optimally swizzled group iteration traversing the L2 cache efficiently
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            # Utilizing mathematical b.T inside the dot aligns natively with Hopper physical fast paths
            acc = tl.dot(a, b.T, acc)

        c_desc.store([offset_m, offset_n], acc.to(dtype))

# Cache SM properties locally to minimize host overhead
_NUM_SMS_CACHE = {}

def run(A, B, C):
    """
    Computes C = A @ B.T directly into the pre-allocated C tensor.
    Exploits Hopper's TMA and WGMMA natively with static shapes and heavily tuned descriptors.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = 7168
    K = 5120
    
    device = A.device
    if device not in _NUM_SMS_CACHE:
        _NUM_SMS_CACHE[device] = torch.cuda.get_device_properties(device).multi_processor_count
    NUM_SMS = _NUM_SMS_CACHE[device]
    
    # Grid is completely persistent and perfectly caps at the active physical SM count
    def grid_fn(META):
        num_pid_m = triton.cdiv(M, META['BLOCK_M'])
        num_pid_n = N // META['BLOCK_N']
        num_tiles = num_pid_m * num_pid_n
        return (min(NUM_SMS, num_tiles),)
            
    _hopper_gemm[grid_fn](
        A, B, C,
        M,
        A.stride(0), B.stride(0), C.stride(0),
        N=N, K=K, NUM_SMS=NUM_SMS
    )