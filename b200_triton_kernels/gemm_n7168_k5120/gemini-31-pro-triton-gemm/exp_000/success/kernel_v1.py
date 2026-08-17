import torch
import triton
import triton.language as tl

# Set allocator for Triton's device-side tensor descriptor creation.
# Required for TMA (Tensor Memory Accelerator) lowerings on Hopper.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # Warp specialized configs (leverage dedicated producer/consumer warps on Hopper)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'WARP_SPECIALIZE': True}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'WARP_SPECIALIZE': True}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'WARP_SPECIALIZE': True}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'WARP_SPECIALIZE': True}, num_stages=4, num_warps=4),
        
        # Standard configs (fallback and baseline for when warp specialization isn't optimal)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'WARP_SPECIALIZE': False}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'WARP_SPECIALIZE': False}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'WARP_SPECIALIZE': False}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'WARP_SPECIALIZE': False}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'WARP_SPECIALIZE': False}, num_stages=4, num_warps=4),
    ],
    key=['M']
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
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    
    # Device-created tensor descriptors for Hopper TMA
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

    # The outer persistent loop that distributes tasks natively to SMs
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE,
    ):
        # Apply L2-friendly swizzling if warp specialization is off.
        # Warp specialization compiler passes are restrictive regarding control flow and grouping math,
        # so we maintain strict simplicity when it is toggled on.
        if WARP_SPECIALIZE:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            GROUP_M: tl.constexpr = 8
            num_pid_in_group = GROUP_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = num_pid_m - first_pid_m
            GROUP_M_ACTUAL = tl.minimum(group_size_m, GROUP_M)
            
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % GROUP_M_ACTUAL)
            pid_n = (tile_id % num_pid_in_group) // GROUP_M_ACTUAL

        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        # Software pipelined load & compute via TMA + WGMMA
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            # The .T transpose is cost-free logistically as matrix dimensions support TMA
            acc = tl.dot(a, b.T, acc)

        c_desc.store([offset_m, offset_n], acc.to(dtype))

def run(A, B, C):
    """
    Computes general matrix multiplication C = A @ B.T mapping to Hopper optimized TMA structures.
    Inputs:
      A: [M, K]
      B: [N, K]
    Outputs:
      C: [M, N] (Preallocated)
    """
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = 7168
    K = 5120
    
    # Query streaming multiprocessors dynamically to clamp grid for ideal hardware scheduling efficiency
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    def grid(META):
        num_pid_m = triton.cdiv(M, META['BLOCK_M'])
        num_pid_n = triton.cdiv(N, META['BLOCK_N'])
        num_tiles = num_pid_m * num_pid_n
        return (min(num_sms, num_tiles),)
        
    _descriptor_persistent_matmul[grid](
        A, B, C,
        M,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        N=N, 
        K=K,
        NUM_SMS=num_sms,
    )