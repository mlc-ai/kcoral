import torch
import triton
import triton.language as tl

# Triton 3.7.1 global allocator setup for device-side descriptor creation.
# This provides infrastructure storage for the TMA descriptors.
def _descriptor_alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device=torch.cuda.current_device(), dtype=torch.int8)

triton.set_allocator(_descriptor_alloc_fn)

@triton.autotune(
    configs=[
        # Baseline software-pipelined configurations (WARP_SPECIALIZE=False)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64,  "GROUP_M": 8, "WARP_SPECIALIZE": False}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "WARP_SPECIALIZE": False}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "WARP_SPECIALIZE": False}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 128, "GROUP_M": 8, "WARP_SPECIALIZE": False}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 256, "GROUP_M": 8, "WARP_SPECIALIZE": False}, num_warps=4, num_stages=4),
        
        # Warp-specialized configurations for Hopper (WARP_SPECIALIZE=True)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64,  "GROUP_M": 8, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "WARP_SPECIALIZE": True}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _descriptor_persistent_matmul(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    GROUP_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    
    # Create device-side descriptors for the TMA hardware. 
    # B is physically [N, K], so its descriptor reflects that physical shape.
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

    # Persistent loop execution strategy capped by SMs
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE,
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
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            
            # Synchronous TMA loads with out-of-bounds protection handled by padding_option="zero"
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            # Since B was loaded as [BLOCK_N, BLOCK_K], transpose it here for the dot operand
            acc = tl.dot(a, b.T, acc)

        # Convert back to destination type before TMA store 
        c_desc.store([offset_m, offset_n], acc.to(dtype))


def run(A, B, C):
    """
    Compute C = A @ B.T into a preallocated CUDA tensor using standard Triton
    Tensor Descriptors mapped to Hopper TMA instructions.
    
    A: [M, K]
    B: [N, K]
    C: [M, N]
    """
    torch.cuda.set_device(A.device)
    if C.numel() == 0:
        return
        
    M, K = A.shape
    N, _ = B.shape
    
    # Target maximum concurrent CTAs across the entire device to feed the persistent kernel
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    def grid(META):
        num_pid_m = triton.cdiv(M, META['BLOCK_M'])
        num_pid_n = triton.cdiv(N, META['BLOCK_N'])
        num_tiles = num_pid_m * num_pid_n
        # Cap the grid to exactly what the device can hold concurrently
        return (min(num_sms, num_tiles), )

    _descriptor_persistent_matmul[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        NUM_SMS=num_sms,
    )