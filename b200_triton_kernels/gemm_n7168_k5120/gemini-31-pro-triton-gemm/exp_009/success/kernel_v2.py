import torch
import triton
import triton.language as tl

# Standard Triton allocator required for device-created tensor descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # High performance WGMMA configs for Hopper
        # BLOCK_K=128 for maximal Tensor Core pipelining
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=2),
        
        # BLOCK_K=64 for larger spatial tiles or deeper pipelines (reduced SMEM usage per stage)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=4, num_stages=5),
        
        # Smaller spatial tiles for better occupancy and latency hiding on specific wave quantizations
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_warps=4, num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _descriptor_matmul_full_grid(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty

    # Create TMA descriptors on the device.
    # We dynamically construct these instead of using a pre-hook, enabling the
    # autotuner to flexibly explore block shapes while exploiting Hopper TMA hardware natively.
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

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2-cache friendly swizzling to group CTAs operating on similar regions of A/B
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m_actual = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m_actual)
    pid_n = (pid % num_pid_in_group) // group_size_m_actual

    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    num_k_tiles = tl.cdiv(K, BLOCK_K)

    # Standard loop software pipelined by Triton based on `num_stages`
    for k_tile in range(num_k_tiles):
        offset_k = k_tile * BLOCK_K
        
        # Asynchronously load tiles through TMA
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        
        # B physically is [N, K], loading [BLOCK_N, BLOCK_K] gives un-transposed shape.
        # Calling b.T conceptually reshapes it to [BLOCK_K, BLOCK_N] 
        # meeting requirements for C = A @ B.T.
        acc = tl.dot(a, b.T, acc)

    c_desc.store([offset_m, offset_n], acc.to(dtype))


def run(A, B, C):
    """
    Computes GEMM C = A @ B.T targeting the standard Hopper TMA & Tensor Core 
    using a full grid for maximal SM occupancy and L2 reuse.
    A: [M, K]
    B: [N, K]
    C: [M, N] (Destination passing)
    """
    if C.numel() == 0:
        return

    torch.cuda.set_device(A.device)
    M, K = A.shape
    N = C.shape[1]
    
    def grid_fn(META):
        num_pid_m = triton.cdiv(M, META['BLOCK_M'])
        num_pid_n = triton.cdiv(N, META['BLOCK_N'])
        return (num_pid_m * num_pid_n,)

    _descriptor_matmul_full_grid[grid_fn](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )