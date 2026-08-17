import torch
import triton
import triton.language as tl

# Configure the Triton allocator for device-side descriptor creation
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    configs = []
    scenarios = [
        # Large tiles for maximum compute throughput
        (256, 128, 128, 8, 2, True),
        (256, 128, 128, 8, 2, False),
        (128, 256, 128, 8, 2, True),
        (128, 256, 128, 8, 2, False),
        
        (256, 256, 64, 8, 3, True),
        (256, 256, 64, 8, 3, False),
        (256, 256, 64, 8, 2, True),
        
        (256, 128, 64, 8, 3, True),
        (256, 128, 64, 8, 3, False),
        (128, 256, 64, 8, 3, True),
        (128, 256, 64, 8, 3, False),
        
        # Medium tiles for higher occupancy and latency hiding
        (128, 128, 128, 8, 3, True),
        (128, 128, 128, 8, 3, False),
        (128, 128, 128, 8, 4, True),
        (128, 128, 128, 8, 4, False),
        
        (128, 128, 64, 8, 4, True),
        (128, 128, 64, 8, 4, False),
        (128, 128, 64, 8, 5, True),
        
        # Fallbacks with fewer warps
        (128, 128, 64, 4, 3, False),
    ]
    for block_m, block_n, block_k, warps, stages, ws in scenarios:
        configs.append(triton.Config(
            {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': 8, 'WARP_SPECIALIZE': ws},
            num_stages=stages, num_warps=warps
        ))
    return configs

@triton.jit
def _grouped_tile_coordinates(
    tile_id,
    num_pid_m,
    num_pid_n,
    GROUP_M: tl.constexpr,
):
    """
    Groups M-tiles to maximize L2 cache reuse of the N-tiles.
    Uses tl.where instead of min() to safely handle dynamic bounds.
    """
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    rem_m = num_pid_m - first_pid_m
    group_size_m = tl.where(rem_m < GROUP_M, rem_m, GROUP_M)

    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    return pid_m, pid_n

@triton.autotune(
    configs=get_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    # Device-side descriptors take advantage of standard padding
    # zero-padding safely pads the matrix bounds for TMA load operations.
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    # B is physically stored as [N, K], so we load blocks of [BLOCK_N, BLOCK_K]
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N], padding_option="zero"
    )

    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Compute L2-optimized output tile
    pid_m, pid_n = _grouped_tile_coordinates(tile_id, num_pid_m, num_pid_n, GROUP_M)
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    # Accumulate directly in fp32 for bfloat16 inputs
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    k_tiles = tl.cdiv(K, BLOCK_K)
    for k0 in tl.range(0, k_tiles, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([offset_m, k0 * BLOCK_K])
        b = b_desc.load([offset_n, k0 * BLOCK_K])
        # b is transposed within the dot to align properly: [BLOCK_M, K] @ [K, BLOCK_N]
        acc = tl.dot(a, b.T, acc, out_dtype=tl.float32)
        
    # Convert accumulated FP32 sums strictly before storing
    acc = acc.to(tl.bfloat16)
    c_desc.store([offset_m, offset_n], acc)

def run(A, B, C):
    """
    Destination-passing entry point for computing C = A @ B.T
    
    Tensors:
      A: [M, K] (bfloat16)
      B: [N, K] (bfloat16)
      C: [M, N] (bfloat16) - Output tensor provided preallocated
    """
    torch.cuda.set_device(A.device)
    
    M, K = A.shape
    N, _ = B.shape
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )