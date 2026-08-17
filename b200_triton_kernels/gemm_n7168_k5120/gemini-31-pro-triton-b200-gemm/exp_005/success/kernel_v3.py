import torch
import triton
import triton.language as tl

# Set up the Triton allocator for fast device-side tensor descriptor creation
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    """
    Provide heavily optimized tuning configurations for SM100.
    Carefully limits `num_stages` based on `BLOCK_K` to stay within the ~228KB SMEM budget.
    """
    configs = []
    scenarios = [
        # (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages, warp_specialize, group_m)
        # Rectangular large tiles (Max SMEM limit safely accommodates 2 stages)
        (128, 256, 128, 8, 2, True, 8),
        (128, 256, 128, 8, 2, True, 4),
        (256, 128, 128, 8, 2, True, 8),
        (256, 128, 128, 8, 2, True, 4),
        
        (128, 256, 128, 8, 2, False, 8),
        (256, 128, 128, 8, 2, False, 8),
        
        # Symmetrical highly pipelined tiles
        (128, 128, 128, 8, 3, True, 8),
        (128, 128, 128, 8, 3, True, 4),
        (128, 128, 128, 8, 3, False, 8),
        (128, 128, 128, 4, 3, True, 8),
        (128, 128, 128, 4, 3, False, 8),
        
        # Deep pipelines with smaller K blocks
        (128, 128, 64, 8, 4, True, 8),
        (128, 128, 64, 8, 4, False, 8),
        (128, 128, 64, 8, 5, True, 8),
        (128, 128, 64, 4, 4, True, 8),
        
        # Fallbacks for extreme wave quantization cases
        (64, 128, 128, 4, 4, True, 8),
        (128, 64, 128, 4, 4, True, 8),
    ]
    for m, n, k, w, s, ws, gm in scenarios:
        configs.append(triton.Config(
            {'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 'GROUP_M': gm, 'WARP_SPECIALIZE': ws},
            num_stages=s, num_warps=w
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
    Maps the 1D launch grid into 2D blocks grouped by M.
    This greatly increases L2 cache hit rate for the loaded N (matrix B) tiles.
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
    key=['M'],
)
@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
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
    WARP_SPECIALIZE: tl.constexpr,
):
    # Construct TMA descriptors for hardware-accelerated memory boundaries
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    # Target memory physically exists as [N, K]. Descriptors track exactly this shape.
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
    
    # Calculate output offsets
    pid_m, pid_n = _grouped_tile_coordinates(tile_id, num_pid_m, num_pid_n, GROUP_M)
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    # Pre-allocate FP32 reduction buffer directly inside TMEM/Registers
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Unrollable compile-time K steps due to `K: tl.constexpr`
    k_tiles = tl.cdiv(K, BLOCK_K)
    for k0 in tl.range(0, k_tiles, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([offset_m, k0 * BLOCK_K])
        b = b_desc.load([offset_n, k0 * BLOCK_K])
        
        # B is transposed inline to resolve [BLOCK_M, K] x [K, BLOCK_N]
        acc = tl.dot(a, b.T, acc, out_dtype=tl.float32)
        
    # Standard bfloat16 downcasting handled strictly after completion of the accumulation loop
    acc = acc.to(tl.bfloat16)
    
    # Store directly by leveraging TMA descriptors
    c_desc.store([offset_m, offset_n], acc)

def run(A, B, C):
    """
    Destination-passing entry point for computing C = A @ B.T.
    Provided pre-allocated parameters natively fit host-based descriptors layout logic.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    # Inject constraint knowledge directly as `tl.constexpr` parameters for aggressive loop optimization
    N = 7168
    K = 5120
    
    # Standard 1D decoupled grid for L2-grouped traversal
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    _gemm_kernel[grid](
        A, B, C,
        M,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        N=N, K=K
    )