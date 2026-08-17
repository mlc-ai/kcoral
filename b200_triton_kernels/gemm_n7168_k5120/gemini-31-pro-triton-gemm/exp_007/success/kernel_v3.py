import torch
import triton
import triton.language as tl

# Set Triton's allocator to enable device-created TensorDescriptors.
# This serves as infrastructure storage for device-side descriptor creation,
# perfectly matching the Hopper TMA / Warp Specialized contract.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.jit
def _grouped_tile_coordinates(tile_id, num_pid_m, num_pid_n, GROUP_M: tl.constexpr):
    """
    Computes an L2-aware grouped grid mapping.
    Maps a linear sequence of program IDs to a 2D grid in groups of M tiles.
    """
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    return pid_m, pid_n

def get_configs():
    """
    Extensive set of carefully calculated configs optimized for Hopper SM90 TMA/WGMMA limits.
    Configurations are constrained by the ~228KB CTA shared memory limit and 255 regs/thread limit.
    """
    cfgs = [
        # Ultra-large blocks -> 1 Prog/SM, maximally exploits WGMMA with 8 warps
        (256, 128, 128, 8, 8, 2, 1),
        (128, 256, 128, 8, 8, 2, 1),
        (256, 128, 64,  8, 8, 3, 1),
        (128, 256, 64,  8, 8, 3, 1),
        
        # Standard large tiles -> 1 Prog/SM
        (128, 128, 128, 8, 8, 3, 1),
        (128, 128, 128, 8, 4, 3, 1),
        
        # Moderate density tiles mapping multiple programs (CTAs) per SM for immense latency hiding
        (128, 128, 64,  8, 4, 3, 2),
        (128, 128, 64,  8, 4, 4, 1),
        
        # Latency-hiding optimized narrower tiles
        (128, 64,  64,  8, 4, 4, 2),
        (64,  128, 64,  8, 4, 4, 2),
    ]

    configs = []
    # 132 acts as a very safe maximum SM count reference for H100 scheduling grids
    for m, n, k, g, w, s, p in cfgs:
        configs.append(triton.Config({
            'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 
            'GROUP_M': g, 'WARP_SPECIALIZE': True, 'NUM_PROGRAMS': p * 132
        }, num_warps=w, num_stages=s))
        
        # Baseline fallback without specialized instruction issuing
        configs.append(triton.Config({
            'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 
            'GROUP_M': g, 'WARP_SPECIALIZE': False, 'NUM_PROGRAMS': p * 132
        }, num_warps=w, num_stages=s))
    return configs

@triton.autotune(
    configs=get_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_tma_persistent_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_bn, stride_cm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_PROGRAMS: tl.constexpr, WARP_SPECIALIZE: tl.constexpr
):
    """
    Highly restrictive Hopper warp-specialized persistent loop kernel using Device-created Descriptors.
    Delegates all instruction partitioning, issue pipelining, and mbarriers explicitly to Triton.
    """
    dtype = C_ptr.dtype.element_ty
    
    # Implicitly align strides allowing TMA to fetch optimally
    a_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[M, K],
        strides=[stride_am, 1],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[N, K],
        strides=[stride_bn, 1],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[M, N],
        strides=[stride_cm, 1],
        block_shape=[BLOCK_M, BLOCK_N]
    )

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Persistent loop utilizing hardware multithreading effectively bounded by optimal execution cap
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_PROGRAMS,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE
    ):
        pid_m, pid_n = _grouped_tile_coordinates(tile_id, num_pid_m, num_pid_n, GROUP_M)
        
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        # Start high precision FP32 accumulator natively optimized onto Hopper Tensor Cores
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            # Since physically B is [N, K], transposing effectively evaluates layout logic to standard column-major internally
            # matching optimal WGMMA instruction mappings natively.
            acc = tl.dot(a, b.T, acc)
            
        # Bounds safe TMA Store silently prevents segfaults organically
        c_desc.store([offset_m, offset_n], acc.to(dtype))

def run(A, B, C):
    """
    Public destination-passing execution entry point.
    Computes C = A @ B.T robustly supporting any dynamic M sequence lengths.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    if M == 0 or N == 0 or K == 0:
        return
        
    def grid_fn(META):
        # Maps dynamically to the exact required tiles minimizing overheads while ensuring persistence limits
        num_tiles = triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"])
        progs = min(META["NUM_PROGRAMS"], num_tiles)
        return (progs,)
    
    _gemm_tma_persistent_kernel[grid_fn](
        A, B, C,
        M, N, K,
        A.stride(0), B.stride(0), C.stride(0)
    )