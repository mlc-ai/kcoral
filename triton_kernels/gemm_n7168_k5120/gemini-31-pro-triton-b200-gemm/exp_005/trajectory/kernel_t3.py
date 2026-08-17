import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Configure the Triton allocator for device-created structures if any
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def pre_hook(kwargs):
    """
    Hook to dynamically instantiate host TMA descriptors per configuration trial.
    This avoids device-side overhead and complies with Blackwell TMA constraints.
    """
    kwargs['a_desc'] = TensorDescriptor.from_tensor(kwargs['A_ptr'], (kwargs['BLOCK_M'], kwargs['BLOCK_K']))
    kwargs['b_desc'] = TensorDescriptor.from_tensor(kwargs['B_ptr'], (kwargs['BLOCK_N'], kwargs['BLOCK_K']))
    kwargs['c_desc'] = TensorDescriptor.from_tensor(kwargs['C_ptr'], (kwargs['BLOCK_M'], kwargs['BLOCK_N']))

def get_configs():
    configs = []
    scenarios = [
        # Configurations structured as: (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, warps, stages, warp_specialize)
        # SM100 max SMEM is ~228KB per SM.
        # S = (M*K + N*K) * 2 bytes. Total SMEM = S * stages.
        
        # Ultra large tiles
        # 256x256x64 (S = 64KB, max stages = 3)
        (256, 256, 64, 8, 8, 3, True),
        (256, 256, 64, 8, 8, 2, True),
        
        # High throughput rectangles
        # 256x128x128 or 128x256x128 (S = 96KB, max stages = 2)
        (256, 128, 128, 8, 8, 2, True),
        (256, 128, 128, 16, 8, 2, True),
        (128, 256, 128, 8, 8, 2, True),
        (128, 256, 128, 16, 8, 2, True),
        
        # 256x128x64 or 128x256x64 (S = 48KB, max stages = 4)
        (256, 128, 64, 8, 8, 4, True),
        (256, 128, 64, 8, 8, 3, True),
        (128, 256, 64, 8, 8, 4, True),
        (128, 256, 64, 8, 8, 3, True),
        
        # Medium symmetric tiles for high occupancy and latency hiding
        # 128x128x128 (S = 64KB, max stages = 3)
        (128, 128, 128, 8, 8, 3, True),
        (128, 128, 128, 8, 8, 2, True),
        (128, 128, 128, 8, 4, 3, True),
        (128, 128, 128, 8, 4, 2, True),
        
        # 128x128x64 (S = 32KB, max stages = 6)
        (128, 128, 64, 8, 4, 5, True),
        (128, 128, 64, 8, 4, 4, True),
        (128, 128, 64, 8, 8, 4, True),
        
        # Fallbacks (without warp specialization)
        (256, 128, 128, 8, 8, 2, False),
        (128, 256, 128, 8, 8, 2, False),
        (128, 128, 128, 8, 8, 3, False),
        (128, 128, 128, 8, 4, 3, False),
        (128, 128, 64, 8, 4, 4, False),
    ]
    for block_m, block_n, block_k, group_m, warps, stages, ws in scenarios:
        configs.append(triton.Config(
            {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': group_m, 'WARP_SPECIALIZE': ws},
            num_stages=stages, num_warps=warps, pre_hook=pre_hook
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
    Groups M-tiles together to maximize L2 cache reuse of N-tiles in the inner loop.
    Uses `tl.where` for safety within compiled JIT instead of native python functions.
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
    A_ptr, B_ptr, C_ptr,    # Actual tensors needed for mapping arguments to kwargs inside JIT logic
    a_desc, b_desc, c_desc, # Descriptors safely set up via `pre_hook` 
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Compute L2-optimized output tile mappings
    pid_m, pid_n = _grouped_tile_coordinates(tile_id, num_pid_m, num_pid_n, GROUP_M)
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    # Initialize FP32 accumulator matrix
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    k_tiles = tl.cdiv(K, BLOCK_K)
    for k0 in tl.range(0, k_tiles, warp_specialize=WARP_SPECIALIZE):
        # TMA fetching by leveraging host generated descriptors
        a = a_desc.load([offset_m, k0 * BLOCK_K])
        b = b_desc.load([offset_n, k0 * BLOCK_K])
        
        # B is transposed on the fly matching the logical storage format (Physically mapped: [N, K])
        acc = tl.dot(a, b.T, acc, out_dtype=tl.float32)
        
    # Standard bfloat16 downcasting explicitly handled after loop reduction
    acc = acc.to(tl.bfloat16)
    
    # Store directly by leveraging TMA descriptors
    c_desc.store([offset_m, offset_n], acc)

def run(A, B, C):
    """
    Destination-passing entry point for computing C = A @ B.T.
    Provided pre-allocated parameters natively fit host-based descriptors layout logic.
    """
    torch.cuda.set_device(A.device)
    
    M, K = A.shape
    N, _ = B.shape
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    # Run GPU grid logic properly delegating tensor metadata
    _gemm_kernel[grid](
        A, B, C,
        None, None, None,
        M, N, K
    )