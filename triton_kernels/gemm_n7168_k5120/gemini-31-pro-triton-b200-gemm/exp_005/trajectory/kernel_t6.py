import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Define the Triton allocator for any required infrastructure storage
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def pre_hook(kwargs):
    """
    Hook to dynamically instantiate host TMA descriptors per configuration trial.
    This safely handles the changing block shapes during autotuning and entirely
    avoids device-side descriptor creation overhead.
    """
    kwargs['a_desc'] = TensorDescriptor.from_tensor(kwargs['A_ptr'], [kwargs['BLOCK_M'], kwargs['BLOCK_K']])
    # B is physically stored as [N, K]. We match this physical layout in the descriptor.
    kwargs['b_desc'] = TensorDescriptor.from_tensor(kwargs['B_ptr'], [kwargs['BLOCK_N'], kwargs['BLOCK_K']])
    kwargs['c_desc'] = TensorDescriptor.from_tensor(kwargs['C_ptr'], [kwargs['BLOCK_M'], kwargs['BLOCK_N']])

def get_configs():
    configs = []
    scenarios = [
        # Format: (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages, warp_specialize)
        
        # Maximize SMEM usage natively while safely staying under the 228KB limit per SM.
        # (128x256x128 block takes 96KB SMEM per stage -> 2 stages = 192KB < 228KB)
        (256, 128, 128, 8, 2, True),
        (128, 256, 128, 8, 2, True),
        (256, 128, 64, 8, 4, True),
        (128, 256, 64, 8, 4, True),
        
        # Balanced highly-pipelined tiles
        (128, 128, 128, 8, 3, True),
        (128, 128, 64, 8, 5, True),
        (128, 128, 64, 8, 4, True),
        
        # Baseline trials without automatic warp specialization
        (256, 128, 128, 8, 2, False),
        (128, 256, 128, 8, 2, False),
        (128, 128, 128, 8, 3, False),
        (128, 128, 64, 8, 4, False),
        
        # Smaller block sizes to act as safety fallbacks for corner case dimensions
        (128, 128, 64, 4, 4, True),
        (128, 128, 64, 4, 4, False),
        (64, 128, 128, 4, 4, True),
        (128, 64, 128, 4, 4, True),
    ]
    
    for m, n, k, w, s, ws in scenarios:
        configs.append(triton.Config(
            {'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 'GROUP_M': 8, 'WARP_SPECIALIZE': ws},
            num_stages=s, num_warps=w, pre_hook=pre_hook
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
    Groups M-tiles into batches to maximize L2 cache hit rates for the reused N-tiles.
    Uses `tl.where` to handle dynamic remainders reliably at runtime without Python conditionals.
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
    A_ptr, B_ptr, C_ptr,
    a_desc, b_desc, c_desc,
    M,
    N: tl.constexpr, 
    K: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, 
    WARP_SPECIALIZE: tl.constexpr,
):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m, pid_n = _grouped_tile_coordinates(tile_id, num_pid_m, num_pid_n, GROUP_M)
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    # Initialize high precision (FP32) accumulation block directly matching destination layout
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Pre-calculated constant K-tile extent allows optimal unrolling
    k_tiles = tl.cdiv(K, BLOCK_K)
    for k0 in tl.range(0, k_tiles, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([offset_m, k0 * BLOCK_K])
        b = b_desc.load([offset_n, k0 * BLOCK_K])
        
        # B is fetched as [BLOCK_N, BLOCK_K] matching storage and dynamically transposed inline
        acc = tl.dot(a, b.T, acc, out_dtype=tl.float32)
        
    # Cast exactly down to requested standard representation out-of-loop
    acc = acc.to(tl.bfloat16)
    
    # Directly store TMA bounds-checked result block
    c_desc.store([offset_m, offset_n], acc)

def run(A, B, C):
    """
    Computes generalized scaled precision destination-passing matrix multiplication
    Target: C = A @ B.T
    A: [M, 5120] (bfloat16)
    B: [7168, 5120] (bfloat16)
    C: [M, 7168] (bfloat16) -> (Supplied externally)
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    
    # Constant known scales from parameters to fully exploit compile-time optimization
    N = 7168
    K = 5120
    
    # Assign the standard grid scaling lambda
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    _gemm_kernel[grid](
        A, B, C,
        None, None, None, # Descriptor proxies properly handled automatically by pre_hook
        M,
        N, K
    )