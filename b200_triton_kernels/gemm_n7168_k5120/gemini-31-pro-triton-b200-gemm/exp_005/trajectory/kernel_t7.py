import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Configure the Triton allocator for any required infrastructure storage
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def pre_hook(kwargs):
    """
    Hook to dynamically instantiate host TMA descriptors per configuration trial.
    Overwrites the dummy descriptors passed from the host launcher with 
    the precisely tuned block shapes for each trial.
    """
    kwargs['a_desc'] = TensorDescriptor.from_tensor(kwargs['a_ptr'], [kwargs['BLOCK_M'], kwargs['BLOCK_K']])
    # B is physically stored as [N, K]. We match this physical layout in the descriptor.
    kwargs['b_desc'] = TensorDescriptor.from_tensor(kwargs['b_ptr'], [kwargs['BLOCK_N'], kwargs['BLOCK_K']])
    kwargs['c_desc'] = TensorDescriptor.from_tensor(kwargs['c_ptr'], [kwargs['BLOCK_M'], kwargs['BLOCK_N']])

def get_tma_configs():
    configs = []
    # Configurations mapped as: (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages, warp_specialize)
    for num_stages in [2, 3, 4, 5]:
        for block_m, block_n, block_k in [
            (256, 128, 64), (128, 256, 64), (128, 128, 64),
            (256, 128, 128), (128, 256, 128), (128, 128, 128)
        ]:
            # Calculate actual SMEM limit bounds to avoid violating SM100 limits (228 KiB).
            smem_per_stage = (block_m * block_k + block_n * block_k) * 2
            if smem_per_stage * num_stages > 228000:
                continue
                
            for num_warps in [4, 8]:
                for ws in [True, False]:
                    configs.append(triton.Config(
                        {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': 8, 'WARP_SPECIALIZE': ws},
                        num_stages=num_stages, num_warps=num_warps, pre_hook=pre_hook
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
    configs=get_tma_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_tma_kernel(
    a_ptr, b_ptr, c_ptr,
    a_desc, b_desc, c_desc,
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
    
    # Locate optimized group boundaries
    pid_m, pid_n = _grouped_tile_coordinates(tile_id, num_pid_m, num_pid_n, GROUP_M)
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    # Initialize high precision (FP32) accumulation block directly matching destination layout
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop across K domain utilizing hardware TMA loading capabilities natively
    k_tiles = tl.cdiv(K, BLOCK_K)
    for k0 in tl.range(0, k_tiles, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([offset_m, k0 * BLOCK_K])
        b = b_desc.load([offset_n, k0 * BLOCK_K])
        
        # B is fetched as [BLOCK_N, BLOCK_K] matching physical layout, properly transposed via compiler abstraction
        acc = tl.dot(a, b.T, acc, out_dtype=tl.float32)
        
    # Cast accumulator down to target precision domain entirely outside of reduction loop
    acc = acc.to(tl.bfloat16)
    
    # Store directly leveraging identical TMA validation
    c_desc.store([offset_m, offset_n], acc)

def run(A, B, C):
    """
    Computes generalized destination-passing matrix multiplication
    Target: C = A @ B.T
    A: [M, 5120] (bfloat16)
    B: [7168, 5120] (bfloat16)
    C: [M, 7168] (bfloat16) -> (Supplied externally)
    """
    torch.cuda.set_device(A.device)
    
    M, K = A.shape
    N, _ = B.shape
    
    # Construct base dummy descriptors resolving strict `NoneType` compilation errors.
    # The actual block sizes are subsequently substituted within the autotune `pre_hook`.
    dummy_a = TensorDescriptor.from_tensor(A, [16, 16])
    dummy_b = TensorDescriptor.from_tensor(B, [16, 16])
    dummy_c = TensorDescriptor.from_tensor(C, [16, 16])
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    _gemm_tma_kernel[grid](
        A, B, C,
        dummy_a, dummy_b, dummy_c,
        M, N, K
    )