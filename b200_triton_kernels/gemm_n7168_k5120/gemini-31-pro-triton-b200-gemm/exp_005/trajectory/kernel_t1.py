import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def desc_pre_hook(kwargs):
    A = kwargs['A']
    B = kwargs['B']
    C = kwargs['C']
    BLOCK_M = kwargs['BLOCK_M']
    BLOCK_N = kwargs['BLOCK_N']
    BLOCK_K = kwargs['BLOCK_K']
    
    kwargs['a_desc'] = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    kwargs['b_desc'] = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    kwargs['c_desc'] = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])

def get_configs():
    configs = []
    # (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages, warp_specialize)
    configurations = [
        # Maximize throughput with large blocks
        (256, 128, 128, 8, 3, False),
        (128, 256, 128, 8, 3, False),
        (256, 128, 128, 8, 3, True),
        (128, 256, 128, 8, 3, True),
        
        # 128x128 blocks 
        (128, 128, 128, 8, 3, False),
        (128, 128, 128, 8, 3, True),
        (128, 128, 128, 4, 3, False),
        (128, 128, 128, 4, 3, True),
        (128, 128, 128, 8, 4, False),
        (128, 128, 128, 8, 4, True),
        
        # Fallbacks
        (128, 128, 64, 4, 4, False),
        (128, 128, 64, 8, 4, False),
    ]
    for m, n, k, w, s, ws in configurations:
        configs.append(
            triton.Config(
                {'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 'GROUP_M': 8, 'WARP_SPECIALIZE': ws},
                num_stages=s, num_warps=w, pre_hook=desc_pre_hook
            )
        )
    return configs

@triton.jit
def _grouped_tile_coordinates(
    tile_id,
    num_pid_m,
    num_pid_n,
    GROUP_M: tl.constexpr,
):
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

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
    A, B, C,
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
    
    pid_m, pid_n = _grouped_tile_coordinates(tile_id, num_pid_m, num_pid_n, GROUP_M)
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k0 in tl.range(0, tl.cdiv(K, BLOCK_K), warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([offset_m, k0 * BLOCK_K])
        b = b_desc.load([offset_n, k0 * BLOCK_K])
        acc = tl.dot(a, b.T, acc, out_dtype=tl.float32)
        
    acc = acc.to(tl.bfloat16)
    c_desc.store([offset_m, offset_n], acc)

def run(A, B, C):
    """
    Computes C = A @ B.T.
    A: [M, K] in bfloat16
    B: [N, K] in bfloat16
    C: [M, N] in bfloat16 (preallocated destination)
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    _gemm_kernel[grid](
        A, B, C,
        None, None, None,
        M, N, K,
    )