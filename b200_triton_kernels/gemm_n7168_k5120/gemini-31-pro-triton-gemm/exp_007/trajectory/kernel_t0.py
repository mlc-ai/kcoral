import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

def tma_pre_hook(args):
    """
    Hook to create host-side TensorDescriptors for TMA.
    Runs on the host before launching the kernel for a specific configuration.
    """
    A = args["A"]
    B = args["B"]
    C = args["C"]
    BLOCK_M = args["BLOCK_M"]
    BLOCK_N = args["BLOCK_N"]
    BLOCK_K = args["BLOCK_K"]
    
    args["desc_a"] = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    args["desc_b"] = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    args["desc_c"] = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])

def get_configs():
    """
    Return a list of configurations for autotuning, utilizing 
    the TMA host descriptor pre-hook.
    """
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3, pre_hook=tma_pre_hook),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3, pre_hook=tma_pre_hook),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4, pre_hook=tma_pre_hook),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4, pre_hook=tma_pre_hook),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4, pre_hook=tma_pre_hook),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4, pre_hook=tma_pre_hook),
    ]

@triton.autotune(
    configs=get_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_tma_kernel(
    desc_a, desc_b, desc_c,
    A, B, C,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Hopper-optimized GEMM kernel using standard host-side TensorDescriptors for TMA.
    Computes C = A @ B.T where B is physically stored as [N, K].
    """
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
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

    # Initialize FP32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for k_tile in range(0, num_k_tiles):
        offset_k = k_tile * BLOCK_K
        
        # TMA descriptor load handles boundaries autonomously
        a = desc_a.load([offset_m, offset_k])
        b = desc_b.load([offset_n, offset_k])
        
        # WGMMA lowering prefers passing explicitly transposed tile view matching physical layout
        acc = tl.dot(a, b.T, acc)

    # Convert accumulator down to destination dtype and trigger TMA bounds-safe store
    desc_c.store([offset_m, offset_n], acc.to(A.dtype.element_ty))

def run(A, B, C):
    """
    Entry point to compute C = A @ B.T into a preallocated CUDA tensor.
    N and K are known to be constant 7168 and 5120 respectively, while M can vary.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    if M == 0 or N == 0 or K == 0:
        return
        
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    
    _gemm_tma_kernel[grid](
        desc_a=None, desc_b=None, desc_c=None,
        A=A, B=B, C=C,
        M=M, N=N, K=K
    )