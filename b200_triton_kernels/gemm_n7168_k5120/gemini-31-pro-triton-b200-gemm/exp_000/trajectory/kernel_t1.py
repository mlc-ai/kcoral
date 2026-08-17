import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

def pre_hook(kwargs):
    # Mutate host descriptors to match the autotuned block shapes for this configuration
    kwargs["a_desc"].block_shape = (kwargs["BLOCK_M"], kwargs["BLOCK_K"])
    kwargs["b_desc"].block_shape = (kwargs["BLOCK_N"], kwargs["BLOCK_K"])
    kwargs["c_desc"].block_shape = (kwargs["BLOCK_M"], kwargs["BLOCK_N"])

@triton.autotune(
    configs=[
        # Highly aggressive Blackwell warp-specialized configs utilizing Tensor Memory (TMEM) and TMA
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPECIALIZE': True, 'NUM_STAGES': 3}, num_stages=3, num_warps=8, pre_hook=pre_hook),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPECIALIZE': True, 'NUM_STAGES': 3}, num_stages=3, num_warps=8, pre_hook=pre_hook),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPECIALIZE': True, 'NUM_STAGES': 4}, num_stages=4, num_warps=8, pre_hook=pre_hook),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPECIALIZE': True, 'NUM_STAGES': 4}, num_stages=4, num_warps=4, pre_hook=pre_hook),
        
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': True, 'NUM_STAGES': 4}, num_stages=4, num_warps=8, pre_hook=pre_hook),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': True, 'NUM_STAGES': 4}, num_stages=4, num_warps=8, pre_hook=pre_hook),

        # Non-warp-specialized robust baseline configurations
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': False, 'NUM_STAGES': 3}, num_stages=3, num_warps=8, pre_hook=pre_hook),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': False, 'NUM_STAGES': 3}, num_stages=3, num_warps=8, pre_hook=pre_hook),
    ],
    key=['M'],
)
@triton.jit
def _gemm_kernel(
    a_desc, b_desc, c_desc,
    M, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, WARP_SPECIALIZE: tl.constexpr, NUM_STAGES: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Calculate dimensions
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Swizzled program scheduling for better L2 cache locality
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Descriptor loads take direct 2D scalar offsets, bounded dynamically by TMA
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N

    # Float32 accumulation natively mapped to Tensor Cores
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Core loop utilizing automatic warp-specialization + pipelined TMA loading + MMA on SM100
    for k0 in tl.range(0, k_tiles, num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([offs_am, k0 * BLOCK_K])
        b = b_desc.load([offs_bn, k0 * BLOCK_K])
        
        # B is physically loaded as [BLOCK_N, BLOCK_K], transposed logically for dot
        acc = tl.dot(a, b.T, acc)

    c = acc.to(tl.bfloat16)
    
    # Descriptor stores safely ignore all out-of-bounds elements automatically
    c_desc.store([offs_am, offs_bn], c)


def run(A, B, C):
    """
    Computes C = A @ B.T.
    A: [M, K]
    B: [N, K] 
    C: [M, N]
    Data is preallocated bfloat16. K=5120, N=7168.
    """
    if A.numel() == 0 or B.numel() == 0:
        return

    torch.cuda.set_device(A.device)
    
    M = A.size(0)
    N = 7168
    K = 5120
    
    # Construct base host TensorDescriptors. 
    # Dummy block shapes (128, 64) are supplied; autotune pre_hook will precisely patch these 
    # for each compile configuration before they ever reach device execution.
    a_desc = TensorDescriptor.from_tensor(A, [128, 64])
    b_desc = TensorDescriptor.from_tensor(B, [128, 64])
    c_desc = TensorDescriptor.from_tensor(C, [128, 128])
    
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
        
    _gemm_kernel[grid](
        a_desc, b_desc, c_desc,
        M, N, K
    )