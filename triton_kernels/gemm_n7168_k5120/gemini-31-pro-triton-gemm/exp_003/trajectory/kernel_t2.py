import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Update tensor descriptors dynamically during autotuning so that the TMA load 
# dimensions perfectly match the chosen BLOCK shapes.
def _update_desc(kwargs):
    kwargs['a_desc'] = TensorDescriptor.from_tensor(kwargs['A'], [kwargs['BLOCK_M'], kwargs['BLOCK_K']])
    kwargs['b_desc'] = TensorDescriptor.from_tensor(kwargs['B'], [kwargs['BLOCK_N'], kwargs['BLOCK_K']])
    kwargs['c_desc'] = TensorDescriptor.from_tensor(kwargs['C'], [kwargs['BLOCK_M'], kwargs['BLOCK_N']])

@triton.autotune(
    configs=[
        # Primary combinations optimized for Hopper WGMMAs and TMA latency windows
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=5, num_warps=8),
        
        # Reduced K-block sizes for more pipeline stages without exceeding shared memory bounds
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        
        # Narrower tile configurations
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M'],
    pre_hook=_update_desc,
)
@triton.jit
def _gemm_tma_kernel(
    A, B, C,
    a_desc, b_desc, c_desc,
    M, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr
):
    # Mapping configuration grid to spatial location
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2 swizzled grouping based on GROUP_M
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    
    diff = num_pid_m - first_pid_m
    group_size_m = tl.where(diff < GROUP_M, diff, GROUP_M)
    
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    # Highest precision accumulation map logic using tl.float32 for dot FP accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    # The inner MAC-loop driven by Hopper TMA block operations
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        offset_k = k0 * BLOCK_K
        
        # Synchronous abstraction of TMA. Automatically handles padding behavior safely.
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        
        # tl.dot leverages physical structure: B mapped as [BLOCK_N, BLOCK_K] transposed to logically match A
        acc = tl.dot(a, b.T, acc)
        
    # Standard format conversion immediately before TMA store
    c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))

def run(A, B, C):
    """
    Computes C = A @ B.T
    A: [M, K]
    B: [N, K]
    C: [M, N]
    where N is exactly 7168 and K is exactly 5120.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = 7168
    K = 5120
    
    # Placeholder descriptors initialization; actual parameters provided by `_update_desc` hook during tuning.
    a_desc = TensorDescriptor.from_tensor(A, [64, 64])
    b_desc = TensorDescriptor.from_tensor(B, [64, 64])
    c_desc = TensorDescriptor.from_tensor(C, [64, 64])
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    _gemm_tma_kernel[grid](
        A, B, C,
        a_desc, b_desc, c_desc,
        M, N, K
    )