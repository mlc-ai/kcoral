import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

def get_autotune_config():
    configs = []
    # Explore a variety of block shapes and pipeline depths.
    # Hopper WGMMA performs best with large tiles and deeper pipelines.
    for num_stages in [3, 4, 5, 6]:
        for block_m, block_n, block_k, num_warps in [
            (256, 128, 64, 8),
            (128, 256, 64, 8),
            (256, 64, 64, 8),
            (64, 256, 64, 8),
            (128, 128, 128, 8),
            (128, 128, 64, 8),
            (128, 128, 64, 4),
            (128, 64, 128, 4),
            (64, 128, 128, 4),
            (256, 128, 128, 8),
            (128, 256, 128, 8),
        ]:
            # Estimate shared memory footprint: 2 bytes per element for bfloat16
            smem_size = (block_m * block_k + block_n * block_k) * 2 * num_stages
            # H100 provides up to ~228KB of usable shared memory per SM
            if smem_size <= 220000:
                configs.append(triton.Config(
                    {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': 8}, 
                    num_stages=num_stages, 
                    num_warps=num_warps
                ))
    return configs

def pre_hook(kwargs):
    """
    Hook to dynamically construct host TensorDescriptors before each kernel launch.
    This fulfills Hopper TMA requirement without repeated device-side allocation.
    """
    kwargs['a_desc'] = TensorDescriptor.from_tensor(kwargs['A'], [kwargs['BLOCK_M'], kwargs['BLOCK_K']])
    kwargs['b_desc'] = TensorDescriptor.from_tensor(kwargs['B'], [kwargs['BLOCK_N'], kwargs['BLOCK_K']])
    kwargs['c_desc'] = TensorDescriptor.from_tensor(kwargs['C'], [kwargs['BLOCK_M'], kwargs['BLOCK_N']])


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
    pre_hook=pre_hook,
    reset_to_zero=['C']
)
@triton.jit
def tma_gemm_kernel(
    A, B, C,
    a_desc, b_desc, c_desc,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2 Cache swizzling to group accesses for reuse
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        offset_k = k0 * BLOCK_K
        
        # Hardware-accelerated TMA loads (implicitly handles bounds masking & padding)
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        
        # Perform warp-group level dot. 'b.T' ensures continuous memory match for WGMMA right operand logic.
        acc = tl.dot(a, b.T, acc)
        
    c = acc.to(tl.bfloat16)
    
    # TMA hardware ignores out-of-bounds stores, eliminating the need for mask calculation.
    c_desc.store([offset_m, offset_n], c)

def run(A, B, C):
    """
    Compute C = A @ B.T directly into a preallocated tensor.
    A: [M, K]
    B: [N, K]
    C: [M, N]
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, K_B = B.shape
    
    if M == 0:
        return C
        
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    tma_gemm_kernel[grid](
        A=A, B=B, C=C,
        a_desc=None, b_desc=None, c_desc=None,
        M=M, N=N, K=K
    )
    
    return C