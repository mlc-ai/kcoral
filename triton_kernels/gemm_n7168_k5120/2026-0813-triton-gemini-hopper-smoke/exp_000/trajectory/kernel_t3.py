import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Cache for host TMA descriptors eliminating repeated Python/CPU overhead in benchmarking loops
_descriptor_cache = {}

def get_descriptor(tensor, block_shape):
    # A tensor's pointer, shape, and stride along with our mapped block shape serves as a safe cache key
    key = (tensor.data_ptr(), tensor.shape[0], tensor.shape[1], 
           tensor.stride(0), tensor.stride(1), block_shape[0], block_shape[1])
    if key not in _descriptor_cache:
        _descriptor_cache[key] = TensorDescriptor.from_tensor(tensor, block_shape)
    return _descriptor_cache[key]

def pre_hook(kwargs):
    kwargs['a_desc'] = get_descriptor(kwargs['A'], [kwargs['BLOCK_M'], kwargs['BLOCK_K']])
    kwargs['b_desc'] = get_descriptor(kwargs['B'], [kwargs['BLOCK_N'], kwargs['BLOCK_K']])

def get_autotune_config():
    configs = []
    for block_m, block_n, block_k, group_m, stages, warps in [
        # Large K for highest Hopper Tensor Core utilization limits
        (128, 256, 128, 8, 3, 8),
        (256, 128, 128, 8, 3, 8),
        (128, 128, 128, 8, 3, 8),
        (128, 128, 128, 8, 4, 8),
        (128, 128, 128, 8, 5, 8),
        # Larger M/N, smaller K
        (128, 256, 64, 8, 4, 8),
        (256, 128, 64, 8, 4, 8),
        (64, 256, 64, 8, 4, 8),
        (256, 64, 64, 8, 4, 8),
        # Varied Group sizes targeting optimal 50MiB L2 Cache reuse
        (128, 256, 128, 16, 3, 8),
        (256, 128, 128, 16, 3, 8),
        (128, 128, 128, 16, 4, 8),
        (128, 256, 128, 4, 3, 8),
    ]:
        configs.append(triton.Config(
            {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': group_m},
            num_stages=stages, num_warps=warps
        ))
    return configs

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
    pre_hook=pre_hook,
)
@triton.jit
def tma_gemm_kernel(
    A, B, C_ptr,
    a_desc, b_desc,
    M, N, K,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # 2D Grid L2 cache swizzling
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offset_k = k * BLOCK_K
        
        # TMA hardware auto bounds checking & loads
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        
        # WGMMA warp-group math - Logical transposition natively mirrors the memory
        acc = tl.dot(a, b.T, acc)
        
    c = acc.to(tl.bfloat16)
    
    # Standard pointer stores are fast and prevent Triton v3.7 descriptor padding faults
    offs_cm = offset_m + tl.arange(0, BLOCK_M)
    offs_cn = offset_n + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # Bypass masks if sizes represent a clean multiple 
    if (M % BLOCK_M == 0) and (N % BLOCK_N == 0):
        tl.store(c_ptrs, c)
    else:
        mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=mask)

def run(A, B, C):
    """
    General matrix multiply (GEMM) C = A @ B.T
    A: [M, K] -> bfloat16
    B: [N, K] -> bfloat16
    C: [M, N] -> bfloat16
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, K_B = B.shape
    
    # Short-circuit valid but empty projections
    if M == 0 or N == 0 or K == 0:
        return C
        
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    tma_gemm_kernel[grid](
        A=A, B=B, C_ptr=C,
        a_desc=None, b_desc=None,
        M=M, N=N, K=K,
        stride_cm=C.stride(0), stride_cn=C.stride(1)
    )
    
    return C