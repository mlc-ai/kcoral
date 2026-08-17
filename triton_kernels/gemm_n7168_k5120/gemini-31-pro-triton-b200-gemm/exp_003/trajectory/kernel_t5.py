import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

def pre_hook(kwargs):
    """
    Hook to dynamically create host TMA descriptors right before kernel launch based on the 
    block dimensions selected by the autotuner.
    """
    bm = kwargs['BLOCK_M']
    bn = kwargs['BLOCK_N']
    bk = kwargs['BLOCK_K']
    
    # Instantiate TMA descriptors natively configured for Blackwell architecture bounds
    kwargs['a_desc'] = TensorDescriptor.from_tensor(kwargs['A'], [bm, bk])
    kwargs['b_desc'] = TensorDescriptor.from_tensor(kwargs['B'], [bn, bk])
    kwargs['c_desc'] = TensorDescriptor.from_tensor(kwargs['C'], [bm, bn])

def get_autotune_config():
    configs = []
    # Test CTAs clustering limits (L2 cache localization)
    for ctas in [1, 4]:
        for ws in [True, False]:
            for warps in [4, 8]:
                # Test the strongest throughput candidates tailored for Blackwell SM100 limits
                for block_m, block_n, block_k, stages in [
                    (128, 256, 64, 4),
                    (256, 128, 64, 4),
                    (128, 128, 128, 3),
                    (128, 128, 64, 5),
                    (128, 128, 64, 4),
                    (256, 256, 64, 2),
                ]:
                    # Filter configurations that would exceed Blackwell's max shared mem (227 KiB / SM)
                    bytes_per_stage = (block_m * block_k + block_n * block_k) * 2  # sizeof(bfloat16) == 2
                    if bytes_per_stage * stages <= 220 * 1024:
                        configs.append(triton.Config(
                            {
                                'BLOCK_M': block_m, 
                                'BLOCK_N': block_n, 
                                'BLOCK_K': block_k, 
                                'GROUP_M': 8, 
                                'WARP_SPECIALIZE': ws,
                                'NUM_STAGES': stages
                            },
                            num_stages=stages, 
                            num_warps=warps,
                            num_ctas=ctas,
                            pre_hook=pre_hook
                        ))
    return configs

@triton.autotune(
    configs=get_autotune_config(),
    key=['M'],
)
@triton.jit
def gemm_kernel(
    A, B, C,  # PyTorch tensors required for pre_hook metadata (Unused directly inside JIT)
    a_desc, b_desc, c_desc,
    M,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Software L2 Cache Swizzling (Group M-axis adjacent tiles together)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = num_pid_m - first_pid_m
    if group_size_m > GROUP_M:
        group_size_m = GROUP_M
        
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    
    # Initialize float32 accumulator (matches standard mixed precision PyTorch mechanics)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Native hardware automatic warp specialization (overlaps asynchronous TMA fetches with TC MMAs)
    for k0 in tl.range(0, k_tiles, num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        # TMA host descriptors take pure scalar coordinates
        a = a_desc.load([offs_m, k0 * BLOCK_K])
        b = b_desc.load([offs_n, k0 * BLOCK_K])
        
        # B physical shape is (N, K). By mapping dynamically to (BLOCK_K, BLOCK_N) via .T we
        # naturally align perfectly with Blackwell's standard orientation constraints natively mapping 
        acc = tl.dot(a, b.T, acc)
        
    # Downcast safely at completion of accumulation
    c = acc.to(tl.bfloat16)
    
    # Standard TMA epilogue storage bypassing manual masking boundary calculations
    c_desc.store([offs_m, offs_n], c)


def run(A, B, C):
    """
    Compute general matrix multiply: C = A @ B.T
    
    Args:
        A: (M, K) Source A
        B: (N, K) Source B
        C: (M, N) Preallocated destination tensor
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    # Base configuration uses flat 1D grid launch
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    gemm_kernel[grid](
        A=A, B=B, C=C,
        a_desc=None, b_desc=None, c_desc=None,
        M=M, N=N, K=K
    )