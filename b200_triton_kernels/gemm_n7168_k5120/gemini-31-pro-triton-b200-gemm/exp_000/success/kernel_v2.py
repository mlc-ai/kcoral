import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

def pre_hook(kwargs):
    # Mutate host descriptors to match the autotuned block shapes for each specific trial configuration
    kwargs["a_desc"].block_shape = (kwargs["BLOCK_M"], kwargs["BLOCK_K"])
    kwargs["b_desc"].block_shape = (kwargs["BLOCK_N"], kwargs["BLOCK_K"])
    kwargs["c_desc"].block_shape = (kwargs["BLOCK_M"], kwargs["BLOCK_N"])

def get_configs():
    configs = []
    
    # Exhaustive tuning space for Blackwell SM100 architecture utilizing TMA and TMEM.
    # Note: We strictly avoid `num_ctas > 1` (Thread Block Clusters) in this standard Triton 
    # environment to prevent MLIR tcgen05 MMA distributed-shared-memory layout legalization failures.
    
    # Configurations with BLOCK_K=64 (allows deep pipelining with 4-5 stages while fitting in 228KB SMEM)
    for num_stages in [4, 5]:
        for gm in [8, 4]:
            configs.append(triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': gm, 'WARP_SPECIALIZE': True, 'NUM_STAGES': num_stages}, num_stages=num_stages, num_warps=8, pre_hook=pre_hook))
            configs.append(triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': gm, 'WARP_SPECIALIZE': True, 'NUM_STAGES': num_stages}, num_stages=num_stages, num_warps=8, pre_hook=pre_hook))
            configs.append(triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': gm, 'WARP_SPECIALIZE': True, 'NUM_STAGES': num_stages}, num_stages=num_stages, num_warps=4, pre_hook=pre_hook))
            configs.append(triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': gm, 'WARP_SPECIALIZE': True, 'NUM_STAGES': num_stages}, num_stages=num_stages, num_warps=8, pre_hook=pre_hook))

    # Configurations with BLOCK_K=128 (fewer stages to respect SMEM limits, yields wider MMA instructions)
    for num_stages in [3]:
        for gm in [8, 4]:
            # 128x128x128 securely fits inside 228KB limit with 3 stages (requires ~192KB)
            configs.append(triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': gm, 'WARP_SPECIALIZE': True, 'NUM_STAGES': num_stages}, num_stages=num_stages, num_warps=8, pre_hook=pre_hook))
            
    # Baseline non-warp-specialized configs (serves as robust fallbacks)
    configs.append(triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': False, 'NUM_STAGES': 4}, num_stages=4, num_warps=8, pre_hook=pre_hook))
    configs.append(triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPECIALIZE': False, 'NUM_STAGES': 4}, num_stages=4, num_warps=8, pre_hook=pre_hook))
    configs.append(triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPECIALIZE': False, 'NUM_STAGES': 3}, num_stages=3, num_warps=8, pre_hook=pre_hook))

    return configs

@triton.autotune(
    configs=get_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_kernel(
    a_desc, b_desc, c_desc,
    M, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, WARP_SPECIALIZE: tl.constexpr, NUM_STAGES: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Calculate dimension grid extents
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2 Cache swizzling for clustered SM locality
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Descriptor loads take direct 2D coordinate offsets indicating the block's top-left elements
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N

    # Native accumulation securely routed to Float32 Tensor Memory (TMEM) pathways
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Core unrolled loop harnessing TCGEN05 MMA mapped asynchronously over TMA memory operations
    for k0 in tl.range(0, k_tiles, num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([offs_am, k0 * BLOCK_K])
        b = b_desc.load([offs_bn, k0 * BLOCK_K])
        
        # Operand B is natively trans-read from its [N, K] layout directly into [K, N] during the dot product
        acc = tl.dot(a, b.T, acc, out_dtype=tl.float32)

    c = acc.to(tl.bfloat16)
    
    # TMA descriptors write directly into the destination storage natively handling edge bounds
    c_desc.store([offs_am, offs_bn], c)


def run(A, B, C):
    """
    Computes general matrix multiplication C = A @ B.T optimized for Blackwell.
    A: [M, K]
    B: [N, K] 
    C: [M, N]
    Data types are entirely bfloat16. N=7168, K=5120 inherently.
    """
    if A.numel() == 0 or B.numel() == 0:
        return

    torch.cuda.set_device(A.device)
    
    M, K = A.shape
    N = B.shape[0]
    
    # Emplace host TensorDescriptors. Dummy initial dimensions (128, 64/128) are constructed;
    # the autotune pre_hook applies precise configuration dimensionality per-kernel transparently.
    a_desc = TensorDescriptor.from_tensor(A, [128, 64])
    b_desc = TensorDescriptor.from_tensor(B, [128, 64])
    c_desc = TensorDescriptor.from_tensor(C, [128, 128])
    
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
        
    _gemm_kernel[grid](
        a_desc, b_desc, c_desc,
        M, N, K
    )