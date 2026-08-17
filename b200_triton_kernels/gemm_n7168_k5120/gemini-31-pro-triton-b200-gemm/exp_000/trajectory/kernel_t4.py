import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

def pre_hook(kwargs):
    # Mutate host descriptors to match the autotuned block shapes dynamically
    kwargs["a_desc"].block_shape = (kwargs["BLOCK_M"], kwargs["BLOCK_K"])
    kwargs["b_desc"].block_shape = (kwargs["BLOCK_N"], kwargs["BLOCK_K"])
    kwargs["c_desc"].block_shape = (kwargs["BLOCK_M"], kwargs["BLOCK_N"])

def get_configs():
    configs = []
    
    # High-performance shapes that fit in 228KB shared memory on Blackwell SM100
    # Avoid BLOCK_K=64 combined with any 256 dimension to prevent TCGEN05 MMA layout compiler errors.
    shapes = [
        (128, 256, 128, 2),
        (256, 128, 128, 2),
        (128, 128, 128, 3),
        (128, 128, 128, 4), # Triton will naturally prune if it marginally exceeds smem max
        (128, 128, 64, 4),
        (64, 128, 128, 3),
        (128, 64, 128, 3),
    ]
    
    # Sweep warps, warp specialization, and thread block cluster sizes (ctas)
    for bm, bn, bk, stages in shapes:
        for warps in [4, 8]:
            for ws in [True, False]:
                for ctas in [1, 2, 4, 8]:
                    configs.append(triton.Config(
                        {'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk, 'GROUP_M': 8, 'WARP_SPECIALIZE': ws, 'NUM_STAGES': stages},
                        num_stages=stages,
                        num_warps=warps,
                        num_ctas=ctas,
                        pre_hook=pre_hook
                    ))
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
    
    # Calculate grid bounds
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Swizzled schedule for L2 cache locality across Thread Block Clusters
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Descriptor loads take direct 2D coordinate offsets (scalar values representing the base of the block)
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N

    # Native accumulation in FP32 leveraging Blackwell's TCGEN05 and Tensor Memory (TMEM) pathways
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Warp Specialized, fully pipelined loop mapping asynchronous TMA and MMA
    for k0 in tl.range(0, k_tiles, num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([offs_am, k0 * BLOCK_K])
        b = b_desc.load([offs_bn, k0 * BLOCK_K])
        
        # B is physically loaded as [BLOCK_N, BLOCK_K], transposing logically for the dot operation to [BLOCK_K, BLOCK_N]
        acc = tl.dot(a, b.T, acc, out_dtype=tl.float32)

    c = acc.to(tl.bfloat16)
    
    # TMA descriptors write directly into the destination tensor, implicitly handling bounds checks
    c_desc.store([offs_am, offs_bn], c)


def run(A, B, C):
    """
    Computes general matrix multiplication C = A @ B.T.
    A: [M, K]
    B: [N, K] 
    C: [M, N]
    Data types are bfloat16. N=7168, K=5120.
    """
    if A.numel() == 0 or B.numel() == 0:
        return

    torch.cuda.set_device(A.device)
    
    M = A.size(0)
    N = 7168
    K = 5120
    
    # Construct base host TensorDescriptors. 
    # Dummy block shapes (128, 64) are provided; pre_hook patches these dynamically during autotune dispatch.
    a_desc = TensorDescriptor.from_tensor(A, [128, 64])
    b_desc = TensorDescriptor.from_tensor(B, [128, 64])
    c_desc = TensorDescriptor.from_tensor(C, [128, 128])
    
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
        
    _gemm_kernel[grid](
        a_desc, b_desc, c_desc,
        M, N, K
    )