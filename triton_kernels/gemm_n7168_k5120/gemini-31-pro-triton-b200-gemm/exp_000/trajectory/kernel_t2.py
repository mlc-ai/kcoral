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
    # Exhaustive combination of optimal block shapes for SM100 limits (<= 228KB Shared Memory)
    shapes = [
        (128, 128, 128, 3), 
        (128, 128, 128, 2), 
        (128, 256, 128, 2), 
        (256, 128, 128, 2), 
        (256, 256, 64, 2),  
        (256, 256, 64, 3),  
        (128, 256, 64, 3),  
        (128, 256, 64, 4),  
        (256, 128, 64, 3),
        (256, 128, 64, 4),
    ]
    
    # We explore various warps, TMA cluster sizes, and Warp Specialization (WS)
    for (bm, bn, bk, stages) in shapes:
        for warps in [4, 8]:
            for ctas in [1, 2, 4, 8]:
                for ws in [True, False]:
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
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, WARP_SPECIALIZE: tl.constexpr, NUM_STAGES: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Calculate dimension grid
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Swizzled schedule for L2 cache locality over Thread Block Clusters
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Descriptor loads take direct 2D coordinate offsets 
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N

    # Native accumulation to float32 utilizing TCGEN05 and TMEM paths
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Warp Specialized, fully pipelined loop mapping TMA and MMA
    for k0 in tl.range(0, k_tiles, num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([offs_am, k0 * BLOCK_K])
        b = b_desc.load([offs_bn, k0 * BLOCK_K])
        
        # B is physically loaded as [BLOCK_N, BLOCK_K], transposed logically for dot ([BLOCK_K, BLOCK_N])
        acc = tl.dot(a, b.T, acc, out_dtype=tl.float32)

    c = acc.to(tl.bfloat16)
    
    # TMA descriptors write directly into the destination tensor safely avoiding OOB
    c_desc.store([offs_am, offs_bn], c)


def run(A, B, C):
    """
    Computes general matrix multiplication C = A @ B.T with PyTorch.
    A: [M, K]
    B: [N, K] 
    C: [M, N]
    Data types are bfloat16. K=5120, N=7168. 
    """
    if A.numel() == 0 or B.numel() == 0:
        return

    torch.cuda.set_device(A.device)
    
    M, K = A.shape
    N, _ = B.shape
    
    # Construct base host TensorDescriptors. 
    # Dummy shapes (128, 64/128) provided here; pre_hook patches these for each evaluated kernel trial.
    a_desc = TensorDescriptor.from_tensor(A, [128, 64])
    b_desc = TensorDescriptor.from_tensor(B, [128, 64])
    c_desc = TensorDescriptor.from_tensor(C, [128, 128])
    
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
        
    _gemm_kernel[grid](
        a_desc=a_desc, b_desc=b_desc, c_desc=c_desc,
        M=M, N=N, K=K
    )