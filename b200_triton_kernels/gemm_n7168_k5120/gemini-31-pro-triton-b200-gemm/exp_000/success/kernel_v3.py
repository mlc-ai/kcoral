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
    # Block sizes are strictly curated to remain within the 228KB Shared Memory limit per SM.
    # We omit thread block clusters (`num_ctas`) to avoid MLIR legalization faults on certain MMA shapes.
    
    for ws in [True, False]:
        for gm in [8]:
            # Ultra-large footprint (maximizes compute-to-memory ratio)
            # 256x256x64 uses ~64KB per stage. 2 stages = 128KB (Fits easily)
            configs.append(triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': gm, 'WARP_SPECIALIZE': ws, 'NUM_STAGES': 2}, num_stages=2, num_warps=8, pre_hook=pre_hook))
            
            # Large footprint with wider K
            # 128x256x128 uses ~96KB per stage. 2 stages = 192KB (Fits)
            configs.append(triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': gm, 'WARP_SPECIALIZE': ws, 'NUM_STAGES': 2}, num_stages=2, num_warps=8, pre_hook=pre_hook))
            configs.append(triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': gm, 'WARP_SPECIALIZE': ws, 'NUM_STAGES': 2}, num_stages=2, num_warps=8, pre_hook=pre_hook))
            
            # Medium footprint with wider K (allows deep pipelining)
            # 128x128x128 uses ~64KB per stage. 3 stages = 192KB (Fits)
            configs.append(triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': gm, 'WARP_SPECIALIZE': ws, 'NUM_STAGES': 3}, num_stages=3, num_warps=8, pre_hook=pre_hook))
            configs.append(triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': gm, 'WARP_SPECIALIZE': ws, 'NUM_STAGES': 2}, num_stages=2, num_warps=8, pre_hook=pre_hook))

            # Narrower K, allowing highest pipeline depth (4 stages)
            # 128x256x64 uses ~48KB per stage. 4 stages = 192KB (Fits)
            configs.append(triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': gm, 'WARP_SPECIALIZE': ws, 'NUM_STAGES': 4}, num_stages=4, num_warps=8, pre_hook=pre_hook))
            configs.append(triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': gm, 'WARP_SPECIALIZE': ws, 'NUM_STAGES': 4}, num_stages=4, num_warps=8, pre_hook=pre_hook))
            configs.append(triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': gm, 'WARP_SPECIALIZE': ws, 'NUM_STAGES': 3}, num_stages=3, num_warps=8, pre_hook=pre_hook))
            configs.append(triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': gm, 'WARP_SPECIALIZE': ws, 'NUM_STAGES': 3}, num_stages=3, num_warps=8, pre_hook=pre_hook))
            
            # Standard fast configurations
            configs.append(triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': gm, 'WARP_SPECIALIZE': ws, 'NUM_STAGES': 4}, num_stages=4, num_warps=8, pre_hook=pre_hook))
            configs.append(triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': gm, 'WARP_SPECIALIZE': ws, 'NUM_STAGES': 4}, num_stages=4, num_warps=4, pre_hook=pre_hook))
            configs.append(triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': gm, 'WARP_SPECIALIZE': ws, 'NUM_STAGES': 3}, num_stages=3, num_warps=4, pre_hook=pre_hook))
            
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
    
    # Grid Dimensions calculation
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2 Cache Swizzling for spatial locality across memory accesses
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Coordinate offsets indicating the top-left element for TMA block requests
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N

    # Floating point Native Accumulator allocated directly to Blackwell's Tensor Memory (TMEM) array
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Deeply pipelined loop seamlessly interleaving TMA memory fetches with asynchronous TCGEN05 MMA execution
    for k0 in tl.range(0, k_tiles, num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([offs_am, k0 * BLOCK_K])
        b = b_desc.load([offs_bn, k0 * BLOCK_K])
        
        # Operand B is effectively native-transposed from its original [N, K] topology implicitly 
        # into a [K, N] alignment required by mathematical dot operation constraints automatically.
        acc = tl.dot(a, b.T, acc)

    c = acc.to(tl.bfloat16)
    
    # 2D TMA Descriptor directly commits the epilogue securely guarding output tensor bounds natively
    c_desc.store([offs_am, offs_bn], c)


def run(A, B, C):
    """
    Computes general matrix multiplication C = A @ B.T highly optimized for Blackwell architectures.
    A: [M, K]
    B: [N, K] 
    C: [M, N]
    Data types are entirely bfloat16. Extents: K=5120, N=7168 inherently.
    """
    if A.numel() == 0 or B.numel() == 0:
        return

    torch.cuda.set_device(A.device)
    
    M, K = A.shape
    N = B.shape[0]
    
    # Construct persistent Host-side TensorDescriptors. Initial block-shape dimensionality (128x64) is simply a placeholder;
    # the triton `pre_hook` mechanism dynamically bridges these attributes during Autotune dispatch matching each trial cleanly.
    a_desc = TensorDescriptor.from_tensor(A, [128, 64])
    b_desc = TensorDescriptor.from_tensor(B, [128, 64])
    c_desc = TensorDescriptor.from_tensor(C, [128, 128])
    
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
        
    _gemm_kernel[grid](
        a_desc, b_desc, c_desc,
        M, N, K
    )