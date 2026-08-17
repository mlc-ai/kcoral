import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def config_pre_hook(kwargs):
    """
    Dynamically adjusts the TMA descriptor's physical block layouts for each corresponding 
    autotune permutation without generating manual device-descriptor recompilation costs.
    """
    kwargs["a_desc"].block_shape = (kwargs["BLOCK_M"], kwargs["BLOCK_K"])
    kwargs["b_desc"].block_shape = (kwargs["BLOCK_N"], kwargs["BLOCK_K"])
    kwargs["c_desc"].block_shape = (kwargs["BLOCK_M"], kwargs["BLOCK_N"])


def get_configs():
    """
    Systematically generates an exhaustive set of Blackwell-friendly tile mapping targets.
    """
    configs = []
    for ws in [True, False]:
        for block_m, block_n, block_k in [
            (256, 128, 128),
            (128, 256, 128),
            (128, 128, 128),
            (256, 128, 64),
            (128, 256, 64),
            (128, 128, 64),
            (256, 256, 64),
        ]:
            for group_m in [4, 8]:
                for num_warps in [4, 8]:
                    for num_stages in [3, 4, 5]:
                        # Pruning heuristics designed strictly to prevent compile/spill failures
                        if ws and num_warps == 8 and num_stages > 3:
                            continue
                        if block_k == 128 and num_stages > 4:
                            continue
                        if block_m == 256 and block_n == 256 and num_stages > 3:
                            continue
                        if block_m * block_n >= 65536 and num_warps == 4:
                            continue
                            
                        configs.append(triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k, 
                             "GROUP_M": group_m, "WARP_SPECIALIZE": ws, "LOOP_STAGES": num_stages},
                            num_warps=num_warps, num_stages=num_stages, pre_hook=config_pre_hook
                        ))
    return configs


@triton.autotune(
    configs=get_configs(),
    key=["M"]
)
@triton.jit
def _gemm_tma_kernel(
    a_desc, b_desc, c_desc,
    M, 
    N: tl.constexpr, 
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    tile_id = tl.program_id(0)
    
    # Grid dimensions resolve automatically at compile time when N is constexpr
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2 cache-aware grouped tile traversing map
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    
    # 2D scalar base coordinate mapped for native TMA boundaries
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    # The inner math pipe relies implicitly on standard 5th-Gen Tensor Core generation instructions 
    for k0 in tl.range(0, k_tiles, num_stages=LOOP_STAGES, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([offs_m, k0 * BLOCK_K])
        b = b_desc.load([offs_n, k0 * BLOCK_K])
        
        # Matrix B implicit layout mapping to expected orientation (B transpose internally loaded) 
        acc = tl.dot(a, b.T, acc)
        
    # Result natively routed; any unaligned bounds resolve harmlessly out-of-bounds in Blackwell hardware limits
    c_desc.store([offs_m, offs_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """
    Computes generalized scaled precision matrix multiplication C = A @ B.T.
    Targeting specific structural workloads mapped to Qwen-3 bounds properties.
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    if M == 0 or N == 0 or K == 0:
        return
        
    # Establish TMA alignment mapping templates utilizing Python. 
    # Shapes internally re-orchestrated at compile time directly by Triton's hook system.
    a_desc = TensorDescriptor.from_tensor(A, [128, 128])
    b_desc = TensorDescriptor.from_tensor(B, [128, 128])
    c_desc = TensorDescriptor.from_tensor(C, [128, 128])
    
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    
    _gemm_tma_kernel[grid](
        a_desc, b_desc, c_desc,
        M, N, K,
    )