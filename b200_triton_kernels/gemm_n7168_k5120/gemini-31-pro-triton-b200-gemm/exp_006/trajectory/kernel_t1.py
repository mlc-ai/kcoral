import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def config_pre_hook(kwargs):
    """
    Dynamically adjusts the underlying TMA block shapes corresponding to each 
    benchmarked trial configuration during autotuning.
    """
    kwargs["a_desc"].block_shape = (kwargs["BLOCK_M"], kwargs["BLOCK_K"])
    kwargs["b_desc"].block_shape = (kwargs["BLOCK_N"], kwargs["BLOCK_K"])
    kwargs["c_desc"].block_shape = (kwargs["BLOCK_M"], kwargs["BLOCK_N"])


@triton.autotune(
    configs=[
        # --- Deeply Pipelined Warp-Specialized Configurations ---
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 16, "WARP_SPECIALIZE": True, "LOOP_STAGES": 4}, num_warps=8, num_stages=4, pre_hook=config_pre_hook),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 16, "WARP_SPECIALIZE": True, "LOOP_STAGES": 4}, num_warps=8, num_stages=4, pre_hook=config_pre_hook),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "WARP_SPECIALIZE": True, "LOOP_STAGES": 3}, num_warps=8, num_stages=3, pre_hook=config_pre_hook),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8, "WARP_SPECIALIZE": True, "LOOP_STAGES": 3}, num_warps=8, num_stages=3, pre_hook=config_pre_hook),
        
        # --- Ultra-Large Spatial Tiling ---
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64,  "GROUP_M": 8,  "WARP_SPECIALIZE": True, "LOOP_STAGES": 3}, num_warps=8, num_stages=3, pre_hook=config_pre_hook),
        
        # --- Balanced Blocks ---
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 16, "WARP_SPECIALIZE": True, "LOOP_STAGES": 4}, num_warps=8, num_stages=4, pre_hook=config_pre_hook),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8,  "WARP_SPECIALIZE": True, "LOOP_STAGES": 4}, num_warps=8, num_stages=4, pre_hook=config_pre_hook),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8,  "WARP_SPECIALIZE": True, "LOOP_STAGES": 3}, num_warps=8, num_stages=3, pre_hook=config_pre_hook),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_M": 8,  "WARP_SPECIALIZE": True, "LOOP_STAGES": 3}, num_warps=8, num_stages=3, pre_hook=config_pre_hook),
        
        # --- Standard Unspecialized Baselines ---
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8,  "WARP_SPECIALIZE": False, "LOOP_STAGES": 3}, num_warps=8, num_stages=3, pre_hook=config_pre_hook),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8,  "WARP_SPECIALIZE": False, "LOOP_STAGES": 3}, num_warps=8, num_stages=3, pre_hook=config_pre_hook),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8,  "WARP_SPECIALIZE": False, "LOOP_STAGES": 4}, num_warps=4, num_stages=4, pre_hook=config_pre_hook),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8,  "WARP_SPECIALIZE": False, "LOOP_STAGES": 3}, num_warps=4, num_stages=3, pre_hook=config_pre_hook),
    ],
    key=["M", "N", "K"]
)
@triton.jit
def _gemm_tma_kernel(
    a_desc, b_desc, c_desc,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    tile_id = tl.program_id(0)
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
    
    # 2D scalar base coordinate mapped for Host Descriptor constraints
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Central accumulation loop mapped exactly matching native Blackwell requirements
    for k0 in tl.range(0, k_tiles, num_stages=LOOP_STAGES, warp_specialize=WARP_SPECIALIZE):
        # TMA bounds masking completely mitigated locally
        a = a_desc.load([offs_m, k0 * BLOCK_K])
        b = b_desc.load([offs_n, k0 * BLOCK_K])
        
        # Operand B is mapped implicitly transposed logic via .T from load 
        acc = tl.dot(a, b.T, acc)
        
    # Result stored seamlessly via TMA with out-of-bounds elements bypassed intrinsically 
    c_desc.store([offs_m, offs_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """
    Computes generalized scaled precision matrix multiplication C = A @ B.T.
    Arguments are assumed compliant with the Qwen-3 projection sequence bounds constraints.
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    if M == 0 or N == 0 or K == 0:
        return
        
    # Host descriptors initialize hardware memory boundaries once via python overhead
    # Target configurations are manipulated and resolved entirely within our Config pre_hook
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