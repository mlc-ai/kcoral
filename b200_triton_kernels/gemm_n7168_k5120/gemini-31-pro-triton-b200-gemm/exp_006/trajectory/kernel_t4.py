import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def config_pre_hook(kwargs):
    """
    Surgically maps autotune-derived tile boundaries to the native TMA hardware
    descriptor's physical block layouts for zero-cost loop iterations.
    """
    kwargs["a_desc"].block_shape = (kwargs["BLOCK_M"], kwargs["BLOCK_K"])
    kwargs["b_desc"].block_shape = (kwargs["BLOCK_N"], kwargs["BLOCK_K"])
    kwargs["c_desc"].block_shape = (kwargs["BLOCK_M"], kwargs["BLOCK_N"])


def get_configs():
    configs = []
    # Exhaustively tuned for Blackwell B200 TMEM/TMA path limits
    # Structure: (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps, num_ctas, warp_specialize)
    candidates = [
        # --- Deeply Pipelined 256-Large Tiles --- 
        (256, 128, 128, 2, 8, 8, True),
        (256, 128, 128, 2, 12, 8, True),  # Broaden warps to evade register spills
        (256, 128, 128, 2, 8, 1, True),
        (256, 128, 64, 4, 8, 8, True),
        (256, 128, 64, 4, 12, 8, True),
        (256, 128, 64, 3, 8, 8, True),

        (128, 256, 128, 2, 8, 8, True),
        (128, 256, 128, 2, 12, 8, True),
        (128, 256, 128, 2, 8, 1, True),
        (128, 256, 64, 4, 8, 8, True),
        (128, 256, 64, 4, 12, 8, True),
        (128, 256, 64, 3, 8, 8, True),

        # --- Balanced Standard Tiles ---
        (128, 128, 128, 3, 8, 8, True),
        (128, 128, 128, 3, 4, 8, True),
        (128, 128, 128, 3, 8, 1, True),
        (128, 128, 128, 2, 8, 8, True),

        # --- Ultra-Large Orthogonal Tiles ---
        (256, 256, 64, 2, 8, 8, True),
        (256, 256, 64, 2, 12, 8, True),

        # --- Non-WS Baselines to guarantee fallback correctness ---
        (256, 128, 128, 2, 8, 8, False),
        (128, 256, 128, 2, 8, 8, False),
        (128, 128, 128, 3, 8, 8, False),
        (128, 128, 64, 3, 4, 1, False),
    ]

    for (m, n, k, stages, warps, ctas, ws) in candidates:
        configs.append(
            triton.Config(
                {
                    "BLOCK_M": m,
                    "BLOCK_N": n,
                    "BLOCK_K": k,
                    "GROUP_M": 8,
                    "WARP_SPECIALIZE": ws,
                    "LOOP_STAGES": stages,
                },
                num_warps=warps,
                num_stages=stages,
                num_ctas=ctas,
                pre_hook=config_pre_hook
            )
        )
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
    
    # 2D scalar base coordinate mapped strictly for native TMA boundaries
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    # The inner math pipe mapped matching strictly to native 5th-Gen Tensor Core layouts 
    for k0 in tl.range(0, k_tiles, num_stages=LOOP_STAGES, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([offs_m, k0 * BLOCK_K])
        b = b_desc.load([offs_n, k0 * BLOCK_K])
        
        # Matrix B implicitly load mapped to TMA expected spatial orientation (B transpose internally represented)
        acc = tl.dot(a, b.T, acc)
        
    # Standard hardware-routed resolution cleanly bypassing dynamic boundary conditions  
    c_desc.store([offs_m, offs_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """
    Computes generalized matrix multiplication C = A @ B.T in bfloat16.
    Targeting specific structural workloads mapped to Qwen-3 projection configurations.
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    if M == 0 or N == 0 or K == 0:
        return
        
    # Dummy setup boundaries generated sequentially purely for Triton Host hooking system.
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