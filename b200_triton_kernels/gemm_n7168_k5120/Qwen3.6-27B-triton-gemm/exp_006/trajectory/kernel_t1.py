import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    A_desc_ptr,
    B_desc_ptr,
    C_desc_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # --- Grouped 1-D to 2-D tile mapping for L2 locality ---
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        offset_k = k_tile * BLOCK_K
        a_tile = A_desc_ptr.load([offset_m, offset_k])
        b_tile = B_desc_ptr.load([offset_k, offset_n])
        acc = tl.dot(a_tile, b_tile, acc=acc)

    C_desc_ptr.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute C = A @ B.T into preallocated output C."""
    torch.cuda.set_device(A.device)

    M = A.shape[0]
    N = B.shape[0]   # 7168
    K = B.shape[1]   # 5120

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    GROUP_M = 8

    # Build tensor descriptors for TMA on Hopper
    # A: [M, K], load blocks of [BLOCK_M, BLOCK_K]
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    # B_T: logically [K, N], load blocks of [BLOCK_K, BLOCK_N]
    # B is physically [N, K]; we build a view treating axis 1 (K) as rows, axis 0 (N) as cols
    b_t = B.T.contiguous()  # [K, N] for proper descriptor strides
    b_desc = TensorDescriptor.from_tensor(b_t, [BLOCK_K, BLOCK_N])
    # C: [M, N], store blocks of [BLOCK_M, BLOCK_N]
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])

    grid = (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
    )

    _gemm_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_warps=8,
        num_stages=4,
    )