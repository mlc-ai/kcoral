import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _grouped_tile_coordinates(
    tile_id,
    num_pid_m,
    num_pid_n,
    GROUP_M: tl.constexpr,
):
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    return pid_m, pid_n


@triton.autotune(
    configs=[
        # Various tile shapes with GROUP_M=1 (default row-major)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 1},
                      num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 1},
                      num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 1},
                      num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 1},
                      num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 1},
                      num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 1},
                      num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 1},
                      num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 1},
                      num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 1},
                      num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 1},
                      num_warps=8, num_stages=3),
        # Grouped variants
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 4},
                      num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
                      num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 4},
                      num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_m, pid_n = _grouped_tile_coordinates(
        tl.program_id(0),
        tl.cdiv(M, BLOCK_M),
        tl.cdiv(N, BLOCK_N),
        GROUP_M,
    )

    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        offset_k = k_idx * BLOCK_K
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        acc = tl.dot(a, b.T, acc)

    c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute C = A @ B.T into preallocated output tensor C."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = B.shape[1]

    def _make_descriptors(BLOCK_M, BLOCK_N, BLOCK_K):
        return (
            TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K]),
            TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K]),
            TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N]),
        )

    # Precompute grid; autotune will pick config then relaunch
    num_pid_m_max = max(triton.cdiv(M, cfg.kwargs["BLOCK_M"])
                        for cfg in _gemm_kernel._autotuner.configs)
    num_pid_n_max = max(triton.cdiv(N, cfg.kwargs["BLOCK_N"])
                        for cfg in _gemm_kernel._autotuner.configs)
    grid_size = num_pid_m_max * num_pid_n_max

    def grid(META):
        bm = META["BLOCK_M"]
        bn = META["BLOCK_N"]
        return (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

    # Create descriptors using default block shape (will be overwritten by pre_hook if needed)
    a_desc, b_desc, c_desc = _make_descriptors(128, 256, 64)

    _gemm_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        num_warps=8,
        num_stages=3,
    )