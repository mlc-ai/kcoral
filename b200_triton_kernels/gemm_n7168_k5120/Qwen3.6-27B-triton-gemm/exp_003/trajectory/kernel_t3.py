import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
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
):
    """Tiled GEMM with TMA descriptors: C[M,N] = A[M,K] @ B[N,K].T"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        offset_k = k_tile * BLOCK_K
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        acc = tl.dot(a, b.T, acc=acc)

    c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute C = A @ B.T into preallocated output C."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]

    # Build descriptors with a dummy block_shape; autotune will override
    a_desc = TensorDescriptor.from_tensor(A, block_shape=[64, 64])
    b_desc = TensorDescriptor.from_tensor(B, block_shape=[64, 64])
    c_desc = TensorDescriptor.from_tensor(C, block_shape=[64, 64])

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _gemm_kernel[grid](
        a_desc, b_desc, c_desc,
        M, N, K,
    )