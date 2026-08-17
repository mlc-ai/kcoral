import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    A_desc, B0_desc, B1_desc, C_desc,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n_half = tl.program_id(1)
    pid_n = tl.program_id(2)

    row_C = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_C = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    if row_C[0] >= M:
        return

    n_offset = pid_n_half * (N // 2)
    b_desc = B0_desc if pid_n_half == 0 else B1_desc

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k_iter in range(K // BLOCK_K):
        a = A_desc.load([row_C[0], k_iter * BLOCK_K])
        b = b_desc.load([n_offset + col_C[0], k_iter * BLOCK_K])
        acc = tl.dot(a, b.T, acc)

    c_desc.store([row_C[0], n_offset + col_C[0]], acc.to(C_desc.dtype.element_ty))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    assert N == 7168, f"N must be 7168, got {N}"
    assert K == 5120, f"K must be 5120, got {K}"
    assert C.shape == (M, N), f"C shape mismatch: expected {(M, N)}, got {C.shape}"
    assert A.stride(0) == K and A.stride(1) == 1, "A must be contiguous"
    assert B.stride(0) == K and B.stride(1) == 1, "B must be contiguous"
    assert C.stride(0) == N and C.stride(1) == 1, "C must be contiguous"
    
    A_desc = TensorDescriptor.from_tensor(A, [64, 64])
    B0_desc = TensorDescriptor.from_tensor(B[:N//2, :], [128, 64])
    B1_desc = TensorDescriptor.from_tensor(B[N//2:, :], [128, 64])
    C_desc = TensorDescriptor.from_tensor(C, [64, 128])

    grid = (triton.cdiv(M, 64), 2, 28)
    
    _gemm_kernel[grid](
        A_desc, B0_desc, B1_desc, C_desc,
        M, N, K,
        BLOCK_M=64, BLOCK_N=128, BLOCK_K=64,
    )