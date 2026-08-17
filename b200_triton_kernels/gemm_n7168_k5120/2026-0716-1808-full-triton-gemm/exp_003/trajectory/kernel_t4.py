import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    A_desc, B_desc, C_desc,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    row_C_start = pid_m * BLOCK_M
    col_C_start = pid_n * BLOCK_N
    
    if row_C_start >= M:
        return

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    num_k_iters = K // BLOCK_K
    for k_iter in range(num_k_iters):
        k_start = k_iter * BLOCK_K
        a = A_desc.load([row_C_start, k_start])
        b = B_desc.load([col_C_start, k_start])
        acc = tl.dot(a, b.T, acc)

    c_desc.store([row_C_start, col_C_start], acc.to(tl.bfloat16))


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
    
    A_desc = TensorDescriptor.from_tensor(A, [128, 128])
    B_desc = TensorDescriptor.from_tensor(B, [128, 128])
    C_desc = TensorDescriptor.from_tensor(C, [128, 128])

    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    
    _gemm_kernel[grid](
        A_desc, B_desc, C_desc,
        M, N, K,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=128,
        num_warps=8, num_stages=3,
    )