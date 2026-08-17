import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    row_A = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_B = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_k_iters = K // BLOCK_K
    
    # Prologue load
    col_A0 = tl.arange(0, BLOCK_K)
    ptrs_a0 = A + row_A[:, None] * stride_am + col_A0[None, :] * stride_ak
    mask_a0 = (row_A[:, None] < M) & (col_A0[None, :] < K)
    a_tile0 = tl.load(ptrs_a0, mask=mask_a0, other=0.0)
    
    for k_iter in range(0, num_k_iters - 1):
        col_A1 = (k_iter + 1) * BLOCK_K + tl.arange(0, BLOCK_K)
        ptrs_a1 = A + row_A[:, None] * stride_am + col_A1[None, :] * stride_ak
        mask_a1 = (row_A[:, None] < M) & (col_A1[None, :] < K)
        a_tile1 = tl.load(ptrs_a1, mask=mask_a1, other=0.0)
        
        row_B = k_iter * BLOCK_K + tl.arange(0, BLOCK_K)
        ptrs_b = B + row_B[:, None] * stride_bk + col_B[None, :] * stride_bn
        mask_b = (row_B[:, None] < K) & (col_B[None, :] < N)
        b_tile = tl.load(ptrs_b, mask=mask_b, other=0.0)
        
        acc = tl.dot(a_tile0, b_tile, acc)
        
        a_tile0 = a_tile1
    
    # Epilogue
    row_B = (num_k_iters - 1) * BLOCK_K + tl.arange(0, BLOCK_K)
    ptrs_b = B + row_B[:, None] * stride_bk + col_B[None, :] * stride_bn
    mask_b = (row_B[:, None] < K) & (col_B[None, :] < N)
    b_tile = tl.load(ptrs_b, mask=mask_b, other=0.0)
    
    acc = tl.dot(a_tile0, b_tile, acc)
    
    row_C = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_C = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptr = C + row_C[:, None] * stride_cm + col_C[None, :] * stride_cn
    mask_c = (row_C[:, None] < M) & (col_C[None, :] < N)
    tl.store(out_ptr, acc.to(C.dtype.element_ty), mask=mask_c)


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
    
    grid = (triton.cdiv(M, 64), N // 128)
    
    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am=K, stride_ak=1,
        stride_bk=1, stride_bn=K,
        stride_cm=N, stride_cn=1,
        BLOCK_M=64, BLOCK_N=128, BLOCK_K=256,
        num_warps=8, num_stages=2,
    )