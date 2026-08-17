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
    
    for k_step in tl.range(0, K, BLOCK_K, num_stages=3):
        col_A = k_step + tl.arange(0, BLOCK_K)
        ptrs_a = A + row_A[:, None] * stride_am + col_A[None, :] * stride_ak
        mask_a = (row_A[:, None] < M) & (col_A[None, :] < K)
        a_tile = tl.load(ptrs_a, mask=mask_a, other=0.0)
        
        row_B = k_step + tl.arange(0, BLOCK_K)
        ptrs_b = B + col_B[None, :] * stride_bn + row_B[:, None] * stride_bk
        mask_b = (row_B[:, None] < K) & (col_B[None, :] < N)
        b_tile = tl.load(ptrs_b, mask=mask_b, other=0.0)
        
        acc = tl.dot(a_tile, b_tile, acc)
    
    out_ptr = C + row_A[:, None] * stride_cm + col_B[None, :] * stride_cn
    
    valid_m = row_A < M
    valid_n = col_B < N
    store_mask = valid_m[:, None] & valid_n[None, :]
    
    tl.store(out_ptr, acc.to(C.dtype.element_ty), mask=store_mask)


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
    
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 256))
    
    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am=K, stride_ak=1,
        stride_bk=1, stride_bn=K,
        stride_cm=N, stride_cn=1,
        BLOCK_M=128, BLOCK_N=256, BLOCK_K=128,
        num_warps=8, num_stages=3,
    )