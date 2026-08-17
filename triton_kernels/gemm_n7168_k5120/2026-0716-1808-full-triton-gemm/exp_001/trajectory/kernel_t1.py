import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    ],
    key=["M"],
)
@triton.jit
def _gemm_kernel(
    A, B, C, M, N, K, 
    stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    row_ptr = A + pid_m * BLOCK_M * stride_am
    col_ptr = B + pid_n * BLOCK_N * stride_bn
    
    outs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = outs_m < M
    outs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = outs_n < N
    
    for k0 in range(num_k_tiles):
        k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = k < K
        
        a = tl.load(row_ptr + outs_m[:, None] * stride_am + k[None, :] * stride_ak, 
                     mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        b = tl.load(col_ptr + k[:, None] * stride_bk + outs_n[None, :] * stride_bn, 
                     mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        acc = tl.dot(a, b.T, acc)
    
    c_ptr = C + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptr + outs_m[:, None] * stride_cm + outs_n[None, :] * stride_cn, 
             acc.to(C.dtype), mask=mask_c)


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bn, stride_bk = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    _gemm_kernel[grid](
        A, B, C, M, N, K,
        stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
    )