import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(A, B, C, M, N, K, stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn):
    """
    Fast GEMM kernel optimized for the Qwen3 14B qkv_proj configuration.
    Computes C = A @ B.T with constant tile sizes derived from common Hopper Tensor Core layouts.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    row_ptr = A + pid_m * BLOCK_M * stride_am
    col_ptr = B + pid_n * BLOCK_N * stride_bn
    c_ptr = C + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    
    for k0 in range(num_k_tiles):
        k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = k < K
        
        outs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = outs_m < M
        mask_a = mask_m[:, None] & mask_k[None, :]
        
        a = tl.load(row_ptr + outs_m[:, None] * stride_am + k[None, :] * stride_ak, 
                     mask=mask_a, other=0.0)
        
        outs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = outs_n < N
        mask_b = mask_k[:, None] & mask_n[None, :]
        
        b = tl.load(col_ptr + k[:, None] * stride_bk + outs_n[None, :] * stride_bn, 
                     mask=mask_b, other=0.0)
        
        acc = tl.dot(a, b.T, acc)
    
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

    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    _gemm_kernel[grid](
        A, B, C, M, N, K,
        stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
        num_warps=4,
        num_stages=3,
    )


if __name__ == "__main__":
    print("Running local sanity check...")
    M_val = 1024
    N_val = 7168
    K_val = 5120
    
    A = torch.randn(M_val, K_val, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N_val, K_val, device="cuda", dtype=torch.bfloat16)
    C_torch = torch.matmul(A, B.T)
    
    C = torch.empty_like(C_torch)
    run(A, B, C)
    
    max_err = (C - C_torch).abs().max().item()
    print(f"Max error: {max_err}")
    if max_err <= 1e-2:
        print("Local sanity check PASSED.")
    else:
        print("Local sanity check FAILED!")