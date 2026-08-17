import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A, B, C, M, N, K, 
    stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn
):
    """
    Fast GEMM kernel optimized for the Qwen3 14B qkv_proj configuration.
    Computes C = A @ B.T with constant tile sizes derived from common Hopper Tensor Core layouts.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Local tile definitions
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    # Safely initialize accumulator in FP32 to avoid precision loss during inner reductions
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    # Define contiguous linear ranges for indexing
    arange_m = tl.arange(0, BLOCK_M)
    arange_n = tl.arange(0, BLOCK_N)
    arange_k = tl.arange(0, BLOCK_K)
    
    # Extracted chunked base offsets to avoid repeatedly computing the same math inside the loop.
    row_ptr = A + offset_m * stride_am
    col_ptr = B + offset_n * stride_bn
    c_ptr = C + offset_m * stride_cm + offset_n * stride_cn
    
    # Since N=7168 and K=5120 are guaranteed multiples of our configured blocks, 
    # bounding loops cleanly avoids boundary checks inside the hot path.
    for k0 in range(5120 // 128):
        k_base = k0 * BLOCK_K
        
        a = tl.load(row_ptr + arange_m[:, None] * stride_am + (k_base + arange_k[None, :]) * stride_ak, 
                     mask=(offset_m + arange_m < M)[:, None], other=0.0)
        
        # B mask omitted as N is fully divisible by block size (7168 % 128 == 0)
        b = tl.load(col_ptr + (k_base + arange_k[:, None]) * stride_bk + arange_n[None, :] * stride_bn)
        
        acc = tl.dot(a, b.T, acc)
    
    M_left = M - offset_m
    
    # Mask only uninitialized rows out of store bounds when writing final outputs back to memory
    mask_m = (arange_m < M_left)
    
    tl.store(c_ptr + arange_m[:, None] * stride_cm + arange_n[None, :] * stride_cn, 
             acc.to(C.dtype), mask=mask_m[:, None])


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
        num_stages=1,
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