import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel_int4(
    A, B, C, M, N, K, STRIDE_A_M, STRIDE_B_N, STRIDE_C_M,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    assert BLOCK_K % 8 == 0
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    A_int4 = (A + offset_m * STRIDE_A_M).to(tl.int4.pointer_dtype())
    B_int4 = (B + offset_n * STRIDE_B_N).to(tl.int4.pointer_dtype())
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    num_k_steps = K // BLOCK_K
    
    for k_idx in range(num_k_steps):
        offset_k = k_idx * BLOCK_K
        
        row_offsets_a = (offset_m + tl.arange(0, BLOCK_M)) * STRIDE_A_M
        a_ptrs = (row_offsets_a[:, None] + (offset_k + tl.arange(0, BLOCK_K // 8) * 8)[None, :]) // 8
        a_packed = tl.load(A_int4 + a_ptrs)
        a_tile = a_packed.reshape((BLOCK_M, BLOCK_K // 8, 8))
        a_tile = a_tile.cast(tl.bfloat16, bitcast=True)
        a_tile = a_tile.reshape((BLOCK_M, BLOCK_K))
        
        row_offsets_b = (offset_n + tl.arange(0, BLOCK_N)) * STRIDE_B_N
        b_ptrs = (row_offsets_b[:, None] + (offset_k + tl.arange(0, BLOCK_K // 8) * 8)[None, :]) // 8
        b_packed = tl.load(B_int4 + b_ptrs)
        b_tile = b_packed.reshape((BLOCK_N, BLOCK_K // 8, 8))
        b_tile = b_tile.cast(tl.bfloat16, bitcast=True)
        b_tile = b_tile.reshape((BLOCK_N, BLOCK_K))
        
        acc = tl.dot(a_tile, b_tile.T, acc)
    
    off_m = tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    out_ptr = C + (offset_m + off_m) * STRIDE_C_M + (offset_n + off_n)
    mask_m = (offset_m + off_m)[:, None] < M
    tl.store(out_ptr, acc.to(tl.bfloat16), mask=mask_m)


@triton.jit
def _gemm_kernel(
    A, B, C, M, N, K, STRIDE_A_M, STRIDE_B_N, STRIDE_C_M,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    num_k_steps = K // BLOCK_K
    
    for k_idx in range(num_k_steps):
        offset_k = k_idx * BLOCK_K
        k = offset_k + tl.arange(0, BLOCK_K)
        
        a_ptrs = A + (offset_m + tl.arange(0, BLOCK_M))[:, None] * STRIDE_A_M + k[None, :]
        mask_m = (offset_m + tl.arange(0, BLOCK_M))[:, None] < M
        a_tile = tl.load(a_ptrs, mask=mask_m, other=0.0)
        
        b_ptrs = B + (offset_n + tl.arange(0, BLOCK_N))[:, None] * STRIDE_B_N + k[None, :]
        mask_n = (offset_n + tl.arange(0, BLOCK_N))[:, None] < N
        b_tile = tl.load(b_ptrs, mask=mask_n, other=0.0)
        
        acc = tl.dot(a_tile, b_tile.T, acc)
    
    off_m = tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    out_ptr = C + (offset_m + off_m) * STRIDE_C_M + (offset_n + off_n)
    mask_m = (offset_m + off_m)[:, None] < M
    tl.store(out_ptr, acc.to(tl.bfloat16), mask=mask_m)


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    grid = (num_pid_m, num_pid_n)
    
    STRIDE_A_M = A.stride[0]
    STRIDE_B_N = B.stride[0]
    STRIDE_C_M = C.stride[0]
    
    if A.stride[1] == 1 and B.stride[1] == 1:
        _gemm_kernel_int4[grid](A, B, C, M, N, K, STRIDE_A_M, STRIDE_B_N, STRIDE_C_M,
                                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                                num_warps=8, num_stages=4)
    else:
        _gemm_kernel[grid](A, B, C, M, N, K, STRIDE_A_M, STRIDE_B_N, STRIDE_C_M,
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                           num_warps=8, num_stages=4)