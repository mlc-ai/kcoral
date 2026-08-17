import torch
import triton
import triton.language as tl


@triton.jit
def issue_async_copy(dst_ptr, src_ptr, size_bytes):
    """Issue a cp.async.shared.global copy and commit it."""
    tl.inline_asm_elementwise(
        "cp.async.shared.global $0, $1, $2;\n"
        "cp.async.commit_group;\n",
        "",
        "r, r, n",
        (dst_ptr, src_ptr, size_bytes),
        is_pure=False
    )

@triton.jit
def wait_async_copy():
    """Wait for all committed async copies to complete."""
    tl.inline_asm_elementwise(
        "cp.async.cg;\n",
        "",
        "",
        (),
        is_pure=False
    )


@triton.jit
def _gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N_out,
    K,
    stride_A_m,
    stride_A_k,
    stride_B_n,
    stride_B_k,
    stride_C_m,
    stride_C_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Implements C = A @ B^T on a 2D grid with software double buffering using cp.async.
    Accumulates internally in FP32 utilizing Hopper tensor cores.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Declare double buffered shared memory banks
    shared_a0 = extern extern_shared((BLOCK_M * BLOCK_K * 2,), dtype=tl.int8)
    shared_a1 = extern extern_shared((BLOCK_M * BLOCK_K * 2,), dtype=tl.int8)
    shared_b0 = extern extern_shared((BLOCK_N * BLOCK_K * 2,), dtype=tl.int8)
    shared_b1 = extern extern_shared((BLOCK_N * BLOCK_K * 2,), dtype=tl.int8)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    # PROLOGUE: prefetch k=0 chunk into buffer 0 async
    if num_k_tiles > 0:
        a_ptr = A_ptr + row_idx[:, None] * stride_A_m + (0 * BLOCK_K) * stride_A_k
        b_ptr = B_ptr + col_idx[:, None] * stride_B_n + (0 * BLOCK_K) * stride_B_k
        
        issue_async_copy(shared_a0.to(tl.pointer(tl.int8)), a_ptr.to(tl.pointer(tl.int8)), BLOCK_M * BLOCK_K * 2)
        issue_async_copy(shared_b0.to(tl.pointer(tl.int8)), b_ptr.to(tl.pointer(tl.int8)), BLOCK_N * BLOCK_K * 2)
    
    current = 0
    for k in range(num_k_tiles):
        next_val = (k + 1) % 2
        
        # PREFETCH NEXT k-chunk ASYNC
        if k + 1 < num_k_tiles:
            next_a_ptr = A_ptr + row_idx[:, None] * stride_A_m + ((k + 1) * BLOCK_K + k_idx[0]) * stride_A_k
            next_b_ptr = B_ptr + col_idx[:, None] * stride_B_n + ((k + 1) * BLOCK_K + k_idx[0]) * stride_B_k
            
            next_shared_a = shared_a1 if next_val == 1 else shared_a0
            next_shared_b = shared_b1 if next_val == 1 else shared_b0
            
            issue_async_copy(next_shared_a.to(tl.pointer(tl.int8)), next_a_ptr.to(tl.pointer(tl.int8)), BLOCK_M * BLOCK_K * 2)
            issue_async_copy(next_shared_b.to(tl.pointer(tl.int8)), next_b_ptr.to(tl.pointer(tl.int8)), BLOCK_N * BLOCK_K * 2)

        # WAIT for current chunk's data to be ready
        wait_async_copy()
        
        # Define current buffers dynamically
        curr_shared_a = shared_a1 if current == 1 else shared_a0
        curr_shared_b = shared_b1 if current == 1 else shared_b0
        
        # Interpret buffers as contiguous matrices [BLOCK_M, BLOCK_K] and [BLOCK_N, BLOCK_K]
        s_A = curr_shared_a.to(tl.pointer(tl.bfloat16))
        s_B = curr_shared_b.to(tl.pointer(tl.bfloat16))
        
        # GATHER SCATTER logic effectively loading them cohesively 
        a = s_A + tl.arange(0, BLOCK_M)[:, None] * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
        b = s_B + tl.arange(0, BLOCK_N)[:, None] * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
        
        a = tl.load(a)
        b = tl.load(b)
        
        # Accumulate in FP32 leveraging Hopper tensor cores
        acc = tl.dot(a, b.T, acc, input_precision="tf32")
        
        current = next_val # rotate to next double buffer pair for iteration k+1
        
    # Write Out Epilogue
    ptr_C = C_ptr + row_idx[:, None] * stride_C_m + col_idx[None, :] * stride_C_n
    mask_m = row_idx < M
    mask_n = col_idx < N_out
    tl.store(ptr_C, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def run(A, B, C):
    """Compute ``C = A @ B.T`` into the preallocated tensor C."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N_out = B.shape[0]
    K = B.shape[1]
    
    grid = (triton.cdiv(M, 128), triton.cdiv(N_out, 128))
    _gemm_kernel[grid](
        A, 
        B, 
        C, 
        M, N_out, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=128, 
        BLOCK_N=128, 
        BLOCK_K=128,
        num_warps=8, 
        num_stages=2
    )