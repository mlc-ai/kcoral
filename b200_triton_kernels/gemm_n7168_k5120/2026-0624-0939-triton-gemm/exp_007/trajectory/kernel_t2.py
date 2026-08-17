import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.heuristics(values={"NUM_SMS": lambda args: 132})
@triton.jit
def _persistent_gemm_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    NUM_SMS: tl.constexpr,
):
    """
    Persistent tiled GEMM with double buffered A loads and grouped M-tile iteration.
    Computes C = A @ B.T. Each CTA owns a contiguous chunk of the M dimension and 
    a single tile of the N dimension, looping internally to fill its assigned rows.
    """
    start_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset_n = pid_n * 160
    
    num_k_steps = K // 160  # 32 for K=5120
    num_m_iters = (M - start_m * 160 + 159) // 160
    max_m_iters = (M + 159) // 160
    
    # Double buffered M offsets to overlap loads and computation
    offset_m_curr = start_m * 160
    offset_m_next = offset_m_curr + 160
    start_m_valid = (start_m * 160) < M
    next_m_valid = (offset_m_next) < M
    
    # Pre-issue the very first A load so the first loop iteration has it ready.
    a_curr = tl.zeros((160, 160), tl.bfloat16)
    if start_m_valid:
        a_curr = a_desc.load([offset_m_curr, 0])

    for m_iter in range(num_m_iters):
        acc = tl.zeros((160, 160), tl.float32)
        
        for k_step in tl.range(0, num_k_steps, 1, num_stages=4):
            offset_k = k_step * 160
            
            # Double buffered fetch for the subsequent M block.
            if next_m_valid:
                a_next = a_desc.load([offset_m_next, offset_k])
            
            b_curr = b_desc.load([offset_n, offset_k])
            acc = tl.dot(a_curr, b_curr.T, acc)
        
        # Before wrapping to the next M iteration, shift our working buffers forward.
        if num_k_steps > 0:
            if (num_k_steps - 1) == (num_k_steps - 1):
                a_curr = a_next
                offset_m_curr = offset_m_next
                start_m_valid = next_m_valid
                next_m_valid = (offset_m_next + 160) < M
                offset_m_next += 160
        
        if m_iter < num_m_iters - 1:
            if start_m_valid:
                c_desc.store([offset_m_curr, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into the preallocated output tensor ``C``."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    a_desc = TensorDescriptor.from_tensor(A, [160, 160])
    b_desc = TensorDescriptor.from_tensor(B, [160, 160])
    c_desc = TensorDescriptor.from_tensor(C, [160, 160])
    
    grid = (triton.cdiv(M, 160), triton.cdiv(N, 160))
    
    _persistent_gemm_kernel[grid](
        a_desc, b_desc, c_desc, M, N, K,
        num_warps=8, num_stages=4
    )