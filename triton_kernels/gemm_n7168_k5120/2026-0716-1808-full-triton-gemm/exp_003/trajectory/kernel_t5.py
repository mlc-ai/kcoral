import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    A_desc, B_desc, C,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    C_tmp,
    C_stride_0,
):
    # We map one CTA to one specific 128-row chunk of the entire C matrix.
    tile_id = tl.program_id(0)
    row_start = tile_id * BLOCK_M
    
    if row_start >= M:
        return

    # Pre-allocate space in shared memory for C's output tiles to allow overlap
    C_tmp = tl.empty((2, BLOCK_M, BLOCK_N), tl.bfloat16, bindings=[("buffer", "shared")])

    # Pre-compute row math and masks 
    row_C = row_start + tl.arange(0, BLOCK_M)
    col_C = tl.arange(0, BLOCK_N)
    offs_out = row_C[:, None] * C_stride_0 + col_C[None, :] * 1
    valid_mask = (row_start + tl.arange(0, BLOCK_M)) < M

    for i in range(2):
        C_tmp[i] = tl.zeros((BLOCK_M, BLOCK_N), tl.bfloat16)

    prev_buf = None
    col_iter = 0
    prev_col_start = -1
    prev_valid_mask_n = None
    
    for col_start in range(0, N, BLOCK_N):
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        valid_mask_n = (col_start + tl.arange(0, BLOCK_N)) < N
        
        for k_step in tl.range(0, K, BLOCK_K, num_stages=3):
            a0 = A_desc.load([row_start, k_step])
            a1 = A_desc.load([row_start, k_step + 128])
            
            b0 = B_desc.load([col_start, k_step])
            b1 = B_desc.load([col_start, k_step + 128])
            
            acc = tl.dot(a0, b0.T, acc)
            acc = tl.dot(a1, b1.T, acc)
            
        buf_idx = col_iter % 2
        
        # Assign current active state temporarily so subsequent cycles utilize TMA overlapping
        C_tmp[buf_idx] = acc 
        
        if prev_buf is not None:
            ptr_c = C + row_start * C_stride_0 + prev_col_start * 1
            prev_acc = C_tmp[prev_buf]
            masked_acc = tl.where(valid_mask[:, None], prev_acc, 0.0)
            tl.store(ptr_c + offs_out, masked_acc.to(tl.bfloat16), 
                     mask=valid_mask[:, None] & prev_valid_mask_n[None, :], boundary_check=(0, 1))
            
        prev_buf = buf_idx
        prev_col_start = col_start
        prev_valid_mask_n = valid_mask_n
        col_iter += 1
    
    if prev_buf is not None:
        ptr_c = C + row_start * C_stride_0 + prev_col_start * 1
        final_acc = C_tmp[prev_buf]
        masked_acc = tl.where(valid_mask[:, None], final_acc, 0.0)
        tl.store(ptr_c + offs_out, masked_acc.to(tl.bfloat16), 
                 mask=valid_mask[:, None] & prev_valid_mask_n[None, :], boundary_check=(0, 1))


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

    # Limit launch size to 64, mapping each CTA to processing strictly its unique subset of 128 rows
    num_ctas = min(64, triton.cdiv(M, 128))
    grid = (num_ctas,)
    
    _gemm_kernel[grid](
        A_desc, B_desc, C,
        M, N, K,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=256,
        C_tmp=None,
        C_stride_0=N,
        num_warps=8, num_stages=3,
    )