import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    A_desc, B_desc, C_desc_ptr,
    M, N, K, N_half,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n_half = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    offset_m = pid_m * BLOCK_M
    offset_n_base = pid_n_half * N_half + pid_n * BLOCK_N
    offset_n_rel = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_k_tiles = K // BLOCK_K
    for k_tile in range(num_k_tiles):
        a = A_desc.load([offset_m, k_tile * BLOCK_K])
        b = B_desc.load([offset_n_base + offset_n_rel, k_tile * BLOCK_K])
        acc = tl.dot(a, b.T, acc)
    
    # Process the K dimension remainder
    k_offset = num_k_tiles * BLOCK_K
    rem = K - k_offset
    if rem > 0:
        a_rem = A_desc.load([offset_m, k_offset])
        b_rem = B_desc.load([offset_n_base + offset_n_rel, k_offset])
        
        k_idx = tl.arange(0, BLOCK_K)
        k_idx_full = k_idx[:, None]
        a_rem_masked = a_rem[k_idx_full < rem, k_idx_full]
        b_rem_masked = b_rem[k_idx_full < rem, k_idx_full]
        
        acc = tl.dot(a_rem_masked, b_rem_masked.T, acc)
    
    if pid_n_half == 0:
        C_desc_ptr[0].store([offset_m, offset_n_rel], acc.to(tl.bfloat16))
    else:
        C_desc_ptr[1].store([offset_m, offset_n_rel], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    N_half = N // 2
    
    BLOCK_M = 128
    BLOCK_N = 2048
    BLOCK_K = 2048
    
    A_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    B_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc_0 = TensorDescriptor.from_tensor(C[:, 0:N_half], [BLOCK_M, BLOCK_N])
    c_desc_1 = TensorDescriptor.from_tensor(C[:, N_half:2*N_half], [BLOCK_M, BLOCK_N])
    
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]), 
        2, 
        triton.cdiv(N_half, META["BLOCK_N"])
    )
    
    _gemm_kernel[grid](
        A_desc, B_desc, [c_desc_0, c_desc_1],
        M, N, K, N_half,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3, maxnreg=255,
    )