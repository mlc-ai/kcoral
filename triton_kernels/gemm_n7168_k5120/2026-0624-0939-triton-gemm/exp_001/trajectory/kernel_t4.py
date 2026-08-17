import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    A_desc, B_desc, C_ptr,
    M, N, K,
    stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Standard 2D grid mapping: each program resolves one output tile coordinate
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    # Establish output storage coordinates
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    offs_n = offset_n + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_k_tiles = K // BLOCK_K
    for k_tile in range(num_k_tiles):
        # Efficient Hopper TMA loads tracking internal loop state automatically
        a = A_desc.load([offset_m, k_tile * BLOCK_K])
        b = B_desc.load([offset_n, k_tile * BLOCK_K])
        
        # Standard Matrix multiplication logic expectation: A[M, K] @ B[N, K].T -> C[M, N]
        acc = tl.dot(a, b.T, acc)
    
    # Destination passing output contract 
    ptrs_c = C_ptr + offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(ptrs_c, acc.to(tl.bfloat16), mask=mask_c)


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 256
    
    # Host-side configurations bypass invalid shape translation limits entirely
    A_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    B_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"])
    )
    
    _gemm_kernel[grid](
        A_desc, B_desc, C,
        M, N, K,
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )