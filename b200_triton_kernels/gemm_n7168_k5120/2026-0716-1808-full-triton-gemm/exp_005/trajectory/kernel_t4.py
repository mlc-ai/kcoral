import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M,
    N,
    K,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid is (num_m_tiles * 2, num_n_tiles). 
    # program_id(0) encodes both the M-tile index and which half of the N-tile we are processing.
    tile_id = tl.program_id(0)
    n_tile = tl.program_id(1)
    
    pid_n = n_tile
    half = tile_id % 2
    
    offset_n = pid_n * BLOCK_N
    m_tile = tile_id // 2
    offset_m = m_tile * BLOCK_M
    
    acc_0 = tl.zeros((BLOCK_M, BLOCK_N // 2), dtype=tl.float32)
    acc_1 = tl.zeros((BLOCK_M, BLOCK_N // 2), dtype=tl.float32)
    
    # K is fixed at 5120. We iterate exactly 5120 / 1024 = 5 times.
    num_k_tiles = tl.cdiv(K, 1024)
    
    for k_iter in range(num_k_tiles):
        k_step = k_iter * 1024
        
        a = a_desc.load([offset_m, k_step])
        
        b_0 = b_desc.load([offset_n + 0 * (BLOCK_N // 2), k_step])
        b_1 = b_desc.load([offset_n + 1 * (BLOCK_N // 2), k_step])
        
        acc_0 = tl.dot(a, b_0.T, acc_0)
        acc_1 = tl.dot(a, b_1.T, acc_1)

    mask_0 = (offset_m + tl.arange(0, BLOCK_M)[:, None] < M) & \
             (offset_n + tl.arange(0, BLOCK_N // 2)[None, :] < N)
    
    mask_1 = (offset_m + tl.arange(0, BLOCK_M)[:, None] < M) & \
             (offset_n + BLOCK_N // 2 + tl.arange(0, BLOCK_N // 2)[None, :] < N)
    
    c_val_0 = acc_0.to(tl.bfloat16)
    c_val_1 = acc_1.to(tl.bfloat16)
    
    base_ptr_0 = c_ptr + (offset_m + tl.arange(0, BLOCK_M)[:, None]) * stride_cm + \
                 (offset_n + tl.arange(0, BLOCK_N // 2)[None, :]) * stride_cn
    base_ptr_1 = c_ptr + (offset_m + tl.arange(0, BLOCK_M)[:, None]) * stride_cm + \
                 (offset_n + BLOCK_N // 2 + tl.arange(0, BLOCK_N // 2)[None, :]) * stride_cn
    
    tl.store(base_ptr_0, c_val_0, mask=mask_0, eviction_policy="evict_first")
    tl.store(base_ptr_1, c_val_1, mask=mask_1, eviction_policy="evict_first")


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    
    if A.numel() == 0:
        return

    M, K = A.shape
    N, K_b = B.shape
    assert K == K_b, f"K dimension mismatch: {K} vs {K_b}"
    
    BLOCK_M = 256
    BLOCK_N = 2048
    
    num_m_tiles = triton.cdiv(M, BLOCK_M)
    num_n_tiles = triton.cdiv(N, BLOCK_N)
    
    grid = (num_m_tiles * 2, num_n_tiles)
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, 1024])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N // 2, 1024])
    
    _gemm_kernel[grid](
        a_desc, b_desc, C,
        M, N, K,
        stride_cm=N, stride_cn=1,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=4,
        maxnreg=128,
    )


if __name__ == "__main__":
    M, N, K = 8192, 7168, 5120
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    C_ref = torch.matmul(A, B.T)
    
    run(A, B, C)
    
    print(f"Max error: {torch.max(torch.abs(C - C_ref))}")