import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.tools.tiletuple import tiletuple


@triton.jit
def _gemm_kernel_2d(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Standard 2D-grid tiled GEMM. Computes C = A @ B.T
    Utilizes Hopper TMA and Tensor Cores.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    if offset_m >= M or offset_n >= N:
        return
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    # Capture the K dimension iteration space as a tensor to enable 
    # asynchronous TMA software pipelining across K blocks.
    num_k_steps = K // BLOCK_K
    ti = tl.range(0, num_k_steps)
    
    a = a_desc.load([offset_m, ti[:, None] * BLOCK_K])
    b = b_desc.load([offset_n, ti[:, None] * BLOCK_K])
    
    for a_tile, b_tile in tiletuple(a[0], b[0]):
        acc = tl.dot(a_tile, b_tile.T, acc)
    
    c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    # 128x128 matches Hopper WGMMA row/column tiles perfectly.
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64 
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    c_ptr = C.data_ptr() & 15
    
    _gemm_kernel_2d[grid](
        a_desc, b_desc, c_desc, M, N, K, c_ptr,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=3,
    )


if __name__ == "__main__":
    import time
    
    M, N, K = 8192, 7168, 5120
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    C = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    run(A, B, C)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    C_ref = torch.matmul(A, B.T)
    is_close = torch.allclose(C, C_ref, atol=1e-2)
    print(f"Correct: {is_close}, Time: {(end - start) * 1000:.2f} ms")