import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_kernel_128x128(
    a_desc, b_desc, c_ptr,
    M, N, K,
    c_stride_m, c_stride_n,
    TP_BLOCKS_M: tl.constexpr,
    TP_BLOCKS_N: tl.constexpr,
    TP_BLOCKS_K: tl.constexpr,
):
    m_idx = tl.program_id(0)
    n_idx = tl.program_id(1)
    k_idx = tl.program_id(2)
    
    a = a_desc.load([m_idx * 128, k_idx * 128])
    b = b_desc.load([n_idx * 128, k_idx * 128])
    
    acc = tl.dot(a, b.T)
    
    out = acc.to(tl.bfloat16)
    
    base_ptr = c_ptr + m_idx * 128 * c_stride_m + n_idx * 128 * c_stride_n
    
    row = tl.arange(0, 128)
    col = tl.arange(0, 128)
    out_offset_m = (m_idx * 128) + row[:, None]
    out_offset_n = (n_idx * 128) + col[None, :]
    ptrs = base_ptr + out_offset_m * c_stride_m + out_offset_n * c_stride_n
    
    tl.atomic_add(ptrs, out)


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    
    M, K = A.shape
    N = B.shape[0]
    
    assert K == 5120, f"Unsupported K={K}"
    assert N == 7168, f"Unsupported N={N}"
    
    C_fp32 = torch.empty((M, N), dtype=torch.float32, device=C.device)
    
    a_desc = TensorDescriptor.from_tensor(A, [128, 128])
    b_desc = TensorDescriptor.from_tensor(B, [128, 128])
    
    grid = (M // 128, N // 128, K // 128)
    
    _gemm_kernel_128x128[grid](
        a_desc, b_desc, C_fp32, M, N, K, N, 1, 
        TP_BLOCKS_M=128, TP_BLOCKS_N=128, TP_BLOCKS_K=128,
    )
    
    C.copy_(C_fp32.to(C.dtype))