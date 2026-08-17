import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=5),
    ],
    key=["M", "N", "K"],
    pre_hook=lambda args: None,
)
@triton.jit
def _gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Blocked GEMM: C = A @ B.T using device-side tensor descriptors."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create tensor descriptors inside kernel for TMA access
    # A is [M, K] row-major, stride_k=1
    a_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero",
    )
    
    # B is [N, K] row-major, stride_k=1  
    b_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero",
    )
    
    # C is [M, N] row-major, stride_n=1
    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N

    num_k_iters = tl.cdiv(K, BLOCK_K)
    for k in range(0, num_k_iters):
        offset_k = k * BLOCK_K

        # Load A tile [BLOCK_M, BLOCK_K]
        a_tile = a_desc.load([offset_m, offset_k])
        
        # Load B tile [BLOCK_N, BLOCK_K], then transpose to [BLOCK_K, BLOCK_N]
        b_tile = b_desc.load([offset_n, offset_k])

        acc = tl.dot(a_tile, b_tile.T, acc)

    # Store result
    c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """
    Compute C = A @ B.T into preallocated output C.
    
    A: [M, K] bfloat16
    B: [N, K] bfloat16
    C: [M, N] bfloat16 (output)
    """
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = B.shape[1]

    # Need an allocator for device-side tensor descriptor storage
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    _gemm_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
    )