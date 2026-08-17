import torch
import triton
import triton.language as tl


def _alloc_fn(size: int, alignment: int, stream):
    """Allocator for Triton descriptor infrastructure storage."""
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_alloc_fn)

NUM_SMS = 132


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
                      num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
                      num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128},
                      num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64},
                      num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
                      num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
                      num_warps=4, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_persistent(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Persistent tiled GEMM using Hopper device-side tensor descriptors.

    Physical layouts: A[M,K], B[N,K], C[M,N] all contiguous row-major.
    Logical operation: C = A @ B.T  →  each element C[m,n] = sum_k A[m,k]*B[n,k]
    Descriptor loads [BLOCK_M,BLOCK_K] and [BLOCK_N,BLOCK_K], transpose B tile for dot.
    """
    out_dtype = C_ptr.dtype.element_ty

    # Device-side tensor descriptors for TMA-backed loads on Hopper
    a_desc = tl.make_tensor_descriptor(
        A_ptr, shape=[M, K], strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero")
    b_desc = tl.make_tensor_descriptor(
        B_ptr, shape=[N, K], strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K], padding_option="zero")
    c_desc = tl.make_tensor_descriptor(
        C_ptr, shape=[M, N], strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N])

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_k_steps = tl.cdiv(K, BLOCK_K)

    # Persistent grid-strided loop: each of ~NUM_SMS programs covers
    # approximately num_tiles / NUM_SMS (M,N) tile pairs.
    for tile_idx in tl.range(pid, num_tiles, NUM_SMS):
        pid_m = tile_idx // num_pid_n
        pid_n = tile_idx % num_pid_n

        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_step in range(num_k_steps):
            off_k = k_step * BLOCK_K
            a = a_desc.load([off_m, off_k])
            b = b_desc.load([off_n, off_k])
            acc = tl.dot(a, b.T, acc=acc)

        c_desc.store([off_m, off_n], acc.to(out_dtype))


def run(A, B, C):
    """Compute C = A @ B.T into preallocated output C.

    Args:
        A: [M, K] bfloat16 input
        B: [N, K] bfloat16 input (logically transposed)
        C: [M, N] bfloat16 preallocated output
    """
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]

    # Clamp grid at SM count for persistent scheduling
    nm = triton.cdiv(M, 64)  # worst-case small blocks
    nn = triton.cdiv(N, 64)
    max_tiles = nm * nn
    grid_size = min(NUM_SMS, max_tiles)

    _gemm_persistent[(grid_size,)](
        A, B, C,
        M, N, K,
        NUM_SMS=NUM_SMS,
    )