import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    A_desc_ptr,
    B_desc_ptr,
    C_desc_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Persistent GEMM kernel using device tensor descriptors for TMA."""
    start_pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = grid_m * grid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False):
        pid_m = tile_id // grid_n
        pid_n = tile_id % grid_n

        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        num_k_tiles = tl.cdiv(K, BLOCK_K)
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            # A desc load: [BLOCK_M, BLOCK_K]
            a = A_desc_ptr.load([offset_m, offset_k])
            # B desc load: [BLOCK_N, BLOCK_K] (physical [N, K] layout)
            b = B_desc_ptr.load([offset_n, offset_k])
            # tl.dot(a[BLOCK_M, BLOCK_K], b[BLOCK_N, BLOCK_K]) -> [BLOCK_M, BLOCK_N]
            acc = tl.dot(a, b, acc=acc)

        # Store via C descriptor
        C_desc_ptr.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute C = A @ B.T into preallocated output tensor using TMA descriptors."""
    torch.cuda.set_device(A.device)

    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]

    assert A.shape == (M, K), f"A shape {A.shape} != ({M}, {K})"
    assert B.shape == (N, K), f"B shape {B.shape} != ({N}, {K})"
    assert C.shape == (M, N), f"C shape {C.shape} != ({M}, {N})"

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    NUM_SMS = 132  # Hopper H100

    # Create host tensor descriptors with matching block shapes
    # A: [M, K] -> tile [BLOCK_M, BLOCK_K]
    a_desc = TensorDescriptor.from_tensor(
        A,
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero",
    )
    # B: [N, K] physically -> tile [BLOCK_N, BLOCK_K]
    b_desc = TensorDescriptor.from_tensor(
        B,
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero",
    )
    # C: [M, N] -> tile [BLOCK_M, BLOCK_N]
    c_desc = TensorDescriptor.from_tensor(
        C,
        block_shape=[BLOCK_M, BLOCK_N],
    )

    # Number of program instances: cap at NUM_SMS for persistent scheduling
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    num_programs = min(NUM_SMS, num_pid_m * num_pid_n)

    grid = (num_programs,)

    _gemm_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_SMS=NUM_SMS,
        num_warps=8,
        num_stages=5,
    )