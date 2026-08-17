import torch
import triton
import triton.language as tl


# Allocate infrastructure storage for device-created tensor descriptors
triton.set_allocator(
    lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
)


@triton.jit
def _gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    """
    Persistent GEMM using device tensor descriptors on Hopper.
    
    Computes C[M,N] = A[M,K] @ B[N,K].T into preallocated C.
    B is physically stored as [N, K]; we transpose in the dot operation.
    Accumulation uses FP32; output is cast to BF16.
    """
    # Create device-side tensor descriptors for TMA-backed access
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero",
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero",
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    # Persistent loop: each program cycles through multiple output tiles
    for tile_id in tl.range(
        pid,
        num_tiles,
        NUM_SMS,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE,
    ):
        pm = tile_id // num_pid_n
        pn = tile_id % num_pid_n
        offset_m = pm * BLOCK_M
        offset_n = pn * BLOCK_N

        # Fresh accumulator for each tile
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        num_k_steps = tl.cdiv(K, BLOCK_K)
        for ki in range(num_k_steps):
            offset_k = ki * BLOCK_K
            a_tile = a_desc.load([offset_m, offset_k])
            b_tile = b_desc.load([offset_n, offset_k])
            # b_tile[BLOCK_N, BLOCK_K].T -> [BLOCK_K, BLOCK_N]
            # dot(a_tile[BLOCK_M, BLOCK_K], b_tile.T[BLOCK_K, BLOCK_N]) -> [BLOCK_M, BLOCK_N]
            acc = tl.dot(a_tile, b_tile.T, acc)

        # Store result; OOB writes are ignored by descriptor
        c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C.
    
    A: [M, K=5120] bfloat16
    B: [N=7168, K=5120] bfloat16 (physical layout)
    C: [M, N=7168] bfloat16 (preallocated destination)
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N = B.shape[0]

    # Tile configuration tuned for Hopper (SM90/SM90a)
    # - BLOCK_M=128, BLOCK_N=256 maximizes output tile size for L2 reuse
    # - BLOCK_K=64 balances register pressure (80% FLOP efficiency on WMMA units)
    # - 8 warps saturates Hopper WGMMA (requires warp-group of 4+ warps)
    # - Persistent scheduling keeps SMs fully utilized across variable M
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64

    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    grid_size = min(num_sms, num_tiles)

    # Launch with warp_specialize=False (stable, well-tested path on Hopper)
    _gemm_kernel[(grid_size,)](
        A,
        B,
        C,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_SMS=num_sms,
        WARP_SPECIALIZE=False,
        num_warps=8,
        num_stages=3,
    )