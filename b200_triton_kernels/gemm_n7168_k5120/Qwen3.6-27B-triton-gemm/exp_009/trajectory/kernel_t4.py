import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Persistent GEMM using host-side tensor descriptors on Hopper.
    
    Computes C[M,N] = A[M,K] @ B[N,K].T into preallocated C.
    Accumulation uses FP32; output is cast to BF16.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    # Persistent loop: each program processes multiple output tiles
    for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=False):
        pm = tile_id // num_pid_n
        pn = tile_id % num_pid_n
        offset_m = pm * BLOCK_M
        offset_n = pn * BLOCK_N

        # Fresh accumulator for each tile
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        num_k_steps = tl.cdiv(K, BLOCK_K)
        for ki in range(num_k_steps):
            offset_k = ki * BLOCK_K
            # Load [BLOCK_M, BLOCK_K] from A's physical [M, K] layout
            a_tile = a_desc.load([offset_m, offset_k])
            # Load [BLOCK_N, BLOCK_K] from B's physical [N, K] layout
            b_tile = b_desc.load([offset_n, offset_k])
            # dot(a[BLOCK_M,BLOCK_K], b.T[BLOCK_K,BLOCK_N]) -> [BLOCK_M,BLOCK_N]
            acc = tl.dot(a_tile, b_tile.T, acc)

        # Store converted to BF16; descriptor ignores OOB writes
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

    # Tile configuration optimized for Hopper (SM90/SM90a)
    # - Large tiles amortize launch overhead and maximize FLOPs per tile
    # - BLOCK_M=128, BLOCK_N=128 gives balanced work per program
    # - BLOCK_K=32 means fewer iterations but better pipeline overlap (K/BLOCK_K=160)
    # - 8 warps saturates WGMMA units; 3 stages hide memory latency
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # Create host-side tensor descriptors — lower to TMA on Hopper
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])

    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    grid_size = min(num_sms, num_tiles)

    _gemm_kernel[(grid_size,)](
        a_desc, b_desc, c_desc,
        M, N, K,
        NUM_SMS=num_sms,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=3,
    )