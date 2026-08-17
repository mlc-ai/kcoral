import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    A_desc,
    B_desc,
    C_desc,
    M,
    N,
    K,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Persistent-scheduled GEMM: C[*,*] = A[*,*] @ B[*,:].T using TMA descriptors."""
    start_pid = tl.program_id(0)
    pid_stride = tl.num_programs(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, pid_stride):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        num_k = tl.cdiv(K, BLOCK_K)
        for ki in range(num_k):
            off_k = ki * BLOCK_K
            a = A_desc.load([off_m, off_k])
            b = B_desc.load([off_n, off_k])
            acc = tl.dot(a, b.T, acc)

        C_desc.store([off_m, off_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Destination-passing: compute C = A @ B.T on-preallocated bf16 buffer."""
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N = B.shape[0]

    # Hopper-optimized block sizes
    # BLOCK_K=64 matches WGMMA granularity for BF16->FP32 acc
    # BLOCK_M=128, BLOCK_N=256 give balanced compute/occupancy
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64

    # Build host descriptors (lower to TMA on Hopper)
    A_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    B_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    C_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])

    props = torch.cuda.get_device_properties(A.device)
    num_sms = props.multi_processor_count

    # Persistent grid: cap at NUM_SMS to avoid oversubscription
    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid_size = min(num_sms, num_tiles)
    grid = (grid_size,)

    _gemm_kernel[grid](
        A_desc,
        B_desc,
        C_desc,
        M,
        N,
        K,
        NUM_SMS=num_sms,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )