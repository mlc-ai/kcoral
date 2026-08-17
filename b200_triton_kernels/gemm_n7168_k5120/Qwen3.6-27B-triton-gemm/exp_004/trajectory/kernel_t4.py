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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # L2-aware grouped tile mapping
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_steps = tl.cdiv(K, BLOCK_K)
    for k in range(num_k_steps):
        a_tile = A_desc.load([off_m, k * BLOCK_K])
        b_tile = B_desc.load([off_n, k * BLOCK_K])
        acc = tl.dot(a_tile, b_tile.T, acc=acc)

    C_desc.store([off_m, off_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute C = A @ B.T into preallocated output tensor C.

    Hand-tuned configurations for N=7168, K=5120 BF16 GEMM on Hopper.
    Runs the best matching config based on M dimension.
    """
    torch.cuda.set_device(A.device)

    M, K = A.shape
    N = B.shape[0]

    # Candidate configs sorted by expected performance for large M,N,K
    candidates = [
        # (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages)
        (256, 128, 64, 8, 8, 4),
        (256, 128, 64, 8, 8, 3),
        (256, 128, 128, 8, 8, 4),
        (256, 128, 128, 4, 8, 4),
        (128, 256, 64, 8, 8, 4),
        (128, 256, 64, 4, 8, 4),
        (128, 256, 128, 8, 8, 4),
        (128, 128, 64, 8, 8, 4),
        (128, 128, 64, 4, 8, 4),
        (256, 256, 64, 8, 8, 3),
        (256, 256, 64, 8, 8, 4),
        (128, 128, 64, 8, 4, 4),
    ]

    best_time = None
    best_params = None

    for i, (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages) in enumerate(candidates):
        A_desc = TensorDescriptor.from_tensor(A, block_shape=[BLOCK_M, BLOCK_K])
        B_desc = TensorDescriptor.from_tensor(B, block_shape=[BLOCK_N, BLOCK_K])
        C_desc = TensorDescriptor.from_tensor(C, block_shape=[BLOCK_M, BLOCK_N])

        num_pid_m = triton.cdiv(M, BLOCK_M)
        num_pid_n = triton.cdiv(N, BLOCK_N)
        grid = (num_pid_m * num_pid_n,)

        # Zero output before trial
        C.zero_()

        prof_start = torch.cuda.Event(enable_timing=True)
        prof_end = torch.cuda.Event(enable_timing=True)
        prof_start.record()

        _gemm_kernel[grid](
            A_desc, B_desc, C_desc,
            M, N, K,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        prof_end.record()
        torch.cuda.synchronize()
        elapsed_ms = prof_start.elapsed_time(prof_end)

        if best_time is None or elapsed_ms < best_time:
            best_time = elapsed_ms
            best_params = (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages)

    # Run final with best params
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages = best_params
    A_desc = TensorDescriptor.from_tensor(A, block_shape=[BLOCK_M, BLOCK_K])
    B_desc = TensorDescriptor.from_tensor(B, block_shape=[BLOCK_N, BLOCK_K])
    C_desc = TensorDescriptor.from_tensor(C, block_shape=[BLOCK_M, BLOCK_N])

    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    grid = (num_pid_m * num_pid_n,)

    _gemm_kernel[grid](
        A_desc, B_desc, C_desc,
        M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )