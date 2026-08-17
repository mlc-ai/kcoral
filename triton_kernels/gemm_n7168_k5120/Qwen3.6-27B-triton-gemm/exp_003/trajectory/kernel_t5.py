import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def run(A, B, C):
    """Compute C = A @ B.T into preallocated output C."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]  # 7168
    K = A.shape[1]  # 5120

    _launch_mixed(M, N, K, A, B, C)


def _launch_mixed(M, N, K, A, B, C):
    """Try multiple configurations and pick the fastest one.
    
    Since N=7168 and K=5120 are fixed constants, we can target specific tile sizes.
    We benchmark a few options and keep the best result in C.
    """
    # Candidate tile configurations optimized for Hopper WGMMA
    candidates = [
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 8, "num_stages": 3},
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 8, "num_stages": 3},
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 8, "num_stages": 3},
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "num_warps": 8, "num_stages": 3},
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 8, "num_stages": 3},
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 8, "num_stages": 4},
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 8, "num_stages": 5},
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 3},
    ]

    best_time = float("inf")
    C_save = C.clone()

    for cfg in candidates:
        BM, BN, BK = cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"]
        
        # Reset C before trial
        C.copy_(C_save)
        
        a_desc = TensorDescriptor.from_tensor(A, block_shape=[BM, BK])
        b_desc = TensorDescriptor.from_tensor(B, block_shape=[BN, BK])
        c_desc = TensorDescriptor.from_tensor(C, block_shape=[BM, BN])

        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

        def time_it():
            stream = torch.cuda.current_stream()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            _gemm_desc_kernel[grid](
                a_desc, b_desc, c_desc, M, N, K,
                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
            )
            end.record(stream)
            end.synchronize()
            return start.elapsed_time(end)

        try:
            elapsed = time_it()
            if elapsed < best_time:
                best_time = elapsed
                best_cfg = cfg
        except Exception:
            continue

    # Run with best config
    cfg = best_cfg
    BM, BN, BK = cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"]
    a_desc = TensorDescriptor.from_tensor(A, block_shape=[BM, BK])
    b_desc = TensorDescriptor.from_tensor(B, block_shape=[BN, BK])
    c_desc = TensorDescriptor.from_tensor(C, block_shape=[BM, BN])
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _gemm_desc_kernel[grid](
        a_desc, b_desc, c_desc, M, N, K,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )


@triton.jit
def _gemm_desc_kernel(
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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        offset_k = k_tile * BLOCK_K
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        acc = tl.dot(a, b.T, acc=acc)

    c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))