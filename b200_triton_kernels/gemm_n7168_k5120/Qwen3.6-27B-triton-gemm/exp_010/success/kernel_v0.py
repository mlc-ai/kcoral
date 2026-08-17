import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes C = A @ B.T using tiled matrix-multiply.
    
    A: [M, K], B: [N, K] → C: [M, N]
    B.T[k, n] = B[n, k], so we load B tiles as [BLOCK_N, BLOCK_K] then transpose.
    
    Tile shapes:
      - a_tile: [BLOCK_M, BLOCK_K]
      - b_tile: [BLOCK_N, BLOCK_K] → b_tile.T: [BLOCK_K, BLOCK_N]
      - acc:    [BLOCK_M, BLOCK_N]
    """
    start_pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    for pid in tl.range(start_pid, num_tiles, num_pids):
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for start_k in range(0, tl.cdiv(K, BLOCK_K)):
            offs_k = start_k * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]
            k_mask = offs_k < K

            # Load A[BLOCK_M, BLOCK_K]: ptr shape [BLOCK_M, BLOCK_K]
            a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (k_mask[None, :]),
                other=0.0,
            )

            # Load B[BLOCK_N, BLOCK_K]: ptr shape [BLOCK_N, BLOCK_K]
            # B is stored [N, K], so index B[n, k] uses stride_bn for n, stride_bk for k
            b_ptrs = B + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
            b = tl.load(
                b_ptrs,
                mask=(offs_n[:, None] < N) & (k_mask[None, :]),
                other=0.0,
            )

            # tl.dot([BM, BK], [BK, BN]) -> [BM, BN]
            # b has shape [BLOCK_N, BLOCK_K], so b.T has shape [BLOCK_K, BLOCK_N]
            acc = tl.dot(a, b.T, acc=acc)

        # Write output tile C[BLOCK_M, BLOCK_N]
        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(
            c_ptrs,
            acc.to(tl.bfloat16),
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def run(A, B, C):
    """Compute C = A @ B.T into preallocated output tensor C."""
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N = B.shape[0]

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    # Persistent scheduling: cap grid at SM count
    num_sm = torch.cuda.get_device_properties(A.device).multi_processor_count
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    grid_size = min(num_sm, num_tiles)
    grid = (grid_size,)

    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )