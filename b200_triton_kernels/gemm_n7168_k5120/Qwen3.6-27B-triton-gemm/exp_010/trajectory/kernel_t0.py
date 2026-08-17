import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes C = A @ B.T using a tiled matrix-multiply approach.
    
    Tile layout:
      - A tile: [BLOCK_M, BLOCK_K] loaded from A[start_m, start_k]
      - B tile: [BLOCK_N, BLOCK_K] loaded from B[start_n, start_k], transposed for dot
      - Output tile: [BLOCK_M, BLOCK_N] written to C[start_m, start_n]
    
    B is stored physically as [N, K], so B.T[k, n] accesses B[n, k].
    """
    start_pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    for pid in tl.range(start_pid, num_tiles, num_pids):
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Base pointers for each tile's row/column slice
        a_ptrs = A + offs_m[:, None] * stride_am
        b_ptrs = B + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for start_k in range(0, tl.cdiv(K, BLOCK_K)):
            offs_k = start_k * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            # Load A[BLOCK_M, BLOCK_K] and B[BLOCK_N, BLOCK_K] tiles
            a = tl.load(
                a_ptrs + offs_k[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (k_mask[None, :]),
                other=0.0,
            )
            b = tl.load(
                b_ptrs + offs_k[:, None] * stride_bk,
                mask=(k_mask[:, None]) & (offs_n[None, :] < N),
                other=0.0,
            )

            # dot([BM, BK], [BK, BN]) -> [BM, BN]
            acc = tl.dot(a, b.T, acc=acc)

        # Write result tile
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
    grid = (min(num_sm, num_tiles),)

    # Stride mapping:
    #   A[M, K]:     stride_am=A.stride(0), stride_ak=A.stride(1)
    #   B[N, K]:     stride_bn=B.stride(0) for N-axis, stride_bk=B.stride(1) for K-axis
    #   C[M, N]:     stride_cm=C.stride(0), stride_cn=C.stride(1)
    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(1), B.stride(0),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )