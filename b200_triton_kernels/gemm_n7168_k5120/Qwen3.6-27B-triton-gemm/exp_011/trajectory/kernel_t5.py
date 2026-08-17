import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Persistent GEMM with group-M-sweeping for L2 cache reuse."""
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    
    # Number of full groups
    num_groups = tl.cdiv(num_pid_m, GROUP_SIZE_M)
    
    # Grouped block row assignment for better L2 reuse
    group_id = pid // num_pid_n
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(GROUP_SIZE_M, num_pid_m - first_pid_m)
    
    pid_m = first_pid_m + ((pid % num_pid_n) % group_size_m)
    pid_n = (pid % num_pid_n) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_steps = tl.cdiv(K, BLOCK_K)
    for ki in range(num_k_steps):
        offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)

        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = B + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
        b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b.T, acc)

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


def run(A, B, C):
    """Destination-passing: compute C = A @ B.T into preallocated bf16 buffer."""
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N = B.shape[0]

    props = torch.cuda.get_device_properties(A.device)
    num_sms = props.multi_processor_count

    # Tuned configuration for Hopper BF16 WGMMA
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_SIZE_M = 8

    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n
    
    # Persistent-ish: cap grid at SM count to limit launch overhead
    grid_size = min(num_sms * 4, total_tiles)
    grid = (grid_size,)

    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        GROUP_SIZE_M=GROUP_SIZE_M,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )