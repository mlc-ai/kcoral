import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

def pre_hook(kwargs):
    # Establish host-side TMA descriptors dynamically for the autotuner's tile dimensions.
    # We sidestep the runtime device allocator completely overhead.
    kwargs["A_desc"] = TensorDescriptor.from_tensor(kwargs["A"], [kwargs["BLOCK_M"], kwargs["BLOCK_K"]])
    kwargs["B_desc"] = TensorDescriptor.from_tensor(kwargs["B"], [kwargs["BLOCK_N"], kwargs["BLOCK_K"]])

def get_configs():
    configs = []
    # Exhaustive testing tuned explicitly for Blackwell SM100 architecture and L2 caches
    for ws in [True, False]:
        for bm, bn, bk in [
            (256, 128, 128),
            (128, 256, 128),
            (128, 128, 128),
            (256, 256, 64),
            (128, 256, 64),
            (256, 128, 64),
            (128, 128, 64),
        ]:
            for ns in [2, 3, 4, 5]:
                for nw in [4, 8]:
                    # Keep shared memory allocations beneath Blackwell's 227 KiB constraint
                    shmem_bytes = (bm + bn) * bk * 2 * ns
                    if shmem_bytes <= 227000:
                        configs.append(triton.Config({
                            "BLOCK_M": bm,
                            "BLOCK_N": bn,
                            "BLOCK_K": bk,
                            "NUM_STAGES": ns,
                            "WARP_SPECIALIZE": ws,
                            "GROUP_M": 8,
                        }, num_warps=nw, num_stages=ns))
    return configs

@triton.autotune(
    configs=get_configs(),
    key=["M", "N", "K"],
    pre_hook=pre_hook,
)
@triton.jit
def gemm_kernel_tma_host(
    A, B, C_ptr,            # PyTorch inputs only passed so `pre_hook` can intercept them efficiently 
    A_desc, B_desc,         # Descriptors built on host minimizing launch latency
    M, N, K,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr, WARP_SPECIALIZE: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    # Swizzling matrix block indexing ensures the L2 Cache re-utilizes data effectively
    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # TMA descriptors natively index memory in coordinates vs ptr byte arithmetic
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Warp specialization operates at max peak inside a simple clean loop 
    for k0 in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        a = A_desc.load([offs_m, k0 * BLOCK_K])
        b = B_desc.load([offs_n, k0 * BLOCK_K])
        
        # Orient Matrix 'B' natively via `.T` property mimicking SM100 architecture needs
        acc = tl.dot(a, b.T, acc)
        
    c_offs_m = offs_m + tl.arange(0, BLOCK_M)
    c_offs_n = offs_n + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + (c_offs_m[:, None] * stride_cm + c_offs_n[None, :] * stride_cn)
    c_mask = (c_offs_m[:, None] < M) & (c_offs_n[None, :] < N)
    
    # Explicit pointer epilogue handles boundary margins securely
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)

def run(A, B, C):
    """
    Computes a general matrix multiply C = A @ B.T.
    A is logically/physically aligned as [M, K]
    B is logically/physically aligned as [N, K]
    C is initialized and written back logically as [M, N]
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    
    gemm_kernel_tma_host[grid](
        A=A, B=B, C_ptr=C,
        A_desc=None, B_desc=None,
        M=M, N=N, K=K,
        stride_cm=C.stride(0), stride_cn=C.stride(1)
    )