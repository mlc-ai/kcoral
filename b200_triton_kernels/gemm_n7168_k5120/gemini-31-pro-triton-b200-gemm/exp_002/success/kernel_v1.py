import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    configs = []
    # Test with and without warp specialization to find the optimal path
    for ws in [True, False]:
        for bm, bn, bk in [
            (128, 256, 128),
            (256, 128, 128),
            (128, 128, 128),
            (256, 256, 64),
            (128, 256, 64),
            (256, 128, 64),
            (128, 128, 64),
        ]:
            for ns in [3, 4, 5]:
                for nw in [4, 8]:
                    # Filter configs that would exceed Blackwell's ~227 KiB shared memory per CTA limit
                    shmem_bytes = (bm + bn) * bk * 2 * ns
                    if shmem_bytes > 220000:
                        continue
                    configs.append(triton.Config({
                        "BLOCK_M": bm,
                        "BLOCK_N": bn,
                        "BLOCK_K": bk,
                        "NUM_STAGES": ns,
                        "WARP_SPECIALIZE": ws
                    }, num_warps=nw))
    return configs

@triton.autotune(
    configs=get_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def gemm_kernel_tma(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr, WARP_SPECIALIZE: tl.constexpr
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    # Swizzle the launch sequence to improve L2 cache locality
    GROUP_M = 8
    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    
    # Device-side TMA descriptors creation
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Simple loop for TMA and warp specialization compatability
    for k0 in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([offs_m, k0 * BLOCK_K])
        b = b_desc.load([offs_n, k0 * BLOCK_K])
        acc = tl.dot(a, b.T, acc)
        
    c_offs_m = offs_m + tl.arange(0, BLOCK_M)
    c_offs_n = offs_n + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (c_offs_m[:, None] * stride_cm + c_offs_n[None, :] * stride_cn)
    c_mask = (c_offs_m[:, None] < M) & (c_offs_n[None, :] < N)
    
    # Store epilogue
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)

def run(A, B, C):
    """
    Computes a general matrix multiply C = A @ B.T.
    A is expected to be [M, K] logically and physically.
    B is expected to be [N, K] logically and physically.
    C is preallocated as [M, N].
    Leverages SM100 TMA memory acceleration and automatic warp-specialization.
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    
    gemm_kernel_tma[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )