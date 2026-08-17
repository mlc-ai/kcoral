import torch
import triton
import triton.language as tl

# Set Triton allocator for device-created TMA descriptors (required on SM100)
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    configs = []
    # Test with and without warp specialization for optimal TMA / TMEM usage
    for ws in [True, False]:
        for bm, bn, bk in [
            (256, 128, 128),
            (128, 256, 128),
            (128, 128, 128),
            (256, 256, 64),
            (128, 256, 64),
            (256, 128, 64),
        ]:
            for ns in [2, 3, 4, 5]:
                for nw in [4, 8]: # Blackwell optimally runs with 4 or 8 warps for standard fp16/bf16 tcgen05
                    # Enforce the SM100 ~227 KiB shared memory hard limit 
                    shmem = (bm + bn) * bk * 2 * ns
                    if shmem > 227000:
                        continue
                        
                    configs.append(triton.Config({
                        "BLOCK_M": bm,
                        "BLOCK_N": bn,
                        "BLOCK_K": bk,
                        "NUM_STAGES": ns,
                        "WARP_SPECIALIZE": ws,
                        "GROUP_M": 8
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
    NUM_STAGES: tl.constexpr, WARP_SPECIALIZE: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    # Standard robust L2 cache swizzling
    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    
    # Device-side TMA Descriptors created safely inside JIT
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Simple unrolled TMA loop. Automatic warp specialization safely handles producer/consumer mapping
    for k0 in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([offs_m, k0 * BLOCK_K])
        b = b_desc.load([offs_n, k0 * BLOCK_K])
        
        # 'b' physically [N, K], loaded [BLOCK_N, BLOCK_K], supplied as '.T' for correct Tensor Core orientation
        acc = tl.dot(a, b.T, acc)
        
    c_offs_m = offs_m + tl.arange(0, BLOCK_M)
    c_offs_n = offs_n + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (c_offs_m[:, None] * stride_cm + c_offs_n[None, :] * stride_cn)
    c_mask = (c_offs_m[:, None] < M) & (c_offs_n[None, :] < N)
    
    # Cast to final precision and store via regular robust pointer masking
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)

def run(A, B, C):
    """
    Computes a general matrix multiply C = A @ B.T where:
    A is logically [M, K]
    B is physically [N, K]
    C is preallocated as [M, N]
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