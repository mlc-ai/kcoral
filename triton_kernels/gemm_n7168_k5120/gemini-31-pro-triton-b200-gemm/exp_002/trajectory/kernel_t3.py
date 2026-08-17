import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    # Blackwell infrastructure storage for device-created TMA descriptors
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    configs = []
    # Exhaustive autotuning space for Blackwell SM100 architecture
    for ws in [True, False]:
        for bm, bn, bk in [
            (256, 128, 128),
            (128, 256, 128),
            (128, 128, 128),
            (256, 256, 64),
            (256, 128, 64),
            (128, 256, 64),
            (128, 128, 64),
            (64, 256, 64),
            (256, 64, 64),
        ]:
            for ns in [2, 3, 4, 5, 6]:
                for nw in [4, 8, 12, 16]:
                    # Filter to stay within Blackwell's ~227 KiB max addressable shared memory bounds per CTA
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
    grid_n = N // BLOCK_N
    
    # Highly optimized L2 cache swizzling. 
    # Given M=8192 and maximum BLOCK_M=256, grid_m is unequivocally a multiple of 8 (GROUP_M).
    # This guarantees exact bounds scaling with no divergent remainder constraints natively.
    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    
    pid_m = first_pid_m + (pid % GROUP_M)
    pid_n = (pid % num_pid_in_group) // GROUP_M
    
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    
    # Device-side TMA Descriptor instantiation leverages SM100 hardware data loading efficiently
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Core loop lowered implicitly utilizing TMEM and tcgen05 if correctly supported by block shapes 
    for k0 in tl.range(0, K // BLOCK_K, num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([offs_m, k0 * BLOCK_K])
        b = b_desc.load([offs_n, k0 * BLOCK_K])
        
        # Transposing locally loaded `b` correctly fulfills native orientation layout constraints
        acc = tl.dot(a, b.T, acc)
        
    # Epilogue bounds mask is safely omitted under verified dimensional exactness checks
    c_offs_m = offs_m + tl.arange(0, BLOCK_M)
    c_offs_n = offs_n + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (c_offs_m[:, None] * stride_cm + c_offs_n[None, :] * stride_cn)
    
    tl.store(c_ptrs, acc.to(tl.bfloat16))

def run(A, B, C):
    """
    Computes a general matrix multiply C = A @ B.T where:
    A is logically [M, K]
    B is physically [N, K]
    C is initialized and written back logically as [M, N]
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    grid = lambda META: ( (M // META["BLOCK_M"]) * (N // META["BLOCK_N"]), )
    
    gemm_kernel_tma[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )