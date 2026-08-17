import torch
import triton
import triton.language as tl

# Set Triton allocator for device-created TMA descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    configs = []
    # Tuned explicitly to saturate SM100 TMA and TMEM bandwidth limits
    # Ensures no configuration exceeds Blackwell's ~227 KiB max shared memory limit
    for ws in [True, False]:
        for bm, bn, bk, ns, nw in [
            (256, 256, 64, 2, 8),
            (128, 256, 128, 2, 8),
            (256, 128, 128, 2, 8),
            (128, 128, 128, 3, 8),
            (128, 256, 64, 3, 8),
            (256, 128, 64, 3, 8),
            (128, 128, 64, 4, 8),
        ]:
            configs.append(triton.Config({
                'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk, 
                'NUM_STAGES': ns, 'WARP_SPECIALIZE': ws, 'GROUP_M': 8
            }, num_warps=nw))
    return configs

@triton.autotune(
    configs=get_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def gemm_kernel_tma_store(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr, WARP_SPECIALIZE: tl.constexpr, GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    # Block scheduling L2 cache swizzling (Groups M iterations together to reuse B's rows/columns)
    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # TMA descriptors natively index memory in coordinates vs byte offset arithmetic
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    
    # Device-side TMA descriptors created safely inside JIT
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N], padding_option="zero"
    )
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Fully unrolled loop leveraging automatic warp specialization asynchronously
    for k0 in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([offs_m, k0 * BLOCK_K])
        b = b_desc.load([offs_n, k0 * BLOCK_K])
        
        # 'b' physically [N, K], loaded [BLOCK_N, BLOCK_K], supplied as '.T' for correct Tensor Core orientation
        acc = tl.dot(a, b.T, acc)
        
    # Asynchronous TMA store back directly drops any out-of-bounds bounds natively (no masking required!)
    c_desc.store([offs_m, offs_n], acc.to(tl.bfloat16))

def run(A, B, C):
    """
    Computes a general matrix multiply C = A @ B.T.
    A is expected to be [M, K] logically and physically.
    B is expected to be [N, K] logically and physically.
    C is preallocated as [M, N].
    Leverages SM100 TMA memory acceleration for loads and stores, plus automatic warp-specialization.
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    
    gemm_kernel_tma_store[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )