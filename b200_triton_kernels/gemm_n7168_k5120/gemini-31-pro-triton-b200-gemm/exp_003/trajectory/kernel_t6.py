import torch
import triton
import triton.language as tl

# Standard allocator for device-created tensor descriptors on Blackwell
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_autotune_config():
    configs = []
    # Explore multiple dimensions of the Blackwell optimization space:
    # 1. CTAs clustering (num_ctas): enables L2 cache locality and TMA multicast across clusters.
    # 2. Warp specialization (WARP_SPECIALIZE): enables asynchronous overlap of TMA loads and Tensor Core MMA.
    # 3. Pipeline depth (stages) and thread count (warps).
    for ctas in [1, 2, 4, 8]:
        for ws in [True, False]:
            for stages in [3, 4, 5]:
                for warps in [4, 8]:
                    for block_m, block_n, block_k in [
                        (128, 256, 128),
                        (256, 128, 128),
                        (128, 128, 128),
                        (256, 256, 64),
                        (128, 256, 64),
                        (256, 128, 64),
                        (128, 128, 64),
                    ]:
                        # A bfloat16 is 2 bytes. Calculate buffer sizes for A and B.
                        bytes_per_stage = (block_m * block_k + block_n * block_k) * 2
                        
                        # Limit to ~200 KiB to comfortably fit within Blackwell's 228 KiB SM limit 
                        # while leaving room for warp specialization overhead.
                        if bytes_per_stage * stages <= 200 * 1024:
                            configs.append(triton.Config(
                                {
                                    'BLOCK_M': block_m, 
                                    'BLOCK_N': block_n, 
                                    'BLOCK_K': block_k, 
                                    'GROUP_M': 8, 
                                    'WARP_SPECIALIZE': ws,
                                    'NUM_STAGES': stages
                                },
                                num_stages=stages, 
                                num_warps=warps,
                                num_ctas=ctas
                            ))
    return configs

@triton.autotune(
    configs=get_autotune_config(),
    key=['M'],
)
@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr
):
    # Construct TMA descriptors for A, B, and C on device. 
    # TMA natively handles out-of-bounds padding during loads, and ignores out-of-bounds stores.
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

    # Calculate grid position with L2 cache Swizzling.
    # Grouping CTAs improves locality, and perfectly aligns with TMA multicast when num_ctas > 1.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = num_pid_m - first_pid_m
    if group_size_m > GROUP_M:
        group_size_m = GROUP_M
        
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    
    # FP32 Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Core unrolled inner loop. By pipelining num_stages and toggling WARP_SPECIALIZE, 
    # the Blackwell hardware can asynchronously overlap descriptor fetches, epilogue, and MMA steps.
    for k0 in tl.range(0, k_tiles, warp_specialize=WARP_SPECIALIZE, num_stages=NUM_STAGES):
        # Descriptors accept scalar coordinates.
        a = a_desc.load([offs_m, k0 * BLOCK_K])
        b = b_desc.load([offs_n, k0 * BLOCK_K])
        
        # Logically B is (N, K). Matrix `b` is loaded as (BLOCK_N, BLOCK_K).
        # We transact `b.T` (BLOCK_K, BLOCK_N) to complete the gemm orientation.
        acc = tl.dot(a, b.T, acc)
        
    # Downcast safely post-reduction
    c = acc.to(tl.bfloat16)
    
    # Efficient descriptor-based writeback; TMA handles any partial block boundary safely.
    c_desc.store([offs_m, offs_n], c)

def run(A, B, C):
    """
    Compute C = A @ B.T
    
    Args:
        A: Tensor of shape (M, K)
        B: Tensor of shape (N, K)
        C: Preallocated destination tensor of shape (M, N)
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    # Flat 1D grid launch
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    gemm_kernel[grid](
        A, B, C,
        M, N=N, K=K,
        stride_am=A.stride(0), stride_ak=A.stride(1),
        stride_bn=B.stride(0), stride_bk=B.stride(1),
        stride_cm=C.stride(0), stride_cn=C.stride(1)
    )