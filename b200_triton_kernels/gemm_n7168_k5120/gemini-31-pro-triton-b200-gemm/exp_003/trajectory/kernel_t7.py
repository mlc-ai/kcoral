import torch
import triton
import triton.language as tl

# Standard Blackwell allocator for device-created tensor descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_autotune_config():
    configs = []
    # Explore clustered CTAs, warp specialization, and pipeline depths
    for ctas in [1, 2, 4]:
        for ws in [True, False]:
            for warps in [4, 8]:
                # We strictly avoid block combinations (like BLOCK_N=256 and BLOCK_K=64)
                # that map to tcgen05.mma layouts currently unsupported by the SM100 compiler backend.
                for block_m, block_n, block_k, stages in [
                    (128, 128, 128, 3),
                    (128, 128, 64, 4),
                    (128, 256, 128, 2),
                    (256, 128, 128, 2),
                    (64, 128, 128, 4),
                    (128, 64, 128, 4),
                    (64, 64, 128, 5),
                ]:
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
    # Blackwell TMA tensor descriptors handle loading and padding asynchronously.
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

    # Grid mapping with L2 cache swizzling
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
    
    # High-precision accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Asynchronous load-compute pipeline utilizing 5th-gen Tensor Cores and TMA
    for k0 in tl.range(0, k_tiles, warp_specialize=WARP_SPECIALIZE, num_stages=NUM_STAGES):
        a = a_desc.load([offs_m, k0 * BLOCK_K])
        b = b_desc.load([offs_n, k0 * BLOCK_K])
        
        # Orient the B tile to (BLOCK_K, BLOCK_N) matching standard MMA conventions
        acc = tl.dot(a, b.T, acc)
        
    c = acc.to(tl.bfloat16)
    
    # Store epilogue directly via the TMA unit
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
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    gemm_kernel[grid](
        A, B, C,
        M, N=N, K=K,
        stride_am=A.stride(0), stride_ak=A.stride(1),
        stride_bn=B.stride(0), stride_bk=B.stride(1),
        stride_cm=C.stride(0), stride_cn=C.stride(1)
    )