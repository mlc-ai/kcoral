import torch
import triton
import triton.language as tl

# Standard Blackwell allocator for device-created tensor descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_autotune_config():
    configs = []
    # Test across both specialized and non-specialized warp layouts 
    for ws in [True, False]:
        # Typical stage and warp bounds for Blackwell.
        # We constrain shared memory per block across pipeline stages to remain under ~200 KiB.
        for stages, warps in [(3, 4), (4, 4), (5, 4), (3, 8), (4, 8), (5, 8)]:
            for block_m, block_n, block_k in [
                (128, 256, 64),
                (256, 128, 64),
                (128, 128, 64),
                (64, 256, 64),
                (256, 64, 64),
                (128, 128, 128),
                (256, 256, 64),
            ]:
                bytes_per_stage = (block_m * block_k + block_n * block_k) * 2
                if bytes_per_stage * stages <= 200 * 1024:
                    configs.append(triton.Config(
                        {
                            'BLOCK_M': block_m, 
                            'BLOCK_N': block_n, 
                            'BLOCK_K': block_k, 
                            'GROUP_M': 8, 
                            'WARP_SPECIALIZE': ws
                        },
                        num_stages=stages, 
                        num_warps=warps
                    ))
    return configs

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr
):
    # Blackwell TMA tensor descriptors handle boundary padding natively
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )

    # Calculate grid position and L2 cache swizzling
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
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Core loop utilizing automatic warp specialization when configured
    for k0 in tl.range(0, k_tiles, warp_specialize=WARP_SPECIALIZE):
        # TMA descriptor loads take scalar coordinates as offsets
        a = a_desc.load([offs_m, k0 * BLOCK_K])
        b = b_desc.load([offs_n, k0 * BLOCK_K])
        
        # Matrix B is effectively (BLOCK_N, BLOCK_K), transpsoing yields (BLOCK_K, BLOCK_N)
        acc = tl.dot(a, b.T, acc)
        
    c = acc.to(tl.bfloat16)
    
    # Store epilogue via masked standard pointers
    offs_cm = offs_m + tl.arange(0, BLOCK_M)
    offs_cn = offs_n + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

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
    
    # Merge block shapes for flat 1D grid launch
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )