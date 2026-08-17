import torch
import triton
import triton.language as tl

# Install Triton's descriptor allocator to provide TMA layout storage for device-created descriptors.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    configs = []
    # Use reliable block configurations known to map efficiently to Hopper WGMMA layouts.
    # Avoiding num_ctas > 1 and num_stages > 3 to prevent complex shared memory layouts
    # that could trigger MLIR legalization failures for transposed B operands.
    valid_combinations = [
        (128, 128, 64, 4, 3),
        (128, 128, 64, 8, 3),
        (128, 256, 64, 8, 3),
        (256, 128, 64, 8, 3),
        (128, 128, 128, 8, 3),
    ]
    for block_m, block_n, block_k, warps, stages in valid_combinations:
        configs.append(triton.Config(
            {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': 8},
            num_warps=warps, 
            num_stages=stages
        ))
    return configs

@triton.autotune(
    configs=get_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    
    # Construct device TMA descriptors per Hopper specifications.
    # B is physically stored as [N, K], yielding a descriptor that loads [BLOCK_N, BLOCK_K] tiles.
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero",
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero",
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Grid Swizzling: Improve L2 Reuse natively through layout ordering.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    # Accumulate precision in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Software-pipelined TMA loads + WGMMA dots
    for k_tile in range(num_k_tiles):
        offset_k = k_tile * BLOCK_K
        
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        
        # Hardware WGMMA execution on natively transposed B load
        acc = tl.dot(a, b.T, acc)
        
    # Cast back to target dtype and perform async TMA store
    c_desc.store([offset_m, offset_n], acc.to(dtype))

def run(A, B, C):
    """
    Compute GEMM C = A @ B.T directly into C.
    Hardware-Scheduled TMA implementation exploiting Hopper pipelines.
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, K_B = B.shape
    assert K == K_B, "Inner dimension mismatch."
    
    # Delegate tracking explicitly down to the active wave-schedulers.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    
    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )