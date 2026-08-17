import torch
import triton
import triton.language as tl

# Install Triton's descriptor allocator to provide TMA layout storage for device-created descriptors.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


def get_configs():
    configs = []
    # Exhaustive hardware-aligned candidates to hit optimal memory constraints and cluster dimensions
    valid_combinations = [
        # (block_m, block_n, block_k, warps, stages, ctas)
        (256, 128, 64, 8, 3, 1),
        (256, 128, 64, 8, 4, 1),
        (128, 256, 64, 8, 3, 1),
        (128, 256, 64, 8, 4, 1),
        (128, 128, 128, 8, 3, 1),
        (128, 128, 128, 4, 3, 1),
        (256, 256, 64, 8, 3, 1),
        (128, 128, 64, 8, 4, 1),
        (128, 128, 64, 8, 5, 1),
        (128, 128, 64, 4, 4, 1),
        
        # Thread Block Cluster configs to heavily capitalize on H100 L2 Cache pooling
        (128, 256, 64, 8, 4, 2),
        (128, 256, 64, 8, 4, 4),
        (256, 128, 64, 8, 4, 2),
        (256, 128, 64, 8, 4, 4),
        (128, 128, 128, 8, 3, 2),
        (128, 128, 128, 8, 3, 4),
        (256, 256, 64, 8, 3, 2),
        (256, 256, 64, 8, 3, 4),
    ]
    for block_m, block_n, block_k, warps, stages, ctas in valid_combinations:
        configs.append(triton.Config(
            {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': 8},
            num_warps=warps, 
            num_stages=stages, 
            num_ctas=ctas
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
    # Construct device TMA descriptors per Hopper specifications
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
    
    # Grid Swizzling: Improve L2 Reuse natively through layout ordering
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Software-pipelined TMA loads + WGMMA dots
    for k_tile in range(num_k_tiles):
        offset_k = k_tile * BLOCK_K
        
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        
        acc = tl.dot(a, b.T, acc)
        
    c_desc.store([offset_m, offset_n], acc.to(c_ptr.dtype.element_ty))


def run(A, B, C):
    """
    Compute GEMM C = A @ B.T directly into C.
    Hardware-Scheduled TMA implementation aligned with H100 optimal pipelines.
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, K_B = B.shape
    assert K == K_B, "Inner dimension mismatch."
    
    # Delegate tracking explicitly down to the active wave-schedulers, removing persistent looping.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    
    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )