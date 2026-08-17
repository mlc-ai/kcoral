import torch
import triton
import triton.language as tl

# Configure Triton allocator for device-side tensor descriptors (Hopper TMA requirement)
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    configs = []
    # Rigorous bounds to avoid MLIR Hopper WGMMA layout bugs for bfloat16.
    # Restricting to max block size of 128 guarantees perfectly legal instruction shapes 
    # without exceeding the WGMMA operand constraints while preserving top-tier performance.
    for num_stages in [3, 4, 5]:
        for block_m, block_n, block_k in [
            (128, 128, 128),
            (128, 128, 64),
            (64, 128, 128),
            (128, 64, 128),
            (64, 128, 64),
            (128, 64, 64),
            (64, 64, 64),
        ]:
            for num_warps in [4, 8]:
                # Hopper Shared Memory hardware limit per CTA is ~228 KiB. 
                bytes_per_stage = (block_m + block_n) * block_k * 2
                if bytes_per_stage * num_stages <= 220000:
                    configs.append(triton.Config(
                        {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': 8},
                        num_warps=num_warps, num_stages=num_stages
                    ))
    return configs

@triton.autotune(
    configs=get_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _tma_gemm(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    
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
    
    # L2 Cache Swizzle mapping assigns logically neighboring tiles to execute simultaneously on the cluster
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = num_pid_m - first_pid_m
    if group_size_m > GROUP_M:
        group_size_m = GROUP_M
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k_tile in range(tl.cdiv(K, BLOCK_K)):
        offset_k = k_tile * BLOCK_K
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        
        # WGMMA operations on Hopper thrive when the right operand natively corresponds 
        # to physical row-major storage traversed as col-major logic, accomplished natively via b.T
        acc = tl.dot(a, b.T, acc)

    c_desc.store([offset_m, offset_n], acc.to(dtype))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_pointer_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Safely fallback to memory pointer logic when bounds and alignment prohibit TMA capabilities
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)

    m_mask = offs_m[:, None] < M
    n_mask = offs_n[:, None] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_ptrs, mask=m_mask, other=0.0)
        b = tl.load(B_ptrs, mask=n_mask, other=0.0)

        acc = tl.dot(a, b.T, acc)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    C_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(C_ptrs, acc.to(C.dtype.element_ty), mask=m_mask & n_mask)

def run(A, B, C):
    """
    Compute GEMM C = A @ B.T into a preallocated output tensor C.
    A is [M, K], B is [N, K], C is [M, N].
    """
    if A.numel() == 0 or B.numel() == 0 or C.numel() == 0:
        return
        
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, K_b = B.shape
    
    # TMA alignment requirement validation for Hopper Descriptor Path
    tma_supported = True
    if A.stride(1) != 1 or B.stride(1) != 1 or C.stride(1) != 1:
        tma_supported = False
    if (A.stride(0) * A.element_size()) % 16 != 0:
        tma_supported = False
    if (B.stride(0) * B.element_size()) % 16 != 0:
        tma_supported = False
    if (C.stride(0) * C.element_size()) % 16 != 0:
        tma_supported = False

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    if tma_supported:
        _tma_gemm[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1)
        )
    else:
        _gemm_pointer_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1)
        )