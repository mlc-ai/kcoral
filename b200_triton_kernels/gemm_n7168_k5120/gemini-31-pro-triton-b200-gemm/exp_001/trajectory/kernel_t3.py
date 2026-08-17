import torch
import triton
import triton.language as tl

# Configure Triton allocator to support device-side tensor descriptor creation for TMA
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_tma_configs():
    configs = []
    cases = [
        # High-performance shapes strictly utilizing BLOCK_K=64 to align natively 
        # with fifth-generation Tensor Cores (tcgen05.mma) for BF16 without layout conflicts.
        # Format: (BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, NUM_WARPS, WARP_SPEC)
        
        # Deep pipelines with standard shapes
        (128, 256, 64, 4, 8, True),
        (256, 128, 64, 4, 8, True),
        (128, 256, 64, 3, 8, True),
        (256, 128, 64, 3, 8, True),
        
        # Maximize occupancy with smaller blocks
        (128, 128, 64, 4, 4, True),
        (128, 128, 64, 5, 8, True),
        (128, 128, 64, 3, 4, True),
        
        # Baselines without warp specialization
        (128, 256, 64, 3, 8, False),
        (256, 128, 64, 3, 8, False),
        (128, 128, 64, 4, 8, False),
    ]
    for m, n, k, stages, warps, ws in cases:
        configs.append(triton.Config({
            'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 
            'GROUP_M': 8, 'NUM_STAGES': stages, 'WARP_SPEC': ws
        }, num_warps=warps, num_stages=stages, num_ctas=1))
        
        # Enable cluster-level scheduling for larger kernels 
        # (promotes L2 multicast / shared-memory distribution benefits on Blackwell)
        if m * n >= 32768 and ws:
            configs.append(triton.Config({
                'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 
                'GROUP_M': 8, 'NUM_STAGES': stages, 'WARP_SPEC': ws
            }, num_warps=warps, num_stages=stages, num_ctas=2))
            
    return configs

def get_ptr_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    ]


@triton.autotune(
    configs=get_tma_configs(),
    key=['M'],
)
@triton.jit
def _gemm_kernel_tma(
    A_ptr, B_ptr, C_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_STAGES: tl.constexpr, WARP_SPEC: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Grid swizzle to enhance L2 cache hit rate across B matrices.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    
    # Create Native Blackwell TMA Descriptors
    a_desc = tl.make_tensor_descriptor(
        A_ptr, shape=[M, K], strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        B_ptr, shape=[N, K], strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        C_ptr, shape=[M, N], strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N], padding_option="zero"
    )
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in tl.range(0, K // BLOCK_K, num_stages=NUM_STAGES, warp_specialize=WARP_SPEC):
        offs_k = k * BLOCK_K
        a = a_desc.load([offs_m, offs_k])
        
        # Logically loaded as [BLOCK_N, BLOCK_K] since B is stored natively as [N, K]
        b = b_desc.load([offs_n, offs_k])
        
        # b.T correctly bridges [BLOCK_N, BLOCK_K] -> [BLOCK_K, BLOCK_N], computing A @ B.T natively
        acc = tl.dot(a, b.T, acc, out_dtype=tl.float32)
        
    c_desc.store([offs_m, offs_n], acc.to(tl.bfloat16))


@triton.autotune(
    configs=get_ptr_configs(),
    key=['M'],
)
@triton.jit
def _gemm_kernel_ptr(
    A_ptr, B_ptr, C_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    
    mask_m = offs_m[:, None] < M
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K // BLOCK_K):
        a = tl.load(a_ptrs, mask=mask_m, other=0.0)
        # N and K strictly cover block dimension intervals seamlessly.
        b = tl.load(b_ptrs)
        
        acc = tl.dot(a, b.T, acc, out_dtype=tl.float32)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_m)


def is_aligned(tensor):
    """Verifies that the physical tensor conforms to TMA 16-byte alignment boundaries."""
    if tensor.dim() != 2:
        return False
    if tensor.data_ptr() % 16 != 0:
        return False
    if (tensor.stride(0) * tensor.element_size()) % 16 != 0:
        return False
    if tensor.stride(1) != 1:
        return False
    return True


def run(A, B, C):
    """
    Computes C = A @ B.T.
    A is [M, 5120], B is [7168, 5120] and C is natively preallocated [M, 7168].
    """
    torch.cuda.set_device(A.device)
    
    M = A.size(0)
    N = 7168
    K = 5120
    
    if M == 0:
        return
        
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    # Fast path: dispatch to Blackwell's native TMA descriptor pipeline if tensors conform.
    if is_aligned(A) and is_aligned(B) and is_aligned(C):
        _gemm_kernel_tma[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
        )
    else:
        _gemm_kernel_ptr[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
        )