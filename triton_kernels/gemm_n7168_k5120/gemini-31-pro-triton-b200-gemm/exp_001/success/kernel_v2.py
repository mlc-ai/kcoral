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
        # Proven safe TMA hardware pipeline mappings on SM100 using native warp specialization
        (128, 256, 64, 3, 8, True),
        (256, 128, 64, 3, 8, True),
        (128, 128, 64, 4, 8, True),
        (128, 128, 64, 3, 4, True),
        (64, 256, 64, 3, 8, True),
        
        # Aggressive deep pipelines explicitly bypassing warp specialization 
        # to cleanly avoid MLIR layout/swizzling compilation failures on `tcgen05.mma`
        (128, 256, 64, 4, 8, False),
        (256, 128, 64, 4, 8, False),
        (128, 128, 64, 5, 8, False),
        (128, 128, 64, 6, 8, False),
        (256, 256, 64, 3, 8, False),
        
        # Thin-M configurations for heavily variable batch size efficiency
        (64, 128, 64, 3, 4, False),
        (32, 256, 64, 3, 4, False),
        (64, 128, 64, 4, 4, False),
    ]
    
    for m, n, k, stages, warps, ws in cases:
        # Exploring L2 data-locality optimizations across standard tuning grids
        for group in [4, 8, 16]:
            configs.append(triton.Config({
                'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 
                'GROUP_M': group, 'NUM_STAGES': stages, 'WARP_SPEC': ws
            }, num_warps=warps, num_stages=stages))
            
    return configs

def get_ptr_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_STAGES': 3}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_STAGES': 3}, num_warps=4, num_stages=3),
    ]


@triton.autotune(
    configs=get_tma_configs(),
    key=['M'],
)
@triton.jit
def _gemm_kernel_tma(
    A_ptr, B_ptr, C_ptr,
    M,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_STAGES: tl.constexpr, WARP_SPEC: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # 2D Grid grouping to maintain contiguous L2 cache loads from operand B (held stationary across M block column)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    
    # Instantiate TMA hardware tensor descriptors internally 
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
    
    # Core loop natively accelerating bounds-free asynchronous TMA fetches integrated with TMEM
    for k in tl.range(0, K // BLOCK_K, num_stages=NUM_STAGES, warp_specialize=WARP_SPEC):
        offs_k = k * BLOCK_K
        a = a_desc.load([offs_m, offs_k])
        
        # Load block from physical B shape [N, K]
        b = b_desc.load([offs_n, offs_k])
        
        # `b.T` invokes a TMA-MMA coordinated swizzle organically projecting [BLOCK_K, BLOCK_N]
        acc = tl.dot(a, b.T, acc, out_dtype=tl.float32)
        
    c_desc.store([offs_m, offs_n], acc.to(tl.bfloat16))


@triton.autotune(
    configs=get_ptr_configs(),
    key=['M'],
)
@triton.jit
def _gemm_kernel_ptr(
    A_ptr, B_ptr, C_ptr,
    M,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_STAGES: tl.constexpr,
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
    
    # Boundary mask strictly required here as M is dynamically sized
    mask_m = offs_m[:, None] < M
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in tl.range(0, K // BLOCK_K, num_stages=NUM_STAGES):
        a = tl.load(a_ptrs, mask=mask_m, other=0.0)
        # N=7168 and K=5120 inherently encapsulate valid dimensions (no edge bounds processing required)
        b = tl.load(b_ptrs)
        
        acc = tl.dot(a, b.T, acc, out_dtype=tl.float32)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_m)


def is_aligned(tensor):
    """Verifies physical tensor strictly maps to natively required 16-byte alignment bounds on SM100 TMA hardware."""
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
    Computes standard destination-passing GEMM: C = A @ B.T.
    A mapped dynamically as [M, 5120], B is [7168, 5120] mapped into preallocated C [M, 7168].
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
    
    # Priority Fast Path: natively map supported tensors cleanly to SM100 TMA processing pipeline 
    if is_aligned(A) and is_aligned(B) and is_aligned(C):
        _gemm_kernel_tma[grid](
            A, B, C,
            M,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            N=N, K=K
        )
    else:
        _gemm_kernel_ptr[grid](
            A, B, C,
            M,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            N=N, K=K
        )