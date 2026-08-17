import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Configure Triton allocator to support device-side tensor descriptor creation for TMA
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_tma_configs():
    configs = []
    cases = [
        # High-performance geometries explicitly optimized for SM100 TMA and tcgen05 MMA lowering.
        # Strict adherence to BLOCK_K=64 natively maps to 128-byte swizzled bfloat16 layouts.
        # Format: (BLOCK_M, BLOCK_N, BLOCK_K, LOOP_STAGES, NUM_WARPS, WARP_SPEC, NUM_CTAS)
        
        # Deeply pipelined pipelines for massive blocks
        (256, 128, 64, 4, 8, True, 2),
        (256, 128, 64, 4, 8, True, 4),
        (128, 256, 64, 4, 8, True, 2),
        (128, 256, 64, 4, 8, True, 4),
        
        # Broad accumulation footprints
        (256, 256, 64, 3, 16, True, 2),
        (256, 256, 64, 3, 8, True, 2),
        
        # Intensive pipelines avoiding clustered boundaries
        (128, 128, 64, 6, 8, True, 2),
        (128, 128, 64, 5, 8, True, 1),
        
        # Baselines without warp specialization (serves well to offset spill register pressure limits)
        (256, 128, 64, 4, 8, False, 1),
        (128, 256, 64, 4, 8, False, 1),
        (256, 256, 64, 3, 16, False, 1),
        (128, 128, 64, 5, 8, False, 1),
    ]
    for m, n, k, stages, warps, ws, ctas in cases:
        # Guarantee we stay strictly within Blackwell's maximum 228 KiB shared memory per CTA limit
        smem_req = stages * (m * k + n * k) * 2  # 2 bytes per bfloat16
        if smem_req <= 227 * 1024:
            configs.append(triton.Config({
                'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 
                'GROUP_M': 8, 'LOOP_STAGES': stages, 'WARP_SPEC': ws
            }, num_warps=warps, num_stages=stages, num_ctas=ctas))
    return configs

def get_ptr_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'LOOP_STAGES': 4}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'LOOP_STAGES': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'LOOP_STAGES': 4}, num_warps=4, num_stages=4),
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
    GROUP_M: tl.constexpr, LOOP_STAGES: tl.constexpr, WARP_SPEC: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # 2D Grid mapping targeting L2 cache locality reuse
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    
    # Device-created Standard Blackwell TMA Descriptors
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
    
    for k in tl.range(0, K // BLOCK_K, num_stages=LOOP_STAGES, warp_specialize=WARP_SPEC):
        offs_k = k * BLOCK_K
        a = a_desc.load([offs_m, offs_k])
        # Logically loaded as [BLOCK_N, BLOCK_K] effectively bridging physical layout [N, K]
        b = b_desc.load([offs_n, offs_k])
        
        # Hardware transposition explicitly achieves B.T transposition organically over native tcgen05 TMEM pathways
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
    GROUP_M: tl.constexpr, LOOP_STAGES: tl.constexpr,
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
    
    for k in tl.range(0, K // BLOCK_K, num_stages=LOOP_STAGES):
        a = tl.load(a_ptrs, mask=mask_m, other=0.0)
        # N and K are strictly mapped bounds with purely clean divisions, avoiding extra mask handling logic.
        b = tl.load(b_ptrs)
        
        acc = tl.dot(a, b.T, acc, out_dtype=tl.float32)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_m)


def is_aligned(tensor):
    """Verifies that physical tensors comply with Native TMA hardware 16-byte alignment bounds."""
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
    A is [M, 5120], B is [7168, 5120] and C is [M, 7168] natively preallocated.
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
    
    # Fast path: dispatch safely to hardware-accelerated TMA + Tensor Core generation pipeline
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