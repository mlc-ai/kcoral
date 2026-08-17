import torch
import triton
import triton.language as tl

# Set Triton allocator for device-side tensor descriptor creation.
# This infrastructure storage is necessary to utilize Hopper TMA lowerings natively.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # TMA (Tensor Memory Accelerator) Configs 
        # Offloads memory bounds checking and moves data via async memory copy directly to shared memory
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': True}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': True}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': True}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'USE_TMA': True}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'USE_TMA': True}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'USE_TMA': True}, num_stages=4, num_warps=8),
        
        # Standard Pointer Configs (with eliminated masking)
        # Bypasses descriptor overhead entirely for shapes perfectly divisible by blocks
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': False}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': False}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': False}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'USE_TMA': False}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'USE_TMA': False}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'USE_TMA': False}, num_stages=4, num_warps=8),
    ],
    key=['M']
)
@triton.jit
def _gemm_kernel(
    A, B, C,
    M, 
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    N: tl.constexpr, 
    K: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    USE_TMA: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2 Cache-optimized grouped swizzling algorithm 
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    GROUP_M_ACTUAL = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % GROUP_M_ACTUAL)
    pid_n = (pid % num_pid_in_group) // GROUP_M_ACTUAL

    # Branch entirely removed at compile-time by `constexpr` annotation 
    if USE_TMA:
        # Hopper TMA lowers bounds checking logic to device intrinsic descriptors implicitly 
        a_desc = tl.make_tensor_descriptor(
            A, shape=[M, K], strides=[stride_am, stride_ak], block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
        )
        b_desc = tl.make_tensor_descriptor(
            B, shape=[N, K], strides=[stride_bn, stride_bk], block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
        )
        c_desc = tl.make_tensor_descriptor(
            C, shape=[M, N], strides=[stride_cm, stride_cn], block_shape=[BLOCK_M, BLOCK_N]
        )
        
        offs_m = pid_m * BLOCK_M
        offs_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k0 in range(tl.cdiv(K, BLOCK_K)):
            offs_k = k0 * BLOCK_K
            a = a_desc.load([offs_m, offs_k])
            
            # WGMMA prefers col-major layout for the right operand; physically matching `.T` layout triggers optimal fast path
            b = b_desc.load([offs_n, offs_k]) 
            acc = tl.dot(a, b.T, acc)
            
        c_desc.store([offs_m, offs_n], acc.to(C.dtype.element_ty))
        
    else:
        # Standard software-pipelined LDG/LDSM fallback without TMA overhead
        offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        
        a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # Scalar compile-time-friendly condition eliminating mask bottleneck for perfectly divisible dimensions (e.g. M=8192)
        if M % BLOCK_M == 0:
            for _ in range(0, tl.cdiv(K, BLOCK_K)):
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                acc = tl.dot(a, b.T, acc)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
                
            c_ptrs = C + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
            tl.store(c_ptrs, acc.to(C.dtype.element_ty))
        else:
            mask_m = offs_am[:, None] < M
            for _ in range(0, tl.cdiv(K, BLOCK_K)):
                a = tl.load(a_ptrs, mask=mask_m, other=0.0)
                b = tl.load(b_ptrs)
                acc = tl.dot(a, b.T, acc)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
                
            c_ptrs = C + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
            tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mask_m)

def run(A, B, C):
    """
    Computes general matrix multiplication C = A @ B.T highly optimized for Hopper architecture.
    Inputs:
      A: [M, K] bfloat16
      B: [N, K] bfloat16
    Outputs:
      C: [M, N] (Preallocated) bfloat16
    """
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = 7168
    K = 5120
    
    def grid(META):
        # A standard grid utilizes cuBLAS-like round-robin work distribution directly managed by SM hardware schedulers
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
        
    _gemm_kernel[grid](
        A, B, C,
        M, 
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        N=N, K=K,
    )