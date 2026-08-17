import torch
import triton
import triton.language as tl

# Infrastructure storage allocator for device-created TMA descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # --- TMA (Tensor Memory Accelerator) Configurations ---
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': True}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': True}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': True}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': True}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'USE_TMA': True}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'USE_TMA': True}, num_stages=4, num_warps=8),

        # --- Standard cp.async Pointer Configurations ---
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': False}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': False}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': False}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': False}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'USE_TMA': False}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'USE_TMA': False}, num_stages=4, num_warps=8),
    ],
    key=['M']
)
@triton.jit
def _hybrid_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = N // BLOCK_N
    
    # L2-optimizing group swizzle layout
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    num_k_tiles = K // BLOCK_K
    dtype = c_ptr.dtype.element_ty

    # Branch entirely resolved at compile time for zero execution overhead.
    if USE_TMA:
        # Descriptor-backed TMA loads map to asynchronous hardware blocks
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
            block_shape=[BLOCK_M, BLOCK_N]
        )
        
        for k_tile in tl.range(0, num_k_tiles):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            # tl.dot dynamically orchestrates WGMMA Tensor Core instructions internally 
            acc = tl.dot(a, b.T, acc)
            
        c_desc.store([offset_m, offset_n], acc.to(dtype))
        
    else:
        # Standard loop, heavily optimized as K and N are constexpr multiples 
        offs_m = offset_m + tl.arange(0, BLOCK_M)
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
        
        # Only M is dynamic, so only mask_m is needed for loads/stores
        mask_m = offs_m < M
        
        for k_tile in tl.range(0, num_k_tiles):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs) # N and K boundaries are guaranteed safe
            
            acc = tl.dot(a, b.T, acc)
            
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
            
        c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        tl.store(c_ptrs, acc.to(dtype), mask=mask_m[:, None])

def run(A, B, C):
    """
    Computes C = A @ B.T where:
      A is of shape [M, 5120]
      B is of shape [7168, 5120]
      C is of shape [M, 7168]
    """
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    
    # Passing fixed constants structurally ensures the compiler applies unrolling heuristics
    N = 7168
    K = 5120
    
    def grid_fn(META):
        num_pid_m = triton.cdiv(M, META['BLOCK_M'])
        num_pid_n = N // META['BLOCK_N']
        return (num_pid_m * num_pid_n,)
        
    _hybrid_gemm_kernel[grid_fn](
        A, B, C,
        M,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        N=N, K=K
    )