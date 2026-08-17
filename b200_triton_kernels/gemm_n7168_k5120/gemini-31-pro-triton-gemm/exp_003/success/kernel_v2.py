import torch
import triton
import triton.language as tl

# Install Triton's descriptor allocator to support device-side TMA descriptors
def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

if not hasattr(triton, "_alloc_setup_done"):
    triton.set_allocator(_alloc_fn)
    triton._alloc_setup_done = True

@triton.autotune(
    configs=[
        # Large optimal TMA WG tiles (Tuned for H100 Hopper SM90)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        
        # Medium tiles for higher occupancy limit scaling
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        
        # High K-compute density tiles
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 256, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M'],
)
@triton.jit
def _gemm_device_tma_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2 Cache Swizzling for better data reuse
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    
    diff = num_pid_m - first_pid_m
    group_size_m = tl.where(diff < GROUP_M, diff, GROUP_M)
    
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    # Hopper Device TMA Descriptors 
    # Constructed dynamically per program avoiding host-signature baking conflicts
    a_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[N, K],
        strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N]
    )
    
    # Accumulate in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    # K is guaranteed to be a multiple of BLOCK_K for this GEMM layer
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        offset_k = k0 * BLOCK_K
        
        # Async TMA hardware handles loading and padding boundary conditions 
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        
        # B is loaded physically transposed, so logically apply B.T in dot operation
        acc = tl.dot(a, b.T, acc)
        
    # Standard format conversion and TMA store
    c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))

def run(A, B, C):
    """
    Computes general matrix multiply C = A @ B.T where:
    A: [M, K]
    B: [N, K]
    C: [M, N]
    With N = 7168, K = 5120.
    """
    torch.cuda.set_device(A.device)
    
    # Problem definition boundaries
    M = A.shape[0]
    N = 7168
    K = 5120
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    _gemm_device_tma_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )