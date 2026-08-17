import torch
import triton
import triton.language as tl

def get_configs():
    configs = []
    # Exhaustive sweep of large Hopper-optimized configurations
    # These tile sizes focus on saturating WGMMA usage and perfectly hiding memory latency
    for block_m, block_n, block_k, num_warps, num_stages in [
        # Large tile max-throughput configs
        (128, 256, 64, 8, 3),
        (128, 256, 64, 8, 4),
        (256, 128, 64, 8, 3),
        (256, 128, 64, 8, 4),
        (128, 128, 128, 8, 3),
        
        # Balanced tile configs 
        (128, 128, 64, 4, 3),
        (128, 128, 64, 4, 4),
        (128, 128, 64, 4, 5),
        (128, 128, 64, 8, 3),
        (128, 128, 64, 8, 4),
        (128, 128, 64, 8, 5),
        
        # Smaller block chunks optimizing dynamic load-balancing (tail latency prevention)
        (64, 256, 64, 4, 3),
        (64, 256, 64, 4, 4),
        (256, 64, 64, 4, 3),
        (256, 64, 64, 4, 4),
        
        (64, 128, 64, 4, 3),
        (64, 128, 64, 4, 4),
        (64, 128, 64, 4, 5),
        
        (128, 64, 64, 4, 3),
        (128, 64, 64, 4, 4),
        (128, 64, 64, 4, 5),
        
        (128, 64, 128, 4, 3),
        (64, 128, 128, 4, 3),
    ]:
        for group_m in [8, 16]:
            configs.append(triton.Config(
                {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': group_m},
                num_warps=num_warps, num_stages=num_stages
            ))
    return configs

@triton.autotune(
    configs=get_configs(),
    key=['M', 'N', 'K']
)
@triton.jit
def _gemm_pointer(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    
    # 1. Hierarchical Grid Swizzling to Maximize L2 Cache Reuse 
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 2. Structure Affine Pointer Blocks
    # A is row-major loaded structurally as [BLOCK_M, BLOCK_K]. Last dim accesses contiguous chunks.
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    
    # B is physically [N, K]. Stride BK is 1. We map it structurally as [BLOCK_N, BLOCK_K] 
    # to maintain strict memory contiguous alignment (coalesced reads) on its last dimension block slice.
    b_ptrs = b_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # M is dynamically sized, requires boundary safety check.
    mask_m = offs_m[:, None] < M

    # 3. Main Product Loop
    for _ in range(0, K, BLOCK_K):
        # We drop condition masks for dimensions strictly proven bound-divisible by constraints
        a = tl.load(a_ptrs, mask=mask_m, other=0.0)
        b = tl.load(b_ptrs)
        
        # Virtual `b.T` transpose here dynamically reorders into [BLOCK_K, BLOCK_N] to validate A @ B.T structure. 
        # Resolves completely inside hardware registers/smem maintaining WGMMA operand legality natively.
        acc = tl.dot(a, b.T, acc)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 4. Epilogue
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask_m)

def run(A, B, C):
    """
    Computes generalized matrix multiply C = A @ B.T.
    
    Inputs:
    A: [M, K]
    B: [N, K] 
    C: [M, N] preallocated definition output tensor
    """
    torch.cuda.set_device(A.device)
    
    M = A.size(0)
    N = B.size(0)
    K = A.size(1)

    def grid_fn(meta):
        # Hardware scheduled layout ensures load-balancing eliminates the tail straggler penalty 
        # faced by simple persistent software loops.
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    _gemm_pointer[grid_fn](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )