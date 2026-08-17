import torch
import triton
import triton.language as tl

_allocator_set = False

@triton.autotune(
    configs=[
        # 128x128x64: 32KB per stage. Safe up to 6 stages (192KB shared memory)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=6),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 16}, num_warps=4, num_stages=5),
        
        # 256x128x64: 48KB per stage. Safe up to 4 stages (192KB)
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 16}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        
        # 128x256x64: 48KB per stage. Safe up to 4 stages (192KB)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 16}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        
        # 128x128x128 (high compute intensity): 64KB per stage. Safe up to 3 stages (192KB)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 16}, num_warps=8, num_stages=3),
        
        # 256x128x128 / 128x256x128 (maxed out limits): 96KB per stage. Safe up to 2 stages (192KB)
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=2),
    ],
    key=["M"],
)
@triton.jit
def _tma_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M,
    stride_am, stride_bn, stride_cm,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Device Descriptors for optimal Hopper TMA execution
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, 1], block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[stride_bn, 1], block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[stride_cm, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2-Aware grouped swizzling algorithm
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    # We use a standard Python range loop alongside TMA.
    # This invokes the Hopper pipeliner to naturally emit TMA loads & WGMMA math without
    # triggering the bug present in `warp_specialize`/`tl.range` configurations.
    for k_tile in range(0, num_k_tiles):
        offset_k = k_tile * BLOCK_K
        
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        
        # Hopper fully supports b.T natively as long as `b` comes loaded exactly as [BLOCK_N, BLOCK_K] 
        acc = tl.dot(a, b.T, acc)
        
    c_desc.store([offset_m, offset_n], acc.to(c_ptr.dtype.element_ty))


def run(A, B, C):
    global _allocator_set
    torch.cuda.set_device(A.device)
    
    if not _allocator_set:
        def alloc_fn(size: int, alignment: int, stream):
            # Safe host-side allocator for internal TMA device descriptors 
            return torch.empty(size, device=torch.cuda.current_device(), dtype=torch.int8)
        triton.set_allocator(alloc_fn)
        _allocator_set = True

    M = A.shape[0]
    
    # Fixed constrained inputs via Prompt Contract
    N = 7168
    K = 5120
    
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    
    _tma_gemm_kernel[grid](
        A, B, C,
        M,
        A.stride(0), B.stride(0), C.stride(0),
        N=N, K=K
    )