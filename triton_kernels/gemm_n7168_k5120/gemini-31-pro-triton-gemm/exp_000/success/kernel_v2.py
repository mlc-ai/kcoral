import torch
import triton
import triton.language as tl

# Set Triton allocator for device-side tensor descriptor creation.
# This infrastructure storage is required for Hopper TMA descriptors.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # PATH 1: TMA Persistent Warp Specialized
        # Maximize L2 cache locality and overlap descriptor loads with specialized consumer warps
        # Max stages bounded by 228KB Shared Memory limit
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': True, 'WARP_SPECIALIZE': True}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': True, 'WARP_SPECIALIZE': True}, num_stages=2, num_warps=8),
        
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': True, 'WARP_SPECIALIZE': True}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': True, 'WARP_SPECIALIZE': True}, num_stages=3, num_warps=8),
        
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'USE_TMA': True, 'WARP_SPECIALIZE': True}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'USE_TMA': True, 'WARP_SPECIALIZE': True}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'USE_TMA': True, 'WARP_SPECIALIZE': True}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'USE_TMA': True, 'WARP_SPECIALIZE': True}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'USE_TMA': True, 'WARP_SPECIALIZE': True}, num_stages=4, num_warps=4),

        # PATH 2: TMA Persistent Standard (warp_specialize=False)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': True, 'WARP_SPECIALIZE': False}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': True, 'WARP_SPECIALIZE': False}, num_stages=2, num_warps=8),
        
        # PATH 3: Standard Pointers
        # Relies on dynamic hardware scheduler and classic software pipelined LDSM+WGMMA
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': False, 'WARP_SPECIALIZE': False}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': False, 'WARP_SPECIALIZE': False}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'USE_TMA': False, 'WARP_SPECIALIZE': False}, num_stages=3, num_warps=8),
        
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'USE_TMA': False, 'WARP_SPECIALIZE': False}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'USE_TMA': False, 'WARP_SPECIALIZE': False}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'USE_TMA': False, 'WARP_SPECIALIZE': False}, num_stages=4, num_warps=4),
    ],
    key=['M']
)
@triton.jit
def _super_gemm(
    a_ptr, b_ptr, c_ptr,
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
    NUM_SMS: tl.constexpr,
    USE_TMA: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    
    if USE_TMA:
        # ==========================================
        # PATH 1 & 2: Hopper TMA with Persistent Grid
        # ==========================================
        a_desc = tl.make_tensor_descriptor(
            a_ptr, shape=[M, K], strides=[stride_am, stride_ak], block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
        )
        b_desc = tl.make_tensor_descriptor(
            b_ptr, shape=[N, K], strides=[stride_bn, stride_bk], block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
        )
        c_desc = tl.make_tensor_descriptor(
            c_ptr, shape=[M, N], strides=[stride_cm, stride_cn], block_shape=[BLOCK_M, BLOCK_N]
        )

        start_pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_tiles = num_pid_m * num_pid_n
        num_k_tiles = tl.cdiv(K, BLOCK_K)

        for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False, warp_specialize=WARP_SPECIALIZE):
            # L2 Cache Swizzling math (matches classic standard grouping but applied iteratively)
            num_pid_in_group = GROUP_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            GROUP_M_ACTUAL = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
            
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % GROUP_M_ACTUAL)
            pid_n = (tile_id % num_pid_in_group) // GROUP_M_ACTUAL

            offset_m = pid_m * BLOCK_M
            offset_n = pid_n * BLOCK_N
            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

            for k_tile in range(num_k_tiles):
                offset_k = k_tile * BLOCK_K
                a = a_desc.load([offset_m, offset_k])
                b = b_desc.load([offset_n, offset_k])
                # b is technically loaded [BLOCK_N, BLOCK_K]. b.T perfectly aligns with fast path Hopper physical strides
                acc = tl.dot(a, b.T, acc)

            c_desc.store([offset_m, offset_n], acc.to(dtype))

    else:
        # ==========================================
        # PATH 3: Standard Pointer arithmetic
        # ==========================================
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        
        # Standard L2 cache grouped swizzle logic
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        GROUP_M_ACTUAL = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        
        pid_m = first_pid_m + ((pid % num_pid_in_group) % GROUP_M_ACTUAL)
        pid_n = (pid % num_pid_in_group) // GROUP_M_ACTUAL

        offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            # B dims are constant multiples of optimal blocks. Masking is entirely eliminated for inner `b` loop
            a = tl.load(a_ptrs, mask=offs_am[:, None] < M, other=0.0)
            b = tl.load(b_ptrs) 
            
            # WGMMA lowering will hit peak tensor core usage through standard pipeline
            acc = tl.dot(a, b.T, acc)
            
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
            
        acc = acc.to(dtype)
        
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + (stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :])
        
        tl.store(c_ptrs, acc, mask=offs_cm[:, None] < M)

def run(A, B, C):
    """
    Computes general matrix multiplication C = A @ B.T.
    Inputs:
      A: [M, K]
      B: [N, K]
    Outputs:
      C: [M, N] (Preallocated)
    """
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = 7168
    K = 5120
    
    # Retrieve SM count to scale persistent grids perfectly dynamically across varying Hopper SKUs (H100/H800/etc.)
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    def grid(META):
        num_pid_m = triton.cdiv(M, META['BLOCK_M'])
        num_pid_n = triton.cdiv(N, META['BLOCK_N'])
        num_tiles = num_pid_m * num_pid_n
        if META.get('USE_TMA', False):
            # Limit the persistent SM launch to exactly the physical cluster/SMs count
            return (min(num_sms, num_tiles),)
        else:
            # Let hardware scheduler natively dispatch all tiles round robin for standard path
            return (num_tiles,)
            
    _super_gemm[grid](
        A, B, C,
        M,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        N=N, K=K,
        NUM_SMS=num_sms,
    )