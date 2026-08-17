import torch
import triton
import triton.language as tl

# Configure Triton allocator for device-side tensor descriptors (Hopper TMA requirement)
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_tma_configs():
    configs = []
    # Exhaustively test standard proven WGMMA configs without blowing up the register limits.
    # We restrict BLOCK dimensions to 128/256 which guarantees compliance with Hopper TMA instructions
    # without running into the PTX assembler's "Insufficient registers" limit.
    for ws in [True, False]:
        for block_m, block_n, block_k, num_warps, num_stages in [
            (128, 128, 128, 8, 3),
            (128, 128, 128, 8, 4),
            (128, 256, 64, 8, 3),
            (256, 128, 64, 8, 3),
            (128, 128, 64, 4, 3),
            (128, 128, 64, 8, 3),
        ]:
            configs.append(triton.Config(
                {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'WARP_SPECIALIZE': ws, 'GROUP_M': 8},
                num_warps=num_warps, num_stages=num_stages
            ))
    return configs

@triton.autotune(
    configs=get_tma_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _tma_persistent_gemm(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
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
    
    # B is loaded logically as [N, K], perfectly capturing its row-major storage in memory.
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

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_k_tiles = tl.cdiv(K, BLOCK_K)

    num_pid_in_group = GROUP_M * num_pid_n

    # Specialized persistent grid tightly looping over all SMs to maximize hardware occupancy.
    # Leveraging Hopper warp specialization drastically reduces register pressure and hides fetch latency.
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE,
    ):
        # L2 Cache Swizzle mapping assigns logically neighboring tiles to execute simultaneously on the cluster
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = num_pid_m - first_pid_m
        if group_size_m > GROUP_M:
            group_size_m = GROUP_M
        
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            # Since B natively loads as [BLOCK_N, BLOCK_K], b.T evaluates mathematically 
            # to [BLOCK_K, BLOCK_N], seamlessly matching WGMMA's optimal col-major requirement.
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
    # Minimal fallback pointer kernel mapped safely when Hopper descriptors strictly reject unaligned memory.
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
    N, _ = B.shape
    
    # Check boundaries against absolute 16-byte Hopper TMA descriptor restrictions
    tma_supported = True
    if A.stride(1) != 1 or B.stride(1) != 1 or C.stride(1) != 1:
        tma_supported = False
    if (A.stride(0) * A.element_size()) % 16 != 0:
        tma_supported = False
    if (B.stride(0) * B.element_size()) % 16 != 0:
        tma_supported = False
    if (C.stride(0) * C.element_size()) % 16 != 0:
        tma_supported = False

    if tma_supported:
        num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
        grid = lambda META: (
            min(num_sms, triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N'])),
        )
        _tma_persistent_gemm[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            NUM_SMS=num_sms
        )
    else:
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        _gemm_pointer_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
        )