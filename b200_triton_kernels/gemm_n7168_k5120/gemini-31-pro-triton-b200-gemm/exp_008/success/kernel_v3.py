import torch
import triton
import triton.language as tl

# Standard Triton device-created descriptors require setting an infrastructure allocator
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.jit
def _subtile_accumulator(
    acc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SUBTILE_FACTOR: tl.constexpr,
):
    tl.static_assert(SUBTILE_FACTOR > 0)
    tl.static_assert((SUBTILE_FACTOR & (SUBTILE_FACTOR - 1)) == 0)
    if SUBTILE_FACTOR == 1:
        return (acc,)
    else:
        tl.static_assert(BLOCK_N % 2 == 0)
        acc = tl.reshape(acc, (BLOCK_M, 2, BLOCK_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        left, right = tl.split(acc)
        return _subtile_accumulator(
            left, BLOCK_M, BLOCK_N // 2, SUBTILE_FACTOR // 2
        ) + _subtile_accumulator(
            right, BLOCK_M, BLOCK_N // 2, SUBTILE_FACTOR // 2
        )

def get_configs():
    configs = []
    # Exhaustively search optimal bandwidth paths across TMA & Pointer configurations
    for use_tma in [True, False]:
        for warp_spec in [True, False]:
            # Standard Triton dictates automatic warp_specialize performs best on simple descriptor loops
            if warp_spec and not use_tma:
                continue
                
            for block_m, block_n, block_k, num_warps, num_stages, epilogue in [
                # Largest shapes
                (256, 256, 64, 8, 3, 4),
                (128, 256, 128, 8, 3, 2),
                (256, 128, 128, 8, 3, 2),
                
                # Large standard shapes
                (128, 256, 64, 8, 4, 2),
                (256, 128, 64, 8, 4, 2),
                (128, 256, 64, 8, 3, 2),
                (256, 128, 64, 8, 3, 2),
                
                # Medium shapes avoiding subtile overhead
                (128, 128, 128, 8, 4, 1),
                (128, 128, 128, 8, 3, 1),
                (128, 128, 64, 4, 4, 1),
                (128, 128, 64, 4, 3, 1),
                (128, 128, 64, 8, 4, 1),
                (128, 128, 64, 8, 3, 1),
                
                # Aspect ratio edge shapes
                (64, 256, 64, 4, 3, 2),
                (256, 64, 64, 4, 3, 1),
            ]:
                if warp_spec and (num_warps < 4 or num_stages < 2):
                    continue
                    
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_M": block_m,
                            "BLOCK_N": block_n,
                            "BLOCK_K": block_k,
                            "GROUP_M": 8,
                            "EPILOGUE_SUBTILE": epilogue,
                            "WARP_SPECIALIZE": warp_spec,
                            "NUM_STAGES": num_stages,
                            "USE_TMA": use_tma,
                        },
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs

@triton.autotune(
    configs=get_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    # L2 Cache blocking optimization: grouped sequence traversal
    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    offset_n = (pid_n * BLOCK_N).to(tl.int32)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    # Competitively benchmark Standard Fetching vs Device Descriptors mapped TMA loads 
    if USE_TMA:
        a_desc = tl.make_tensor_descriptor(
            a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
            block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
        )
        b_desc = tl.make_tensor_descriptor(
            b_ptr, shape=[N, K], strides=[stride_bn, stride_bk],
            block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
        )
        # Lowers optimally to native tcgen05 instructions combining pipelined fetch requests
        for k0 in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
            offset_k = (k0 * BLOCK_K).to(tl.int32)
            a_tile = a_desc.load([offset_m, offset_k])
            b_tile = b_desc.load([offset_n, offset_k])
            acc = tl.dot(a_tile, b_tile.T, acc)
    else:
        offs_am = offset_m + tl.arange(0, BLOCK_M)
        offs_bn = offset_n + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk
        for k0 in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
            a_mask = (offs_am[:, None] < M)
            b_mask = (offs_bn[:, None] < N)
            a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc = tl.dot(a_tile, b_tile.T, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    subtiles = _subtile_accumulator(acc, BLOCK_M, BLOCK_N, EPILOGUE_SUBTILE)
    
    # Store directly via Global Mem element pointers bypassing Descriptor Store overhead limitations
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    
    for i in tl.static_range(EPILOGUE_SUBTILE):
        subtile_offset_n = offset_n + i * (BLOCK_N // EPILOGUE_SUBTILE)
        offs_n = subtile_offset_n + tl.arange(0, BLOCK_N // EPILOGUE_SUBTILE)
        mask_n = offs_n < N
        
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = mask_m[:, None] & mask_n[None, :]
        tl.store(c_ptrs, subtiles[i].to(tl.bfloat16), mask=mask)

def run(A, B, C):
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N = B.shape[0]

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )