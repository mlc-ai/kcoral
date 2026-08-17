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
        return _subtile_accumulator(left, BLOCK_M, BLOCK_N // 2, SUBTILE_FACTOR // 2) + \
               _subtile_accumulator(right, BLOCK_M, BLOCK_N // 2, SUBTILE_FACTOR // 2)


def get_configs():
    configs = []
    # Configurations mapped specifically towards WGMMA throughput with L2 caching (GROUP_M=8)
    # The autotuner finds optimal stage/warp mappings avoiding both SMEM & Register limits
    for block_m, block_n, num_warps, num_stages, epilogue, warp_spec in [
        # Large tiles - Subtile 1 avoids permutation/split math entirely maximizing epilogue speeds
        (128, 256, 8, 4, 1, True),
        (256, 128, 8, 4, 1, True),
        (128, 256, 8, 3, 1, True),
        (256, 128, 8, 3, 1, True),
        
        # Subtile 2 fallback variants if register spilling triggers on any unexpected layouts
        (128, 256, 8, 4, 2, True),
        (256, 128, 8, 4, 2, True),
        
        # Unspecialized Large tiles baselines
        (128, 256, 8, 4, 1, False),
        (256, 128, 8, 4, 1, False),
        (128, 256, 8, 3, 1, False),
        (256, 128, 8, 3, 1, False),
        
        # Mid-size robust tiles
        (128, 128, 8, 4, 1, True),
        (128, 128, 8, 5, 1, True),
        (128, 128, 4, 4, 1, True),
        (128, 128, 4, 5, 1, True),
        (128, 128, 8, 4, 1, False),
        (128, 128, 4, 4, 1, False),
    ]:
        configs.append(
            triton.Config(
                {
                    "BLOCK_M": block_m,
                    "BLOCK_N": block_n,
                    "BLOCK_K": 64,  # Hardlocked to 64 avoiding `tcgen05.mma` invalid layout issues
                    "GROUP_M": 8,
                    "EPILOGUE_SUBTILE": epilogue,
                    "WARP_SPECIALIZE": warp_spec,
                    "NUM_STAGES": num_stages,
                },
                num_warps=num_warps,
                num_stages=num_stages,
            )
        )
    return configs


@triton.heuristics({
    "EVEN_M": lambda args: args["M"] % args["BLOCK_M"] == 0,
    "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
})
@triton.autotune(
    configs=get_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _tma_gemm(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero",
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero",
    )
    
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    # L2 Cache Optimization: grouping mapping minimizes loading iterations of `B` Matrix
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

    # Blackwell Native TMA loads piped perfectly onto 5th Gen TC units
    for k0 in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        offset_k = (k0 * BLOCK_K).to(tl.int32)
        a_tile = a_desc.load([offset_m, offset_k])
        b_tile = b_desc.load([offset_n, offset_k])
        acc = tl.dot(a_tile, b_tile.T, acc)

    subtiles = _subtile_accumulator(acc, BLOCK_M, BLOCK_N, EPILOGUE_SUBTILE)
    
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    for i in tl.static_range(EPILOGUE_SUBTILE):
        subtile_offset_n = offset_n + i * (BLOCK_N // EPILOGUE_SUBTILE)
        offs_n = subtile_offset_n + tl.arange(0, BLOCK_N // EPILOGUE_SUBTILE)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        
        # Bypass heavy boundary constraint logic if shapes perfectly align with block definitions
        if EVEN_M and EVEN_N:
            tl.store(c_ptrs, subtiles[i].to(tl.bfloat16))
        else:
            mask_m = offs_m < M
            mask_n = offs_n < N
            mask = mask_m[:, None] & mask_n[None, :]
            tl.store(c_ptrs, subtiles[i].to(tl.bfloat16), mask=mask)


def run(A, B, C):
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N = B.shape[0]

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _tma_gemm[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )