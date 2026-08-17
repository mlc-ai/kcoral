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
    # Exhaustive sweep optimally avoiding Register Ceilings & SMEM spilling on Blackwell target.
    # We strictly enforce `num_warps=4` on `warp_specialize=True` setups because Standard 
    # Triton will silently duplicate logical warps up to 8 internally for concurrent MMA/Memory roles!
    for block_m, block_n, block_k, num_warps, num_stages, epilogue, warp_spec, prog_mult in [
        # Highly specialized configs with safe SMEM footprints (~192KB/144KB limits)
        (128, 256, 64, 4, 4, 2, True, 1),
        (256, 128, 64, 4, 4, 2, True, 1),
        (128, 256, 64, 4, 4, 2, True, 2),
        (256, 128, 64, 4, 4, 2, True, 2),
        (128, 256, 64, 4, 3, 2, True, 1),
        (256, 128, 64, 4, 3, 2, True, 1),
        (128, 256, 64, 4, 3, 2, True, 2),
        
        # Square sizes matching cleanly against Tensor Core alignment boundaries
        (128, 128, 64, 4, 4, 1, True, 1),
        (128, 128, 64, 4, 4, 1, True, 2),
        (128, 128, 64, 4, 5, 1, True, 1),
        
        # Unspecialized baseline configurations safely scaling `num_warps=8` statically
        (128, 256, 64, 8, 4, 2, False, 1),
        (256, 128, 64, 8, 4, 2, False, 1),
        (128, 128, 64, 8, 4, 1, False, 1),
        (128, 256, 64, 8, 3, 2, False, 1),
        (256, 128, 64, 8, 3, 2, False, 1),
    ]:
        configs.append(
            triton.Config(
                {
                    "BLOCK_M": block_m,
                    "BLOCK_N": block_n,
                    "BLOCK_K": block_k,
                    "GROUP_M": 8,
                    "EPILOGUE_SUBTILE": epilogue,
                    "FLATTEN": True,
                    "WARP_SPECIALIZE": warp_spec,
                    "PROG_MULT": prog_mult,
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
def _persistent_tma_gemm(
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
    FLATTEN: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    PROG_MULT: tl.constexpr,
):
    NUM_PROGRAMS = tl.num_programs(0)
    
    # Device descriptors for Async TMA mapped loads / stores
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
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N // EPILOGUE_SUBTILE],
    )

    start_pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = grid_m * grid_n
    
    # Pre-calculated swizzle limits mapped per logical group size bounding localized SM caching.
    num_pid_in_group = GROUP_M * grid_n

    # The outer persistent loop natively implements flattened boundaries spanning across
    # sequentially grouped tiles, perfectly overlapping epilogues mapped onto WGMMA cycles.
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_PROGRAMS,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE,
    ):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)
        pid_in_group = tile_id % num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m

        offset_m = (pid_m * BLOCK_M).to(tl.int32)
        offset_n = (pid_n * BLOCK_N).to(tl.int32)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        # Reconstruct logical dot boundaries natively within pipelined loops lowered efficiently
        # leveraging backend `tcgen05` scheduling semantics spanning memory stages mapped on TMA paths. 
        for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
            offset_k = (k_tile * BLOCK_K).to(tl.int32)
            a_tile = a_desc.load([offset_m, offset_k])
            b_tile = b_desc.load([offset_n, offset_k])
            acc = tl.dot(a_tile, b_tile.T, acc)

        subtiles = _subtile_accumulator(acc, BLOCK_M, BLOCK_N, EPILOGUE_SUBTILE)
        
        # Stream pipelined flush onto global memory bypassing restrictive Pointer bounds calculations.
        for i in tl.static_range(EPILOGUE_SUBTILE):
            offset_n_i = offset_n + i * (BLOCK_N // EPILOGUE_SUBTILE)
            c_desc.store([offset_m, offset_n_i], subtiles[i].to(tl.bfloat16))


_device_sms = None

def run(A, B, C):
    global _device_sms
    if _device_sms is None:
        _device_sms = torch.cuda.get_device_properties(A.device).multi_processor_count

    M, K = A.shape
    N = B.shape[0]

    def grid(META):
        grid_m = triton.cdiv(M, META["BLOCK_M"])
        grid_n = triton.cdiv(N, META["BLOCK_N"])
        num_tiles = grid_m * grid_n
        
        # Scaling parameter determining execution parallel scaling avoiding strict hardware cap bounds.
        mult = META.get("PROG_MULT", 1)
        num_programs = min(_device_sms * mult, num_tiles)
        return (num_programs,)

    _persistent_tma_gemm[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )