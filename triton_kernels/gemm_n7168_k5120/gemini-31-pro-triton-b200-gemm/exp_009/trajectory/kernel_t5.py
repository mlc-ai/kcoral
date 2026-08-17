import torch
import triton
import triton.language as tl

# Configure Triton descriptor allocator for Blackwell TMA descriptors
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
    """
    Splits the fp32 accumulator iteratively to reduce register pressure 
    during type conversion and descriptor store in the epilogue.
    Returns a tuple of length SUBTILE_FACTOR containing the subtiles.
    """
    tl.static_assert(SUBTILE_FACTOR > 0)
    tl.static_assert((SUBTILE_FACTOR & (SUBTILE_FACTOR - 1)) == 0)
    if SUBTILE_FACTOR == 1:
        return (acc,)
    else:
        tl.static_assert(BLOCK_N % 2 == 0)
        # Group columns into two halves
        acc = tl.reshape(acc, (BLOCK_M, 2, BLOCK_N // 2))
        # Bring the 'halves' dimension to the end for standard split
        acc = tl.permute(acc, (0, 2, 1))
        left, right = tl.split(acc)
        # Recurse and combine tuples
        return _subtile_accumulator(
            left, BLOCK_M, BLOCK_N // 2, SUBTILE_FACTOR // 2
        ) + _subtile_accumulator(
            right, BLOCK_M, BLOCK_N // 2, SUBTILE_FACTOR // 2
        )


@triton.autotune(
    configs=[
        # --- High-Warp configs for large tiles to hide latency & distribute registers (B200 has 1024 threads/CTA) ---
        
        # 128x256x128
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8, "EPILOGUE_SUBTILE": 1, "PROGS_PER_SM": 1, "USE_PERSISTENT": True, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8, "EPILOGUE_SUBTILE": 1, "PROGS_PER_SM": 1, "USE_PERSISTENT": True, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=12, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8, "EPILOGUE_SUBTILE": 1, "PROGS_PER_SM": 1, "USE_PERSISTENT": True, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=16, num_stages=2),
        
        # 256x128x128
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "EPILOGUE_SUBTILE": 1, "PROGS_PER_SM": 1, "USE_PERSISTENT": True, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "EPILOGUE_SUBTILE": 1, "PROGS_PER_SM": 1, "USE_PERSISTENT": True, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=12, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "EPILOGUE_SUBTILE": 1, "PROGS_PER_SM": 1, "USE_PERSISTENT": True, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=16, num_stages=2),
        
        # 256x256x64 (Largest Tile, uses subtile to prevent spill during FP32->BF16 conversion)
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "EPILOGUE_SUBTILE": 2, "PROGS_PER_SM": 1, "USE_PERSISTENT": True, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=12, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "EPILOGUE_SUBTILE": 2, "PROGS_PER_SM": 1, "USE_PERSISTENT": True, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=16, num_stages=3),
        
        # 128x256x64
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "EPILOGUE_SUBTILE": 1, "PROGS_PER_SM": 1, "USE_PERSISTENT": True, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "EPILOGUE_SUBTILE": 1, "PROGS_PER_SM": 1, "USE_PERSISTENT": True, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=12, num_stages=4),
        
        # 256x128x64
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "EPILOGUE_SUBTILE": 1, "PROGS_PER_SM": 1, "USE_PERSISTENT": True, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "EPILOGUE_SUBTILE": 1, "PROGS_PER_SM": 1, "USE_PERSISTENT": True, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=12, num_stages=4),
        
        # 128x128x128 (Smaller tiles allow more stages for latency hiding, Progs per SM = 2)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "EPILOGUE_SUBTILE": 1, "PROGS_PER_SM": 2, "USE_PERSISTENT": True, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "EPILOGUE_SUBTILE": 1, "PROGS_PER_SM": 2, "USE_PERSISTENT": True, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=8, num_stages=4),
        
        # Non-persistent fallbacks (standard SM CTA scheduling)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8, "EPILOGUE_SUBTILE": 1, "PROGS_PER_SM": 1, "USE_PERSISTENT": False, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=12, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "EPILOGUE_SUBTILE": 1, "PROGS_PER_SM": 1, "USE_PERSISTENT": False, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=12, num_stages=2),
    ],
    key=["M"], # N and K are passed as tl.constexpr, so tuning uniquely varies by M
)
@triton.jit
def _tma_gemm_kernel(
    A, B, C,
    M,
    stride_am, stride_bn, stride_cm,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    FLATTEN: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    USE_PERSISTENT: tl.constexpr,
    PROGS_PER_SM: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = N // BLOCK_N
    num_tiles = grid_m * grid_n
    num_pid_in_group = GROUP_M * grid_n

    # Device descriptors for TMA backed loads/stores
    # Note: inner stride is hardcoded to 1 to satisfy TMA alignment requirements
    a_desc = tl.make_tensor_descriptor(
        A, shape=[M, K], strides=[stride_am, 1], block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        B, shape=[N, K], strides=[stride_bn, 1], block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        C, shape=[M, N], strides=[stride_cm, 1], block_shape=[BLOCK_M, BLOCK_N // EPILOGUE_SUBTILE]
    )

    # When USE_PERSISTENT is False, num_programs == num_tiles, 
    # so the loop executes exactly once acting as a standard non-persistent kernel.
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        num_programs,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE,
    ):
        # L2-Grouped tile ordering 
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(grid_m - first_pid_m, GROUP_M)
        pid_in_group = tile_id % num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m

        offset_m = (pid_m * BLOCK_M).to(tl.int32)
        offset_n = (pid_n * BLOCK_N).to(tl.int32)
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        # Inner loop dynamically pipelined seamlessly by Triton compiler
        for k_tile in range(0, K // BLOCK_K):
            offset_k = (k_tile * BLOCK_K).to(tl.int32)
            
            a_tile = a_desc.load([offset_m, offset_k])
            b_tile = b_desc.load([offset_n, offset_k])
            
            # b_tile is loaded as [BLOCK_N, BLOCK_K], logical dot requires .T
            acc = tl.dot(a_tile, b_tile.T, acc)
        
        # Fast path for EPILOGUE_SUBTILE == 1 completely avoids overhead 
        if EPILOGUE_SUBTILE == 1:
            c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))
        else:
            # Epilogue subtiling alleviates registers footprint limit on huge tiles
            subtiles = _subtile_accumulator(acc, BLOCK_M, BLOCK_N, EPILOGUE_SUBTILE)
            for i in tl.static_range(EPILOGUE_SUBTILE):
                offset_n_i = (offset_n + i * (BLOCK_N // EPILOGUE_SUBTILE)).to(tl.int32)
                # Write results matching requested public output dtype (bfloat16)
                c_desc.store([offset_m, offset_n_i], subtiles[i].to(tl.bfloat16))


def run(A, B, C):
    """
    Computes C = A @ B.T in bfloat16 precision using Blackwell native 
    tensor-descriptor architecture via an optimized tuning space.
    """
    if A.numel() == 0 or B.numel() == 0 or C.numel() == 0:
        return
        
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N = C.shape[1]
    
    device_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    # Launch grid resolves to device_sms if persistent, otherwise full grid mapping
    def grid_fn(META):
        num_tiles = triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"])
        if META.get("USE_PERSISTENT", True):
            progs_per_sm = META.get("PROGS_PER_SM", 1)
            return (min(device_sms * progs_per_sm, num_tiles),)
        else:
            return (num_tiles,)
    
    # Note: K and N are strictly consts in the problem statement 
    # Passing them as Python keyword kwargs resolves them as `tl.constexpr`s
    _tma_gemm_kernel[grid_fn](
        A, B, C,
        M,
        A.stride(0), B.stride(0), C.stride(0),
        N=N, K=K
    )