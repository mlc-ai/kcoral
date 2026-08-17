import torch
import triton
import triton.language as tl

# Configure Triton descriptor allocator for device-side descriptor storage
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

@triton.autotune(
    configs=[
        # Non-persistent configs (Multiplier 0) - allows hardware to fully schedule all blocks
        # Large blocks, optimal for high throughput and reduced TMA overhead
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8,  "EPILOGUE_SUBTILE": 1, "FLATTEN": True, "WARP_SPECIALIZE": True, "PERSISTENT_MULTIPLIER": 0}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8,  "EPILOGUE_SUBTILE": 1, "FLATTEN": True, "WARP_SPECIALIZE": True, "PERSISTENT_MULTIPLIER": 0}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64,  "GROUP_M": 8,  "EPILOGUE_SUBTILE": 1, "FLATTEN": True, "WARP_SPECIALIZE": True, "PERSISTENT_MULTIPLIER": 0}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64,  "GROUP_M": 8,  "EPILOGUE_SUBTILE": 1, "FLATTEN": True, "WARP_SPECIALIZE": True, "PERSISTENT_MULTIPLIER": 0}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8,  "EPILOGUE_SUBTILE": 1, "FLATTEN": True, "WARP_SPECIALIZE": True, "PERSISTENT_MULTIPLIER": 0}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8,  "EPILOGUE_SUBTILE": 1, "FLATTEN": True, "WARP_SPECIALIZE": True, "PERSISTENT_MULTIPLIER": 0}, num_warps=4, num_stages=3),
        
        # Vary GROUP_M for L2 Cache Tuning
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 16, "EPILOGUE_SUBTILE": 1, "FLATTEN": True, "WARP_SPECIALIZE": True, "PERSISTENT_MULTIPLIER": 0}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 32, "EPILOGUE_SUBTILE": 1, "FLATTEN": True, "WARP_SPECIALIZE": True, "PERSISTENT_MULTIPLIER": 0}, num_warps=8, num_stages=2),
        
        # Enable clustered scheduling (num_ctas=2, 4) to exploit L2 hardware clusters natively
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8,  "EPILOGUE_SUBTILE": 1, "FLATTEN": True, "WARP_SPECIALIZE": True, "PERSISTENT_MULTIPLIER": 0}, num_warps=8, num_stages=2, num_ctas=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8,  "EPILOGUE_SUBTILE": 1, "FLATTEN": True, "WARP_SPECIALIZE": True, "PERSISTENT_MULTIPLIER": 0}, num_warps=8, num_stages=2, num_ctas=4),
        
        # Persistent configs (Multiplier 1, 2, 4) - limits overhead of grid scheduling by reusing programs
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8,  "EPILOGUE_SUBTILE": 1, "FLATTEN": True, "WARP_SPECIALIZE": True, "PERSISTENT_MULTIPLIER": 1}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8,  "EPILOGUE_SUBTILE": 1, "FLATTEN": True, "WARP_SPECIALIZE": True, "PERSISTENT_MULTIPLIER": 2}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8,  "EPILOGUE_SUBTILE": 1, "FLATTEN": True, "WARP_SPECIALIZE": True, "PERSISTENT_MULTIPLIER": 4}, num_warps=8, num_stages=2),

        # Configurations with Epilogue Subtiling (saves register file pressure in epilogue)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8,  "EPILOGUE_SUBTILE": 2, "FLATTEN": True, "WARP_SPECIALIZE": True, "PERSISTENT_MULTIPLIER": 0}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8,  "EPILOGUE_SUBTILE": 2, "FLATTEN": True, "WARP_SPECIALIZE": True, "PERSISTENT_MULTIPLIER": 0}, num_warps=8, num_stages=2),

        # Non-warp-specialized fallback (standard pipelined TMA without automatic async role partitioning)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8,  "EPILOGUE_SUBTILE": 1, "FLATTEN": False, "WARP_SPECIALIZE": False, "PERSISTENT_MULTIPLIER": 0}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8,  "EPILOGUE_SUBTILE": 1, "FLATTEN": False, "WARP_SPECIALIZE": False, "PERSISTENT_MULTIPLIER": 0}, num_warps=8, num_stages=3),
    ],
    key=["M"],
)
@triton.jit
def _tma_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
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
    PERSISTENT_MULTIPLIER: tl.constexpr,
):
    # Descriptors created once per grid program iteration (efficient setup)
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero"
    )
    # Physically N x K mapping. Transposed dynamically in the inner loop with b_tile.T
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N // EPILOGUE_SUBTILE]
    )

    start_pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = grid_m * grid_n
    num_pid_in_group = GROUP_M * grid_n

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        num_programs,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE,
    ):
        # L2-optimal Swizzling
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(grid_m - first_pid_m, GROUP_M)
        pid_in_group = tile_id % num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m

        offset_m = (pid_m * BLOCK_M).to(tl.int32)
        offset_n = (pid_n * BLOCK_N).to(tl.int32)
        
        # Explicit accumulator precision management
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        # Loop over K
        for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
            offset_k = (k_tile * BLOCK_K).to(tl.int32)
            a_tile = a_desc.load([offset_m, offset_k])
            b_tile = b_desc.load([offset_n, offset_k])
            
            # WGMMA integration with TMA operand delivery
            acc = tl.dot(a_tile, b_tile.T, acc)

        # Accumulator Epilogue and Storage
        subtiles = _subtile_accumulator(
            acc, BLOCK_M, BLOCK_N, EPILOGUE_SUBTILE
        )

        for i in tl.static_range(EPILOGUE_SUBTILE):
            offset_n_i = offset_n + i * (BLOCK_N // EPILOGUE_SUBTILE)
            c_desc.store([offset_m, offset_n_i], subtiles[i].to(tl.bfloat16))


def run(A, B, C):
    """
    Computes general matrix multiplication C = A @ B.T.
    Explores comprehensive non-persistent and persistent grid configurations
    tailored for optimal Blackwell SM100 hardware thread occupancy and L2 re-use.
    """
    if A.shape[0] == 0:
        return
    torch.cuda.set_device(A.device)
    
    M, K_val = A.shape
    N_val, _ = B.shape
    
    device_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    def grid_fn(META):
        num_tiles = triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N_val, META['BLOCK_N'])
        pm = META['PERSISTENT_MULTIPLIER']
        if pm == 0:
            return (num_tiles, )
        else:
            return (min(device_sms * pm, num_tiles), )

    _tma_gemm_kernel[grid_fn](
        A, B, C,
        M, N_val, K_val,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )