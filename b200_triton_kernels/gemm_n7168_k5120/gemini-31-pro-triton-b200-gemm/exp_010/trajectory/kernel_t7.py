import torch
import triton
import triton.language as tl

# Configure Triton descriptor allocator for device-side descriptor storage
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    configs = []
    # Exhaustive tuning space balancing register footprint, shared memory, and warp specialization.
    for ws in [True, False]:
        for block_m, block_n, block_k, num_warps, num_stages in [
            (128, 256, 64, 4, 4),
            (128, 256, 64, 8, 4),
            (256, 128, 64, 4, 4),
            (256, 128, 64, 8, 4),
            
            (128, 256, 128, 4, 2),
            (128, 256, 128, 8, 2),
            (256, 128, 128, 4, 2),
            (256, 128, 128, 8, 2),
            
            (128, 128, 128, 4, 3),
            (128, 128, 128, 8, 3),
            (128, 128, 128, 4, 4),
            (128, 128, 128, 8, 4),
            
            (128, 128, 64, 4, 5),
            (128, 128, 64, 8, 5),

            (256, 256, 64, 8, 2),
            (256, 256, 64, 8, 3),
        ]:
            # Hardware requirements for warp specialization
            if ws and num_warps < 4: continue
            if ws and num_stages < 2: continue
            
            # Enforce 228 KiB limit for SM100
            bytes_per_stage = (block_m * block_k + block_n * block_k) * 2
            if bytes_per_stage * num_stages > 225 * 1024:
                continue
                
            for epilogue in [1, 2, 4]:
                if epilogue > 1 and block_n // epilogue < 64: continue
                if epilogue == 4 and block_n < 256: continue
                
                # Explore swizzling intensity to maximize L2 hit rate for matrix B
                for group_m in [4, 8]:
                    configs.append(triton.Config(
                        {
                            "BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k, 
                            "GROUP_M": group_m, "EPILOGUE_SUBTILE": epilogue, 
                            "WARP_SPECIALIZE": ws
                        },
                        num_warps=num_warps, num_stages=num_stages
                    ))
    return configs

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
    configs=get_configs(),
    key=["M"],
)
@triton.jit
def _persistent_tma_gemm(
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
    WARP_SPECIALIZE: tl.constexpr,
):
    # Blackwell TMA Descriptors declared outside the persistent loop
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero"
    )
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

    # Native persistent grid loop leveraging automatic warp specialization & flattening
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        num_programs,
        flatten=True,
        warp_specialize=WARP_SPECIALIZE,
    ):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(grid_m - first_pid_m, GROUP_M)
        pid_in_group = tile_id % num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m

        offset_m = (pid_m * BLOCK_M).to(tl.int32)
        offset_n = (pid_n * BLOCK_N).to(tl.int32)
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
            offset_k = (k_tile * BLOCK_K).to(tl.int32)
            a_tile = a_desc.load([offset_m, offset_k])
            b_tile = b_desc.load([offset_n, offset_k])
            acc = tl.dot(a_tile, b_tile.T, acc)

        # Apply epilogue logic resolving register allocations gracefully
        subtiles = _subtile_accumulator(
            acc, BLOCK_M, BLOCK_N, EPILOGUE_SUBTILE
        )

        for i in tl.static_range(EPILOGUE_SUBTILE):
            offset_n_i = offset_n + i * (BLOCK_N // EPILOGUE_SUBTILE)
            c_desc.store([offset_m, offset_n_i], subtiles[i].to(tl.bfloat16))


def run(A, B, C):
    """
    Computes general matrix multiplication C = A @ B.T.
    Follows a highly optimized persistent device-created descriptor TMA kernel pattern
    targeting Blackwell SM100 architecture with specialized TMEM utilization mapping.
    """
    if A.shape[0] == 0:
        return
    torch.cuda.set_device(A.device)
    
    M, K_val = A.shape
    N_val, _ = B.shape
    
    device_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    def grid_fn(META):
        num_tiles = triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N_val, META['BLOCK_N'])
        # Launch up to 2x SM limit to ensure hardware scheduler hides thread dispatch latencies completely
        return (min(device_sms * 2, num_tiles), )

    _persistent_tma_gemm[grid_fn](
        A, B, C,
        M, N_val, K_val,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )