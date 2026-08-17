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
    """
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
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "EPILOGUE_SUBTILE": 4, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "EPILOGUE_SUBTILE": 2, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8, "EPILOGUE_SUBTILE": 4, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "EPILOGUE_SUBTILE": 2, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "EPILOGUE_SUBTILE": 2, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=4),
        # Fallbacks without warp specialization
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "EPILOGUE_SUBTILE": 2, "FLATTEN": False, "WARP_SPECIALIZE": False}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "EPILOGUE_SUBTILE": 2, "FLATTEN": False, "WARP_SPECIALIZE": False}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8, "EPILOGUE_SUBTILE": 1, "FLATTEN": False, "WARP_SPECIALIZE": False}, num_warps=4, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _persistent_tma_gemm(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    NUM_PROGRAMS,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    FLATTEN: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    start_pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = grid_m * grid_n
    num_pid_in_group = GROUP_M * grid_n

    # Descriptors created once per program
    a_desc = tl.make_tensor_descriptor(
        A, shape=[M, K], strides=[stride_am, stride_ak], block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        B, shape=[N, K], strides=[stride_bn, stride_bk], block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        C, shape=[M, N], strides=[stride_cm, stride_cn], block_shape=[BLOCK_M, BLOCK_N // EPILOGUE_SUBTILE]
    )

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_PROGRAMS,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE,
    ):
        # Determine L2-grouped tile indices
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
            
            # PMA lowering via device descriptors
            a_tile = a_desc.load([offset_m, offset_k])
            b_tile = b_desc.load([offset_n, offset_k])
            
            acc = tl.dot(a_tile, b_tile.T, acc)
        
        # Epilogue subtiling to alleviate registers footprint
        subtiles = _subtile_accumulator(acc, BLOCK_M, BLOCK_N, EPILOGUE_SUBTILE)
        
        for i in tl.static_range(EPILOGUE_SUBTILE):
            offset_n_i = offset_n + i * (BLOCK_N // EPILOGUE_SUBTILE)
            # Write results matching requested public output dtype (bfloat16)
            c_desc.store([offset_m, offset_n_i], subtiles[i].to(tl.bfloat16))


def run(A, B, C):
    """
    Computes C = A @ B.T in bfloat16 precision using Blackwell native 
    tensor-descriptor architecture via persistent TMA GEMM.
    """
    if A.numel() == 0 or B.numel() == 0 or C.numel() == 0:
        return
        
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N = C.shape[1]
    
    device_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    # Estimate the maximum number of useful parallel tiles
    max_num_tiles = triton.cdiv(M, 64) * triton.cdiv(N, 64)
    num_programs = min(device_sms, max_num_tiles)
    
    _persistent_tma_gemm[(num_programs,)](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        NUM_PROGRAMS=num_programs,
    )