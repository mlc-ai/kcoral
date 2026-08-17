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
        # Warp Specialize = True with num_warps=4 
        # (Warp specialization increases warps implicitly. Starting from 4 helps avoid register spilling.)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8, "EPILOGUE_SUBTILE": 4, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "EPILOGUE_SUBTILE": 4, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "EPILOGUE_SUBTILE": 2, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=3),

        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64,  "GROUP_M": 8, "EPILOGUE_SUBTILE": 4, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64,  "GROUP_M": 8, "EPILOGUE_SUBTILE": 4, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64,  "GROUP_M": 8, "EPILOGUE_SUBTILE": 2, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=4),

        # Warp Specialize = True with num_warps=8
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8, "EPILOGUE_SUBTILE": 4, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "EPILOGUE_SUBTILE": 2, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64,  "GROUP_M": 8, "EPILOGUE_SUBTILE": 4, "FLATTEN": True, "WARP_SPECIALIZE": True}, num_warps=8, num_stages=4),

        # Baseline (WARP_SPECIALIZE = False)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8, "EPILOGUE_SUBTILE": 4, "FLATTEN": False, "WARP_SPECIALIZE": False}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "EPILOGUE_SUBTILE": 2, "FLATTEN": False, "WARP_SPECIALIZE": False}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64,  "GROUP_M": 8, "EPILOGUE_SUBTILE": 4, "FLATTEN": False, "WARP_SPECIALIZE": False}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64,  "GROUP_M": 8, "EPILOGUE_SUBTILE": 2, "FLATTEN": False, "WARP_SPECIALIZE": False}, num_warps=8, num_stages=4),
    ],
    key=["M"],
)
@triton.jit
def _persistent_tma_gemm(
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
    EPILOGUE_SUBTILE: tl.constexpr,
    FLATTEN: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero"
    )
    # B is physically [N, K], transposing the tile at dot product.
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
        # Grouped M ordering to maximize L2 Cache reuse of B matrix
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

        subtiles = _subtile_accumulator(
            acc, BLOCK_M, BLOCK_N, EPILOGUE_SUBTILE
        )

        for i in tl.static_range(EPILOGUE_SUBTILE):
            offset_n_i = offset_n + i * (BLOCK_N // EPILOGUE_SUBTILE)
            c_desc.store([offset_m, offset_n_i], subtiles[i].to(tl.bfloat16))

def run(A, B, C):
    """
    Computes general matrix multiplication C = A @ B.T where:
      A is [M, K]
      B is [N, K]
      C is [M, N]
    Uses a highly optimized persistent device-created descriptor TMA kernel logic 
    targeting NVIDIA Blackwell SM100 limits and behavior.
    """
    if A.shape[0] == 0:
        return
    torch.cuda.set_device(A.device)
    
    M, K_val = A.shape
    N_val, _ = B.shape
    
    device_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    def grid_fn(META):
        num_tiles = triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N_val, META['BLOCK_N'])
        # A 1D persistent grid mapped to max concurrent hardware thread blocks
        return (min(device_sms, num_tiles), )
        
    _persistent_tma_gemm[grid_fn](
        A, B, C,
        M,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        N_val, K_val
    )