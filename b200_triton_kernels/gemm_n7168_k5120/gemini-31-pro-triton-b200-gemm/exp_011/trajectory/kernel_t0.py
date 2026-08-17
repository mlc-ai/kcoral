import torch
import triton
import triton.language as tl

# Install Triton's descriptor allocator to provide descriptor storage for TMA ops
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
    Recursively split the large N accumulator tile into smaller ones for conversion and storage.
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

@triton.jit
def _persistent_tma_gemm(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_bn, stride_cm,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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

    # Descriptors efficiently replace mask checking and correctly handle alignment logic in TMA.
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, 1],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero"
    )
    # B is physically layout [N, K], transposing its loaded logical [BLOCK_N, BLOCK_K] block inline with .T
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[stride_bn, 1],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[stride_cm, 1],
        block_shape=[BLOCK_M, BLOCK_N // EPILOGUE_SUBTILE]
    )

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_PROGRAMS,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE,
    ):
        # Grouped Tile Swizzle
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
            
            # [BLOCK_M, BLOCK_K] @ [BLOCK_N, BLOCK_K].T -> [BLOCK_M, BLOCK_N]
            acc = tl.dot(a_tile, b_tile.T, acc)

        subtiles = _subtile_accumulator(
            acc, BLOCK_M, BLOCK_N, EPILOGUE_SUBTILE
        )

        for i in tl.static_range(EPILOGUE_SUBTILE):
            offset_n_i = offset_n + i * (BLOCK_N // EPILOGUE_SUBTILE)
            c_desc.store([offset_m, offset_n_i], subtiles[i].to(tl.bfloat16))

def run(A, B, C):
    """
    Computes a general matrix multiply C = A @ B.T.
    Arguments must be preallocated CUDA tensors. C will not be re-allocated.
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N = C.shape[1]

    # Selected Blackwell config for heavy QKV projection workload shapes 
    block_m = 128
    block_n = 256
    block_k = 64
    
    num_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    if num_tiles == 0:
        return
        
    device_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    # Cap 1D grid at hardware-useful SM counts for persistence
    num_programs = min(device_sms, num_tiles)
    if num_programs == 0:
        num_programs = 1

    _persistent_tma_gemm[(num_programs,)](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        B.stride(0),
        C.stride(0),
        NUM_PROGRAMS=num_programs,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        EPILOGUE_SUBTILE=2,
        FLATTEN=True,
        WARP_SPECIALIZE=True,
        num_warps=8,
        num_stages=4,
    )