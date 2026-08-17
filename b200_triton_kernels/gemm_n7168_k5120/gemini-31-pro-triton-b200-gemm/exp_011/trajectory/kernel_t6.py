import torch
import triton
import triton.language as tl

# Install Triton's descriptor allocator to provide descriptor infrastructure storage for TMA ops
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    configs = []
    
    # Dense configuration space specifically targeting heavy Blackwell tensor throughput.
    for bm, bn, bk in [
        (128, 256, 128),
        (256, 128, 128),
        (256, 256, 64),
        (128, 128, 128),
        (128, 256, 64),
        (256, 128, 64),
        (128, 128, 64),
    ]:
        for ns in [2, 3, 4, 5]:
            for num_warps in [8, 12]:
                for group_m in [8, 16]:
                    for ws, flat in [(True, True), (False, False)]:
                        # Strict SMEM allocation bounds for SM100 limits (227 KiB max usable shared per SM)
                        # We must buffer 'ns' stages of both physical A and physical B inputs
                        shm_req = (bm * bk + bn * bk) * 2 * ns
                        if shm_req > 220 * 1024:
                            continue
                        
                        # Only apply Epilogue Subtiling if accumulator registers threaten to exceed limits
                        # e.g., 256x256 generates 65,536 elements -> necessitates 2 subtiles to clear limits safely.
                        subtiles = 2 if (bm == 256 and bn == 256) else 1
                        
                        configs.append(triton.Config(
                            {
                                'BLOCK_M': bm, 
                                'BLOCK_N': bn, 
                                'BLOCK_K': bk, 
                                'GROUP_M': group_m, 
                                'WARP_SPECIALIZE': ws, 
                                'FLATTEN': flat, 
                                'EPILOGUE_SUBTILE': subtiles,
                            },
                            num_warps=num_warps, 
                            num_stages=ns
                        ))
    return configs


@triton.jit
def _subtile_accumulator(
    acc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SUBTILE_FACTOR: tl.constexpr,
):
    """
    Recursively split large N accumulator tiles into smaller storage chunks to strictly enforce
    register bounds during conversions prior to TMEM flush. Only utilized when required.
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
    configs=get_configs(),
    key=["M", "N", "K"],
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

    # Device descriptors eliminate mask overheads through native bound wrapping in the TMA pipeline
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, 1],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero"
    )
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

    # 1D Persistent Loop matching hardware SM count prevents deep scheduler exhaustion.  
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_PROGRAMS,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE,
    ):
        # L2-Optimized Grouped Tile Swizzle Mapping
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(grid_m - first_pid_m, GROUP_M)
        pid_in_group = tile_id % num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m

        offset_m = (pid_m * BLOCK_M).to(tl.int32)
        offset_n = (pid_n * BLOCK_N).to(tl.int32)
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        K_TILES = K // BLOCK_K
        for k_tile in range(0, K_TILES):
            offset_k = (k_tile * BLOCK_K).to(tl.int32)
            
            a_tile = a_desc.load([offset_m, offset_k])
            b_tile = b_desc.load([offset_n, offset_k])
            
            # Transposed matrix b applied implicitly by B.T to force Col-Major optimization inside tcgen05
            acc = tl.dot(a_tile, b_tile.T, acc, out_dtype=tl.float32)

        if EPILOGUE_SUBTILE == 1:
            c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))
        else:
            subtiles = _subtile_accumulator(
                acc, BLOCK_M, BLOCK_N, EPILOGUE_SUBTILE
            )
            for i in tl.static_range(EPILOGUE_SUBTILE):
                offset_n_i = (offset_n + i * (BLOCK_N // EPILOGUE_SUBTILE)).to(tl.int32)
                c_desc.store([offset_m, offset_n_i], subtiles[i].to(tl.bfloat16))


def run(A, B, C):
    """
    Computes a general matrix multiply C = A @ B.T.
    Arguments must be preallocated CUDA tensors. C will not be re-allocated.
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N = C.shape[1]
        
    device_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    # Restricting launched blocks exactly to the count of useful Multi-Processors
    num_tiles = triton.cdiv(M, 128) * triton.cdiv(N, 128)
    num_programs = min(device_sms, num_tiles)
    if num_programs == 0:
        num_programs = 1

    # Using 1D capped execution grid allows optimal SM saturation mapped cleanly to loop bounds 
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
    )