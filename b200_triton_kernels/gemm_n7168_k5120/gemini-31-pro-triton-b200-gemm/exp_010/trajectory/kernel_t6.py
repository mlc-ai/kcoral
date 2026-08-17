import torch
import triton
import triton.language as tl

# Configure Triton descriptor allocator for device-side descriptor storage
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    configs = []
    
    # 1. TMA (Tensor Memory Accelerator) paths targeting maximum Tensor Core throughput
    for ws in [True, False]:
        for block_m, block_n, block_k, num_warps, num_stages in [
            (128, 256, 128, 4, 2),
            (128, 256, 128, 8, 2),
            (256, 128, 128, 4, 2),
            (256, 128, 128, 8, 2),
            (128, 128, 128, 4, 3),
            (128, 128, 128, 8, 3),
            (128, 256, 64, 4, 3),
            (256, 128, 64, 4, 3),
            (128, 64, 128, 4, 3),
            (64, 128, 128, 4, 3),
        ]:
            if ws and num_warps < 4: continue
            if ws and num_stages < 2: continue
            
            for epilogue in [1, 2]:
                # Don't subtile small blocks (wasteful)
                if block_n <= 128 and epilogue > 1: continue
                configs.append(triton.Config(
                    {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k, "GROUP_M": 8, "EPILOGUE_SUBTILE": epilogue, "USE_TMA": True, "WARP_SPECIALIZE": ws},
                    num_warps=num_warps, num_stages=num_stages
                ))
                
    # 2. Pure Pointer-based fallback paths (sometimes optimally compiled for low setup overhead)
    for ws in [False]:
        for block_m, block_n, block_k, num_warps, num_stages in [
            (128, 256, 128, 8, 2),
            (256, 128, 128, 8, 2),
            (128, 128, 128, 4, 3),
            (128, 128, 64, 8, 4),
        ]:
            configs.append(triton.Config(
                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k, "GROUP_M": 8, "EPILOGUE_SUBTILE": 1, "USE_TMA": False, "WARP_SPECIALIZE": ws},
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
def _gemm_kernel(
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
    USE_TMA: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    tile_id = tl.program_id(0)
    
    # L2-optimal Grouped M ordering
    num_pid_in_group = GROUP_M * grid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(grid_m - first_pid_m, GROUP_M)
    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    offset_n = (pid_n * BLOCK_N).to(tl.int32)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    if USE_TMA:
        # Create TMA descriptors once per CTA
        a_desc = tl.make_tensor_descriptor(
            a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
            block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
        )
        b_desc = tl.make_tensor_descriptor(
            b_ptr, shape=[N, K], strides=[stride_bn, stride_bk],
            block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
        )
        
        # Warp specialization applied accurately to the inner K loop
        for k_tile in tl.range(0, tl.cdiv(K, BLOCK_K), warp_specialize=WARP_SPECIALIZE):
            offset_k = (k_tile * BLOCK_K).to(tl.int32)
            a_tile = a_desc.load([offset_m, offset_k])
            b_tile = b_desc.load([offset_n, offset_k])
            acc = tl.dot(a_tile, b_tile.T, acc)
    else:
        offs_m = offset_m + tl.arange(0, BLOCK_M)
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
        
        m_mask = offs_m < M
        
        # Warp specialization for standard pipelined pointer loop
        for k_tile in tl.range(0, tl.cdiv(K, BLOCK_K), warp_specialize=WARP_SPECIALIZE):
            a_tile = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
            b_tile = tl.load(b_ptrs)
            acc = tl.dot(a_tile, b_tile.T, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    subtiles = _subtile_accumulator(
        acc, BLOCK_M, BLOCK_N, EPILOGUE_SUBTILE
    )

    if USE_TMA:
        c_desc = tl.make_tensor_descriptor(
            c_ptr, shape=[M, N], strides=[stride_cm, stride_cn],
            block_shape=[BLOCK_M, BLOCK_N // EPILOGUE_SUBTILE]
        )
        for i in tl.static_range(EPILOGUE_SUBTILE):
            offset_n_i = offset_n + i * (BLOCK_N // EPILOGUE_SUBTILE)
            c_desc.store([offset_m, offset_n_i], subtiles[i].to(tl.bfloat16))
    else:
        offs_m_store = offset_m + tl.arange(0, BLOCK_M)
        m_mask = offs_m_store < M
        for i in tl.static_range(EPILOGUE_SUBTILE):
            sub_n = BLOCK_N // EPILOGUE_SUBTILE
            offs_n_store = offset_n + i * sub_n + tl.arange(0, sub_n)
            c_ptrs = c_ptr + (offs_m_store[:, None] * stride_cm + offs_n_store[None, :] * stride_cn)
            tl.store(c_ptrs, subtiles[i].to(tl.bfloat16), mask=m_mask[:, None])


def run(A, B, C):
    """
    Computes general matrix multiplication C = A @ B.T.
    Follows an aggressive auto-tuning schedule mapping perfectly to the SM100 architecture.
    Utilizes hardware scheduled CTA dispatching avoiding integer math overhead inside kernels.
    """
    if A.shape[0] == 0:
        return
    torch.cuda.set_device(A.device)
    
    M, K_val = A.shape
    N_val, _ = B.shape
    
    def grid_fn(META):
        num_tiles = triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N_val, META['BLOCK_N'])
        return (num_tiles, )

    _gemm_kernel[grid_fn](
        A, B, C,
        M, N_val, K_val,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )