import torch
import triton
import triton.language as tl

_allocator_set = False

def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

def get_configs():
    configs = []
    # Hopper H100 SM90 usable shared memory max per CTA
    max_shared_memory = 227 * 1024
    
    # We allow the auto-tuner to empirically pick between a persistent loop dispatch (with or without
    # restricted warp specialization) and a full standard hardware-scheduled grid dispatch.
    for persistent in [True, False]:
        for warp_specialize in ([True, False] if persistent else [False]):
            for block_m, block_n, block_k, num_warps, num_stages in [
                (128, 128, 64, 4, 3),
                (128, 128, 64, 4, 4),
                (128, 128, 64, 4, 5),
                (128, 128, 64, 8, 3),
                (128, 128, 64, 8, 4),
                (128, 256, 64, 8, 3),
                (128, 256, 64, 8, 4),
                (256, 128, 64, 8, 3),
                (256, 128, 64, 8, 4),
                (128, 128, 128, 4, 3),
                (128, 128, 128, 8, 3),
                (128, 128, 128, 8, 4),
                (128, 256, 128, 8, 2),
                (256, 128, 128, 8, 2),
            ]:
                # Guard against configs that would catastrophically spill registers. 
                # (e.g. 128x256 / 4 warps -> 256 registers per thread > 255 max allowed)
                acc_size = block_m * block_n
                threads = num_warps * 32
                if acc_size / threads > 128:
                    continue
                    
                # Guard against exceeding SMEM bounds for TMA descriptors and WGMMA buffer pipeline
                a_size = block_m * block_k * 2  # 2 bytes for bfloat16
                b_size = block_n * block_k * 2
                total_smem = (a_size + b_size) * num_stages
                
                if total_smem <= max_shared_memory:
                    configs.append(triton.Config(
                        {
                            'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 
                            'GROUP_M': 8, 'WARP_SPECIALIZE': warp_specialize, 'PERSISTENT': persistent
                        },
                        num_warps=num_warps, num_stages=num_stages
                    ))
    return configs

@triton.autotune(
    configs=get_configs(),
    key=['M', 'N', 'K']
)
@triton.jit
def _descriptor_matmul(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    PERSISTENT: tl.constexpr,
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
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    num_pid_in_group = GROUP_M * num_pid_n

    if PERSISTENT:
        start_pid = tl.program_id(0)
        # Bounded program instances sequential processing of tiles
        for tile_id in tl.range(
            start_pid,
            num_tiles,
            NUM_SMS,
            flatten=False,
            warp_specialize=WARP_SPECIALIZE,
        ):
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            offset_m = pid_m * BLOCK_M
            offset_n = pid_n * BLOCK_N
            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

            for k_tile in range(num_k_tiles):
                offset_k = k_tile * BLOCK_K
                a = a_desc.load([offset_m, offset_k])
                b = b_desc.load([offset_n, offset_k])
                acc = tl.dot(a, b.T, acc)

            c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))
    else:
        # Standard software-pipeline 1D clustered dispatch (hardware tail execution balancing)
        tile_id = tl.program_id(0)
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            acc = tl.dot(a, b.T, acc)

        c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """
    Computes C = A @ B.T directly into pre-allocated memory C.
    Inputs:
    A: [M, K]
    B: [N, K] 
    C: [M, N] preallocated output tensor
    """
    global _allocator_set
    if not _allocator_set:
        triton.set_allocator(_alloc_fn)
        _allocator_set = True

    torch.cuda.set_device(A.device)
    M, K = A.shape
    N = B.shape[0]

    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count

    def grid_fn(meta):
        num_pid_m = triton.cdiv(M, meta['BLOCK_M'])
        num_pid_n = triton.cdiv(N, meta['BLOCK_N'])
        num_tiles = num_pid_m * num_pid_n
        
        # Grid adaptation based on dynamic compile-time persistent branching
        if meta['PERSISTENT']:
            return (min(num_sms, num_tiles),)
        else:
            return (num_tiles,)

    _descriptor_matmul[grid_fn](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        NUM_SMS=num_sms,
    )