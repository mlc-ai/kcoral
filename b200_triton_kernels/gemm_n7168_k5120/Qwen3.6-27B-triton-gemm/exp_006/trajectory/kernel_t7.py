import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Device-side tensor descriptors
    a_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero",
    )
    b_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero",
    )
    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
        padding_option="zero",
    )

    # Grouped 1-D grid mapping
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    start_pid = tl.program_id(0)
    tile_stride = tl.num_programs(0)

    for tile_id in tl.range(start_pid, num_tiles, tile_stride, num_stages=0):
        # Compute pid_m, pid_n with grouped ordering
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

        pid_in_group = tile_id % num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m

        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        num_k_tiles = tl.cdiv(K, BLOCK_K)
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            # B stored [N,K], load [BLOCK_N, BLOCK_K], then transpose to [BLOCK_K, BLOCK_N]
            b = b_desc.load([offset_n, offset_k])
            acc = tl.dot(a, b.T, acc)

        c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute C = A @ B.T into preallocated output C."""
    torch.cuda.set_device(A.device)

    # Install allocator for device-created tensor descriptors
    def _alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(_alloc_fn)

    M = A.shape[0]
    N = B.shape[0]   # 7168
    K = B.shape[1]   # 5120

    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 128
    GROUP_M = 8

    # Launch many CTAs for persistent scheduling
    NUM_SMS = 132  # H100 SXM
    max_ctas = min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N))
    grid = (max_ctas,)

    _gemm_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_warps=8,
        num_stages=4,
    )