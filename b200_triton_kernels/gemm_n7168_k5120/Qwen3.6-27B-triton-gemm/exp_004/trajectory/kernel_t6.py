import torch
import triton
import triton.language as tl


# Install descriptor allocator for device-created tensor descriptors.
def _alloc_fn(size: int, alignment: int, stream=None):
    return torch.empty(size, device="cuda", dtype=torch.int8)


try:
    triton.set_allocator(_alloc_fn)
except Exception:
    pass


@triton.jit
def _gemm_persistent_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    # Create device-side tensor descriptors
    A_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero",
    )
    B_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[N, K],
        strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero",
    )
    C_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N],
        padding_option="zero",
    )

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_k_steps = tl.cdiv(K, BLOCK_K)

    dtype = tl.bfloat16

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE,
    ):
        # L2-aware grouped tile mapping
        pid_in_group = tile_id % (GROUP_M * num_pid_n)
        group_id = tile_id // (GROUP_M * num_pid_n)
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m

        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(num_k_steps):
            a_tile = A_desc.load([off_m, k * BLOCK_K])
            b_tile = B_desc.load([off_n, k * BLOCK_K])
            acc = tl.dot(a_tile, b_tile.T, acc=acc)

        C_desc.store([off_m, off_n], acc.to(dtype))


def run(A, B, C):
    """Compute C = A @ B.T into preallocated output tensor C.

    Persistent kernel with optional warp specialization on Hopper:
      - Producer warps execute TMA async loads
      - Consumer warps execute WGMMA tensor core math
    L2-aware grouped ordering maximizes operand cache residency.
    """
    torch.cuda.set_device(A.device)

    M, K = A.shape
    N = B.shape[0]

    NUM_SMS = 132  # H100 SXM SM count

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_M = 8

    # Cap grid at number of available SMs for persistent scheduling
    total_programs = min(
        NUM_SMS,
        triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
    )
    grid = (total_programs,)

    # Launch with warp_specialize=True for Hopper producer/consumer pipeline
    _gemm_persistent_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        NUM_SMS=NUM_SMS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        WARP_SPECIALIZE=True,
        num_warps=8,
        num_stages=3,
    )