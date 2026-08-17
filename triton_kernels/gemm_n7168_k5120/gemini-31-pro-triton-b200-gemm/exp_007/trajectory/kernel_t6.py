import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Configure Triton allocator for device-side infrastructure required for TMA descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.jit
def _grouped_tile_coordinates(
    tile_id,
    num_pid_m,
    num_pid_n,
    GROUP_M: tl.constexpr,
):
    """
    Groups output tiles to improve L2 cache residency by computing 
    multiple M tiles sequentially for a given N group. This promotes 
    temporal reuse of the B operand matrix in the L2 cache.
    """
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    return pid_m, pid_n

def get_configs():
    configs = []
    # Broad config sweep targeting peak standard Tensor Core throughput on Blackwell.
    # Excludes `warp_specialize` to maintain a robust baseline free from layout restrictions,
    # relying strictly on optimal software pipelining (NUM_STAGES) and TMA overlaps.
    for group_m in [1, 4, 8]:
        for stages in [2, 3, 4, 5]:
            for bm, bn, bk, warps in [
                (64, 64, 64, 4),
                (128, 128, 64, 4),
                (128, 128, 64, 8),
                (128, 256, 64, 4),
                (128, 256, 64, 8),
                (256, 128, 64, 4),
                (256, 128, 64, 8),
                (256, 256, 64, 8),
                (128, 128, 128, 4),
                (128, 128, 128, 8),
                (128, 256, 128, 8),
                (256, 128, 128, 8),
            ]:
                # SMEM capacity check for B200 (max 228 KiB per SM). 
                # Formula assumes bf16 (2 bytes) for A and B.
                smem_kb = (bm * bk + bn * bk) * 2 * stages / 1024
                if smem_kb >= 220:
                    continue
                
                configs.append(triton.Config(
                    {
                        "BLOCK_M": bm,
                        "BLOCK_N": bn,
                        "BLOCK_K": bk,
                        "GROUP_M": group_m,
                        "NUM_STAGES": stages,
                    },
                    num_warps=warps,
                    num_stages=stages,
                ))
    return configs

@triton.autotune(
    configs=get_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # 2D execution ordering translates directly to temporal cache locality hints
    pid_m, pid_n = _grouped_tile_coordinates(
        tile_id,
        num_pid_m,
        num_pid_n,
        GROUP_M,
    )
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    # Dynamic tensor memory descriptor configurations map directly to Blackwell TMA endpoints
    # Out-of-bounds coordinates are autonomously zero-padded.
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )
    
    # MMA operates strictly against unscaled accumulations inherently mapped to TMEM
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Canonical TMA loop structured with `num_stages` ensuring hardware issues asynchronous prefetch steps
    for k0 in tl.range(0, k_tiles, num_stages=NUM_STAGES):
        a_tile = a_desc.load([offset_m, k0 * BLOCK_K])
        b_tile = b_desc.load([offset_n, k0 * BLOCK_K])
        
        # B is physically structured as [BLOCK_N, BLOCK_K]. `.T` correctly applies logical [BLOCK_K, BLOCK_N] geometry
        acc = tl.dot(a_tile, b_tile.T, acc)
        
    # Standard output casting deferred explicitly beyond full accumulator reduction
    acc_cast = acc.to(tl.bfloat16)
    
    # TMA-bound epilogue perfectly satisfies boundary limitations 
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N], padding_option="zero"
    )
    c_desc.store([offset_m, offset_n], acc_cast)


def run(A, B, C):
    """
    Computes generalized matrix multiplication C = A @ B.T.
    Strictly observes destination-passing behavior writing back closely to preallocated arrays.
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    
    gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )