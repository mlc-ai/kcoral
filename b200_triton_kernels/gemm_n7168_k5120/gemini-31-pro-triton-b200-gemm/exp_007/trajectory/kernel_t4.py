import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Configure Triton allocator for device-side tensor descriptors (TMA lowering infrastructure).
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
    # Construct an extensive configuration sweep covering block sizes, warp specialization,
    # staging depths, clustering, and execution widths to uncover the peak Blackwell TMA schedule.
    for group_m in [8]:
        for ws in [False, True]:
            for stages in [2, 3, 4, 5]:
                for bm, bn, bk in [
                    (256, 128, 128),
                    (128, 256, 128),
                    (128, 128, 128),
                    (256, 256, 64),
                    (128, 256, 64),
                    (256, 128, 64),
                    (128, 128, 64),
                    (64, 128, 64),
                    (128, 64, 64),
                ]:
                    for warps in [4, 8, 12]:
                        # Discard candidates that exceed the B200 228 KiB SMEM limit per SM
                        smem = (bm * bk + bn * bk) * 2 * stages
                        if smem > 220 * 1024:
                            continue
                        
                        # Discard functionally sub-optimal warp allocations
                        if bm * bn >= 32768 and warps == 4:
                            continue
                        if bm * bn <= 16384 and warps == 12:
                            continue
                        
                        # Include clustered compilation choices to broaden thread block scheduling overlaps
                        for ctas in [1, 2]:
                            configs.append(triton.Config(
                                {
                                    "BLOCK_M": bm,
                                    "BLOCK_N": bn,
                                    "BLOCK_K": bk,
                                    "GROUP_M": group_m,
                                    "WARP_SPECIALIZE": ws,
                                    "NUM_STAGES": stages,
                                },
                                num_warps=warps,
                                num_stages=stages,
                                num_ctas=ctas,
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
    WARP_SPECIALIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Obtain logically grouped coordinates that maximize cache reuse
    pid_m, pid_n = _grouped_tile_coordinates(
        tile_id,
        num_pid_m,
        num_pid_n,
        GROUP_M,
    )
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    # Establish dynamic tensor memory descriptor configurations mapping accurately to hardware TMAs
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )
    
    # Blackwell MMA naturally operates against unscaled accumulations natively allocated within TMEM
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    # A canonical pipelined TMA loop leveraging asynchronous stages, explicitly targeting tcgen05 paths
    for k0 in tl.range(0, k_tiles, num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        a_tile = a_desc.load([offset_m, k0 * BLOCK_K])
        b_tile = b_desc.load([offset_n, k0 * BLOCK_K])
        
        # Matrix B resides physically as [BLOCK_N, BLOCK_K], transposing exposes the appropriate layout cleanly
        acc = tl.dot(a_tile, b_tile.T, acc, out_dtype=tl.float32)
        
    # Standard output casting delayed specifically following complete block reduction
    acc_cast = acc.to(tl.bfloat16)
    
    # TMA bound store gracefully accommodates boundary rules across output shapes safely
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N], padding_option="zero"
    )
    c_desc.store([offset_m, offset_n], acc_cast)

def run(A, B, C):
    """
    Computes a generalized block matrix multiplication C = A @ B.T structurally targeting Blackwell devices.
    Strictly observes destination-passing behavior mapping execution traces tightly into predefined allocated targets.
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