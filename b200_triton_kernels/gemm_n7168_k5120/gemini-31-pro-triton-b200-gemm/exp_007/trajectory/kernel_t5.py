import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Configure Triton JIT to allow device-side creation of tensor descriptors for Blackwell TMA.
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
    multiple M tiles sequentially for a given N group.
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
    # Exhaustively search across grouping behavior, TMA staging depths, and Blackwell warp roles.
    for group_m in [4, 8]:
        for ws in [False, True]:
            # Automatic warp specialization distributes instruction roles asymmetrically across warps. 
            # Providing higher counts strictly when WS is enabled resolves SM resource pressure issues.
            warps_options = [8, 12] if ws else [4, 8]
            for stages in [2, 3, 4, 5]:
                for bm, bn in [
                    (256, 128),
                    (128, 256),
                    (128, 128),
                    (256, 64),
                    (64, 256),
                    (128, 64),
                    (64, 128),
                ]:
                    # To completely bypass "illegal tcgen05 MMA layout" errors observed with specific B-tile Transpositions,
                    # we statically anchor BLOCK_K at 64 elements which is natively legal for 16-bit WGMMA.
                    bk = 64
                    for warps in warps_options:
                        # Cull configurations that underserve large register blocks or exceed B200's SMEM
                        if bm * bn >= 32768 and warps == 4:
                            continue
                        
                        smem_kb = (bm * bk + bn * bk) * 2 * stages / 1024
                        if smem_kb >= 220:
                            continue
                            
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
    
    # 2D execution ordering translates directly to temporal cache locality hints
    pid_m, pid_n = _grouped_tile_coordinates(
        tile_id,
        num_pid_m,
        num_pid_n,
        GROUP_M,
    )
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    # Device-created descriptors lower seamlessly into Blackwell TMA asynchronous loads.
    # Out-of-bounds coordinates are autonomously zero-padded based on 'shape' and 'padding_option'.
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Pipelined loop optionally subdivided into async TMA/MMA producer and consumer partitions
    for k0 in tl.range(0, k_tiles, num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        a_tile = a_desc.load([offset_m, k0 * BLOCK_K])
        b_tile = b_desc.load([offset_n, k0 * BLOCK_K])
        
        # B is loaded physically coalesced as [BLOCK_N, BLOCK_K]. 
        # Its `.T` mapping yields [BLOCK_K, BLOCK_N] efficiently matching matrix multiplication rules.
        acc = tl.dot(a_tile, b_tile.T, acc)
        
    # Scale conversion delayed to post-reduction ensures minimal numerical precision collapse
    acc_cast = acc.to(tl.bfloat16)
    
    # Destination storing leverages TMA hardware bypassing manual pointer bounds arithmetic routines
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N], padding_option="zero"
    )
    c_desc.store([offset_m, offset_n], acc_cast)

def run(A, B, C):
    """
    Computes a generalized matrix multiplication C = A @ B.T structurally targeting Blackwell TMA endpoints.
    Abides by destination-passing syntax allocating results securely into predefined storage blocks.
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