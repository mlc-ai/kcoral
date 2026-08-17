import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Configure Triton JIT allocator for device-side tensor descriptors (TMA infrastructure).
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
    Computes a 2D tile layout mapping that groups outputs along M to 
    significantly improve the temporal reuse of the B matrix in L2 cache.
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
    # Exhaustively search across grouping behavior, TMA staging depths, clustering, and warp roles.
    for group_m in [4, 8]:
        for ws in [False, True]:
            for stages in [3, 4, 5]:
                for bm, bn in [
                    (128, 128),
                    (128, 256),
                    (256, 128),
                    (256, 256),
                ]:
                    # Firmly fix K=64 natively compatible with Blackwell's 16-bit WGMMA tcgen05 instructions, 
                    # evading layout transpositions errors entirely while sustaining peak throughput.
                    bk = 64
                    for warps in [4, 8]:
                        # Broaden threadblock overlaps by adding clustered compilation where supported
                        for ctas in [1, 2]:
                            # Filter combinations breaking the B200 Shared Memory ceiling (228 KiB per SM)
                            smem_kb = (bm * bk + bn * bk) * 2 * stages / 1024
                            if smem_kb >= 220:
                                continue
                            
                            # Prune warp allocations insufficient to cover exceptionally wide tile topologies
                            if bm * bn >= 65536 and warps == 4:
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
    
    # M-grouped execution translates tightly to physical L2 locality 
    pid_m, pid_n = _grouped_tile_coordinates(
        tile_id,
        num_pid_m,
        num_pid_n,
        GROUP_M,
    )
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    # Device-created descriptors lower straight into Blackwell hardware-bounded TMA requests.
    # Out-of-bounds coordinates securely evaluate to zero via 'padding_option'.
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )
    
    # FP32 accumulator initializes inherently against fast Blackwell TMEM architectures
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    # A canonical pipelined TMA loop incorporating native hardware async staging mechanisms
    for k0 in tl.range(0, k_tiles, num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        a_tile = a_desc.load([offset_m, k0 * BLOCK_K])
        b_tile = b_desc.load([offset_n, k0 * BLOCK_K])
        
        # B arrives structured logically as [BLOCK_N, BLOCK_K]. Matrix multiplication `.T` transpose 
        # converts it gracefully to RHS operand dimensions [BLOCK_K, BLOCK_N].
        acc = tl.dot(a_tile, b_tile.T, acc)
        
    # Standard precision casting executed completely outside the critical inner math loop
    acc_cast = acc.to(tl.bfloat16)
    
    # Destination storing operates via TMA bounds-free epilogue mechanics
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N], padding_option="zero"
    )
    c_desc.store([offset_m, offset_n], acc_cast)


def run(A, B, C):
    """
    Computes a generalized matrix multiplication C = A @ B.T structurally targeting Blackwell 
    fifth-generation WGMMA / TMA endpoints.
    Strictly observes destination-passing conventions mutating predefined blocks transparently.
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