import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Configure Triton allocator for device-side infrastructure (if any internal lowering needs it)
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
    for group_m in [4, 8]:
        for ws in [False, True]:
            # Warp specialization splits producer/consumer roles, so higher warp counts can be vital.
            warps_list = [4, 8, 12] if ws else [4, 8]
            for stages in [2, 3, 4, 5]:
                for bm, bn, bk in [
                    (256, 128, 128),
                    (128, 256, 128),
                    (128, 128, 128),
                    (256, 128, 64),
                    (128, 256, 64),
                    (256, 64, 128),
                    (64, 256, 128),
                    (128, 128, 64),
                    (64, 128, 64),
                ]:
                    # B200 SMEM per SM is up to 228 KiB.
                    # Each element is 2 bytes (bfloat16).
                    smem_kb = (bm * bk + bn * bk) * 2 * stages / 1024
                    
                    if smem_kb > 220:
                        continue
                    
                    for warps in warps_list:
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

def update_descriptors(kwargs):
    """
    Config pre_hook: Mutates the host descriptors with the chosen autotuned block shapes.
    Preferring host descriptors (rather than tl.make_tensor_descriptor) prevents repeated
    device-side descriptor overhead and guarantees safe autotuner exploration.
    """
    bm = kwargs["BLOCK_M"]
    bn = kwargs["BLOCK_N"]
    bk = kwargs["BLOCK_K"]
    kwargs["a_desc"] = TensorDescriptor.from_tensor(kwargs["A"], [bm, bk])
    kwargs["b_desc"] = TensorDescriptor.from_tensor(kwargs["B"], [bn, bk])
    kwargs["c_desc"] = TensorDescriptor.from_tensor(kwargs["C"], [bm, bn])

@triton.autotune(
    configs=get_configs(),
    key=["M", "N", "K"],
    pre_hook=update_descriptors,
)
@triton.jit
def gemm_kernel(
    A, B, C,
    a_desc, b_desc, c_desc,
    M, N, K,
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
    
    pid_m, pid_n = _grouped_tile_coordinates(
        tile_id,
        num_pid_m,
        num_pid_n,
        GROUP_M,
    )
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    # Initialize FP32 accumulator to map natively into Blackwell TMEM
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    # TMA memory loops benefit from num_stages tracking in the range block structure
    for k0 in tl.range(0, k_tiles, num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        # TMA bounds checks are naturally covered by the hardware descriptors
        a_tile = a_desc.load([offset_m, k0 * BLOCK_K])
        b_tile = b_desc.load([offset_n, k0 * BLOCK_K])
        
        # B is physically `[N, K]` loaded as `[BLOCK_N, BLOCK_K]`. 
        # By passing b_tile.T, it transforms correctly into `[BLOCK_K, BLOCK_N]` for matrix multiplication.
        acc = tl.dot(a_tile, b_tile.T, acc, out_dtype=tl.float32)
        
    # Convert fully reduced accumulator precision just before epilogue
    acc_cast = acc.to(tl.bfloat16)
    c_desc.store([offset_m, offset_n], acc_cast)

def run(A, B, C):
    """
    Computes generalized matrix multiplication C = A @ B.T.
    Follows destination-passing format: evaluates results into allocated output tensors.
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    # Pre-instantiate dummy shape descriptors that pass Triton JIT's type checker initially.
    # These gets successfully overwritten by the pre_hook logic with optimized configurations.
    dummy_a = TensorDescriptor.from_tensor(A, [128, 64])
    dummy_b = TensorDescriptor.from_tensor(B, [128, 64])
    dummy_c = TensorDescriptor.from_tensor(C, [128, 128])
    
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    
    gemm_kernel[grid](
        A=A, B=B, C=C,
        a_desc=dummy_a, b_desc=dummy_b, c_desc=dummy_c,
        M=M, N=N, K=K,
    )