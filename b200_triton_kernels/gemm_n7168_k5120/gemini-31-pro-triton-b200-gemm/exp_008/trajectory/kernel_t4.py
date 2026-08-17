import torch
import triton
import triton.language as tl

# Standard Triton device-created descriptors require setting an infrastructure allocator
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    configs = []
    # (block_m, block_n, block_k, num_warps, num_stages, num_ctas, warp_spec)
    # SM100 limits tuning space bounds:
    # Tiles are strictly sized with `num_stages` combinations that fit within 228 KB SMEM.
    candidate_params = [
        # Clustered - 4 CTAs (Leveraging hardware TMA Multicasting)
        (256, 128, 128, 8, 2, 4, True),
        (128, 256, 128, 8, 2, 4, True),
        (256, 128, 64,  8, 4, 4, True),
        (128, 256, 64,  8, 4, 4, True),
        (128, 128, 128, 8, 3, 4, True),
        (128, 128, 128, 4, 3, 4, True),
        (128, 128, 64,  8, 4, 4, True),
        (128, 128, 64,  4, 4, 4, True),
        
        # Clustered - 8 CTAs (Maximizes L2 locality and large cluster multicast)
        (256, 128, 128, 8, 2, 8, True),
        (128, 256, 128, 8, 2, 8, True),
        (128, 128, 128, 8, 3, 8, True),
        
        # Unclustered - 1 CTA (Baselining fallback)
        (256, 128, 128, 8, 2, 1, True),
        (128, 256, 128, 8, 2, 1, True),
        (128, 128, 128, 8, 3, 1, True),
        (128, 128, 128, 4, 3, 1, True),
        
        # Unspecialized baselines
        (256, 128, 128, 8, 2, 4, False),
        (128, 256, 128, 8, 2, 4, False),
        (128, 128, 128, 8, 3, 4, False),
        (128, 128, 128, 4, 3, 1, False),
    ]
    
    for m, n, k, w, s, c, ws in candidate_params:
        configs.append(
            triton.Config(
                {
                    "BLOCK_M": m,
                    "BLOCK_N": n,
                    "BLOCK_K": k,
                    "GROUP_M": 8,
                    "WARP_SPECIALIZE": ws,
                    "NUM_STAGES": s,
                },
                num_warps=w,
                num_stages=s,
                num_ctas=c,
            )
        )
    return configs

@triton.autotune(
    configs=get_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _tma_gemm(
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
    # Device descriptors mapped directly into global memory arrays via TMA 
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
    
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    # Swizzling mapping optimized to cluster the same `pid_n` across `GROUP_M`
    # When matched with `num_ctas`, B tiles naturally leverage hardware TMA multicast.
    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    offset_n = (pid_n * BLOCK_N).to(tl.int32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    # Core computation lowered via async TMA pipelining mapping natively to WGMMA WGs 
    for k0 in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        offset_k = (k0 * BLOCK_K).to(tl.int32)
        a_tile = a_desc.load([offset_m, offset_k])
        b_tile = b_desc.load([offset_n, offset_k])
        acc = tl.dot(a_tile, b_tile.T, acc)

    # Natively flushes TMEM straight out to Global over TMA
    c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))

def run(A, B, C):
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N = B.shape[0]

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _tma_gemm[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )