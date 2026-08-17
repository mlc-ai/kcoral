import torch
import triton
import triton.language as tl

# Triton requires an allocator setup to utilize fast device-created TMA descriptors
def _descriptor_allocator(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_descriptor_allocator)

def get_configs():
    configs = []
    # Exhaustively tuned for Blackwell B200 TMEM/TMA path limits
    for ws in [True, False]:
        for block_m, block_n, block_k in [
            (256, 128, 128),
            (128, 256, 128),
            (128, 128, 128),
            (256, 64, 128),
            (64, 256, 128),
            (128, 64, 128),
            (64, 128, 128),
            (128, 128, 64),
            (256, 128, 64),
            (128, 256, 64),
        ]:
            for warps in [4, 8]:
                for stages in [3, 4, 5]:
                    # Rigorous filter heuristics to prevent compile/spill failures
                    if ws and warps == 4:
                        continue
                    if stages == 5 and block_k == 128:
                        continue
                    if block_m * block_n >= 32768 and warps == 4:
                        continue
                    
                    configs.append(triton.Config(
                        {
                            'BLOCK_M': block_m, 
                            'BLOCK_N': block_n, 
                            'BLOCK_K': block_k, 
                            'GROUP_M': 8, 
                            'WARP_SPECIALIZE': ws, 
                            'LOOP_STAGES': stages
                        },
                        num_warps=warps, 
                        num_stages=stages
                    ))
    return configs


@triton.autotune(configs=get_configs(), key=["M", "N", "K"])
@triton.jit
def _gemm_tma_device_kernel(
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
    LOOP_STAGES: tl.constexpr,
):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2 cache-aware grouped tile traversing map bounds memory HBM re-reads implicitly
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    
    # Base spatial coordinates for native TMA translation boundaries
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    
    # Native device-created descriptors for A and B entirely bypasses slow python cuTensorMapEncode driver calls
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero"
    )
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Accelerated core pipelined async loads utilizing Automatic Warp Specialization 
    for k0 in tl.range(0, k_tiles, num_stages=LOOP_STAGES, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([offs_m, k0 * BLOCK_K])
        b = b_desc.load([offs_n, k0 * BLOCK_K])
        
        # Matrix B implicit layout mapping to expected orientation (B transpose internally loaded)
        acc = tl.dot(a, b.T, acc)
        
    # Epilogue standard pointer store entirely isolates boundaries to prevent MMAv5 legalization TMEM issues 
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


def run(A, B, C):
    """
    Computes generalized precision matrix multiplication C = A @ B.T.
    Targeting specific structural workloads mapped strictly to Qwen-3 bounds.
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    if M == 0 or N == 0 or K == 0:
        return
        
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    
    _gemm_tma_device_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )