import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

def pre_hook(kwargs):
    # Establish host-side TMA descriptors dynamically for the current autotuning block sizes
    kwargs['A_desc'] = TensorDescriptor.from_tensor(kwargs['A'], [kwargs['BLOCK_M'], kwargs['BLOCK_K']])
    kwargs['B_desc'] = TensorDescriptor.from_tensor(kwargs['B'], [kwargs['BLOCK_N'], kwargs['BLOCK_K']])
    kwargs['C_desc'] = TensorDescriptor.from_tensor(kwargs['C'], [kwargs['BLOCK_M'], kwargs['BLOCK_N']])

def get_configs():
    configs = []
    # Exhaustive config selection designed specifically for Blackwell TMA & TMEM capacities
    for ws in [True, False]:
        for bm, bn, bk, ns in [
            (128, 256, 128, 2),
            (256, 128, 128, 2),
            (128, 128, 128, 3),
            (128, 128, 128, 4),
            (128, 256, 64, 3),
            (256, 128, 64, 3),
            (128, 128, 64, 4),
            (256, 256, 64, 2),
        ]:
            for nw in [4, 8, 12]:
                # Maximum shared memory per SM / CTA on Blackwell is 227 KiB
                shmem_bytes_needed = (bm + bn) * bk * 2 * ns
                if shmem_bytes_needed > 220000:
                    continue
                configs.append(triton.Config({
                    "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk,
                    "NUM_STAGES": ns, "WARP_SPECIALIZE": ws
                }, num_warps=nw, num_stages=ns))
    return configs

@triton.autotune(
    configs=get_configs(),
    key=["M", "N", "K"],
    pre_hook=pre_hook,
)
@triton.jit
def gemm_kernel_tma(
    A, B, C,                # Passed so the autotuner pre_hook has access to the tensors
    A_desc, B_desc, C_desc, # TMA Descriptors injected by pre_hook
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr, WARP_SPECIALIZE: tl.constexpr
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    # Block scheduling L2 cache swizzling (Group M iterations together to reuse B's columns)
    GROUP_M = 8
    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # TMA utilizes logical coordinate scalars, not vector offsets
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Core unrolled warp-specialized loop loading directly via TMA
    for k0 in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        a = A_desc.load([offs_m, k0 * BLOCK_K])
        b = B_desc.load([offs_n, k0 * BLOCK_K])
        
        # Matrix B implicitly transposed to correctly orient layout shape requirements
        acc = tl.dot(a, b.T, acc)
        
    # Asynchronous TMA store back directly dropping any partial/out-of-bounds segments natively
    C_desc.store([offs_m, offs_n], acc.to(tl.bfloat16))

def run(A, B, C):
    """
    Computes a general matrix multiply C = A @ B.T.
    A is expected to be [M, K] logically and physically.
    B is expected to be [N, K] logically and physically.
    C is preallocated as [M, N].
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    
    gemm_kernel_tma[grid](
        A=A, B=B, C=C,
        A_desc=None, B_desc=None, C_desc=None, # Injected efficiently via the Autotuner `pre_hook` 
        M=M, N=N, K=K
    )