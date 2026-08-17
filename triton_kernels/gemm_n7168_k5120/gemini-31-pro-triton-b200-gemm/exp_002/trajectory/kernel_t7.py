import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

def pre_hook(kwargs):
    # Establish host-side TMA descriptors dynamically tuned to the autotuner's block dimensions.
    # This completely eliminates JIT device memory allocations and associated dispatcher latency.
    bm = kwargs["BLOCK_M"]
    bn = kwargs["BLOCK_N"]
    bk = kwargs["BLOCK_K"]
    kwargs["A_desc"] = TensorDescriptor.from_tensor(kwargs["A_ptr"], [bm, bk])
    kwargs["B_desc"] = TensorDescriptor.from_tensor(kwargs["B_ptr"], [bn, bk])
    kwargs["C_desc"] = TensorDescriptor.from_tensor(kwargs["C_ptr"], [bm, bn])

def get_configs():
    configs = []
    # Exhaustive tuning space tailored exactly to Blackwell's 227 KiB max shared memory ceiling.
    # High num_stages efficiently hides global memory latency when TMA is deployed.
    for ws in [True, False]:
        for bm, bn, bk in [
            (256, 256, 64),
            (128, 256, 128),
            (256, 128, 128),
            (128, 256, 64),
            (256, 128, 64),
            (128, 128, 128),
        ]:
            for ns in [2, 3, 4, 5]:
                for nw in [4, 8]:
                    shmem = (bm + bn) * bk * 2 * ns
                    if shmem <= 227000:
                        configs.append(triton.Config({
                            'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk,
                            'NUM_STAGES': ns, 'WARP_SPECIALIZE': ws, 'GROUP_M': 8
                        }, num_warps=nw))
    return configs


@triton.autotune(
    configs=get_configs(),
    key=["M", "N", "K"],
    pre_hook=pre_hook,
)
@triton.jit
def gemm_kernel_tma_host(
    A_ptr, B_ptr, C_ptr,      # Standard PyTorch input references (harvested by pre_hook)
    A_desc, B_desc, C_desc,   # Host TMA descriptors
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr, WARP_SPECIALIZE: tl.constexpr, GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    # Mathematical L2 cache swizzling. Unrolled safely knowing sizes are unconditionally divisible.
    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % GROUP_M)
    pid_n = (pid % num_pid_in_group) // GROUP_M
    
    # TMA descriptors logically coordinate element indexing directly
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Fully unrolled loop leveraging automatic warp specialization seamlessly
    for k0 in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        a = A_desc.load([offs_m, k0 * BLOCK_K])
        b = B_desc.load([offs_n, k0 * BLOCK_K])
        
        # B Matrix transposes in logical flow to conform to SM100 TC hardware orientation requirements
        acc = tl.dot(a, b.T, acc)
        
    # Standard TMA asynchronous store natively handling constraints
    C_desc.store([offs_m, offs_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """
    Computes a general matrix multiply C = A @ B.T.
    A is logically/physically aligned as [M, K]
    B is logically/physically aligned as [N, K]
    C is initialized and written back logically as [M, N]
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    # Dummy descriptors establish expected cache types to avoid compilation dispatcher confusion.
    # Their actual block configurations are rapidly corrected & overridden inside the Config pre_hook.
    dummy_A = TensorDescriptor.from_tensor(A, [16, 16])
    dummy_B = TensorDescriptor.from_tensor(B, [16, 16])
    dummy_C = TensorDescriptor.from_tensor(C, [16, 16])
    
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    
    gemm_kernel_tma_host[grid](
        A_ptr=A, B_ptr=B, C_ptr=C,
        A_desc=dummy_A, B_desc=dummy_B, C_desc=dummy_C,
        M=M, N=N, K=K
    )