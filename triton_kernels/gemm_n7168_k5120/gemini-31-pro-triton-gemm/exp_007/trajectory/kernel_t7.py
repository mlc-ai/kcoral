import torch
import triton
import triton.language as tl

# Standard Triton descriptor allocator for device-created TensorDescriptors.
# This serves as infrastructure storage mapped on the host, specifically 
# required for the Hopper TMA warp-specialized persistent contract.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    """
    Carefully filtered configurations strictly tailored to Hopper WGMMA constraints.
    Avoids 256-sized blocks which can trigger WGMMA layout lowering failures.
    Calculates exact shared memory footprint to compute the optimal PROGS_PER_SM 
    for maximum occupancy on the H100 device.
    """
    configs = []
    
    # Base shapes (BLOCK_M, BLOCK_N, BLOCK_K, num_stages)
    shapes = [
        # Standard robust paths (Max computing efficiency)
        (128, 128, 128, 2),
        (128, 128, 128, 3),
        
        # Latency-hiding heavy paths
        (128, 128, 64, 2),
        (128, 128, 64, 3),
        (128, 128, 64, 4),
        (128, 128, 64, 5),
        
        # High M/N ratio alternatives
        (128, 64, 128, 2),
        (128, 64, 128, 3),
        (64, 128, 128, 2),
        (64, 128, 128, 3),
        
        # Skinny fallback paths
        (128, 64, 64, 3),
        (128, 64, 64, 4),
        (64, 128, 64, 3),
        (64, 128, 64, 4),
    ]
    
    for m, n, k, s in shapes:
        # Calculate precise Shared Memory requirements for TMA AB staging and C store
        smem_a = m * k * 2
        smem_b = n * k * 2
        smem_c = m * n * 2
        total_smem = (smem_a + smem_b) * s + smem_c
        
        # Hopper H100 CTA Shared Memory limit is ~228 KB
        if total_smem > 233000:
            continue
            
        # Determine maximum concurrently executing CTAs per SM
        progs_per_sm = 1
        if total_smem <= 114000:
            progs_per_sm = 2
            
        for warp_spec in [True, False]:
            for w in [4, 8]:
                for group in [8]:
                    configs.append(triton.Config(
                        {'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 
                         'GROUP_M': group, 'WARP_SPECIALIZE': warp_spec,
                         'PROGS_PER_SM': progs_per_sm},
                        num_warps=w, num_stages=s
                    ))
    return configs

@triton.autotune(
    configs=get_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_tma_persistent_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_bn, stride_cm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr, 
    PROGS_PER_SM: tl.constexpr, WARP_SPECIALIZE: tl.constexpr
):
    """
    Standard Hopper warp-specialized descriptor loop.
    Maps an entire GEMM over persistently scheduled thread blocks distributed across all SMs.
    TMA inherently bypasses boundary software-masks for clean bounds handling.
    """
    dtype = C_ptr.dtype.element_ty
    
    a_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[M, K],
        strides=[stride_am, 1],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[N, K],
        strides=[stride_bn, 1],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[M, N],
        strides=[stride_cm, 1],
        block_shape=[BLOCK_M, BLOCK_N]
    )

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    TOTAL_PROGRAMS = NUM_SMS * PROGS_PER_SM
    
    # Persistent hardware loop bounded identically by optimal active CTAs
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        TOTAL_PROGRAMS,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE
    ):
        # L2-Aware Grouped Tile Mapping 
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        
        pid_in_group = tile_id % num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m
        
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        # Accumulate natively in FP32 on Hopper Tensor Cores
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            # Since physically B is [N, K], transposing b guarantees passing a proper column-major 
            # register layout to the dot operation triggering optimal WGMMA instruction mappings automatically.
            acc = tl.dot(a, b.T, acc)
            
        c_desc.store([offset_m, offset_n], acc.to(dtype))

def run(A, B, C):
    """
    Public destination-passing execution function evaluating C = A @ B.T robustly across sequences.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    if M == 0 or N == 0 or K == 0:
        return
        
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    def grid_fn(META):
        num_tiles = triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"])
        total_programs = num_sms * META["PROGS_PER_SM"]
        # Limit grid instantiation to maximal active device capacity, relying natively on hardware loop persistence.
        return (min(total_programs, num_tiles),)
    
    _gemm_tma_persistent_kernel[grid_fn](
        A, B, C,
        M, N, K,
        A.stride(0), B.stride(0), C.stride(0),
        NUM_SMS=num_sms
    )