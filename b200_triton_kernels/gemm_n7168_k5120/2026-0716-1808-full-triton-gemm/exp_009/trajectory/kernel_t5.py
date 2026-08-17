import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    A_desc,
    B_desc,
    C_desc,
    M,
    N_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BK: tl.constexpr,
):
    start_tile = tl.program_id(0)
    stride = tl.num_programs(0)
    
    num_tiles_m = tl.cdiv(M, BLOCK_M)
    num_tiles_n = tl.cdiv(N_dim, BLOCK_N * 2)
    num_tiles = num_tiles_m * num_tiles_n
    
    for tile_id in tl.range(start_tile, num_tiles, stride):
        pid_m = tile_id // num_tiles_n
        pid_n = tile_id % num_tiles_n
        
        m_offset = pid_m * BLOCK_M
        n_offset = pid_n * (BLOCK_N * 2)
        
        if m_offset >= M:
            break
            
        acc0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        acc1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        # K is hardcoded to 5120, split evenly into two loops of 2560.
        # Using static_range and literal integer division allows full compiler unrolling.
        for k0 in tl.static_range(0, 2560 // BK):
            k = k0 * BK
            a0 = A_desc.load([m_offset, k])
            b0_0 = B_desc.load([n_offset, k])
            b1_0 = B_desc.load([n_offset + BLOCK_N, k])
            
            acc0 = tl.dot(a0, b0_0, acc0)
            acc1 = tl.dot(a0, b1_0, acc1)
            
        for k0 in tl.static_range(0, 2560 // BK):
            k = 2560 + k0 * BK
            a1 = A_desc.load([m_offset, k])
            b0_1 = B_desc.load([n_offset, k])
            b1_1 = B_desc.load([n_offset + BLOCK_N, k])
            
            acc0 = tl.dot(a1, b0_1, acc0)
            acc1 = tl.dot(a1, b1_1, acc1)
        
        C_desc.store([m_offset, n_offset], acc0.to(tl.bfloat16))
        C_desc.store([m_offset, n_offset + BLOCK_N], acc1.to(tl.bfloat16))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N_dim = B.shape[0]
    
    # Software pipelining parameters and tiling configuration
    block_m = 128
    block_n = 256
    block_k = 128
    
    # Utilize persistent execution mapping capped by available SMs.
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    num_tiles_m = (M + block_m - 1) // block_m
    num_tiles_n = (N_dim + block_n * 2 - 1) // (block_n * 2)
    num_tiles = num_tiles_m * num_tiles_n
    grid = (min(num_sms, num_tiles),) 
    
    # Pass tensors directly via TensorDescriptors; block shapes define TMA tile sizes.
    A_desc = TensorDescriptor.from_tensor(A, [block_m, block_k])
    B_desc = TensorDescriptor.from_tensor(B, [block_n, block_k])
    C_desc = TensorDescriptor.from_tensor(C, [block_m, block_n * 2])
    
    _gemm_kernel[grid](
        A_desc,
        B_desc,
        C_desc,
        M,
        N_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BK=block_k,
        num_warps=4,
        num_stages=3,
    )