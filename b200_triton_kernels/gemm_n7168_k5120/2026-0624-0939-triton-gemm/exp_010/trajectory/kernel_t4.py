import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from itertools import product


@triton.jit
def persistent_gemm(
    a_desc, b_desc, c_desc,
    M, N, K,
    num_pid_m, num_pid_n,
    l2_order,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    start_pid = tl.program_id(0)
    tile_stride = tl.num_programs(0)
    
    num_tiles = num_pid_m * num_pid_n
    
    for tile_id in range(start_pid, num_tiles, tile_stride):
        idx = tile_id // num_pid_n
        if idx < len(l2_order):
            pid_m, pid_n = l2_order[idx]
        else:
            pid_m = idx // num_pid_n
            pid_n = idx % num_pid_n
        
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        num_k_tiles = tl.cdiv(K, BLOCK_K)
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            acc = tl.dot(a, b.T, acc)
        
        out = acc.to(tl.bfloat16)
        c_desc.store([offset_m, offset_n], out)


def run(A, B, C):
    """Compute C = A @ B.T into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    
    M, K = A.shape
    N = B.shape[0]
    
    assert K == 5120
    assert N == 7168
    
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    # Reorder output iterations to massively increase L2 hit rates
    order = list(product(range(num_pid_m), range(num_pid_n)))
    l2_order = [None] * len(order)
    for i, (m, n) in enumerate(order):
        interleaved = 0
        max_bits = max(m, n).bit_length()
        for bit in range(max_bits):
            interleaved |= ((m >> bit) & 1) << (2 * bit)
            interleaved |= ((n >> bit) & 1) << (2 * bit + 1)
        l2_order[interleaved] = (m, n)
    
    NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count
    grid = (min(NUM_SMS, num_tiles),)
    
    persistent_gemm[grid](
        a_desc, b_desc, c_desc, M, N, K,
        num_pid_m, num_pid_n,
        l2_order,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=4,
    )