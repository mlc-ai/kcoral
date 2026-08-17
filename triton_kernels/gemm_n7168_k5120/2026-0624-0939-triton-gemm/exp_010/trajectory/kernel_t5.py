import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def persistent_gemm(
    a_desc, b_desc, c_desc, l2_order_ptr,
    M, N, K,
    num_pid_m, num_pid_n, num_tiles,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    start_pid = tl.program_id(0)
    tile_stride = tl.num_programs(0)
    
    for tile_id in range(start_pid, num_tiles, tile_stride):
        idx = tile_id // num_pid_n
        if idx < len(l2_order):
            m_offset, n_offset = tl.load(l2_order_ptr + tile_id * 2, mask=tile_id < num_tiles)
        else:
            idx = tile_id // num_pid_n
            pid_m = idx // num_pid_n
            pid_n = idx % num_pid_n
            m_offset = pid_m * BLOCK_M
            n_offset = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        num_k_tiles = tl.cdiv(K, BLOCK_K)
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([m_offset, offset_k])
            b = b_desc.load([n_offset, offset_k])
            acc = tl.dot(a, b.T, acc)
        
        out = acc.to(tl.bfloat16)
        c_desc.store([m_offset, n_offset], out)


def run(A, B, C):
    """Compute C = A @ B.T into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    
    M, K = A.shape
    N = B.shape[0]
    
    assert K == 5120
    assert N == 7168
    
    BLOCK_M = 128
    BLOCK_N = 512
    BLOCK_K = 128
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    l2_order = torch.empty((num_tiles, 2), dtype=torch.float32, device=A.device)
    for n in range(num_pid_n):
        for m in range(num_pid_m):
            n_outer_idx = n * num_pid_m + m
            l2_order[n_outer_idx, 0] = m * BLOCK_M
            l2_order[n_outer_idx, 1] = n * BLOCK_N
            
    l2_order_ptr = l2_order.contiguous().data_ptr()
    
    NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count
    grid = (min(NUM_SMS, num_tiles),)
    
    persistent_gemm[grid](
        a_desc, b_desc, c_desc, l2_order_ptr, M, N, K,
        num_pid_m, num_pid_n, num_tiles,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=4, maxnreg=255,
    )