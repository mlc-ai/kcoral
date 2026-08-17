import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    a_desc, b_desc, c_desc,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    num_pid_m = M // BLOCK_M
    num_pid_n = N // BLOCK_N
    num_tiles = num_pid_m * num_pid_n
    
    pid = tl.program_id(0)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_k_steps = K // BLOCK_K
    
    for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=False):
        id_m = tile_id // num_pid_n
        idx = tile_id % num_pid_n
        
        offset_m = id_m * BLOCK_M
        offset_n = idx * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_idx in range(num_k_steps):
            offset_k = k_idx * BLOCK_K
            a_tile = a_desc.load([offset_m, offset_k])
            b_tile = b_desc.load([offset_n, offset_k])
            acc = tl.dot(a_tile, b_tile.T, acc)
        
        c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 128
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    NUM_SMS = 132
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    grid = (min(NUM_SMS, num_tiles),)
    
    _gemm_kernel[grid](a_desc, b_desc, c_desc, M, N, K,
                       BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                       NUM_SMS=NUM_SMS,
                       num_warps=8, num_stages=2)