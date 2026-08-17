import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_persistent(
    A_desc, B_desc, C_desc,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr
):
    """
    Persistent GEMM kernel targeting Hopper.
    
    Uses Hardware Transactional Memory Access (TMA) via compile-time TensorDescriptors 
    and optimized tensor core reduction to compute C = A @ B^T.
    """
    
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False, warp_specialize=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            
            a = A_desc.load([offset_m, offset_k])
            b = B_desc.load([offset_n, offset_k])
            
            acc = tl.dot(a, b.T, acc)
            
        c = acc.to(tl.bfloat16)
        C_desc.store([offset_m, offset_n], c)


def run(A, B, C):
    """
    Compute C = A @ B.T on Hopper using a persistent grid and TMA.
    
    Args:
        A: Input tensor of shape [M, 5120] and dtype bfloat16.
        B: Input tensor of shape [7168, 5120] and dtype bfloat16.
        C: Preallocated output tensor of shape [M, 7168] and dtype bfloat16.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = 7168
    K = 5120
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    NUM_SMS = 132 
    
    A_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    B_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    C_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    grid = (min(NUM_SMS, num_tiles),)
    
    _gemm_persistent[grid](
        A_desc, B_desc, C_desc,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_SMS=NUM_SMS,
        num_warps=4, num_stages=2,
    )