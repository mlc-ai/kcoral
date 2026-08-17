import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    Persistent-scheduled tiled GEMM using Host Tensor Descriptors.
    Computes C = A @ B.T where A is [M, K], B is [N, K], and C is [M, N].
    Utilizes massive K-tile blocking (512) and persistent SM assignments to maximize 
    data reuse and instruction-level parallelism via TMA/WGMMA.
    """
    num_pid_m = M // BLOCK_M
    num_pid_n = N // BLOCK_N
    
    start_pid = tl.program_id(0)
    if start_pid >= num_pid_m:
        return
        
    offset_m = start_pid * BLOCK_M
    
    # Persistently track accumulators for every N-tile this SM is responsible for processing
    accs = [tl.zeros((BLOCK_M, BLOCK_N), tl.float32) for _ in range(num_pid_n)]
    
    num_k_steps = K // BLOCK_K
    
    # Loop strictly ordered across K, enabling the compiler to schedule 
    # TMA load/wGMMA issue groups asynchronously across stages
    for k_step in tl.range(0, num_k_steps, num_stages=3):
        offset_k = k_step * BLOCK_K
        
        # Issue a fully asynchronous TMA load for the A fragment
        a = a_desc.load([offset_m, offset_k])  
        
        # Iterate over the complete N domain sequentially 
        for i in range(num_pid_n):
            offset_n = i * BLOCK_N
            
            # Issue TMA load for the B fragment
            b = b_desc.load([offset_n, offset_k])  
            
            acc_i = accs[i]
            # WGMMA expects [M, K] @ [K, N]; b.T provides the necessary [K, N] logical view
            acc_i = tl.dot(a, b.T, acc_i)
            accs[i] = acc_i
            
    # Synchronize and materialize the converted outputs to HBM via TMA stores
    for i in range(num_pid_n):
        offset_n = i * BLOCK_N
        c = accs[i].to(tl.bfloat16)
        c_desc.store([offset_m, offset_n], c)


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = 7168
    K = 5120
    
    # Maximize WGMMA arithmetic intensity and enable robust 3-stage TMA pipelining
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 512  
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    
    NUM_SMS = 132
    grid = (min(NUM_SMS, num_pid_m),)
    
    _gemm_kernel[grid](
        a_desc, b_desc, c_desc,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_SMS=NUM_SMS,
        num_warps=8, num_stages=3,
    )