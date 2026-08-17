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
    STAGES: tl.constexpr,
):
    dtype = c_desc.dtype.element_ty
    
    start_pid = tl.program_id(0)
    num_n_tiles = (N + BLOCK_N - 1) // BLOCK_N
    num_steps = ((M + BLOCK_M - 1) // BLOCK_M) * num_n_tiles
    
    tmp_c_0 = c_desc.load([-1, -1])
    tmp_c_1 = c_desc.load([-1, -1])
    
    for step in tl.range(
        start_pid,
        num_steps,
        NUM_SMS,
        flatten=False,
        warp_specialize=False,
    ):
        pid_n = step // num_n_tiles
        pid_m = step % num_n_tiles
        
        offset_n = pid_n * BLOCK_N
        offset_m = pid_m * BLOCK_M
        
        if offset_m >= M:
            continue
        if offset_n >= N:
            continue
            
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        num_k_tiles = (K + BLOCK_K - 1) // BLOCK_K
        for k_tile in tl.range(
            0,
            num_k_tiles,
            1,
            flatten=False,
            warp_specialize=False,
            num_stages=STAGES,
        ):
            offset_k = k_tile * BLOCK_K
            
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            acc = tl.dot(a, b.T, acc)
        
        if step % 2 == 0:
            local_c = tmp_c_0
        else:
            local_c = tmp_c_1
            
        local_c[:] = acc.to(dtype)
        c_desc.store([offset_m, offset_n], local_c)


def run(A, B, C):
    """Compute C = A @ B.T using optimized Hopper GEMM logic."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    if M == 0:
        return

    BLOCK_M = 256
    BLOCK_N = 128
    BLOCK_K = 16
    NUM_SMS = 128
    STAGES = 3
    
    dummy_a_ptr = torch.empty((1, 1), dtype=torch.bfloat16, device=A.device)
    a_desc = TensorDescriptor.from_tensor(dummy_a_ptr, [BLOCK_M, BLOCK_K])
    a_desc._base = A.data_ptr()
    a_desc._shape = [M, K]
    a_desc._strides = [K, 1]
    
    dummy_b_ptr = torch.empty((1, 1), dtype=torch.bfloat16, device=B.device)
    b_desc = TensorDescriptor.from_tensor(dummy_b_ptr, [BLOCK_N, BLOCK_K])
    b_desc._base = B.data_ptr()
    b_desc._shape = [N, K]
    b_desc._strides = [K, 1]
    
    dummy_c_ptr = torch.empty((1, 1), dtype=torch.bfloat16, device=C.device)
    c_desc = TensorDescriptor.from_tensor(dummy_c_ptr, [BLOCK_M, BLOCK_N])
    c_desc._base = C.data_ptr()
    c_desc._shape = [M, N]
    c_desc._strides = [N, 1]
    
    grid = (NUM_SMS,)
    
    _gemm_kernel[grid](
        a_desc, b_desc, c_desc,
        M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_SMS=NUM_SMS,
        STAGES=STAGES,
        num_warps=4,
    )