import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    A_desc, B_desc, C_ptr,
    M, N, K,
    stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    NUM_SM = 132
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    num_sm = min(NUM_SM, num_m_tiles)
    
    start_m_idx = tl.program_id(0)
    
    for m_idx in range(start_m_idx, num_m_tiles, num_sm):
        offset_m = m_idx * 256
        offs_m = offset_m + tl.arange(0, BLOCK_M)
        
        num_k_levels = 7
        batch_start = m_idx * 40
        
        for _ in range(num_k_levels):
            n_idx_0 = batch_start // 40 + 0 * 7
            n_idx_1 = batch_start // 40 + 1 * 7
            n_idx_2 = batch_start // 40 + 2 * 7
            n_idx_3 = batch_start // 40 + 3 * 7
            
            offset_n_0 = n_idx_0 * BLOCK_N
            offset_n_1 = n_idx_1 * BLOCK_N
            offset_n_2 = n_idx_2 * BLOCK_N
            offset_n_3 = n_idx_3 * BLOCK_N
            
            offs_n_0 = offset_n_0 + tl.arange(0, BLOCK_N)
            offs_n_1 = offset_n_1 + tl.arange(0, BLOCK_N)
            offs_n_2 = offset_n_2 + tl.arange(0, BLOCK_N)
            offs_n_3 = offset_n_3 + tl.arange(0, BLOCK_N)
            
            acc_0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            acc_1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            acc_2 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            acc_3 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            
            for k_step in range(40):
                if k_step == 0:
                    acc_0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
                    acc_1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
                    acc_2 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
                    acc_3 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
                
                k_off_0 = k_step * 64
                k_off_1 = k_off_0 + 64
                
                a_0 = A_desc.load([offset_m, k_off_0])
                a_1 = A_desc.load([offset_m, k_off_1])
                
                b_0_0 = B_desc.load([offset_n_0, k_off_0])
                b_0_1 = B_desc.load([offset_n_0, k_off_1])
                acc_0 = tl.dot(a_0, b_0_0.T, acc_0)
                acc_0 = tl.dot(a_1, b_0_1.T, acc_0)
                
                b_1_0 = B_desc.load([offset_n_1, k_off_0])
                b_1_1 = B_desc.load([offset_n_1, k_off_1])
                acc_1 = tl.dot(a_0, b_1_0.T, acc_1)
                acc_1 = tl.dot(a_1, b_1_1.T, acc_1)
                
                b_2_0 = B_desc.load([offset_n_2, k_off_0])
                b_2_1 = B_desc.load([offset_n_2, k_off_1])
                acc_2 = tl.dot(a_0, b_2_0.T, acc_2)
                acc_2 = tl.dot(a_1, b_2_1.T, acc_2)
                
                b_3_0 = B_desc.load([offset_n_3, k_off_0])
                b_3_1 = B_desc.load([offset_n_3, k_off_1])
                acc_3 = tl.dot(a_0, b_3_0.T, acc_3)
                acc_3 = tl.dot(a_1, b_3_1.T, acc_3)
            
            ptrs_c_0 = C_ptr + offs_m[:, None] * stride_c_m + offs_n_0[None, :] * stride_c_n
            mask_c_0 = (offs_m[:, None] < M) & (offs_n_0[None, :] < N)
            tl.store(ptrs_c_0, acc_0.to(tl.bfloat16), mask=mask_c_0)
            
            ptrs_c_1 = C_ptr + offs_m[:, None] * stride_c_m + offs_n_1[None, :] * stride_c_n
            mask_c_1 = (offs_m[:, None] < M) & (offs_n_1[None, :] < N)
            tl.store(ptrs_c_1, acc_1.to(tl.bfloat16), mask=mask_c_1)
            
            ptrs_c_2 = C_ptr + offs_m[:, None] * stride_c_m + offs_n_2[None, :] * stride_c_n
            mask_c_2 = (offs_m[:, None] < M) & (offs_n_2[None, :] < N)
            tl.store(ptrs_c_2, acc_2.to(tl.bfloat16), mask=mask_c_2)
            
            ptrs_c_3 = C_ptr + offs_m[:, None] * stride_c_m + offs_n_3[None, :] * stride_c_n
            mask_c_3 = (offs_m[:, None] < M) & (offs_n_3[None, :] < N)
            tl.store(ptrs_c_3, acc_3.to(tl.bfloat16), mask=mask_c_3)


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    
    A_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    B_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    
    NUM_SM = 132
    persistent_batch_size = triton.cdiv(M, BLOCK_M)
    num_sm = min(NUM_SM, persistent_batch_size)
    grid = num_sm
    
    _gemm_kernel[grid](
        A_desc, B_desc, C,
        M, N, K,
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=3, maxnreg=255,
    )