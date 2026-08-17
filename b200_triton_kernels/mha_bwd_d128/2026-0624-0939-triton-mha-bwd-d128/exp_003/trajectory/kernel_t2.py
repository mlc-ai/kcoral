import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def load_3d(desc, base_ptr, mask_row):
    row_off = tl.arange(0, 64)[:, None]
    col_off = tl.arange(0, 128)[None, :]
    ptr = base_ptr + row_off * 128 + col_off * 1
    mask = mask_row[:, None]
    return tl.load(ptr, mask=mask, other=0.0)


@triton.jit
def load_1d(base_ptr, mask):
    off = tl.arange(0, 64)
    return tl.load(base_ptr + off, mask=mask, other=0.0)


@triton.jit
def store_3d(desc, base_ptr, val, mask_row):
    row_off = tl.arange(0, 64)[:, None]
    col_off = tl.arange(0, 128)[None, :]
    ptr = base_ptr + row_off * 128 + col_off * 1
    mask = mask_row[:, None]
    tl.store(ptr, val, mask=mask)


@triton.jit
def _mha_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, L_ptr_float,
    dQ_ptr, dK_ptr, dV_ptr,
    S_len, d, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_b_h = tl.program_id(0)
    pid_m = tl.program_id(1)
    num_pid_m = tl.cdiv(S_len, BLOCK_M)
    num_pid_n = tl.cdiv(S_len, BLOCK_N)

    # Phase 1: Compute dQ
    if pid_m < num_pid_m:
        offset_m = pid_m * BLOCK_M
        mask_r_m = (offset_m + tl.arange(0, BLOCK_M)) < S_len
        
        Q_m_0 = load_3d(None, Q_ptr + pid_b_h * S_len * d + offset_m * d, mask_r_m)
        dO_m_0 = load_3d(None, dO_ptr + pid_b_h * S_len * d + offset_m * d, mask_r_m)
        O_m_0 = load_3d(None, O_ptr + pid_b_h * S_len * d + offset_m * d, mask_r_m)
        
        Q_m_0 = Q_m_0.to(tl.float32)
        dO_m_0 = dO_m_0.to(tl.float32)
        O_m_0 = O_m_0.to(tl.float32)
        
        D_m = tl.sum(dO_m_0 * O_m_0, axis=1)
        L_m = load_1d(L_ptr_float + pid_b_h * S_len + offset_m, mask_r_m)
        
        acc_dQ_0 = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
        
        for n_idx in range(0, S_len, BLOCK_N):
            mask_r_n = (n_idx + tl.arange(0, BLOCK_N)) < S_len
            
            K_n_0 = load_3d(None, K_ptr + pid_b_h * S_len * d + n_idx * d, mask_r_n)
            V_n_0 = load_3d(None, V_ptr + pid_b_h * S_len * d + n_idx * d, mask_r_n)
            
            K_n_0 = K_n_0.to(tl.float32)
            V_n_0 = V_n_0.to(tl.float32)
            
            S = tl.dot(Q_m_0, K_n_0.T) * scale
            P = tl.exp(S - L_m[:, None])
            
            dP = tl.dot(dO_m_0, V_n_0.T)
            dS = P * (dP - D_m[:, None]) * scale
            
            acc_dQ_0 = tl.dot(dS, K_n_0, acc_dQ_0)
        
        dq_out_0 = acc_dQ_0.to(Q_ptr.dtype.element_ty)
        mask_m = (offset_m + tl.arange(0, 64)) < S_len
        store_3d(None, dQ_ptr + pid_b_h * S_len * d + offset_m * d, dq_out_0, mask_m)

    # Phase 2: Compute dK and dV
    pid_n = pid_m
    if pid_n < num_pid_n:
        offset_n = pid_n * BLOCK_N
        mask_r_n = (offset_n + tl.arange(0, BLOCK_N)) < S_len
        
        K_n_0 = load_3d(None, K_ptr + pid_b_h * S_len * d + offset_n * d, mask_r_n)
        V_n_0 = load_3d(None, V_ptr + pid_b_h * S_len * d + offset_n * d, mask_r_n)
        
        K_n_0 = K_n_0.to(tl.float32)
        V_n_0 = V_n_0.to(tl.float32)
        
        acc_dK_0 = tl.zeros((BLOCK_N, 128), dtype=tl.float32)
        acc_dV_0 = tl.zeros((BLOCK_N, 128), dtype=tl.float32)
        
        for m_idx in range(0, S_len, BLOCK_M):
            mask_r_m = (m_idx + tl.arange(0, BLOCK_M)) < S_len
            
            Q_m_0 = load_3d(None, Q_ptr + pid_b_h * S_len * d + m_idx * d, mask_r_m)
            dO_m_0 = load_3d(None, dO_ptr + pid_b_h * S_len * d + m_idx * d, mask_r_m)
            O_m_0 = load_3d(None, O_ptr + pid_b_h * S_len * d + m_idx * d, mask_r_m)
            
            Q_m_0 = Q_m_0.to(tl.float32)
            dO_m_0 = dO_m_0.to(tl.float32)
            O_m_0 = O_m_0.to(tl.float32)
            
            D_m = tl.sum(dO_m_0 * O_m_0, axis=1)
            L_m = load_1d(L_ptr_float + pid_b_h * S_len + m_idx, mask_r_m)
            
            S = tl.dot(Q_m_0, K_n_0.T) * scale
            P = tl.exp(S - L_m[:, None])
            
            dP = tl.dot(dO_m_0, V_n_0.T)
            dS = P * (dP - D_m[:, None]) * scale
            
            acc_dV_0 = tl.dot(P.T, dO_m_0, acc_dV_0)
            acc_dK_0 = tl.dot(dS.T, Q_m_0, acc_dK_0)
        
        dk_out_0 = acc_dK_0.to(K_ptr.dtype.element_ty)
        dv_out_0 = acc_dV_0.to(V_ptr.dtype.element_ty)
        mask_n = (offset_n + tl.arange(0, 64)) < S_len
        store_3d(None, dK_ptr + pid_b_h * S_len * d + offset_n * d, dk_out_0, mask_n)
        store_3d(None, dV_ptr + pid_b_h * S_len * d + offset_n * d, dv_out_0, mask_n)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    L_float = L.float()
    
    BLOCK_M = 64
    Q_desc = TensorDescriptor.from_tensor(Q, [1, BLOCK_M, 128])
    K_desc = TensorDescriptor.from_tensor(K, [1, BLOCK_M, 128])
    V_desc = TensorDescriptor.from_tensor(V, [1, BLOCK_M, 128])
    O_desc = TensorDescriptor.from_tensor(O, [1, BLOCK_M, 128])
    dO_desc = TensorDescriptor.from_tensor(dO, [1, BLOCK_M, 128])
    L_desc = TensorDescriptor.from_tensor(L_float, [1, BLOCK_M])
    
    assert Q.is_contiguous(), "Q must be contiguous for valid TMA strides"
    assert (Q.stride(2) % 16 == 0) and (Q.stride(1) % 16 == 0)
    
    grid = (B * H, triton.cdiv(S, 64))
    _mha_bwd_kernel[grid](
        Q, K, V, O, dO, L, L_float,
        dQ, dK, dV,
        S, d, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=64,
        num_warps=4, num_stages=5,
    )