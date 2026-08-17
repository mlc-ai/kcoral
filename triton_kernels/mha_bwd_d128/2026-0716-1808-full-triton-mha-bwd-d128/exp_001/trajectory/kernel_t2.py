import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dq_kernel(
    desc_Q, desc_K, desc_V, desc_dO, desc_dQ,
    L_ptr, dQ_ptr,
    S_seq, scale,
    stride_bh, stride_s, stride_d,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    bh = tl.program_id(2)
    
    i_start = pid_i * 64
    j_start = pid_j * 64
    
    smem_Q = tl.empty((2, 64, 64), dtype=tl.bfloat16, shared=True)
    smem_dO = tl.empty((2, 64, 64), dtype=tl.bfloat16, shared=True)
    smem_K = tl.empty((2, 64, 64), dtype=tl.bfloat16, shared=True)
    smem_V = tl.empty((2, 64, 64), dtype=tl.bfloat16, shared=True)
    
    for chunk in range(2):
        q_tile = desc_Q.load([bh, i_start, chunk * 64])
        do_tile = desc_dO.load([bh, i_start, chunk * 64])
        k_tile = desc_K.load([bh, j_start, chunk * 64])
        v_tile = desc_V.load([bh, j_start, chunk * 64])
        
        tl.store(smem_Q[chunk], q_tile)
        tl.store(smem_dO[chunk], do_tile)
        tl.store(smem_K[chunk], k_tile)
        tl.store(smem_V[chunk], v_tile)
        
    dQ_tile = tl.zeros((64, 128), dtype=tl.float32)
    
    expert_kernel_D_prev = tl.zeros((64, 64), dtype=tl.float32)
    for chunk in range(2):
        expert_kernel_Q = tl.load(smem_Q[chunk]).to(tl.float32)
        expert_kernel_dO = tl.load(smem_dO[chunk]).to(tl.float32)
        expert_kernel_K = tl.load(smem_K[chunk]).to(tl.float32)
        expert_kernel_V = tl.load(smem_V[chunk]).to(tl.float32)
        
        expert_kernel_D = tl.dot(expert_kernel_dO, expert_kernel_V.T)
        expert_kernel_D_prev += expert_kernel_D
        
    Q_full = tl.concat(smem_Q[0].to(tl.float32), smem_Q[1].to(tl.float32), dim=1)
    K_full = tl.concat(smem_K[0].to(tl.float32), smem_K[1].to(tl.float32), dim=1)
    
    S = tl.dot(Q_full, K_full.T)
    
    l_row = tl.load(L_ptr + bh * 4096 + i_start + tl.arange(0, 64), 
                    mask=(i_start + tl.arange(0, 64) < S_seq))
    
    P = tl.exp(S * scale - l_row[:, None])
    
    dQ_tile[:, :64] = (expert_kernel_D_prev * P)
    dQ_tile[:, 64:] = (expert_kernel_D_prev * P)
    
    dQ_tile = (dQ_tile * scale)
    
    mask = ((i_start + tl.arange(0, 64)[:, None]) < S_seq) & ((tl.arange(0, 128)[None, :]) < 128)
    
    c_ptr = dQ_ptr + bh * (i_start * 128) + pid_i * 64 * 128
    tl.store(c_ptr, dQ_tile, mask=mask)


@triton.jit
def _bwd_dkv_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dK, desc_dV,
    L_ptr, dK_ptr, dV_ptr,
    S_seq, scale,
    stride_bh, stride_s, stride_d,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    bh = tl.program_id(2)
    
    i_start = pid_i * 64
    j_start = pid_j * 64
    
    smem_Q = tl.empty((2, 64, 64), dtype=tl.bfloat16, shared=True)
    smem_dO = tl.empty((2, 64, 64), dtype=tl.bfloat16, shared=True)
    smem_O = tl.empty((2, 64, 64), dtype=tl.bfloat16, shared=True)
    smem_K = tl.empty((2, 64, 64), dtype=tl.bfloat16, shared=True)
    smem_V = tl.empty((2, 64, 64), dtype=tl.bfloat16, shared=True)
    
    for chunk in range(2):
        q_tile = desc_Q.load([bh, i_start, chunk * 64])
        do_tile = desc_dO.load([bh, i_start, chunk * 64])
        o_tile = desc_O.load([bh, i_start, chunk * 64])
        k_tile = desc_K.load([bh, j_start, chunk * 64])
        v_tile = desc_V.load([bh, j_start, chunk * 64])
        
        tl.store(smem_Q[chunk], q_tile)
        tl.store(smem_dO[chunk], do_tile)
        tl.store(smem_O[chunk], o_tile)
        tl.store(smem_K[chunk], k_tile)
        tl.store(smem_V[chunk], v_tile)
        
    dK_tile = tl.zeros((64, 128), dtype=tl.float32)
    dV_tile = tl.zeros((64, 128), dtype=tl.float32)
    
    expert_kernel_D_prev = tl.zeros((64, 64), dtype=tl.float32)
    for chunk in range(2):
        expert_kernel_Q = tl.load(smem_Q[chunk]).to(tl.float32)
        expert_kernel_dO = tl.load(smem_dO[chunk]).to(tl.float32)
        expert_kernel_O = tl.load(smem_O[chunk]).to(tl.float32)
        expert_kernel_K = tl.load(smem_K[chunk]).to(tl.float32)
        expert_kernel_V = tl.load(smem_V[chunk]).to(tl.float32)
        
        expert_kernel_D = tl.dot(expert_kernel_dO, expert_kernel_V.T)
        
        # Compute correction term E_i = \sum_d dO_{id} O_{id}
        E = (expert_kernel_dO * expert_kernel_O)
        expert_kernel_D = expert_kernel_D_prev - E[:, None]
        
        expert_kernel_D_prev += expert_kernel_D
        
    Q_full = tl.concat(smem_Q[0].to(tl.float32), smem_Q[1].to(tl.float32), dim=1)
    K_full = tl.concat(smem_K[0].to(tl.float32), smem_K[1].to(tl.float32), dim=1)
    
    S = tl.dot(Q_full, K_full.T)
    
    l_row = tl.load(L_ptr + bh * 4096 + i_start + tl.arange(0, 64), 
                    mask=(i_start + tl.arange(0, 64) < S_seq))
    
    P = tl.exp(S * scale - l_row[:, None])
    
    dK_tile[:, :64] = (expert_kernel_D_prev.T * P.T)
    dK_tile[:, 64:] = (expert_kernel_D_prev.T * P.T)
    
    dK_tile = (dK_tile * scale)
    
    dV_tile[:, :64] = P.T
    dV_tile[:, 64:] = P.T
    
    mask_k = ((j_start + tl.arange(0, 64)[:, None]) < S_seq) & ((tl.arange(0, 128)[None, :]) < 128)
    c_ptr_k = dK_ptr + bh * (j_start * 128) + pid_j * 64 * 128
    tl.store(c_ptr_k, dK_tile, mask=mask_k)
    
    mask_v = ((j_start + tl.arange(0, 64)[:, None]) < S_seq) & ((tl.arange(0, 128)[None, :]) < 128)
    c_ptr_v = dV_ptr + bh * (j_start * 128) + pid_j * 64 * 128
    tl.store(c_ptr_v, dV_tile, mask=mask_v)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward ``dQ, dK, dV`` in place."""
    torch.cuda.set_device(Q.device)
    S_seq = Q.shape[-2]
    scale = 1.0 / tl.sqrt(128.0)
    
    bh_total = Q.shape[0] * Q.shape[1]
    
    desc_Q = TensorDescriptor.from_tensor(Q.reshape(bh_total, S_seq, 128), [64, 64, 64])
    desc_K = TensorDescriptor.from_tensor(K.reshape(bh_total, S_seq, 128), [64, 64, 64])
    desc_V = TensorDescriptor.from_tensor(V.reshape(bh_total, S_seq, 128), [64, 64, 64])
    desc_O = TensorDescriptor.from_tensor(O.reshape(bh_total, S_seq, 128), [64, 64, 64])
    desc_dO = TensorDescriptor.from_tensor(dO.reshape(bh_total, S_seq, 128), [64, 64, 64])
    desc_dQ = TensorDescriptor.from_tensor(dQ.reshape(bh_total, S_seq, 128), [64, 64, 64])
    desc_dK = TensorDescriptor.from_tensor(dK.reshape(bh_total, S_seq, 128), [64, 64, 64])
    desc_dV = TensorDescriptor.from_tensor(dV.reshape(bh_total, S_seq, 128), [64, 64, 64])
    
    grid_dq = (S_seq // 64, S_seq // 64, bh_total)
    _bwd_dq_kernel[grid_dq](
        desc_Q, desc_K, desc_V, desc_dO, desc_dQ,
        L, dQ, S_seq, scale,
        1, 1, 1, 
        num_warps=4,
        num_stages=4,
    )
    
    grid_dkv = (S_seq // 64, S_seq // 64, bh_total)
    _bwd_dkv_kernel[grid_dkv](
        desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dK, desc_dV,
        L, dK, dV, S_seq, scale,
        1, 1, 1,
        num_warps=4,
        num_stages=4,
    )