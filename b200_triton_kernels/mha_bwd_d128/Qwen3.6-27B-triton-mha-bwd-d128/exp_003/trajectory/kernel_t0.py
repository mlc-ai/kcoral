import torch
import triton
import triton.language as tl


@triton.jit
def _dV_kernel(
    dO_ptr, P_ptr, dV_ptr,
    stride_OB, stride_OH, stride_OS, stride_OD,
    stride_PB, stride_PH, stride_PS,
    stride_VB, stride_VH, stride_VS, stride_VD,
    B, H, S, d,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute dV[b,h,:,:] = P[b,h,:,:]^T @ dO[b,h,:,:] for one (b,h) pair."""
    pid = tl.program_id(0)
    total = B * H * tl.cdiv(S, BLOCK_S)
    if pid >= total:
        return
    
    bh_div = pid // tl.cdiv(S, BLOCK_S)
    bh = bh_div
    b = bh // H
    h = bh % H
    
    n_seqs = tl.cdiv(S, BLOCK_S)
    seq_idx = pid % n_seqs
    
    offs_s = seq_idx * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    
    acc = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    
    for i in range(n_seqs):
        i_s = i * BLOCK_S + tl.arange(0, BLOCK_S)
        
        mask_dO = (offs_s[:, None] < S) & (offs_d[None, :] < d)
        mask_P = (i_s[:, None] < S) & (offs_d[None, :] < d)
        
        dO_ptrs = dO_ptr + b * stride_OB + h * stride_OH + offs_s[:, None] * stride_OS + offs_d[None, :] * stride_OD
        P_ptrs = P_ptr + b * stride_PB + h * stride_PH + i_s[:, None] * stride_PS + offs_d[None, :] * stride_PD
        
        dO_tile = tl.load(dO_ptrs, mask=mask_dO, other=0.0)
        P_tile = tl.load(P_ptrs, mask=mask_P, other=0.0)
        
        acc = tl.dot(P_tile, dO_tile, acc)
    
    mask_out = (offs_s[:, None] < S) & (offs_d[None, :] < d)
    out_ptrs = dV_ptr + b * stride_VB + h * stride_VH + offs_s[:, None] * stride_VS + offs_d[None, :] * stride_VD
    tl.store(out_ptrs, acc.to(P_ptr.dtype.element_ty), mask=mask_out)


@triton.jit
def _dQ_dK_kernel(
    Q_ptr, K_ptr, dO_ptr, V_ptr, L_ptr,
    dQ_ptr, dK_ptr,
    stride_QB, stride_QH, stride_QS, stride_QD,
    stride_KB, stride_KH, stride_KS, stride_KD,
    stride_OB, stride_OH, stride_OS, stride_OD,
    stride_VB, stride_VH, stride_VS, stride_VD,
    stride_LB, stride_LH, stride_LS,
    stride_dQB, stride_dQH, stride_dQS, stride_dQD,
    stride_dKB, stride_dKH, stride_dKS, stride_dKD,
    B, H, S, d,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute dQ and dK for one (b,h) pair using tiled approach."""
    pid = tl.program_id(0)
    total_bh = B * H
    total_sd = tl.cdiv(S, BLOCK_S) * tl.cdiv(d, BLOCK_D)
    total = total_bh * total_sd
    
    if pid >= total:
        return
    
    bh = pid // total_sd
    b = bh // H
    h = bh % H
    
    sd_idx = pid % total_sd
    ns = tl.cdiv(S, BLOCK_S)
    nd = tl.cdiv(d, BLOCK_D)
    s_idx = sd_idx // nd
    d_idx = sd_idx % nd
    
    inv_scale = 1.0 / tl.sqrt(float(d))
    
    offs_qs = s_idx * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_dq = d_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    
    mask_qs = offs_qs < S
    mask_dq = offs_dq < d
    mask_full = mask_qs[:, None] & mask_dq[None, :]
    
    Q_ptrs = Q_ptr + b * stride_QB + h * stride_QH + offs_qs[:, None] * stride_QS + tl.arange(0, d)[None, :] * stride_QD
    Q_mask_row = mask_qs[:, None] & (tl.arange(0, d)[None, :] < d)
    
    acc_dQ = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    acc_dK = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    
    for ks in range(ns):
        ks_off = ks * BLOCK_S + tl.arange(0, BLOCK_S)
        mask_ks = ks_off < S
        
        K_ptrs = K_ptr + b * stride_KB + h * stride_KH + ks_off[:, None] * stride_KS + tl.arange(0, d)[None, :] * stride_KD
        K_mask = mask_ks[:, None] & (tl.arange(0, d)[None, :] < d)
        
        Q_tile = tl.load(Q_ptrs, mask=Q_mask_row, other=0.0)
        K_tile = tl.load(K_ptrs, mask=K_mask, other=0.0)
        
        S_mat = tl.dot(Q_tile, K_tile.T) * inv_scale
        
        mask_S = mask_qs[:, None] & mask_ks[None, :]
        S_mat = tl.where(mask_S, S_mat, 0.0)
        
        off_L = b * stride_LB + h * stride_LH + offs_qs * stride_LS
        L_vals = tl.load(L_ptr + off_L, mask=mask_qs, other=0.0)[:, None]
        
        dO_ptrs_t = dO_ptr + b * stride_OB + h * stride_OH + ks_off[:, None] * stride_OS + offs_dq[None, :] * stride_OD
        mask_dO = mask_ks[:, None] & mask_dq[None, :]
        dO_tile = tl.load(dO_ptrs_t, mask=mask_dO, other=0.0)
        
        V_ptrs_t = V_ptr + b * stride_VB + h * stride_VH + ks_off[:, None] * stride_VS + tl.arange(0, d)[None, :] * stride_VD
        mask_V = mask_ks[:, None] & (tl.arange(0, d)[None, :] < d)
        V_tile = tl.load(V_ptrs_t, mask=mask_V, other=0.0)
        
        dOV = tl.dot(dO_tile, V_tile.T)
        mask_dOV = mask_qs[:, None] & mask_ks[None, :]
        dOV = tl.where(mask_dOV, dOV, 0.0)
        
        P = tl.exp(S_mat - L_vals)
        
        dOV_colsum = tl.sum(dOV, axis=1, keep_dims=True)
        
        dS_mat = P * (dOV - dOV_colsum)
        
        dS_mat = tl.where(mask_qs[:, None] & mask_ks[None, :], dS_mat, 0.0)
        
        dQ_add = tl.dot(dS_mat, K_tile) * inv_scale
        acc_dQ += dQ_add
        
        dK_add = tl.dot(dS_mat.T, Q_tile) * inv_scale
        acc_dK += dK_add
    
    acc_dQ = acc_dQ.to(Q_ptr.dtype.element_ty)
    acc_dK = acc_dK.to(K_ptr.dtype.element_ty)
    
    dQ_out = dQ_ptr + b * stride_dQB + h * stride_dQH + offs_qs[:, None] * stride_dQS + offs_dq[None, :] * stride_dQD
    tl.store(dQ_out, acc_dQ, mask=mask_full)
    
    dK_out = dK_ptr + b * stride_dKB + h * stride_dKH + offs_qs[:, None] * stride_dKS + offs_dq[None, :] * stride_dKD
    tl.store(dK_out, acc_dK, mask=mask_full)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Multi-head attention backward pass.
    Computes dQ, dK, dV given Q, K, V, O, dO, and logsumexp L.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    
    if L.dim() == 3:
        L_expanded = L.unsqueeze(-1)
    else:
        L_expanded = L
    
    inv_scale = 1.0 / math.sqrt(d)
    
    BLOCK_S = 64
    BLOCK_D = 128
    
    stride_OB, stride_OH, stride_OS, stride_OD = O.stride()
    stride_QB, stride_QH, stride_QS, stride_QD = Q.stride()
    stride_KB, stride_KH, stride_KS, stride_KD = K.stride()
    stride_VB, stride_VH, stride_VS, stride_VD = V.stride()
    stride_dOB, stride_dOH, stride_dOS, stride_dOD = dO.stride()
    stride_LB, stride_LH, stride_LS = L.stride()
    stride_dQB, stride_dQH, stride_dQS, stride_dQD = dQ.stride()
    stride_dKB, stride_dKH, stride_dKS, stride_dKD = dK.stride()
    stride_dVB, stride_dVH, stride_dVS, stride_dVD = dV.stride()
    
    grid_dV = (B * H * triton.cdiv(S, BLOCK_S),)
    
    @triton.heuristics(values={"BLOCK_S": 64, "BLOCK_D": 128})
    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=4, num_stages=2),
            triton.Config({}, num_warps=8, num_stages=3),
        ],
        key=["S", "d"],
    )
    @triton.jit
    def _dV_impl(
        dO_ptr, P_logsumexp_ptr, dV_ptr,
        stride_dOB, stride_dOH, stride_dOS, stride_dOD,
        stride_LB, stride_LH, stride_LS,
        stride_dVB, stride_dVH, stride_dVS, stride_dVD,
        stride_QB, stride_QH, stride_QS, stride_QD,
        stride_KB, stride_KH, stride_KS, stride_KD,
        B, H, S, d,
        BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        total = B * H * tl.cdiv(S, BLOCK_S)
        if pid >= total:
            return
        
        bh_div = pid // tl.cdiv(S, BLOCK_S)
        bh = bh_div
        b = bh // H
        h = bh % H
        
        n_seqs = tl.cdiv(S, BLOCK_S)
        seq_idx = pid % n_seqs
        
        offs_vs = seq_idx * BLOCK_S + tl.arange(0, BLOCK_S)
        offs_d = tl.arange(0, BLOCK_D)
        
        inv_scale = 1.0 / tl.sqrt(float(d))
        
        acc = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
        
        for qs in range(n_seqs):
            qs_off = qs * BLOCK_S + tl.arange(0, BLOCK_S)
            
            Q_ptrs = Q_ptr + b * stride_QB + h * stride_QH + qs_off[:, None] * stride_QS + tl.arange(0, d)[None, :] * stride_QD
            Q_mask = (qs_off[:, None] < S) & (tl.arange(0, d)[None, :] < d)
            Q_tile = tl.load(Q_ptrs, mask=Q_mask, other=0.0)
            
            K_ptrs = K_ptr + b * stride_KB + h * stride_KH + offs_vs[:, None] * stride_KS + tl.arange(0, d)[None, :] * stride_KD
            K_mask = (offs_vs[:, None] < S) & (tl.arange(0, d)[None, :] < d)
            K_tile = tl.load(K_ptrs, mask=K_mask, other=0.0)
            
            S_mat = tl.dot(Q_tile, K_tile.T) * inv_scale
            
            L_ptrs = L_ptr + b * stride_LB + h * stride_LH + qs_off * stride_LS
            L_vals = tl.load(L_ptrs, mask=qs_off < S, other=0.0)[:, None]
            
            P_mat = tl.exp(S_mat - L_vals)
            
            dO_ptrs = dO_ptr + b * stride_dOB + h * stride_dOH + qs_off[:, None] * stride_dOS + offs_d[None, :] * stride_dOD
            dO_mask = (qs_off[:, None] < S) & (offs_d[None, :] < d)
            dO_tile = tl.load(dO_ptrs, mask=dO_mask, other=0.0)
            
            acc = tl.dot(P_mat, dO_tile, acc)
        
        out_mask = (offs_vs[:, None] < S) & (offs_d[None, :] < d)
        out_ptrs = dV_ptr + b * stride_dVB + h * stride_dVH + offs_vs[:, None] * stride_dVS + offs_d[None, :] * stride_dVD
        tl.store(out_ptrs, acc.to(Q_ptr.dtype.element_ty), mask=out_mask)
    
    _dV_impl[grid_dV](
        dO, L, dV,
        stride_dOB, stride_dOH, stride_dOS, stride_dOD,
        stride_LB, stride_LH, stride_LS,
        stride_dVB, stride_dVH, stride_dVS, stride_dVD,
        stride_QB, stride_QH, stride_QS, stride_QD,
        stride_KB, stride_KH, stride_KS, stride_KD,
        B, H, S, d,
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=3,
    )
    
    grid_dQdK = (B * H * triton.cdiv(S, BLOCK_S) * triton.cdiv(d, BLOCK_D),)
    
    _dQ_dK_kernel[grid_dQdK](
        Q, K, dO, V, L,
        dQ, dK,
        stride_QB, stride_QH, stride_QS, stride_QD,
        stride_KB, stride_KH, stride_KS, stride_KD,
        stride_dOB, stride_dOH, stride_dOS, stride_dOD,
        stride_VB, stride_VH, stride_VS, stride_VD,
        stride_LB, stride_LH, stride_LS,
        stride_dQB, stride_dQH, stride_dQS, stride_dQD,
        stride_dKB, stride_dKH, stride_dKS, stride_dKD,
        B, H, S, d,
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )


if __name__ == "__main__":
    import math
    print("MHA backward kernel implemented")