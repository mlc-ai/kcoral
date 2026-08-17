import math
import torch
import triton
import triton.language as tl

# Standard infrastructure allocator solely for device-created descriptor storage directives
def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M_DQ': 128, 'BLOCK_N_DQ': 128, 'BLOCK_M_DKDV': 128, 'BLOCK_N_DKDV': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M_DQ': 128, 'BLOCK_N_DQ': 64,  'BLOCK_M_DKDV': 64,  'BLOCK_N_DKDV': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M_DQ': 64,  'BLOCK_N_DQ': 128, 'BLOCK_M_DKDV': 128, 'BLOCK_N_DKDV': 64},  num_warps=8, num_stages=4),
    ],
    key=['S']
)
@triton.jit
def _bwd_kernel(
    Q, K, V, O, dO, LSE, dQ, dK, dV,
    stride_qb, stride_qh, stride_qs: tl.constexpr, stride_qd: tl.constexpr,
    stride_kb, stride_kh, stride_ks: tl.constexpr, stride_kd: tl.constexpr,
    stride_vb, stride_vh, stride_vs: tl.constexpr, stride_vd: tl.constexpr,
    stride_ob, stride_oh, stride_os: tl.constexpr, stride_od: tl.constexpr,
    stride_dob, stride_doh, stride_dos: tl.constexpr, stride_dod: tl.constexpr,
    stride_dqb, stride_dqh, stride_dqs: tl.constexpr, stride_dqd: tl.constexpr,
    stride_dkb, stride_dkh, stride_dks: tl.constexpr, stride_dkd: tl.constexpr,
    stride_dvb, stride_dvh, stride_dvs: tl.constexpr, stride_dvd: tl.constexpr,
    stride_lseb, stride_lseh, stride_lses: tl.constexpr,
    S: tl.constexpr, softmax_scale,
    BLOCK_M_DQ: tl.constexpr, BLOCK_N_DQ: tl.constexpr,
    BLOCK_M_DKDV: tl.constexpr, BLOCK_N_DKDV: tl.constexpr,
    d: tl.constexpr
):
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    num_kv_tiles = tl.cdiv(S, BLOCK_N_DKDV)
    num_q_tiles = tl.cdiv(S, BLOCK_M_DQ)
    
    off_q = pid_b * stride_qb + pid_h * stride_qh
    off_k = pid_b * stride_kb + pid_h * stride_kh
    off_v = pid_b * stride_vb + pid_h * stride_vh
    off_o = pid_b * stride_ob + pid_h * stride_oh
    off_do = pid_b * stride_dob + pid_h * stride_doh
    off_dq = pid_b * stride_dqb + pid_h * stride_dqh
    off_dk = pid_b * stride_dkb + pid_h * stride_dkh
    off_dv = pid_b * stride_dvb + pid_h * stride_dvh
    off_lse = pid_b * stride_lseb + pid_h * stride_lseh
    
    # -------------------------------------------------------------
    # Region 1: dK / dV Owners (exclusively mapped)
    # -------------------------------------------------------------
    if pid < num_kv_tiles:
        kv_tile = pid
        
        # TMA Device Descriptors fully resolving pipelining layout natively 
        k_desc = tl.make_tensor_descriptor(
            K + off_k, shape=(S, d), strides=(stride_ks, stride_kd),
            block_shape=(BLOCK_N_DKDV, d), padding_option="zero"
        )
        v_desc = tl.make_tensor_descriptor(
            V + off_v, shape=(S, d), strides=(stride_vs, stride_vd),
            block_shape=(BLOCK_N_DKDV, d), padding_option="zero"
        )
        dk_desc = tl.make_tensor_descriptor(
            dK + off_dk, shape=(S, d), strides=(stride_dks, stride_dkd),
            block_shape=(BLOCK_N_DKDV, d), padding_option="zero"
        )
        dv_desc = tl.make_tensor_descriptor(
            dV + off_dv, shape=(S, d), strides=(stride_dvs, stride_dvd),
            block_shape=(BLOCK_N_DKDV, d), padding_option="zero"
        )
        
        k = tl.load(k_desc, [kv_tile * BLOCK_N_DKDV, 0])
        v = tl.load(v_desc, [kv_tile * BLOCK_N_DKDV, 0])
        
        dk = tl.zeros((BLOCK_N_DKDV, d), tl.float32)
        dv = tl.zeros((BLOCK_N_DKDV, d), tl.float32)
        
        q_desc = tl.make_tensor_descriptor(
            Q + off_q, shape=(S, d), strides=(stride_qs, stride_qd),
            block_shape=(BLOCK_M_DKDV, d), padding_option="zero"
        )
        o_desc = tl.make_tensor_descriptor(
            O + off_o, shape=(S, d), strides=(stride_os, stride_od),
            block_shape=(BLOCK_M_DKDV, d), padding_option="zero"
        )
        do_desc = tl.make_tensor_descriptor(
            dO + off_do, shape=(S, d), strides=(stride_dos, stride_dod),
            block_shape=(BLOCK_M_DKDV, d), padding_option="zero"
        )
        
        offs_n = kv_tile * BLOCK_N_DKDV + tl.arange(0, BLOCK_N_DKDV)
        
        for q_tile in range(tl.cdiv(S, BLOCK_M_DKDV)):
            q = tl.load(q_desc, [q_tile * BLOCK_M_DKDV, 0])
            o = tl.load(o_desc, [q_tile * BLOCK_M_DKDV, 0])
            do = tl.load(do_desc, [q_tile * BLOCK_M_DKDV, 0])
            
            offs_m = q_tile * BLOCK_M_DKDV + tl.arange(0, BLOCK_M_DKDV)
            lse = tl.load(LSE + off_lse + offs_m * stride_lses, mask=offs_m < S, other=0.0)
            
            delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
            
            qk_t = tl.dot(k, q.T, out_dtype=tl.float32) * softmax_scale
            p_t = tl.math.exp(qk_t - lse[None, :])
            
            mask_nm = (offs_n[:, None] < S) & (offs_m[None, :] < S)
            p_t = tl.where(mask_nm, p_t, 0.0)
            
            dv += tl.dot(p_t.to(q.dtype), do, out_dtype=tl.float32)
            
            dp_t = tl.dot(v, do.T, out_dtype=tl.float32)
            ds_t = p_t * (dp_t - delta[None, :]) * softmax_scale
            
            dk += tl.dot(ds_t.to(q.dtype), q, out_dtype=tl.float32)
            
        tl.store(dk_desc, [kv_tile * BLOCK_N_DKDV, 0], dk.to(k.dtype))
        tl.store(dv_desc, [kv_tile * BLOCK_N_DKDV, 0], dv.to(v.dtype))
        
    # -------------------------------------------------------------
    # Region 2: dQ Owners (exclusively mapped)
    # -------------------------------------------------------------
    else:
        q_tile = pid - num_kv_tiles
        
        q_desc = tl.make_tensor_descriptor(
            Q + off_q, shape=(S, d), strides=(stride_qs, stride_qd),
            block_shape=(BLOCK_M_DQ, d), padding_option="zero"
        )
        o_desc = tl.make_tensor_descriptor(
            O + off_o, shape=(S, d), strides=(stride_os, stride_od),
            block_shape=(BLOCK_M_DQ, d), padding_option="zero"
        )
        do_desc = tl.make_tensor_descriptor(
            dO + off_do, shape=(S, d), strides=(stride_dos, stride_dod),
            block_shape=(BLOCK_M_DQ, d), padding_option="zero"
        )
        dq_desc = tl.make_tensor_descriptor(
            dQ + off_dq, shape=(S, d), strides=(stride_dqs, stride_dqd),
            block_shape=(BLOCK_M_DQ, d), padding_option="zero"
        )
        
        q = tl.load(q_desc, [q_tile * BLOCK_M_DQ, 0])
        o = tl.load(o_desc, [q_tile * BLOCK_M_DQ, 0])
        do = tl.load(do_desc, [q_tile * BLOCK_M_DQ, 0])
        
        offs_m = q_tile * BLOCK_M_DQ + tl.arange(0, BLOCK_M_DQ)
        lse = tl.load(LSE + off_lse + offs_m * stride_lses, mask=offs_m < S, other=0.0)
        
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        dq = tl.zeros((BLOCK_M_DQ, d), tl.float32)
        
        k_desc = tl.make_tensor_descriptor(
            K + off_k, shape=(S, d), strides=(stride_ks, stride_kd),
            block_shape=(BLOCK_N_DQ, d), padding_option="zero"
        )
        v_desc = tl.make_tensor_descriptor(
            V + off_v, shape=(S, d), strides=(stride_vs, stride_vd),
            block_shape=(BLOCK_N_DQ, d), padding_option="zero"
        )
        
        for kv_tile in range(tl.cdiv(S, BLOCK_N_DQ)):
            k = tl.load(k_desc, [kv_tile * BLOCK_N_DQ, 0])
            v = tl.load(v_desc, [kv_tile * BLOCK_N_DQ, 0])
            
            qk = tl.dot(q, k.T, out_dtype=tl.float32) * softmax_scale
            p = tl.math.exp(qk - lse[:, None])
            
            offs_n = kv_tile * BLOCK_N_DQ + tl.arange(0, BLOCK_N_DQ)
            mask_mn = (offs_m[:, None] < S) & (offs_n[None, :] < S)
            p = tl.where(mask_mn, p, 0.0)
            
            dp = tl.dot(do, v.T, out_dtype=tl.float32)
            ds = p * (dp - delta[:, None]) * softmax_scale
            
            dq += tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32)
            
        tl.store(dq_desc, [q_tile * BLOCK_M_DQ, 0], dq.to(q.dtype))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes SDPA backward updating precisely defined destination tensors seamlessly targeting FA TMA structures.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    softmax_scale = 1.0 / math.sqrt(d)
    
    def grid_fn(META):
        num_kv = triton.cdiv(S, META['BLOCK_N_DKDV'])
        num_q = triton.cdiv(S, META['BLOCK_M_DQ'])
        return (num_kv + num_q, B, H)
        
    _bwd_kernel[grid_fn](
        Q=Q, K=K, V=V, O=O, dO=dO, LSE=L, dQ=dQ, dK=dK, dV=dV,
        stride_qb=Q.stride(0), stride_qh=Q.stride(1), stride_qs=Q.stride(2), stride_qd=Q.stride(3),
        stride_kb=K.stride(0), stride_kh=K.stride(1), stride_ks=K.stride(2), stride_kd=K.stride(3),
        stride_vb=V.stride(0), stride_vh=V.stride(1), stride_vs=V.stride(2), stride_vd=V.stride(3),
        stride_ob=O.stride(0), stride_oh=O.stride(1), stride_os=O.stride(2), stride_od=O.stride(3),
        stride_dob=dO.stride(0), stride_doh=dO.stride(1), stride_dos=dO.stride(2), stride_dod=dO.stride(3),
        stride_dqb=dQ.stride(0), stride_dqh=dQ.stride(1), stride_dqs=dQ.stride(2), stride_dqd=dQ.stride(3),
        stride_dkb=dK.stride(0), stride_dkh=dK.stride(1), stride_dks=dK.stride(2), stride_dkd=dK.stride(3),
        stride_dvb=dV.stride(0), stride_dvh=dV.stride(1), stride_dvs=dV.stride(2), stride_dvd=dV.stride(3),
        stride_lseb=L.stride(0), stride_lseh=L.stride(1), stride_lses=L.stride(2),
        S=S, softmax_scale=softmax_scale, d=d
    )