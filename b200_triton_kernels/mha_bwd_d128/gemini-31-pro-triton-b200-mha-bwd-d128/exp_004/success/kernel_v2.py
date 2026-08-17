import torch
import triton
import triton.language as tl

# Configure the infrastructure allocator mandated by standard Triton for device-created TMA descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def bwd_kernel(
    Q, K, V, O, dO, L, dQ, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr
):
    pid = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // 48
    h = pid_bh % 48

    num_kv_tiles = tl.cdiv(S, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    
    if pid < num_kv_tiles:
        # ---------------------------------------------------------
        # dK, dV owner region
        # ---------------------------------------------------------
        n0 = pid
        offs_n = n0 * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k_desc = tl.make_tensor_descriptor(
            K + b * stride_kb + h * stride_kh,
            shape=[S, HEAD_DIM], strides=[stride_ks, 1],
            block_shape=[BLOCK_N, HEAD_DIM], padding_option="zero"
        )
        v_desc = tl.make_tensor_descriptor(
            V + b * stride_vb + h * stride_vh,
            shape=[S, HEAD_DIM], strides=[stride_vs, 1],
            block_shape=[BLOCK_N, HEAD_DIM], padding_option="zero"
        )
        
        k = k_desc.load([n0 * BLOCK_N, 0])
        v = v_desc.load([n0 * BLOCK_N, 0])
        
        dk = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
        dv = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
        
        q_desc = tl.make_tensor_descriptor(
            Q + b * stride_qb + h * stride_qh,
            shape=[S, HEAD_DIM], strides=[stride_qs, 1],
            block_shape=[BLOCK_M, HEAD_DIM], padding_option="zero"
        )
        do_desc = tl.make_tensor_descriptor(
            dO + b * stride_dob + h * stride_doh,
            shape=[S, HEAD_DIM], strides=[stride_dos, 1],
            block_shape=[BLOCK_M, HEAD_DIM], padding_option="zero"
        )
        
        l_ptrs = L + b * stride_lb + h * stride_lh + tl.arange(0, BLOCK_M) * stride_ls

        # tl.range ensures proper pipelining of the memory accesses
        for m0 in tl.range(0, tl.cdiv(S, BLOCK_M)):
            q = q_desc.load([m0 * BLOCK_M, 0])
            do = do_desc.load([m0 * BLOCK_M, 0])
            
            offs_m = m0 * BLOCK_M + tl.arange(0, BLOCK_M)
            mask_m = offs_m < S
            
            # Load O via pointers directly to registers. This prevents 
            # exceeding Blackwell's 228KiB SMEM limit, allowing larger 128x128 blocks!
            o_ptrs = O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
            o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
            
            l_val = tl.load(l_ptrs + m0 * BLOCK_M * stride_ls, mask=mask_m, other=0.0)
            
            # Recomputing delta is significantly faster than storing/loading from global memory
            # CUDA cores will execute this concurrently with Tensor Cores doing dot operations
            delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
            
            scores_t = tl.dot(k, tl.trans(q), out_dtype=tl.float32) * scale
            scores_t = tl.where(mask_n[:, None] & mask_m[None, :], scores_t, float("-inf"))
            
            p_t = tl.math.exp(scores_t - l_val[None, :])
            
            dv = tl.dot(p_t.to(q.dtype), do, acc=dv)
            
            dp_t = tl.dot(v, tl.trans(do), out_dtype=tl.float32)
            ds_t = p_t * (dp_t - delta[None, :]) * scale
            ds_t = tl.where(mask_n[:, None] & mask_m[None, :], ds_t, 0.0)
            
            dk = tl.dot(ds_t.to(q.dtype), q, acc=dk)
            
        dk_desc = tl.make_tensor_descriptor(
            dK + b * stride_dkb + h * stride_dkh,
            shape=[S, HEAD_DIM], strides=[stride_dks, 1],
            block_shape=[BLOCK_N, HEAD_DIM], padding_option="zero"
        )
        dv_desc = tl.make_tensor_descriptor(
            dV + b * stride_dvb + h * stride_dvh,
            shape=[S, HEAD_DIM], strides=[stride_dvs, 1],
            block_shape=[BLOCK_N, HEAD_DIM], padding_option="zero"
        )
        dk_desc.store([n0 * BLOCK_N, 0], dk.to(dK.dtype.element_ty))
        dv_desc.store([n0 * BLOCK_N, 0], dv.to(dV.dtype.element_ty))
        
    else:
        # ---------------------------------------------------------
        # dQ owner region
        # ---------------------------------------------------------
        m0 = pid - num_kv_tiles
        offs_m = m0 * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q_desc = tl.make_tensor_descriptor(
            Q + b * stride_qb + h * stride_qh,
            shape=[S, HEAD_DIM], strides=[stride_qs, 1],
            block_shape=[BLOCK_M, HEAD_DIM], padding_option="zero"
        )
        do_desc = tl.make_tensor_descriptor(
            dO + b * stride_dob + h * stride_doh,
            shape=[S, HEAD_DIM], strides=[stride_dos, 1],
            block_shape=[BLOCK_M, HEAD_DIM], padding_option="zero"
        )
        
        q = q_desc.load([m0 * BLOCK_M, 0])
        do = do_desc.load([m0 * BLOCK_M, 0])
        
        # Pointer load straight to registers
        o_ptrs = O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        
        l_ptrs = L + b * stride_lb + h * stride_lh + offs_m * stride_ls
        l_val = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        dq = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
        
        k_desc = tl.make_tensor_descriptor(
            K + b * stride_kb + h * stride_kh,
            shape=[S, HEAD_DIM], strides=[stride_ks, 1],
            block_shape=[BLOCK_N, HEAD_DIM], padding_option="zero"
        )
        v_desc = tl.make_tensor_descriptor(
            V + b * stride_vb + h * stride_vh,
            shape=[S, HEAD_DIM], strides=[stride_vs, 1],
            block_shape=[BLOCK_N, HEAD_DIM], padding_option="zero"
        )
        
        for n0 in tl.range(0, tl.cdiv(S, BLOCK_N)):
            k = k_desc.load([n0 * BLOCK_N, 0])
            v = v_desc.load([n0 * BLOCK_N, 0])
            
            offs_n = n0 * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < S
            
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
            scores = tl.where(mask_m[:, None] & mask_n[None, :], scores, float("-inf"))
            
            p = tl.math.exp(scores - l_val[:, None])
            
            dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
            ds = p * (dp - delta[:, None]) * scale
            ds = tl.where(mask_m[:, None] & mask_n[None, :], ds, 0.0)
            
            dq = tl.dot(ds.to(q.dtype), k, acc=dq)
            
        dq_desc = tl.make_tensor_descriptor(
            dQ + b * stride_dqb + h * stride_dqh,
            shape=[S, HEAD_DIM], strides=[stride_dqs, 1],
            block_shape=[BLOCK_M, HEAD_DIM], padding_option="zero"
        )
        dq_desc.store([m0 * BLOCK_M, 0], dq.to(dQ.dtype.element_ty))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes standard SDPA backward for Q, K, V passing gradients destination-style 
    to dQ, dK, dV via Blackwell TMA hardware and single-kernel split ownership partition.
    """
    torch.cuda.set_device(Q.device)
    
    B = 4
    H = 48
    S = Q.shape[2]
    d = 128
    
    scale = 1.0 / (d ** 0.5)
    
    grid = lambda META: (triton.cdiv(S, META["BLOCK_N"]) + triton.cdiv(S, META["BLOCK_M"]), B * H)
    
    bwd_kernel[grid](
        Q, K, V, O, dO, L, dQ, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        S, scale, HEAD_DIM=d
    )