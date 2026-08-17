import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def _fwd_kernel_tma(
    q_desc, k_desc, v_desc, o_desc,
    sm_scale_log2, LSE,
    stride_lsez, stride_lseh, stride_lsem,
    B, H, N_CTX,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid = tl.program_id(0)
    num_m_tiles = tl.cdiv(N_CTX, BLOCK_M)
    num_tiles = B * H * num_m_tiles
    
    # Cyclic assignment ensures multiple SMs process the same head concurrently,
    # leading to perfect L2 cache broadcast for the K and V matrices.
    for tile_id in range(pid, num_tiles, NUM_SMS):
        off_hz = tile_id // num_m_tiles
        start_m = tile_id % num_m_tiles
        
        off_z = off_hz // H
        off_h = off_hz % H
        
        # Load Q via TMA
        q = q_desc.load([off_z, off_h, start_m * BLOCK_M, 0])
        q = tl.reshape(q, [BLOCK_M, BLOCK_DMODEL])
        # Premultiply by scale * log2(e) for base-2 exp speedup
        q = (q * sm_scale_log2).to(q.dtype.element_ty)
        
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        
        limit = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M)
        unmasked_steps = tl.minimum(N_CTX, start_m * BLOCK_M) // BLOCK_N
        
        # Unmasked loop: high-throughput WGMMA sequence
        for step in range(0, unmasked_steps):
            start_n = step * BLOCK_N
            k = k_desc.load([off_z, off_h, start_n, 0])
            k = tl.reshape(k, [BLOCK_N, BLOCK_DMODEL])
            v = v_desc.load([off_z, off_h, start_n, 0])
            v = tl.reshape(v, [BLOCK_N, BLOCK_DMODEL])
            
            # WGMMA native transpose layout for Hopper
            qk = tl.dot(q, k.T)
            
            m_i_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp2(m_i - m_i_new)
            p = tl.exp2(qk - m_i_new[:, None])
            
            l_i = l_i * alpha + tl.sum(p, 1)
            p_cast = p.to(v.dtype.element_ty)
            
            acc = acc * alpha[:, None]
            acc = tl.dot(p_cast, v, acc)
            
            m_i = m_i_new
            
        # Masked loop: causal boundary processing
        total_steps = tl.cdiv(limit, BLOCK_N)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n_base = tl.arange(0, BLOCK_N)
        
        for step in range(unmasked_steps, total_steps):
            start_n = step * BLOCK_N
            k = k_desc.load([off_z, off_h, start_n, 0])
            k = tl.reshape(k, [BLOCK_N, BLOCK_DMODEL])
            v = v_desc.load([off_z, off_h, start_n, 0])
            v = tl.reshape(v, [BLOCK_N, BLOCK_DMODEL])
            
            qk = tl.dot(q, k.T)
            
            offs_n = start_n + offs_n_base
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            seq_mask = offs_n[None, :] < N_CTX
            
            qk = tl.where(causal_mask & seq_mask, qk, float("-inf"))
            
            m_i_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp2(m_i - m_i_new)
            p = tl.exp2(qk - m_i_new[:, None])
            
            l_i = l_i * alpha + tl.sum(p, 1)
            p_cast = p.to(v.dtype.element_ty)
            
            acc = acc * alpha[:, None]
            acc = tl.dot(p_cast, v, acc)
            
            m_i = m_i_new
            
        # Epilogue: Resolve output and calculate matching fp32 analytical LSE
        ln2 = 0.6931471805599453
        l_i_log2 = tl.log2(l_i)
        lse = (m_i + l_i_log2) * ln2
        
        out = acc / l_i[:, None]
        out_4d = tl.reshape(out.to(q.dtype.element_ty), [1, 1, BLOCK_M, BLOCK_DMODEL])
        
        # TMA store guarantees bounds checking natively
        o_desc.store([off_z, off_h, start_m * BLOCK_M, 0], out_4d)
        
        lse_ptrs = LSE + (off_z * stride_lsez + off_h * stride_lseh) + offs_m * stride_lsem
        tl.store(lse_ptrs, lse, mask=offs_m < N_CTX)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    if S == 0:
        return
        
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = 128
    
    # Pre-create Host TensorDescriptors to map safely over the whole input block bounds
    q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, BLOCK_DMODEL])
    k_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, BLOCK_DMODEL])
    v_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, BLOCK_DMODEL])
    o_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, BLOCK_DMODEL])
    
    # sm_scale integrated directly against log2(e) precomputation overhead
    sm_scale = 1.0 / (D ** 0.5)
    sm_scale_log2 = sm_scale * 1.4426950408889634
    
    num_sms = torch.cuda.get_device_properties(Q.device).multi_processor_count
    num_m_tiles = triton.cdiv(S, BLOCK_M)
    num_tiles = B * H * num_m_tiles
    
    # Optimal persistent grid scaled exactly against device hardware limit
    grid = (min(num_sms, num_tiles), )
    
    _fwd_kernel_tma[grid](
        q_desc, k_desc, v_desc, o_desc,
        sm_scale_log2, LSE,
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        NUM_SMS=grid[0],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=8, num_stages=3
    )