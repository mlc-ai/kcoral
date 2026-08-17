import torch
import triton
import triton.language as tl

# Set up Triton allocator for device-side descriptor infrastructure
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE,
    sm_scale_log2,
    B, H, S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    num_pid_m = tl.cdiv(S, BLOCK_M)
    num_tiles = B * H * num_pid_m

    # Create 4D TMA descriptors OUTSIDE the loop.
    # This mitigates device-side allocations completely, running exactly once per SM
    # and ensuring flawless WGMMA backend execution.
    q_desc = tl.make_tensor_descriptor(
        Q, shape=[B, H, S, D], strides=[stride_qb, stride_qh, stride_qs, stride_qd],
        block_shape=[1, 1, BLOCK_M, D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K, shape=[B, H, S, D], strides=[stride_kb, stride_kh, stride_ks, stride_kd],
        block_shape=[1, 1, BLOCK_N, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V, shape=[B, H, S, D], strides=[stride_vb, stride_vh, stride_vs, stride_vd],
        block_shape=[1, 1, BLOCK_N, D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O, shape=[B, H, S, D], strides=[stride_ob, stride_oh, stride_os, stride_od],
        block_shape=[1, 1, BLOCK_M, D]
    )
    
    # Persistent scheduling tightly controls SM grouping logic and naturally 
    # synchronizes contiguous iterations for maximal L2 Cache hits
    for tile_id in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        batch_head = tile_id // num_pid_m
        start_m = tile_id % num_pid_m
        
        batch_idx = batch_head // H
        head_idx = batch_head % H
        
        # Fetch Query via TMA and logically reshape to native 2D layout for WGMMA execution
        q_4d = q_desc.load([batch_idx, head_idx, start_m * BLOCK_M, 0])
        q = tl.reshape(q_4d, [BLOCK_M, D])
        
        # Maintain FP32 Accumulators
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        
        max_n = tl.minimum((start_m + 1) * BLOCK_M, S)
        num_full_blocks = tl.minimum(S // BLOCK_N, start_m * BLOCK_M // BLOCK_N)
        end_n_blocks = (max_n + BLOCK_N - 1) // BLOCK_N
        
        # Phase 1: Rapid loop over fully valid sequence blocks (no causal masking required)
        for start_n in range(0, num_full_blocks):
            offset_n = start_n * BLOCK_N
            k_4d = k_desc.load([batch_idx, head_idx, offset_n, 0])
            v_4d = v_desc.load([batch_idx, head_idx, offset_n, 0])
            
            k = tl.reshape(k_4d, [BLOCK_N, D])
            v = tl.reshape(v_4d, [BLOCK_N, D])
            
            qk = tl.dot(q, k.T)
            qk = qk * sm_scale_log2
            
            # Sub tl.exp for optimized tl.exp2 mapping saving immense instruction overhead natively
            m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp2(m_i - m_i_new)
            p = tl.exp2(qk - m_i_new[:, None])
            
            l_i_new = alpha * l_i + tl.sum(p, axis=1)
            
            acc = acc * alpha[:, None]
            p_bf16 = p.to(tl.bfloat16)
            acc = tl.dot(p_bf16, v, acc=acc)
            
            m_i = m_i_new
            l_i = l_i_new

        # Phase 2: Iterate over constrained causal/sequence-masked boundary blocks
        for start_n in range(num_full_blocks, end_n_blocks):
            offset_n = start_n * BLOCK_N
            k_4d = k_desc.load([batch_idx, head_idx, offset_n, 0])
            v_4d = v_desc.load([batch_idx, head_idx, offset_n, 0])
            
            k = tl.reshape(k_4d, [BLOCK_N, D])
            v = tl.reshape(v_4d, [BLOCK_N, D])
            
            qk = tl.dot(q, k.T)
            qk = qk * sm_scale_log2
            
            offs_n = offset_n + tl.arange(0, BLOCK_N)
            k_mask = offs_n < S
            
            valid_mask = (offs_m[:, None] >= offs_n[None, :]) & k_mask[None, :]
            
            # Keep fundamentally padded out-of-bounds Q computations as 0 explicitly averting NaN collapse
            qk = tl.where(valid_mask | (offs_m[:, None] >= S), qk, float("-inf"))
            
            m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp2(m_i - m_i_new)
            p = tl.exp2(qk - m_i_new[:, None])
            
            l_i_new = alpha * l_i + tl.sum(p, axis=1)
            
            acc = acc * alpha[:, None]
            p_bf16 = p.to(tl.bfloat16)
            acc = tl.dot(p_bf16, v, acc=acc)
            
            m_i = m_i_new
            l_i = l_i_new

        acc = acc / l_i[:, None]
        
        # TMA naturally overrides writes accurately targeting any functionally padded memory offsets
        out = tl.reshape(acc.to(tl.bfloat16), [1, 1, BLOCK_M, D])
        o_desc.store([batch_idx, head_idx, start_m * BLOCK_M, 0], out)
        
        # Map specialized scaled log2 math back to natural log LSE
        ln2 = 0.6931471805599453
        lse = (m_i + tl.log2(l_i)) * ln2
        
        lse_base = LSE + batch_idx * stride_lseb + head_idx * stride_lseh
        lse_ptrs = lse_base + offs_m * stride_lses
        q_mask = offs_m < S
        tl.store(lse_ptrs, lse, mask=q_mask)

def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    sm_scale = 1.0 / (D ** 0.5)
    log2_e = 1.4426950408889634
    sm_scale_log2 = sm_scale * log2_e
    
    num_sms = torch.cuda.get_device_properties(Q.device).multi_processor_count
    
    # Cap total launched programs strictly to exact SM capacities achieving optimal persistent residency
    grid = lambda META: (min(num_sms, B * H * triton.cdiv(S, META["BLOCK_M"])),)
    
    _attn_fwd_kernel[grid](
        Q, K, V, O, LSE,
        sm_scale_log2,
        B, H, S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        D=D,
    )