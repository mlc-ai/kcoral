import torch
import triton
import triton.language as tl

# Triton allocator for device-side descriptor infrastructure
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE,
    sm_scale,
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
    # Persistent scheduling calculations
    num_pid_m = tl.cdiv(S, BLOCK_M)
    num_tiles = B * H * num_pid_m

    # Create 4D TMA descriptors for optimal hardware-accelerated memory routing
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
    
    # Persistent grid loop to maximize occupancy and L2 cache locality
    for tile_id in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        batch_head = tile_id // num_pid_m
        start_m = tile_id % num_pid_m
        
        batch_idx = batch_head // H
        head_idx = batch_head % H
        
        # Load query block using TMA and reshape to 2D
        q_4d = q_desc.load([batch_idx, head_idx, start_m * BLOCK_M, 0])
        q = tl.reshape(q_4d, [BLOCK_M, D])
        
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        
        # Causal logic parameters
        max_n = tl.minimum((start_m + 1) * BLOCK_M, S)
        num_full_blocks = tl.minimum(S // BLOCK_N, start_m * BLOCK_M // BLOCK_N)
        end_n_blocks = (max_n + BLOCK_N - 1) // BLOCK_N
        
        # Phase 1: Fully valid blocks (no causal masking overhead)
        for start_n in range(0, num_full_blocks):
            offset_n = start_n * BLOCK_N
            k_4d = k_desc.load([batch_idx, head_idx, offset_n, 0])
            v_4d = v_desc.load([batch_idx, head_idx, offset_n, 0])
            
            k = tl.reshape(k_4d, [BLOCK_N, D])
            v = tl.reshape(v_4d, [BLOCK_N, D])
            
            # WGMMA execution natively targets this representation
            qk = tl.dot(q, k.T)
            qk = qk * sm_scale
            
            # FlashAttention-v2 style scaling
            m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new[:, None])
            
            l_i_new = alpha * l_i + tl.sum(p, axis=1)
            
            acc = acc * alpha[:, None]
            p_bf16 = p.to(tl.bfloat16)
            acc = tl.dot(p_bf16, v, acc=acc)
            
            m_i = m_i_new
            l_i = l_i_new

        # Phase 2: Causal boundary blocks
        for start_n in range(num_full_blocks, end_n_blocks):
            offset_n = start_n * BLOCK_N
            k_4d = k_desc.load([batch_idx, head_idx, offset_n, 0])
            v_4d = v_desc.load([batch_idx, head_idx, offset_n, 0])
            
            k = tl.reshape(k_4d, [BLOCK_N, D])
            v = tl.reshape(v_4d, [BLOCK_N, D])
            
            qk = tl.dot(q, k.T)
            qk = qk * sm_scale
            
            offs_n = offset_n + tl.arange(0, BLOCK_N)
            k_mask = offs_n < S
            
            valid_mask = (offs_m[:, None] >= offs_n[None, :]) & k_mask[None, :]
            # Allow out-of-bound query rows to compute 0 to bypass NaNs securely
            qk = tl.where(valid_mask | (offs_m[:, None] >= S), qk, float("-inf"))
            
            m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new[:, None])
            
            l_i_new = alpha * l_i + tl.sum(p, axis=1)
            
            acc = acc * alpha[:, None]
            p_bf16 = p.to(tl.bfloat16)
            acc = tl.dot(p_bf16, v, acc=acc)
            
            m_i = m_i_new
            l_i = l_i_new

        # Final normalization
        acc = acc / l_i[:, None]
        
        # TMA naturally ignores writes to dynamically out-of-bounds offsets
        out = tl.reshape(acc.to(tl.bfloat16), [1, 1, BLOCK_M, D])
        o_desc.store([batch_idx, head_idx, start_m * BLOCK_M, 0], out)
        
        # Safe explicit masking for normal pointer-based LSE output
        lse = m_i + tl.log(l_i)
        lse_base = LSE + batch_idx * stride_lseb + head_idx * stride_lseh
        lse_ptrs = lse_base + offs_m * stride_lses
        q_mask = offs_m < S
        tl.store(lse_ptrs, lse, mask=q_mask)

def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)
    
    # Cap grid sizes at the SM count to support fully persistent execution layout
    num_sms = torch.cuda.get_device_properties(Q.device).multi_processor_count
    max_estimated_tiles = B * H * triton.cdiv(S, 64)
    grid = (min(num_sms, max_estimated_tiles),)
    
    _attn_fwd_kernel[grid](
        Q, K, V, O, LSE,
        sm_scale,
        B, H, S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        D=D,
    )