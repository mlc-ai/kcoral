import math
import torch
import triton
import triton.language as tl

# Set Triton allocator for device-created infrastructure TMA descriptors
def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_alloc_fn)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ],
    key=['S']
)
@triton.jit
def _causal_fwd_kernel(
    Q, K, V, O, LSE,
    sm_scale_log2, S, B, H,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    # Create 4D Tensor Descriptors globally across the entire CTA once (Hopper natively supports up to 5D)
    # This fully amortizes the setup overhead and safely handles boundary out-of-bounds padding
    q_desc = tl.make_tensor_descriptor(
        Q, shape=[B, H, S, BLOCK_DMODEL], strides=[stride_qb, stride_qh, stride_qs, stride_qd],
        block_shape=[1, 1, BLOCK_M, BLOCK_DMODEL], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K, shape=[B, H, S, BLOCK_DMODEL], strides=[stride_kb, stride_kh, stride_ks, stride_kd],
        block_shape=[1, 1, BLOCK_N, BLOCK_DMODEL], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V, shape=[B, H, S, BLOCK_DMODEL], strides=[stride_vb, stride_vh, stride_vs, stride_vd],
        block_shape=[1, 1, BLOCK_N, BLOCK_DMODEL], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O, shape=[B, H, S, BLOCK_DMODEL], strides=[stride_ob, stride_oh, stride_os, stride_od],
        block_shape=[1, 1, BLOCK_M, BLOCK_DMODEL]
    )
    
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(S, BLOCK_M)
    num_tiles = B * H * num_pid_m
    
    # Persistent Grid Execution targeting hardware capacity mapping
    # Sequential threads naturally chunk identical heads concurrently allowing huge L2 cache multicasting
    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        batch_head = tile_id // num_pid_m
        pid_m = tile_id % num_pid_m
        
        batch_idx = batch_head // H
        head_idx = batch_head % H
        offset_m = pid_m * BLOCK_M
        
        # Out-of-bounds block catching for padding handling 
        if offset_m >= S:
            continue
            
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        
        # Fetch 4D chunks natively mapping to contiguous 2D SMEM registers
        q_4d = q_desc.load([batch_idx, head_idx, offset_m, 0])
        q = tl.reshape(q_4d, [BLOCK_M, BLOCK_DMODEL])
        
        limit_n = offset_m // BLOCK_N
        max_k = tl.minimum(S, offset_m + BLOCK_M)
        num_k_blocks = tl.cdiv(max_k, BLOCK_N)
        
        offs_m = offset_m + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        
        for start_n in range(0, num_k_blocks):
            offset_n = start_n * BLOCK_N
            k_4d = k_desc.load([batch_idx, head_idx, offset_n, 0])
            k = tl.reshape(k_4d, [BLOCK_N, BLOCK_DMODEL])
            
            # WGMMA SS: Q in SMEM, K^T in SMEM natively executing without constraints
            qk = tl.dot(q, k.trans(1, 0), out_dtype=tl.float32)
            qk = qk * sm_scale_log2
            
            # Conditionally apply causal limits strictly bounding diagonal masks
            if start_n >= limit_n:
                offs_n_curr = offset_n + offs_n
                causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
                seq_mask = offs_n_curr[None, :] < S
                mask = causal_mask & seq_mask
                qk = tl.where(mask, qk, float("-inf"))
                
            m_ij = tl.max(qk, 1)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp2(m_i - m_i_new)
            p = tl.exp2(qk - m_i_new[:, None])
            l_i_new = alpha * l_i + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            
            v_4d = v_desc.load([batch_idx, head_idx, offset_n, 0])
            v = tl.reshape(v_4d, [BLOCK_N, BLOCK_DMODEL])
            
            p_cast = p.to(Q.dtype.element_ty)
            
            # WGMMA RS: Softmax scaled weights maintained in Registers, V mapped seamlessly from SMEM
            acc = tl.dot(p_cast, v, acc, out_dtype=tl.float32)
            
            m_i = m_i_new
            l_i = l_i_new

        acc = acc * (1.0 / l_i[:, None])
        acc_4d = tl.reshape(acc, [1, 1, BLOCK_M, BLOCK_DMODEL])
        
        # TMA Engine strictly handles out-of-bounds dropping logically bounding the tensor to S sequences
        o_desc.store([batch_idx, head_idx, offset_m, 0], acc_4d.to(Q.dtype.element_ty))
        
        # Convert internal analytic base-2 representation safely back to exact base `e` expected standard LSE
        lse = m_i * 0.6931471805599453 + tl.log(l_i)
        lse_ptrs = LSE + batch_idx * stride_lseb + head_idx * stride_lseh + offs_m * stride_lses
        tl.store(lse_ptrs, lse, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    if Q.shape[2] == 0:
        return
        
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Scale translation to match exact FP32 execution expectations 
    sm_scale = 1.0 / math.sqrt(D)
    sm_scale_log2 = sm_scale * 1.4426950408889634
    
    # Restrict CTA Launch dynamically to hardware SM count achieving perfect persistent streaming bounds
    NUM_SMS = torch.cuda.get_device_properties(Q.device).multi_processor_count
    grid = (NUM_SMS,)

    _causal_fwd_kernel[grid](
        Q, K, V, O, LSE,
        sm_scale_log2, S, B, H,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        NUM_SMS,
        BLOCK_DMODEL=128
    )