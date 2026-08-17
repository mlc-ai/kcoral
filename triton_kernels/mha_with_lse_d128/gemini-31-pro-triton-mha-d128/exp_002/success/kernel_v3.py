import torch
import triton
import triton.language as tl
import math

# Configure Triton allocator for device-created tensor descriptors (Hopper TMA)
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=5, num_warps=4),
    ],
    key=['S']
)
@triton.jit
def _attn_fwd_kernel_persistent(
    Q_ptr, K_ptr, V_ptr, sm_scale_log2,
    O_ptr, LSE_ptr,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    EVEN_S: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    num_m = tl.cdiv(S, BLOCK_M)
    num_tiles = num_m * B * H
    
    start_pid = tl.program_id(0)
    
    # Persistent scheduling over all tiles
    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        # Decode tile_id: pid_m varies fastest to group same pid_bh together for optimal L2 cache multicast
        pid_m = tile_id % num_m
        pid_bh = tile_id // num_m
        
        pid_b = pid_bh // H
        pid_h = pid_bh % H
        
        q_offset = pid_b * stride_qb + pid_h * stride_qh
        k_offset = pid_b * stride_kb + pid_h * stride_kh
        v_offset = pid_b * stride_vb + pid_h * stride_vh
        o_offset = pid_b * stride_ob + pid_h * stride_oh
        lse_offset = pid_b * stride_lseb + pid_h * stride_lseh
        
        Q_base = Q_ptr + q_offset
        K_base = K_ptr + k_offset
        V_base = V_ptr + v_offset
        O_base = O_ptr + o_offset
        LSE_base = LSE_ptr + lse_offset
        
        # Create TMA descriptors (statically handles bounding conditions without mask instructions)
        q_desc = tl.make_tensor_descriptor(
            Q_base, shape=[S, D], strides=[stride_qs, 1],
            block_shape=[BLOCK_M, D], padding_option="zero"
        )
        k_desc = tl.make_tensor_descriptor(
            K_base, shape=[S, D], strides=[stride_ks, 1],
            block_shape=[BLOCK_N, D], padding_option="zero"
        )
        v_desc = tl.make_tensor_descriptor(
            V_base, shape=[S, D], strides=[stride_vs, 1],
            block_shape=[BLOCK_N, D], padding_option="zero"
        )
        o_desc = tl.make_tensor_descriptor(
            O_base, shape=[S, D], strides=[stride_os, 1],
            block_shape=[BLOCK_M, D]
        )
        
        start_m = pid_m * BLOCK_M
        
        # Load Q tile and pre-scale dynamically to avoid extra instructions per inner loop
        q = q_desc.load([start_m, 0])
        q = (q * sm_scale_log2).to(tl.bfloat16)
        
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        
        num_n_blocks = tl.cdiv(S, BLOCK_N)
        
        # Diverge completely based on static EVEN_S sequence bounds
        if EVEN_S:
            for n_idx in range(num_n_blocks):
                start_n = n_idx * BLOCK_N
                
                k = k_desc.load([start_n, 0])
                v = v_desc.load([start_n, 0])
                
                # Q @ K.T (Hop Tensor Core FP32 acc)
                qk = tl.dot(q, k.T, out_dtype=tl.float32)
                
                m_ij = tl.max(qk, 1)
                m_i_new = tl.maximum(m_i, m_ij)
                
                # Using exp2 strictly maps to EX2 hardware unit
                alpha = tl.exp2(m_i - m_i_new)
                beta = tl.exp2(qk - m_i_new[:, None])
                
                l_i_new = alpha * l_i + tl.sum(beta, 1)
                
                p = beta.to(tl.bfloat16)
                acc = acc * alpha[:, None]
                acc = tl.dot(p, v, acc, out_dtype=tl.float32)
                
                m_i = m_i_new
                l_i = l_i_new
        else:
            offs_m = start_m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < S
            for n_idx in range(num_n_blocks):
                start_n = n_idx * BLOCK_N
                
                k = k_desc.load([start_n, 0])
                v = v_desc.load([start_n, 0])
                
                qk = tl.dot(q, k.T, out_dtype=tl.float32)
                
                offs_n = start_n + tl.arange(0, BLOCK_N)
                mask_qk = mask_m[:, None] & (offs_n[None, :] < S)
                qk = tl.where(mask_qk, qk, float("-inf"))
                
                m_ij = tl.max(qk, 1)
                m_i_new = tl.maximum(m_i, m_ij)
                m_i_new = tl.where(mask_m, m_i_new, 0.0)
                
                alpha = tl.exp2(m_i - m_i_new)
                beta = tl.exp2(qk - m_i_new[:, None])
                
                l_i_new = alpha * l_i + tl.sum(beta, 1)
                
                p = beta.to(tl.bfloat16)
                acc = acc * alpha[:, None]
                acc = tl.dot(p, v, acc, out_dtype=tl.float32)
                
                m_i = m_i_new
                l_i = l_i_new
                
        # Finalize Output
        acc = acc * (1.0 / l_i[:, None])
        O_val = acc.to(tl.bfloat16)
        
        # TMA automatically handles bounds checking when writing
        o_desc.store([start_m, 0], O_val)
        
        # Standard pointer store for LSE as TMA requires inner contiguous dimensions
        offs_m_store = start_m + tl.arange(0, BLOCK_M)
        lse_ptrs = LSE_base + (offs_m_store * stride_lses)
        lse_val = m_i * 0.6931471805599453 + tl.log(l_i)
        
        if EVEN_S:
            tl.store(lse_ptrs, lse_val)
        else:
            tl.store(lse_ptrs, lse_val, mask=offs_m_store < S)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    if S == 0:
        return
        
    # sm_scale incorporates Log2(E) factor upstream of the hardware EX2 exponentiation
    sm_scale_log2 = (1.0 / math.sqrt(D)) * 1.4426950408889634
    
    # Statically determine if the sequence length guarantees safe alignment without any branching
    even_s = (S % 256 == 0)
    
    # Determine the number of SMs available on the active hardware architecture for persistent workers
    num_sms = torch.cuda.get_device_properties(Q.device).multi_processor_count
    
    # Cap the launch configuration to the active processor count
    num_tiles_max = B * H * triton.cdiv(S, 64)
    grid = (min(num_sms, num_tiles_max),)
    
    _attn_fwd_kernel_persistent[grid](
        Q, K, V, sm_scale_log2,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D=D, EVEN_S=even_s, NUM_SMS=num_sms,
    )