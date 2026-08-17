import torch
import triton
import triton.language as tl

# Configure the allocator for device-created TMA descriptors natively supported on SM90+
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # Safe baselines with ordinary software-pipelining 
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'WARP_SPECIALIZE': False}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'WARP_SPECIALIZE': False}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'WARP_SPECIALIZE': False}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'WARP_SPECIALIZE': False}, num_warps=8, num_stages=4),
        
        # Warp specialized variants triggering pingpong scheduling 
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'WARP_SPECIALIZE': True}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'WARP_SPECIALIZE': True}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'WARP_SPECIALIZE': True}, num_warps=8, num_stages=4),
    ],
    key=['S']
)
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale_log2, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr
):
    start_pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(S, BLOCK_M)
    num_tiles = B * H * num_pid_m
    
    # Hopper persistent loop maximizing L2 Cache reuse 
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE,
    ):
        bh = tile_id // num_pid_m
        pid_m = tile_id % num_pid_m
        b = bh // H
        h = bh % H
        
        q_ptr = Q + b * stride_qb + h * stride_qh
        k_ptr = K + b * stride_kb + h * stride_kh
        v_ptr = V + b * stride_vb + h * stride_vh
        o_ptr = O + b * stride_ob + h * stride_oh
        lse_ptr = LSE + b * stride_lseb + h * stride_lseh
        
        # Fast device-side TMA descriptors allocation bypassing 4D stride limits
        q_desc = tl.make_tensor_descriptor(
            q_ptr, shape=[S, BLOCK_D], strides=[stride_qs, stride_qd], 
            block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
        )
        k_desc = tl.make_tensor_descriptor(
            k_ptr, shape=[S, BLOCK_D], strides=[stride_ks, stride_kd], 
            block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
        )
        v_desc = tl.make_tensor_descriptor(
            v_ptr, shape=[S, BLOCK_D], strides=[stride_vs, stride_vd], 
            block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
        )
        o_desc = tl.make_tensor_descriptor(
            o_ptr, shape=[S, BLOCK_D], strides=[stride_os, stride_od], 
            block_shape=[BLOCK_M, BLOCK_D]
        )
        
        offset_m = pid_m * BLOCK_M
        q = q_desc.load([offset_m, 0])
        
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        
        num_steps = tl.cdiv(S, BLOCK_N)
        
        for i in range(num_steps):
            offset_n = i * BLOCK_N
            k = k_desc.load([offset_n, 0])
            v = v_desc.load([offset_n, 0])
            
            # WGMMA paths preserved natively. Q (SMEM) @ K.T (SMEM) guarantees optimum SS WGMMA hardware execution
            qk = tl.dot(q, k.T)
            qk = qk * sm_scale_log2
            
            # Fast out-of-bounds safety cleanly handling tail blocks dynamically
            if S % BLOCK_N != 0:
                if i == num_steps - 1:
                    offs_n = offset_n + tl.arange(0, BLOCK_N)
                    qk = tl.where(offs_n[None, :] < S, qk, float("-inf"))
                    
            m_ij = tl.max(qk, 1)
            m_i_new = tl.maximum(m_i, m_ij)
            
            # SFU MUFU.EX2 operations mapping intrinsically avoiding high latency FP32 conversions
            alpha = tl.exp2(m_i - m_i_new)
            p = tl.exp2(qk - m_i_new[:, None])
            
            l_i_new = alpha * l_i + tl.sum(p, 1)
            
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.bfloat16), v, acc)
            
            m_i = m_i_new
            l_i = l_i_new

        offs_m = offset_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        l_i_safe = tl.where(mask_m, l_i, 1.0)
        
        # Eliminates O(BLOCK_M * D) division translating to localized vector reciprocal scaling 
        rcp_l = 1.0 / l_i_safe
        acc = acc * rcp_l[:, None]
        
        # Revert log2 scaling factor mapping exactly to standard mathematical bounds 
        lse = m_i * 0.6931471805599453 + tl.log(l_i_safe)
        
        # O bounds cleanly clipped
        o_desc.store([offset_m, 0], acc.to(tl.bfloat16))
        
        lse_offs = offs_m * stride_lses
        tl.store(lse_ptr + lse_offs, lse, mask=mask_m)

def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Exponent folds efficiently converting log rules upfront 
    sm_scale_log2 = (1.0 / (D ** 0.5)) * 1.4426950408889634
    NUM_SMS = torch.cuda.get_device_properties(Q.device).multi_processor_count
    
    # 1D Persistent launch capped inherently avoiding CTA overhead limits 
    grid = (NUM_SMS,)
    
    _fwd_kernel[grid](
        Q, K, V, sm_scale_log2, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        NUM_SMS=NUM_SMS,
        BLOCK_D=D
    )