import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Provide Triton's allocator for creating device-side TMA descriptors dynamically
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "LOOP_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "LOOP_STAGES": 4}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "LOOP_STAGES": 2}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "LOOP_STAGES": 4}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "LOOP_STAGES": 4}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "LOOP_STAGES": 4}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _mha_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lseb, stride_lseh, stride_lses,
    S, H, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    start_m = pid_m * BLOCK_M
    
    # Contextual bases mapped strictly to identical alignments 
    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = O + b * stride_ob + h * stride_oh
    
    # Establish fully hardware-backed Tensor Descriptors (TMA capabilities)
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, D], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, D], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, D], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, D], strides=[stride_os, 1],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    
    # Native query block retrieval
    q = q_desc.load([start_m, 0])
    
    # Immediately apply log2 scaling externally to mitigate sequential MUFU serialization later 
    RCP_LN2: tl.constexpr = 1.4426950408889634
    scale_base2 = scale * RCP_LN2
    q = (q.to(tl.float32) * scale_base2).to(tl.bfloat16)
    
    # Initialize online-softmax context states
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    
    # Fast iteration parameters cleanly bypassing loop remainder bounds internally
    num_blocks = S // BLOCK_N
    
    # Direct multi-buffering hardware mapping over fully in-bounds tensor tiles
    for block_idx in tl.range(0, num_blocks, num_stages=LOOP_STAGES):
        start_n = block_idx * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        # Native WGMMA Matrix Multiplication 
        scores = tl.dot(q, k.T)
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # Resolves trailing sequence elements with targeted masking independently 
    if S % BLOCK_N != 0:
        start_n = num_blocks * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        scores = tl.dot(q, k.T)
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid_score = offs_n[None, :] < S
        scores = tl.where(valid_score, scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # Establish globally stable values without conditionals because S > 0 forces valid sequence interactions inherently
    inv_l_i = 1.0 / l_i
    output = acc * inv_l_i[:, None]
    
    # O outputs seamlessly ignoring queries inherently zero-padded by TMA configurations externally
    o_desc.store([start_m, 0], output.to(tl.bfloat16))
    
    # Recover standard Log Sum Exp output definitions mapped to natural base configurations natively
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(l_i)) * LN2
    
    # Safeguard vector stores individually 
    offs_m = start_m + tl.arange(0, BLOCK_M)
    m_mask = offs_m < S
    lse_ptrs = LSE + b * stride_lseb + h * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=m_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = D ** -0.5
    
    # X Grid prioritizes SM proximity logic implicitly caching common K / V sequences locally for equivalent query sets
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    
    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, H, scale,
        D=128
    )