import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor

# Global cache for host-side TensorDescriptors to entirely eliminate Python/C++ 
# descriptor construction overhead during rapid benchmark or training loops.
_desc_cache = {}

def get_desc(tensor, block_shape):
    key = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tuple(block_shape))
    if key not in _desc_cache:
        _desc_cache[key] = TensorDescriptor.from_tensor(tensor, block_shape)
    return _desc_cache[key]

def desc_pre_hook(kwargs):
    # Dynamically constructs & injects required 4D descriptors for whichever configuration the autotuner currently evaluates
    BM = kwargs["BLOCK_M"]
    BN = kwargs["BLOCK_N"]
    BD = kwargs["BLOCK_D"]
    
    kwargs["q_desc"] = get_desc(kwargs["Q"], [1, 1, BM, BD])
    kwargs["k_desc"] = get_desc(kwargs["K"], [1, 1, BN, BD])
    kwargs["v_desc"] = get_desc(kwargs["V"], [1, 1, BN, BD])
    kwargs["o_desc"] = get_desc(kwargs["O"], [1, 1, BM, BD])


_configs = [
    # Top-tier configs pushing Latency-Hiding through maximum Register / TMEM arithmetic intensity
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "PIPE_STAGES": 2}, num_warps=8, num_stages=2, pre_hook=desc_pre_hook),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "PIPE_STAGES": 3}, num_warps=8, num_stages=3, pre_hook=desc_pre_hook),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "PIPE_STAGES": 4}, num_warps=8, num_stages=4, pre_hook=desc_pre_hook),
    
    # Standard high-performance fallback structures
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "PIPE_STAGES": 3}, num_warps=8, num_stages=3, pre_hook=desc_pre_hook),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "PIPE_STAGES": 4}, num_warps=8, num_stages=4, pre_hook=desc_pre_hook),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "PIPE_STAGES": 3}, num_warps=4, num_stages=3, pre_hook=desc_pre_hook),
]

@triton.autotune(configs=_configs, key=["S"])
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE, sm_scale_log2,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, D,
    q_desc, k_desc, v_desc, o_desc,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    PIPE_STAGES: tl.constexpr
):
    start_m = tl.program_id(0) * BLOCK_M
    if start_m >= S:
        return

    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H

    # Seamless hardware-side striding and boundary padded loads via TMA
    q_blk = q_desc.load([off_b, off_h, start_m, 0])
    q = tl.reshape(q_blk, (BLOCK_M, BLOCK_D))
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    
    # Safe -inf init: causally valid bounds assure every participating m-row intercepts 
    # at least `n=0`, bypassing terminal NaN poisoning issues on `qk - m_ij`
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # === UNMASKED PIPELINED BLOCKS ===
    # Free of sequence boundary conditions avoiding tl.where bottlenecks
    for start_n in tl.range(0, start_m, BLOCK_N, num_stages=PIPE_STAGES):
        k_blk = k_desc.load([off_b, off_h, start_n, 0])
        k = tl.reshape(k_blk, (BLOCK_N, BLOCK_D))
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale_log2
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        v_blk = v_desc.load([off_b, off_h, start_n, 0])
        v = tl.reshape(v_blk, (BLOCK_N, BLOCK_D))
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # === MASKED DIAGONAL BLOCKS ===
    end_n = tl.minimum(S, start_m + BLOCK_M)
    for start_n in range(start_m, end_n, BLOCK_N):
        k_blk = k_desc.load([off_b, off_h, start_n, 0])
        k = tl.reshape(k_blk, (BLOCK_N, BLOCK_D))
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale_log2
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid_mask = (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < S)
        qk = tl.where(valid_mask, qk, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        v_blk = v_desc.load([off_b, off_h, start_n, 0])
        v = tl.reshape(v_blk, (BLOCK_N, BLOCK_D))
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # Safely normalize the cumulative sequence values
    out = acc / l_i[:, None]
    out = out.to(tl.bfloat16)
    
    out_blk = tl.reshape(out, (1, 1, BLOCK_M, BLOCK_D))
    o_desc.store([off_b, off_h, start_m, 0], out_blk)
    
    # Store Natural Log-Sum-Exp via element-wise multiplier fallback logic correctly onto Natural Log
    lse = m_i + tl.log2(l_i)
    lse = lse * 0.6931471805599453 
    lse_ptrs = LSE + off_b * stride_lseb + off_h * stride_lseh + offs_m * stride_lses
    mask_m = offs_m < S
    tl.store(lse_ptrs, lse, mask=mask_m)

def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
        
    # Prematurely multiplying scalar constants eliminates downstream arithmetic conversions mapping directly into native EX2 hardware instructions
    sm_scale_log2 = (1.0 / math.sqrt(D)) * 1.4426950408889634
    
    # Structuring execution layout aligning M bounds perfectly utilizing local concurrent Head/Batch L2 hit rates 
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H, 1)
    
    _attn_fwd_kernel[grid](
        Q, K, V, O, LSE, sm_scale_log2,
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D,
        None, None, None, None, # Intercepted and injected securely inside desc_pre_hook configs
        BLOCK_D=128
    )