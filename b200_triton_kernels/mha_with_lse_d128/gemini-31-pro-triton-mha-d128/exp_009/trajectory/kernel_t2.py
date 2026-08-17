import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

def _update_desc(kwargs):
    """
    Hook to create efficient host-side TMA tensor descriptors 
    based on the current autotune trial's block configurations.
    """
    BLOCK_M = kwargs['BLOCK_M']
    BLOCK_N = kwargs['BLOCK_N']
    BLOCK_D = kwargs['BLOCK_D']
    
    # Generate 4D TensorDescriptors preserving logical coordinates mapping
    kwargs['Q_desc'] = TensorDescriptor.from_tensor(kwargs['Q'], [1, 1, BLOCK_M, BLOCK_D])
    kwargs['K_desc'] = TensorDescriptor.from_tensor(kwargs['K'], [1, 1, BLOCK_N, BLOCK_D])
    kwargs['V_desc'] = TensorDescriptor.from_tensor(kwargs['V'], [1, 1, BLOCK_N, BLOCK_D])
    kwargs['O_desc'] = TensorDescriptor.from_tensor(kwargs['O'], [1, 1, BLOCK_M, BLOCK_D])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=4, num_stages=4),
    ],
    key=['S'],
    pre_hook=_update_desc,
)
@triton.jit
def _fwd_kernel(
    Q, K, V, O, # Consumed/managed primarily by pre_hook to seed descriptors
    Q_desc, K_desc, V_desc, O_desc, LSE,
    sm_scale, H, S,
    stride_lseb, stride_lseh, stride_lses,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    
    offset_m = start_m * BLOCK_M
    
    # Asynchronous TMA descriptor load for query
    q = Q_desc.load([b, h, offset_m, 0])
    
    # Numerical stability trackers
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    num_steps = tl.cdiv(S, BLOCK_N)
    
    for i in tl.range(0, num_steps):
        offset_n = i * BLOCK_N
        
        # Asynchronous pipelined TMA loads for keys and values
        k = K_desc.load([b, h, offset_n, 0])
        v = V_desc.load([b, h, offset_n, 0])
        
        # Emits WGMMA instructions; physically transposed dot handles matching layouts
        qk = tl.dot(q, k.T)
        qk = qk * sm_scale
        
        # Bounds checking dynamically drops gracefully across sequences not perfectly divisible
        if S % BLOCK_N != 0:
            if i == num_steps - 1:
                offs_n = offset_n + tl.arange(0, BLOCK_N)
                qk = tl.where(offs_n[None, :] < S, qk, float("-inf"))
                
        # Running maximums, exponentially scaled probabilities
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        l_i_new = alpha * l_i + tl.sum(p, 1)
        
        # Multiply accumulators inline effectively casting on the fly bridging register spaces
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_i_new
        l_i = l_i_new

    # Epilogue rescaling and LSE calculation
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    l_i_safe = tl.where(mask_m, l_i, 1.0)
    acc = acc / l_i_safe[:, None]
    lse = m_i + tl.log(l_i_safe)
    
    # TMA store automatically drops sequence length trailing elements safely out-of-bounds
    O_desc.store([b, h, offset_m, 0], acc.to(tl.bfloat16))
    
    # Ordinary pointer store for the log sum exp values representing each sequence head explicitly
    lse_offs = b * stride_lseb + h * stride_lseh + offs_m * stride_lses
    tl.store(LSE + lse_offs, lse, mask=mask_m)

def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)
    
    # Align X-fastest grid directly along M sequence length to maximize natural L2 hits 
    # as keys and values scan alongside constant batches & heads
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    
    _fwd_kernel[grid](
        Q, K, V, O,
        None, None, None, None, LSE,
        sm_scale, H, S,
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        BLOCK_D=D
    )