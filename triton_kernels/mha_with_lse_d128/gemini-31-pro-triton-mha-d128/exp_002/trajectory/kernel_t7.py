import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math

def get_config(block_m, block_n, num_stages, num_warps):
    """
    Creates a tuning configuration that dynamically prepares 4D host TensorDescriptors.
    This avoids the significant overhead of device-side descriptor creation in every CTA.
    """
    def hook(kwargs):
        Q = kwargs['Q']
        K = kwargs['K']
        V = kwargs['V']
        O = kwargs['O']
        D = kwargs['D']
        
        # Q, K, V, O are typically contiguous in the inner dimension, matching Hopper TMA constraints.
        # The block shape applies to the [B, H, M, D] structure logically as [1, 1, BLOCK, D].
        kwargs['q_desc'] = TensorDescriptor.from_tensor(Q, [1, 1, block_m, D])
        kwargs['k_desc'] = TensorDescriptor.from_tensor(K, [1, 1, block_n, D])
        kwargs['v_desc'] = TensorDescriptor.from_tensor(V, [1, 1, block_n, D])
        kwargs['o_desc'] = TensorDescriptor.from_tensor(O, [1, 1, block_m, D])
        
    return triton.Config(
        {'BLOCK_M': block_m, 'BLOCK_N': block_n},
        num_stages=num_stages,
        num_warps=num_warps,
        pre_hook=hook
    )


@triton.autotune(
    configs=[
        get_config(128, 128, 2, 8),
        get_config(128, 128, 3, 8),
        get_config(128, 64, 4, 4),
        get_config(128, 64, 5, 8),
        get_config(64, 128, 3, 4),
        get_config(64, 128, 4, 8),
        get_config(64, 64, 5, 4),
    ],
    key=['S']
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, # Passed primarily so the pre_hook can access the raw tensors
    q_desc, k_desc, v_desc, o_desc,
    sm_scale_log2,
    LSE_ptr,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    EVEN_S: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    start_m = pid_m * BLOCK_M
    
    # Load 4D TMA block and squeeze logically for the dot operations
    q_4d = q_desc.load([pid_b, pid_h, start_m, 0])
    q = tl.reshape(q_4d, (BLOCK_M, D))
    
    # Pre-scale statically using log2(e) to use hardware exp2 in the inner loop natively
    q = (q * sm_scale_log2).to(tl.bfloat16)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    if not EVEN_S:
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    
    # Pipelined hardware inner loop
    for n_idx in range(num_n_blocks):
        start_n = n_idx * BLOCK_N
        
        # TMA naturally pads out-of-bounds rows with zeros natively
        k_4d = k_desc.load([pid_b, pid_h, start_n, 0])
        v_4d = v_desc.load([pid_b, pid_h, start_n, 0])
        
        k = tl.reshape(k_4d, (BLOCK_N, D))
        v = tl.reshape(v_4d, (BLOCK_N, D))
        
        # WGMMA execution natively supported for non-transposed BF16 right operands
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        if not EVEN_S:
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_qk = mask_m[:, None] & (offs_n[None, :] < S)
            qk = tl.where(mask_qk, qk, float("-inf"))
            
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        if not EVEN_S:
            # Prevent fully out-of-bounds rows from evaluating `-inf - (-inf) = NaN`
            m_i_new = tl.where(mask_m, m_i_new, 0.0)
            
        # Standard software EX2 SASS mapping
        alpha = tl.exp2(m_i - m_i_new)
        beta = tl.exp2(qk - m_i_new[:, None])
        
        l_i_new = alpha * l_i + tl.sum(beta, 1)
        
        p = beta.to(tl.bfloat16)
        acc = acc * alpha[:, None]
        acc = tl.dot(p, v, acc, out_dtype=tl.float32)
        
        m_i = m_i_new
        l_i = l_i_new
        
    inv_l_i = 1.0 / l_i
    
    # Safe protection against completely out-of-bounds rows
    if not EVEN_S:
        inv_l_i = tl.where(mask_m, inv_l_i, 0.0)
        
    acc = acc * inv_l_i[:, None]
    O_val = acc.to(tl.bfloat16)
    
    # TMA implicitly masks out-of-bounds bounds without instructions when writing 
    O_val_4d = tl.reshape(O_val, (1, 1, BLOCK_M, D))
    o_desc.store([pid_b, pid_h, start_m, 0], O_val_4d)
    
    # Standard linear memory pointer stores for LSE
    offs_m_store = start_m + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE_ptr + pid_b * stride_lseb + pid_h * stride_lseh + offs_m_store * stride_lses
    
    # Shift base-2 back to natural logarithmic domain equivalent to PyTorch reference
    lse_val = (m_i + tl.log2(l_i)) * 0.6931471805599453
    
    if EVEN_S:
        tl.store(lse_ptrs, lse_val)
    else:
        tl.store(lse_ptrs, lse_val, mask=offs_m_store < S)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
        
    # Standard scaled attention conversion to Log2 scale
    sm_scale_log2 = (1.0 / math.sqrt(D)) * 1.4426950408889634
    even_s = (S % 128 == 0)
    
    # Natural 3D grid groups queries of the same Head together automatically
    # This acts as an organic L2 swizzle strategy for Hopper
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), H, B)
    
    # Initialize dummy host descriptors. The Triton config pre_hook executes at runtime
    # and populates these kwargs with strictly sized descriptors mapped to autotuned choices.
    dummy_q = TensorDescriptor.from_tensor(Q, [1, 1, 128, D])
    dummy_k = TensorDescriptor.from_tensor(K, [1, 1, 128, D])
    dummy_v = TensorDescriptor.from_tensor(V, [1, 1, 128, D])
    dummy_o = TensorDescriptor.from_tensor(O, [1, 1, 128, D])
    
    _attn_fwd_kernel[grid](
        Q, K, V, O,
        dummy_q, dummy_k, dummy_v, dummy_o,
        sm_scale_log2,
        LSE,
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D=D, EVEN_S=even_s
    )