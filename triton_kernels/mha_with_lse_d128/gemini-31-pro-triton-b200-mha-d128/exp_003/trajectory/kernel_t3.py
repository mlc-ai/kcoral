import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "WARP_SPECIALIZE": False, "PIPELINE_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "WARP_SPECIALIZE": True,  "PIPELINE_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "WARP_SPECIALIZE": False, "PIPELINE_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "WARP_SPECIALIZE": False, "PIPELINE_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "WARP_SPECIALIZE": False, "PIPELINE_STAGES": 4}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "WARP_SPECIALIZE": False, "PIPELINE_STAGES": 4}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "WARP_SPECIALIZE": False, "PIPELINE_STAGES": 5}, num_warps=8, num_stages=5),
    ],
    key=["S"],
)
@triton.jit
def _attn_fwd_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE,
    sm_scale,
    stride_lsez, stride_lseh, stride_lses,
    B, H, S, D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr, PIPELINE_STAGES: tl.constexpr
):
    start_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    
    batch_idx = batch_head // H
    head_idx = batch_head % H
    
    # Load Q tile using 4D TMA descriptor for zero-overhead stride and boundary handling
    q_4d = Q_desc.load([batch_idx, head_idx, start_m * BLOCK_M, 0])
    q = tl.reshape(q_4d, [BLOCK_M, D_HEAD])
    
    # Scale Q outside the hot loop to save 1 FP32 multiplication per element inside the loop.
    q = (q.to(tl.float32) * sm_scale).to(tl.bfloat16)
    
    # Initialize running softmax states
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    num_full_blocks = S // BLOCK_N
    
    # Process full blocks completely devoid of bounds-checking masks
    for block_idx in tl.range(0, num_full_blocks, num_stages=PIPELINE_STAGES, warp_specialize=WARP_SPECIALIZE):
        start_n = block_idx * BLOCK_N
        
        # Native TMA tensor descriptor loads for K and V tiles
        k_4d = K_desc.load([batch_idx, head_idx, start_n, 0])
        v_4d = V_desc.load([batch_idx, head_idx, start_n, 0])
        
        k = tl.reshape(k_4d, [BLOCK_N, D_HEAD])
        v = tl.reshape(v_4d, [BLOCK_N, D_HEAD])
        
        # Q @ K^T (q is already scaled)
        qk = tl.dot(q, k.T)
        
        # Softmax stats mapping
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        # Accumulate P @ V along with dynamically rescaling running result
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc)
        
        m_i = m_ij

    # Process the tail sequence block if sequence length is not purely divisible by BLOCK_N
    if not EVEN_N:
        start_n = num_full_blocks * BLOCK_N
        if start_n < S:
            k_4d = K_desc.load([batch_idx, head_idx, start_n, 0])
            v_4d = V_desc.load([batch_idx, head_idx, start_n, 0])
            
            k = tl.reshape(k_4d, [BLOCK_N, D_HEAD])
            v = tl.reshape(v_4d, [BLOCK_N, D_HEAD])
            
            qk = tl.dot(q, k.T)
            
            # Mask out-of-bounds padded keys to -inf prior to max operations
            offs_n_base = tl.arange(0, BLOCK_N)
            mask_k = (start_n + offs_n_base) < S
            qk = tl.where(mask_k[None, :], qk, float("-inf"))
            
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.bfloat16), v, acc=acc)
            
            m_i = m_ij

    # Normalize accumulated matrix and compute the exact LogSumExp
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    # Store matrix output O utilizing precise TMA ignoring out-of-bounds implicitly
    acc_4d = tl.reshape(acc, [1, 1, BLOCK_M, D_HEAD])
    O_desc.store([batch_idx, head_idx, start_m * BLOCK_M, 0], acc_4d.to(tl.bfloat16))
    
    # Store output LSE with pointers (using straightforward masking if necessary)
    lse_offset = batch_idx * stride_lsez + head_idx * stride_lseh
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    
    if EVEN_M:
        tl.store(lse_ptrs, lse)
    else:
        mask_m = offs_m < S
        tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    """
    Computes Non-causal Multi-Head Attention targeting optimized NVIDIA Blackwell SM100 limits.
    Utilizes Host TMA Descriptors and inner-loop scaling elimination for maximal hardware utilization.
    Outputs are written directly to preallocated destination tensors O and LSE.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)

    # Dynamic Host TMA descriptor allocation using structural physical shapes allows zero-overhead
    # descriptor alignment across complex stride layouts seamlessly backing up TMA capability
    # No allocations or reassignments are triggered; all inputs map natively matching physical layout dimensions
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, 128, D]) # Placeholder, dynamically adjusted in kernel/hooks
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, 128, D]) 
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, 128, D])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, 128, D])

    # Pre-hooks required due to compile-time dynamic tuning on block_shapes dynamically mutating the descriptors
    def config_pre_hook(args):
        kwargs = args['kwargs']
        BLOCK_M = kwargs['BLOCK_M']
        BLOCK_N = kwargs['BLOCK_N']
        args['kwargs']['Q_desc'] = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, D])
        args['kwargs']['K_desc'] = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, D])
        args['kwargs']['V_desc'] = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, D])
        args['kwargs']['O_desc'] = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, D])

    # Unravel grid preserving start_m adjacent bounds to maximize potential sequence-level structural cache reuse
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    
    # Exploit static evaluation attributes stripping loop boundaries where naturally divisible sequences exist
    _attn_fwd_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        sm_scale,
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D_HEAD=D,
        EVEN_M=(S % 128 == 0),
        EVEN_N=(S % 128 == 0),
        pre_hook=config_pre_hook
    )