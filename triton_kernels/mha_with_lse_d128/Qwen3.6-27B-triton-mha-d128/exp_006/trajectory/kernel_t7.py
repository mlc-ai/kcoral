import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_alloc_fn)


@triton.jit
def _attention_kernel(
    Q_desc, K_desc, V_desc, O_desc,
    LSE_ptr,
    stride_lsb, stride_lsh, stride_lss,
    B, H, S, D,
    inv_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    # Persistent grid: iterate over (batch, head, q_tile) assignments
    pid = tl.program_id(0)
    n_sms = NUM_SMS
    
    num_q_tiles = tl.cdiv(S, BLOCK_M)
    num_pairs = B * H * num_q_tiles
    
    off_m = tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_D)
    
    for pair_id in range(pid, num_pairs, n_sms):
        # Decompose pair_id into (batch, head, q_tile)
        bid = pair_id // (H * num_q_tiles)
        remainder = pair_id % (H * num_q_tiles)
        hid = remainder // num_q_tiles
        pid_m = remainder % num_q_tiles
        
        q_offset_m = pid_m * BLOCK_M
        
        mask_m = (q_offset_m + off_m) < S
        
        # Load Q tile using descriptor - padded out-of-bounds elements to zero
        q = Q_desc.load([bid * S + hid * S + q_offset_m, 0])
        
        # Initialize online softmax accumulators
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc_o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        
        num_kv_iters = tl.cdiv(S, BLOCK_N)
        
        for kv_it in range(num_kv_iters):
            offset_n = kv_it * BLOCK_N
            
            k = K_desc.load([bid * S + hid * S + offset_n, 0])
            v = V_desc.load([bid * S + hid * S + offset_n, 0])
            
            s = tl.dot(q, k.T, out_dtype=tl.float32)
            s = s * inv_scale
            
            m_ij = tl.max(s, axis=1)
            m_i_new = tl.maximum(m_i, m_ij)
            
            p = tl.exp(s - m_i_new[:, None])
            
            old_scale = tl.exp(m_i - m_i_new)
            l_i_new = old_scale * l_i + tl.sum(p, axis=1)
            acc_o = old_scale[:, None] * acc_o + tl.dot(
                p.to(tl.bfloat16), v, acc=None, out_dtype=tl.float32)
            
            m_i = m_i_new
            l_i = l_i_new
        
        # Normalize and convert output
        o_final = acc_o / l_i[:, None]
        o_stored = o_final.to(tl.bfloat16)
        
        # Store O tile using descriptor
        O_desc.store([bid * S + hid * S + q_offset_m, 0], o_stored)
        
        # Store LSE values - still need pointer arithmetic for this rank-3 tensor
        row_idx = q_offset_m + off_m
        lse_val = m_i + tl.log(l_i)
        lse_ptrs = (LSE_ptr
                    + bid * stride_lsb + hid * stride_lsh
                    + row_idx * stride_lss)
        tl.store(lse_ptrs, lse_val, mask=mask_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape[0], Q.shape[1], Q.shape[2], Q.shape[3]

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D  # = 128

    # Flatten BH into a single logical sequence dimension for descriptor simplicity
    # Logical shape per-descriptor: [B*H*S, D]
    SQ = B * H * S
    strides_sq = 1
    strides_d = D
    
    # Create descriptors for each tensor (treating BH* flattened)
    # We flatten B,H,S into one dimension: linear index = ((b*H)+h)*S + s
    def reshape_for_descriptor(t):
        """Flatten [B,H,S,D] -> [B*H*S, D]"""
        return t.reshape(B * H * S, D)
    
    Q_flat = reshape_for_descriptor(Q)
    K_flat = reshape_for_descriptor(K)
    V_flat = reshape_for_descriptor(V)
    O_flat = reshape_for_descriptor(O)
    
    Q_desc = TensorDescriptor.from_tensor(
        Q_flat, [BLOCK_M, BLOCK_D], padding_option="zero")
    K_desc = TensorDescriptor.from_tensor(
        K_flat, [BLOCK_N, BLOCK_D], padding_option="zero")
    V_desc = TensorDescriptor.from_tensor(
        V_flat, [BLOCK_N, BLOCK_D], padding_option="zero")
    O_desc = TensorDescriptor.from_tensor(
        O_flat, [BLOCK_M, BLOCK_D])
    
    # Grid: min(NUM_SMS, total_work) programs for persistent execution
    num_q_tiles = triton.cdiv(S, BLOCK_M)
    num_pairs = B * H * num_q_tiles
    
    NUM_SMS = 132  # H100 has 132 SMs
    grid = (min(NUM_SMS, num_pairs),)
    
    inv_scale = 1.0 / (D ** 0.5)
    
    _attention_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc,
        LSE,
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D,
        inv_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        NUM_SMS=NUM_SMS,
        num_warps=8,
        num_stages=4,
    )