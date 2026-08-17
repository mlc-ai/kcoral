import torch
import triton
import triton.language as tl


@triton.jit
def rescale_O(O_0, O_1, exp_scale):
    return O_0 * exp_scale[:, None], O_1 * exp_scale[:, None]


@triton.jit
def _mha_fwd_causal(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, H, B,
    SCALE,
    NUM_SMS: tl.constexpr,
    D: tl.constexpr = 128,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 64,
):
    start_pid = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    
    bh_idx = batch_idx * H + head_idx
    
    q_strides = [S * D, D, 1]
    k_strides = [S * D, D, 1]
    v_strides = [S * D, D, 1]
    o_strides = [S * D, D, 1]
    
    Q_desc = tl.make_tensor_descriptor(Q_ptr, shape=[B * H, S, D], strides=q_strides, block_shape=[1, BLOCK_M, D], padding_option="zero")
    K_desc = tl.make_tensor_descriptor(K_ptr, shape=[B * H, S, D], strides=k_strides, block_shape=[1, BLOCK_N, D], padding_option="zero")
    V_desc = tl.make_tensor_descriptor(V_ptr, shape=[B * H, S, D], strides=v_strides, block_shape=[1, BLOCK_N, D], padding_option="zero")
    O_desc = tl.make_tensor_descriptor(O_ptr, shape=[B * H, S, D], strides=o_strides, block_shape=[1, BLOCK_M, D])
    
    num_pid_m = tl.cdiv(S, BLOCK_M)
    
    for m in range(start_pid, num_pid_m, NUM_SMS):
        Q = Q_desc.load([bh_idx, m * BLOCK_M, 0])
        Q_0, Q_1 = tl.split(Q, dim=-1)
        
        O_acc_0 = tl.zeros((BLOCK_M, D//2), dtype=tl.float32)
        O_acc_1 = tl.zeros((BLOCK_M, D//2), dtype=tl.float32)
        
        m_val = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
        l_val = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
        
        q_seq = (m * BLOCK_M) + tl.arange(0, BLOCK_M)
        
        for j in range(j + 1):
            K = K_desc.load([bh_idx, j * BLOCK_N, 0])
            K_0, K_1 = tl.split(K, dim=-1)
            
            acc = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
            acc = acc * SCALE
            
            k_seq = (j * BLOCK_N) + tl.arange(0, BLOCK_N)
            causal_mask = (q_seq[:, None] >= k_seq[None, :]) & (q_seq[:, None] < S) & (k_seq[None, :] < S)
            acc = tl.where(causal_mask, acc, -1.0e20)
            
            m_old = m_val
            m_curr = tl.max(acc, axis=1)
            m_val = tl.maximum(m_old, m_curr)
            
            exp_scale = tl.exp(m_old - m_val)
            P = tl.exp(acc - m_val)
            P = tl.where(causal_mask, P, 0.0)
            
            l_curr = tl.sum(P, axis=1)
            l_val = l_val * exp_scale + l_curr
            
            O_acc_0, O_acc_1 = rescale_O(O_acc_0, O_acc_1, exp_scale)
            
            V = V_desc.load([bh_idx, j * BLOCK_N, 0])
            V_0, V_1 = tl.split(V, dim=-1)
            
            O_acc_0 = O_acc_0 + tl.dot(P, V_0)
            O_acc_1 = O_acc_1 + tl.dot(P, V_1)
            
        inv_l = 1.0 / l_val
        O_0 = (O_acc_0 * inv_l[:, None]).to(tl.bfloat16)
        O_1 = (O_acc_1 * inv_l[:, None]).to(tl.bfloat16)
        
        O = tl.join(O_0, O_1, dim=-1)
        O_desc.store([bh_idx, m * BLOCK_M, 0], O)


@triton.jit
def _compute_lse(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, H, B,
    SCALE,
    NUM_SMS: tl.constexpr,
    D: tl.constexpr = 128,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 64,
):
    start_pid = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    
    bh_idx = batch_idx * H + head_idx
    
    q_strides = [S * D, D, 1]
    k_strides = [S * D, D, 1]
    lse_strides = [S, 1, 1]
    
    Q_desc = tl.make_tensor_descriptor(Q_ptr, shape=[B * H, S, D], strides=q_strides, block_shape=[1, BLOCK_M, D], padding_option="zero")
    K_desc = tl.make_tensor_descriptor(K_ptr, shape=[B * H, S, D], strides=k_strides, block_shape=[1, BLOCK_N, D], padding_option="zero")
    LSE_desc = tl.make_tensor_descriptor(LSE_ptr, shape=[B * H, S, 1], strides=lse_strides, block_shape=[1, BLOCK_M, 1])
    
    num_pid_m = tl.cdiv(S, BLOCK_M)
    
    for m in range(start_pid, num_pid_m, NUM_SMS):
        Q = Q_desc.load([bh_idx, m * BLOCK_M, 0])
        Q_0, Q_1 = tl.split(Q, dim=-1)
        
        m_val = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
        l_val = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
        
        q_seq = (m * BLOCK_M) + tl.arange(0, BLOCK_M)
        
        for j in range(j + 1):
            K = K_desc.load([bh_idx, j * BLOCK_N, 0])
            K_0, K_1 = tl.split(K, dim=-1)
            
            acc = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
            acc = acc * SCALE
            
            k_seq = (j * BLOCK_N) + tl.arange(0, BLOCK_N)
            causal_mask = (q_seq[:, None] >= k_seq[None, :]) & (q_seq[:, None] < S) & (k_seq[None, :] < S)
            acc = tl.where(causal_mask, acc, -1.0e20)
            
            m_old = m_val
            m_curr = tl.max(acc, axis=1)
            m_val = tl.maximum(m_old, m_curr)
            
            exp_scale = tl.exp(m_old - m_val)
            P = tl.exp(acc - m_val)
            P = tl.where(causal_mask, P, 0.0)
            
            l_curr = tl.sum(P, axis=1)
            l_val = l_val * exp_scale + l_curr
            
        L_val = m_val + tl.log(l_val)
        LSE_desc.store([bh_idx, m * BLOCK_M, 0], L_val[:, None])


def run(Q, K, V, O, LSE):
    """Compute Causal MHA Forward Pass with LSE output."""
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)
    
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    Q_desc = TensorDescriptor.from_tensor(Q, [B * H, S, D], padding_option="zero")
    K_desc = TensorDescriptor.from_tensor(K, [B * H, S, D], padding_option="zero")
    V_desc = TensorDescriptor.from_tensor(V, [B * H, S, D], padding_option="zero")
    O_desc = TensorDescriptor.from_tensor(O, [B * H, S, D])
    LSE_desc = TensorDescriptor.from_tensor(LSE, [B * H, S, 1], padding_option="zero")
    
    scale = 1.0 / (D ** 0.5)
    NUM_SMS = 132
    
    grid = (min(NUM_SMS, triton.cdiv(S, 128)), H, B)
    
    _mha_fwd_causal[grid](
        Q, K, V, O, LSE,
        S, H, B,
        scale,
        NUM_SMS=NUM_SMS,
        num_warps=4,
        num_stages=3,
    )
    
    _compute_lse[grid](
        Q, K, V, O, LSE,
        S, H, B,
        scale,
        NUM_SMS=NUM_SMS,
        num_warps=4,
        num_stages=3,
    )