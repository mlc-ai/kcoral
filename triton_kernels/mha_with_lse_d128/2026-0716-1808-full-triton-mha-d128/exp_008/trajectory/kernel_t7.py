import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def create_barrier():
    bar = tl.empty(1, dtype=tl.int32)
    tl.mbarrier_init(bar, 1)
    return bar


@triton.jit
def _attention_kernel(
    q_desc, k_descs, v_descs, o_desc,
    LSE_ptr,
    S, scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    q_start = tl.program_id(0) * BLOCK_M
    b_h = tl.program_id(1)
    WARP_ID = tl.program_id(2)
    
    buf_q = tl.shared_array((BLOCK_M, 128), dtype=tl.bfloat16)
    buf_k = [tl.shared_array((BLOCK_N, 128), dtype=tl.bfloat16) for _ in range(2)]
    buf_v = [tl.shared_array((BLOCK_N, 128), dtype=tl.bfloat16) for _ in range(2)]
    
    bar_q = create_barrier()
    bar_k = [create_barrier(), create_barrier()]
    bar_v = [create_barrier(), create_barrier()]
    bar_consumer_done = [create_barrier(), create_barrier()]
    
    num_kv_blocks = (S + BLOCK_N - 1) // BLOCK_N
    
    if WARP_ID < 4:  # Producer
        if WARP_ID == 0:
            q_desc.load_async(([b_h, q_start, 0]), barrier=bar_q)
        
        if WARP_ID == 0 and num_kv_blocks > 0:
            k_descs[0].load_async(([b_h, 0, 0]), barrier=bar_k[0])
            v_descs[0].load_async(([b_h, 0, 0]), barrier=bar_v[0])
        
        for j in range(1, num_kv_blocks):
            phase = j % 2
            if WARP_ID == 0:
                if j >= 2:
                    tl.mbarrier_arrive_and_expect_wakeup(bar_consumer_done[phase])
                
                k_descs[phase].load_async(([b_h, j * BLOCK_N, 0]), barrier=bar_k[phase])
                v_descs[phase].load_async(([b_h, j * BLOCK_N, 0]), barrier=bar_v[phase])
    
    else:  # Consumer
        tl.mbarrier_wait(bar_q)
        
        q0 = q_desc.load(([b_h, q_start, 0]))
        q1 = q_desc.load(([b_h, q_start, 64]))
        
        o_acc0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
        o_acc1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
        m_row = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        ell_row = tl.zeros((BLOCK_M,), dtype=tl.float32)
        
        row_idx = tl.arange(0, BLOCK_M)
        row_mask = (q_start + row_idx < S).to(tl.float32)
        
        for j in range(num_kv_blocks):
            phase = j % 2
            
            tl.mbarrier_wait(bar_k[phase])
            tl.mbarrier_wait(bar_v[phase])
            
            k0 = k_descs[phase].load(([b_h, j * BLOCK_N, 0]))
            k1 = k_descs[phase].load(([b_h, j * BLOCK_N, 64]))
            v0 = v_descs[phase].load(([b_h, j * BLOCK_N, 0]))
            v1 = v_descs[phase].load(([b_h, j * BLOCK_N, 64]))
            
            acc_o = tl.dot(q0, k0.T)
            acc_o = tl.dot(q1, k1.T, acc_o)
            
            acc_o = acc_o * scale
            
            col_idx = tl.arange(0, BLOCK_N)
            col_mask = (j * BLOCK_N + col_idx < S)[None, :]
            acc_o = tl.where(col_mask, acc_o, -float("inf"))
            
            local_m = tl.max(acc_o, axis=1)
            local_m = local_m * row_mask + (-float("inf")) * (1 - row_mask)
            
            old_m = m_row
            new_m = tl.maximum(old_m, local_m)
            
            alpha = tl.where(old_m > -1e38, tl.exp(old_m - new_m), 0.0)
            
            o_acc0 = o_acc0 * alpha[:, None]
            o_acc1 = o_acc1 * alpha[:, None]
            
            p_cur = tl.exp(acc_o - new_m[:, None])
            p_cur = p_cur * row_mask[:, None]
            
            ell_row = ell_row * alpha + tl.sum(p_cur, axis=1)
            
            p_cur_bf16 = p_cur.to(tl.bfloat16)
            
            o_acc0 = tl.dot(p_cur_bf16, v0, o_acc0)
            o_acc1 = tl.dot(p_cur_bf16, v1, o_acc1)
            
            m_row = new_m
            
            if WARP_ID == 4:
                tl.mbarrier_signal(bar_consumer_done[phase])
        
        inv_ell = 1.0 / ell_row
        
        out0 = (o_acc0 * inv_ell[:, None]).to(tl.bfloat16)
        out1 = (o_acc1 * inv_ell[:, None]).to(tl.bfloat16)
        
        out_unscaled = tl.cat([out0, out1], dim=1)
        
        o_desc.store([b_h, q_start, 0], out_unscaled)
        
        row_idx_32 = tl.arange(0, BLOCK_M)
        lse_ptr = LSE_ptr + (b_h // H) * (H * S) + (b_h % H) * S + q_start + row_idx_32
        lse_val = m_row + tl.log(ell_row)
        
        lse_val = tl.where(m_row > -1e38, lse_val, 0.0)
        
        tl.store(lse_ptr, lse_val, mask=q_start + row_idx_32 < S)


BLOCK_M = 64
BLOCK_N = 64

def run(Q, K, V, O, LSE):
    """Compute Multi-Head Attention O and LSE into preallocated CUDA tensors."""
    B, H, S, D = Q.shape
    device = Q.device
    scale = 1.0 / (D ** 0.5)
    
    q_desc = TensorDescriptor.from_tensor(Q.view(B * H, S, D), [1, BLOCK_M, D])
    k_desc = TensorDescriptor.from_tensor(K.view(B * H, S, D), [1, BLOCK_N, D])
    v_desc = TensorDescriptor.from_tensor(V.view(B * H, S, D), [1, BLOCK_N, D])
    o_desc = TensorDescriptor.from_tensor(O.view(B * H, S, D), [1, BLOCK_M, D])
    
    k_descs = [k_desc, k_desc]
    v_descs = [v_desc, v_desc]
    
    num_blocks = triton.cdiv(S, BLOCK_M)
    grid = (num_blocks, B * H, 8)
    
    print(f"Triton MHA Config: Grid={grid}, S={S}, Scale={scale}")
    
    _attention_kernel[grid](
        q_desc, k_descs, v_descs, o_desc, LSE,
        S, scale, H,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=2
    )