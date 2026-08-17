import torch
import triton
import triton.language as tl


@triton.jit
def _attention_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    S,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    
    b_h = pid_b * 48 + pid_h
    
    row_idx = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    row_idx_max = pid_s * BLOCK_S + BLOCK_S - 1
    
    if row_idx_max >= S:
        row_idx_max = S - 1
        
    valid_rows = row_idx < S
    
    q_base = q_ptr + b_h * S * 128
    k_base = k_ptr + b_h * S * 128
    v_base = v_ptr + b_h * S * 128
    o_base = o_ptr + b_h * S * 128
    
    m_i = tl.full([BLOCK_S], -float("inf"), dtype=tl.float32)
    D_i = tl.full([BLOCK_S], 0.0, dtype=tl.float32)
    
    out_acc = tl.zeros([BLOCK_S, 128], dtype=tl.float32)
    
    scale = 1.0 / (128.0 ** 0.5)
    
    for k_tile in range(pid_s + 1):
        if k_tile * BLOCK_S > row_idx_max:
            break
        
        k_tile_row = k_tile * BLOCK_S + tl.arange(0, BLOCK_S)
        
        for chunk in range(0, 128, 128):
            col_chunk = chunk + tl.arange(0, 128)
            
            q_ptr = q_base + row_idx[:, None] * 128 + col_chunk[None, :]
            q = tl.load(q_ptr, other=0.0)
            
            k_ptr = k_base + k_tile_row[:, None] * 128 + col_chunk[None, :]
            k = tl.load(k_ptr, other=0.0)
            
            p_chunk = tl.dot(q, k.T)
            
            p_chunk = p_chunk * scale
            
            col_idx = k_tile * BLOCK_S + tl.arange(0, BLOCK_S)
            mask = row_idx[:, None] >= col_idx[None, :]
            p_chunk[~mask] = -float('inf')
            
            curr_max = tl.maximum(m_i, tl.max(p_chunk, axis=1))
            exp_diff = tl.exp(curr_max - m_i)
            D_i = D_i * exp_diff
            m_i = curr_max
            
            D_i += tl.sum(tl.exp(p_chunk - m_i), axis=1)
            
            out_acc *= exp_diff[:, None]
            
            row_d = tl.arange(0, 128)
            v_ptr = v_base + k_tile_row[:, None] * 128 + row_d[None, :]
            v = tl.load(v_ptr, other=0.0)
            
            out = tl.exp(p_chunk - m_i)
            out_acc = tl.dot(out.to(tl.bfloat16), v)
            
    final_out = out_acc.to(tl.bfloat16)
    
    col_d = tl.arange(0, 128)
    valid = valid_rows[:, None] & (col_d[None, :] < 128)
    tl.store(o_base + row_idx[:, None] * 128 + col_d[None, :], final_out, mask=valid)
    
    lse = m_i + tl.log(D_i)
    
    lse_ptr = lse_ptr + b_h * S
    lse_store = lse if valid_rows else 0.0
    tl.store(lse_ptr + row_idx, lse_store, mask=valid_rows)


def run(Q, K, V, O, LSE):
    """
    Computes causal multi-head attention O and LSE given Q, K, V on the current device.
    Expected shapes:
      Q, K, V : [B, H, S, D]
      O       : [B, H, S, D]
      LSE     : [B, H, S]
    Where B=4, H=48, D=128 are implicit in the layout / strides.
    """
    torch.cuda.set_device(Q.device)
    
    S = Q.shape[2]
    
    block_size = 128
    grid = (
        4,   # B
        48,  # H
        triton.cdiv(S, block_size)
    )
    
    _attention_forward_kernel[grid](
        Q, K, V, O, LSE,
        S,
        BLOCK_S=block_size,
        num_warps=4,
        num_stages=1,
    )


if __name__ == "__main__":
    S = 1024
    device = "cuda"
    Q = torch.randn(4, 48, S, 128, device=device, dtype=torch.bfloat16)
    K = torch.randn(4, 48, S, 128, device=device, dtype=torch.bfloat16)
    V = torch.randn(4, 48, S, 128, device=device, dtype=torch.bfloat16)
    
    O = torch.empty(4, 48, S, 128, device=device, dtype=torch.bfloat16)
    LSE = torch.empty(4, 48, S, device=device, dtype=torch.float32)
    
    run(Q, K, V, O, LSE)
    
    print("Custom kernel outputs computed.")
    print(f"O min={O.min().item():.4f}, max={O.max().item():.4f}, mean={O.mean().item():.4f}")
    print(f"LSE min={LSE.min().item():.4f}, max={LSE.max().item():.4f}, mean={LSE.mean().item():.4f}")
    
    with torch.no_grad():
        torch.cuda.set_device(Q.device)
        ref_O, ref_LSE = torch.ops.aten._scaled_dot_product_cudnn_attention(
            Q, K, V,
            None,
            True,
            0.0,
            True,
            False,
        )
        ref_LSE = ref_LSE.squeeze(-1)
    
    print("\nReference outputs computed.")
    print(f"ref_O min={ref_O.min().item():.4f}, max={ref_O.max().item():.4f}, mean={ref_O.mean().item():.4f}")
    print(f"ref_LSE min={ref_LSE.min().item():.4f}, max={ref_LSE.max().item():.4f}, mean={ref_LSE.mean().item():.4f}")
    
    close_O = torch.allclose(O, ref_O, atol=1e-2, rtol=1e-2)
    close_LSE = torch.allclose(LSE, ref_LSE, atol=1e-3, rtol=1e-3)
    
    print(f"\nMatches reference (O atol=1e-2, rtol=1e-2): {close_O}")
    print(f"Matches reference (LSE atol=1e-3, rtol=1e-3): {close_LSE}")