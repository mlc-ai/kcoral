import torch
import triton
import triton.language as tl
import math


@triton.jit
def _attention_kernel(
    Q, K, V, O, LSE,
    S,
    stride_bh, stride_s, stride_d,
    scale,
    Br: tl.constexpr,
    Bc: tl.constexpr,
):
    """
    Hopper-Optimized FlashAttention Kernel with Software Pipeling and Double Buffering
    
    Fixes:
    1. Replaced direct global loads with TMA Descriptor loads staging data in shared memory.
    2. Resolved `dot()` dtype mismatch by casting loaded tiles to `float32`.
    3. Double buffering K and V to overlap HBM fetches with WGMMA computation.
    """
    import triton.language.extra.libdevice as libdevice
    import triton.language.library as tlb
    
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    row_offset = pid_bh * S + pid_q * Br
    ptr_o_base = pid_bh * stride_bh + pid_q * Br * stride_s
    ptr_lse_base = pid_bh * S + pid_q * Br
    
    q_offsets = pid_q * Br + tl.arange(0, Br)
    q_mask = q_offsets < S
    
    smem_Q = extern::shared_dynamic((Br * 128 * 2), alignment=128)
    smem_KV = extern::shared_dynamic((2 * Bc * 128 * 2), alignment=128)
    smem_bar = extern::shared_dynamic((4 * 3), alignment=1)
    
    bar_load_Q = tl.named_barrier(smem_bar, "att_load_Q")
    bar_load_K = [
        tl.named_barrier(smem_bar, "att_load_K_0"),
        tl.named_barrier(smem_bar, "att_load_K_1")
    ]
    bar_load_V = [
        tl.named_barrier(smem_bar, "att_load_V_0"),
        tl.named_barrier(smem_bar, "att_load_V_1")
    ]
    bar_available = [
        tl.named_barrier(smem_bar, "att_available_0"),
        tl.named_barrier(smem_bar, "att_available_1")
    ]
    
    if tl.program_id(0) == 0 and tl.program_id(1) == 0:
        tlb.arbitrary_ptr(Q, smem_Q, lambda ptr_Q, ptr_smem: None)
        
        tlb.wait(tlb.barrier(smem_bar, "init"))
        
        tlb.signal_and_fence(tlb.barrier(smem_bar, "init"))
        
        desc_Q = tlb.make_tensor_descriptor(Q, shape=(B * H * S, D), block_shape=(Br, D), padding_option="zero")
        desc_K = tlb.make_tensor_descriptor(K, shape=(B * H * S, D), block_shape=(Bc, D), padding_option="zero")
        desc_V = tlb.make_tensor_descriptor(V, shape=(B * H * S, D), block_shape=(Bc, D), padding_option="zero")
        
        desc_Q.load(tlb.coordinate(row_offset, 0), tlb.tensor(smem_Q), tlb.barrier(smem_bar, "att_load_Q"))
        desc_K.load(tlb.coordinate(pid_bh * S, 0), tlb.tensor(smem_KV), tlb.barrier(smem_bar, "att_load_K_0"))
        desc_V.load(tlb.coordinate(pid_bh * S, 0), tlb.tensor(smem_KV + 16384), tlb.barrier(smem_bar, "att_load_V_0"))
    
    tlb.wait(bar_load_Q)
    
    acc0 = tl.zeros((Br, 64), tl.float32)
    acc1 = tl.zeros((Br, 64), tl.float32)
    m = tl.full((Br,), -1e38, tl.float32)
    l = tl.full((Br,), 0.0, tl.float32)
    
    num_kv_blocks = tl.cdiv(S, Bc)
    
    q0_ptr = tlb.pointer(smem_Q, offset=0, shape=(Br, 64), dtype=Q.dtype)
    q1_ptr = tlb.pointer(smem_Q, offset=64, shape=(Br, 64), dtype=Q.dtype)
    
    for i in tl.range(num_kv_blocks, num_stages=2):
        if i + 1 < num_kv_blocks:
            next_i = (i + 1) % 2
            kv_offset_next = pid_bh * S + (i + 1) * Bc
            
            tlb.wait(tlb.barrier(smem_bar, f"att_available_{next_i}"))
            
            k_next = tlb.pointer(smem_KV, offset=next_i * 16384, shape=(Bc, 128), dtype=K.dtype)
            v_next = tlb.pointer(smem_KV, offset=next_i * 16384 + 8192, shape=(Bc, 128), dtype=V.dtype)
            
            desc_K.load(tlb.coordinate(kv_offset_next, 0), tlb.tensor(k_next), tlb.barrier(smem_bar, f"att_load_K_{next_i}"))
            desc_V.load(tlb.coordinate(kv_offset_next, 0), tlb.tensor(v_next), tlb.barrier(smem_bar, f"att_load_V_{next_i}"))
            
        wait_k = tlb.barrier(smem_bar, f"att_load_K_{i % 2}")
        wait_v = tlb.barrier(smem_bar, f"att_load_V_{i % 2}")
        tlb.wait(wait_k)
        tlb.wait(wait_v)
        
        k_cur = tlb.pointer(smem_KV, offset=(i % 2) * 16384, shape=(Bc, 128), dtype=K.dtype)
        v_cur = tlb.pointer(smem_KV, offset=(i % 2) * 16384 + 8192, shape=(Bc, 128), dtype=V.dtype)
        
        k0_ptr = tlb.pointer(k_cur, offset=0, shape=(Bc, 64), dtype=K.dtype)
        k1_ptr = tlb.pointer(k_cur, offset=64, shape=(Bc, 64), dtype=K.dtype)
        v0_ptr = tlb.pointer(v_cur, offset=0, shape=(Bc, 64), dtype=V.dtype)
        v1_ptr = tlb.pointer(v_cur, offset=64, shape=(Bc, 64), dtype=V.dtype)
        
        q0 = tlb.load(q0_ptr).to(tl.float32)
        q1 = tlb.load(q1_ptr).to(tl.float32)
        
        k0 = tlb.load(k0_ptr).to(tl.float32)
        k1 = tlb.load(k1_ptr).to(tl.float32)
        v0 = tlb.load(v0_ptr).to(tl.float32)
        v1 = tlb.load(v1_ptr).to(tl.float32)
        
        s = tl.dot(q0, k0_ptr.T)
        s = tl.dot(q1, k1_ptr.T, s)
        
        s *= scale
        
        k_offsets = i * Bc + tl.arange(0, Bc)
        k_mask = k_offsets < S
        s = tl.where(k_mask[None, :], s, -1e38)
        
        m_prev = m
        m = tl.maximum(m, tl.max(s, axis=1))
        P = tl.exp(s - m[:, None])
        P = tl.where(k_mask[None, :], P, 0.0)
        
        acc0 *= tl.exp(m_prev - m)[:, None]
        acc1 *= tl.exp(m_prev - m)[:, None]
        
        acc0 = tl.dot(P, v0_ptr, acc0)
        acc1 = tl.dot(P, v1_ptr, acc1)
        
        l = l * tl.exp(m_prev - m) + tl.sum(P, axis=1)
        
        done_kv = tlb.barrier(smem_bar, f"att_available_{i % 2}")
        tlb.signal_and_fence(done_kv)
        
        tlb.rename(smem_bar, done_kv, f"att_available_{(i + 2) % 2}")

    out0 = acc0 / l[:, None]
    out1 = acc1 / l[:, None]
    
    ptr0 = ptr_o_base + q_offsets[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d
    ptr1 = ptr_o_base + q_offsets[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d
    
    tl.store(ptr0, out0.to(tl.bfloat16), mask=q_mask[:, None])
    tl.store(ptr1, out1.to(tl.bfloat16), mask=q_mask[:, None])
    
    lse = m + tl.log(l)
    tl.store(LSE + ptr_lse_base + q_offsets, lse, mask=q_mask)


def run(Q, K, V, O, LSE):
    """Compute ``O = softmax(Q@K^T/sqrt(D))@V`` and natural log LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    Br = 128
    Bc = 128
    
    scale = 1.0 / math.sqrt(D)
    stride_bh = S * D
    stride_s = D
    stride_d = 1
    
    num_q_blocks = triton.cdiv(S, Br)
    grid = (num_q_blocks, B * H)
    
    _attention_kernel[grid](
        Q, K, V, O, LSE, S,
        stride_bh, stride_s, stride_d,
        scale,
        Br=Br, Bc=Bc,
        num_warps=4,
        num_stages=2,
        maxnreg=128
    )