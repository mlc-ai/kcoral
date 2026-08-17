import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _dQ_kernel(
    Q_desc,
    K_desc,
    V_desc,
    O, dO, L, dQ,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lsb, stride_lsh, stride_lss,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    B, H, S,
    scale,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Total number of work items: each combines (batch, head, q_block)
    num_pid_bh = B * H
    num_pid_m = tl.cdiv(S, BLOCK_M)
    total_tiles = num_pid_bh * num_pid_m
    
    # Persistent scheduling: each CTA cycles through assigned tiles
    for tile_id in range(pid, total_tiles, NUM_SMS):
        pid_bh = tile_id // num_pid_m
        pid_m = tile_id % num_pid_m
        
        batch = pid_bh // H
        head = pid_bh % H
        
        offs_d = tl.arange(0, BLOCK_D)
        
        m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_boundary = m_off < S
        m_d_mask = m_boundary[:, None]
        
        l_off_batch_head = batch * stride_lsb + head * stride_lsh
        dq_off_base = batch * stride_dqb + head * stride_dqh
        
        # Load Q tile via descriptor
        q_tile = Q_desc.load([m_off, offs_d], boundary_check=(0, 1))
        
        # Load dO tile
        do_ptrs = dO + batch * stride_dob + head * stride_doh + m_off[:, None] * stride_dos + offs_d[None, :] * stride_dod
        do_tile = tl.load(do_ptrs, mask=m_d_mask, other=0.0)
        
        # Load O tile
        o_ptrs = O + batch * stride_ob + head * stride_oh + m_off[:, None] * stride_os + offs_d[None, :] * stride_od
        o_tile = tl.load(o_ptrs, mask=m_d_mask, other=0.0)
        
        # D[i] = sum(dO[i,k] * O[i,k])
        D = tl.sum(do_tile * o_tile, axis=1)
        
        # Load L
        l_ptrs = L + l_off_batch_head + m_off * stride_lss
        L_tile = tl.load(l_ptrs, mask=m_boundary, other=float("inf"))
        
        # Accumulator for dQ
        dQ_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        
        # Iterate over all KV blocks
        num_n_blocks = tl.cdiv(S, BLOCK_N)
        for pid_n in range(num_n_blocks):
            n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            n_boundary = n_off < S
            n_d_mask = n_boundary[:, None]
            
            # Load K via descriptor
            k_tile = K_desc.load([n_off, offs_d], boundary_check=(0, 1))
            
            # Load V via descriptor
            v_tile = V_desc.load([n_off, offs_d], boundary_check=(0, 1))
            
            # S = Q @ K^T * scale
            S_tile = tl.dot(q_tile, k_tile.T) * scale
            
            # P = exp(S - L)
            P = tl.exp(S_tile - L_tile[:, None])
            
            # dP = dO @ V^T
            dP = tl.dot(do_tile, v_tile.T)
            
            # dS = P * (dP - D) * scale
            dS = P * (dP - D[:, None]) * scale
            
            # dQ += dS @ K
            dQ_acc = tl.dot(dS.to(tl.bfloat16), k_tile, dQ_acc)
        
        # Store dQ
        dq_ptrs = dQ + dq_off_base + m_off[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
        tl.store(dq_ptrs, dQ_acc.to(tl.bfloat16), mask=m_d_mask)


@triton.jit
def _dKV_kernel(
    Q_desc,
    K_desc,
    V_desc,
    O, dO, L, dK, dV,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lsb, stride_lsh, stride_lss,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    B, H, S,
    scale,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Total work items: each combines (batch, head, kv_block)
    num_pid_bh = B * H
    num_pid_n = tl.cdiv(S, BLOCK_N)
    total_tiles = num_pid_bh * num_pid_n
    
    for tile_id in range(pid, total_tiles, NUM_SMS):
        pid_bh = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        
        batch = pid_bh // H
        head = pid_bh % H
        
        offs_d = tl.arange(0, BLOCK_D)
        
        n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_boundary = n_off < S
        n_d_mask = n_boundary[:, None]
        
        dk_off_base = batch * stride_dkb + head * stride_dkh
        dv_off_base = batch * stride_dvb + head * stride_dvh
        
        # Load K via descriptor
        k_tile = K_desc.load([n_off, offs_d], boundary_check=(0, 1))
        
        # Load V via descriptor
        v_tile = V_desc.load([n_off, offs_d], boundary_check=(0, 1))
        
        # Accumulators
        dK_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        dV_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        
        num_m_blocks = tl.cdiv(S, BLOCK_M)
        
        for pid_m in range(num_m_blocks):
            m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            m_boundary = m_off < S
            m_d_mask = m_boundary[:, None]
            
            # Load Q via descriptor
            q_tile = Q_desc.load([m_off, offs_d], boundary_check=(0, 1))
            
            # Load dO tile
            do_ptrs = dO + batch * stride_dob + head * stride_doh + m_off[:, None] * stride_dos + offs_d[None, :] * stride_dod
            do_tile = tl.load(do_ptrs, mask=m_d_mask, other=0.0)
            
            # Load O tile
            o_ptrs = O + batch * stride_ob + head * stride_oh + m_off[:, None] * stride_os + offs_d[None, :] * stride_od
            o_tile = tl.load(o_ptrs, mask=m_d_mask, other=0.0)
            
            # D[i] = sum(dO[i,k] * O[i,k])
            D = tl.sum(do_tile * o_tile, axis=1)
            
            # Load L
            l_ptrs = L + batch * stride_lsb + head * stride_lsh + m_off * stride_lss
            L_tile = tl.load(l_ptrs, mask=m_boundary, other=float("inf"))
            
            # S = Q @ K^T * scale
            S_tile = tl.dot(q_tile, k_tile.T) * scale
            
            # P = exp(S - L)
            P = tl.exp(S_tile - L_tile[:, None])
            
            # dP = dO @ V^T
            dP = tl.dot(do_tile, v_tile.T)
            
            # dS = P * (dP - D) * scale
            dS = P * (dP - D[:, None]) * scale
            
            # dK += dS^T @ Q
            dK_acc = tl.dot(dS.T.to(tl.bfloat16), q_tile, dK_acc)
            
            # dV += P^T @ dO
            dV_acc = tl.dot(P.T.to(tl.bfloat16), do_tile, dV_acc)
        
        # Store dK
        dk_ptrs = dK + dk_off_base + n_off[:, None] * stride_dks + offs_d[None, :] * stride_dkd
        tl.store(dk_ptrs, dK_acc.to(tl.bfloat16), mask=n_d_mask)
        
        # Store dV
        dv_ptrs = dV + dv_off_base + n_off[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
        tl.store(dv_ptrs, dV_acc.to(tl.bfloat16), mask=n_d_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Multi-head attention backward pass using TMA on Hopper."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128

    BH = B * H

    qs = Q.stride()
    ks = K.stride()
    vs = V.stride()
    os_s = O.stride()
    dos = dO.stride()
    ls = L.stride()
    dqs = dQ.stride()
    dks = dK.stride()
    dvs = dV.stride()

    # Flatten Q, K, V to [B*H, S, D] for descriptor construction
    Q_flat = Q.reshape(BH, S, D)
    K_flat = K.reshape(BH, S, D)
    V_flat = V.reshape(BH, S, D)

    # Create TMA-compatible contiguous buffers if needed
    Q_c = Q_flat.contiguous()
    K_c = K_flat.contiguous()
    V_c = V_flat.contiguous()

    # Build tensor descriptors for Q, K, V
    Q_desc = TensorDescriptor.from_tensor(Q_c, block_shape=[BLOCK_M, BLOCK_D])
    K_desc = TensorDescriptor.from_tensor(K_c, block_shape=[BLOCK_N, BLOCK_D])
    V_desc = TensorDescriptor.from_tensor(V_c, block_shape=[BLOCK_N, BLOCK_D])

    NUM_SMS = 132

    # Kernel 1: compute dQ
    grid_dQ = (min(NUM_SMS, BH * triton.cdiv(S, BLOCK_M)),)
    _dQ_kernel[grid_dQ](
        Q_desc, K_desc, V_desc, O, dO, L, dQ,
        os_s[0], os_s[1], os_s[2], os_s[3],
        dos[0], dos[1], dos[2], dos[3],
        ls[0], ls[1], ls[2],
        dqs[0], dqs[1], dqs[2], dqs[3],
        B, H, S,
        scale,
        NUM_SMS=NUM_SMS,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=8, num_stages=4,
    )

    # Kernel 2: compute dK and dV
    grid_dKV = (min(NUM_SMS, BH * triton.cdiv(S, BLOCK_N)),)
    _dKV_kernel[grid_dKV](
        Q_desc, K_desc, V_desc, O, dO, L, dK, dV,
        os_s[0], os_s[1], os_s[2], os_s[3],
        dos[0], dos[1], dos[2], dos[3],
        ls[0], ls[1], ls[2],
        dks[0], dks[1], dks[2], dks[3],
        dvs[0], dvs[1], dvs[2], dvs[3],
        B, H, S,
        scale,
        NUM_SMS=NUM_SMS,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=8, num_stages=4,
    )