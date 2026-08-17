import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
    ],
    key=["S_len"],
)
@triton.jit
def bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S_len, stride_m, stride_d, HEAD_STRIDE,
    BLOCK: tl.constexpr, SCALE: tl.constexpr,
):
    i_start = tl.program_id(0) * BLOCK
    head_idx = tl.program_id(1)
    
    row_offsets_128 = tl.arange(0, BLOCK)
    col_offsets_64 = tl.arange(0, 64)
    
    sq0 = __shared__ array('sq0', BLOCK * 64)
    sq1 = __shared__ array('sq1', BLOCK * 64)
    sk0 = __shared__ array('sk0', BLOCK * 64)
    sk1 = __shared__ array('sk1', BLOCK * 64)
    sv0 = __shared__ array('sv0', BLOCK * 64)
    sv1 = __shared__ array('sv1', BLOCK * 64)
    so0 = __shared__ array('so0', BLOCK * 64)
    so1 = __shared__ array('so1', BLOCK * 64)
    sdo0 = __shared__ array('sdo0', BLOCK * 64)
    sdo1 = __shared__ array('sdo1', BLOCK * 64)
    
    s_Q0 = extern_shared_array("s_Q0", BLOCK * 64 * 2)
    s_Q1 = extern_shared_array("s_Q1", BLOCK * 64 * 2)
    s_K0 = extern_shared_array("s_O0", BLOCK * 64 * 2)
    s_K1 = extern_shared_array("s_O1", BLOCK * 64 * 2)
    s_V0 = extern_shared_array("s_dO0", BLOCK * 64 * 2)
    s_V1 = extern_shared_array("s_dO1", BLOCK * 64 * 2)
    s_O0 = extern_shared_array("s_O0", BLOCK * 64 * 2)
    s_O1 = extern_shared_array("s_O1", BLOCK * 64 * 2)
    s_dO0 = extern_shared_array("s_dO0", BLOCK * 64 * 2)
    s_dO1 = extern_shared_array("s_dO1", BLOCK * 64 * 2)
    s_D = extern_shared_array("s_D", BLOCK * 4)
    s_L = extern_shared_array("s_L", BLOCK * 4)
    
    def load_q(base_addr, s_offset):
        for i in range(0, BLOCK * 64, 8):
            idx = tl.program_id(2) * 8 + i
            off = s_offset + idx
            val = tl.load(base_addr + off, mask=off < (s_offset + BLOCK * 64), other=0)
            sq0[idx] = val.to(tl.uint32)
            
    def load_q_1(base_addr, s_offset):
        for i in range(0, BLOCK * 64, 8):
            idx = tl.program_id(2) * 8 + i
            off = s_offset + idx
            val = tl.load(base_addr + off, mask=off < (s_offset + BLOCK * 64), other=0)
            sq1[idx] = val.to(tl.uint32)
            
    def load_o(base_addr, s_offset):
        for i in range(0, BLOCK * 64, 8):
            idx = tl.program_id(2) * 8 + i
            off = s_offset + idx
            val = tl.load(base_addr + off, mask=off < (s_offset + BLOCK * 64), other=0)
            so0[idx] = val.to(tl.uint32)
            
    def load_o_1(base_addr, s_offset):
        for i in range(0, BLOCK * 64, 8):
            idx = tl.program_id(2) * 8 + i
            off = s_offset + idx
            val = tl.load(base_addr + off, mask=off < (s_offset + BLOCK * 64), other=0)
            so1[idx] = val.to(tl.uint32)
            
    def load_do(base_addr, s_offset):
        for i in range(0, BLOCK * 64, 8):
            idx = tl.program_id(2) * 8 + i
            off = s_offset + idx
            val = tl.load(base_addr + off, mask=off < (s_offset + BLOCK * 64), other=0)
            sdo0[idx] = val.to(tl.uint32)
            
    def load_do_1(base_addr, s_offset):
        for i in range(0, BLOCK * 64, 8):
            idx = tl.program_id(2) * 8 + i
            off = s_offset + idx
            val = tl.load(base_addr + off, mask=off < (s_offset + BLOCK * 64), other=0)
            sdo1[idx] = val.to(tl.uint32)
            
    def load_k(base_addr, s_offset):
        for i in range(0, BLOCK * 64, 8):
            idx = tl.program_id(2) * 8 + i
            off = s_offset + idx
            val = tl.load(base_addr + off, mask=off < (s_offset + BLOCK * 64), other=0)
            sk0[idx] = val.to(tl.uint32)
            
    def load_k_1(base_addr, s_offset):
        for i in range(0, BLOCK * 64, 8):
            idx = tl.program_id(2) * 8 + i
            off = s_offset + idx
            val = tl.load(base_addr + off, mask=off < (s_offset + BLOCK * 64), other=0)
            sk1[idx] = val.to(tl.uint32)
            
    def load_v(base_addr, s_offset):
        for i in range(0, BLOCK * 64, 8):
            idx = tl.program_id(2) * 8 + i
            off = s_offset + idx
            val = tl.load(base_addr + off, mask=off < (s_offset + BLOCK * 64), other=0)
            sv0[idx] = val.to(tl.uint32)
            
    def load_v_1(base_addr, s_offset):
        for i in range(0, BLOCK * 64, 8):
            idx = tl.program_id(2) * 8 + i
            off = s_offset + idx
            val = tl.load(base_addr + off, mask=off < (s_offset + BLOCK * 64), other=0)
            sv1[idx] = val.to(tl.uint32)

    if i_start < S_len:
        load_q(Q_ptr, i_start * 64)
        load_q_1(Q_ptr, i_start * 64 + 64)
        load_o(O_ptr, i_start * 64)
        load_o_1(O_ptr, i_start * 64 + 64)
        load_do(dO_ptr, i_start * 64)
        load_do_1(dO_ptr, i_start * 64 + 64)
    
    ptr_O0 = s_O0
    o0 = ptr_O0[(i_start//BLOCK * 64 + row_offsets_128) * 64 + col_offsets_64]
    
    ptr_O1 = s_O1
    o1 = ptr_O1[(i_start//BLOCK * 64 + row_offsets_128) * 64 + col_offsets_64]
    
    ptr_dO0 = s_dO0
    do0 = ptr_dO0[(i_start//BLOCK * 64 + row_offsets_128) * 64 + col_offsets_64]
    
    ptr_dO1 = s_dO1
    do1 = ptr_dO1[(i_start//BLOCK * 64 + row_offsets_128) * 64 + col_offsets_64]
    
    d = o0 * do0 + o1 * do1
    
    d_sum = tl.sum(d, axis=1)
    d_sum = d_sum[:, None]
    
    s_D[(i_start//BLOCK * BLOCK + row_offsets_128)] = d_sum.squeeze(1)
    s_L[(i_start//BLOCK * BLOCK + row_offsets_128)] = tl.load(L_ptr + head_idx * HEAD_STRIDE + i_start + row_offsets_128, mask=(i_start + row_offsets_128) < S_len, other=0.0)
    
    dQ0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dQ1_acc = tl.zeros((BLOCK, 64), tl.float32)
    
    for j_start in range(0, S_len, BLOCK):
        load_k(K_ptr, j_start * 64)
        load_k_1(K_ptr, j_start * 64 + 64)
        load_v(V_ptr, j_start * 64)
        load_v_1(V_ptr, j_start * 64 + 64)
        
        ptr_K0 = s_K0
        k0 = ptr_K0[(j_start//BLOCK * 64 + row_offsets_128) * 64 + col_offsets_64]
        
        ptr_K1 = s_K1
        k1 = ptr_K1[(j_start//BLOCK * 64 + row_offsets_128) * 64 + col_offsets_64]
        
        ptr_V0 = s_V0
        v0 = ptr_V0[(j_start//BLOCK * 64 + row_offsets_128) * 64 + col_offsets_64]
        
        ptr_V1 = s_V1
        v1 = ptr_V1[(j_start//BLOCK * 64 + row_offsets_128) * 64 + col_offsets_64]
        
        ptr_Q0 = s_Q0
        q0 = ptr_Q0[(i_start//BLOCK * 64 + row_offsets_128) * 64 + col_offsets_64]
        
        ptr_Q1 = s_Q1
        q1 = ptr_Q1[(i_start//BLOCK * 64 + row_offsets_128) * 64 + col_offsets_64]
        
        S = tl.dot(q0, k0) + tl.dot(q1, k1)
        
        l_val = s_L[(i_start//BLOCK * BLOCK + row_offsets_128)]
        P = tl.exp(S * SCALE - l_val[:, None])
        
        dP = tl.dot(do0, v0) + tl.dot(do1, v1)
        
        D_i = s_D[(i_start//BLOCK * BLOCK + row_offsets_128)]
        
        dS = P * (dP - D_i[:, None]) * SCALE
        
        dQ0_acc += tl.dot(dS, k0)
        dQ1_acc += tl.dot(dS, k1)
        
    if i_start < S_len:
        ptr_dQ0 = dQ_ptr + head_idx * HEAD_STRIDE + i_start * stride_m
        ptr_dQ1 = dQ_ptr + head_idx * HEAD_STRIDE + i_start * stride_m + 64
        valid = (i_start + row_offsets_128) < S_len
        tl.store(ptr_dQ0 + row_offsets_128[:, None] * stride_m + col_offsets_64[None, :] * stride_d, dQ0_acc.to(tl.bfloat16), mask=valid[:, None])
        tl.store(ptr_dQ1 + row_offsets_128[:, None] * stride_m + col_offsets_64[None, :] * stride_d, dQ1_acc.to(tl.bfloat16), mask=valid[:, None])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
    ],
    key=["S_len"],
)
@triton.jit
def bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S_len, stride_m, stride_d, HEAD_STRIDE,
    BLOCK: tl.constexpr, SCALE: tl.constexpr,
):
    j_start = tl.program_id(0) * BLOCK
    head_idx = tl.program_id(1)
    
    row_offsets_128 = tl.arange(0, BLOCK)
    col_offsets_64 = tl.arange(0, 64)
    
    sq0 = __shared__ array('sq0', BLOCK * 64)
    sq1 = __shared__ array('sq1', BLOCK * 64)
    sk0 = __shared__ array('sk0', BLOCK * 64)
    sk1 = __shared__ array('sk1', BLOCK * 64)
    sv0 = __shared__ array('sv0', BLOCK * 64)
    sv1 = __shared__ array('sv1', BLOCK * 64)
    so0 = __shared__ array('so0', BLOCK * 64)
    so1 = __shared__ array('so1', BLOCK * 64)
    sdo0 = __shared__ array('sdo0', BLOCK * 64)
    sdo1 = __shared__ array('sdo1', BLOCK * 64)
    
    s_Q0 = extern_shared_array("s_Q0", BLOCK * 64 * 2)
    s_Q1 = extern_shared_array("s_Q1", BLOCK * 64 * 2)
    s_K0 = extern_shared_array("s_K0", BLOCK * 64 * 2)
    s_K1 = extern_shared_array("s_K1", BLOCK * 64 * 2)
    s_V0 = extern_shared_array("s_V0", BLOCK * 64 * 2)
    s_V1 = extern_shared_array("s_V1", BLOCK * 64 * 2)
    s_O0 = extern_shared_array("s_O0", BLOCK * 64 * 2)
    s_O1 = extern_shared_array("s_O1", BLOCK * 64 * 2)
    s_dO0 = extern_shared_array("s_dO0", BLOCK * 64 * 2)
    s_dO1 = extern_shared_array("s_dO1", BLOCK * 64 * 2)
    s_D = extern_shared_array("s_D", BLOCK * 4)
    s_L = extern_shared_array("s_L", BLOCK * 4)
    
    def load_q(base_addr, s_offset):
        for i in range(0, BLOCK * 64, 8):
            idx = tl.program_id(2) * 8 + i
            off = s_offset + idx
            val = tl.load(base_addr + off, mask=off < (s_offset + BLOCK * 64), other=0)
            sq0[idx] = val.to(tl.uint32)
            
    def load_q_1(base_addr, s_offset):
        for i in range(0, BLOCK * 64, 8):
            idx = tl.program_id(2) * 8 + i
            off = s_offset + idx
            val = tl.load(base_addr + off, mask=off < (s_offset + BLOCK * 64), other=0)
            sq1[idx] = val.to(tl.uint32)
            
    def load_o(base_addr, s_offset):
        for i in range(0, BLOCK * 64, 8):
            idx = tl.program_id(2) * 8 + i
            off = s_offset + idx
            val = tl.load(base_addr + off, mask=off < (s_offset + BLOCK * 64), other=0)
            so0[idx] = val.to(tl.uint32)
            
    def load_o_1(base_addr, s_offset):
        for i in range(0, BLOCK * 64, 8):
            idx = tl.program_id(2) * 8 + i
            off = s_offset + idx
            val = tl.load(base_addr + off, mask=off < (s_offset + BLOCK * 64), other=0)
            so1[idx] = val.to(tl.uint32)
            
    def load_do(base_addr, s_offset):
        for i in range(0, BLOCK * 64, 8):
            idx = tl.program_id(2) * 8 + i
            off = s_offset + idx
            val = tl.load(base_addr + off, mask=off < (s_offset + BLOCK * 64), other=0)
            sdo0[idx] = val.to(tl.uint32)
            
    def load_do_1(base_addr, s_offset):
        for i in range(0, BLOCK * 64, 8):
            idx = tl.program_id(2) * 8 + i
            off = s_offset + idx
            val = tl.load(base_addr + off, mask=off < (s_offset + BLOCK * 64), other=0)
            sdo1[idx] = val.to(tl.uint32)
            
    def load_k(base_addr, s_offset):
        for i in range(0, BLOCK * 64, 8):
            idx = tl.program_id(2) * 8 + i
            off = s_offset + idx
            val = tl.load(base_addr + off, mask=off < (s_offset + BLOCK * 64), other=0)
            sk0[idx] = val.to(tl.uint32)
            
    def load_k_1(base_addr, s_offset):
        for i in range(0, BLOCK * 64, 8):
            idx = tl.program_id(2) * 8 + i
            off = s_offset + idx
            val = tl.load(base_addr + off, mask=off < (s_offset + BLOCK * 64), other=0)
            sk1[idx] = val.to(tl.uint32)
            
    def load_v(base_addr, s_offset):
        for i in range(0, BLOCK * 64, 8):
            idx = tl.program_id(2) * 8 + i
            off = s_offset + idx
            val = tl.load(base_addr + off, mask=off < (s_offset + BLOCK * 64), other=0)
            sv0[idx] = val.to(tl.uint32)
            
    def load_v_1(base_addr, s_offset):
        for i in range(0, BLOCK * 64, 8):
            idx = tl.program_id(2) * 8 + i
            off = s_offset + idx
            val = tl.load(base_addr + off, mask=off < (s_offset + BLOCK * 64), other=0)
            sv1[idx] = val.to(tl.uint32)

    if j_start < S_len:
        load_k(K_ptr, j_start * 64)
        load_k_1(K_ptr, j_start * 64 + 64)
        load_v(V_ptr, j_start * 64)
        load_v_1(V_ptr, j_start * 64 + 64)
    
    ptr_K0 = s_K0
    k0 = ptr_K0[(j_start//BLOCK * 64 + row_offsets_128) * 64 + col_offsets_64]
    
    ptr_K1 = s_K1
    k1 = ptr_K1[(j_start//BLOCK * 64 + row_offsets_128) * 64 + col_offsets_64]
    
    ptr_V0 = s_V0
    v0 = ptr_V0[(j_start//BLOCK * 64 + row_offsets_128) * 64 + col_offsets_64]
    
    ptr_V1 = s_V1
    v1 = ptr_V1[(j_start//BLOCK * 64 + row_offsets_128) * 64 + col_offsets_64]
    
    dK0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dK1_acc = tl.zeros((BLOCK, 64), tl.float32)
    dV0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dV1_acc = tl.zeros((BLOCK, 64), tl.float32)
    
    for i_start in range(0, S_len, BLOCK):
        
        load_q(Q_ptr, i_start * 64)
        load_q_1(Q_ptr, i_start * 64 + 64)
        load_o(O_ptr, i_start * 64)
        load_o_1(O_ptr, i_start * 64 + 64)
        load_do(dO_ptr, i_start * 64)
        load_do_1(dO_ptr, i_start * 64 + 64)
        
        next_i = i_start + BLOCK
        if next_i < S_len:
            load_q(Q_ptr, next_i * 64)
            load_q_1(Q_ptr, next_i * 64 + 64)
            load_o(O_ptr, next_i * 64)
            load_o_1(O_ptr, next_i * 64 + 64)
            load_do(dO_ptr, next_i * 64)
            load_do_1(dO_ptr, next_i * 64 + 64)
        
        ptr_O0 = s_O0
        o0 = ptr_O0[(i_start//BLOCK * 64 + row_offsets_128) * 64 + col_offsets_64]
        
        ptr_O1 = s_O1
        o1 = ptr_O1[(i_start//BLOCK * 64 + row_offsets_128) * 64 + col_offsets_64]
        
        ptr_dO0 = s_dO0
        do0 = ptr_dO0[(i_start//BLOCK * 64 + row_offsets_128) * 64 + col_offsets_64]
        
        ptr_dO1 = s_dO1
        do1 = ptr_dO1[(i_start//BLOCK * 64 + row_offsets_128) * 64 + col_offsets_64]
        
        d = o0 * do0 + o1 * do1
        
        d_sum = tl.sum(d, axis=1)
        d_sum = d_sum[:, None]
        
        s_D[(i_start//BLOCK * BLOCK + row_offsets_128)] = d_sum.squeeze(1)
        s_L[(i_start//BLOCK * BLOCK + row_offsets_128)] = tl.load(L_ptr + head_idx * HEAD_STRIDE + i_start + row_offsets_128, mask=(i_start + row_offsets_128) < S_len, other=0.0)
        
        ptr_Q0 = s_Q0
        q0 = ptr_Q0[(i_start//BLOCK * 64 + row_offsets_128) * 64 + col_offsets_64]
        
        ptr_Q1 = s_Q1
        q1 = ptr_Q1[(i_start//BLOCK * 64 + row_offsets_128) * 64 + col_offsets_64]
        
        S = tl.dot(q0, k0) + tl.dot(q1, k1)
        
        l_val = s_L[(i_start//BLOCK * BLOCK + row_offsets_128)]
        P = tl.exp(S * SCALE - l_val[:, None])
        
        dP = tl.dot(do0, v0) + tl.dot(do1, v1)
        
        D_i = s_D[(i_start//BLOCK * BLOCK + row_offsets_128)]
        
        dS = P * (dP - D_i[:, None]) * SCALE
        
        dK0_acc += tl.dot(dS.T, q0)
        dK1_acc += tl.dot(dS.T, q1)
        
        dV0_acc += tl.dot(P.T, do0)
        dV1_acc += tl.dot(P.T, do1)
        
    if j_start < S_len:
        ptr_dK0 = dK_ptr + head_idx * HEAD_STRIDE + j_start * stride_m
        ptr_dK1 = dK_ptr + head_idx * HEAD_STRIDE + j_start * stride_m + 64
        valid = (j_start + row_offsets_128) < S_len
        tl.store(ptr_dK0 + row_offsets_128[:, None] * stride_m + col_offsets_64[None, :] * stride_d, dK0_acc.to(tl.bfloat16), mask=valid[:, None])
        tl.store(ptr_dK1 + row_offsets_128[:, None] * stride_m + col_offsets_64[None, :] * stride_d, dK1_acc.to(tl.bfloat16), mask=valid[:, None])
        
        ptr_dV0 = dV_ptr + head_idx * HEAD_STRIDE + j_start * stride_m
        ptr_dV1 = dV_ptr + head_idx * HEAD_STRIDE + j_start * stride_m + 64
        tl.store(ptr_dV0 + row_offsets_128[:, None] * stride_m + col_offsets_64[None, :] * stride_d, dV0_acc.to(tl.bfloat16), mask=valid[:, None])
        tl.store(ptr_dV1 + row_offsets_128[:, None] * stride_m + col_offsets_64[None, :] * stride_d, dV1_acc.to(tl.bfloat16), mask=valid[:, None])


NUM_WARPS = 4
NUM_STAGES = 2

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    b, h, s_len, d = Q.shape
    grid = (triton.cdiv(s_len, 256), b * h, NUM_WARPS)
    
    scale = 1.0 / math.sqrt(d)
    
    bwd_dq_kernel[grid](
        Q, K, V, O, dO, L, dQ,
        s_len, Q.stride(2), Q.stride(3), Q.stride(1),
        SCALE=scale,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    
    bwd_dkv_kernel[grid](
        Q, K, V, O, dO, L, dK, dV,
        s_len, K.stride(2), K.stride(3), K.stride(1),
        SCALE=scale,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )