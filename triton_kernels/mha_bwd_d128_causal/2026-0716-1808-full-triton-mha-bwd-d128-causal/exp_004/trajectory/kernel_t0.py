import math
import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_dQ(
    Q, K, V, O, dO, L, dQ,
    S_len, S, stride_h, stride_s, stride_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """
    First backward kernel computing the gradient w.r.t. the queries (dQ).
    
    Logic:
    Each program instance owns a 16x16 query tile (i.e. 16 contiguous rows of Q within
    a particular attention head). It plays the role of the 'anchor' and sequentially 
    sweeps across every fully-legal causal key block j ranging from 0 up to i.
    
    For each key block j, the kernel:
      1. Issues synchronous global loads for the 16x128 tiles of K_j, V_j.
      2. Computes the raw attention logits S = Q_i @ K_j^T using a 128-wide tensor-core dot.
      3. Applies the scalar softmax normalization L (per query row) and performs a causal 
         element-wise mask (zero-padding illegal off-diagonal transitions to 0 to avert NaN propagation).
      4. Computes the upstream probability gradient dP = dO_i @ V_j^T.
      5. Synthesizes the causal numerically-scaled logit Jacobian dS = P * (dP - D) * scale.
      6. Accumulates the raw dQ update step via a rank-1 expansion dS @ K_j mapped over all 128 features.
      
    Boundary conditions:
      - Out-of-bounds rows beyond S in global memory loads utilize an `other=0.0` fallback 
        ensuring downstream reductions propagate cleanly without NaN leakage.
      - If a program's assigned query block i falls completely outside S limits, execution 
        terminates instantly via `return`.
    """
    i_block = tl.program_id(0)
    num_blocks_per_head = (S + BLOCK_M - 1) // BLOCK_M
    h = i_block // num_blocks_per_head
    i = i_block % num_blocks_per_head
    
    offset_m = i * BLOCK_M
    if offset_m >= S:
        return
    
    base_ptr = h * stride_h
    
    q_row = tl.arange(0, BLOCK_M)
    col = tl.arange(0, HEAD_DIM)
    
    q_load = tl.load(Q + base_ptr + (offset_m + q_row[:, None]) * stride_s + col[None, :] * stride_d, 
                      mask=(offset_m + q_row[:, None]) < S, other=0.0)
    o_load = tl.load(O + base_ptr + (offset_m + q_row[:, None]) * stride_s + col[None, :] * stride_d, 
                      mask=(offset_m + q_row[:, None]) < S, other=0.0)
    do_load = tl.load(dO + base_ptr + (offset_m + q_row[:, None]) * stride_s + col[None, :] * stride_d, 
                       mask=(offset_m + q_row[:, None]) < S, other=0.0)
    
    d_val = (do_load * o_load).sum(axis=1)
    
    l_load = tl.load(L + h * S + offset_m + q_row, mask=(offset_m + q_row) < S, other=0.0)
    
    acc_dQ = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    k_row = tl.arange(0, BLOCK_N)
    
    for j in range(i + 1):
        offset_n = j * BLOCK_N
        
        k_load = tl.load(K + base_ptr + (offset_n + k_row[:, None]) * stride_s + col[None, :] * stride_d, 
                          mask=(offset_n + k_row[:, None]) < S, other=0.0)
        v_load = tl.load(V + base_ptr + (offset_n + k_row[:, None]) * stride_s + col[None, :] * stride_d, 
                          mask=(offset_n + k_row[:, None]) < S, other=0.0)
        
        acc_S = tl.dot(q_load, k_load.T, tl.zeros((BLOCK_M, BLOCK_N), tl.float32))
        
        P = tl.exp(acc_S * scale - l_load[:, None])
        
        mask = ((i * BLOCK_M + q_row[:, None]) >= (j * BLOCK_N + k_row[None, :])) & ((j * BLOCK_N + k_row[None, :]) < S)
        P = tl.where(mask, P, 0.0)
        
        acc_dP = tl.dot(do_load, v_load.T, tl.zeros((BLOCK_M, BLOCK_N), tl.float32))
        
        dS = tl.where(mask, P * (acc_dP - d_val[:, None]) * scale, 0.0)
        
        acc_dQ = tl.dot(dS.to(tl.bfloat16), k_load, acc_dQ)
        
    out_ptr = dQ + base_ptr + (offset_m + q_row[:, None]) * stride_s + col[None, :] * stride_d
    tl.store(out_ptr, acc_dQ, mask=(offset_m + q_row[:, None]) < S)


@triton.jit
def _bwd_dK_dV(
    Q, K, V, O, dO, L, dK, dV,
    S_len, S, stride_h, stride_s, stride_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """
    Second backward kernel resolving gradients w.r.t. the memory bank keys and values (dK, dV).
    
    Logic:
    Each program instance operates on a specific 16x16 key/value block j for a designated 
    attention head. Acting as the static pivot, it cascades exclusively across legal 
    causal query blocks i ranging from j up to the sequence horizon limit (T_r - 1).
    
    Per query block i, the kernel dynamically resolves:
      1. Asynchronous fetches of the 16x128 outer tiles Q_i, O_i, and dO_i.
      2. Row-wise divergence correction scalar D_i = rowsum(dO_i * O_i).
      3. The standard attention map S = Q_i @ K_j^T followed by causal masking and exponentiation.
      4. The value-space projection gradient dP = dO_i @ V_j^T.
      5. A dual-branch GEMM epilogue pushing weighted updates into both dK_j and dV_j 
         utilizing transposed intermediate states.
         
    Boundary conditions:
      - Identical masking policy and semantic out-of-bounds nomenclature (`other=0.0`) as 
        `_bwd_dQ` protecting all tensor arithmetic from garbage inputs.
      - Early termination if the targeted key block j exceeds sequence extent limits.
    """
    j_block = tl.program_id(0)
    num_blocks_per_head = (S + BLOCK_N - 1) // BLOCK_N
    h = j_block // num_blocks_per_head
    j = j_block % num_blocks_per_head
    
    offset_n = j * BLOCK_N
    if offset_n >= S:
        return
    
    base_ptr = h * stride_h
    
    k_row = tl.arange(0, BLOCK_N)
    col = tl.arange(0, HEAD_DIM)
    
    k_load = tl.load(K + base_ptr + (offset_n + k_row[:, None]) * stride_s + col[None, :] * stride_d, 
                      mask=(offset_n + k_row[:, None]) < S, other=0.0)
    v_load = tl.load(V + base_ptr + (offset_n + k_row[:, None]) * stride_s + col[None, :] * stride_d, 
                      mask=(offset_n + k_row[:, None]) < S, other=0.0)
    
    acc_dK = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    acc_dV = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    q_row = tl.arange(0, BLOCK_M)
    
    for i in range(j, num_blocks_per_head):
        offset_m = i * BLOCK_M
        
        q_load = tl.load(Q + base_ptr + (offset_m + q_row[:, None]) * stride_s + col[None, :] * stride_d, 
                          mask=(offset_m + q_row[:, None]) < S, other=0.0)
        o_load = tl.load(O + base_ptr + (offset_m + q_row[:, None]) * stride_s + col[None, :] * stride_d, 
                          mask=(offset_m + q_row[:, None]) < S, other=0.0)
        do_load = tl.load(dO + base_ptr + (offset_m + q_row[:, None]) * stride_s + col[None, :] * stride_d, 
                           mask=(offset_m + q_row[:, None]) < S, other=0.0)
        
        d_val = (do_load * o_load).sum(axis=1)
        
        l_load = tl.load(L + h * S + offset_m + q_row, mask=(offset_m + q_row) < S, other=0.0)
        
        acc_S = tl.dot(q_load, k_load.T, tl.zeros((BLOCK_M, BLOCK_N), tl.float32))
        
        P = tl.exp(acc_S * scale - l_load[:, None])
        
        mask = ((i * BLOCK_M + q_row[:, None]) >= (j * BLOCK_N + k_row[None, :])) & ((j * BLOCK_N + k_row[None, :]) < S)
        P = tl.where(mask, P, 0.0)
        
        acc_dP = tl.dot(do_load, v_load.T, tl.zeros((BLOCK_M, BLOCK_N), tl.float32))
        
        dS = tl.where(mask, P * (acc_dP - d_val[:, None]) * scale, 0.0)
        
        acc_dV = tl.dot(P.to(tl.bfloat16).T, do_load, acc_dV)
        acc_dK = tl.dot(dS.to(tl.bfloat16).T, q_load, acc_dK)
        
    out_ptr_K = dK + base_ptr + (offset_n + k_row[:, None]) * stride_s + col[None, :] * stride_d
    tl.store(out_ptr_K, acc_dK, mask=(offset_n + k_row[:, None]) < S)
    
    out_ptr_V = dV + base_ptr + (offset_n + k_row[:, None]) * stride_s + col[None, :] * stride_d
    tl.store(out_ptr_V, acc_dV, mask=(offset_n + k_row[:, None]) < S)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Host bridge launching the two sequential Triton device-pass routines.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    S_len = B * H * S
    
    stride_b = H * S * d
    stride_h = S * d
    stride_s = d
    stride_d = 1
    
    grid = (num_blocks_per_head * B * H,)
    
    _bwd_dQ[grid](Q, K, V, O, dO, L, dQ, S_len, S, stride_h, stride_s, stride_d, BLOCK_M=16, BLOCK_N=16, HEAD_DIM=128)
    _bwd_dK_dV[grid](Q, K, V, O, dO, L, dK, dV, S_len, S, stride_h, stride_s, stride_d, BLOCK_M=16, BLOCK_N=16, HEAD_DIM=128)