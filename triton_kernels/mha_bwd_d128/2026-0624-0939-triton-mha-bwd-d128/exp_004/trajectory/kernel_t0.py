import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _dQ_kernel(
    q_desc,
    k_desc,
    v_desc,
    do_desc,
    o_desc,
    l_desc,
    dq_desc,
    S,
    tau,
    H,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    i = tl.program_id(0)
    offset_i = i * BLOCK_S
    row_offset_i = b * H * S + h * S + offset_i

    Q0 = q_desc.load([row_offset_i, 0])
    Q1 = q_desc.load([row_offset_i, 64])
    dO0 = do_desc.load([row_offset_i, 0])
    dO1 = do_desc.load([row_offset_i, 64])
    O0 = o_desc.load([row_offset_i, 0])
    O1 = o_desc.load([row_offset_i, 64])

    D = tl.sum(dO0 * O0 + dO1 * O1, axis=1)
    L = l_desc.load([row_offset_i])

    seq_idx_q = offset_i + tl.arange(0, BLOCK_S)
    q_mask = (seq_idx_q < S)[:, None]

    acc_dQ0 = tl.zeros((BLOCK_S, 64), tl.float32)
    acc_dQ1 = tl.zeros((BLOCK_S, 64), tl.float32)

    num_k_tiles = tl.cdiv(S, BLOCK_S)

    for j in tl.range(num_k_tiles, num_stages=2):
        offset_j = j * BLOCK_S
        row_offset_j = b * H * S + h * S + offset_j

        K0 = k_desc.load([row_offset_j, 0])
        K1 = k_desc.load([row_offset_j, 64])
        V0 = v_desc.load([row_offset_j, 0])
        V1 = v_desc.load([row_offset_j, 64])

        seq_idx_kv = offset_j + tl.arange(0, BLOCK_S)
        kv_mask = (seq_idx_kv < S)[None, :]

        S_mat = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)

        P = tl.exp(S_mat * tau - L[:, None])
        P = P * q_mask * kv_mask

        dS = P * (dP - D[:, None]) * tau
        dS = dS * q_mask * kv_mask

        acc_dQ0 = tl.dot(dS, K0, acc_dQ0)
        acc_dQ1 = tl.dot(dS, K1, acc_dQ1)

    dq_desc.store([row_offset_i, 0], acc_dQ0)
    dq_desc.store([row_offset_i, 64], acc_dQ1)


@triton.jit
def _dK_dV_kernel(
    q_desc,
    k_desc,
    v_desc,
    do_desc,
    o_desc,
    l_desc,
    dk_desc,
    dv_desc,
    S,
    tau,
    H,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    j = tl.program_id(0)
    offset_j = j * BLOCK_S
    row_offset_j = b * H * S + h * S + offset_j

    K0 = k_desc.load([row_offset_j, 0])
    K1 = k_desc.load([row_offset_j, 64])
    V0 = v_desc.load([row_offset_j, 0])
    V1 = v_desc.load([row_offset_j, 64])

    acc_dK0 = tl.zeros((BLOCK_S, 64), tl.float32)
    acc_dK1 = tl.zeros((BLOCK_S, 64), tl.float32)
    acc_dV0 = tl.zeros((BLOCK_S, 64), tl.float32)
    acc_dV1 = tl.zeros((BLOCK_S, 64), tl.float32)

    seq_idx_kv = offset_j + tl.arange(0, BLOCK_S)
    kv_mask = (seq_idx_kv < S)[None, :]

    num_q_tiles = tl.cdiv(S, BLOCK_S)

    for i in tl.range(num_q_tiles, num_stages=2):
        offset_i = i * BLOCK_S
        row_offset_i = b * H * S + h * S + offset_i

        Q0 = q_desc.load([row_offset_i, 0])
        Q1 = q_desc.load([row_offset_i, 64])
        dO0 = do_desc.load([row_offset_i, 0])
        dO1 = do_desc.load([row_offset_i, 64])
        O0 = o_desc.load([row_offset_i, 0])
        O1 = o_desc.load([row_offset_i, 64])

        D = tl.sum(dO0 * O0 + dO1 * O1, axis=1)
        L = l_desc.load([row_offset_i])

        seq_idx_q = offset_i + tl.arange(0, BLOCK_S)
        q_mask = (seq_idx_q < S)[:, None]

        S_mat = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)

        P = tl.exp(S_mat * tau - L[:, None])
        P = P * q_mask * kv_mask

        dS = P * (dP - D[:, None]) * tau
        dS = dS * q_mask * kv_mask

        acc_dV0 = tl.dot(P.T, dO0, acc_dV0)
        acc_dV1 = tl.dot(P.T, dO1, acc_dV1)
        acc_dK0 = tl.dot(dS.T, Q0, acc_dK0)
        acc_dK1 = tl.dot(dS.T, Q1, acc_dK1)

    dk_desc.store([row_offset_j, 0], acc_dK0)
    dk_desc.store([row_offset_j, 64], acc_dK1)
    dv_desc.store([row_offset_j, 0], acc_dV0)
    dv_desc.store([row_offset_j, 64], acc_dV1)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape

    tau = 1.0 / (d ** 0.5)
    BLOCK_S = 64

    q_2d = Q.reshape(-1, d)
    k_2d = K.reshape(-1, d)
    v_2d = V.reshape(-1, d)
    o_2d = O.reshape(-1, d)
    do_2d = dO.reshape(-1, d)
    l_1d = L.reshape(-1)

    dq_2d = dQ.reshape(-1, d)
    dk_2d = dK.reshape(-1, d)
    dv_2d = dV.reshape(-1, d)

    q_desc = TensorDescriptor.from_tensor(q_2d, [BLOCK_S, 64])
    k_desc = TensorDescriptor.from_tensor(k_2d, [BLOCK_S, 64])
    v_desc = TensorDescriptor.from_tensor(v_2d, [BLOCK_S, 64])
    o_desc = TensorDescriptor.from_tensor(o_2d, [BLOCK_S, 64])
    do_desc = TensorDescriptor.from_tensor(do_2d, [BLOCK_S, 64])
    l_desc = TensorDescriptor.from_tensor(l_1d, [BLOCK_S])

    dq_desc = TensorDescriptor.from_tensor(dq_2d, [BLOCK_S, 64])
    dk_desc = TensorDescriptor.from_tensor(dk_2d, [BLOCK_S, 64])
    dv_desc = TensorDescriptor.from_tensor(dv_2d, [BLOCK_S, 64])

    grid = (triton.cdiv(S, BLOCK_S), H, B)

    _dQ_kernel[grid](
        q_desc,
        k_desc,
        v_desc,
        do_desc,
        o_desc,
        l_desc,
        dq_desc,
        S,
        tau,
        H,
        BLOCK_S=BLOCK_S,
        num_warps=4,
        num_stages=2,
    )

    _dK_dV_kernel[grid](
        q_desc,
        k_desc,
        v_desc,
        do_desc,
        o_desc,
        l_desc,
        dk_desc,
        dv_desc,
        S,
        tau,
        H,
        BLOCK_S=BLOCK_S,
        num_warps=4,
        num_stages=2,
    )