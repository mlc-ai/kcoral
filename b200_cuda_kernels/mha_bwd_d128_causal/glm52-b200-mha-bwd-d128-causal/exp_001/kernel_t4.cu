#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace mha_bwd_d128_causal {

constexpr int BQ = 64;
constexpr int BK = 64;
constexpr int D = 128;
constexpr int WARPS = 8;
constexpr int THREADS = 256;
constexpr float SCALE = 0.08838834764831843f;
constexpr int B_CONST = 4;
constexpr int H_CONST = 48;

constexpr int S_M_TILES = BQ / 16;
constexpr int S_N_TILES = BK / 16;
constexpr int S_K_TILES = D / 16;
constexpr int S_TILES = S_M_TILES * S_N_TILES;
constexpr int S_TPW = (S_TILES + WARPS - 1) / WARPS;

constexpr int DQ_M_TILES = BQ / 16;
constexpr int DQ_N_TILES = D / 16;
constexpr int DQ_TILES = DQ_M_TILES * DQ_N_TILES;
constexpr int DQ_TPW = (DQ_TILES + WARPS - 1) / WARPS;

__global__ void dQ_dK_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ_out,
    float* __restrict__ dK_ws,
    int S)
{
    int bh = blockIdx.x;
    int b = bh / H_CONST;
    int h = bh % H_CONST;
    int q_blk = blockIdx.y;
    int q_start = q_blk * BQ;
    int tid = threadIdx.x;
    int warp_id = tid / 32;

    int64_t head_offset = (int64_t)b * H_CONST * S * D + (int64_t)h * S * D;
    int64_t lse_offset = (int64_t)b * H_CONST * S + (int64_t)h * S;

    const __nv_bfloat16* Q_base = Q + head_offset;
    const __nv_bfloat16* K_base = K + head_offset;
    const __nv_bfloat16* V_base = V + head_offset;
    const __nv_bfloat16* O_base = O + head_offset;
    const __nv_bfloat16* dO_base = dO + head_offset;
    const float* L_base = L + lse_offset;
    float* dK_base = dK_ws + head_offset;
    __nv_bfloat16* dQ_base = dQ_out + head_offset;

    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem = (__nv_bfloat16*)smem;
    __nv_bfloat16* dO_smem = Q_smem + BQ * D;
    __nv_bfloat16* K_smem = dO_smem + BQ * D;
    __nv_bfloat16* V_smem = K_smem + BK * D;
    float* P_smem = (float*)(V_smem + BK * D);
    __nv_bfloat16* dS_bf16 = (__nv_bfloat16*)(P_smem + BQ * BK);
    float* dQ_smem = (float*)(dS_bf16 + BQ * BK);
    float* D_smem = dQ_smem + BQ * D;

    for (int i = tid; i < BQ * D / 8; i += THREADS) {
        int row = i / (D / 8), col8 = i % (D / 8);
        int gidx = q_start + row;
        if (gidx < S) {
            *reinterpret_cast<int4*>(&Q_smem[row * D + col8 * 8]) =
                *reinterpret_cast<const int4*>(&Q_base[(int64_t)gidx * D + col8 * 8]);
            *reinterpret_cast<int4*>(&dO_smem[row * D + col8 * 8]) =
                *reinterpret_cast<const int4*>(&dO_base[(int64_t)gidx * D + col8 * 8]);
        } else {
            *reinterpret_cast<int4*>(&Q_smem[row * D + col8 * 8]) = make_int4(0,0,0,0);
            *reinterpret_cast<int4*>(&dO_smem[row * D + col8 * 8]) = make_int4(0,0,0,0);
        }
    }
    __syncthreads();

    for (int i = tid; i < BQ; i += THREADS) {
        int gidx = q_start + i;
        if (gidx >= S) { D_smem[i] = 0.0f; continue; }
        float sum = 0.0f;
        for (int d = 0; d < D; d += 2) {
            __nv_bfloat162 o2 = *reinterpret_cast<const __nv_bfloat162*>(&O_base[(int64_t)gidx * D + d]);
            __nv_bfloat162 d2 = *reinterpret_cast<const __nv_bfloat162*>(&dO_smem[i * D + d]);
            float2 of = __bfloat1622float2(o2);
            float2 df = __bfloat1622float2(d2);
            sum += of.x * df.x + of.y * df.y;
        }
        D_smem[i] = sum;
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dq_frag[DQ_TPW];
    #pragma unroll
    for (int t = 0; t < DQ_TPW; t++)
        wmma::fill_fragment(dq_frag[t], 0.0f);

    int num_k_blocks = (q_start + BQ + BK - 1) / BK;
    for (int k_blk = 0; k_blk < num_k_blocks; k_blk++) {
        int k_start = k_blk * BK;

        for (int i = tid; i < BK * D / 8; i += THREADS) {
            int row = i / (D / 8), col8 = i % (D / 8);
            int gidx = k_start + row;
            if (gidx < S) {
                *reinterpret_cast<int4*>(&K_smem[row * D + col8 * 8]) =
                    *reinterpret_cast<const int4*>(&K_base[(int64_t)gidx * D + col8 * 8]);
                *reinterpret_cast<int4*>(&V_smem[row * D + col8 * 8]) =
                    *reinterpret_cast<const int4*>(&V_base[(int64_t)gidx * D + col8 * 8]);
            } else {
                *reinterpret_cast<int4*>(&K_smem[row * D + col8 * 8]) = make_int4(0,0,0,0);
                *reinterpret_cast<int4*>(&V_smem[row * D + col8 * 8]) = make_int4(0,0,0,0);
            }
        }
        __syncthreads();

        // S = Q @ K^T
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[S_TPW];
            #pragma unroll
            for (int t = 0; t < S_TPW; t++) wmma::fill_fragment(c_frag[t], 0.0f);
            #pragma unroll
            for (int t = 0; t < S_TPW; t++) {
                int tile_idx = warp_id * S_TPW + t;
                int tm = tile_idx / S_N_TILES, tn = tile_idx % S_N_TILES;
                #pragma unroll
                for (int k = 0; k < S_K_TILES; k++) {
                    wmma::load_matrix_sync(a_frag, Q_smem + tm * 16 * D + k * 16, D);
                    wmma::load_matrix_sync(b_frag, K_smem + tn * 16 * D + k * 16, D);
                    wmma::mma_sync(c_frag[t], a_frag, b_frag, c_frag[t]);
                }
            }
            #pragma unroll
            for (int t = 0; t < S_TPW; t++) {
                int tile_idx = warp_id * S_TPW + t;
                int tm = tile_idx / S_N_TILES, tn = tile_idx % S_N_TILES;
                wmma::store_matrix_sync(P_smem + tm * 16 * BK + tn * 16, c_frag[t], BK, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // P = exp(S * scale - L) with causal mask
        for (int i = tid; i < BQ * BK; i += THREADS) {
            int row = i / BK, col = i % BK;
            int q_idx = q_start + row, k_idx = k_start + col;
            if (q_idx >= S || k_idx >= S || k_idx > q_idx)
                P_smem[i] = 0.0f;
            else
                P_smem[i] = expf(P_smem[i] * SCALE - L_base[q_idx]);
        }
        __syncthreads();

        // dP = dO @ V^T
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[S_TPW];
            #pragma unroll
            for (int t = 0; t < S_TPW; t++) wmma::fill_fragment(c_frag[t], 0.0f);
            #pragma unroll
            for (int t = 0; t < S_TPW; t++) {
                int tile_idx = warp_id * S_TPW + t;
                int tm = tile_idx / S_N_TILES, tn = tile_idx % S_N_TILES;
                #pragma unroll
                for (int k = 0; k < S_K_TILES; k++) {
                    wmma::load_matrix_sync(a_frag, dO_smem + tm * 16 * D + k * 16, D);
                    wmma::load_matrix_sync(b_frag, V_smem + tn * 16 * D + k * 16, D);
                    wmma::mma_sync(c_frag[t], a_frag, b_frag, c_frag[t]);
                }
            }
            __syncthreads();
            float* dP_smem = (float*)V_smem;
            #pragma unroll
            for (int t = 0; t < S_TPW; t++) {
                int tile_idx = warp_id * S_TPW + t;
                int tm = tile_idx / S_N_TILES, tn = tile_idx % S_N_TILES;
                wmma::store_matrix_sync(dP_smem + tm * 16 * BK + tn * 16, c_frag[t], BK, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // dS = P * (dP - D) in-place
        {
            float* dP_smem = (float*)V_smem;
            for (int i = tid; i < BQ * BK; i += THREADS) {
                int row = i / BK;
                P_smem[i] = P_smem[i] * (dP_smem[i] - D_smem[row]);
            }
        }
        __syncthreads();

        // Convert dS to bf16
        for (int i = tid; i < BQ * BK; i += THREADS)
            dS_bf16[i] = __float2bfloat16(P_smem[i]);
        __syncthreads();

        // dQ += dS_bf16 @ K
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
            #pragma unroll
            for (int t = 0; t < DQ_TPW; t++) {
                int tile_idx = warp_id * DQ_TPW + t;
                int tm = tile_idx / DQ_N_TILES, tn = tile_idx % DQ_N_TILES;
                #pragma unroll
                for (int k = 0; k < BK / 16; k++) {
                    wmma::load_matrix_sync(a_frag, dS_bf16 + tm * 16 * BK + k * 16, BK);
                    wmma::load_matrix_sync(b_frag, K_smem + k * 16 * D + tn * 16, D);
                    wmma::mma_sync(dq_frag[t], a_frag, b_frag, dq_frag[t]);
                }
            }
        }

        // dK = dS_bf16^T @ Q * SCALE
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> dk_frag[DQ_TPW];
            #pragma unroll
            for (int t = 0; t < DQ_TPW; t++) wmma::fill_fragment(dk_frag[t], 0.0f);
            #pragma unroll
            for (int t = 0; t < DQ_TPW; t++) {
                int tile_idx = warp_id * DQ_TPW + t;
                int tm = tile_idx / DQ_N_TILES, tn = tile_idx % DQ_N_TILES;
                #pragma unroll
                for (int k = 0; k < BQ / 16; k++) {
                    wmma::load_matrix_sync(a_frag, dS_bf16 + k * 16 * BK + tm * 16, BK);
                    wmma::load_matrix_sync(b_frag, Q_smem + k * 16 * D + tn * 16, D);
                    wmma::mma_sync(dk_frag[t], a_frag, b_frag, dk_frag[t]);
                }
            }
            #pragma unroll
            for (int t = 0; t < DQ_TPW; t++)
                #pragma unroll
                for (int i = 0; i < dk_frag[t].num_elements; i++)
                    dk_frag[t].x[i] *= SCALE;
            float* dK_temp = dQ_smem;
            #pragma unroll
            for (int t = 0; t < DQ_TPW; t++) {
                int tile_idx = warp_id * DQ_TPW + t;
                int tm = tile_idx / DQ_N_TILES, tn = tile_idx % DQ_N_TILES;
                wmma::store_matrix_sync(dK_temp + tm * 16 * D + tn * 16, dk_frag[t], D, wmma::mem_row_major);
            }
        }
        __syncthreads();

        float* dK_temp = dQ_smem;
        for (int i = tid; i < BK * D; i += THREADS) {
            int row = i / D, col = i % D;
            int k_idx = k_start + row;
            if (k_idx < S && dK_temp[i] != 0.0f)
                atomicAdd(&dK_base[(int64_t)k_idx * D + col], dK_temp[i]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int t = 0; t < DQ_TPW; t++)
        #pragma unroll
        for (int i = 0; i < dq_frag[t].num_elements; i++)
            dq_frag[t].x[i] *= SCALE;

    #pragma unroll
    for (int t = 0; t < DQ_TPW; t++) {
        int tile_idx = warp_id * DQ_TPW + t;
        int tm = tile_idx / DQ_N_TILES, tn = tile_idx % DQ_N_TILES;
        wmma::store_matrix_sync(dQ_smem + tm * 16 * D + tn * 16, dq_frag[t], D, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = tid; i < BQ * D / 8; i += THREADS) {
        int row = i / (D / 8), col8 = i % (D / 8);
        int q_idx = q_start + row;
        if (q_idx < S) {
            float* src = &dQ_smem[row * D + col8 * 8];
            __nv_bfloat16* dst = &dQ_base[(int64_t)q_idx * D + col8 * 8];
            __nv_bfloat162 v0 = __floats2bfloat162_rn(src[0], src[1]);
            __nv_bfloat162 v1 = __floats2bfloat162_rn(src[2], src[3]);
            __nv_bfloat162 v2 = __floats2bfloat162_rn(src[4], src[5]);
            __nv_bfloat162 v3 = __floats2bfloat162_rn(src[6], src[7]);
            *reinterpret_cast<int4*>(dst) = make_int4(
                *reinterpret_cast<int*>(&v0), *reinterpret_cast<int*>(&v1),
                *reinterpret_cast<int*>(&v2), *reinterpret_cast<int*>(&v3));
        }
    }
}

__global__ void dV_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dV_out,
    int S)
{
    int bh = blockIdx.x;
    int b = bh / H_CONST;
    int h = bh % H_CONST;
    int k_blk = blockIdx.y;
    int k_start = k_blk * BK;
    int tid = threadIdx.x;
    int warp_id = tid / 32;

    int64_t head_offset = (int64_t)b * H_CONST * S * D + (int64_t)h * S * D;
    int64_t lse_offset = (int64_t)b * H_CONST * S + (int64_t)h * S;

    const __nv_bfloat16* Q_base = Q + head_offset;
    const __nv_bfloat16* K_base = K + head_offset;
    const __nv_bfloat16* dO_base = dO + head_offset;
    const float* L_base = L + lse_offset;
    __nv_bfloat16* dV_base = dV_out + head_offset;

    extern __shared__ char smem[];
    __nv_bfloat16* K_smem = (__nv_bfloat16*)smem;
    __nv_bfloat16* Q_smem = K_smem + BK * D;
    __nv_bfloat16* dO_smem = Q_smem + BQ * D;
    float* P_smem = (float*)(dO_smem + BQ * D);
    __nv_bfloat16* P_bf16 = (__nv_bfloat16*)(P_smem + BQ * BK);
    float* dV_smem = (float*)(P_bf16 + BQ * BK);

    for (int i = tid; i < BK * D / 8; i += THREADS) {
        int row = i / (D / 8), col8 = i % (D / 8);
        int gidx = k_start + row;
        if (gidx < S) {
            *reinterpret_cast<int4*>(&K_smem[row * D + col8 * 8]) =
                *reinterpret_cast<const int4*>(&K_base[(int64_t)gidx * D + col8 * 8]);
        } else {
            *reinterpret_cast<int4*>(&K_smem[row * D + col8 * 8]) = make_int4(0,0,0,0);
        }
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dv_frag[DQ_TPW];
    #pragma unroll
    for (int t = 0; t < DQ_TPW; t++) wmma::fill_fragment(dv_frag[t], 0.0f);

    int num_q_blocks = (S + BQ - 1) / BQ;
    for (int q_blk = k_blk; q_blk < num_q_blocks; q_blk++) {
        int q_start = q_blk * BQ;

        for (int i = tid; i < BQ * D / 8; i += THREADS) {
            int row = i / (D / 8), col8 = i % (D / 8);
            int gidx = q_start + row;
            if (gidx < S) {
                *reinterpret_cast<int4*>(&Q_smem[row * D + col8 * 8]) =
                    *reinterpret_cast<const int4*>(&Q_base[(int64_t)gidx * D + col8 * 8]);
                *reinterpret_cast<int4*>(&dO_smem[row * D + col8 * 8]) =
                    *reinterpret_cast<const int4*>(&dO_base[(int64_t)gidx * D + col8 * 8]);
            } else {
                *reinterpret_cast<int4*>(&Q_smem[row * D + col8 * 8]) = make_int4(0,0,0,0);
                *reinterpret_cast<int4*>(&dO_smem[row * D + col8 * 8]) = make_int4(0,0,0,0);
            }
        }
        __syncthreads();

        // S = Q @ K^T
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[S_TPW];
            #pragma unroll
            for (int t = 0; t < S_TPW; t++) wmma::fill_fragment(c_frag[t], 0.0f);
            #pragma unroll
            for (int t = 0; t < S_TPW; t++) {
                int tile_idx = warp_id * S_TPW + t;
                int tm = tile_idx / S_N_TILES, tn = tile_idx % S_N_TILES;
                #pragma unroll
                for (int k = 0; k < S_K_TILES; k++) {
                    wmma::load_matrix_sync(a_frag, Q_smem + tm * 16 * D + k * 16, D);
                    wmma::load_matrix_sync(b_frag, K_smem + tn * 16 * D + k * 16, D);
                    wmma::mma_sync(c_frag[t], a_frag, b_frag, c_frag[t]);
                }
            }
            #pragma unroll
            for (int t = 0; t < S_TPW; t++) {
                int tile_idx = warp_id * S_TPW + t;
                int tm = tile_idx / S_N_TILES, tn = tile_idx % S_N_TILES;
                wmma::store_matrix_sync(P_smem + tm * 16 * BK + tn * 16, c_frag[t], BK, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // P = exp(S * scale - L) with causal mask
        for (int i = tid; i < BQ * BK; i += THREADS) {
            int row = i / BK, col = i % BK;
            int q_idx = q_start + row, k_idx = k_start + col;
            if (q_idx >= S || k_idx >= S || k_idx > q_idx)
                P_smem[i] = 0.0f;
            else
                P_smem[i] = expf(P_smem[i] * SCALE - L_base[q_idx]);
        }
        __syncthreads();

        // Convert P to bf16
        for (int i = tid; i < BQ * BK; i += THREADS)
            P_bf16[i] = __float2bfloat16(P_smem[i]);
        __syncthreads();

        // dV += P_bf16^T @ dO
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
            #pragma unroll
            for (int t = 0; t < DQ_TPW; t++) {
                int tile_idx = warp_id * DQ_TPW + t;
                int tm = tile_idx / DQ_N_TILES, tn = tile_idx % DQ_N_TILES;
                #pragma unroll
                for (int k = 0; k < BQ / 16; k++) {
                    wmma::load_matrix_sync(a_frag, P_bf16 + k * 16 * BK + tm * 16, BK);
                    wmma::load_matrix_sync(b_frag, dO_smem + k * 16 * D + tn * 16, D);
                    wmma::mma_sync(dv_frag[t], a_frag, b_frag, dv_frag[t]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int t = 0; t < DQ_TPW; t++) {
        int tile_idx = warp_id * DQ_TPW + t;
        int tm = tile_idx / DQ_N_TILES, tn = tile_idx % DQ_N_TILES;
        wmma::store_matrix_sync(dV_smem + tm * 16 * D + tn * 16, dv_frag[t], D, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = tid; i < BK * D / 8; i += THREADS) {
        int row = i / (D / 8), col8 = i % (D / 8);
        int k_idx = k_start + row;
        if (k_idx < S) {
            float* src = &dV_smem[row * D + col8 * 8];
            __nv_bfloat16* dst = &dV_base[(int64_t)k_idx * D + col8 * 8];
            __nv_bfloat162 v0 = __floats2bfloat162_rn(src[0], src[1]);
            __nv_bfloat162 v1 = __floats2bfloat162_rn(src[2], src[3]);
            __nv_bfloat162 v2 = __floats2bfloat162_rn(src[4], src[5]);
            __nv_bfloat162 v3 = __floats2bfloat162_rn(src[6], src[7]);
            *reinterpret_cast<int4*>(dst) = make_int4(
                *reinterpret_cast<int*>(&v0), *reinterpret_cast<int*>(&v1),
                *reinterpret_cast<int*>(&v2), *reinterpret_cast<int*>(&v3));
        }
    }
}

__global__ void convert_to_bf16(const float* src, __nv_bfloat16* dst, int64_t n) {
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride)
        dst[idx] = __float2bfloat16(src[idx]);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t S = Q.size(2);
    int64_t total_elements = (int64_t)B_CONST * H_CONST * S * D;

    float* dK_ws = nullptr;
    CUDA_CHECK(cudaMalloc(&dK_ws, total_elements * sizeof(float)));
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    CUDA_CHECK(cudaMemsetAsync(dK_ws, 0, total_elements * sizeof(float), stream));

    const __nv_bfloat16* Q_ptr  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr          = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr       = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr       = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr       = static_cast<__nv_bfloat16*>(dV.data_ptr());

    int num_q_blocks = (int)((S + BQ - 1) / BQ);
    int num_k_blocks = (int)((S + BK - 1) / BK);

    {
        dim3 grid(B_CONST * H_CONST, num_q_blocks);
        dim3 block(THREADS);
        int smem_size = BQ * D * 2 * 2 + BK * D * 2 * 2 + BQ * BK * 4 + BQ * BK * 2 + BQ * D * 4 + BQ * 4;
        cudaFuncSetAttribute(dQ_dK_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        dQ_dK_kernel<<<grid, block, smem_size, stream>>>(
            Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr, dK_ws, (int)S);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        dim3 grid(B_CONST * H_CONST, num_k_blocks);
        dim3 block(THREADS);
        int smem_size = BK * D * 2 + BQ * D * 2 * 2 + BQ * BK * 4 + BQ * BK * 2 + BK * D * 4;
        cudaFuncSetAttribute(dV_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        dV_kernel<<<grid, block, smem_size, stream>>>(
            Q_ptr, K_ptr, dO_ptr, L_ptr, dV_ptr, (int)S);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        int ct = 256;
        int64_t cb = (total_elements + ct - 1) / ct;
        if (cb > 2147483647LL) cb = 2147483647LL;
        convert_to_bf16<<<(int)cb, ct, 0, stream>>>(dK_ws, dK_ptr, total_elements);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(dK_ws));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal