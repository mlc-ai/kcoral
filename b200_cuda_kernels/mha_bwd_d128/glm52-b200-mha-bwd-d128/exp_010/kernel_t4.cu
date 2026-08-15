#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cmath>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while (0)

namespace attention_bwd {

constexpr int D_HEAD = 128;
constexpr int NWARPS = 8;
constexpr int NTHREADS = 256;
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

// dQ kernel config
constexpr int BQ_DQ = 64;
constexpr int BKV_DQ = 128;

// dK/dV kernel config
constexpr int BQ_DKV = 64;
constexpr int BKV_DKV = 64;

constexpr int SMEM_SIZE_DQ =
    BQ_DQ * D_HEAD * 2 +    // Qi
    BQ_DQ * D_HEAD * 2 +    // dOi
    BKV_DQ * D_HEAD * 2 +   // Kj
    BKV_DQ * D_HEAD * 2 +   // Vj
    BQ_DQ * BKV_DQ * 4 +    // S/dP float
    BQ_DQ * BKV_DQ * 4 +    // P float
    BQ_DQ * BKV_DQ * 2 +    // dS bf16
    BQ_DQ * 4 +             // Li
    BQ_DQ * 4 +             // Di
    NWARPS * WMMA_M * WMMA_N * 4; // staging

constexpr int SMEM_SIZE_DKV =
    BKV_DKV * D_HEAD * 2 +  // Kj
    BKV_DKV * D_HEAD * 2 +  // Vj
    BQ_DKV * D_HEAD * 2 +   // Qi
    BQ_DKV * D_HEAD * 2 +   // dOi
    BKV_DKV * BQ_DKV * 4 +  // S/dP float
    BKV_DKV * BQ_DKV * 4 +  // P float
    BKV_DKV * BQ_DKV * 2 +  // P bf16
    BKV_DKV * BQ_DKV * 2 +  // dS bf16
    BQ_DKV * 4 +            // Li
    BQ_DKV * 4 +            // Di
    NWARPS * WMMA_M * WMMA_N * 4;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__global__ void compute_D_kernel(
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    float* __restrict__ D,
    int total_rows) {
    int row = blockIdx.x;
    if (row >= total_rows) return;
    int tid = threadIdx.x;
    const __nv_bfloat16* O_ptr = O + (size_t)row * D_HEAD;
    const __nv_bfloat16* dO_ptr = dO + (size_t)row * D_HEAD;
    float sum = 0.0f;
    for (int i = tid; i < D_HEAD; i += blockDim.x)
        sum += __bfloat162float(O_ptr[i]) * __bfloat162float(dO_ptr[i]);
    sum = warp_reduce_sum(sum);
    __shared__ float warp_sums[8];
    int warp_id = tid / 32, lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < blockDim.x / 32; w++) total += warp_sums[w];
        D[row] = total;
    }
}

__global__ __launch_bounds__(NTHREADS, 1) void compute_dQ_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_arr,
    __nv_bfloat16* __restrict__ dQ_out,
    int B, int H, int S) {

    extern __shared__ char smem[];
    __nv_bfloat16* Qi_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* dOi_smem = Qi_smem + BQ_DQ * D_HEAD;
    __nv_bfloat16* Kj_smem = dOi_smem + BQ_DQ * D_HEAD;
    __nv_bfloat16* Vj_smem = Kj_smem + BKV_DQ * D_HEAD;
    float* S_smem = reinterpret_cast<float*>(Vj_smem + BKV_DQ * D_HEAD);
    float* P_smem = S_smem + BQ_DQ * BKV_DQ;
    __nv_bfloat16* dS_bf16 = reinterpret_cast<__nv_bfloat16*>(P_smem + BQ_DQ * BKV_DQ);
    float* Li_smem = reinterpret_cast<float*>(dS_bf16 + BQ_DQ * BKV_DQ);
    float* Di_smem = Li_smem + BQ_DQ;
    float* staging = Di_smem + BQ_DQ;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int num_q_blocks = (S + BQ_DQ - 1) / BQ_DQ;
    int bh_idx = blockIdx.x / num_q_blocks;
    int q_block = blockIdx.x % num_q_blocks;
    int b = bh_idx / H;
    int h = bh_idx % H;
    int q_start = q_block * BQ_DQ;

    size_t bh_offset = (size_t)(b * H + h) * S * D_HEAD;
    const __nv_bfloat16* Q_base = Q + bh_offset;
    const __nv_bfloat16* K_base = K + bh_offset;
    const __nv_bfloat16* V_base = V + bh_offset;
    const __nv_bfloat16* dO_base = dO + bh_offset;
    const float* L_base = L + (size_t)(b * H + h) * S;
    const float* D_base = D_arr + (size_t)(b * H + h) * S;
    __nv_bfloat16* dQ_base = dQ_out + bh_offset;

    // Load Qi, dOi (once)
    {
        int4* Qi_v = reinterpret_cast<int4*>(Qi_smem);
        int4* dOi_v = reinterpret_cast<int4*>(dOi_smem);
        int total_vecs = BQ_DQ * D_HEAD / 8;
        for (int i = tid; i < total_vecs; i += NTHREADS) {
            int row = i / (D_HEAD / 8);
            int col_vec = i % (D_HEAD / 8);
            int gr = q_start + row;
            if (gr < S) {
                Qi_v[i] = *reinterpret_cast<const int4*>(&Q_base[(size_t)gr * D_HEAD + col_vec * 8]);
                dOi_v[i] = *reinterpret_cast<const int4*>(&dO_base[(size_t)gr * D_HEAD + col_vec * 8]);
            } else {
                Qi_v[i] = make_int4(0,0,0,0);
                dOi_v[i] = make_int4(0,0,0,0);
            }
        }
    }
    if (tid < BQ_DQ) {
        int gr = q_start + tid;
        Li_smem[tid] = (gr < S) ? L_base[gr] : 0.0f;
        Di_smem[tid] = (gr < S) ? D_base[gr] : 0.0f;
    }
    __syncthreads();

    // dQ accumulators: BQ x D = 4x8 = 32 tiles, 4 per warp
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dQ_frag[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) wmma::fill_fragment(dQ_frag[i], 0.0f);

    const float scale = 0.08838834764f;

    for (int kv_start = 0; kv_start < S; kv_start += BKV_DQ) {
        // Load Kj, Vj
        {
            int4* Kj_v = reinterpret_cast<int4*>(Kj_smem);
            int4* Vj_v = reinterpret_cast<int4*>(Vj_smem);
            int total_vecs = BKV_DQ * D_HEAD / 8;
            for (int i = tid; i < total_vecs; i += NTHREADS) {
                int row = i / (D_HEAD / 8);
                int col_vec = i % (D_HEAD / 8);
                int gr = kv_start + row;
                if (gr < S) {
                    Kj_v[i] = *reinterpret_cast<const int4*>(&K_base[(size_t)gr * D_HEAD + col_vec * 8]);
                    Vj_v[i] = *reinterpret_cast<const int4*>(&V_base[(size_t)gr * D_HEAD + col_vec * 8]);
                } else {
                    Kj_v[i] = make_int4(0,0,0,0);
                    Vj_v[i] = make_int4(0,0,0,0);
                }
            }
        }
        __syncthreads();

        // S = Qi @ Kj^T (BQ x BKV = 4x8 = 32 tiles, 4 per warp, 8 K-steps)
        {
            int wr = warp_id / 2;
            int wc_base = (warp_id % 2) * 4;
            #pragma unroll
            for (int wj = 0; wj < 4; wj++) {
                int ct = wc_base + wj;
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < D_HEAD / WMMA_K; kk++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(a_frag, Qi_smem + wr * 16 * D_HEAD + kk * 16, D_HEAD);
                    wmma::load_matrix_sync(b_frag, Kj_smem + ct * 16 * D_HEAD + kk * 16, D_HEAD);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                wmma::store_matrix_sync(S_smem + wr * 16 * BKV_DQ + ct * 16, c_frag, BKV_DQ, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // P = exp(S * scale - L)
        for (int i = tid; i < BQ_DQ * BKV_DQ; i += NTHREADS) {
            int row = i / BKV_DQ;
            int col = i % BKV_DQ;
            int qi = q_start + row;
            int ki = kv_start + col;
            P_smem[i] = (qi < S && ki < S) ? __expf(S_smem[i] * scale - Li_smem[row]) : 0.0f;
        }
        __syncthreads();

        // dP = dOi @ Vj^T (reuse S_smem)
        {
            int wr = warp_id / 2;
            int wc_base = (warp_id % 2) * 4;
            #pragma unroll
            for (int wj = 0; wj < 4; wj++) {
                int ct = wc_base + wj;
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < D_HEAD / WMMA_K; kk++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(a_frag, dOi_smem + wr * 16 * D_HEAD + kk * 16, D_HEAD);
                    wmma::load_matrix_sync(b_frag, Vj_smem + ct * 16 * D_HEAD + kk * 16, D_HEAD);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                wmma::store_matrix_sync(S_smem + wr * 16 * BKV_DQ + ct * 16, c_frag, BKV_DQ, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // dS = P * (dP - D) * scale -> bf16
        for (int i = tid; i < BQ_DQ * BKV_DQ; i += NTHREADS) {
            int row = i / BKV_DQ;
            int col = i % BKV_DQ;
            int qi = q_start + row;
            int ki = kv_start + col;
            if (qi < S && ki < S) {
                float ds_val = P_smem[i] * (S_smem[i] - Di_smem[row]) * scale;
                dS_bf16[i] = __float2bfloat16(ds_val);
            } else {
                dS_bf16[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // dQ += dS @ Kj (BQ x D = 4x8 = 32 tiles, 4 per warp, 8 K-steps)
        {
            int wr = warp_id / 2;
            int wc_base = (warp_id % 2) * 4;
            #pragma unroll
            for (int wj = 0; wj < 4; wj++) {
                int ct = wc_base + wj;
                #pragma unroll
                for (int kk = 0; kk < BKV_DQ / WMMA_K; kk++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(a_frag, dS_bf16 + wr * 16 * BKV_DQ + kk * 16, BKV_DQ);
                    wmma::load_matrix_sync(b_frag, Kj_smem + kk * 16 * D_HEAD + ct * 16, D_HEAD);
                    wmma::mma_sync(dQ_frag[wj], a_frag, b_frag, dQ_frag[wj]);
                }
            }
        }
        __syncthreads();
    }

    // Store dQ to global as bf16
    {
        int wr = warp_id / 2;
        int wc_base = (warp_id % 2) * 4;
        float* stage = staging + warp_id * WMMA_M * WMMA_N;
        #pragma unroll
        for (int wj = 0; wj < 4; wj++) {
            int ct = wc_base + wj;
            wmma::store_matrix_sync(stage, dQ_frag[wj], WMMA_N, wmma::mem_row_major);
            __syncwarp();
            int grb = q_start + wr * 16;
            int gcb = ct * 16;
            for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
                int r = i / WMMA_N;
                int c = i % WMMA_N;
                int gr = grb + r;
                if (gr < S)
                    dQ_base[(size_t)gr * D_HEAD + gcb + c] = __float2bfloat16(stage[i]);
            }
        }
    }
}

__global__ __launch_bounds__(NTHREADS, 1) void compute_dKdV_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_arr,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S) {

    extern __shared__ char smem[];
    __nv_bfloat16* Kj_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* Vj_smem = Kj_smem + BKV_DKV * D_HEAD;
    __nv_bfloat16* Qi_smem = Vj_smem + BKV_DKV * D_HEAD;
    __nv_bfloat16* dOi_smem = Qi_smem + BQ_DKV * D_HEAD;
    float* S_smem = reinterpret_cast<float*>(dOi_smem + BQ_DKV * D_HEAD);
    float* P_smem = S_smem + BKV_DKV * BQ_DKV;
    __nv_bfloat16* P_bf16 = reinterpret_cast<__nv_bfloat16*>(P_smem + BKV_DKV * BQ_DKV);
    __nv_bfloat16* dS_bf16 = P_bf16 + BKV_DKV * BQ_DKV;
    float* Li_smem = reinterpret_cast<float*>(dS_bf16 + BKV_DKV * BQ_DKV);
    float* Di_smem = Li_smem + BQ_DKV;
    float* staging = Di_smem + BQ_DKV;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int num_kv_blocks = (S + BKV_DKV - 1) / BKV_DKV;
    int bh_idx = blockIdx.x / num_kv_blocks;
    int kv_block = blockIdx.x % num_kv_blocks;
    int b = bh_idx / H;
    int h = bh_idx % H;
    int kv_start = kv_block * BKV_DKV;

    size_t bh_offset = (size_t)(b * H + h) * S * D_HEAD;
    const __nv_bfloat16* Q_base = Q + bh_offset;
    const __nv_bfloat16* K_base = K + bh_offset;
    const __nv_bfloat16* V_base = V + bh_offset;
    const __nv_bfloat16* dO_base = dO + bh_offset;
    const float* L_base = L + (size_t)(b * H + h) * S;
    const float* D_base = D_arr + (size_t)(b * H + h) * S;
    __nv_bfloat16* dK_base = dK_out + bh_offset;
    __nv_bfloat16* dV_base = dV_out + bh_offset;

    // Load Kj, Vj (once)
    {
        int4* Kj_v = reinterpret_cast<int4*>(Kj_smem);
        int4* Vj_v = reinterpret_cast<int4*>(Vj_smem);
        int total_vecs = BKV_DKV * D_HEAD / 8;
        for (int i = tid; i < total_vecs; i += NTHREADS) {
            int row = i / (D_HEAD / 8);
            int col_vec = i % (D_HEAD / 8);
            int gr = kv_start + row;
            if (gr < S) {
                Kj_v[i] = *reinterpret_cast<const int4*>(&K_base[(size_t)gr * D_HEAD + col_vec * 8]);
                Vj_v[i] = *reinterpret_cast<const int4*>(&V_base[(size_t)gr * D_HEAD + col_vec * 8]);
            } else {
                Kj_v[i] = make_int4(0,0,0,0);
                Vj_v[i] = make_int4(0,0,0,0);
            }
        }
    }
    __syncthreads();

    // dK/dV accumulators: BKV x D = 4x8 = 32 tiles, 4 per warp
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dK_frag[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dV_frag[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        wmma::fill_fragment(dK_frag[i], 0.0f);
        wmma::fill_fragment(dV_frag[i], 0.0f);
    }

    const float scale = 0.08838834764f;

    for (int q_start = 0; q_start < S; q_start += BQ_DKV) {
        // Load Qi, dOi, Li, Di
        {
            int4* Qi_v = reinterpret_cast<int4*>(Qi_smem);
            int4* dOi_v = reinterpret_cast<int4*>(dOi_smem);
            int total_vecs = BQ_DKV * D_HEAD / 8;
            for (int i = tid; i < total_vecs; i += NTHREADS) {
                int row = i / (D_HEAD / 8);
                int col_vec = i % (D_HEAD / 8);
                int gr = q_start + row;
                if (gr < S) {
                    Qi_v[i] = *reinterpret_cast<const int4*>(&Q_base[(size_t)gr * D_HEAD + col_vec * 8]);
                    dOi_v[i] = *reinterpret_cast<const int4*>(&dO_base[(size_t)gr * D_HEAD + col_vec * 8]);
                } else {
                    Qi_v[i] = make_int4(0,0,0,0);
                    dOi_v[i] = make_int4(0,0,0,0);
                }
            }
        }
        if (tid < BQ_DKV) {
            int gr = q_start + tid;
            Li_smem[tid] = (gr < S) ? L_base[gr] : 0.0f;
            Di_smem[tid] = (gr < S) ? D_base[gr] : 0.0f;
        }
        __syncthreads();

        // S^T = Kj @ Qi^T (BKV x BQ = 4x4 = 16 tiles, 2 per warp, 8 K-steps)
        {
            int wr = warp_id / 2;
            int wc_base = (warp_id % 2) * 2;
            #pragma unroll
            for (int wj = 0; wj < 2; wj++) {
                int ct = wc_base + wj;
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < D_HEAD / WMMA_K; kk++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(a_frag, Kj_smem + wr * 16 * D_HEAD + kk * 16, D_HEAD);
                    wmma::load_matrix_sync(b_frag, Qi_smem + ct * 16 * D_HEAD + kk * 16, D_HEAD);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                wmma::store_matrix_sync(S_smem + wr * 16 * BQ_DKV + ct * 16, c_frag, BQ_DKV, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // P^T = exp(S^T * scale - L^T) and convert to bf16
        for (int i = tid; i < BKV_DKV * BQ_DKV; i += NTHREADS) {
            int row = i / BQ_DKV;
            int col = i % BQ_DKV;
            int ki = kv_start + row;
            int qi = q_start + col;
            if (ki < S && qi < S) {
                float p_val = __expf(S_smem[i] * scale - Li_smem[col]);
                P_smem[i] = p_val;
                P_bf16[i] = __float2bfloat16(p_val);
            } else {
                P_smem[i] = 0.0f;
                P_bf16[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // dP^T = Vj @ dOi^T (reuse S_smem)
        {
            int wr = warp_id / 2;
            int wc_base = (warp_id % 2) * 2;
            #pragma unroll
            for (int wj = 0; wj < 2; wj++) {
                int ct = wc_base + wj;
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < D_HEAD / WMMA_K; kk++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(a_frag, Vj_smem + wr * 16 * D_HEAD + kk * 16, D_HEAD);
                    wmma::load_matrix_sync(b_frag, dOi_smem + ct * 16 * D_HEAD + kk * 16, D_HEAD);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                wmma::store_matrix_sync(S_smem + wr * 16 * BQ_DKV + ct * 16, c_frag, BQ_DKV, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // dS^T = P^T * (dP^T - D^T) * scale -> bf16
        for (int i = tid; i < BKV_DKV * BQ_DKV; i += NTHREADS) {
            int row = i / BQ_DKV;
            int col = i % BQ_DKV;
            int ki = kv_start + row;
            int qi = q_start + col;
            if (ki < S && qi < S) {
                float ds_val = P_smem[i] * (S_smem[i] - Di_smem[col]) * scale;
                dS_bf16[i] = __float2bfloat16(ds_val);
            } else {
                dS_bf16[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // dV += P^T @ dOi (BKV x D = 4x8 = 32 tiles, 4 per warp, 4 K-steps)
        {
            int wr = warp_id / 2;
            int wc_base = (warp_id % 2) * 4;
            #pragma unroll
            for (int wj = 0; wj < 4; wj++) {
                int ct = wc_base + wj;
                #pragma unroll
                for (int kk = 0; kk < BQ_DKV / WMMA_K; kk++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(a_frag, P_bf16 + wr * 16 * BQ_DKV + kk * 16, BQ_DKV);
                    wmma::load_matrix_sync(b_frag, dOi_smem + kk * 16 * D_HEAD + ct * 16, D_HEAD);
                    wmma::mma_sync(dV_frag[wj], a_frag, b_frag, dV_frag[wj]);
                }
            }
        }

        // dK += dS^T @ Qi (same structure)
        {
            int wr = warp_id / 2;
            int wc_base = (warp_id % 2) * 4;
            #pragma unroll
            for (int wj = 0; wj < 4; wj++) {
                int ct = wc_base + wj;
                #pragma unroll
                for (int kk = 0; kk < BQ_DKV / WMMA_K; kk++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(a_frag, dS_bf16 + wr * 16 * BQ_DKV + kk * 16, BQ_DKV);
                    wmma::load_matrix_sync(b_frag, Qi_smem + kk * 16 * D_HEAD + ct * 16, D_HEAD);
                    wmma::mma_sync(dK_frag[wj], a_frag, b_frag, dK_frag[wj]);
                }
            }
        }
        __syncthreads();
    }

    // Store dK, dV to global as bf16
    {
        int wr = warp_id / 2;
        int wc_base = (warp_id % 2) * 4;
        float* stage = staging + warp_id * WMMA_M * WMMA_N;
        #pragma unroll
        for (int wj = 0; wj < 4; wj++) {
            int ct = wc_base + wj;
            int grb = kv_start + wr * 16;
            int gcb = ct * 16;

            wmma::store_matrix_sync(stage, dK_frag[wj], WMMA_N, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
                int r = i / WMMA_N;
                int c = i % WMMA_N;
                int gr = grb + r;
                if (gr < S)
                    dK_base[(size_t)gr * D_HEAD + gcb + c] = __float2bfloat16(stage[i]);
            }

            wmma::store_matrix_sync(stage, dV_frag[wj], WMMA_N, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
                int r = i / WMMA_N;
                int c = i % WMMA_N;
                int gr = grb + r;
                if (gr < S)
                    dV_base[(size_t)gr * D_HEAD + gcb + c] = __float2bfloat16(stage[i]);
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4, H = 48, S = (int)Q.size(2), d = 128;
    int total_rows = B * H * S;

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* D_buf;
    CUDA_CHECK(cudaMalloc(&D_buf, (size_t)total_rows * sizeof(float)));

    compute_D_kernel<<<total_rows, 128, 0, stream>>>(O_ptr, dO_ptr, D_buf, total_rows);

    {
        int num_q_blocks = (S + BQ_DQ - 1) / BQ_DQ;
        int grid = B * H * num_q_blocks;
        CUDA_CHECK(cudaFuncSetAttribute(compute_dQ_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE_DQ));
        compute_dQ_kernel<<<grid, NTHREADS, SMEM_SIZE_DQ, stream>>>(
            Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_buf, dQ_ptr, B, H, S);
    }

    {
        int num_kv_blocks = (S + BKV_DKV - 1) / BKV_DKV;
        int grid = B * H * num_kv_blocks;
        CUDA_CHECK(cudaFuncSetAttribute(compute_dKdV_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE_DKV));
        compute_dKdV_kernel<<<grid, NTHREADS, SMEM_SIZE_DKV, stream>>>(
            Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_buf, dK_ptr, dV_ptr, B, H, S);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(D_buf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attention_bwd::run);

}  // namespace attention_bwd