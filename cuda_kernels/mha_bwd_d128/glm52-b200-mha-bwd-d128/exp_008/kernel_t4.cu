#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cmath>
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

namespace mha_bwd_d128 {

constexpr int B = 4;
constexpr int H = 48;
constexpr int D = 128;
constexpr int D8 = D / 8;
constexpr int WMMA = 16;

__global__ void compute_D_kernel(
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    float* __restrict__ D_buf, int S) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S;
    if (idx >= total) return;
    int i = idx % S;
    int bh = idx / S;
    const __nv_bfloat16* O_ptr = O + (size_t)bh * S * D + (size_t)i * D;
    const __nv_bfloat16* dO_ptr = dO + (size_t)bh * S * D + (size_t)i * D;
    float d_val = 0.0f;
    #pragma unroll 8
    for (int k = 0; k < D; k += 2) {
        float2 o = __bfloat1622float2(*(const __nv_bfloat162*)&O_ptr[k]);
        float2 g = __bfloat1622float2(*(const __nv_bfloat162*)&dO_ptr[k]);
        d_val = fmaf(o.x, g.x, d_val);
        d_val = fmaf(o.y, g.y, d_val);
    }
    D_buf[idx] = d_val;
}

// cp.async helpers
__device__ __forceinline__ void cp_async_16B(uint32_t smem_addr, const void* gmem_addr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_addr));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}
template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

// dQ kernel: BR=128, BC=64, 128 threads, 4 warps, 16 dQ_frag, double-buffered K/V
__global__ void compute_dQ_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_buf,
    __nv_bfloat16* __restrict__ dQ, int S, float scale) {

    constexpr int BR = 128, BC = 64;
    int num_q_blocks = (S + BR - 1) / BR;
    int bh = blockIdx.x / num_q_blocks;
    int qi_start = (blockIdx.x % num_q_blocks) * BR;
    int br = min(BR, S - qi_start);
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int row_base = warp_id * 32;

    const __nv_bfloat16* Q_bh = Q + (size_t)bh * S * D;
    const __nv_bfloat16* K_bh = K + (size_t)bh * S * D;
    const __nv_bfloat16* V_bh = V + (size_t)bh * S * D;
    const __nv_bfloat16* dO_bh = dO + (size_t)bh * S * D;
    const float* L_bh = L + (size_t)bh * S;
    const float* D_bh = D_buf + (size_t)bh * S;
    __nv_bfloat16* dQ_bh = dQ + (size_t)bh * S * D;

    extern __shared__ char smem[];
    __nv_bfloat16* sQ = (__nv_bfloat16*)smem;
    __nv_bfloat16* sdO = sQ + BR * D;
    __nv_bfloat16* sK = sdO + BR * D;
    __nv_bfloat16* sV = sK + BC * D;
    float* sS = (float*)(sV + BC * D);
    float* sdP = sS + BR * BC;
    float* sL = sdP + BR * BC;
    float* sDrow = sL + BR;

    // Load Q, dO, L, D
    for (int idx = tid; idx < BR * D8; idx += 128) {
        int i = idx / D8, k8 = idx % D8;
        if (i < br) {
            *(int4*)&sQ[i * D + k8 * 8] = *(const int4*)&Q_bh[(qi_start + i) * D + k8 * 8];
            *(int4*)&sdO[i * D + k8 * 8] = *(const int4*)&dO_bh[(qi_start + i) * D + k8 * 8];
        } else {
            *(int4*)&sQ[i * D + k8 * 8] = make_int4(0,0,0,0);
            *(int4*)&sdO[i * D + k8 * 8] = make_int4(0,0,0,0);
        }
    }
    for (int i = tid; i < BR; i += 128) {
        sL[i] = (i < br) ? L_bh[qi_start + i] : -1e30f;
        sDrow[i] = (i < br) ? D_bh[qi_start + i] : 0.0f;
    }
    __syncthreads();

    using FragA = wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, __nv_bfloat16, wmma::row_major>;
    using FragBc = wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, __nv_bfloat16, wmma::col_major>;
    using FragBr = wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, __nv_bfloat16, wmma::row_major>;
    using FragC = wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, float>;

    FragA a_frag;
    FragBc b_frag_col;
    FragBr b_frag_row;
    FragC c_frag[4];
    FragC dQ_frag[16];

    #pragma unroll
    for (int i = 0; i < 16; i++) wmma::fill_fragment(dQ_frag[i], 0.0f);

    for (int kj = 0; kj < S; kj += BC) {
        int bc = min(BC, S - kj);
        for (int idx = tid; idx < BC * D8; idx += 128) {
            int j = idx / D8, k8 = idx % D8;
            if (j < bc) {
                *(int4*)&sK[j * D + k8 * 8] = *(const int4*)&K_bh[(kj + j) * D + k8 * 8];
                *(int4*)&sV[j * D + k8 * 8] = *(const int4*)&V_bh[(kj + j) * D + k8 * 8];
            } else {
                *(int4*)&sK[j * D + k8 * 8] = make_int4(0,0,0,0);
                *(int4*)&sV[j * D + k8 * 8] = make_int4(0,0,0,0);
            }
        }
        __syncthreads();

        // S = Q @ K^T
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            #pragma unroll
            for (int n = 0; n < 4; n++) wmma::fill_fragment(c_frag[n], 0.0f);
            #pragma unroll
            for (int k = 0; k < D / WMMA; k++) {
                wmma::load_matrix_sync(a_frag, &sQ[(row_base + r * WMMA) * D + k * WMMA], D);
                #pragma unroll
                for (int n = 0; n < 4; n++) {
                    wmma::load_matrix_sync(b_frag_col, &sK[n * WMMA * D + k * WMMA], D);
                    wmma::mma_sync(c_frag[n], a_frag, b_frag_col, c_frag[n]);
                }
            }
            #pragma unroll
            for (int n = 0; n < 4; n++)
                wmma::store_matrix_sync(&sS[(row_base + r * WMMA) * BC + n * WMMA], c_frag[n], BC, wmma::mem_row_major);
        }
        __syncthreads();

        // P = exp(S * scale - L)
        for (int idx = tid; idx < BR * BC; idx += 128) {
            int i = idx / BC;
            sS[idx] = (i < br) ? __expf(sS[idx] * scale - sL[i]) : 0.0f;
        }
        __syncthreads();

        // dP = dO @ V^T
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            #pragma unroll
            for (int n = 0; n < 4; n++) wmma::fill_fragment(c_frag[n], 0.0f);
            #pragma unroll
            for (int k = 0; k < D / WMMA; k++) {
                wmma::load_matrix_sync(a_frag, &sdO[(row_base + r * WMMA) * D + k * WMMA], D);
                #pragma unroll
                for (int n = 0; n < 4; n++) {
                    wmma::load_matrix_sync(b_frag_col, &sV[n * WMMA * D + k * WMMA], D);
                    wmma::mma_sync(c_frag[n], a_frag, b_frag_col, c_frag[n]);
                }
            }
            #pragma unroll
            for (int n = 0; n < 4; n++)
                wmma::store_matrix_sync(&sdP[(row_base + r * WMMA) * BC + n * WMMA], c_frag[n], BC, wmma::mem_row_major);
        }
        __syncthreads();

        // dS = P * (dP - D) * scale, convert to bf16 in sV space
        __nv_bfloat16* sDS_bf16 = (__nv_bfloat16*)sV;
        for (int idx = tid; idx < BR * BC; idx += 128) {
            int i = idx / BC;
            if (i < br) {
                float p = sS[idx];
                float dp = sdP[idx];
                sDS_bf16[idx] = __float2bfloat16(p * (dp - sDrow[i]) * scale);
            } else {
                sDS_bf16[idx] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // dQ += dS @ K
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            #pragma unroll
            for (int k = 0; k < BC / WMMA; k++) {
                wmma::load_matrix_sync(a_frag, &sDS_bf16[(row_base + r * WMMA) * BC + k * WMMA], BC);
                #pragma unroll
                for (int n = 0; n < 8; n++) {
                    wmma::load_matrix_sync(b_frag_row, &sK[k * WMMA * D + n * WMMA], D);
                    wmma::mma_sync(dQ_frag[r * 8 + n], a_frag, b_frag_row, dQ_frag[r * 8 + n]);
                }
            }
        }
        __syncthreads();
    }

    // Store dQ
    float* sdQ_tmp = (float*)sK;
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        #pragma unroll
        for (int n = 0; n < 8; n++)
            wmma::store_matrix_sync(&sdQ_tmp[(row_base + r * WMMA) * D + n * WMMA], dQ_frag[r * 8 + n], D, wmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = tid; idx < BR * D; idx += 128) {
        int i = idx / D, k = idx % D;
        if (i < br)
            dQ_bh[(qi_start + i) * D + k] = __float2bfloat16(sdQ_tmp[idx]);
    }
}

// dV kernel: BR=64, BC=128, 128 threads, 4 warps, 16 dV_frag
__global__ void compute_dV_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_buf,
    __nv_bfloat16* __restrict__ dV, int S, float scale) {

    constexpr int BR = 64, BC = 128;
    int num_k_blocks = (S + BC - 1) / BC;
    int bh = blockIdx.x / num_k_blocks;
    int kj_start = (blockIdx.x % num_k_blocks) * BC;
    int bc = min(BC, S - kj_start);
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int s_row = warp_id * WMMA;     // For S: 1 row tile
    int v_row = warp_id * 32;        // For dV: 2 row tiles

    const __nv_bfloat16* Q_bh = Q + (size_t)bh * S * D;
    const __nv_bfloat16* K_bh = K + (size_t)bh * S * D;
    const __nv_bfloat16* V_bh = V + (size_t)bh * S * D;
    const __nv_bfloat16* dO_bh = dO + (size_t)bh * S * D;
    const float* L_bh = L + (size_t)bh * S;
    const float* D_bh = D_buf + (size_t)bh * S;
    __nv_bfloat16* dV_bh = dV + (size_t)bh * S * D;

    extern __shared__ char smem[];
    __nv_bfloat16* sK = (__nv_bfloat16*)smem;
    __nv_bfloat16* sV = sK + BC * D;
    __nv_bfloat16* sQ = sV + BC * D;
    __nv_bfloat16* sdO = sQ + BR * D;
    float* sS = (float*)(sdO + BR * D);
    float* sdP = sS + BR * BC;
    float* sL = sdP + BR * BC;
    float* sDrow = sL + BR;

    // Load K, V
    for (int idx = tid; idx < BC * D8; idx += 128) {
        int j = idx / D8, k8 = idx % D8;
        if (j < bc) {
            *(int4*)&sK[j * D + k8 * 8] = *(const int4*)&K_bh[(kj_start + j) * D + k8 * 8];
            *(int4*)&sV[j * D + k8 * 8] = *(const int4*)&V_bh[(kj_start + j) * D + k8 * 8];
        } else {
            *(int4*)&sK[j * D + k8 * 8] = make_int4(0,0,0,0);
            *(int4*)&sV[j * D + k8 * 8] = make_int4(0,0,0,0);
        }
    }
    __syncthreads();

    using FragAr = wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, __nv_bfloat16, wmma::row_major>;
    using FragAc = wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, __nv_bfloat16, wmma::col_major>;
    using FragBc = wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, __nv_bfloat16, wmma::col_major>;
    using FragBr = wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, __nv_bfloat16, wmma::row_major>;
    using FragC = wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, float>;

    FragAr a_frag_row;
    FragAc a_frag_col;
    FragBc b_frag_col;
    FragBr b_frag_row;
    FragC c_frag[4];
    FragC dV_frag[16];

    #pragma unroll
    for (int i = 0; i < 16; i++) wmma::fill_fragment(dV_frag[i], 0.0f);

    for (int qi = 0; qi < S; qi += BR) {
        int br = min(BR, S - qi);
        for (int idx = tid; idx < BR * D8; idx += 128) {
            int i = idx / D8, k8 = idx % D8;
            if (i < br) {
                *(int4*)&sQ[i * D + k8 * 8] = *(const int4*)&Q_bh[(qi + i) * D + k8 * 8];
                *(int4*)&sdO[i * D + k8 * 8] = *(const int4*)&dO_bh[(qi + i) * D + k8 * 8];
            } else {
                *(int4*)&sQ[i * D + k8 * 8] = make_int4(0,0,0,0);
                *(int4*)&sdO[i * D + k8 * 8] = make_int4(0,0,0,0);
            }
        }
        for (int i = tid; i < BR; i += 128) {
            sL[i] = (i < br) ? L_bh[qi + i] : -1e30f;
            sDrow[i] = (i < br) ? D_bh[qi + i] : 0.0f;
        }
        __syncthreads();

        // S = Q @ K^T (1 row tile, 8 col tiles in 2 batches, 8 K tiles)
        #pragma unroll
        for (int batch = 0; batch < 2; batch++) {
            int n_start = batch * 4;
            #pragma unroll
            for (int n = 0; n < 4; n++) wmma::fill_fragment(c_frag[n], 0.0f);
            #pragma unroll
            for (int k = 0; k < D / WMMA; k++) {
                wmma::load_matrix_sync(a_frag_row, &sQ[s_row * D + k * WMMA], D);
                #pragma unroll
                for (int n = 0; n < 4; n++) {
                    wmma::load_matrix_sync(b_frag_col, &sK[(n_start + n) * WMMA * D + k * WMMA], D);
                    wmma::mma_sync(c_frag[n], a_frag_row, b_frag_col, c_frag[n]);
                }
            }
            #pragma unroll
            for (int n = 0; n < 4; n++)
                wmma::store_matrix_sync(&sS[s_row * BC + (n_start + n) * WMMA], c_frag[n], BC, wmma::mem_row_major);
        }
        __syncthreads();

        // P = exp(S * scale - L), convert to bf16
        __nv_bfloat16* sP_bf16 = (__nv_bfloat16*)sdP;
        for (int idx = tid; idx < BR * BC; idx += 128) {
            int i = idx / BC;
            float p = (i < br) ? __expf(sS[idx] * scale - sL[i]) : 0.0f;
            sP_bf16[idx] = __float2bfloat16(p);
        }
        __syncthreads();

        // dV += P^T @ dO (2 row tiles, 8 col tiles, 4 K tiles)
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            #pragma unroll
            for (int k = 0; k < BR / WMMA; k++) {
                wmma::load_matrix_sync(a_frag_col, &sP_bf16[k * WMMA * BC + v_row + r * WMMA], BC);
                #pragma unroll
                for (int n = 0; n < 8; n++) {
                    wmma::load_matrix_sync(b_frag_row, &sdO[k * WMMA * D + n * WMMA], D);
                    wmma::mma_sync(dV_frag[r * 8 + n], a_frag_col, b_frag_row, dV_frag[r * 8 + n]);
                }
            }
        }
        __syncthreads();
    }

    // Store dV
    float* sdV_tmp = (float*)sQ;
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        #pragma unroll
        for (int n = 0; n < 8; n++)
            wmma::store_matrix_sync(&sdV_tmp[(v_row + r * WMMA) * D + n * WMMA], dV_frag[r * 8 + n], D, wmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = tid; idx < BC * D; idx += 128) {
        int j = idx / D, k = idx % D;
        if (j < bc)
            dV_bh[(kj_start + j) * D + k] = __float2bfloat16(sdV_tmp[idx]);
    }
}

// dK kernel: BR=64, BC=128, 128 threads, 4 warps, 16 dK_frag
__global__ void compute_dK_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_buf,
    __nv_bfloat16* __restrict__ dK, int S, float scale) {

    constexpr int BR = 64, BC = 128;
    int num_k_blocks = (S + BC - 1) / BC;
    int bh = blockIdx.x / num_k_blocks;
    int kj_start = (blockIdx.x % num_k_blocks) * BC;
    int bc = min(BC, S - kj_start);
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int s_row = warp_id * WMMA;
    int v_row = warp_id * 32;

    const __nv_bfloat16* Q_bh = Q + (size_t)bh * S * D;
    const __nv_bfloat16* K_bh = K + (size_t)bh * S * D;
    const __nv_bfloat16* V_bh = V + (size_t)bh * S * D;
    const __nv_bfloat16* dO_bh = dO + (size_t)bh * S * D;
    const float* L_bh = L + (size_t)bh * S;
    const float* D_bh = D_buf + (size_t)bh * S;
    __nv_bfloat16* dK_bh = dK + (size_t)bh * S * D;

    extern __shared__ char smem[];
    __nv_bfloat16* sK = (__nv_bfloat16*)smem;
    __nv_bfloat16* sV = sK + BC * D;
    __nv_bfloat16* sQ = sV + BC * D;
    __nv_bfloat16* sdO = sQ + BR * D;
    float* sS = (float*)(sdO + BR * D);
    float* sdP = sS + BR * BC;
    __nv_bfloat16* sDS_bf16 = (__nv_bfloat16*)(sdP + BR * BC);
    float* sL = (float*)(sDS_bf16 + BR * BC);
    float* sDrow = sL + BR;

    for (int idx = tid; idx < BC * D8; idx += 128) {
        int j = idx / D8, k8 = idx % D8;
        if (j < bc) {
            *(int4*)&sK[j * D + k8 * 8] = *(const int4*)&K_bh[(kj_start + j) * D + k8 * 8];
            *(int4*)&sV[j * D + k8 * 8] = *(const int4*)&V_bh[(kj_start + j) * D + k8 * 8];
        } else {
            *(int4*)&sK[j * D + k8 * 8] = make_int4(0,0,0,0);
            *(int4*)&sV[j * D + k8 * 8] = make_int4(0,0,0,0);
        }
    }
    __syncthreads();

    using FragAr = wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, __nv_bfloat16, wmma::row_major>;
    using FragAc = wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, __nv_bfloat16, wmma::col_major>;
    using FragBc = wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, __nv_bfloat16, wmma::col_major>;
    using FragBr = wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, __nv_bfloat16, wmma::row_major>;
    using FragC = wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, float>;

    FragAr a_frag_row;
    FragAc a_frag_col;
    FragBc b_frag_col;
    FragBr b_frag_row;
    FragC c_frag[4];
    FragC dK_frag[16];

    #pragma unroll
    for (int i = 0; i < 16; i++) wmma::fill_fragment(dK_frag[i], 0.0f);

    for (int qi = 0; qi < S; qi += BR) {
        int br = min(BR, S - qi);
        for (int idx = tid; idx < BR * D8; idx += 128) {
            int i = idx / D8, k8 = idx % D8;
            if (i < br) {
                *(int4*)&sQ[i * D + k8 * 8] = *(const int4*)&Q_bh[(qi + i) * D + k8 * 8];
                *(int4*)&sdO[i * D + k8 * 8] = *(const int4*)&dO_bh[(qi + i) * D + k8 * 8];
            } else {
                *(int4*)&sQ[i * D + k8 * 8] = make_int4(0,0,0,0);
                *(int4*)&sdO[i * D + k8 * 8] = make_int4(0,0,0,0);
            }
        }
        for (int i = tid; i < BR; i += 128) {
            sL[i] = (i < br) ? L_bh[qi + i] : -1e30f;
            sDrow[i] = (i < br) ? D_bh[qi + i] : 0.0f;
        }
        __syncthreads();

        // S = Q @ K^T
        #pragma unroll
        for (int batch = 0; batch < 2; batch++) {
            int n_start = batch * 4;
            #pragma unroll
            for (int n = 0; n < 4; n++) wmma::fill_fragment(c_frag[n], 0.0f);
            #pragma unroll
            for (int k = 0; k < D / WMMA; k++) {
                wmma::load_matrix_sync(a_frag_row, &sQ[s_row * D + k * WMMA], D);
                #pragma unroll
                for (int n = 0; n < 4; n++) {
                    wmma::load_matrix_sync(b_frag_col, &sK[(n_start + n) * WMMA * D + k * WMMA], D);
                    wmma::mma_sync(c_frag[n], a_frag_row, b_frag_col, c_frag[n]);
                }
            }
            #pragma unroll
            for (int n = 0; n < 4; n++)
                wmma::store_matrix_sync(&sS[s_row * BC + (n_start + n) * WMMA], c_frag[n], BC, wmma::mem_row_major);
        }
        __syncthreads();

        // P = exp(S * scale - L)
        for (int idx = tid; idx < BR * BC; idx += 128) {
            int i = idx / BC;
            sS[idx] = (i < br) ? __expf(sS[idx] * scale - sL[i]) : 0.0f;
        }
        __syncthreads();

        // dP = dO @ V^T
        #pragma unroll
        for (int batch = 0; batch < 2; batch++) {
            int n_start = batch * 4;
            #pragma unroll
            for (int n = 0; n < 4; n++) wmma::fill_fragment(c_frag[n], 0.0f);
            #pragma unroll
            for (int k = 0; k < D / WMMA; k++) {
                wmma::load_matrix_sync(a_frag_row, &sdO[s_row * D + k * WMMA], D);
                #pragma unroll
                for (int n = 0; n < 4; n++) {
                    wmma::load_matrix_sync(b_frag_col, &sV[(n_start + n) * WMMA * D + k * WMMA], D);
                    wmma::mma_sync(c_frag[n], a_frag_row, b_frag_col, c_frag[n]);
                }
            }
            #pragma unroll
            for (int n = 0; n < 4; n++)
                wmma::store_matrix_sync(&sdP[s_row * BC + (n_start + n) * WMMA], c_frag[n], BC, wmma::mem_row_major);
        }
        __syncthreads();

        // dS = P * (dP - D) * scale, convert to bf16
        for (int idx = tid; idx < BR * BC; idx += 128) {
            int i = idx / BC;
            if (i < br) {
                float p = sS[idx];
                float dp = sdP[idx];
                sDS_bf16[idx] = __float2bfloat16(p * (dp - sDrow[i]) * scale);
            } else {
                sDS_bf16[idx] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // dK += dS^T @ Q (2 row tiles, 8 col tiles, 4 K tiles)
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            #pragma unroll
            for (int k = 0; k < BR / WMMA; k++) {
                wmma::load_matrix_sync(a_frag_col, &sDS_bf16[k * WMMA * BC + v_row + r * WMMA], BC);
                #pragma unroll
                for (int n = 0; n < 8; n++) {
                    wmma::load_matrix_sync(b_frag_row, &sQ[k * WMMA * D + n * WMMA], D);
                    wmma::mma_sync(dK_frag[r * 8 + n], a_frag_col, b_frag_row, dK_frag[r * 8 + n]);
                }
            }
        }
        __syncthreads();
    }

    // Store dK
    float* sdK_tmp = (float*)sQ;
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        #pragma unroll
        for (int n = 0; n < 8; n++)
            wmma::store_matrix_sync(&sdK_tmp[(v_row + r * WMMA) * D + n * WMMA], dK_frag[r * 8 + n], D, wmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = tid; idx < BC * D; idx += 128) {
        int j = idx / D, k = idx % D;
        if (j < bc)
            dK_bh[(kj_start + j) * D + k] = __float2bfloat16(sdK_tmp[idx]);
    }
}

void run(
    tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
    tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
    tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int S = (int)Q.size(2);
    float scale = 1.0f / sqrtf((float)D);

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* D_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&D_buf, (size_t)B * H * S * sizeof(float)));

    {   int total = B * H * S, t = 256;
        compute_D_kernel<<<(total+t-1)/t, t, 0, stream>>>(O_ptr, dO_ptr, D_buf, S);
    }
    {   int nqb = (S + 127) / 128;
        size_t sz = (size_t)2*128*D*2 + 2*64*D*2 + 2*128*64*4 + 2*128*4;
        CUDA_CHECK(cudaFuncSetAttribute(compute_dQ_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sz));
        compute_dQ_kernel<<<B*H*nqb, 128, sz, stream>>>(
            Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_buf, dQ_ptr, S, scale);
    }
    {   int nkb = (S + 127) / 128;
        size_t sz = (size_t)2*128*D*2 + 2*64*D*2 + 2*64*128*4 + 2*64*4;
        CUDA_CHECK(cudaFuncSetAttribute(compute_dV_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sz));
        compute_dV_kernel<<<B*H*nkb, 128, sz, stream>>>(
            Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_buf, dV_ptr, S, scale);
    }
    {   size_t sz = (size_t)2*128*D*2 + 2*64*D*2 + 2*64*128*4 + 64*128*2 + 2*64*4;
        CUDA_CHECK(cudaFuncSetAttribute(compute_dK_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sz));
        compute_dK_kernel<<<B*H*nkb, 128, sz, stream>>>(
            Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_buf, dK_ptr, S, scale);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(D_buf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128