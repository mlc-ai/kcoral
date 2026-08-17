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
constexpr int BR = 64;
constexpr int BC = 64;
constexpr int THREADS = 256;
constexpr int WARPS = 8;

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

__global__ __launch_bounds__(256, 2)
void compute_dQ_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_buf,
    __nv_bfloat16* __restrict__ dQ, int S, float scale) {

    int num_q_blocks = (S + BR - 1) / BR;
    int bh = blockIdx.x / num_q_blocks;
    int qi_start = (blockIdx.x % num_q_blocks) * BR;
    int br = min(BR, S - qi_start);
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int row_base = (warp_id / 2) * WMMA;
    int d_tile_start = (warp_id % 2) * 4;
    constexpr int DTILES = 4;

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
    __nv_bfloat16* sP_dS = (__nv_bfloat16*)(sS + BR * BC);
    float* sL = (float*)(sP_dS + BR * BC);
    float* sDrow = sL + BR;
    float* sdP = (float*)sV;

    for (int idx = tid; idx < BR * D8; idx += THREADS) {
        int i = idx / D8, k8 = idx % D8;
        if (i < br) {
            *(int4*)&sQ[i * D + k8 * 8] = *(const int4*)&Q_bh[(qi_start + i) * D + k8 * 8];
            *(int4*)&sdO[i * D + k8 * 8] = *(const int4*)&dO_bh[(qi_start + i) * D + k8 * 8];
        } else {
            *(int4*)&sQ[i * D + k8 * 8] = make_int4(0,0,0,0);
            *(int4*)&sdO[i * D + k8 * 8] = make_int4(0,0,0,0);
        }
    }
    for (int i = tid; i < BR; i += THREADS) {
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
    FragC dQ_frag[DTILES];

    #pragma unroll
    for (int i = 0; i < DTILES; i++) wmma::fill_fragment(dQ_frag[i], 0.0f);

    for (int kj = 0; kj < S; kj += BC) {
        int bc = min(BC, S - kj);
        for (int idx = tid; idx < BC * D8; idx += THREADS) {
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
        for (int n = 0; n < 4; n++) wmma::fill_fragment(c_frag[n], 0.0f);
        #pragma unroll
        for (int k = 0; k < D / WMMA; k++) {
            wmma::load_matrix_sync(a_frag, &sQ[row_base * D + k * WMMA], D);
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                wmma::load_matrix_sync(b_frag_col, &sK[n * WMMA * D + k * WMMA], D);
                wmma::mma_sync(c_frag[n], a_frag, b_frag_col, c_frag[n]);
            }
        }
        #pragma unroll
        for (int n = 0; n < 4; n++)
            wmma::store_matrix_sync(&sS[row_base * BC + n * WMMA], c_frag[n], BC, wmma::mem_row_major);
        __syncthreads();

        // P = exp(S * scale - L)
        for (int idx = tid; idx < BR * BC; idx += THREADS) {
            int i = idx / BC;
            float p = (i < br) ? __expf(sS[idx] * scale - sL[i]) : 0.0f;
            sP_dS[idx] = __float2bfloat16(p);
        }
        __syncthreads();

        // dP = dO @ V^T
        #pragma unroll
        for (int n = 0; n < 4; n++) wmma::fill_fragment(c_frag[n], 0.0f);
        #pragma unroll
        for (int k = 0; k < D / WMMA; k++) {
            wmma::load_matrix_sync(a_frag, &sdO[row_base * D + k * WMMA], D);
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                wmma::load_matrix_sync(b_frag_col, &sV[n * WMMA * D + k * WMMA], D);
                wmma::mma_sync(c_frag[n], a_frag, b_frag_col, c_frag[n]);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int n = 0; n < 4; n++)
            wmma::store_matrix_sync(&sdP[row_base * BC + n * WMMA], c_frag[n], BC, wmma::mem_row_major);
        __syncthreads();

        // dS = P * (dP - D) * scale
        for (int idx = tid; idx < BR * BC; idx += THREADS) {
            int i = idx / BC;
            if (i < br) {
                float p = __bfloat162float(sP_dS[idx]);
                float dp = sdP[idx];
                sP_dS[idx] = __float2bfloat16(p * (dp - sDrow[i]) * scale);
            } else {
                sP_dS[idx] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // dQ += dS @ K (D-split: each warp does 4 D-tiles)
        #pragma unroll
        for (int k = 0; k < BC / WMMA; k++) {
            wmma::load_matrix_sync(a_frag, &sP_dS[row_base * BC + k * WMMA], BC);
            #pragma unroll
            for (int n = 0; n < DTILES; n++) {
                int d_tile = d_tile_start + n;
                wmma::load_matrix_sync(b_frag_row, &sK[k * WMMA * D + d_tile * WMMA], D);
                wmma::mma_sync(dQ_frag[n], a_frag, b_frag_row, dQ_frag[n]);
            }
        }
        __syncthreads();
    }

    // Store dQ
    float* sdQ_tmp = (float*)sK;
    #pragma unroll
    for (int n = 0; n < DTILES; n++)
        wmma::store_matrix_sync(&sdQ_tmp[row_base * D + (d_tile_start + n) * WMMA], dQ_frag[n], D, wmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < BR * D; idx += THREADS) {
        int i = idx / D, k = idx % D;
        if (i < br)
            dQ_bh[(qi_start + i) * D + k] = __float2bfloat16(sdQ_tmp[idx]);
    }
}

__global__ __launch_bounds__(256, 2)
void compute_dKV_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_buf,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV, int S, float scale) {

    int num_k_blocks = (S + BC - 1) / BC;
    int bh = blockIdx.x / num_k_blocks;
    int kj_start = (blockIdx.x % num_k_blocks) * BC;
    int bc = min(BC, S - kj_start);
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int row_base = (warp_id / 2) * WMMA;
    int d_tile_start = (warp_id % 2) * 4;
    constexpr int DTILES = 4;

    const __nv_bfloat16* Q_bh = Q + (size_t)bh * S * D;
    const __nv_bfloat16* K_bh = K + (size_t)bh * S * D;
    const __nv_bfloat16* V_bh = V + (size_t)bh * S * D;
    const __nv_bfloat16* dO_bh = dO + (size_t)bh * S * D;
    const float* L_bh = L + (size_t)bh * S;
    const float* D_bh = D_buf + (size_t)bh * S;
    __nv_bfloat16* dK_bh = dK + (size_t)bh * S * D;
    __nv_bfloat16* dV_bh = dV + (size_t)bh * S * D;

    extern __shared__ char smem[];
    __nv_bfloat16* sK = (__nv_bfloat16*)smem;
    __nv_bfloat16* sV = sK + BC * D;
    __nv_bfloat16* sQ = sV + BC * D;
    __nv_bfloat16* sdO = sQ + BR * D;
    float* sS = (float*)(sdO + BR * D);
    __nv_bfloat16* sP_dS = (__nv_bfloat16*)(sS + BR * BC);
    float* sL = (float*)(sP_dS + BR * BC);
    float* sDrow = sL + BR;
    float* sdP = (float*)sdO;

    for (int idx = tid; idx < BC * D8; idx += THREADS) {
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
    FragC dV_frag[DTILES];
    FragC dK_frag[DTILES];

    #pragma unroll
    for (int i = 0; i < DTILES; i++) {
        wmma::fill_fragment(dV_frag[i], 0.0f);
        wmma::fill_fragment(dK_frag[i], 0.0f);
    }

    for (int qi = 0; qi < S; qi += BR) {
        int br = min(BR, S - qi);
        for (int idx = tid; idx < BR * D8; idx += THREADS) {
            int i = idx / D8, k8 = idx % D8;
            if (i < br) {
                *(int4*)&sQ[i * D + k8 * 8] = *(const int4*)&Q_bh[(qi + i) * D + k8 * 8];
                *(int4*)&sdO[i * D + k8 * 8] = *(const int4*)&dO_bh[(qi + i) * D + k8 * 8];
            } else {
                *(int4*)&sQ[i * D + k8 * 8] = make_int4(0,0,0,0);
                *(int4*)&sdO[i * D + k8 * 8] = make_int4(0,0,0,0);
            }
        }
        for (int i = tid; i < BR; i += THREADS) {
            sL[i] = (i < br) ? L_bh[qi + i] : -1e30f;
            sDrow[i] = (i < br) ? D_bh[qi + i] : 0.0f;
        }
        __syncthreads();

        // S = Q @ K^T
        #pragma unroll
        for (int n = 0; n < 4; n++) wmma::fill_fragment(c_frag[n], 0.0f);
        #pragma unroll
        for (int k = 0; k < D / WMMA; k++) {
            wmma::load_matrix_sync(a_frag_row, &sQ[row_base * D + k * WMMA], D);
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                wmma::load_matrix_sync(b_frag_col, &sK[n * WMMA * D + k * WMMA], D);
                wmma::mma_sync(c_frag[n], a_frag_row, b_frag_col, c_frag[n]);
            }
        }
        #pragma unroll
        for (int n = 0; n < 4; n++)
            wmma::store_matrix_sync(&sS[row_base * BC + n * WMMA], c_frag[n], BC, wmma::mem_row_major);
        __syncthreads();

        // P = exp(S * scale - L)
        for (int idx = tid; idx < BR * BC; idx += THREADS) {
            int i = idx / BC;
            float p = (i < br) ? __expf(sS[idx] * scale - sL[i]) : 0.0f;
            sP_dS[idx] = __float2bfloat16(p);
        }
        __syncthreads();

        // dV += P^T @ dO
        #pragma unroll
        for (int k = 0; k < BR / WMMA; k++) {
            wmma::load_matrix_sync(a_frag_col, &sP_dS[k * WMMA * BC + row_base], BC);
            #pragma unroll
            for (int n = 0; n < DTILES; n++) {
                int d_tile = d_tile_start + n;
                wmma::load_matrix_sync(b_frag_row, &sdO[k * WMMA * D + d_tile * WMMA], D);
                wmma::mma_sync(dV_frag[n], a_frag_col, b_frag_row, dV_frag[n]);
            }
        }

        // dP = dO @ V^T
        #pragma unroll
        for (int n = 0; n < 4; n++) wmma::fill_fragment(c_frag[n], 0.0f);
        #pragma unroll
        for (int k = 0; k < D / WMMA; k++) {
            wmma::load_matrix_sync(a_frag_row, &sdO[row_base * D + k * WMMA], D);
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                wmma::load_matrix_sync(b_frag_col, &sV[n * WMMA * D + k * WMMA], D);
                wmma::mma_sync(c_frag[n], a_frag_row, b_frag_col, c_frag[n]);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int n = 0; n < 4; n++)
            wmma::store_matrix_sync(&sdP[row_base * BC + n * WMMA], c_frag[n], BC, wmma::mem_row_major);
        __syncthreads();

        // dS = P * (dP - D) * scale
        for (int idx = tid; idx < BR * BC; idx += THREADS) {
            int i = idx / BC;
            if (i < br) {
                float p = __bfloat162float(sP_dS[idx]);
                float dp = sdP[idx];
                sP_dS[idx] = __float2bfloat16(p * (dp - sDrow[i]) * scale);
            } else {
                sP_dS[idx] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // dK += dS^T @ Q
        #pragma unroll
        for (int k = 0; k < BR / WMMA; k++) {
            wmma::load_matrix_sync(a_frag_col, &sP_dS[k * WMMA * BC + row_base], BC);
            #pragma unroll
            for (int n = 0; n < DTILES; n++) {
                int d_tile = d_tile_start + n;
                wmma::load_matrix_sync(b_frag_row, &sQ[k * WMMA * D + d_tile * WMMA], D);
                wmma::mma_sync(dK_frag[n], a_frag_col, b_frag_row, dK_frag[n]);
            }
        }
        __syncthreads();
    }

    // Store dV
    float* sdV_tmp = (float*)sQ;
    #pragma unroll
    for (int n = 0; n < DTILES; n++)
        wmma::store_matrix_sync(&sdV_tmp[row_base * D + (d_tile_start + n) * WMMA], dV_frag[n], D, wmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < BC * D; idx += THREADS) {
        int j = idx / D, k = idx % D;
        if (j < bc) dV_bh[(kj_start + j) * D + k] = __float2bfloat16(sdV_tmp[idx]);
    }
    __syncthreads();

    // Store dK
    #pragma unroll
    for (int n = 0; n < DTILES; n++)
        wmma::store_matrix_sync(&sdV_tmp[row_base * D + (d_tile_start + n) * WMMA], dK_frag[n], D, wmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < BC * D; idx += THREADS) {
        int j = idx / D, k = idx % D;
        if (j < bc) dK_bh[(kj_start + j) * D + k] = __float2bfloat16(sdV_tmp[idx]);
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
    {   int nqb = (S + BR - 1) / BR;
        size_t sz = (size_t)4*BR*D*sizeof(__nv_bfloat16) + BR*BC*sizeof(float) +
                    BR*BC*sizeof(__nv_bfloat16) + 2*BR*sizeof(float);
        CUDA_CHECK(cudaFuncSetAttribute(compute_dQ_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sz));
        compute_dQ_kernel<<<B*H*nqb, THREADS, sz, stream>>>(
            Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_buf, dQ_ptr, S, scale);
    }
    {   int nkb = (S + BC - 1) / BC;
        size_t sz = (size_t)2*BC*D*sizeof(__nv_bfloat16) + 2*BR*D*sizeof(__nv_bfloat16) +
                    BR*BC*sizeof(float) + BR*BC*sizeof(__nv_bfloat16) + 2*BR*sizeof(float);
        CUDA_CHECK(cudaFuncSetAttribute(compute_dKV_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sz));
        compute_dKV_kernel<<<B*H*nkb, THREADS, sz, stream>>>(
            Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_buf, dK_ptr, dV_ptr, S, scale);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(D_buf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128