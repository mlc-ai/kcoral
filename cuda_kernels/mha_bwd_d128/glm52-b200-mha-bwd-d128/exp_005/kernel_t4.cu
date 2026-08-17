#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <math.h>
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
} while(0)

namespace attn_bwd {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int D = 128;
constexpr int THREADS = 128;

__global__ void precompute_D_kernel(
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ O,
    float* __restrict__ D_out,
    int total_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < total_rows) {
        const __nv_bfloat16* dO_row = dO + (size_t)row * D;
        const __nv_bfloat16* O_row = O + (size_t)row * D;
        float sum = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; d += 8) {
            int4 dO_v = *(const int4*)(dO_row + d);
            int4 O_v = *(const int4*)(O_row + d);
            __nv_bfloat16* dO_h = (__nv_bfloat16*)&dO_v;
            __nv_bfloat16* O_h = (__nv_bfloat16*)&O_v;
            #pragma unroll
            for (int j = 0; j < 8; j++)
                sum += __bfloat162float(dO_h[j]) * __bfloat162float(O_h[j]);
        }
        D_out[row] = sum;
    }
}

__global__ __launch_bounds__(THREADS, 2)
void dkv_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_pre,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S)
{
    int bh = blockIdx.x;
    int kv_block = blockIdx.y;
    int h = bh % H;
    int b = bh / H;
    int kv_start = kv_block * BN;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int wr = warp_id / 2;
    int wc = warp_id % 2;

    const float scale = 1.0f / sqrtf((float)D);
    size_t bh_off = ((size_t)b * H + h) * S * D;
    size_t stat_off = ((size_t)b * H + h) * S;

    const __nv_bfloat16* Q_bh = Q + bh_off;
    const __nv_bfloat16* K_bh = K + bh_off;
    const __nv_bfloat16* V_bh = V + bh_off;
    const __nv_bfloat16* dO_bh = dO + bh_off;
    const float* L_bh = L + stat_off;
    const float* D_bh = D_pre + stat_off;
    __nv_bfloat16* dK_bh = dK + bh_off;
    __nv_bfloat16* dV_bh = dV + bh_off;

    extern __shared__ char smem[];
    char* ptr = smem;
    __nv_bfloat16* K_smem  = (__nv_bfloat16*)ptr;  ptr += BN * D * 2;
    __nv_bfloat16* V_smem  = (__nv_bfloat16*)ptr;  ptr += BN * D * 2;
    __nv_bfloat16* Q_smem  = (__nv_bfloat16*)ptr;  ptr += BM * D * 2;
    __nv_bfloat16* dO_smem = (__nv_bfloat16*)ptr;  ptr += BM * D * 2;
    float* temp_smem       = (float*)ptr;           ptr += BM * BN * 4;
    __nv_bfloat16* P_smem  = (__nv_bfloat16*)ptr;   ptr += BM * BN * 2;
    float* L_smem          = (float*)ptr;           ptr += BM * 4;
    float* D_smem          = (float*)ptr;

    for (int i = tid; i < BN * D / 8; i += THREADS) {
        int row = (i * 8) / D, col = (i * 8) % D;
        int gr = kv_start + row;
        if (gr < S) {
            *(int4*)(&K_smem[i*8])  = *(int4*)(&K_bh[gr*D + col]);
            *(int4*)(&V_smem[i*8])  = *(int4*)(&V_bh[gr*D + col]);
        } else {
            *(int4*)(&K_smem[i*8])  = make_int4(0,0,0,0);
            *(int4*)(&V_smem[i*8])  = make_int4(0,0,0,0);
        }
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dK_frag[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dV_frag[2][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(dK_frag[i][j], 0.0f);
            wmma::fill_fragment(dV_frag[i][j], 0.0f);
        }

    int num_q = (S + BM - 1) / BM;
    for (int qb = 0; qb < num_q; qb++) {
        int q_start = qb * BM;

        for (int i = tid; i < BM * D / 8; i += THREADS) {
            int row = (i * 8) / D, col = (i * 8) % D;
            int gr = q_start + row;
            if (gr < S) {
                *(int4*)(&Q_smem[i*8])  = *(int4*)(&Q_bh[gr*D + col]);
                *(int4*)(&dO_smem[i*8]) = *(int4*)(&dO_bh[gr*D + col]);
            } else {
                *(int4*)(&Q_smem[i*8])  = make_int4(0,0,0,0);
                *(int4*)(&dO_smem[i*8]) = make_int4(0,0,0,0);
            }
        }
        for (int i = tid; i < BM; i += THREADS) {
            int gr = q_start + i;
            L_smem[i] = (gr < S) ? L_bh[gr] : 0.0f;
            D_smem[i] = (gr < S) ? D_bh[gr] : 0.0f;
        }
        __syncthreads();

        // S = Q @ K^T
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cF[2][2];
            #pragma unroll
            for (int i = 0; i < 2; i++) #pragma unroll
                for (int j = 0; j < 2; j++) wmma::fill_fragment(cF[i][j], 0.0f);
            #pragma unroll
            for (int k = 0; k < D; k += 16) {
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    wmma::load_matrix_sync(a, &Q_smem[(wr*32+i*16)*D + k], D);
                    #pragma unroll
                    for (int j = 0; j < 2; j++) {
                        wmma::load_matrix_sync(b, &K_smem[(wc*32+j*16)*D + k], D);
                        wmma::mma_sync(cF[i][j], a, b, cF[i][j]);
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < 2; i++) #pragma unroll
                for (int j = 0; j < 2; j++)
                    wmma::store_matrix_sync(&temp_smem[(wr*32+i*16)*BN + wc*32+j*16], cF[i][j], BN, wmma::mem_row_major);
        }
        __syncthreads();

        for (int i = tid; i < BM * BN; i += THREADS) {
            int row = i / BN, col = i % BN;
            if (q_start+row < S && kv_start+col < S)
                P_smem[i] = __float2bfloat16(expf(temp_smem[i] * scale - L_smem[row]));
            else
                P_smem[i] = __float2bfloat16(0.0f);
        }
        __syncthreads();

        // dV += P^T @ dO
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            #pragma unroll
            for (int k = 0; k < BM; k += 16) {
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    wmma::load_matrix_sync(a, &P_smem[k*BN + wr*32+i*16], BN);
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        wmma::load_matrix_sync(b, &dO_smem[k*D + wc*64+j*16], D);
                        wmma::mma_sync(dV_frag[i][j], a, b, dV_frag[i][j]);
                    }
                }
            }
        }

        // dP = dO @ V^T
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cF[2][2];
            #pragma unroll
            for (int i = 0; i < 2; i++) #pragma unroll
                for (int j = 0; j < 2; j++) wmma::fill_fragment(cF[i][j], 0.0f);
            #pragma unroll
            for (int k = 0; k < D; k += 16) {
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    wmma::load_matrix_sync(a, &dO_smem[(wr*32+i*16)*D + k], D);
                    #pragma unroll
                    for (int j = 0; j < 2; j++) {
                        wmma::load_matrix_sync(b, &V_smem[(wc*32+j*16)*D + k], D);
                        wmma::mma_sync(cF[i][j], a, b, cF[i][j]);
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < 2; i++) #pragma unroll
                for (int j = 0; j < 2; j++)
                    wmma::store_matrix_sync(&temp_smem[(wr*32+i*16)*BN + wc*32+j*16], cF[i][j], BN, wmma::mem_row_major);
        }
        __syncthreads();

        // dS = P * (dP - D) * scale
        for (int i = tid; i < BM * BN; i += THREADS) {
            int row = i / BN;
            float p = __bfloat162float(P_smem[i]);
            P_smem[i] = __float2bfloat16(p * (temp_smem[i] - D_smem[row]) * scale);
        }
        __syncthreads();

        // dK += dS^T @ Q
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            #pragma unroll
            for (int k = 0; k < BM; k += 16) {
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    wmma::load_matrix_sync(a, &P_smem[k*BN + wr*32+i*16], BN);
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        wmma::load_matrix_sync(b, &Q_smem[k*D + wc*64+j*16], D);
                        wmma::mma_sync(dK_frag[i][j], a, b, dK_frag[i][j]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // Store dK, dV via two-pass staging
    float* stage = (float*)Q_smem;
    #pragma unroll
    for (int pass = 0; pass < 2; pass++) {
        if (wr == pass) {
            #pragma unroll
            for (int i = 0; i < 2; i++) #pragma unroll
                for (int j = 0; j < 4; j++)
                    wmma::store_matrix_sync(&stage[i*16*128 + wc*64+j*16], dK_frag[i][j], 128, wmma::mem_row_major);
        }
        __syncthreads();
        for (int idx = tid; idx < 32*128; idx += THREADS) {
            int row = idx/128, col = idx%128;
            int gr = kv_start + pass*32 + row;
            if (gr < S) dK_bh[gr*D + col] = __float2bfloat16(stage[idx]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int pass = 0; pass < 2; pass++) {
        if (wr == pass) {
            #pragma unroll
            for (int i = 0; i < 2; i++) #pragma unroll
                for (int j = 0; j < 4; j++)
                    wmma::store_matrix_sync(&stage[i*16*128 + wc*64+j*16], dV_frag[i][j], 128, wmma::mem_row_major);
        }
        __syncthreads();
        for (int idx = tid; idx < 32*128; idx += THREADS) {
            int row = idx/128, col = idx%128;
            int gr = kv_start + pass*32 + row;
            if (gr < S) dV_bh[gr*D + col] = __float2bfloat16(stage[idx]);
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(THREADS, 2)
void dq_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_pre,
    __nv_bfloat16* __restrict__ dQ,
    int B, int H, int S)
{
    int bh = blockIdx.x;
    int q_block = blockIdx.y;
    int h = bh % H;
    int b = bh / H;
    int q_start = q_block * BM;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int wr = warp_id / 2;
    int wc = warp_id % 2;

    const float scale = 1.0f / sqrtf((float)D);
    size_t bh_off = ((size_t)b * H + h) * S * D;
    size_t stat_off = ((size_t)b * H + h) * S;

    const __nv_bfloat16* Q_bh = Q + bh_off;
    const __nv_bfloat16* K_bh = K + bh_off;
    const __nv_bfloat16* V_bh = V + bh_off;
    const __nv_bfloat16* dO_bh = dO + bh_off;
    const float* L_bh = L + stat_off;
    const float* D_bh = D_pre + stat_off;
    __nv_bfloat16* dQ_bh = dQ + bh_off;

    extern __shared__ char smem[];
    char* ptr = smem;
    __nv_bfloat16* Q_smem  = (__nv_bfloat16*)ptr;  ptr += BM * D * 2;
    __nv_bfloat16* dO_smem = (__nv_bfloat16*)ptr;  ptr += BM * D * 2;
    __nv_bfloat16* K_smem  = (__nv_bfloat16*)ptr;  ptr += BN * D * 2;
    __nv_bfloat16* V_smem  = (__nv_bfloat16*)ptr;  ptr += BN * D * 2;
    float* temp_smem       = (float*)ptr;           ptr += BM * BN * 4;
    __nv_bfloat16* P_smem  = (__nv_bfloat16*)ptr;   ptr += BM * BN * 2;
    float* L_smem          = (float*)ptr;           ptr += BM * 4;
    float* D_smem          = (float*)ptr;

    for (int i = tid; i < BM * D / 8; i += THREADS) {
        int row = (i * 8) / D, col = (i * 8) % D;
        int gr = q_start + row;
        if (gr < S) {
            *(int4*)(&Q_smem[i*8])  = *(int4*)(&Q_bh[gr*D + col]);
            *(int4*)(&dO_smem[i*8]) = *(int4*)(&dO_bh[gr*D + col]);
        } else {
            *(int4*)(&Q_smem[i*8])  = make_int4(0,0,0,0);
            *(int4*)(&dO_smem[i*8]) = make_int4(0,0,0,0);
        }
    }
    for (int i = tid; i < BM; i += THREADS) {
        int gr = q_start + i;
        L_smem[i] = (gr < S) ? L_bh[gr] : 0.0f;
        D_smem[i] = (gr < S) ? D_bh[gr] : 0.0f;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dQ_frag[2][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(dQ_frag[i][j], 0.0f);

    __syncthreads();

    int num_kv = (S + BN - 1) / BN;
    for (int kvb = 0; kvb < num_kv; kvb++) {
        int kv_start = kvb * BN;

        for (int i = tid; i < BN * D / 8; i += THREADS) {
            int row = (i * 8) / D, col = (i * 8) % D;
            int gr = kv_start + row;
            if (gr < S) {
                *(int4*)(&K_smem[i*8]) = *(int4*)(&K_bh[gr*D + col]);
                *(int4*)(&V_smem[i*8]) = *(int4*)(&V_bh[gr*D + col]);
            } else {
                *(int4*)(&K_smem[i*8]) = make_int4(0,0,0,0);
                *(int4*)(&V_smem[i*8]) = make_int4(0,0,0,0);
            }
        }
        __syncthreads();

        // S = Q @ K^T
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cF[2][2];
            #pragma unroll
            for (int i = 0; i < 2; i++) #pragma unroll
                for (int j = 0; j < 2; j++) wmma::fill_fragment(cF[i][j], 0.0f);
            #pragma unroll
            for (int k = 0; k < D; k += 16) {
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    wmma::load_matrix_sync(a, &Q_smem[(wr*32+i*16)*D + k], D);
                    #pragma unroll
                    for (int j = 0; j < 2; j++) {
                        wmma::load_matrix_sync(b, &K_smem[(wc*32+j*16)*D + k], D);
                        wmma::mma_sync(cF[i][j], a, b, cF[i][j]);
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < 2; i++) #pragma unroll
                for (int j = 0; j < 2; j++)
                    wmma::store_matrix_sync(&temp_smem[(wr*32+i*16)*BN + wc*32+j*16], cF[i][j], BN, wmma::mem_row_major);
        }
        __syncthreads();

        for (int i = tid; i < BM * BN; i += THREADS) {
            int row = i / BN, col = i % BN;
            if (q_start+row < S && kv_start+col < S)
                P_smem[i] = __float2bfloat16(expf(temp_smem[i] * scale - L_smem[row]));
            else
                P_smem[i] = __float2bfloat16(0.0f);
        }
        __syncthreads();

        // dP = dO @ V^T
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cF[2][2];
            #pragma unroll
            for (int i = 0; i < 2; i++) #pragma unroll
                for (int j = 0; j < 2; j++) wmma::fill_fragment(cF[i][j], 0.0f);
            #pragma unroll
            for (int k = 0; k < D; k += 16) {
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    wmma::load_matrix_sync(a, &dO_smem[(wr*32+i*16)*D + k], D);
                    #pragma unroll
                    for (int j = 0; j < 2; j++) {
                        wmma::load_matrix_sync(b, &V_smem[(wc*32+j*16)*D + k], D);
                        wmma::mma_sync(cF[i][j], a, b, cF[i][j]);
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < 2; i++) #pragma unroll
                for (int j = 0; j < 2; j++)
                    wmma::store_matrix_sync(&temp_smem[(wr*32+i*16)*BN + wc*32+j*16], cF[i][j], BN, wmma::mem_row_major);
        }
        __syncthreads();

        // dS = P * (dP - D) * scale
        for (int i = tid; i < BM * BN; i += THREADS) {
            int row = i / BN;
            float p = __bfloat162float(P_smem[i]);
            P_smem[i] = __float2bfloat16(p * (temp_smem[i] - D_smem[row]) * scale);
        }
        __syncthreads();

        // dQ += dS @ K
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            #pragma unroll
            for (int k = 0; k < BN; k += 16) {
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    wmma::load_matrix_sync(a, &P_smem[(wr*32+i*16)*BN + k], BN);
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        wmma::load_matrix_sync(b, &K_smem[k*D + wc*64+j*16], D);
                        wmma::mma_sync(dQ_frag[i][j], a, b, dQ_frag[i][j]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // Store dQ via two-pass staging
    float* stage = (float*)K_smem;
    #pragma unroll
    for (int pass = 0; pass < 2; pass++) {
        if (wr == pass) {
            #pragma unroll
            for (int i = 0; i < 2; i++) #pragma unroll
                for (int j = 0; j < 4; j++)
                    wmma::store_matrix_sync(&stage[i*16*128 + wc*64+j*16], dQ_frag[i][j], 128, wmma::mem_row_major);
        }
        __syncthreads();
        for (int idx = tid; idx < 32*128; idx += THREADS) {
            int row = idx/128, col = idx%128;
            int gr = q_start + pass*32 + row;
            if (gr < S) dQ_bh[gr*D + col] = __float2bfloat16(stage[idx]);
        }
        __syncthreads();
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = 4, H = 48, S = static_cast<int>(Q.size(2));

    const __nv_bfloat16* Q_data  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_data  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_data = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_data          = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_data       = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_data       = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_data       = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int total_rows = B * H * S;
    float* D_pre = nullptr;
    CUDA_CHECK(cudaMalloc(&D_pre, total_rows * sizeof(float)));

    int d_threads = 256;
    int d_blocks = (total_rows + d_threads - 1) / d_threads;
    precompute_D_kernel<<<d_blocks, d_threads, 0, stream>>>(
        dO_data, O_data, D_pre, total_rows);

    int smem_size = BN*D*2*2 + BM*D*2*2 + BM*BN*4 + BM*BN*2 + BM*4*2;

    CUDA_CHECK(cudaFuncSetAttribute(dkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    int num_kv = (S + BN - 1) / BN;
    dim3 grid1(B * H, num_kv);
    dkv_kernel<<<grid1, THREADS, smem_size, stream>>>(
        Q_data, K_data, V_data, dO_data, L_data, D_pre,
        dK_data, dV_data, B, H, S);

    int num_q = (S + BM - 1) / BM;
    dim3 grid2(B * H, num_q);
    dq_kernel<<<grid2, THREADS, smem_size, stream>>>(
        Q_data, K_data, V_data, dO_data, L_data, D_pre,
        dQ_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(D_pre));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_bwd::run);

}  // namespace attn_bwd