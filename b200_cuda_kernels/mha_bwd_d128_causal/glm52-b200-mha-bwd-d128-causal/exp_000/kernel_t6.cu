#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

constexpr int D = 128;
constexpr int BQ = 64;
constexpr int BK = 16;
constexpr int WM = 16, WN = 16, WK = 16;
constexpr int WARPS = 4;
constexpr int THREADS = 128;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xFFFFFFFF, val, o);
    return val;
}

__global__ void compute_D_kernel(
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ O,
    float* __restrict__ Dbuf, int S)
{
    int bh = blockIdx.x;
    int qi = blockIdx.y * WARPS + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (qi >= S) return;
    int base = bh * S * D + qi * D;
    float s = 0;
    #pragma unroll
    for (int e = 0; e < 4; e++)
        s += __bfloat162float(dO[base + lane*4 + e]) * __bfloat162float(O[base + lane*4 + e]);
    s = warp_reduce_sum(s);
    if (lane == 0) Dbuf[bh * S + qi] = s;
}

__global__ void dQ_kernel(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L, const float* __restrict__ Dbuf,
    __nv_bfloat16* __restrict__ dQ, int S)
{
    int bh = blockIdx.x;
    int q_start = blockIdx.y * BQ;
    int max_qi = min(q_start + BQ - 1, S - 1);
    int num_q = max_qi - q_start + 1;
    int wid = threadIdx.x / 32;

    __shared__ __nv_bfloat16 sQ[BQ * D];
    __shared__ __nv_bfloat16 sDO[BQ * D];
    __shared__ __nv_bfloat16 sK[BK * D];
    __shared__ __nv_bfloat16 sV[BK * D];
    __shared__ float sS[BQ * BK];
    __shared__ float sDOV[BQ * BK];
    __shared__ __nv_bfloat16 sDS[BQ * BK];
    __shared__ float sLD[BQ * 2];
    __shared__ float sOut[BQ * D];

    {
        int qb = bh * S * D + q_start * D;
        for (int i = threadIdx.x * 8; i < num_q * D; i += THREADS * 8) {
            *reinterpret_cast<uint4*>(&sQ[i]) = *reinterpret_cast<const uint4*>(&Q[qb + i]);
            *reinterpret_cast<uint4*>(&sDO[i]) = *reinterpret_cast<const uint4*>(&dO[qb + i]);
        }
        for (int i = num_q * D + threadIdx.x * 8; i < BQ * D; i += THREADS * 8) {
            *reinterpret_cast<uint4*>(&sQ[i]) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&sDO[i]) = make_uint4(0, 0, 0, 0);
        }
        for (int i = threadIdx.x; i < num_q; i += THREADS) {
            sLD[i] = L[bh * S + q_start + i];
            sLD[BQ + i] = Dbuf[bh * S + q_start + i];
        }
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> fBcm;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> fC;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> fBrm;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> dQacc[D / WN];
    #pragma unroll
    for (int n = 0; n < D / WN; n++) wmma::fill_fragment(dQacc[n], 0.0f);

    float scale = rsqrtf((float)D);

    for (int kt = 0; kt <= max_qi; kt += BK) {
        int tk = min(BK, max_qi - kt + 1);
        {
            int kb = bh * S * D + kt * D;
            for (int i = threadIdx.x * 8; i < tk * D; i += THREADS * 8) {
                *reinterpret_cast<uint4*>(&sK[i]) = *reinterpret_cast<const uint4*>(&K[kb + i]);
                *reinterpret_cast<uint4*>(&sV[i]) = *reinterpret_cast<const uint4*>(&V[kb + i]);
            }
            for (int i = tk * D + threadIdx.x * 8; i < BK * D; i += THREADS * 8) {
                *reinterpret_cast<uint4*>(&sK[i]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&sV[i]) = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // S = Q @ K^T
        wmma::fill_fragment(fC, 0.0f);
        #pragma unroll
        for (int k = 0; k < D; k += WK) {
            wmma::load_matrix_sync(fA, &sQ[wid * WM * D + k], D);
            wmma::load_matrix_sync(fBcm, &sK[k], D);
            wmma::mma_sync(fC, fA, fBcm, fC);
        }
        wmma::store_matrix_sync(&sS[wid * WM * BK], fC, BK, wmma::mem_row_major);
        __syncthreads();

        // dOV = dO @ V^T
        wmma::fill_fragment(fC, 0.0f);
        #pragma unroll
        for (int k = 0; k < D; k += WK) {
            wmma::load_matrix_sync(fA, &sDO[wid * WM * D + k], D);
            wmma::load_matrix_sync(fBcm, &sV[k], D);
            wmma::mma_sync(fC, fA, fBcm, fC);
        }
        wmma::store_matrix_sync(&sDOV[wid * WM * BK], fC, BK, wmma::mem_row_major);
        __syncthreads();

        // Softmax -> dS (with boundary masking)
        for (int idx = threadIdx.x; idx < BQ * BK; idx += THREADS) {
            int i = idx / BK, j = idx % BK;
            int qi = q_start + i, kj = kt + j;
            if (i >= num_q || kj > qi || kj >= S) {
                sDS[idx] = __float2bfloat16(0.0f);
            } else {
                float p = __expf(sS[idx] * scale - sLD[i]);
                float ds = p * (sDOV[idx] - sLD[BQ + i]);
                sDS[idx] = __float2bfloat16(ds);
            }
        }
        __syncthreads();

        // dQ += dS @ K
        wmma::load_matrix_sync(fA, &sDS[wid * WM * BK], BK);
        #pragma unroll
        for (int n = 0; n < D / WN; n++) {
            wmma::load_matrix_sync(fBrm, &sK[n * WN], D);
            wmma::mma_sync(dQacc[n], fA, fBrm, dQacc[n]);
        }
        __syncthreads();
    }

    // Store dQ
    #pragma unroll
    for (int n = 0; n < D / WN; n++)
        wmma::store_matrix_sync(&sOut[wid * WM * D + n * WN], dQacc[n], D, wmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x * 4; i < BQ * D; i += THREADS * 4) {
        int row = i / D, col = i % D;
        int qi_l = row;
        if (qi_l < num_q) {
            __nv_bfloat162 lo = __float22bfloat162_rn(make_float2(sOut[i], sOut[i+1]));
            __nv_bfloat162 hi = __float22bfloat162_rn(make_float2(sOut[i+2], sOut[i+3]));
            uint2 out = {*(uint*)&lo, *(uint*)&hi};
            *(uint2*)&dQ[bh * S * D + (q_start + qi_l) * D + col] = out;
        }
    }
}

__global__ void dKV_kernel(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L, const float* __restrict__ Dbuf,
    __nv_bfloat16* __restrict__ dK, __nv_bfloat16* __restrict__ dV, int S)
{
    int bh = blockIdx.x;
    int k_start = blockIdx.y * BK;
    int num_k = min(BK, S - k_start);
    int wid = threadIdx.x / 32;

    __shared__ __nv_bfloat16 sK[BK * D];
    __shared__ __nv_bfloat16 sV[BK * D];
    __shared__ __nv_bfloat16 sQ[BQ * D];
    __shared__ __nv_bfloat16 sDO[BQ * D];
    __shared__ float sST[BK * BQ];
    __shared__ float sDOVT[BK * BQ];
    __shared__ __nv_bfloat16 sDST[BK * BQ];
    __shared__ __nv_bfloat16 sPT[BK * BQ];
    __shared__ float sLD[BQ * 2];
    __shared__ float sOutK[BK * D];
    __shared__ float sOutV[BK * D];

    {
        int kb = bh * S * D + k_start * D;
        for (int i = threadIdx.x * 8; i < num_k * D; i += THREADS * 8) {
            *reinterpret_cast<uint4*>(&sK[i]) = *reinterpret_cast<const uint4*>(&K[kb + i]);
            *reinterpret_cast<uint4*>(&sV[i]) = *reinterpret_cast<const uint4*>(&V[kb + i]);
        }
        for (int i = num_k * D + threadIdx.x * 8; i < BK * D; i += THREADS * 8) {
            *reinterpret_cast<uint4*>(&sK[i]) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&sV[i]) = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> fBcm;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> fC;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> fBrm;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> dKacc[2], dVacc[2];
    #pragma unroll
    for (int n = 0; n < 2; n++) {
        wmma::fill_fragment(dKacc[n], 0.0f);
        wmma::fill_fragment(dVacc[n], 0.0f);
    }

    float scale = rsqrtf((float)D);

    for (int qt = k_start; qt < S; qt += BQ) {
        int tq = min(BQ, S - qt);
        {
            int qb = bh * S * D + qt * D;
            for (int i = threadIdx.x * 8; i < tq * D; i += THREADS * 8) {
                *reinterpret_cast<uint4*>(&sQ[i]) = *reinterpret_cast<const uint4*>(&Q[qb + i]);
                *reinterpret_cast<uint4*>(&sDO[i]) = *reinterpret_cast<const uint4*>(&dO[qb + i]);
            }
            for (int i = tq * D + threadIdx.x * 8; i < BQ * D; i += THREADS * 8) {
                *reinterpret_cast<uint4*>(&sQ[i]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&sDO[i]) = make_uint4(0, 0, 0, 0);
            }
            for (int i = threadIdx.x; i < tq; i += THREADS) {
                sLD[i] = L[bh * S + qt + i];
                sLD[BQ + i] = Dbuf[bh * S + qt + i];
            }
        }
        __syncthreads();

        // S_T = K @ Q^T
        wmma::fill_fragment(fC, 0.0f);
        #pragma unroll
        for (int k = 0; k < D; k += WK) {
            wmma::load_matrix_sync(fA, &sK[k], D);
            wmma::load_matrix_sync(fBcm, &sQ[wid * WM * D + k], D);
            wmma::mma_sync(fC, fA, fBcm, fC);
        }
        wmma::store_matrix_sync(&sST[wid * WN], fC, BQ, wmma::mem_row_major);
        __syncthreads();

        // dOV_T = V @ dO^T
        wmma::fill_fragment(fC, 0.0f);
        #pragma unroll
        for (int k = 0; k < D; k += WK) {
            wmma::load_matrix_sync(fA, &sV[k], D);
            wmma::load_matrix_sync(fBcm, &sDO[wid * WM * D + k], D);
            wmma::mma_sync(fC, fA, fBcm, fC);
        }
        wmma::store_matrix_sync(&sDOVT[wid * WN], fC, BQ, wmma::mem_row_major);
        __syncthreads();

        // Softmax -> dS_T, P_T (with boundary masking)
        for (int idx = threadIdx.x; idx < BK * BQ; idx += THREADS) {
            int j = idx / BQ, i = idx % BQ;
            int kj = k_start + j, qi = qt + i;
            if (i >= tq || qi < kj || kj >= S) {
                sDST[idx] = __float2bfloat16(0.0f);
                sPT[idx] = __float2bfloat16(0.0f);
            } else {
                float p = __expf(sST[idx] * scale - sLD[i]);
                float ds = p * (sDOVT[idx] - sLD[BQ + i]);
                sDST[idx] = __float2bfloat16(ds);
                sPT[idx] = __float2bfloat16(p);
            }
        }
        __syncthreads();

        // dK += dS_T @ Q, dV += P_T @ dO
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            int dt = wid * 32 + n * 16;
            #pragma unroll
            for (int k = 0; k < BQ; k += WK) {
                wmma::load_matrix_sync(fA, &sDST[k], BQ);
                wmma::load_matrix_sync(fBrm, &sQ[k * D + dt], D);
                wmma::mma_sync(dKacc[n], fA, fBrm, dKacc[n]);
                wmma::load_matrix_sync(fA, &sPT[k], BQ);
                wmma::load_matrix_sync(fBrm, &sDO[k * D + dt], D);
                wmma::mma_sync(dVacc[n], fA, fBrm, dVacc[n]);
            }
        }
        __syncthreads();
    }

    // Store dK
    #pragma unroll
    for (int n = 0; n < 2; n++) {
        int dt = wid * 32 + n * 16;
        wmma::store_matrix_sync(&sOutK[dt], dKacc[n], D, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = threadIdx.x * 4; i < BK * D; i += THREADS * 4) {
        int kj_l = i / D, d = i % D;
        if (kj_l < num_k) {
            __nv_bfloat162 lo = __float22bfloat162_rn(make_float2(sOutK[i], sOutK[i+1]));
            __nv_bfloat162 hi = __float22bfloat162_rn(make_float2(sOutK[i+2], sOutK[i+3]));
            uint2 out = {*(uint*)&lo, *(uint*)&hi};
            *(uint2*)&dK[bh * S * D + (k_start + kj_l) * D + d] = out;
        }
    }

    // Store dV
    #pragma unroll
    for (int n = 0; n < 2; n++) {
        int dt = wid * 32 + n * 16;
        wmma::store_matrix_sync(&sOutV[dt], dVacc[n], D, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = threadIdx.x * 4; i < BK * D; i += THREADS * 4) {
        int kj_l = i / D, d = i % D;
        if (kj_l < num_k) {
            __nv_bfloat162 lo = __float22bfloat162_rn(make_float2(sOutV[i], sOutV[i+1]));
            __nv_bfloat162 hi = __float22bfloat162_rn(make_float2(sOutV[i+2], sOutV[i+3]));
            uint2 out = {*(uint*)&lo, *(uint*)&hi};
            *(uint2*)&dV[bh * S * D + (k_start + kj_l) * D + d] = out;
        }
    }
}

namespace mha_bwd_d128_causal {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int S = static_cast<int>(Q.size(2));
    int BH = 4 * 48;

    const __nv_bfloat16 *Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16 *Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16 *Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16 *Op = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16 *dOp = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float *Lp = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16 *dQp = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16 *dKp = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16 *dVp = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* Dbuf = nullptr;
    CUDA_CHECK(cudaMalloc(&Dbuf, BH * S * sizeof(float)));

    dim3 gridD(BH, (S + WARPS - 1) / WARPS);
    dim3 gridQ(BH, (S + BQ - 1) / BQ);
    dim3 gridKV(BH, (S + BK - 1) / BK);
    dim3 block(THREADS);

    compute_D_kernel<<<gridD, block, 0, stream>>>(dOp, Op, Dbuf, S);
    dQ_kernel<<<gridQ, block, 0, stream>>>(Qp, Kp, Vp, dOp, Lp, Dbuf, dQp, S);
    dKV_kernel<<<gridKV, block, 0, stream>>>(Qp, Kp, Vp, dOp, Lp, Dbuf, dKp, dVp, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(Dbuf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal