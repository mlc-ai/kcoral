#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
    } \
} while(0)

namespace mha_bwd {

using bf16 = __nv_bfloat16;

#define BK 64
#define BQ 64
#define HD 128
#define NT 128

// D[row] = sum_k O[row,k] * dO[row,k]
__global__ void compute_D_kernel(const bf16* __restrict__ O, const bf16* __restrict__ dO,
                                 float* __restrict__ Dout, int total_rows) {
    int row = blockIdx.x;
    if (row >= total_rows) return;
    int tid = threadIdx.x;
    float v = __bfloat162float(O[(size_t)row*HD + tid]) * __bfloat162float(dO[(size_t)row*HD + tid]);
    __shared__ float sm[NT];
    sm[tid] = v;
    __syncthreads();
    for (int s = NT/2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid+s];
        __syncthreads();
    }
    if (tid == 0) Dout[row] = sm[0];
}

// Parallel over key blocks: compute dK, dV
__global__ void __launch_bounds__(NT) bwd_dkv_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
    const bf16* __restrict__ dO, const float* __restrict__ Lv, const float* __restrict__ Dv,
    bf16* __restrict__ dK, bf16* __restrict__ dV, int S, float scale)
{
    int bh = blockIdx.y;
    int kb = blockIdx.x;
    int kj0 = kb*BK;
    if (kj0 >= S) return;
    int tid = threadIdx.x;

    const bf16* Kp  = K   + (size_t)bh*S*HD;
    const bf16* Vp  = V   + (size_t)bh*S*HD;
    const bf16* Qp  = Q   + (size_t)bh*S*HD;
    const bf16* dOp = dO  + (size_t)bh*S*HD;
    const float* Lp = Lv  + (size_t)bh*S;
    const float* Dp = Dv  + (size_t)bh*S;
    bf16* dKp = dK + (size_t)bh*S*HD;
    bf16* dVp = dV + (size_t)bh*S*HD;

    extern __shared__ char smem_raw[];
    bf16* Ks   = (bf16*)smem_raw;
    bf16* Vs   = Ks + BK*HD;
    bf16* Qs   = Vs + BK*HD;
    bf16* dOs  = Qs + BQ*HD;
    float* Ls  = (float*)(dOs + BQ*HD);
    float* Ds  = Ls + BQ;
    float* Pt  = Ds + BQ;         // Pt[j*BQ + i] = P_{query i, key j}
    float* dSt = Pt + BK*BQ;      // dSt[j*BQ + i] = dS_{i,j}
    float* dKacc = dSt + BK*BQ;
    float* dVacc = dKacc + BK*HD;

    for (int idx = tid; idx < BK*HD; idx += NT) {
        int j = idx / HD, k = idx % HD;
        int gj = kj0 + j;
        if (gj < S) { Ks[idx] = Kp[(size_t)gj*HD + k]; Vs[idx] = Vp[(size_t)gj*HD + k]; }
        else { Ks[idx] = __float2bfloat16(0.f); Vs[idx] = __float2bfloat16(0.f); }
        dKacc[idx] = 0.f; dVacc[idx] = 0.f;
    }
    __syncthreads();

    int numQB = (S + BQ - 1)/BQ;
    for (int qb = kb; qb < numQB; qb++) {
        int qi0 = qb*BQ;
        for (int idx = tid; idx < BQ*HD; idx += NT) {
            int i = idx / HD, k = idx % HD;
            int gi = qi0 + i;
            if (gi < S) { Qs[idx] = Qp[(size_t)gi*HD + k]; dOs[idx] = dOp[(size_t)gi*HD + k]; }
            else { Qs[idx] = __float2bfloat16(0.f); dOs[idx] = __float2bfloat16(0.f); }
        }
        for (int idx = tid; idx < BQ; idx += NT) {
            int gi = qi0 + idx;
            Ls[idx] = (gi < S) ? Lp[gi] : 0.f;
            Ds[idx] = (gi < S) ? Dp[gi] : 0.f;
        }
        __syncthreads();

        bool diag = (qb == kb);

        // S^T then P^T
        for (int e = tid; e < BK*BQ; e += NT) {
            int j = e / BQ, i = e % BQ;
            const bf16* kr = &Ks[j*HD];
            const bf16* qr = &Qs[i*HD];
            float acc = 0.f;
            #pragma unroll 8
            for (int k = 0; k < HD; k++) acc += __bfloat162float(kr[k]) * __bfloat162float(qr[k]);
            acc *= scale;
            int gi = qi0 + i, gj = kj0 + j;
            bool valid = (gi < S) && (gj < S);
            if (diag) valid = valid && (gj <= gi);
            Pt[e] = valid ? __expf(acc - Ls[i]) : 0.f;
        }
        __syncthreads();

        // dP^T then dS^T
        for (int e = tid; e < BK*BQ; e += NT) {
            int j = e / BQ, i = e % BQ;
            const bf16* vr  = &Vs[j*HD];
            const bf16* dor = &dOs[i*HD];
            float acc = 0.f;
            #pragma unroll 8
            for (int k = 0; k < HD; k++) acc += __bfloat162float(vr[k]) * __bfloat162float(dor[k]);
            dSt[e] = Pt[e] * (acc - Ds[i]);
        }
        __syncthreads();

        // accumulate dV, dK
        for (int e = tid; e < BK*HD; e += NT) {
            int j = e / HD, k = e % HD;
            float accV = 0.f, accK = 0.f;
            #pragma unroll 8
            for (int i = 0; i < BQ; i++) {
                float p  = Pt[j*BQ + i];
                float ds = dSt[j*BQ + i];
                accV += p  * __bfloat162float(dOs[i*HD + k]);
                accK += ds * __bfloat162float(Qs[i*HD + k]);
            }
            dVacc[e] += accV;
            dKacc[e] += scale * accK;
        }
        __syncthreads();
    }

    for (int e = tid; e < BK*HD; e += NT) {
        int j = e / HD, k = e % HD;
        int gj = kj0 + j;
        if (gj < S) {
            dKp[(size_t)gj*HD + k] = __float2bfloat16(dKacc[e]);
            dVp[(size_t)gj*HD + k] = __float2bfloat16(dVacc[e]);
        }
    }
}

// Parallel over query blocks: compute dQ
__global__ void __launch_bounds__(NT) bwd_dq_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
    const bf16* __restrict__ dO, const float* __restrict__ Lv, const float* __restrict__ Dv,
    bf16* __restrict__ dQ, int S, float scale)
{
    int bh = blockIdx.y;
    int qb = blockIdx.x;
    int qi0 = qb*BQ;
    if (qi0 >= S) return;
    int tid = threadIdx.x;

    const bf16* Kp  = K   + (size_t)bh*S*HD;
    const bf16* Vp  = V   + (size_t)bh*S*HD;
    const bf16* Qp  = Q   + (size_t)bh*S*HD;
    const bf16* dOp = dO  + (size_t)bh*S*HD;
    const float* Lp = Lv  + (size_t)bh*S;
    const float* Dp = Dv  + (size_t)bh*S;
    bf16* dQp = dQ + (size_t)bh*S*HD;

    extern __shared__ char smem_raw[];
    bf16* Qs   = (bf16*)smem_raw;
    bf16* dOs  = Qs + BQ*HD;
    bf16* Ks   = dOs + BQ*HD;
    bf16* Vs   = Ks + BK*HD;
    float* Ls  = (float*)(Vs + BK*HD);
    float* Ds  = Ls + BQ;
    float* Sp  = Ds + BQ;         // Sp[i*BK + j] = P_{i,j}
    float* dSp = Sp + BQ*BK;      // dSp[i*BK + j] = dS_{i,j}
    float* dQacc = dSp + BQ*BK;

    for (int idx = tid; idx < BQ*HD; idx += NT) {
        int i = idx / HD, k = idx % HD;
        int gi = qi0 + i;
        if (gi < S) { Qs[idx] = Qp[(size_t)gi*HD + k]; dOs[idx] = dOp[(size_t)gi*HD + k]; }
        else { Qs[idx] = __float2bfloat16(0.f); dOs[idx] = __float2bfloat16(0.f); }
        dQacc[idx] = 0.f;
    }
    for (int idx = tid; idx < BQ; idx += NT) {
        int gi = qi0 + idx;
        Ls[idx] = (gi < S) ? Lp[gi] : 0.f;
        Ds[idx] = (gi < S) ? Dp[gi] : 0.f;
    }
    __syncthreads();

    for (int kb = 0; kb <= qb; kb++) {
        int kj0 = kb*BK;
        for (int idx = tid; idx < BK*HD; idx += NT) {
            int j = idx / HD, k = idx % HD;
            int gj = kj0 + j;
            if (gj < S) { Ks[idx] = Kp[(size_t)gj*HD + k]; Vs[idx] = Vp[(size_t)gj*HD + k]; }
            else { Ks[idx] = __float2bfloat16(0.f); Vs[idx] = __float2bfloat16(0.f); }
        }
        __syncthreads();

        bool diag = (kb == qb);

        for (int e = tid; e < BQ*BK; e += NT) {
            int i = e / BK, j = e % BK;
            const bf16* qr = &Qs[i*HD];
            const bf16* kr = &Ks[j*HD];
            float acc = 0.f;
            #pragma unroll 8
            for (int k = 0; k < HD; k++) acc += __bfloat162float(qr[k]) * __bfloat162float(kr[k]);
            acc *= scale;
            int gi = qi0 + i, gj = kj0 + j;
            bool valid = (gi < S) && (gj < S);
            if (diag) valid = valid && (gj <= gi);
            Sp[e] = valid ? __expf(acc - Ls[i]) : 0.f;
        }
        __syncthreads();

        for (int e = tid; e < BQ*BK; e += NT) {
            int i = e / BK, j = e % BK;
            const bf16* dor = &dOs[i*HD];
            const bf16* vr  = &Vs[j*HD];
            float acc = 0.f;
            #pragma unroll 8
            for (int k = 0; k < HD; k++) acc += __bfloat162float(dor[k]) * __bfloat162float(vr[k]);
            dSp[e] = Sp[e] * (acc - Ds[i]);
        }
        __syncthreads();

        for (int e = tid; e < BQ*HD; e += NT) {
            int i = e / HD, k = e % HD;
            float acc = 0.f;
            #pragma unroll 8
            for (int j = 0; j < BK; j++) {
                acc += dSp[i*BK + j] * __bfloat162float(Ks[j*HD + k]);
            }
            dQacc[e] += scale * acc;
        }
        __syncthreads();
    }

    for (int e = tid; e < BQ*HD; e += NT) {
        int i = e / HD, k = e % HD;
        int gi = qi0 + i;
        if (gi < S) dQp[(size_t)gi*HD + k] = __float2bfloat16(dQacc[e]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = Q.size(0), H = Q.size(1), S = Q.size(2), d = Q.size(3);
    float scale = 1.0f / sqrtf((float)d);

    const bf16* Qp  = (const bf16*)Q.data_ptr();
    const bf16* Kp  = (const bf16*)K.data_ptr();
    const bf16* Vp  = (const bf16*)V.data_ptr();
    const bf16* Op  = (const bf16*)O.data_ptr();
    const bf16* dOp = (const bf16*)dO.data_ptr();
    const float* Lp = (const float*)L.data_ptr();
    bf16* dQp = (bf16*)dQ.data_ptr();
    bf16* dKp = (bf16*)dK.data_ptr();
    bf16* dVp = (bf16*)dV.data_ptr();

    cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id);

    float* Dscr = nullptr;
    CUDA_CHECK(cudaMallocAsync((void**)&Dscr, (size_t)B*H*S*sizeof(float), stream));

    int total_rows = B*H*S;
    compute_D_kernel<<<total_rows, HD, 0, stream>>>(Op, dOp, Dscr, total_rows);
    CUDA_CHECK(cudaGetLastError());

    size_t sm1 = (size_t)(4*BK*HD)*sizeof(bf16) + (size_t)(2*BQ)*sizeof(float)
                 + (size_t)(2*BK*BQ)*sizeof(float) + (size_t)(2*BK*HD)*sizeof(float);
    size_t sm2 = (size_t)(4*BQ*HD)*sizeof(bf16) + (size_t)(2*BQ)*sizeof(float)
                 + (size_t)(2*BQ*BK)*sizeof(float) + (size_t)(BQ*HD)*sizeof(float);

    cudaFuncSetAttribute(bwd_dkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm1);
    cudaFuncSetAttribute(bwd_dq_kernel,  cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm2);

    int numKB = (S + BK - 1)/BK;
    dim3 g1(numKB, B*H);
    bwd_dkv_kernel<<<g1, NT, sm1, stream>>>(Qp, Kp, Vp, dOp, Lp, Dscr, dKp, dVp, S, scale);
    CUDA_CHECK(cudaGetLastError());

    int numQB = (S + BQ - 1)/BQ;
    dim3 g2(numQB, B*H);
    bwd_dq_kernel<<<g2, NT, sm2, stream>>>(Qp, Kp, Vp, dOp, Lp, Dscr, dQp, S, scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(Dscr, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd