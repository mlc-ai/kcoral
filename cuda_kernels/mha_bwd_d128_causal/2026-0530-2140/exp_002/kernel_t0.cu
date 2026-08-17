#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd_d128_causal {

using bf16 = __nv_bfloat16;

constexpr int Hc = 48;
constexpr int Dc = 128;
constexpr int BM = 64;   // query tile
constexpr int BN = 64;   // key tile

// ---------------- D = rowsum(dO ⊙ O) ----------------
__global__ void compute_D_kernel(const bf16* __restrict__ O,
                                 const bf16* __restrict__ dO,
                                 float* __restrict__ D, int S) {
    long row = (long)blockIdx.x;          // global row index in [0, B*H*S)
    int tid = threadIdx.x;                // 0..127
    const bf16* o = O + row * Dc;
    const bf16* g = dO + row * Dc;
    float v = __bfloat162float(o[tid]) * __bfloat162float(g[tid]);
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    __shared__ float ws[4];
    int warp = tid >> 5, lane = tid & 31;
    if (lane == 0) ws[warp] = v;
    __syncthreads();
    if (tid == 0) D[row] = ws[0] + ws[1] + ws[2] + ws[3];
}

// ---------------- dK / dV kernel ----------------
__launch_bounds__(128)
__global__ void dkdv_kernel(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                            const bf16* __restrict__ V, const bf16* __restrict__ dO,
                            const float* __restrict__ L, const float* __restrict__ D,
                            bf16* __restrict__ dK, bf16* __restrict__ dV,
                            int S, float scale) {
    extern __shared__ char smem[];
    bf16* Ksh  = (bf16*)smem;             // [BN][Dc]
    bf16* Vsh  = Ksh + BN * Dc;           // [BN][Dc]
    bf16* Qsh  = Vsh + BN * Dc;           // [BM][Dc]
    bf16* dOsh = Qsh + BM * Dc;           // [BM][Dc]
    float* St  = (float*)(dOsh + BM * Dc);// [BN][BM]  (P^T)
    float* dSt = St + BN * BM;            // [BN][BM]  (dS^T)
    float* Lsh = dSt + BN * BM;           // [BM]
    float* Dsh = Lsh + BM;                // [BM]

    int kv_tile = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;                // = head-dim column
    int kv_start = kv_tile * BN;

    long bh = (long)(b * Hc + h);
    const bf16* Qb  = Q  + bh * S * Dc;
    const bf16* Kb  = K  + bh * S * Dc;
    const bf16* Vb  = V  + bh * S * Dc;
    const bf16* dOb = dO + bh * S * Dc;
    const float* Lb = L  + bh * S;
    const float* Db = D  + bh * S;
    bf16* dKb = dK + bh * S * Dc;
    bf16* dVb = dV + bh * S * Dc;

    // load K, V tile (each thread loads its column for all rows)
    for (int i = 0; i < BN; i++) {
        int kg = kv_start + i;
        bf16 z = __float2bfloat16(0.f);
        Ksh[i * Dc + tid] = (kg < S) ? Kb[(long)kg * Dc + tid] : z;
        Vsh[i * Dc + tid] = (kg < S) ? Vb[(long)kg * Dc + tid] : z;
    }
    __syncthreads();

    float dKacc[BN], dVacc[BN];
    #pragma unroll
    for (int n = 0; n < BN; n++) { dKacc[n] = 0.f; dVacc[n] = 0.f; }

    int numQ = (S + BM - 1) / BM;
    for (int qt = kv_tile; qt < numQ; qt++) {
        int q_start = qt * BM;
        for (int i = 0; i < BM; i++) {
            int qg = q_start + i;
            bf16 z = __float2bfloat16(0.f);
            Qsh[i * Dc + tid]  = (qg < S) ? Qb[(long)qg * Dc + tid]  : z;
            dOsh[i * Dc + tid] = (qg < S) ? dOb[(long)qg * Dc + tid] : z;
        }
        if (tid < BM) {
            int qg = q_start + tid;
            Lsh[tid] = (qg < S) ? Lb[qg] : 0.f;
            Dsh[tid] = (qg < S) ? Db[qg] : 0.f;
        }
        __syncthreads();

        // Step A+B: S^T -> P^T
        for (int o = tid; o < BN * BM; o += 128) {
            int n = o / BM, m = o % BM;
            const __nv_bfloat162* Kr = (const __nv_bfloat162*)(Ksh + n * Dc);
            const __nv_bfloat162* Qr = (const __nv_bfloat162*)(Qsh + m * Dc);
            float acc = 0.f;
            #pragma unroll
            for (int k = 0; k < Dc / 2; k++) {
                float2 a = __bfloat1622float2(Kr[k]);
                float2 q = __bfloat1622float2(Qr[k]);
                acc += a.x * q.x + a.y * q.y;
            }
            acc *= scale;
            int kg = kv_start + n, qg = q_start + m;
            float p = (kg < S && qg < S && kg <= qg) ? __expf(acc - Lsh[m]) : 0.f;
            St[n * BM + m] = p;
        }
        __syncthreads();

        // Step D+E: dP^T -> dS^T
        for (int o = tid; o < BN * BM; o += 128) {
            int n = o / BM, m = o % BM;
            const __nv_bfloat162* Vr = (const __nv_bfloat162*)(Vsh + n * Dc);
            const __nv_bfloat162* Or = (const __nv_bfloat162*)(dOsh + m * Dc);
            float acc = 0.f;
            #pragma unroll
            for (int k = 0; k < Dc / 2; k++) {
                float2 v = __bfloat1622float2(Vr[k]);
                float2 g = __bfloat1622float2(Or[k]);
                acc += v.x * g.x + v.y * g.y;
            }
            float p = St[n * BM + m];
            dSt[n * BM + m] = p * (acc - Dsh[m]);
        }
        __syncthreads();

        // Step C+F: accumulate dV and dK (thread owns column = tid)
        for (int m = 0; m < BM; m++) {
            float dval = __bfloat162float(dOsh[m * Dc + tid]);
            float qval = __bfloat162float(Qsh[m * Dc + tid]);
            #pragma unroll
            for (int n = 0; n < BN; n++) {
                dVacc[n] += St[n * BM + m]  * dval;
                dKacc[n] += dSt[n * BM + m] * qval;
            }
        }
        __syncthreads();
    }

    // write
    for (int n = 0; n < BN; n++) {
        int kg = kv_start + n;
        if (kg < S) {
            dKb[(long)kg * Dc + tid] = __float2bfloat16(scale * dKacc[n]);
            dVb[(long)kg * Dc + tid] = __float2bfloat16(dVacc[n]);
        }
    }
}

// ---------------- dQ kernel ----------------
__launch_bounds__(128)
__global__ void dq_kernel(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                          const bf16* __restrict__ V, const bf16* __restrict__ dO,
                          const float* __restrict__ L, const float* __restrict__ D,
                          bf16* __restrict__ dQ, int S, float scale) {
    extern __shared__ char smem[];
    bf16* Ksh  = (bf16*)smem;             // [BN][Dc]
    bf16* Vsh  = Ksh + BN * Dc;           // [BN][Dc]
    bf16* Qsh  = Vsh + BN * Dc;           // [BM][Dc]
    bf16* dOsh = Qsh + BM * Dc;           // [BM][Dc]
    float* St  = (float*)(dOsh + BM * Dc);// [BN][BM]
    float* dSt = St + BN * BM;            // [BN][BM]
    float* Lsh = dSt + BN * BM;           // [BM]
    float* Dsh = Lsh + BM;                // [BM]

    int q_tile = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;
    int q_start = q_tile * BM;

    long bh = (long)(b * Hc + h);
    const bf16* Qb  = Q  + bh * S * Dc;
    const bf16* Kb  = K  + bh * S * Dc;
    const bf16* Vb  = V  + bh * S * Dc;
    const bf16* dOb = dO + bh * S * Dc;
    const float* Lb = L  + bh * S;
    const float* Db = D  + bh * S;
    bf16* dQb = dQ + bh * S * Dc;

    // load Q, dO (once), L, D
    for (int i = 0; i < BM; i++) {
        int qg = q_start + i;
        bf16 z = __float2bfloat16(0.f);
        Qsh[i * Dc + tid]  = (qg < S) ? Qb[(long)qg * Dc + tid]  : z;
        dOsh[i * Dc + tid] = (qg < S) ? dOb[(long)qg * Dc + tid] : z;
    }
    if (tid < BM) {
        int qg = q_start + tid;
        Lsh[tid] = (qg < S) ? Lb[qg] : 0.f;
        Dsh[tid] = (qg < S) ? Db[qg] : 0.f;
    }
    __syncthreads();

    float dQacc[BM];
    #pragma unroll
    for (int m = 0; m < BM; m++) dQacc[m] = 0.f;

    for (int kt = 0; kt <= q_tile; kt++) {
        int kv_start = kt * BN;
        for (int i = 0; i < BN; i++) {
            int kg = kv_start + i;
            bf16 z = __float2bfloat16(0.f);
            Ksh[i * Dc + tid] = (kg < S) ? Kb[(long)kg * Dc + tid] : z;
            Vsh[i * Dc + tid] = (kg < S) ? Vb[(long)kg * Dc + tid] : z;
        }
        __syncthreads();

        // S^T -> P^T
        for (int o = tid; o < BN * BM; o += 128) {
            int n = o / BM, m = o % BM;
            const __nv_bfloat162* Kr = (const __nv_bfloat162*)(Ksh + n * Dc);
            const __nv_bfloat162* Qr = (const __nv_bfloat162*)(Qsh + m * Dc);
            float acc = 0.f;
            #pragma unroll
            for (int k = 0; k < Dc / 2; k++) {
                float2 a = __bfloat1622float2(Kr[k]);
                float2 q = __bfloat1622float2(Qr[k]);
                acc += a.x * q.x + a.y * q.y;
            }
            acc *= scale;
            int kg = kv_start + n, qg = q_start + m;
            float p = (kg < S && qg < S && kg <= qg) ? __expf(acc - Lsh[m]) : 0.f;
            St[n * BM + m] = p;
        }
        __syncthreads();

        // dP^T -> dS^T
        for (int o = tid; o < BN * BM; o += 128) {
            int n = o / BM, m = o % BM;
            const __nv_bfloat162* Vr = (const __nv_bfloat162*)(Vsh + n * Dc);
            const __nv_bfloat162* Or = (const __nv_bfloat162*)(dOsh + m * Dc);
            float acc = 0.f;
            #pragma unroll
            for (int k = 0; k < Dc / 2; k++) {
                float2 v = __bfloat1622float2(Vr[k]);
                float2 g = __bfloat1622float2(Or[k]);
                acc += v.x * g.x + v.y * g.y;
            }
            float p = St[n * BM + m];
            dSt[n * BM + m] = p * (acc - Dsh[m]);
        }
        __syncthreads();

        // accumulate dQ[m][tid] += sum_n dS^T[n][m] * K[n][tid]
        for (int n = 0; n < BN; n++) {
            float kval = __bfloat162float(Ksh[n * Dc + tid]);
            #pragma unroll
            for (int m = 0; m < BM; m++) {
                dQacc[m] += dSt[n * BM + m] * kval;
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < BM; m++) {
        int qg = q_start + m;
        if (qg < S) dQb[(long)qg * Dc + tid] = __float2bfloat16(scale * dQacc[m]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);

    const bf16* Qp  = static_cast<const bf16*>(Q.data_ptr());
    const bf16* Kp  = static_cast<const bf16*>(K.data_ptr());
    const bf16* Vp  = static_cast<const bf16*>(V.data_ptr());
    const bf16* Op  = static_cast<const bf16*>(O.data_ptr());
    const bf16* dOp = static_cast<const bf16*>(dO.data_ptr());
    const float* Lp = static_cast<const float*>(L.data_ptr());
    bf16* dQp = static_cast<bf16*>(dQ.data_ptr());
    bf16* dKp = static_cast<bf16*>(dK.data_ptr());
    bf16* dVp = static_cast<bf16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float scale = 1.0f / sqrtf((float)Dc);

    long rows = (long)B * H * S;
    float* Dp = nullptr;
    CUDA_CHECK(cudaMallocAsync(&Dp, rows * sizeof(float), stream));

    // compute D
    compute_D_kernel<<<(unsigned)rows, 128, 0, stream>>>(Op, dOp, Dp, S);

    size_t smem = (size_t)(BN * Dc + BN * Dc + BM * Dc + BM * Dc) * sizeof(bf16)
                + (size_t)(BN * BM + BN * BM + BM + BM) * sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute(dkdv_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    CUDA_CHECK(cudaFuncSetAttribute(dq_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

    int numKV = (S + BN - 1) / BN;
    int numQ  = (S + BM - 1) / BM;

    dim3 g1((unsigned)numKV, (unsigned)H, (unsigned)B);
    dkdv_kernel<<<g1, 128, smem, stream>>>(Qp, Kp, Vp, dOp, Lp, Dp, dKp, dVp, S, scale);

    dim3 g2((unsigned)numQ, (unsigned)H, (unsigned)B);
    dq_kernel<<<g2, 128, smem, stream>>>(Qp, Kp, Vp, dOp, Lp, Dp, dQp, S, scale);

    CUDA_CHECK(cudaFreeAsync(Dp, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal