#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);exit(1);} } while(0)

namespace tvm_ffi_attn_bwd {

constexpr int BQ = 32;
constexpr int BKV = 64;
constexpr int D = 128;
constexpr int THREADS = 256;

struct Smem {
    __nv_bfloat16 Q[BQ*D];
    __nv_bfloat16 K[BKV*D];
    __nv_bfloat16 V[BKV*D];
    __nv_bfloat16 O[BQ*D];
    __nv_bfloat16 dO[BQ*D];
    float L[BQ];
    float Dm[BQ];
    float S[BQ*BKV];
    float P[BQ*BKV];
    float dP[BQ*BKV];
    float dS[BQ*BKV];
    float dQ[BQ*D];
};

__global__ void attn_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    float* __restrict__ dK_fp,
    float* __restrict__ dV_fp,
    int S, int H, float scale)
{
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int64_t base = ((int64_t)(b*H + h)) * S * D;
    int64_t base_lse = ((int64_t)(b*H + h)) * S;

    extern __shared__ __align__(16) char smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);

    int tid = threadIdx.x;
    int nqb = (S + BQ - 1) / BQ;

    for (int ib = 0; ib < nqb; ib++) {
        int gi0 = ib * BQ;
        int BQ_a = (gi0 + BQ <= S) ? BQ : (S - gi0);
        if (BQ_a <= 0) break;

        for (int idx = tid; idx < BQ*D; idx += THREADS) sm.dQ[idx] = 0.f;
        for (int idx = tid; idx < BQ_a*D; idx += THREADS) {
            int i = idx / D, dd = idx % D;
            int gi = gi0 + i;
            sm.Q[i*D+dd]  = Q[base + (int64_t)gi*D + dd];
            sm.O[i*D+dd]  = O[base + (int64_t)gi*D + dd];
            sm.dO[i*D+dd] = dO[base + (int64_t)gi*D + dd];
        }
        for (int i = tid; i < BQ_a; i += THREADS) sm.L[i] = L[base_lse + gi0 + i];
        __syncthreads();

        if (tid < BQ_a) {
            float s = 0.f;
            for (int d = 0; d < D; d++)
                s += __bfloat162float(sm.dO[tid*D+d]) * __bfloat162float(sm.O[tid*D+d]);
            sm.Dm[tid] = s;
        }
        __syncthreads();

        int maxjb = (gi0 + BQ_a - 1) / BKV;
        for (int jb = 0; jb <= maxjb; jb++) {
            int gj0 = jb * BKV;
            int BKV_a = (gj0 + BKV <= S) ? BKV : (S - gj0);
            if (BKV_a <= 0) break;

            for (int idx = tid; idx < BKV_a*D; idx += THREADS) {
                int k = idx / D, dd = idx % D;
                int gk = gj0 + k;
                sm.K[k*D+dd] = K[base + (int64_t)gk*D + dd];
                sm.V[k*D+dd] = V[base + (int64_t)gk*D + dd];
            }
            __syncthreads();

            for (int n = 0; n < 8; n++) {
                int idx = tid + n*THREADS;
                int i = idx / BKV, k = idx % BKV;
                if (i >= BQ_a || k >= BKV_a) continue;
                int gi = gi0 + i, gj = gj0 + k;
                float s = 0.f;
                for (int d = 0; d < D; d++)
                    s += __bfloat162float(sm.Q[i*D+d]) * __bfloat162float(sm.K[k*D+d]);
                s *= scale;
                if (gj > gi) s = -INFINITY;
                sm.S[i*BKV+k] = s;
            }
            __syncthreads();

            for (int idx = tid; idx < BQ_a*BKV_a; idx += THREADS) {
                int i = idx / BKV, k = idx % BKV;
                sm.P[i*BKV+k] = expf(sm.S[i*BKV+k] - sm.L[i]);
            }
            __syncthreads();

            for (int n = 0; n < 8; n++) {
                int idx = tid + n*THREADS;
                int i = idx / BKV, k = idx % BKV;
                if (i >= BQ_a || k >= BKV_a) continue;
                float s = 0.f;
                for (int d = 0; d < D; d++)
                    s += __bfloat162float(sm.dO[i*D+d]) * __bfloat162float(sm.V[k*D+d]);
                sm.dP[i*BKV+k] = s;
            }
            __syncthreads();

            for (int idx = tid; idx < BQ_a*BKV_a; idx += THREADS) {
                int i = idx / BKV, k = idx % BKV;
                sm.dS[i*BKV+k] = sm.P[i*BKV+k] * (sm.dP[i*BKV+k] - sm.Dm[i]);
            }
            __syncthreads();

            for (int n = 0; n < 16; n++) {
                int idx = tid + n*THREADS;
                int i = idx / D, dd = idx % D;
                if (i >= BQ_a) continue;
                float v = 0.f;
                for (int k = 0; k < BKV_a; k++)
                    v += sm.dS[i*BKV+k] * __bfloat162float(sm.K[k*D+dd]);
                sm.dQ[i*D+dd] += v * scale;
            }
            __syncthreads();

            for (int n = 0; n < 32; n++) {
                int idx = tid + n*THREADS;
                int k = idx / D, dd = idx % D;
                if (k >= BKV_a) continue;
                float v = 0.f;
                for (int i = 0; i < BQ_a; i++) {
                    int gi = gi0 + i, gj = gj0 + k;
                    if (gj > gi) continue;
                    v += sm.dS[i*BKV+k] * __bfloat162float(sm.Q[i*D+dd]);
                }
                v *= scale;
                int gk = gj0 + k;
                atomicAdd(&dK_fp[base + (int64_t)gk*D + dd], v);
            }

            for (int n = 0; n < 32; n++) {
                int idx = tid + n*THREADS;
                int k = idx / D, dd = idx % D;
                if (k >= BKV_a) continue;
                float v = 0.f;
                for (int i = 0; i < BQ_a; i++) {
                    int gi = gi0 + i, gj = gj0 + k;
                    if (gj > gi) continue;
                    v += sm.P[i*BKV+k] * __bfloat162float(sm.dO[i*D+dd]);
                }
                int gk = gj0 + k;
                atomicAdd(&dV_fp[base + (int64_t)gk*D + dd], v);
            }
            __syncthreads();
        }

        for (int n = 0; n < 16; n++) {
            int idx = tid + n*THREADS;
            int i = idx / D, dd = idx % D;
            if (i >= BQ_a) continue;
            int gi = gi0 + i;
            dQ[base + (int64_t)gi*D + dd] = __float2bfloat16(sm.dQ[i*D+dd]);
        }
        __syncthreads();
    }
}

__global__ void convert_kernel(const float* __restrict__ in, __nv_bfloat16* __restrict__ out, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2bfloat16(in[idx]);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = 4, H = 48, d = 128;
    int S = (int)Q.size(2);
    int64_t total = (int64_t)B * H * S * d;
    float scale = 1.f / sqrtf((float)d);

    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* Op = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dOp = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* Lp = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQp = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dKp = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dVp = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* dK_fp = nullptr;
    float* dV_fp = nullptr;
    CUDA_CHECK(cudaMallocAsync(&dK_fp, sizeof(float)*total, stream));
    CUDA_CHECK(cudaMallocAsync(&dV_fp, sizeof(float)*total, stream));
    CUDA_CHECK(cudaMemsetAsync(dK_fp, 0, sizeof(float)*total, stream));
    CUDA_CHECK(cudaMemsetAsync(dV_fp, 0, sizeof(float)*total, stream));

    int smem_size = sizeof(Smem);
    CUDA_CHECK(cudaFuncSetAttribute(attn_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid(B*H);
    dim3 block(THREADS);
    attn_bwd_kernel<<<grid, block, smem_size, stream>>>(Qp, Kp, Vp, Op, dOp, Lp, dQp, dK_fp, dV_fp, S, H, scale);
    CUDA_CHECK(cudaGetLastError());

    int64_t conv_total = total;
    int cb = 256;
    int64_t cg = (conv_total + cb - 1) / cb;
    convert_kernel<<<(int)cg, cb, 0, stream>>>(dK_fp, dKp, conv_total);
    convert_kernel<<<(int)cg, cb, 0, stream>>>(dV_fp, dVp, conv_total);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(dK_fp, stream));
    CUDA_CHECK(cudaFreeAsync(dV_fp, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_attn_bwd::run);

}  // namespace tvm_ffi_attn_bwd