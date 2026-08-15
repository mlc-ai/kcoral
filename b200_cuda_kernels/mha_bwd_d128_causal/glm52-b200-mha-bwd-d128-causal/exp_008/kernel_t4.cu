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
constexpr int THREADS = 128;

struct Smem {
    __nv_bfloat16 Q[BQ*D];
    __nv_bfloat16 O[BQ*D];
    __nv_bfloat16 dO[BQ*D];
    __nv_bfloat16 K[BKV*D];
    __nv_bfloat16 V[BKV*D];
    float P[BQ*BKV];
    float dS[BQ*BKV];
    float dQ[BQ*D];
    float L[BQ];
    float Dm[BQ];
};

__global__ void attn_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ_out,
    float* __restrict__ dK_fp,
    float* __restrict__ dV_fp,
    int S, int H, float scale)
{
    int qb = blockIdx.x;
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int64_t base = ((int64_t)(b*H + h)) * S * D;
    int64_t base_lse = ((int64_t)(b*H + h)) * S;

    int gi0 = qb * BQ;
    int BQ_a = (gi0 + BQ <= S) ? BQ : (S - gi0);
    if (BQ_a <= 0) return;

    extern __shared__ __align__(16) char smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
    int tid = threadIdx.x;

    // Vectorized loads for Q, O, dO
    {
        const int4* Q_g = reinterpret_cast<const int4*>(&Q[base + (int64_t)gi0*D]);
        const int4* O_g = reinterpret_cast<const int4*>(&O[base + (int64_t)gi0*D]);
        const int4* dO_g = reinterpret_cast<const int4*>(&dO[base + (int64_t)gi0*D]);
        int4* Q_s = reinterpret_cast<int4*>(sm.Q);
        int4* O_s = reinterpret_cast<int4*>(sm.O);
        int4* dO_s = reinterpret_cast<int4*>(sm.dO);
        int total_vec = BQ_a * D / 8;
        for (int idx = tid; idx < total_vec; idx += THREADS) {
            Q_s[idx] = Q_g[idx];
            O_s[idx] = O_g[idx];
            dO_s[idx] = dO_g[idx];
        }
    }
    
    // Init dQ
    for (int idx = tid; idx < BQ_a*D; idx += THREADS) sm.dQ[idx] = 0.f;
    
    // Load L
    for (int i = tid; i < BQ_a; i += THREADS) sm.L[i] = L[base_lse + gi0 + i];
    __syncthreads();

    // D = rowsum(dO * O)
    if (tid < BQ_a) {
        float s = 0.f;
        for (int d = 0; d < D; d += 2) {
            __nv_bfloat162 do2 = *reinterpret_cast<__nv_bfloat162*>(&sm.dO[tid*D+d]);
            __nv_bfloat162 o2 = *reinterpret_cast<__nv_bfloat162*>(&sm.O[tid*D+d]);
            float2 dof = __bfloat1622float2(do2);
            float2 of = __bfloat1622float2(o2);
            s += dof.x * of.x + dof.y * of.y;
        }
        sm.Dm[tid] = s;
    }
    __syncthreads();

    int max_jb = (gi0 + BQ_a - 1) / BKV;

    for (int jb = 0; jb <= max_jb; jb++) {
        int gj0 = jb * BKV;
        int BKV_a = (gj0 + BKV <= S) ? BKV : (S - gj0);

        // Load K, V
        {
            const int4* K_g = reinterpret_cast<const int4*>(&K[base + (int64_t)gj0*D]);
            const int4* V_g = reinterpret_cast<const int4*>(&V[base + (int64_t)gj0*D]);
            int4* K_s = reinterpret_cast<int4*>(sm.K);
            int4* V_s = reinterpret_cast<int4*>(sm.V);
            int total_vec = BKV_a * D / 8;
            for (int idx = tid; idx < total_vec; idx += THREADS) {
                K_s[idx] = K_g[idx];
                V_s[idx] = V_g[idx];
            }
        }
        __syncthreads();

        // S = Q @ K^T * scale -> P = exp(S - L) (compute directly into P)
        for (int idx = tid; idx < BQ_a*BKV_a; idx += THREADS) {
            int i = idx / BKV, k = idx % BKV;
            int gi = gi0 + i, gj = gj0 + k;
            float s = 0.f;
            for (int d = 0; d < D; d += 2) {
                __nv_bfloat162 q2 = *reinterpret_cast<__nv_bfloat162*>(&sm.Q[i*D+d]);
                __nv_bfloat162 k2 = *reinterpret_cast<__nv_bfloat162*>(&sm.K[k*D+d]);
                float2 qf = __bfloat1622float2(q2);
                float2 kf = __bfloat1622float2(k2);
                s += qf.x * kf.x + qf.y * kf.y;
            }
            s *= scale;
            sm.P[i*BKV+k] = (gj <= gi) ? __expf(s - sm.L[i]) : 0.f;
        }
        __syncthreads();

        // dP = dO @ V^T -> dS = P * (dP - D)
        for (int idx = tid; idx < BQ_a*BKV_a; idx += THREADS) {
            int i = idx / BKV, k = idx % BKV;
            int gi = gi0 + i, gj = gj0 + k;
            if (gj > gi) {
                sm.dS[i*BKV+k] = 0.f;
                continue;
            }
            float dp = 0.f;
            for (int d = 0; d < D; d += 2) {
                __nv_bfloat162 do2 = *reinterpret_cast<__nv_bfloat162*>(&sm.dO[i*D+d]);
                __nv_bfloat162 v2 = *reinterpret_cast<__nv_bfloat162*>(&sm.V[k*D+d]);
                float2 dof = __bfloat1622float2(do2);
                float2 vf = __bfloat1622float2(v2);
                dp += dof.x * vf.x + dof.y * vf.y;
            }
            sm.dS[i*BKV+k] = sm.P[i*BKV+k] * (dp - sm.Dm[i]);
        }
        __syncthreads();

        // dQ[i][d] += scale * sum_k dS[i][k] * K[k][d]
        for (int idx = tid; idx < BQ_a*D/2; idx += THREADS) {
            int i = idx / (D/2), dd = (idx % (D/2)) * 2;
            float v0 = 0.f, v1 = 0.f;
            for (int k = 0; k < BKV_a; k++) {
                float dS = sm.dS[i*BKV+k];
                __nv_bfloat162 k2 = *reinterpret_cast<__nv_bfloat162*>(&sm.K[k*D+dd]);
                float2 kf = __bfloat1622float2(k2);
                v0 += dS * kf.x;
                v1 += dS * kf.y;
            }
            sm.dQ[i*D+dd] += v0 * scale;
            sm.dQ[i*D+dd+1] += v1 * scale;
        }

        // dK[k][d] += scale * sum_i dS[i][k] * Q[i][d]
        // dV[k][d] += sum_i P[i][k] * dO[i][d]
        for (int idx = tid; idx < BKV_a*D/2; idx += THREADS) {
            int k = idx / (D/2), dd = (idx % (D/2)) * 2;
            int gj = gj0 + k;
            int min_i = max(0, gj - gi0);
            if (min_i >= BQ_a) continue;

            float vK0 = 0.f, vK1 = 0.f, vV0 = 0.f, vV1 = 0.f;
            for (int i = min_i; i < BQ_a; i++) {
                float dS = sm.dS[i*BKV+k];
                float P = sm.P[i*BKV+k];
                __nv_bfloat162 q2 = *reinterpret_cast<__nv_bfloat162*>(&sm.Q[i*D+dd]);
                __nv_bfloat162 do2 = *reinterpret_cast<__nv_bfloat162*>(&sm.dO[i*D+dd]);
                float2 qf = __bfloat1622float2(q2);
                float2 dof = __bfloat1622float2(do2);
                vK0 += dS * qf.x;
                vK1 += dS * qf.y;
                vV0 += P * dof.x;
                vV1 += P * dof.y;
            }
            int gk = gj0 + k;
            atomicAdd(&dK_fp[base + (int64_t)gk*D + dd], vK0 * scale);
            atomicAdd(&dK_fp[base + (int64_t)gk*D + dd + 1], vK1 * scale);
            atomicAdd(&dV_fp[base + (int64_t)gk*D + dd], vV0);
            atomicAdd(&dV_fp[base + (int64_t)gk*D + dd + 1], vV1);
        }
        __syncthreads();
    }

    // Write dQ to global bf16
    for (int idx = tid; idx < BQ_a*D; idx += THREADS) {
        dQ_out[base + (int64_t)gi0*D + idx] = __float2bfloat16(sm.dQ[idx]);
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

    int num_q_blocks = (S + BQ - 1) / BQ;
    int smem_size = sizeof(Smem);
    CUDA_CHECK(cudaFuncSetAttribute(attn_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid(num_q_blocks, B*H);
    dim3 block(THREADS);
    attn_bwd_kernel<<<grid, block, smem_size, stream>>>(Qp, Kp, Vp, Op, dOp, Lp, dQp, dK_fp, dV_fp, S, H, scale);
    CUDA_CHECK(cudaGetLastError());

    int cb = 256;
    int64_t cg = (total + cb - 1) / cb;
    convert_kernel<<<(int)cg, cb, 0, stream>>>(dK_fp, dKp, total);
    convert_kernel<<<(int)cg, cb, 0, stream>>>(dV_fp, dVp, total);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(dK_fp, stream));
    CUDA_CHECK(cudaFreeAsync(dV_fp, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_attn_bwd::run);

}  // namespace tvm_ffi_attn_bwd