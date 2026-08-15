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
    __nv_bfloat16 O[BQ*D];
    __nv_bfloat16 dO[BQ*D];
    __nv_bfloat16 K[BKV*D];
    __nv_bfloat16 V[BKV*D];
    float P[BQ*BKV];
    float dS[BQ*BKV];
    float dK[BKV*D];
    float dV[BKV*D];
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
    float* __restrict__ dQ_fp,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int S, int H, float scale)
{
    int jb = blockIdx.x;
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int64_t base = ((int64_t)(b*H + h)) * S * D;
    int64_t base_lse = ((int64_t)(b*H + h)) * S;

    int gj0 = jb * BKV;
    int BKV_a = (gj0 + BKV <= S) ? BKV : (S - gj0);
    if (BKV_a <= 0) return;

    extern __shared__ __align__(16) char smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
    int tid = threadIdx.x;

    // Load K, V with vectorized int4 (8 bf16) loads
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
    // Init dK, dV accumulators
    for (int idx = tid; idx < BKV*D; idx += THREADS) {
        sm.dK[idx] = 0.f;
        sm.dV[idx] = 0.f;
    }
    __syncthreads();

    int min_qb = gj0 / BQ;
    int max_qb = (S - 1) / BQ;

    for (int qi = min_qb; qi <= max_qb; qi++) {
        int gi0 = qi * BQ;
        int BQ_a = (gi0 + BQ <= S) ? BQ : (S - gi0);
        if (BQ_a <= 0) continue;

        // Load Q, O, dO with vectorized loads
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
            float p = (gj <= gi) ? __expf(s - sm.L[i]) : 0.f;
            sm.P[i*BKV+k] = p;
        }
        __syncthreads();

        // dS = P * (dO @ V^T - D)
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

        // dK[k][d] += scale * sum_i dS[i][k] * Q[i][d]
        for (int idx = tid; idx < BKV_a*D; idx += THREADS) {
            int k = idx / D, dd = idx % D;
            int gj = gj0 + k;
            int min_i = max(0, gj - gi0);
            if (min_i >= BQ_a) continue;
            float v = 0.f;
            for (int i = min_i; i < BQ_a; i++) {
                v += sm.dS[i*BKV+k] * __bfloat162float(sm.Q[i*D+dd]);
            }
            sm.dK[k*D+dd] += v * scale;
        }

        // dV[k][d] += sum_i P[i][k] * dO[i][d]
        for (int idx = tid; idx < BKV_a*D; idx += THREADS) {
            int k = idx / D, dd = idx % D;
            int gj = gj0 + k;
            int min_i = max(0, gj - gi0);
            if (min_i >= BQ_a) continue;
            float v = 0.f;
            for (int i = min_i; i < BQ_a; i++) {
                v += sm.P[i*BKV+k] * __bfloat162float(sm.dO[i*D+dd]);
            }
            sm.dV[k*D+dd] += v;
        }

        // dQ[i][d] += scale * sum_k dS[i][k] * K[k][d] (atomic to global fp32)
        for (int idx = tid; idx < BQ_a*D; idx += THREADS) {
            int i = idx / D, dd = idx % D;
            int gi = gi0 + i;
            int max_k = min(BKV_a, gi - gj0 + 1);
            if (max_k <= 0) continue;
            float v = 0.f;
            for (int k = 0; k < max_k; k++) {
                v += sm.dS[i*BKV+k] * __bfloat162float(sm.K[k*D+dd]);
            }
            atomicAdd(&dQ_fp[base + (int64_t)gi*D + dd], v * scale);
        }
        __syncthreads();
    }

    // Write dK, dV to global (fp32 -> bf16)
    for (int idx = tid; idx < BKV_a*D; idx += THREADS) {
        int k = idx / D, dd = idx % D;
        int gk = gj0 + k;
        dK_out[base + (int64_t)gk*D + dd] = __float2bfloat16(sm.dK[k*D+dd]);
        dV_out[base + (int64_t)gk*D + dd] = __float2bfloat16(sm.dV[k*D+dd]);
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

    float* dQ_fp = nullptr;
    CUDA_CHECK(cudaMallocAsync(&dQ_fp, sizeof(float)*total, stream));
    CUDA_CHECK(cudaMemsetAsync(dQ_fp, 0, sizeof(float)*total, stream));

    int num_kv_blocks = (S + BKV - 1) / BKV;
    int smem_size = sizeof(Smem);
    CUDA_CHECK(cudaFuncSetAttribute(attn_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid(num_kv_blocks, B*H);
    dim3 block(THREADS);
    attn_bwd_kernel<<<grid, block, smem_size, stream>>>(Qp, Kp, Vp, Op, dOp, Lp, dQ_fp, dKp, dVp, S, H, scale);
    CUDA_CHECK(cudaGetLastError());

    int cb = 256;
    int64_t cg = (total + cb - 1) / cb;
    convert_kernel<<<(int)cg, cb, 0, stream>>>(dQ_fp, dQp, total);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(dQ_fp, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_attn_bwd::run);

}  // namespace tvm_ffi_attn_bwd