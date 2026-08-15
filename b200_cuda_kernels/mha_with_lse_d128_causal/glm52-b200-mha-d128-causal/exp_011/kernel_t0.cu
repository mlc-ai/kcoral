#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace tvm_ffi_mha {

constexpr int BR = 128;
constexpr int BC = 32;
constexpr int D = 128;
constexpr int Q_STRIDE = 129;  // D + 1 padding for bank conflict avoidance
constexpr int SMEM_SIZE = D * Q_STRIDE * sizeof(__nv_bfloat16) + 2 * BC * D * sizeof(__nv_bfloat16);

__device__ __forceinline__ float fast_exp2f(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__global__ void __launch_bounds__(128, 2)
mha_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int H, int S) {

    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;

    int64_t base = (int64_t)(b * H + h) * S * D;
    const __nv_bfloat16* Q_bh = Q + base;
    const __nv_bfloat16* K_bh = K + base;
    const __nv_bfloat16* V_bh = V + base;
    __nv_bfloat16* O_bh = O + base;
    float* LSE_bh = LSE + (int64_t)(b * H + h) * S;

    extern __shared__ char smem[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);  // [D][Q_STRIDE] transposed
    __nv_bfloat16* sK = sQ + D * Q_STRIDE;                         // [BC][D]
    __nv_bfloat16* sV = sK + BC * D;                                // [BC][D]

    // 1/sqrt(128) * log2(e) for log2-scaled exp2f computation
    const float scale_log2e = 0.12754844765175826f;
    const float inv_log2e = 0.6931471805599453f;  // ln(2)

    for (int qi = 0; qi < S; qi += BR) {
        int q_rows = min(BR, S - qi);

        // Load Q tile: sQ[d][r] = Q_bh[(qi+r)*D + d], coalesced across threads
        for (int r = 0; r < q_rows; r++) {
            sQ[tid * Q_STRIDE + r] = Q_bh[(int64_t)(qi + r) * D + tid];
        }
        __syncthreads();

        float o_local[128];
        float m_local, l_local;

        if (tid < q_rows) {
            #pragma unroll 8
            for (int d = 0; d < 128; d++) o_local[d] = 0.0f;
            m_local = -INFINITY;
            l_local = 0.0f;
        }

        int max_k = qi + q_rows;  // causal: max key pos = max query pos

        for (int kj = 0; kj < max_k; kj += BC) {
            int k_cols = min(BC, S - kj);

            // Load K and V tiles together
            for (int j = 0; j < k_cols; j++) {
                sK[j * D + tid] = K_bh[(int64_t)(kj + j) * D + tid];
                sV[j * D + tid] = V_bh[(int64_t)(kj + j) * D + tid];
            }
            for (int j = k_cols; j < BC; j++) {
                sK[j * D + tid] = __float2bfloat16(0.0f);
                sV[j * D + tid] = __float2bfloat16(0.0f);
            }
            __syncthreads();

            float score[BC];

            if (tid < q_rows) {
                int q_pos = qi + tid;

                #pragma unroll
                for (int j = 0; j < BC; j++) score[j] = 0.0f;

                // Compute QK^T: score[j] = sum_d Q[tid][d] * K[j][d]
                for (int d = 0; d < 128; d += 2) {
                    float qf0 = __bfloat162float(sQ[d * Q_STRIDE + tid]);
                    float qf1 = __bfloat162float(sQ[(d + 1) * Q_STRIDE + tid]);

                    #pragma unroll
                    for (int j = 0; j < BC; j++) {
                        __nv_bfloat162 k2 = *reinterpret_cast<__nv_bfloat162*>(&sK[j * D + d]);
                        float2 kf2 = __bfloat1622float2(k2);
                        score[j] = __fmaf_rn(qf0, kf2.x, __fmaf_rn(qf1, kf2.y, score[j]));
                    }
                }

                // Scale by 1/sqrt(D) * log2(e) and apply causal mask
                #pragma unroll
                for (int j = 0; j < BC; j++) {
                    score[j] *= scale_log2e;
                    int k_pos = kj + j;
                    if (k_pos > q_pos || k_pos >= S) {
                        score[j] = -INFINITY;
                    }
                }

                // Online softmax update
                float m_new = m_local;
                #pragma unroll
                for (int j = 0; j < BC; j++) {
                    m_new = fmaxf(m_new, score[j]);
                }

                float alpha = (m_local == -INFINITY) ? 0.0f : fast_exp2f(m_local - m_new);

                float l_new = 0.0f;
                #pragma unroll
                for (int j = 0; j < BC; j++) {
                    float p = fast_exp2f(score[j] - m_new);
                    score[j] = p;
                    l_new += p;
                }

                l_local = l_local * alpha + l_new;
                m_local = m_new;

                #pragma unroll 8
                for (int d = 0; d < 128; d++) o_local[d] *= alpha;

                // Accumulate O += P @ V
                for (int d = 0; d < 128; d += 2) {
                    float acc0 = o_local[d];
                    float acc1 = o_local[d + 1];
                    #pragma unroll
                    for (int j = 0; j < BC; j++) {
                        float p = score[j];
                        __nv_bfloat162 v2 = *reinterpret_cast<__nv_bfloat162*>(&sV[j * D + d]);
                        float2 vf2 = __bfloat1622float2(v2);
                        acc0 = __fmaf_rn(p, vf2.x, acc0);
                        acc1 = __fmaf_rn(p, vf2.y, acc1);
                    }
                    o_local[d] = acc0;
                    o_local[d + 1] = acc1;
                }
            }

            __syncthreads();
        }

        // Write output: O and LSE
        if (tid < q_rows) {
            float inv_l = 1.0f / l_local;
            for (int d = 0; d < 128; d += 2) {
                __nv_bfloat162 v = __float22bfloat162_rn(
                    make_float2(o_local[d] * inv_l, o_local[d + 1] * inv_l));
                *reinterpret_cast<__nv_bfloat162*>(&O_bh[(int64_t)(qi + tid) * D + d]) = v;
            }
            // LSE = m_nat + log(l) = (m_log2 + log2(l)) * ln(2)
            LSE_bh[qi + tid] = (m_local + log2f(l_local)) * inv_log2e;
        }

        __syncthreads();
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    if (S == 0) return;

    int grid = B * H;
    int block = 128;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(mha_causal_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    mha_causal_kernel<<<grid, block, SMEM_SIZE, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

}  // namespace tvm_ffi_mha