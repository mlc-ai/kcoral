#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
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

namespace mha_bwd_d128 {

constexpr int B_CONST = 4;
constexpr int H_CONST = 48;
constexpr int D_CONST = 128;
constexpr int BR = 64;
constexpr int BC = 64;
constexpr int THREADS = 128;
constexpr int D8 = D_CONST / 8; // 16 int4 elements per row

__global__ void compute_D_kernel(
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    float* __restrict__ D,
    int S) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B_CONST * H_CONST * S;
    if (idx >= total) return;
    int i = idx % S;
    int bh = idx / S;
    const __nv_bfloat16* O_ptr = O + (size_t)bh * S * D_CONST + (size_t)i * D_CONST;
    const __nv_bfloat16* dO_ptr = dO + (size_t)bh * S * D_CONST + (size_t)i * D_CONST;
    float d_val = 0.0f;
    for (int k = 0; k < D_CONST; k += 2) {
        float o0 = __bfloat162float(O_ptr[k]);
        float o1 = __bfloat162float(O_ptr[k+1]);
        float g0 = __bfloat162float(dO_ptr[k]);
        float g1 = __bfloat162float(dO_ptr[k+1]);
        d_val = fmaf(o0, g0, d_val);
        d_val = fmaf(o1, g1, d_val);
    }
    D[idx] = d_val;
}

__global__ void init_zero_kernel(float* __restrict__ ptr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = 0.0f;
}

__global__ void attention_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D,
    __nv_bfloat16* __restrict__ dQ,
    float* __restrict__ dK_fp32,
    float* __restrict__ dV_fp32,
    int S,
    float scale) {

    int bh = blockIdx.x;
    int tid = threadIdx.x;

    const __nv_bfloat16* Q_bh = Q + (size_t)bh * S * D_CONST;
    const __nv_bfloat16* K_bh = K + (size_t)bh * S * D_CONST;
    const __nv_bfloat16* V_bh = V + (size_t)bh * S * D_CONST;
    const __nv_bfloat16* dO_bh = dO + (size_t)bh * S * D_CONST;
    const float* L_bh = L + (size_t)bh * S;
    const float* D_bh = D + (size_t)bh * S;
    __nv_bfloat16* dQ_bh = dQ + (size_t)bh * S * D_CONST;
    float* dK_bh = dK_fp32 + (size_t)bh * S * D_CONST;
    float* dV_bh = dV_fp32 + (size_t)bh * S * D_CONST;

    extern __shared__ char smem[];
    __nv_bfloat16* sQ = (__nv_bfloat16*)smem;              // BR * D
    __nv_bfloat16* sK = sQ + BR * D_CONST;                  // BC * D
    __nv_bfloat16* sV = sK + BC * D_CONST;                  // BC * D
    __nv_bfloat16* sdO = sV + BC * D_CONST;                 // BR * D
    float* sP = (float*)(sdO + BR * D_CONST);               // BR * BC (reused for dS)
    float* sdQ = sP + BR * BC;                               // BR * D
    float* sL = sdQ + BR * D_CONST;                          // BR
    float* sDrow = sL + BR;                                  // BR

    for (int qi = 0; qi < S; qi += BR) {
        int br = min(BR, S - qi);

        // Vectorized load Q, dO (int4 = 8 BF16)
        for (int idx = tid; idx < br * D8; idx += THREADS) {
            int i = idx / D8;
            int k8 = idx % D8;
            *(int4*)&sQ[i * D_CONST + k8 * 8] =
                *(const int4*)&Q_bh[(qi + i) * D_CONST + k8 * 8];
            *(int4*)&sdO[i * D_CONST + k8 * 8] =
                *(const int4*)&dO_bh[(qi + i) * D_CONST + k8 * 8];
        }
        // Load L, D
        for (int i = tid; i < br; i += THREADS) {
            sL[i] = L_bh[qi + i];
            sDrow[i] = D_bh[qi + i];
        }
        // Init dQ
        for (int idx = tid; idx < br * D_CONST; idx += THREADS) {
            sdQ[idx] = 0.0f;
        }
        __syncthreads();

        for (int kj = 0; kj < S; kj += BC) {
            int bc = min(BC, S - kj);

            // Vectorized load K, V
            for (int idx = tid; idx < bc * D8; idx += THREADS) {
                int j = idx / D8;
                int k8 = idx % D8;
                *(int4*)&sK[j * D_CONST + k8 * 8] =
                    *(const int4*)&K_bh[(kj + j) * D_CONST + k8 * 8];
                *(int4*)&sV[j * D_CONST + k8 * 8] =
                    *(const int4*)&V_bh[(kj + j) * D_CONST + k8 * 8];
            }
            __syncthreads();

            // S = Q @ K^T * scale, P = exp(S - L)
            for (int idx = tid; idx < br * bc; idx += THREADS) {
                int i = idx / bc;
                int j = idx % bc;
                float s = 0.0f;
                const __nv_bfloat16* qrow = &sQ[i * D_CONST];
                const __nv_bfloat16* krow = &sK[j * D_CONST];
                for (int k = 0; k < D_CONST; k += 2) {
                    float q0 = __bfloat162float(qrow[k]);
                    float q1 = __bfloat162float(qrow[k+1]);
                    float k0 = __bfloat162float(krow[k]);
                    float k1 = __bfloat162float(krow[k+1]);
                    s = fmaf(q0, k0, s);
                    s = fmaf(q1, k1, s);
                }
                s *= scale;
                sP[i * bc + j] = __expf(s - sL[i]);
            }
            __syncthreads();

            // dV += P^T @ dO  [bc, D]
            for (int idx = tid; idx < bc * D_CONST; idx += THREADS) {
                int j = idx / D_CONST;
                int k = idx % D_CONST;
                float dv = 0.0f;
                for (int i = 0; i < br; i++) {
                    dv = fmaf(sP[i * bc + j],
                              __bfloat162float(sdO[i * D_CONST + k]), dv);
                }
                atomicAdd(&dV_bh[(kj + j) * D_CONST + k], dv);
            }
            __syncthreads();

            // dP = dO @ V^T, dS = P * (dP - D), overwrite sP with dS
            for (int idx = tid; idx < br * bc; idx += THREADS) {
                int i = idx / bc;
                int j = idx % bc;
                float dp = 0.0f;
                const __nv_bfloat16* dorow = &sdO[i * D_CONST];
                const __nv_bfloat16* vrow = &sV[j * D_CONST];
                for (int k = 0; k < D_CONST; k += 2) {
                    float d0 = __bfloat162float(dorow[k]);
                    float d1 = __bfloat162float(dorow[k+1]);
                    float v0 = __bfloat162float(vrow[k]);
                    float v1 = __bfloat162float(vrow[k+1]);
                    dp = fmaf(d0, v0, dp);
                    dp = fmaf(d1, v1, dp);
                }
                float p = sP[i * bc + j];
                sP[i * bc + j] = p * (dp - sDrow[i]);
            }
            __syncthreads();

            // dQ += dS @ K * scale  [br, D]
            for (int idx = tid; idx < br * D_CONST; idx += THREADS) {
                int i = idx / D_CONST;
                int k = idx % D_CONST;
                float dq = 0.0f;
                for (int j = 0; j < bc; j++) {
                    dq = fmaf(sP[i * bc + j],
                              __bfloat162float(sK[j * D_CONST + k]), dq);
                }
                sdQ[i * D_CONST + k] += dq * scale;
            }

            // dK += dS^T @ Q * scale  [bc, D]
            for (int idx = tid; idx < bc * D_CONST; idx += THREADS) {
                int j = idx / D_CONST;
                int k = idx % D_CONST;
                float dk = 0.0f;
                for (int i = 0; i < br; i++) {
                    dk = fmaf(sP[i * bc + j],
                              __bfloat162float(sQ[i * D_CONST + k]), dk);
                }
                atomicAdd(&dK_bh[(kj + j) * D_CONST + k], dk * scale);
            }
            __syncthreads();
        }

        // Store dQ
        for (int idx = tid; idx < br * D_CONST; idx += THREADS) {
            int i = idx / D_CONST;
            int k = idx % D_CONST;
            dQ_bh[(qi + i) * D_CONST + k] = __float2bfloat16(sdQ[i * D_CONST + k]);
        }
        __syncthreads();
    }
}

__global__ void convert_fp32_bf16_kernel(
    const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

void run(
    tvm::ffi::TensorView Q,
    tvm::ffi::TensorView K,
    tvm::ffi::TensorView V,
    tvm::ffi::TensorView O,
    tvm::ffi::TensorView dO,
    tvm::ffi::TensorView L,
    tvm::ffi::TensorView dQ,
    tvm::ffi::TensorView dK,
    tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int S = (int)Q.size(2);
    float scale = 1.0f / sqrtf((float)D_CONST);

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

    size_t D_size = (size_t)B_CONST * H_CONST * S * sizeof(float);
    size_t grad_size = (size_t)B_CONST * H_CONST * S * D_CONST * sizeof(float);

    float* D_buf = nullptr;
    float* dK_fp32 = nullptr;
    float* dV_fp32 = nullptr;

    CUDA_CHECK(cudaMalloc(&D_buf, D_size));
    CUDA_CHECK(cudaMalloc(&dK_fp32, grad_size));
    CUDA_CHECK(cudaMalloc(&dV_fp32, grad_size));

    // Compute D = rowsum(dO * O)
    {
        int total = B_CONST * H_CONST * S;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        compute_D_kernel<<<blocks, threads, 0, stream>>>(O_ptr, dO_ptr, D_buf, S);
    }

    // Init dK_fp32, dV_fp32 to zero
    {
        int total = B_CONST * H_CONST * S * D_CONST;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        init_zero_kernel<<<blocks, threads, 0, stream>>>(dK_fp32, total);
        init_zero_kernel<<<blocks, threads, 0, stream>>>(dV_fp32, total);
    }

    // Main backward kernel
    {
        int blocks = B_CONST * H_CONST;
        size_t smem_size =
            (size_t)BR * D_CONST * sizeof(__nv_bfloat16) +  // sQ
            (size_t)BC * D_CONST * sizeof(__nv_bfloat16) +  // sK
            (size_t)BC * D_CONST * sizeof(__nv_bfloat16) +  // sV
            (size_t)BR * D_CONST * sizeof(__nv_bfloat16) +  // sdO
            (size_t)BR * BC * sizeof(float) +                // sP/dS
            (size_t)BR * D_CONST * sizeof(float) +           // sdQ
            (size_t)BR * sizeof(float) +                     // sL
            (size_t)BR * sizeof(float);                      // sDrow

        CUDA_CHECK(cudaFuncSetAttribute(
            attention_bwd_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));

        attention_bwd_kernel<<<blocks, THREADS, smem_size, stream>>>(
            Q_ptr, K_ptr, V_ptr, dO_ptr,
            L_ptr, D_buf,
            dQ_ptr, dK_fp32, dV_fp32,
            S, scale);
    }

    // Convert dK, dV from FP32 to BF16
    {
        int total = B_CONST * H_CONST * S * D_CONST;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        convert_fp32_bf16_kernel<<<blocks, threads, 0, stream>>>(dK_fp32, dK_ptr, total);
        convert_fp32_bf16_kernel<<<blocks, threads, 0, stream>>>(dV_fp32, dV_ptr, total);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(D_buf));
    CUDA_CHECK(cudaFree(dK_fp32));
    CUDA_CHECK(cudaFree(dV_fp32));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128