#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
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

namespace mha_bwd {

constexpr int D = 128;
constexpr int B_R = 64;
constexpr int B_C = 32;
constexpr int THREADS = 256;
constexpr float SCALE = 0.07071067811865475f; // 1/sqrt(128)

// Kernel 1: Compute dV and Di
__global__ void dv_di_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dV,
    float* __restrict__ Di,
    int H, int S) {

    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;

    size_t offset = (size_t)(b * H + h) * S * D;
    const __nv_bfloat16* Q_bh = Q + offset;
    const __nv_bfloat16* K_bh = K + offset;
    const __nv_bfloat16* V_bh = V + offset;
    const __nv_bfloat16* dO_bh = dO + offset;
    const float* L_bh = L + (size_t)(b * H + h) * S;
    __nv_bfloat16* dV_bh = dV + offset;
    float* Di_bh = Di + (size_t)(b * H + h) * S;

    int tid = threadIdx.x;

    extern __shared__ char smem[];
    __nv_bfloat16* smem_Q = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* smem_K = smem_Q + B_R * D;
    __nv_bfloat16* smem_V = smem_K + B_C * D;
    __nv_bfloat16* smem_dO = smem_V + B_C * D;
    float* smem_S = reinterpret_cast<float*>(smem_dO + B_R * D);
    float* smem_P = smem_S + B_R * B_C;
    float* smem_dP = smem_P + B_R * B_C;
    float* smem_dV = smem_dP + B_R * B_C;

    for (int kv_start = 0; kv_start < S; kv_start += B_C) {
        // Load K, V tiles
        for (int i = tid; i < B_C * D; i += blockDim.x) {
            int row = i / D, col = i % D;
            smem_K[i] = (kv_start + row < S) ? K_bh[(size_t)(kv_start + row) * D + col] : __float2bfloat16(0.f);
            smem_V[i] = (kv_start + row < S) ? V_bh[(size_t)(kv_start + row) * D + col] : __float2bfloat16(0.f);
        }
        for (int i = tid; i < B_C * D; i += blockDim.x) smem_dV[i] = 0.0f;
        __syncthreads();

        for (int q_start = 0; q_start < S; q_start += B_R) {
            // Load Q, dO tiles
            for (int i = tid; i < B_R * D; i += blockDim.x) {
                int row = i / D, col = i % D;
                smem_Q[i] = (q_start + row < S) ? Q_bh[(size_t)(q_start + row) * D + col] : __float2bfloat16(0.f);
                smem_dO[i] = (q_start + row < S) ? dO_bh[(size_t)(q_start + row) * D + col] : __float2bfloat16(0.f);
            }
            __syncthreads();

            // S = scale * Q @ K^T
            for (int idx = tid; idx < B_R * B_C; idx += blockDim.x) {
                int i = idx / B_C, j = idx % B_C;
                float sum = 0.f;
                const __nv_bfloat162* q2 = reinterpret_cast<const __nv_bfloat162*>(&smem_Q[i * D]);
                const __nv_bfloat162* k2 = reinterpret_cast<const __nv_bfloat162*>(&smem_K[j * D]);
                #pragma unroll
                for (int k = 0; k < D / 2; k++) {
                    float2 qf = __bfloat1622float2(q2[k]);
                    float2 kf = __bfloat1622float2(k2[k]);
                    sum += qf.x * kf.x + qf.y * kf.y;
                }
                smem_S[idx] = sum * SCALE;
            }
            __syncthreads();

            // P = exp(S - L), dP = dO @ V^T
            for (int idx = tid; idx < B_R * B_C; idx += blockDim.x) {
                int i = idx / B_C, j = idx % B_C;
                float p_val = 0.f;
                if (q_start + i < S && kv_start + j < S) {
                    p_val = __expf(smem_S[idx] - L_bh[q_start + i]);
                }
                smem_P[idx] = p_val;

                float dp = 0.f;
                const __nv_bfloat162* do2 = reinterpret_cast<const __nv_bfloat162*>(&smem_dO[i * D]);
                const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&smem_V[j * D]);
                #pragma unroll
                for (int k = 0; k < D / 2; k++) {
                    float2 dof = __bfloat1622float2(do2[k]);
                    float2 vf = __bfloat1622float2(v2[k]);
                    dp += dof.x * vf.x + dof.y * vf.y;
                }
                smem_dP[idx] = dp;
            }
            __syncthreads();

            // Di += rowsum(P * dP)
            for (int i = tid; i < B_R; i += blockDim.x) {
                if (q_start + i < S) {
                    float di = 0.f;
                    for (int j = 0; j < B_C; j++) di += smem_P[i * B_C + j] * smem_dP[i * B_C + j];
                    atomicAdd(&Di_bh[q_start + i], di);
                }
            }

            // dV += P^T @ dO
            for (int idx = tid; idx < B_C * D; idx += blockDim.x) {
                int j = idx / D, k = idx % D;
                float sum = 0.f;
                for (int i = 0; i < B_R; i++) sum += smem_P[i * B_C + j] * __bfloat162float(smem_dO[i * D + k]);
                smem_dV[idx] += sum;
            }
            __syncthreads();
        }

        // Write dV
        for (int i = tid; i < B_C * D; i += blockDim.x) {
            int row = i / D, col = i % D;
            if (kv_start + row < S) dV_bh[(size_t)(kv_start + row) * D + col] = __float2bfloat16(smem_dV[i]);
        }
        __syncthreads();
    }
}

// Kernel 2: Compute dQ and dK
__global__ void dq_dk_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ Di,
    __nv_bfloat16* __restrict__ dQ,
    float* __restrict__ dK_ws,
    int H, int S) {

    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;

    size_t offset = (size_t)(b * H + h) * S * D;
    const __nv_bfloat16* Q_bh = Q + offset;
    const __nv_bfloat16* K_bh = K + offset;
    const __nv_bfloat16* V_bh = V + offset;
    const __nv_bfloat16* dO_bh = dO + offset;
    const float* L_bh = L + (size_t)(b * H + h) * S;
    const float* Di_bh = Di + (size_t)(b * H + h) * S;
    __nv_bfloat16* dQ_bh = dQ + offset;
    float* dK_bh = dK_ws + offset;

    int tid = threadIdx.x;

    extern __shared__ char smem[];
    __nv_bfloat16* smem_Q = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* smem_K = smem_Q + B_R * D;
    __nv_bfloat16* smem_V = smem_K + B_C * D;
    __nv_bfloat16* smem_dO = smem_V + B_C * D;
    float* smem_S = reinterpret_cast<float*>(smem_dO + B_R * D);
    float* smem_P = smem_S + B_R * B_C;
    float* smem_dP = smem_P + B_R * B_C;
    float* smem_dS = smem_dP + B_R * B_C;
    float* smem_dQ = smem_dS + B_R * B_C;
    float* smem_Di_loc = smem_dQ + B_R * D;

    for (int q_start = 0; q_start < S; q_start += B_R) {
        // Load Q, dO, Di
        for (int i = tid; i < B_R * D; i += blockDim.x) {
            int row = i / D, col = i % D;
            smem_Q[i] = (q_start + row < S) ? Q_bh[(size_t)(q_start + row) * D + col] : __float2bfloat16(0.f);
            smem_dO[i] = (q_start + row < S) ? dO_bh[(size_t)(q_start + row) * D + col] : __float2bfloat16(0.f);
        }
        for (int i = tid; i < B_R; i += blockDim.x) {
            smem_Di_loc[i] = (q_start + i < S) ? Di_bh[q_start + i] : 0.0f;
        }
        for (int i = tid; i < B_R * D; i += blockDim.x) smem_dQ[i] = 0.0f;
        __syncthreads();

        for (int kv_start = 0; kv_start < S; kv_start += B_C) {
            // Load K, V
            for (int i = tid; i < B_C * D; i += blockDim.x) {
                int row = i / D, col = i % D;
                smem_K[i] = (kv_start + row < S) ? K_bh[(size_t)(kv_start + row) * D + col] : __float2bfloat16(0.f);
                smem_V[i] = (kv_start + row < S) ? V_bh[(size_t)(kv_start + row) * D + col] : __float2bfloat16(0.f);
            }
            __syncthreads();

            // S = scale * Q @ K^T
            for (int idx = tid; idx < B_R * B_C; idx += blockDim.x) {
                int i = idx / B_C, j = idx % B_C;
                float sum = 0.f;
                const __nv_bfloat162* q2 = reinterpret_cast<const __nv_bfloat162*>(&smem_Q[i * D]);
                const __nv_bfloat162* k2 = reinterpret_cast<const __nv_bfloat162*>(&smem_K[j * D]);
                #pragma unroll
                for (int k = 0; k < D / 2; k++) {
                    float2 qf = __bfloat1622float2(q2[k]);
                    float2 kf = __bfloat1622float2(k2[k]);
                    sum += qf.x * kf.x + qf.y * kf.y;
                }
                smem_S[idx] = sum * SCALE;
            }
            __syncthreads();

            // P, dP, dS
            for (int idx = tid; idx < B_R * B_C; idx += blockDim.x) {
                int i = idx / B_C, j = idx % B_C;
                float p_val = 0.f;
                if (q_start + i < S && kv_start + j < S) p_val = __expf(smem_S[idx] - L_bh[q_start + i]);
                smem_P[idx] = p_val;

                float dp = 0.f;
                const __nv_bfloat162* do2 = reinterpret_cast<const __nv_bfloat162*>(&smem_dO[i * D]);
                const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&smem_V[j * D]);
                #pragma unroll
                for (int k = 0; k < D / 2; k++) {
                    float2 dof = __bfloat1622float2(do2[k]);
                    float2 vf = __bfloat1622float2(v2[k]);
                    dp += dof.x * vf.x + dof.y * vf.y;
                }
                smem_dP[idx] = dp;
                smem_dS[idx] = p_val * (dp - smem_Di_loc[i]);
            }
            __syncthreads();

            // dQ += scale * dS @ K
            for (int idx = tid; idx < B_R * D; idx += blockDim.x) {
                int i = idx / D, k = idx % D;
                float sum = 0.f;
                for (int j = 0; j < B_C; j++) sum += smem_dS[i * B_C + j] * __bfloat162float(smem_K[j * D + k]);
                smem_dQ[idx] += sum * SCALE;
            }

            // dK += scale * dS^T @ Q (atomicAdd)
            for (int idx = tid; idx < B_C * D; idx += blockDim.x) {
                int j = idx / D, k = idx % D;
                float sum = 0.f;
                for (int i = 0; i < B_R; i++) sum += smem_dS[i * B_C + j] * __bfloat162float(smem_Q[i * D + k]);
                if (kv_start + j < S) atomicAdd(&dK_bh[(size_t)(kv_start + j) * D + k], sum * SCALE);
            }
            __syncthreads();
        }

        // Write dQ
        for (int i = tid; i < B_R * D; i += blockDim.x) {
            int row = i / D, col = i % D;
            if (q_start + row < S) dQ_bh[(size_t)(q_start + row) * D + col] = __float2bfloat16(smem_dQ[i]);
        }
        __syncthreads();
    }
}

__global__ void convert_f32_bf16(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(src[idx]);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = 4;
    int H = 48;
    int S = (int)Q.size(2);

    int total = B * H;
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* Di = nullptr;
    float* dK_ws = nullptr;
    CUDA_CHECK(cudaMalloc(&Di, (size_t)total * S * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK_ws, (size_t)total * S * D * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(Di, 0, (size_t)total * S * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dK_ws, 0, (size_t)total * S * D * sizeof(float), stream));

    size_t smem1 = (size_t)(B_R * D * 2 * 2 + B_C * D * 2 * 2 + B_R * B_C * 4 * 3 + B_C * D * 4);
    size_t smem2 = (size_t)(B_R * D * 2 * 2 + B_C * D * 2 * 2 + B_R * B_C * 4 * 4 + B_R * D * 4 + B_R * 4);

    CUDA_CHECK(cudaFuncSetAttribute(dv_di_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem1));
    CUDA_CHECK(cudaFuncSetAttribute(dq_dk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem2));

    dv_di_kernel<<<total, THREADS, smem1, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        Di, H, S);
    CUDA_CHECK(cudaGetLastError());

    dq_dk_kernel<<<total, THREADS, smem2, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        Di,
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        dK_ws, H, S);
    CUDA_CHECK(cudaGetLastError());

    int n_elems = total * S * D;
    convert_f32_bf16<<<(n_elems + 255) / 256, 256, 0, stream>>>(
        dK_ws, static_cast<__nv_bfloat16*>(dK.data_ptr()), n_elems);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(Di));
    CUDA_CHECK(cudaFree(dK_ws));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd