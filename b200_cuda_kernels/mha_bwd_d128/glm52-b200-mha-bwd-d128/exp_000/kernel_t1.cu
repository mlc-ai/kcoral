#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
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

namespace mha_bwd_d128 {

constexpr int D = 128;
constexpr int BQ = 64;
constexpr int BK = 32;
constexpr int THREADS = 64;
constexpr float SCALE = 0.0883883476f; // 1.0f / sqrtf(128.0f)

constexpr int SMEM_SIZE =
    BQ * D * 2 + BQ * D * 2 + BK * D * 2 + BK * D * 2 +
    BQ * D * 4 + BK * D * 4 + BK * D * 4 +
    BQ * 4 + BQ * 4; // ~100000 bytes

__global__ void __launch_bounds__(THREADS) mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ_out,
    float* __restrict__ dK_fp32,
    float* __restrict__ dV_fp32,
    int H, int S)
{
    int bh = blockIdx.x;
    int qb = blockIdx.y;
    int tid = threadIdx.x;
    int q_start = qb * BQ;

    int64_t base = (int64_t)bh * S * D;
    const __nv_bfloat16* Q_bh = Q + base;
    const __nv_bfloat16* K_bh = K + base;
    const __nv_bfloat16* V_bh = V + base;
    const __nv_bfloat16* O_bh = O + base;
    const __nv_bfloat16* dO_bh = dO + base;
    const float* L_bh = L + (int64_t)bh * S;
    float* dK_bh = dK_fp32 + base;
    float* dV_bh = dV_fp32 + base;

    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem  = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* dO_smem = Q_smem + BQ * D;
    __nv_bfloat16* K_smem  = dO_smem + BQ * D;
    __nv_bfloat16* V_smem  = K_smem + BK * D;
    float* dQ_smem  = reinterpret_cast<float*>(V_smem + BK * D);
    float* dK_smem  = dQ_smem + BQ * D;
    float* dV_smem  = dK_smem + BK * D;
    float* L_smem   = dV_smem + BK * D;
    float* D_smem   = L_smem + BQ;

    // Load Q, dO into smem; compute D_i = dO_i . O_i
    for (int i = tid; i < BQ; i += THREADS) {
        int qi = q_start + i;
        if (qi < S) {
            #pragma unroll
            for (int k = 0; k < D; k += 8) {
                *reinterpret_cast<int4*>(&Q_smem[i * D + k]) =
                    *reinterpret_cast<const int4*>(&Q_bh[qi * D + k]);
                *reinterpret_cast<int4*>(&dO_smem[i * D + k]) =
                    *reinterpret_cast<const int4*>(&dO_bh[qi * D + k]);
            }
            L_smem[i] = L_bh[qi];

            float dot_val = 0.0f;
            #pragma unroll
            for (int k = 0; k < D; k += 2) {
                __nv_bfloat162 o2 = *reinterpret_cast<const __nv_bfloat162*>(&O_bh[qi * D + k]);
                __nv_bfloat162 do2 = *reinterpret_cast<const __nv_bfloat162*>(&dO_smem[i * D + k]);
                float2 of = __bfloat1622float2(o2);
                float2 dof = __bfloat1622float2(do2);
                dot_val += of.x * dof.x + of.y * dof.y;
            }
            D_smem[i] = dot_val;
        } else {
            L_smem[i] = -1e30f;
            D_smem[i] = 0.0f;
            #pragma unroll
            for (int k = 0; k < D; k += 8) {
                *reinterpret_cast<int4*>(&Q_smem[i * D + k]) = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4*>(&dO_smem[i * D + k]) = make_int4(0, 0, 0, 0);
            }
        }
        #pragma unroll
        for (int k = 0; k < D; k += 4) {
            dQ_smem[i * D + k]     = 0.0f;
            dQ_smem[i * D + k + 1] = 0.0f;
            dQ_smem[i * D + k + 2] = 0.0f;
            dQ_smem[i * D + k + 3] = 0.0f;
        }
    }
    __syncthreads();

    for (int kv_start = 0; kv_start < S; kv_start += BK) {
        for (int j = tid; j < BK; j += THREADS) {
            int kj = kv_start + j;
            if (kj < S) {
                #pragma unroll
                for (int k = 0; k < D; k += 8) {
                    *reinterpret_cast<int4*>(&K_smem[j * D + k]) =
                        *reinterpret_cast<const int4*>(&K_bh[kj * D + k]);
                    *reinterpret_cast<int4*>(&V_smem[j * D + k]) =
                        *reinterpret_cast<const int4*>(&V_bh[kj * D + k]);
                }
            } else {
                #pragma unroll
                for (int k = 0; k < D; k += 8) {
                    *reinterpret_cast<int4*>(&K_smem[j * D + k]) = make_int4(0, 0, 0, 0);
                    *reinterpret_cast<int4*>(&V_smem[j * D + k]) = make_int4(0, 0, 0, 0);
                }
            }
        }
        __syncthreads();

        for (int idx = tid; idx < BK * D; idx += THREADS) {
            dK_smem[idx] = 0.0f;
            dV_smem[idx] = 0.0f;
        }
        __syncthreads();

        if (tid < BQ && q_start + tid < S) {
            float l_val = L_smem[tid];
            float d_val = D_smem[tid];
            float p_vals[BK];
            float ds_vals[BK];

            #pragma unroll
            for (int j = 0; j < BK; j++) {
                if (kv_start + j < S) {
                    float s = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < D; k += 2) {
                        __nv_bfloat162 q2 = *reinterpret_cast<__nv_bfloat162*>(&Q_smem[tid * D + k]);
                        __nv_bfloat162 k2 = *reinterpret_cast<__nv_bfloat162*>(&K_smem[j * D + k]);
                        float2 qf = __bfloat1622float2(q2);
                        float2 kf = __bfloat1622float2(k2);
                        s += qf.x * kf.x + qf.y * kf.y;
                    }
                    s *= SCALE;
                    p_vals[j] = __expf(s - l_val);

                    float dp = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < D; k += 2) {
                        __nv_bfloat162 do2 = *reinterpret_cast<__nv_bfloat162*>(&dO_smem[tid * D + k]);
                        __nv_bfloat162 v2 = *reinterpret_cast<__nv_bfloat162*>(&V_smem[j * D + k]);
                        float2 dof = __bfloat1622float2(do2);
                        float2 vf = __bfloat1622float2(v2);
                        dp += dof.x * vf.x + dof.y * vf.y;
                    }
                    ds_vals[j] = p_vals[j] * (dp - d_val);
                } else {
                    p_vals[j] = 0.0f;
                    ds_vals[j] = 0.0f;
                }
            }

            #pragma unroll
            for (int j = 0; j < BK; j++) {
                float ds_scaled = ds_vals[j] * SCALE;
                #pragma unroll
                for (int k = 0; k < D; k += 2) {
                    __nv_bfloat162 k2 = *reinterpret_cast<__nv_bfloat162*>(&K_smem[j * D + k]);
                    float2 kf = __bfloat1622float2(k2);
                    dQ_smem[tid * D + k]     += ds_scaled * kf.x;
                    dQ_smem[tid * D + k + 1] += ds_scaled * kf.y;
                }
            }

            #pragma unroll
            for (int j = 0; j < BK; j++) {
                if (kv_start + j < S && ds_vals[j] != 0.0f) {
                    float ds_scaled = ds_vals[j] * SCALE;
                    #pragma unroll
                    for (int k = 0; k < D; k += 2) {
                        __nv_bfloat162 q2 = *reinterpret_cast<__nv_bfloat162*>(&Q_smem[tid * D + k]);
                        float2 qf = __bfloat1622float2(q2);
                        atomicAdd(&dK_smem[j * D + k],     ds_scaled * qf.x);
                        atomicAdd(&dK_smem[j * D + k + 1], ds_scaled * qf.y);
                    }
                }
            }

            #pragma unroll
            for (int j = 0; j < BK; j++) {
                if (kv_start + j < S && p_vals[j] != 0.0f) {
                    float p = p_vals[j];
                    #pragma unroll
                    for (int k = 0; k < D; k += 2) {
                        __nv_bfloat162 do2 = *reinterpret_cast<__nv_bfloat162*>(&dO_smem[tid * D + k]);
                        float2 dof = __bfloat1622float2(do2);
                        atomicAdd(&dV_smem[j * D + k],     p * dof.x);
                        atomicAdd(&dV_smem[j * D + k + 1], p * dof.y);
                    }
                }
            }
        }
        __syncthreads();

        for (int idx = tid; idx < BK * D; idx += THREADS) {
            int j = idx / D;
            int k = idx % D;
            int kj = kv_start + j;
            if (kj < S) {
                atomicAdd(&dK_bh[kj * D + k], dK_smem[idx]);
                atomicAdd(&dV_bh[kj * D + k], dV_smem[idx]);
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < BQ; i += THREADS) {
        int qi = q_start + i;
        if (qi < S) {
            #pragma unroll
            for (int k = 0; k < D; k += 2) {
                __nv_bfloat162 res;
                res.x = __float2bfloat16(dQ_smem[i * D + k]);
                res.y = __float2bfloat16(dQ_smem[i * D + k + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&dQ_out[base + qi * D + k]) = res;
            }
        }
    }
}

__global__ void convert_fp32_to_bf16_kernel(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = static_cast<int>(Q.size(0));
    int H = static_cast<int>(Q.size(1));
    int S = static_cast<int>(Q.size(2));

    int total_elements = B * H * S * D;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* dK_fp32 = nullptr;
    float* dV_fp32 = nullptr;
    CUDA_CHECK(cudaMalloc(&dK_fp32, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV_fp32, total_elements * sizeof(float)));

    CUDA_CHECK(cudaMemsetAsync(dK_fp32, 0, total_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_fp32, 0, total_elements * sizeof(float), stream));

    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    dim3 grid(B * H, (S + BQ - 1) / BQ);
    dim3 block(THREADS);

    mha_bwd_kernel<<<grid, block, SMEM_SIZE, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        dK_fp32,
        dV_fp32,
        H, S);

    CUDA_CHECK(cudaGetLastError());

    int convert_threads = 256;
    int convert_blocks = (total_elements + convert_threads - 1) / convert_threads;

    convert_fp32_to_bf16_kernel<<<convert_blocks, convert_threads, 0, stream>>>(
        dK_fp32, static_cast<__nv_bfloat16*>(dK.data_ptr()), total_elements);
    convert_fp32_to_bf16_kernel<<<convert_blocks, convert_threads, 0, stream>>>(
        dV_fp32, static_cast<__nv_bfloat16*>(dV.data_ptr()), total_elements);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(dK_fp32));
    CUDA_CHECK(cudaFree(dV_fp32));

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128