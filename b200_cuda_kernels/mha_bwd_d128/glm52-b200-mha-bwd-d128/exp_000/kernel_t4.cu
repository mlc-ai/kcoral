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
constexpr int THREADS = 128;
constexpr float SCALE = 0.0883883476f; // 1/sqrt(128)

// Shared memory layout:
// Q: BQ*D*2=16384, dO: BQ*D*2=16384, K: BK*D*2=8192, V: BK*D*2=8192
// dQ: BQ*D*4=32768, P: BQ*BK*4=8192, dS: BQ*BK*4=8192, L: BQ*4=256, Dm: BQ*4=256
constexpr int SMEM_SIZE = 16384*2 + 8192*2 + 32768 + 8192*2 + 256*2;

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
    int S)
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
    float* P_smem   = dQ_smem + BQ * D;
    float* dS_smem  = P_smem + BQ * BK;
    float* L_smem   = dS_smem + BQ * BK;
    float* Dm_smem  = L_smem + BQ;

    // Load Q, dO (vectorized 16B); init dQ; compute L and D_i = dO_i . O_i
    for (int idx = tid; idx < BQ * D / 8; idx += THREADS) {
        int i = idx / (D / 8);
        int k = (idx % (D / 8)) * 8;
        int qi = q_start + i;
        if (qi < S) {
            *reinterpret_cast<int4*>(&Q_smem[i * D + k]) =
                *reinterpret_cast<const int4*>(&Q_bh[qi * D + k]);
            *reinterpret_cast<int4*>(&dO_smem[i * D + k]) =
                *reinterpret_cast<const int4*>(&dO_bh[qi * D + k]);
        } else {
            *reinterpret_cast<int4*>(&Q_smem[i * D + k]) = make_int4(0, 0, 0, 0);
            *reinterpret_cast<int4*>(&dO_smem[i * D + k]) = make_int4(0, 0, 0, 0);
        }
    }

    for (int idx = tid; idx < BQ * D / 4; idx += THREADS) {
        int i = idx / (D / 4);
        int k = (idx % (D / 4)) * 4;
        *reinterpret_cast<float4*>(&dQ_smem[i * D + k]) =
            make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    for (int i = tid; i < BQ; i += THREADS) {
        int qi = q_start + i;
        if (qi < S) {
            L_smem[i] = L_bh[qi];
            float dot_val = 0.0f;
            #pragma unroll 8
            for (int k = 0; k < D; k += 2) {
                __nv_bfloat162 o2 = *reinterpret_cast<const __nv_bfloat162*>(&O_bh[qi * D + k]);
                __nv_bfloat162 do2 = *reinterpret_cast<__nv_bfloat162*>(&dO_smem[i * D + k]);
                float2 of = __bfloat1622float2(o2);
                float2 dof = __bfloat1622float2(do2);
                dot_val += of.x * dof.x + of.y * dof.y;
            }
            Dm_smem[i] = dot_val;
        } else {
            L_smem[i] = -1e30f;
            Dm_smem[i] = 0.0f;
        }
    }
    __syncthreads();

    // Main loop over KV blocks
    for (int kv_start = 0; kv_start < S; kv_start += BK) {
        // Load K, V (vectorized 16B)
        for (int idx = tid; idx < BK * D / 8; idx += THREADS) {
            int j = idx / (D / 8);
            int k = (idx % (D / 8)) * 8;
            int kj = kv_start + j;
            if (kj < S) {
                *reinterpret_cast<int4*>(&K_smem[j * D + k]) =
                    *reinterpret_cast<const int4*>(&K_bh[kj * D + k]);
                *reinterpret_cast<int4*>(&V_smem[j * D + k]) =
                    *reinterpret_cast<const int4*>(&V_bh[kj * D + k]);
            } else {
                *reinterpret_cast<int4*>(&K_smem[j * D + k]) = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4*>(&V_smem[j * D + k]) = make_int4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // Phase 1: Compute P and dS (64 threads, one per q row)
        if (tid < BQ) {
            int qi = q_start + tid;
            float l_val = L_smem[tid];
            float d_val = Dm_smem[tid];
            if (qi < S) {
                #pragma unroll 1
                for (int j = 0; j < BK; j++) {
                    if (kv_start + j < S) {
                        float s = 0.0f;
                        #pragma unroll 8
                        for (int k = 0; k < D; k += 2) {
                            __nv_bfloat162 q2 = *reinterpret_cast<__nv_bfloat162*>(&Q_smem[tid * D + k]);
                            __nv_bfloat162 k2 = *reinterpret_cast<__nv_bfloat162*>(&K_smem[j * D + k]);
                            float2 qf = __bfloat1622float2(q2);
                            float2 kf = __bfloat1622float2(k2);
                            s += qf.x * kf.x + qf.y * kf.y;
                        }
                        s *= SCALE;
                        float p = expf(s - l_val);
                        P_smem[tid * BK + j] = p;

                        float dp = 0.0f;
                        #pragma unroll 8
                        for (int k = 0; k < D; k += 2) {
                            __nv_bfloat162 do2 = *reinterpret_cast<__nv_bfloat162*>(&dO_smem[tid * D + k]);
                            __nv_bfloat162 v2 = *reinterpret_cast<__nv_bfloat162*>(&V_smem[j * D + k]);
                            float2 dof = __bfloat1622float2(do2);
                            float2 vf = __bfloat1622float2(v2);
                            dp += dof.x * vf.x + dof.y * vf.y;
                        }
                        // Pre-multiply by SCALE for dQ and dK
                        dS_smem[tid * BK + j] = p * (dp - d_val) * SCALE;
                    } else {
                        P_smem[tid * BK + j] = 0.0f;
                        dS_smem[tid * BK + j] = 0.0f;
                    }
                }
            } else {
                #pragma unroll
                for (int j = 0; j < BK; j++) {
                    P_smem[tid * BK + j] = 0.0f;
                    dS_smem[tid * BK + j] = 0.0f;
                }
            }
        }
        __syncthreads();

        // Phase 2: dQ[i][k] += sum_j dS[i][j] * K[j][k] (no atomics, each (i,k) unique)
        // 128 threads: i = tid/2, k_base = (tid%2)*64
        {
            int i = tid / 2;
            int k_base = (tid % 2) * 64;
            int qi = q_start + i;
            if (qi < S) {
                #pragma unroll 1
                for (int j = 0; j < BK; j++) {
                    float ds = dS_smem[i * BK + j];
                    #pragma unroll 4
                    for (int k = 0; k < 64; k += 2) {
                        __nv_bfloat162 k2 = *reinterpret_cast<__nv_bfloat162*>(&K_smem[j * D + k_base + k]);
                        float2 kf = __bfloat1622float2(k2);
                        dQ_smem[i * D + k_base + k]     += ds * kf.x;
                        dQ_smem[i * D + k_base + k + 1] += ds * kf.y;
                    }
                }
            }
        }

        // Phase 3: dK[j][k] = sum_i dS[i][j] * Q[i][k], dV[j][k] = sum_i P[i][j] * dO[i][k]
        // 128 threads: j = tid/4, k_base = (tid%4)*32
        {
            int j = tid / 4;
            int k_base = (tid % 4) * 32;
            int kj = kv_start + j;
            if (j < BK && kj < S) {
                #pragma unroll 1
                for (int k = 0; k < 32; k += 2) {
                    float dk0 = 0.0f, dk1 = 0.0f;
                    float dv0 = 0.0f, dv1 = 0.0f;
                    #pragma unroll 8
                    for (int i = 0; i < BQ; i++) {
                        float ds = dS_smem[i * BK + j];
                        float p = P_smem[i * BK + j];
                        __nv_bfloat162 q2 = *reinterpret_cast<__nv_bfloat162*>(&Q_smem[i * D + k_base + k]);
                        __nv_bfloat162 do2 = *reinterpret_cast<__nv_bfloat162*>(&dO_smem[i * D + k_base + k]);
                        float2 qf = __bfloat1622float2(q2);
                        float2 dof = __bfloat1622float2(do2);
                        dk0 += ds * qf.x;
                        dk1 += ds * qf.y;
                        dv0 += p * dof.x;
                        dv1 += p * dof.y;
                    }
                    atomicAdd(&dK_bh[kj * D + k_base + k],     dk0);
                    atomicAdd(&dK_bh[kj * D + k_base + k + 1], dk1);
                    atomicAdd(&dV_bh[kj * D + k_base + k],     dv0);
                    atomicAdd(&dV_bh[kj * D + k_base + k + 1], dv1);
                }
            }
        }
        __syncthreads();
    }

    // Write dQ (fp32 -> bf16, vectorized 4B = 2 bf16)
    for (int idx = tid; idx < BQ * D / 2; idx += THREADS) {
        int i = idx / (D / 2);
        int k = (idx % (D / 2)) * 2;
        int qi = q_start + i;
        if (qi < S) {
            __nv_bfloat162 res;
            res.x = __float2bfloat16(dQ_smem[i * D + k]);
            res.y = __float2bfloat16(dQ_smem[i * D + k + 1]);
            *reinterpret_cast<__nv_bfloat162*>(&dQ_out[base + qi * D + k]) = res;
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
        S);

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