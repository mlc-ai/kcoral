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

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int D = 128;
constexpr int THREADS = 256;
constexpr int B_CONST = 4;
constexpr int H_CONST = 48;

__global__ void backward_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    float* __restrict__ dK_float,
    float* __restrict__ dV_float,
    int S)
{
    const float scale = rsqrtf((float)D);

    int num_q_blocks = (S + BM - 1) / BM;
    int q_block = blockIdx.x;
    int bh = q_block / num_q_blocks;
    int q_tile = q_block % num_q_blocks;
    int batch = bh / H_CONST;
    int head = bh % H_CONST;

    int64_t bh_offset = (int64_t)batch * H_CONST * S * D + (int64_t)head * S * D;
    int64_t lse_offset = (int64_t)batch * H_CONST * S + (int64_t)head * S;

    const __nv_bfloat16* Q_ptr = Q + bh_offset;
    const __nv_bfloat16* K_ptr = K + bh_offset;
    const __nv_bfloat16* V_ptr = V + bh_offset;
    const __nv_bfloat16* O_ptr = O + bh_offset;
    const __nv_bfloat16* dO_ptr = dO + bh_offset;
    const float* L_ptr = L + lse_offset;
    __nv_bfloat16* dQ_ptr = dQ + bh_offset;
    float* dK_ptr = dK_float + bh_offset;
    float* dV_ptr = dV_float + bh_offset;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_Q  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_K  = smem_Q + BM * D;
    __nv_bfloat16* smem_V  = smem_K + BN * D;
    __nv_bfloat16* smem_dO = smem_V + BN * D;
    float* smem_P   = reinterpret_cast<float*>(smem_dO + BM * D);
    float* smem_dP  = smem_P + BM * BN;
    float* smem_dQ  = smem_dP + BM * BN;
    float* smem_LSE = smem_dQ + BM * D;
    float* smem_D   = smem_LSE + BM;

    int tid = threadIdx.x;
    int q_start = q_tile * BM;
    int q_len = min(BM, S - q_start);

    // Load Q tile [BM, D]
    for (int i = tid; i < BM * D; i += THREADS) {
        int m = i / D, d = i % D;
        smem_Q[m * D + d] = (m < q_len) ? Q_ptr[(q_start + m) * D + d] : __float2bfloat16(0.0f);
    }

    // Load dO tile [BM, D]
    for (int i = tid; i < BM * D; i += THREADS) {
        int m = i / D, d = i % D;
        smem_dO[m * D + d] = (m < q_len) ? dO_ptr[(q_start + m) * D + d] : __float2bfloat16(0.0f);
    }

    // Load LSE [BM]
    for (int m = tid; m < BM; m += THREADS) {
        smem_LSE[m] = (m < q_len) ? L_ptr[q_start + m] : 0.0f;
    }

    // Compute D[m] = sum_d(dO[m,d] * O[m,d])
    for (int m = tid; m < BM; m += THREADS) {
        if (m < q_len) {
            float d_val = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < D; d++) {
                d_val += __bfloat162float(dO_ptr[(q_start + m) * D + d]) *
                         __bfloat162float(O_ptr[(q_start + m) * D + d]);
            }
            smem_D[m] = d_val;
        } else {
            smem_D[m] = 0.0f;
        }
    }

    // Initialize dQ accumulator to zero
    for (int i = tid; i < BM * D; i += THREADS) {
        smem_dQ[i] = 0.0f;
    }

    __syncthreads();

    // Main loop over KV blocks
    int num_kv_blocks = (S + BN - 1) / BN;
    for (int kv_tile = 0; kv_tile < num_kv_blocks; kv_tile++) {
        int kv_start = kv_tile * BN;
        int kv_len = min(BN, S - kv_start);

        // Load K and V tiles [BN, D]
        for (int i = tid; i < BN * D; i += THREADS) {
            int n = i / D, d = i % D;
            smem_K[n * D + d] = (n < kv_len) ? K_ptr[(kv_start + n) * D + d] : __float2bfloat16(0.0f);
            smem_V[n * D + d] = (n < kv_len) ? V_ptr[(kv_start + n) * D + d] : __float2bfloat16(0.0f);
        }

        __syncthreads();

        // S = Q @ K^T * scale  -> [BM, BN]
        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int m = idx / BN, n = idx % BN;
            float sum = 0.0f;
            #pragma unroll 32
            for (int d = 0; d < D; d++) {
                sum += __bfloat162float(smem_Q[m * D + d]) *
                       __bfloat162float(smem_K[n * D + d]);
            }
            smem_P[m * BN + n] = sum * scale;
        }

        __syncthreads();

        // P = exp(S - LSE)  -> [BM, BN]
        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int m = idx / BN, n = idx % BN;
            if (m < q_len && n < kv_len) {
                smem_P[idx] = expf(smem_P[idx] - smem_LSE[m]);
            } else {
                smem_P[idx] = 0.0f;
            }
        }

        __syncthreads();

        // dP = dO @ V^T  -> [BM, BN]
        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int m = idx / BN, n = idx % BN;
            float sum = 0.0f;
            #pragma unroll 32
            for (int d = 0; d < D; d++) {
                sum += __bfloat162float(smem_dO[m * D + d]) *
                       __bfloat162float(smem_V[n * D + d]);
            }
            smem_dP[idx] = sum;
        }

        __syncthreads();

        // dS = P * (dP - D)  (in-place in smem_dP)
        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int m = idx / BN;
            smem_dP[idx] = smem_P[idx] * (smem_dP[idx] - smem_D[m]);
        }

        __syncthreads();

        // dV += P^T @ dO  -> [BN, D]  (atomic add to global float32)
        for (int idx = tid; idx < BN * D; idx += THREADS) {
            int n = idx / D, d = idx % D;
            float sum = 0.0f;
            for (int m = 0; m < BM; m++) {
                sum += smem_P[m * BN + n] * __bfloat162float(smem_dO[m * D + d]);
            }
            if (n < kv_len) {
                atomicAdd(&dV_ptr[(kv_start + n) * D + d], sum);
            }
        }

        // dK += dS^T @ Q * scale  -> [BN, D]  (atomic add to global float32)
        for (int idx = tid; idx < BN * D; idx += THREADS) {
            int n = idx / D, d = idx % D;
            float sum = 0.0f;
            for (int m = 0; m < BM; m++) {
                sum += smem_dP[m * BN + n] * __bfloat162float(smem_Q[m * D + d]);
            }
            sum *= scale;
            if (n < kv_len) {
                atomicAdd(&dK_ptr[(kv_start + n) * D + d], sum);
            }
        }

        // dQ += dS @ K * scale  -> [BM, D]  (accumulate in shared memory)
        for (int idx = tid; idx < BM * D; idx += THREADS) {
            int m = idx / D, d = idx % D;
            float sum = 0.0f;
            for (int n = 0; n < BN; n++) {
                sum += smem_dP[m * BN + n] * __bfloat162float(smem_K[n * D + d]);
            }
            smem_dQ[idx] += sum * scale;
        }

        __syncthreads();
    }

    // Store dQ to global (convert float32 -> bf16)
    for (int i = tid; i < BM * D; i += THREADS) {
        int m = i / D, d = i % D;
        if (m < q_len) {
            dQ_ptr[(q_start + m) * D + d] = __float2bfloat16(smem_dQ[m * D + d]);
        }
    }
}

__global__ void convert_kernel(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t S = Q.size(2);

    // Shared memory size: Q + K + V + dO (bf16) + P + dP (f32) + dQ (f32) + LSE + D (f32)
    size_t smem_size =
        (size_t)BM * D * sizeof(__nv_bfloat16) +   // Q
        (size_t)BN * D * sizeof(__nv_bfloat16) +   // K
        (size_t)BN * D * sizeof(__nv_bfloat16) +   // V
        (size_t)BM * D * sizeof(__nv_bfloat16) +   // dO
        (size_t)BM * BN * sizeof(float) +          // P
        (size_t)BM * BN * sizeof(float) +          // dP/dS
        (size_t)BM * D * sizeof(float) +           // dQ accumulator
        (size_t)BM * sizeof(float) +               // LSE
        (size_t)BM * sizeof(float);                // D

    CUDA_CHECK(cudaFuncSetAttribute(backward_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    // Allocate temp float32 buffers for dK and dV (for atomic accumulation)
    size_t temp_size = (size_t)B_CONST * H_CONST * S * D * sizeof(float);
    float* dK_float = nullptr;
    float* dV_float = nullptr;
    CUDA_CHECK(cudaMallocAsync(&dK_float, temp_size, stream));
    CUDA_CHECK(cudaMallocAsync(&dV_float, temp_size, stream));
    CUDA_CHECK(cudaMemsetAsync(dK_float, 0, temp_size, stream));
    CUDA_CHECK(cudaMemsetAsync(dV_float, 0, temp_size, stream));

    // Launch backward kernel
    int num_q_blocks = ((int)S + BM - 1) / BM;
    int grid_size = B_CONST * H_CONST * num_q_blocks;

    backward_kernel<<<grid_size, THREADS, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        dK_float, dV_float,
        (int)S);

    CUDA_CHECK(cudaGetLastError());

    // Convert dK_float and dV_float to bf16 output
    int total_elements = (int)((size_t)B_CONST * H_CONST * S * D);
    int convert_threads = 256;
    int convert_blocks = (total_elements + convert_threads - 1) / convert_threads;

    convert_kernel<<<convert_blocks, convert_threads, 0, stream>>>(
        dK_float, static_cast<__nv_bfloat16*>(dK.data_ptr()), total_elements);
    convert_kernel<<<convert_blocks, convert_threads, 0, stream>>>(
        dV_float, static_cast<__nv_bfloat16*>(dV.data_ptr()), total_elements);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(dK_float, stream));
    CUDA_CHECK(cudaFreeAsync(dV_float, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128