#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cmath>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <stdexcept>

static constexpr int32_t TILE_M = 4;
static constexpr int32_t BLOCK_N = 64;
static constexpr int32_t THREADS = 256;

__device__ __forceinline__ float fdev_fmaxf(float a, float b) { return a > b ? a : b; }
__device__ __forceinline__ float fdev_fminf(float a, float b) { return a < b ? a : b; }
__device__ __forceinline__ int idev_min(int a, int b) { return a < b ? a : b; }

__global__ void mha_forward_kernel(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* O,
    float* LSE,
    int B, int H, int S, int D,
    float scale) 
{
    uint32_t tid = threadIdx.x;
    uint32_t q_idx = tid / BLOCK_N;
    uint32_t k_idx = tid % BLOCK_N;

    uint32_t bid = blockIdx.x;
    uint32_t bh = bid % (B * H);
    uint32_t b = bh / H;
    uint32_t h = bh % H;
    uint32_t q_tile_idx = bid / (B * H);
    uint32_t q_start = q_tile_idx * TILE_M;
    uint32_t valid_q_rows = idev_min(TILE_M, S - q_start);

    bool q_valid = (q_idx < valid_q_rows);

    extern __shared__ char smem[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK = sQ + TILE_M * D;
    __nv_bfloat16* sV = sK + BLOCK_N * D;
    float* sO = reinterpret_cast<float*>(sV + BLOCK_N * D);
    float* sMax = sO + TILE_M * D;
    float* sSum = sMax + TILE_M;
    float* sP = sSum + TILE_M;

    // Load Q tile
    for (int i = tid; i < TILE_M * D; i += THREADS) {
        int q = i / D;
        if (q < valid_q_rows) {
            sQ[i] = Q[((uint64_t)b * H + h) * S * D + (q_start + q) * D + (i % D)];
        } else {
            sQ[i] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    // Initialize accumulators
    sMax[q_idx] = q_valid ? -1e20f : 0.0f;
    sSum[q_idx] = 0.0f;
    for (int i = tid; i < TILE_M * D; i += THREADS) {
        sO[i] = 0.0f;
    }
    __syncthreads();

    uint64_t base_bh = ((uint64_t)b * H + h) * S * D;

    for (int k_start = 0; k_start < S; k_start += BLOCK_N) {
        uint32_t k_valid_cnt = idev_min(BLOCK_N, S - k_start);
        bool k_valid = (k_idx < k_valid_cnt) && q_valid;

        // Load K tile
        for (int i = tid; i < BLOCK_N * D; i += THREADS) {
            int k = i / D;
            if (k < k_valid_cnt) {
                sK[i] = K[base_bh + (k_start + k) * D + (i % D)];
            } else {
                sK[i] = __float2bfloat16(0.0f);
            }
        }
        // Load V tile
        for (int i = tid; i < BLOCK_N * D; i += THREADS) {
            int k = i / D;
            if (k < k_valid_cnt) {
                sV[i] = V[base_bh + (k_start + k) * D + (i % D)];
            } else {
                sV[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // Compute P = Q @ K^T * scale
        float p_val = 0.0f;
        if (k_valid) {
            for (int d = 0; d < D; ++d) {
                p_val += __bfloat162float(sQ[q_idx * D + d]) * __bfloat162float(sK[k_idx * D + d]);
            }
            p_val *= scale;
        } else {
            p_val = -1e20f;
        }
        sP[k_idx] = p_val;
        __syncthreads();

        // Warp reduce max
        float w_max = p_val;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            w_max = fdev_fmaxf(w_max, __shfl_down_sync(0xFFFFFFFF, w_max, offset));
        }
            
        float old_max = sMax[q_idx];
        float new_max = fdev_fmaxf(old_max, w_max);
        sMax[q_idx] = new_max;
        __syncthreads();

        float p_exp = expf(sP[k_idx] - new_max);
        float scale_sum = expf(old_max - new_max);

        // Warp reduce sum of p_exp
        float w_sum = p_exp;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            w_sum += __shfl_down_sync(0xFFFFFFFF, w_sum, offset);
        }

        // Partition D dimension among k_idx threads to avoid shared memory bank conflicts/races
        int d_start = (k_idx * D) / BLOCK_N;
        int d_end = ((k_idx + 1) * D) / BLOCK_N;
        
        for (int d = d_start; d < d_end; ++d) {
            float v_val = __bfloat162float(sV[k_idx * D + d]);
            sO[q_idx * D + d] = sO[q_idx * D + d] * scale_sum + p_exp * v_val;
        }

        if (k_idx == 0) {
            sSum[q_idx] = sSum[q_idx] * scale_sum + w_sum;
        }
        __syncthreads();
    }

    // Final normalization and store to global memory
    __syncthreads();
    for (int i = tid; i < TILE_M * D; i += THREADS) {
        int q = i / D;
        int d = i % D;
        if (q < valid_q_rows) {
            float inv_sum = (sSum[q] > 0.0f) ? (1.0f / sSum[q]) : 0.0f;
            O[base_bh + (q_start + q) * D + d] = __float2bfloat16(sO[i] * inv_sum);
            if (d == 0) {
                LSE[((uint64_t)b * H + h) * S + (q_start + q)] = sMax[q] + logf(fdev_fmaxf(sSum[q], 1e-30f));
            }
        }
    }
}

namespace tvm_ffi_mha {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    cudaSetDevice(Q.device().device_id);
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    float scale = 1.0f / sqrtf(static_cast<float>(D));

    int32_t num_q_tiles = (S + TILE_M - 1) / TILE_M;
    dim3 grid(B * H * num_q_tiles);
    dim3 block(THREADS);
    
    size_t smem_size = sizeof(__nv_bfloat16) * ((TILE_M + 2 * BLOCK_N) * D) + 
                       sizeof(float) * ((TILE_M * D) + 2 * TILE_M + BLOCK_N);
                       
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_forward_kernel<<<grid, block, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S), static_cast<int>(D), scale);
        
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TVM_FFI_THROW(std::runtime_error) << "CUDA kernel launch failed: " << cudaGetErrorString(err);
    }
    cudaStreamSynchronize(stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

} // namespace tvm_ffi_mha