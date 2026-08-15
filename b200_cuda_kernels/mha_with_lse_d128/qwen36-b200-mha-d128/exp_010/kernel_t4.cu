#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <stdexcept>

static constexpr int32_t BLOCK_M = 32;
static constexpr int32_t BLOCK_N = 32;
static constexpr int32_t THREADS = BLOCK_M * BLOCK_N;

__device__ __forceinline__ float fdev_fmaxf(float a, float b) { return a > b ? a : b; }
__device__ __forceinline__ int idev_min(int a, int b) { return a < b ? a : b; }

struct __align__(16) SharedMemory {
    __nv_bfloat16 sQ[BLOCK_M * 128];       // Load Q tile
    __nv_bfloat16 sK[BLOCK_N * 128];       // Load K tile  
    __nv_bfloat16 sV[BLOCK_N * 128];       // Load V tile
    float sO[BLOCK_M * 128];               // Accumulate output
    float sMax[BLOCK_M];                   // Per-q-row max
    float sSum[BLOCK_M];                   // Per-q-row sum
};

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
    uint32_t q_idx = tid / BLOCK_N;   // [0..BLOCK_M-1]
    uint32_t k_idx = tid % BLOCK_N;   // [0..BLOCK_N-1]

    uint32_t bid = blockIdx.x;
    uint32_t bh = bid % (B * H);
    uint32_t b = bh / H;
    uint32_t h = bh % H;
    uint32_t q_tile_idx = bid / (B * H);
    uint32_t q_start = q_tile_idx * BLOCK_M;
    uint32_t valid_q_rows = idev_min(BLOCK_M, S - q_start);

    bool q_valid = (q_idx < valid_q_rows);

    extern __shared__ char smem_raw[];
    SharedMemory& smem = *reinterpret_cast<SharedMemory*>(smem_raw);

    // Load Q tile (BLOCK_M x D)
    for (int i = tid; i < BLOCK_M * D; i += THREADS) {
        int q = i / D;
        int d = i % D;
        if (q < valid_q_rows) {
            smem.sQ[i] = Q[((uint64_t)b * H + h) * S * D + (q_start + q) * D + d];
        } else {
            smem.sQ[i] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    // Initialize per-q-row accumulators
    smem.sMax[q_idx] = -FLT_MAX;
    smem.sSum[q_idx] = 0.0f;
    for (int i = tid; i < BLOCK_M * D; i += THREADS) {
        smem.sO[i] = 0.0f;
    }
    __syncthreads();

    uint64_t base_bh = ((uint64_t)b * H + h) * S * D;

    // Iterate over K/V chunks along S dimension
    for (int k_start = 0; k_start < S; k_start += BLOCK_N) {
        uint32_t k_valid_cnt = idev_min(BLOCK_N, S - k_start);
        bool k_valid = (k_idx < k_valid_cnt);

        // Load K tile (BLOCK_N x D)
        for (int i = tid; i < BLOCK_N * D; i += THREADS) {
            int k = i / D;
            int d = i % D;
            if (k < k_valid_cnt) {
                smem.sK[i] = K[base_bh + (k_start + k) * D + d];
            } else {
                smem.sK[i] = __float2bfloat16(0.0f);
            }
        }
        // Load V tile (BLOCK_N x D)
        for (int i = tid; i < BLOCK_N * D; i += THREADS) {
            int k = i / D;
            int d = i % D;
            if (k < k_valid_cnt) {
                smem.sV[i] = V[base_bh + (k_start + k) * D + d];
            } else {
                smem.sV[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // Compute P[q_idx][k_idx] = sum_d(Q * K) * scale
        float p_val = -FLT_MAX;
        if (q_valid && k_valid) {
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) {
                acc += __bfloat162float(smem.sQ[q_idx * D + d]) 
                     * __bfloat162float(smem.sK[k_idx * D + d]);
            }
            p_val = acc * scale;
        }
        // Store in register - no shared mem needed since we reduce per-q-row below
        // Using a per-q-row shared mem to communicate max is enough

        // Write p_val to a per-q-row scratch area in shared memory
        // All threads with same q_idx form a subgroup of BLOCK_N threads
        // We store p_val contiguously per q_idx group
        // Offset = q_idx * BLOCK_N + k_idx
        float* sP_base = reinterpret_cast<float*>(smem.sSum + BLOCK_M);
        sP_base[q_idx * BLOCK_N + k_idx] = p_val;
        __syncthreads();

        // Each q-row leader reduces over its BLOCK_N k values
        float w_max = -FLT_MAX;
        for (int k = 0; k < k_valid_cnt; ++k) {
            float val = sP_base[q_idx * BLOCK_N + k];
            if (val > w_max) w_max = val;
        }

        float old_max = smem.sMax[q_idx];
        float new_max = fdev_fmaxf(old_max, w_max);
        smem.sMax[q_idx] = new_max;
        __syncthreads();

        // Gate: only process if q row is valid AND we have at least one valid k contributing
        if (q_valid && new_max > -1e10f) {
            float p_exp = expf(p_val - new_max);
            
            // Guard against div-by-zero type issues when old_max == -FLT_MAX
            float scale_sum;
            if (old_max <= -1e10f) {
                scale_sum = 0.0f;  // First meaningful iteration: old O is zero
            } else {
                scale_sum = expf(old_max - new_max);
            }

            // Partition D among k_idx threads to avoid race conditions
            int d_start_part = (k_idx * D) / BLOCK_N;
            int d_end_part = ((k_idx + 1) * D) / BLOCK_N;

            for (int d = d_start_part; d < d_end_part; ++d) {
                float o_val = smem.sO[q_idx * D + d] * scale_sum;
                // Only add V contribution if this k is valid
                if (k_valid) {
                    o_val += p_exp * __bfloat162float(smem.sV[k_idx * D + d]);
                }
                smem.sO[q_idx * D + d] = o_val;
            }

            // Sum reduction: only thread 0 in each q-group updates sSum
            if (k_idx == 0) {
                float w_sum = 0.0f;
                for (int k = 0; k < k_valid_cnt; ++k) {
                    float pv = sP_base[q_idx * BLOCK_N + k];
                    if (pv > -1e10f) {
                        w_sum += expf(pv - new_max);
                    }
                }
                smem.sSum[q_idx] = smem.sSum[q_idx] * scale_sum + w_sum;
            }
        }
        __syncthreads();
    }

    // Final normalization and store to global memory
    __syncthreads();
    for (int i = tid; i < BLOCK_M * D; i += THREADS) {
        int q = i / D;
        int d = i % D;
        if (q < valid_q_rows) {
            float inv_sum = (smem.sSum[q] > 1e-40f) ? (1.0f / smem.sSum[q]) : 0.0f;
            O[base_bh + (q_start + q) * D + d] = __float2bfloat16(smem.sO[i] * inv_sum);
            if (d == 0) {
                float lse = smem.sMax[q] + logf(fdev_fmaxf(smem.sSum[q], 1e-30f));
                LSE[((uint64_t)b * H + h) * S + (q_start + q)] = lse;
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

    int32_t num_q_tiles = (S + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(B * H * num_q_tiles);
    dim3 block(THREADS);

    size_t smem_size = sizeof(SharedMemory) + BLOCK_M * BLOCK_N * sizeof(float);
    
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