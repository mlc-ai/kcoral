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

template<int D_>
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
    uint32_t q_start = q_tile_idx * BLOCK_M;
    int valid_q_rows = idev_min(BLOCK_M, S - q_start);

    bool q_valid = (q_idx < valid_q_rows);

    extern __shared__ char smem_raw[];

    // Flat shared memory layout - no struct to avoid alignment surprises
    // sQ: [0, BLOCK_M*D)
    // sK: [BLOCK_M*D, BLOCK_M*D + BLOCK_N*D)  
    // sV: [BLOCK_M*D + BLOCK_N*D, BLOCK_M*D + 2*BLOCK_N*D)
    // sO: [..., + BLOCK_M*D)
    // sMax: [..., + BLOCK_M*4)
    // sSum: [..., + BLOCK_M*4)
    // sP:  [..., + BLOCK_M*BLOCK_N*4) - per-q-row p values
    
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sK = sQ + BLOCK_M * D_;
    __nv_bfloat16* sV = sK + BLOCK_N * D_;
    float* sO = reinterpret_cast<float*>(sV + BLOCK_N * D_);
    volatile float* sMax = sO + BLOCK_M * D_;
    volatile float* sSum = sMax + BLOCK_M;
    volatile float* sP = sSum + BLOCK_M;

    // Load Q tile (BLOCK_M x D)
    for (int i = tid; i < BLOCK_M * D_; i += THREADS) {
        int q = i / D_;
        int d = i % D_;
        if (q < valid_q_rows) {
            sQ[i] = Q[((uint64_t)b * H + h) * S * D_ + (q_start + q) * D_ + d];
        } else {
            sQ[i] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    // Initialize per-row accumulators
    sMax[q_idx] = -1e20f;
    sSum[q_idx] = 0.0f;
    for (int i = tid; i < BLOCK_M * D_; i += THREADS) {
        sO[i] = 0.0f;
    }
    __syncthreads();

    uint64_t base_bh = ((uint64_t)b * H + h) * S * D_;

    for (int k_start = 0; k_start < S; k_start += BLOCK_N) {
        int k_valid_cnt = idev_min(BLOCK_N, S - k_start);
        bool k_valid = (k_idx < k_valid_cnt);

        // Load K tile
        for (int i = tid; i < BLOCK_N * D_; i += THREADS) {
            int k = i / D_;
            int d = i % D_;
            if (k < k_valid_cnt) {
                sK[i] = K[base_bh + (k_start + k) * D_ + d];
            } else {
                sK[i] = __float2bfloat16(0.0f);
            }
        }
        // Load V tile
        for (int i = tid; i < BLOCK_N * D_; i += THREADS) {
            int k = i / D_;
            int d = i % D_;
            if (k < k_valid_cnt) {
                sV[i] = V[base_bh + (k_start + k) * D_ + d];
            } else {
                sV[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // Compute dot product for each (q_idx, k_idx)
        float p_val = -1e20f;
        if (q_valid && k_valid) {
            float acc = 0.0f;
            for (int d = 0; d < D_; ++d) {
                acc += __bfloat162float(sQ[q_idx * D_ + d]) 
                     * __bfloat162float(sK[k_idx * D_ + d]);
            }
            p_val = acc * scale;
        }

        // Write to shared memory for cross-thread reduction
        sP[q_idx * BLOCK_N + k_idx] = p_val;
        __syncthreads();

        // Find max across all k for this q_idx
        float w_max = -1e20f;
        for (int k = 0; k < BLOCK_N; ++k) {
            float v = sP[q_idx * BLOCK_N + k];
            if (v > w_max) w_max = v;
        }

        float old_max = sMax[q_idx];
        float new_max = fdev_fmaxf(old_max, w_max);
        sMax[q_idx] = new_max;
        __syncthreads();

        // Ensure all threads see consistent new_max
        float cur_new_max = sMax[q_idx];

        // Only accumulate if this q row is valid and we've seen at least one finite attention weight
        if (q_valid && cur_new_max > -1e15f) {
            float p_exp = expf(p_val - cur_new_max);
            
            // scale_sum: how to renormalize previously accumulated values
            float scale_sum;
            if (old_max <= -1e15f) {
                scale_sum = 0.0f; // First contribution
            } else {
                scale_sum = expf(old_max - cur_new_max);
            }

            // Partition D dimension: each k_idx handles a disjoint subset
            int d_lo = (k_idx * D_) / BLOCK_N;
            int d_hi = ((k_idx + 1) * D_) / BLOCK_N;

            for (int d = d_lo; d < d_hi; ++d) {
                float o_acc = sO[q_idx * D_ + d] * scale_sum;
                if (k_valid) {
                    o_acc += p_exp * __bfloat162float(sV[k_idx * D_ + d]);
                }
                sO[q_idx * D_ + d] = o_acc;
            }

            // Only k_idx==0 for each q row updates the sum
            if (k_idx == 0) {
                float w_sum = 0.0f;
                for (int k = 0; k < BLOCK_N; ++k) {
                    float pv = sP[q_idx * BLOCK_N + k];
                    if (pv > -1e15f) {
                        w_sum += expf(pv - cur_new_max);
                    }
                }
                sSum[q_idx] = sSum[q_idx] * scale_sum + w_sum;
            }
        }
        __syncthreads();
    }

    // Store results
    __syncthreads();
    for (int i = tid; i < BLOCK_M * D_; i += THREADS) {
        int q = i / D_;
        int d = i % D_;
        if (q < valid_q_rows) {
            float final_sum = sSum[q];
            float inv = (final_sum > 1e-40f) ? (1.0f / final_sum) : 0.0f;
            O[base_bh + (q_start + q) * D_ + d] = __float2bfloat16(sO[i] * inv);
            if (d == 0) {
                float m = sMax[q];
                float log_s = logf(fdev_fmaxf(final_sum, 1e-30f));
                LSE[((uint64_t)b * H + h) * S + (q_start + q)] = m + log_s;
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

    // Calculate shared memory size
    // sQ: BLOCK_M*D*2 + sK: BLOCK_N*D*2 + sV: BLOCK_N*D*2
    // sO: BLOCK_M*D*4 + sMax: BLOCK_M*4 + sSum: BLOCK_M*4 + sP: BLOCK_M*BLOCK_N*4
    size_t smem_size = (size_t)(BLOCK_M + 2*BLOCK_N) * D * sizeof(__nv_bfloat16)
                      + (size_t)BLOCK_M * D * sizeof(float)
                      + (size_t)(2*BLOCK_M + BLOCK_M*BLOCK_N) * sizeof(float);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    // Dispatch based on D (up to template specialization limit)
    switch (D) {
        case 64:  mha_forward_kernel<64><<<grid, block, smem_size, stream>>>(Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, (int)B, (int)H, (int)S, (int)D, scale); break;
        case 128: mha_forward_kernel<128><<<grid, block, smem_size, stream>>>(Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, (int)B, (int)H, (int)S, (int)D, scale); break;
        case 256: mha_forward_kernel<256><<<grid, block, smem_size, stream>>>(Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, (int)B, (int)H, (int)S, (int)D, scale); break;
        case 512: mha_forward_kernel<512><<<grid, block, smem_size, stream>>>(Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, (int)B, (int)H, (int)S, (int)D, scale); break;
        default: TVM_FFI_THROW(std::runtime_error) << "Unsupported head dimension D=" << D;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TVM_FFI_THROW(std::runtime_error) << "CUDA kernel launch failed: " << cudaGetErrorString(err);
    }
    cudaStreamSynchronize(stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

} // namespace tvm_ffi_mha