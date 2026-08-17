#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <stdexcept>

static constexpr int32_t BLOCK_M = 4;
static constexpr int32_t BLOCK_N = 32;
static constexpr int32_t THREADS = BLOCK_M * BLOCK_N; // 128

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
    uint32_t warp_id = tid / 32;       // same as q_idx since BLOCK_M=4
    uint32_t lane_id = tid % 32;       // same as k_idx since BLOCK_N=32

    uint32_t bid = blockIdx.x;
    uint32_t bh = bid % (B * H);
    uint32_t b = bh / H;
    uint32_t h = bh % H;
    uint32_t q_tile_idx = bid / (B * H);
    uint32_t q_start = q_tile_idx * BLOCK_M;
    int valid_q_rows = idev_min(BLOCK_M, S - q_start);

    bool q_valid = (warp_id < valid_q_rows);

    extern __shared__ char smem_raw[];

    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sK = sQ + BLOCK_M * D_;
    __nv_bfloat16* sV = sK + BLOCK_N * D_;
    float* sO = reinterpret_cast<float*>(sV + BLOCK_N * D_);
    float* sMax = sO + BLOCK_M * D_;
    float* sSum = sMax + BLOCK_M;

    // Load Q tile (BLOCK_M x D) into shared memory
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

    // Initialize per-q-row accumulators
    sMax[warp_id] = -1e20f;
    sSum[warp_id] = 0.0f;
    for (int i = tid; i < BLOCK_M * D_; i += THREADS) {
        sO[i] = 0.0f;
    }
    __syncthreads();

    uint64_t base_bh = ((uint64_t)b * H + h) * S * D_;

    for (int k_start = 0; k_start < S; k_start += BLOCK_N) {
        int k_valid_cnt = idev_min(BLOCK_N, S - k_start);
        bool k_valid = (lane_id < k_valid_cnt);

        // Load K tile (BLOCK_N x D)
        for (int i = tid; i < BLOCK_N * D_; i += THREADS) {
            int k = i / D_;
            int d = i % D_;
            if (k < k_valid_cnt) {
                sK[i] = K[base_bh + (k_start + k) * D_ + d];
            } else {
                sK[i] = __float2bfloat16(0.0f);
            }
        }
        // Load V tile (BLOCK_N x D)
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

        // Compute dot product Q[q_idx] . K[k_idx] for each thread in the warp
        // All 32 threads in a warp share the same q_idx (=warp_id)
        float p_val = -1e20f;
        if (q_valid && k_valid) {
            float acc = 0.0f;
            for (int d = 0; d < D_; ++d) {
                acc += __bfloat162float(sQ[warp_id * D_ + d])
                     * __bfloat162float(sK[lane_id * D_ + d]);
            }
            p_val = acc * scale;
        }

        // Warp-wide reduction: find max over all k for this q row
        // All threads in warp contribute their p_val
        float w_max = p_val;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            w_max = fdev_fmaxf(w_max, __shfl_down_sync(0xFFFFFFFF, w_max, offset));
        }
        // w_max now holds the per-k-start chunk max for this q row (same in all threads of warp)

        // Read old max from shared memory
        float old_max = sMax[warp_id];
        float new_max = fdev_fmaxf(old_max, w_max);
        sMax[warp_id] = new_max;
        __syncthreads();

        // Re-read new_max to ensure consistency across warps (though warp already agrees)
        float cur_new_max = sMax[warp_id];

        // Only accumulate if this q row is valid and we've seen real attention scores
        if (q_valid && cur_new_max > -1e15f) {
            float p_exp = expf(p_val - cur_new_max);

            // Renormalization factor
            float scale_sum;
            if (old_max <= -1e15f) {
                scale_sum = 0.0f; // First contribution phase
            } else {
                scale_sum = expf(old_max - cur_new_max);
            }

            // Update sO: renormalize and add new contribution
            // Each lane contributes to 4 different d-values (D/32 = 128/32 = 4)
            for (int d_offset = 0; d_offset < D_; d_offset += BLOCK_N) {
                int d = lane_id + d_offset;
                float o_old = sO[warp_id * D_ + d] * scale_sum;
                if (k_valid) {
                    o_old += p_exp * __bfloat162float(sV[lane_id * D_ + d]);
                }
                sO[warp_id * D_ + d] = o_old;
            }

            // Warp-wide reduction: sum of p_exp
            float w_sum = p_exp;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                w_sum += __shfl_down_sync(0xFFFFFFFF, w_sum, offset);
            }

            // Only lane 0 writes the updated sum
            if (lane_id == 0) {
                sSum[warp_id] = sSum[warp_id] * scale_sum + w_sum;
            }
        }
        __syncthreads();
    }

    // Final normalization and store
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
                float lse = m + logf(fdev_fmaxf(final_sum, 1e-30f));
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

    float sc = 1.0f / sqrtf(static_cast<float>(D));

    int32_t num_q_tiles = (S + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(B * H * num_q_tiles);
    dim3 block(THREADS);

    // Shared memory:
    // sQ: BLOCK_M*D*2, sK: BLOCK_N*D*2, sV: BLOCK_N*D*2
    // sO: BLOCK_M*D*4, sMax: BLOCK_M*4, sSum: BLOCK_M*4
    size_t smem_size = (size_t)(BLOCK_M + 2*BLOCK_N) * D * sizeof(__nv_bfloat16)
                      + (size_t)BLOCK_M * D * sizeof(float)
                      + (size_t)(2*BLOCK_M) * sizeof(float);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    switch (D) {
        case 64:  mha_forward_kernel<64><<<grid, block, smem_size, stream>>>(Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, (int)B, (int)H, (int)S, (int)D, sc); break;
        case 128: mha_forward_kernel<128><<<grid, block, smem_size, stream>>>(Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, (int)B, (int)H, (int)S, (int)D, sc); break;
        case 256: mha_forward_kernel<256><<<grid, block, smem_size, stream>>>(Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, (int)B, (int)H, (int)S, (int)D, sc); break;
        case 512: mha_forward_kernel<512><<<grid, block, smem_size, stream>>>(Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, (int)B, (int)H, (int)S, (int)D, sc); break;
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