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
    uint32_t warp_id = tid >> 5;      // tid / 32, range [0, BLOCK_M-1]
    uint32_t lane_id = tid & 31;      // tid % 32, range [0, BLOCK_N-1]

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
    // Per-q-row max and sum for cross-k communication
    float* sMax = reinterpret_cast<float*>(sV + BLOCK_N * D_);
    float* sSum = sMax + BLOCK_M;
    // Shared storage for p values per (q,k) for reduction
    float* sP = sSum + BLOCK_M;

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

    // Local per-thread accumulators for O[saved_q_idx * D + saved_d]
    // Each thread keeps a subset of the D-dimension in registers
    // With D_=128 and 128 threads: each thread owns exactly 1 output d-element per q-row
    // Actually let's partition differently: thread t owns d = t for q_row = warp_id
    // So each thread owns exactly one d-index for its q-row
    
    uint32_t my_d = lane_id;  // Within a warp, lane_id maps to d-index for OUTPUT
    // But we need D/BLOCK_N = 128/32 = 4 elements per lane. So lane covers d={lane, lane+32, lane+64, lane+96}
    
    // Per-q-row local accumulator (register array)
    float local_o[4];  // D_/BLOCK_N = 128/32 = 4
    #pragma unroll
    for (int i = 0; i < 4; ++i) local_o[i] = 0.0f;
    
    float local_max = -1e20f;
    float local_sum = 0.0f;

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

        // Compute dot product: Q[q] . K[k_lane] where q=warp_id, k=lane_id
        float p_val = -1e20f;
        if (q_valid && k_valid) {
            float acc = 0.0f;
            #pragma unroll
            for (int d = 0; d < D_; ++d) {
                acc += __bfloat162float(sQ[warp_id * D_ + d])
                     * __bfloat162float(sK[lane_id * D_ + d]);
            }
            p_val = acc * scale;
        }

        // Store p_val in shared memory for cross-lane max reduction
        sP[warp_id * BLOCK_N + lane_id] = p_val;
        __syncthreads();

        // Find max across all k for this q_row using shared memory scan
        // (avoids warp-divergence issues with shuffle when only some lanes are valid)
        float w_max = -1e20f;
        #pragma unroll
        for (int k = 0; k < BLOCK_N; ++k) {
            float val = sP[warp_id * BLOCK_N + k];
            if (val > w_max) w_max = val;
        }

        float old_max = local_max;
        float new_max = fdev_fmaxf(old_max, w_max);
        
        // Gate accumulation on validity
        if (q_valid && new_max > -1e15f) {
            float p_exp = expf(p_val - new_max);

            // Renormalization factor
            float renorm_scale;
            if (old_max <= -1e15f) {
                renorm_scale = 0.0f;
            } else {
                renorm_scale = expf(old_max - new_max);
            }

            // Update local output accumulation: each lane owns 4 d-strides
            #pragma unroll
            for (int stride = 0; stride < 4; ++stride) {
                int d = lane_id + stride * BLOCK_N;  // stride in units of BLOCK_N
                float prev_o = local_o[stride] * renorm_scale;
                if (k_valid) {
                    prev_o += p_exp * __bfloat162float(sV[lane_id * D_ + d]);
                }
                local_o[stride] = prev_o;
            }

            // Warp reduce the sum of p_exp using shuffle (all lanes participate)
            float w_sum = p_exp;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                w_sum += __shfl_down_sync(0xFFFFFFFF, w_sum, offset);
            }

            // All lanes compute updated local_sum, but only lane 0 persists to sSum
            local_sum = local_sum * renorm_scale + w_sum;
            local_max = new_max;
        }
        __syncthreads();
    }

    // Write final results to global memory
    if (q_valid) {
        float final_sum = local_sum;
        float final_max = local_max;
        float inv = (final_sum > 1e-40f) ? (1.0f / final_sum) : 0.0f;

        // Each lane writes its 4 output elements
        #pragma unroll
        for (int stride = 0; stride < 4; ++stride) {
            int d = lane_id + stride * BLOCK_N;
            int64_t out_idx = base_bh + (q_start + warp_id) * D_ + d;
            O[out_idx] = __float2bfloat16(local_o[stride] * inv);
        }
        
        // Lane 0 writes LSE
        if (lane_id == 0) {
            int64_t lse_idx = ((uint64_t)b * H + h) * S + (q_start + warp_id);
            LSE[lse_idx] = final_max + logf(fdev_fmaxf(final_sum, 1e-30f));
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
    // sMax: BLOCK_M*4, sSum: BLOCK_M*4, sP: BLOCK_M*BLOCK_N*4
    size_t smem_size = (size_t)(BLOCK_M + 2*BLOCK_N) * D * sizeof(__nv_bfloat16)
                      + (size_t)(2*BLOCK_M + BLOCK_M*BLOCK_N) * sizeof(float);

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