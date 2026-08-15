#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

constexpr int BLOCK_M = 64;      // Query tile rows
constexpr int BLOCK_D = 128;     // Head dimension
constexpr int BLOCK_S = 64;      // Key/Value tile rows
constexpr int THREADS = 256;
constexpr int NUM_TPR = THREADS / BLOCK_M; // 4 threads per query row
constexpr int SEG_LEN = BLOCK_D / NUM_TPR; // 32 D-elements per thread

__global__ void MhaCausalKernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, int D
) {
    int bid = blockIdx.x;
    int num_q_tiles = (S + BLOCK_M - 1) / BLOCK_M;
    int total_tiles = B * H * num_q_tiles;
    if (bid >= total_tiles) return;

    int bh = bid / num_q_tiles;
    int b = bh / H;
    int h = bh % H;
    int q_tile_idx = bid % num_q_tiles;
    
    int q_start = q_tile_idx * BLOCK_M;
    int q_end = q_start + BLOCK_M < S ? q_start + BLOCK_M : S;
    int active_rows = q_end - q_start;

    int tid = threadIdx.x;
    int r = tid / NUM_TPR;          // Row index within tile (0..63)
    int tpr = tid % NUM_TPR;        // Thread-per-row index (0..3)
    int start_d = tpr * SEG_LEN;    // D-dimension segment start (0, 32, 64, 96)
    
    bool is_active = (r < active_rows);

    // Shared memory: sQ[BLOCK_M x BLOCK_D], sK[BLOCK_S x BLOCK_D], sV[BLOCK_S x BLOCK_D]
    extern __shared__ char smem_raw[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sK = sQ + BLOCK_M * BLOCK_D;
    __nv_bfloat16* sV = sK + BLOCK_S * BLOCK_D;

    // Cooperative load Q tile
    for (int i = tid; i < BLOCK_M * BLOCK_D; i += THREADS) {
        int glob_r = i / BLOCK_D;
        int glob_c = i % BLOCK_D;
        if (glob_r < active_rows) {
            sQ[glob_r * BLOCK_D + glob_c] = Q[((b * H + h) * S + q_start + glob_r) * D + glob_c];
        }
    }
    __syncthreads();

    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(D));
    constexpr float NEG_INF = -1e20f;
    float p_max = NEG_INF;
    float p_sum = 0.0f;
    float acc_o[SEG_LEN];
    for(int j = 0; j < SEG_LEN; ++j) acc_o[j] = 0.0f;

    int num_kv_tiles = (S + BLOCK_S - 1) / BLOCK_S;
    int lane_base = r * NUM_TPR; // Starting lane of this row group

    for (int kv_t = 0; kv_t < num_kv_tiles; ++kv_t) {
        int s_start = kv_t * BLOCK_S;
        
        // Cooperative load K & V tiles
        for (int i = tid; i < BLOCK_S * BLOCK_D; i += THREADS) {
            int glob_r = i / BLOCK_D;
            int glob_c = i % BLOCK_D;
            int abs_r = s_start + glob_r;
            if (abs_r < S) {
                int idx = ((b * H + h) * S + abs_r) * D + glob_c;
                sK[glob_r * BLOCK_D + glob_c] = K[idx];
                sV[glob_r * BLOCK_D + glob_c] = V[idx];
            }
        }
        __syncthreads();

        int q_abs = q_start + r;
        for (int k_r = 0; k_r < BLOCK_S; ++k_r) {
            int k_abs = s_start + k_r;
            if (k_abs >= S) break;
            
            // Compute partial dot product over segment
            float partial_score = 0.0f;
            if (is_active) {
                int q_off = r * BLOCK_D + start_d;
                int k_off = k_r * BLOCK_D + start_d;
                for(int j = 0; j < SEG_LEN; ++j) {
                    partial_score += __bfloat162float(sQ[q_off + j]) * 
                                     __bfloat162float(sK[k_off + j]);
                }
            }
            
            // Reduce partial scores across the 4 threads in the row group
            float full_score = partial_score;
            full_score += __shfl_sync(0xFFFFFFFF, partial_score, lane_base + 1);
            full_score += __shfl_sync(0xFFFFFFFF, partial_score, lane_base + 2);
            full_score += __shfl_sync(0xFFFFFFFF, partial_score, lane_base + 3);
            full_score *= inv_sqrt_d;

            if (is_active) {
                bool attend = (k_abs <= q_abs);
                float current_max = p_max;
                
                if (attend) {
                    if (tpr == 0) {
                        if (full_score > current_max) {
                            p_sum *= expf(current_max - full_score);
                            p_sum += 1.0f;
                            p_max = full_score;
                        } else {
                            p_sum += expf(full_score - current_max);
                        }
                    }
                    
                    // Synchronize p_max across the row group
                    p_max = __shfl_sync(0xFFFFFFFF, p_max, lane_base);
                    
                    float acc_ratio = expf(current_max - p_max);
                    float v_scale = expf(full_score - p_max);
                    
                    int v_off = k_r * BLOCK_D + start_d;
                    for(int j = 0; j < SEG_LEN; ++j) {
                        acc_o[j] = acc_o[j] * acc_ratio + v_scale * __bfloat162float(sV[v_off + j]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write output O
    if (is_active) {
        float norm = (p_sum > 0.0f) ? 1.0f / p_sum : 0.0f;
        int base_offset = ((b * H + h) * S + q_start + r) * D + start_d;
        for(int j = 0; j < SEG_LEN; ++j) {
            O[base_offset + j] = __float2bfloat16(acc_o[j] * norm);
        }
    }
    
    // Write LSE (only once per query row)
    if (is_active && tpr == 0) {
        float lse = (p_sum > 0.0f) ? (p_max + logf(p_sum)) : NEG_INF;
        LSE[(b * H + h) * S + q_start + r] = lse;
    }
}

namespace mha_impl {
void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = static_cast<int>(Q.size(0));
    int H = static_cast<int>(Q.size(1));
    int S = static_cast<int>(Q.size(2));
    int D = static_cast<int>(Q.size(3));

    // Dynamic shared memory: sQ(BLOCK_M*D) + sK(BLOCK_S*D) + sV(BLOCK_S*D)
    size_t smem_elements = static_cast<size_t>(BLOCK_M * BLOCK_D) + 
                           2ULL * static_cast<size_t>(BLOCK_S * BLOCK_D);
    size_t smem_bytes = smem_elements * sizeof(__nv_bfloat16);
    
    int blocks = B * H * ((S + BLOCK_M - 1) / BLOCK_M);
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    MhaCausalKernel<<<blocks, THREADS, smem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        B, H, S, D);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
} // namespace mha_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);