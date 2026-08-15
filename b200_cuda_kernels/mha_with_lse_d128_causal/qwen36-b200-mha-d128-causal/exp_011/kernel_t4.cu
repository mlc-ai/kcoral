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

constexpr int BLOCK_M = 64;      // Query tile size
constexpr int BLOCK_D = 128;     // Head dimension
constexpr int BLOCK_S = 64;      // Key/Value tile size
constexpr int THREADS = 256;     // Threads per block
constexpr int COLS_PER_THREAD = 32; // Output columns handled per thread (covers D=128)

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
    int r = tid % BLOCK_M;
    int c_block = tid / BLOCK_M;
    int start_d = c_block * COLS_PER_THREAD;

    // Dynamic shared memory: sQ[BLOCK_M][BLOCK_D], sK[BLOCK_S][BLOCK_D], sV[BLOCK_S][BLOCK_D]
    extern __shared__ char smem_raw[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sK = sQ + BLOCK_M * BLOCK_D;
    __nv_bfloat16* sV = sK + BLOCK_S * BLOCK_D;

    // Cooperative load Q tile
    bool is_active = (r < active_rows);
    for (int i = tid; i < BLOCK_M * BLOCK_D; i += THREADS) {
        if (i / BLOCK_D < active_rows) {
            int glob_r = i / BLOCK_D;
            int glob_c = i % BLOCK_D;
            sQ[i] = Q[((b * H + h) * S + q_start + glob_r) * D + glob_c];
        }
    }
    __syncthreads();

    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(D));
    float p_max = -1e20f;
    float p_sum = 0.0f;
    float acc_o[COLS_PER_THREAD];
    for(int j = 0; j < COLS_PER_THREAD; ++j) acc_o[j] = 0.0f;

    int num_kv_tiles = (S + BLOCK_S - 1) / BLOCK_S;
    for (int kv_t = 0; kv_t < num_kv_tiles; ++kv_t) {
        int s_start = kv_t * BLOCK_S;
        
        // Cooperative load K & V tiles
        for (int i = tid; i < BLOCK_S * BLOCK_D; i += THREADS) {
            int glob_r = i / BLOCK_D;
            int glob_c = i % BLOCK_D;
            int abs_r = s_start + glob_r;
            if (abs_r < S) {
                int idx = ((b * H + h) * S + abs_r) * D + glob_c;
                sK[i] = K[idx];
                sV[i] = V[idx];
            }
        }
        __syncthreads();

        // Cache Q row in registers
        float q_reg[COLS_PER_THREAD];
        if (is_active) {
            for(int j = 0; j < COLS_PER_THREAD; ++j) {
                q_reg[j] = __bfloat162float(sQ[r * BLOCK_D + start_d + j]);
            }
        }

        int q_abs = q_start + r;
        for (int k_r = 0; k_r < BLOCK_S; ++k_r) {
            int k_abs = s_start + k_r;
            if (k_abs >= S) break;
            
            float k_reg[COLS_PER_THREAD];
            float v_reg[COLS_PER_THREAD];
            if (is_active) {
                for(int j = 0; j < COLS_PER_THREAD; ++j) {
                    k_reg[j] = __bfloat162float(sK[k_r * BLOCK_D + start_d + j]);
                    v_reg[j] = __bfloat162float(sV[k_r * BLOCK_D + start_d + j]);
                }
            }

            float score = 0.0f;
            if (is_active) {
                for(int j = 0; j < COLS_PER_THREAD; ++j) {
                    score += q_reg[j] * k_reg[j];
                }
                score *= inv_sqrt_d;
                
                // Causal mask: only attend to key positions <= query position
                if (k_abs <= q_abs) {
                    if (score > p_max) {
                        float scale = expf(p_max - score);
                        p_sum *= scale;
                        p_sum += 1.0f;
                        for(int j = 0; j < COLS_PER_THREAD; ++j) {
                            acc_o[j] = acc_o[j] * scale + v_reg[j];
                        }
                        p_max = score;
                    } else {
                        float weight = expf(score - p_max);
                        p_sum += weight;
                        for(int j = 0; j < COLS_PER_THREAD; ++j) {
                            acc_o[j] += weight * v_reg[j];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write output O
    int base_offset = ((b * H + h) * S + q_start + r) * D + start_d;
    if (is_active) {
        float norm = (p_sum > 0.0f) ? 1.0f / p_sum : 0.0f;
        for(int j = 0; j < COLS_PER_THREAD; ++j) {
            O[base_offset + j] = __float2bfloat16(acc_o[j] * norm);
        }
    }
    
    // Write LSE (only once per query row)
    if (is_active && start_d == 0) {
        float lse = (p_sum > 0.0f) ? (p_max + logf(p_sum)) : -1e20f;
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