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

constexpr int BLOCK_M = 64;      // Query tile height
constexpr int BLOCK_N = 64;      // Key/Value tile height
constexpr int BLOCK_D = 128;     // Head dimension
constexpr int THREADS = 256;
constexpr int TPR = THREADS / BLOCK_M; // Threads per query row (4)
constexpr int SEG_D = BLOCK_D / TPR;   // D-dimension segment per thread (32)

__global__ void MhaCausalKernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, int D) 
{
    int bid = blockIdx.x;
    int num_q_tiles = (S + BLOCK_M - 1) / BLOCK_M;
    int total_tiles = B * H * num_q_tiles;
    if (bid >= total_tiles) return;

    int bh = bid / num_q_tiles;
    int b = bh / H;
    int h = bh % H;
    int q_tile_idx = bid % num_q_tiles;
    
    int q_start = q_tile_idx * BLOCK_M;
    int active_rows = (q_start + BLOCK_M < S) ? BLOCK_M : S - q_start;

    int tid = threadIdx.x;
    int r = tid / TPR;
    int tpr = tid % TPR;
    int seg_start = tpr * SEG_D;
    bool active = (r < active_rows);

    // Shared memory: sQ[BLOCK_M x BLOCK_D], sK[BLOCK_N x BLOCK_D], sV[BLOCK_N x BLOCK_D]
    extern __shared__ char smem_raw[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sK = sQ + BLOCK_M * BLOCK_D;
    __nv_bfloat16* sV = sK + BLOCK_N * BLOCK_D;

    // Cooperative load Q tile
    for (int i = tid; i < BLOCK_M * BLOCK_D; i += THREADS) {
        int mr = i / BLOCK_D;
        int md = i % BLOCK_D;
        if (mr < active_rows) {
            sQ[mr * BLOCK_D + md] = Q[((b * H + h) * S + q_start + mr) * D + md];
        }
    }
    __syncthreads();

    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(D));
    constexpr float NEG_INF = -1e20f;
    float m_i = NEG_INF;
    float l_i = 0.0f;
    float o_reg[SEG_D];
    for(int j=0; j<SEG_D; ++j) o_reg[j] = 0.0f;

    int num_kv_tiles = (S + BLOCK_N - 1) / BLOCK_N;
    int q_abs = q_start + r;

    for (int n_block = 0; n_block < num_kv_tiles; ++n_block) {
        int s_start = n_block * BLOCK_N;
        
        // Cooperative load K & V tiles
        for (int i = tid; i < BLOCK_N * BLOCK_D; i += THREADS) {
            int nr = i / BLOCK_D;
            int nd = i % BLOCK_D;
            int abs_r = s_start + nr;
            if (abs_r < S) {
                int idx = ((b * H + h) * S + abs_r) * D + nd;
                sK[nr * BLOCK_D + nd] = K[idx];
                sV[nr * BLOCK_D + nd] = V[idx];
            }
        }
        __syncthreads();

        // Causal mask boundary within this KV tile
        int max_nr = q_abs - s_start;
        if (max_nr < 0) {
            continue; // Entire KV tile is in the future
        }
        bool full_tile = (max_nr >= BLOCK_N);
        int iter_limit = full_tile ? BLOCK_N : (max_nr + 1);

        for (int nr = 0; nr < BLOCK_N; ++nr) {
            float sum = 0.0f;
            if (active) {
                int base_q = r * BLOCK_D + seg_start;
                int base_k = nr * BLOCK_D + seg_start;
                for(int j=0; j<SEG_D; ++j) {
                    sum += __bfloat162float(sQ[base_q+j]) * __bfloat162float(sK[base_k+j]);
                }
            }
            
            // Reduce dot product across the 4 threads sharing this row
            sum += __shfl_sync(0xffffffff, sum, r*TPR + 1);
            sum += __shfl_sync(0xffffffff, sum, r*TPR + 2);
            sum += __shfl_sync(0xffffffff, sum, r*TPR + 3);
            sum *= inv_sqrt_d;

            if (active && !full_tile && nr >= iter_limit) {
                continue; // Skip past-key positions due to causal mask
            }
            
            if (active) {
                float old_m = m_i;
                if (sum > m_i) {
                    m_i = sum;
                    if (tpr == 0) {
                        l_i = l_i * expf(old_m - m_i) + 1.0f;
                    }
                } else {
                    if (tpr == 0) {
                        l_i += expf(sum - m_i);
                    }
                }
                
                float scale_o = expf(old_m - m_i);
                float scale_p = expf(sum - m_i);
                
                int base_v = nr * BLOCK_D + seg_start;
                for(int j=0; j<SEG_D; ++j) {
                    o_reg[j] = o_reg[j] * scale_o + scale_p * __bfloat162float(sV[base_v+j]);
                }
            }
        }
        __syncthreads();
    }

    // Normalize and write output O
    if (active) {
        float norm = (l_i > 0.0f) ? 1.0f / l_i : 0.0f;
        int out_off = ((b * H + h) * S + q_start + r) * D + seg_start;
        for(int j=0; j<SEG_D; ++j) {
            O[out_off+j] = __float2bfloat16(o_reg[j] * norm);
        }
    }
    
    // Write LogSumExp (only one thread per row computes/writes it)
    if (active && tpr == 0) {
        LSE[(b * H + h) * S + q_start + r] = (l_i > 0.0f) ? (m_i + logf(l_i)) : NEG_INF;
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

    // Dynamic shared memory allocation
    size_t smem_elements = static_cast<size_t>(BLOCK_M * BLOCK_D) + 
                           2ULL * static_cast<size_t>(BLOCK_N * BLOCK_D);
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