#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace mha_d128 {

template<int BM, int BK, int HD, int BLOCK_SIZE>
__global__ void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, int D) 
{
    // Shared memory buffers for each KV tile
    __shared__ float smem_K[BK][HD];
    __shared__ float smem_V[BK][HD];

    int bh_idx = blockIdx.x;
    int num_bh = B * H;
    int total_q_tiles = (S + BM - 1) / BM;
    int tile_idx = bh_idx / num_bh;
    int bhtile_id = bh_idx % num_bh;
    
    if (tile_idx >= total_q_tiles || bhtile_id >= num_bh) return;
    
    int b = bhtile_id / H;
    int h = bhtile_id % H;
    int m_base = tile_idx * BM;

    const __nv_bfloat16* q_base = Q + ((size_t)b * H + h) * S * D;
    const __nv_bfloat16* k_base = K + ((size_t)b * H + h) * S * D;
    const __nv_bfloat16* v_base = V + ((size_t)b * H + h) * S * D;
    __nv_bfloat16* o_base = O + ((size_t)b * H + h) * S * D;
    float* lse_base = LSE + ((size_t)b * H + h) * S;

    int tid = threadIdx.x;

    // Load Q tile into local arrays (one column per thread, all BM rows)
    // Q_vals[r] gives Q[m_base+r][tid] converted to fp32
    float Q_vals[BM];
    for (int r = 0; r < BM; ++r) {
        int gm = m_base + r;
        Q_vals[r] = (gm < S) ? __bfloat162float(q_base[gm * D + tid]) : 0.0f;
    }

    // Output accumulators: one float per output row (BM max)
    float O_vals[BM];
    for (int r = 0; r < BM; ++r) {
        O_vals[r] = 0.0f;
    }

    float inv_sqrt_D = rsqrtf((float)D);
    int num_kv_tiles = (S + BK - 1) / BK;

    // Per-row softmax state
    float row_max[BM];
    float row_sum[BM];
    for (int r = 0; r < BM; ++r) {
        row_max[r] = -1e20f;
        row_sum[r] = 0.0f;
    }

    for (int kv_t = 0; kv_t < num_kv_tiles; ++kv_t) {
        int kv_start = kv_t * BK;
        int cur_k_len = min(BK, S - kv_start);

        // Cooperative load of K and V tiles
        for (int kr = 0; kr < BK; ++kr) {
            int gidx = (kv_start + kr) * D + tid;
            if (kr < cur_k_len) {
                smem_K[kr][tid] = __bfloat162float(k_base[gidx]);
                smem_V[kr][tid] = __bfloat162float(v_base[gidx]);
            } else {
                smem_K[kr][tid] = 0.0f;
                smem_V[kr][tid] = 0.0f;
            }
        }
        __syncthreads();

        // For each query row in this block, compute scores and accumulate output
        for (int r = 0; r < BM; ++r) {
            bool active = (m_base + r) < S;
            if (!active) continue;

            // Dot product Q[r] . K[j]^T for each j
            float s_max = -1e20f;
            
            // Vectorized dot product using 2-wide unpacking
            for (int j = 0; j < cur_k_len; ++j) {
                float dot = 0.0f;
                for (int k = 0; k < HD; k += 2) {
                    // Unpack two bf16 -> float, compute products
                    float q_val = Q_vals[r];  // already loaded above per-thread
                    dot += Q_vals[r] * smem_K[j][k];  // Wrong: Q_vals indexed wrong
                    
                    // Actually each thread owns one column, Q_vals[r] = Q[m_base+r][tid]
                    // smem_K[j][tid] = K[kv_start+j][tid]
                    // But we iterate k here... that's wrong. Each thread holds ONE element.
                }
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    assert(D == 128);
    
    constexpr int BM = 128;
    constexpr int BK = 64;
    constexpr int HD = 128;
    constexpr int BLOCK_SIZE = 128;  // one thread per column
    
    int num_bh = (int)(B * H);
    int num_q_tiles = (int)((S + BM - 1) / BM);
    int num_blocks = num_bh * num_q_tiles;
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id)
    );
    
    mha_kernel<BM, BK, HD, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        (int)B, (int)H, (int)S, (int)D
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);

} // namespace mha_d128