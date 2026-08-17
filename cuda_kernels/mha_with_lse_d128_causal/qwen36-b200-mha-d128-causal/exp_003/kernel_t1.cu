#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <algorithm>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define TILE_S 128
#define HEAD_DIM 128
#define INF 1e20f

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_impl {

__global__ void mha_causal_kernel(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* O,
    float* LSE,
    int S,
    int B,
    int H) {
    
    int bh_idx = blockIdx.x;
    if (bh_idx >= B * H) return;
    
    int d_idx = threadIdx.x; // Matches HEAD_DIM=128 exactly
    
    size_t bh_stride = (size_t)S * HEAD_DIM;
    const __nv_bfloat16* Q_bh = Q + bh_idx * bh_stride;
    const __nv_bfloat16* K_bh = K + bh_idx * bh_stride;
    const __nv_bfloat16* V_bh = V + bh_idx * bh_stride;
    __nv_bfloat16* O_bh = O + bh_idx * bh_stride;
    float* LSE_bh = LSE + bh_idx * S;

    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* smem_K = smem;
    __nv_bfloat16* smem_V = smem + TILE_S * HEAD_DIM;

    float inv_sqrt_d = 1.0f / sqrtf((float)HEAD_DIM);

    // Process query rows in chunks of blockDim.x (128)
    for (int q_start = 0; q_start < S; q_start += blockDim.x) {
        int q_idx = q_start + d_idx;
        if (q_idx >= S) break;

        float acc_o = 0.0f;
        float max_s = -INF;
        float sum_e = 1.0f;

        // Iterate over Key/Value tiles
        for (int k_start = 0; k_start < S; k_start += TILE_S) {
            int k_end = std::min(k_start + TILE_S, S);
            int tile_len = k_end - k_start;

            // Cooperative load of K and V tiles into shared memory
            for (int i = threadIdx.x; i < tile_len * HEAD_DIM; i += blockDim.x) {
                int r = i / HEAD_DIM;
                int c = i % HEAD_DIM;
                smem_K[i] = K_bh[(k_start + r) * HEAD_DIM + c];
                smem_V[i] = V_bh[(k_start + r) * HEAD_DIM + c];
            }
            __syncthreads();

            // Pass 1: Find maximum score in this tile (with causal mask applied)
            float tile_max = -INF;
            const __nv_bfloat16* q_row = Q_bh + q_idx * HEAD_DIM;
            for (int k_off = 0; k_off < tile_len; ++k_off) {
                if (k_start + k_off <= q_idx) {
                    float s = 0.0f;
                    const __nv_bfloat16* k_row = smem_K + k_off * HEAD_DIM;
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        s += __bfloat162float(q_row[d]) * __bfloat162float(k_row[d]);
                    }
                    s *= inv_sqrt_d;
                    if (s > tile_max) tile_max = s;
                }
            }

            // Skip accumulation if entire tile is masked out
            if (tile_max > -0.5f * INF) {
                // Scale previous state by exp(old_max - new_max)
                float p_scale = expf(max_s - tile_max);
                acc_o *= p_scale;
                sum_e *= p_scale;

                // Pass 2: Compute softmax weights and accumulate output
                const __nv_bfloat16* v_tile = smem_V;
                for (int k_off = 0; k_off < tile_len; ++k_off) {
                    float s = 0.0f;
                    const __nv_bfloat16* k_row = smem_K + k_off * HEAD_DIM;
                    
                    if (k_start + k_off <= q_idx) {
                        #pragma unroll
                        for (int d = 0; d < HEAD_DIM; ++d) {
                            s += __bfloat162float(q_row[d]) * __bfloat162float(k_row[d]);
                        }
                        s *= inv_sqrt_d;
                    } else {
                        s = -INF;
                    }
                    
                    float p = expf(s - tile_max);
                    acc_o += p * __bfloat162float(v_tile[k_off * HEAD_DIM + d_idx]);
                    sum_e += p;
                }
                max_s = tile_max;
            }
            __syncthreads();
        }

        // Final normalization and LSE computation
        float final_inv_sum = 1.0f / sum_e;
        O_bh[q_idx * HEAD_DIM + d_idx] = __float2bfloat16(acc_o * final_inv_sum);
        LSE_bh[q_idx] = max_s + logf(sum_e);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    // D is fixed at 128 by problem definition, verified by size(3) if needed
    
    const __nv_bfloat16* d_Q = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* d_K = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* d_V = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* d_O = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* d_LSE = static_cast<float*>(LSE.data_ptr());
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
        
    int blocks = B * H;
    int threads = 128; // Exactly covers HEAD_DIM=128
    size_t smem_bytes = 2ULL * TILE_S * HEAD_DIM * sizeof(__nv_bfloat16);
    
    mha_causal_kernel<<<blocks, threads, smem_bytes, stream>>>(
        d_Q, d_K, d_V, d_O, d_LSE, static_cast<int>(S), static_cast<int>(B), static_cast<int>(H));
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace mha_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);