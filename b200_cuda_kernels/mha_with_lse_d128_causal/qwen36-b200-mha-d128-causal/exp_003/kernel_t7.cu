#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define TILE_N 128
#define D 128
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

__device__ __forceinline__ int my_min(int a, int b) { return a < b ? a : b; }

// Each CTA processes all query positions for ONE (batch, head).
// threads = S. Each thread handles one query position sequentially across KV tiles.
__global__ void mha_causal_kernel(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* O,
    float* LSE_out,
    int S, int B, int H, int D_dim) {

    const int bh_idx = blockIdx.x;
    if (bh_idx >= B * H) return;

    const int q_row = threadIdx.x; // Each thread owns one query row
    if (q_row >= S) return;

    const long long bh_stride = (long long)S * D_dim;
    const __nv_bfloat16* Q_bh = Q + bh_idx * bh_stride;
    const __nv_bfloat16* K_bh = K + bh_idx * bh_stride;
    const __nv_bfloat16* V_bh = V + bh_idx * bh_stride;
    __nv_bfloat16* O_bh = O + bh_idx * bh_stride;
    float* LSE_bh = LSE_out + bh_idx * S;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_K = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_V = smem_K + TILE_N * D_dim;

    const float inv_sqrt_d = 1.0f / sqrtf((float)D_dim);
    const __nv_bfloat16* q_row_ptr = Q_bh + (long long)q_row * D_dim;

    // Load Q row into registers
    float q_reg[D_dim];
    for (int d = 0; d < D_dim; ++d) {
        q_reg[d] = __bfloat162float(q_row_ptr[d]);
    }

    float max_s = -INF;
    float sum_e = 1.0f;
    
    // Accumulate output in local arrays
    float acc_o[D_dim];
    for (int d = 0; d < D_dim; ++d) acc_o[d] = 0.0f;

    // Iterate over Key/Value tiles
    for (int k_tile = 0; k_tile < S; k_tile += TILE_N) {
        int k_end = my_min(k_tile + TILE_N, S);
        int tile_len = k_end - k_tile;

        // Cooperative load of K and V tiles into shared memory
        for (int i = threadIdx.x; i < tile_len * D_dim; i += blockDim.x) {
            int r = i / D_dim;
            int c = i % D_dim;
            smem_K[i] = K_bh[(long long)(k_tile + r) * D_dim + c];
            smem_V[i] = V_bh[(long long)(k_tile + r) * D_dim + c];
        }
        __syncthreads();

        // Pass 1: Find tile maximum (causal mask applied)
        float tile_max = -INF;
        float scores[TILE_N];
        
        for (int k_off = 0; k_off < tile_len; ++k_off) {
            if (k_tile + k_off <= q_row) {
                float s = 0.0f;
                const __nv_bfloat16* k_row = smem_K + k_off * D_dim;
                #pragma unroll
                for (int d = 0; d < D_dim; ++d) {
                    s += q_reg[d] * __bfloat162float(k_row[d]);
                }
                s *= inv_sqrt_d;
                scores[k_off] = s;
                tile_max = fmaxf(tile_max, s);
            } else {
                scores[k_off] = -INF;
            }
        }

        // Skip if entire tile masked out
        if (tile_max > -0.5f * INF) {
            float p_scale = expf(max_s - tile_max);
            
            // Rescale previous accumulators
            #pragma unroll
            for (int d = 0; d < D_dim; ++d) acc_o[d] *= p_scale;
            sum_e *= p_scale;

            // Pass 2: Compute softmax weights and accumulate output
            for (int k_off = 0; k_off < tile_len; ++k_off) {
                float p = expf(scores[k_off] - tile_max);
                const __nv_bfloat16* v_row = smem_V + k_off * D_dim;
                #pragma unroll
                for (int d = 0; d < D_dim; ++d) {
                    acc_o[d] += p * __bfloat162float(v_row[d]);
                }
                sum_e += p;
            }
            max_s = tile_max;
        }
        __syncthreads();
    }

    // Final normalization and write results
    float final_inv = 1.0f / sum_e;
    __nv_bfloat16* o_row = O_bh + (long long)q_row * D_dim;
    #pragma unroll
    for (int d = 0; d < D_dim; ++d) {
        o_row[d] = __float2bfloat16(acc_o[d] * final_inv);
    }
    LSE_bh[q_row] = max_s + logf(sum_e);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    const int64_t B = Q.size(0);
    const int64_t H = Q.size(1);
    const int64_t S_val = Q.size(2);
    const int64_t D_val = Q.size(3);
    
    const __nv_bfloat16* d_Q = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* d_K = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* d_V = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* d_O = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* d_LSE = static_cast<float*>(LSE.data_ptr());
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int S_int = static_cast<int>(S_val);
    int B_int = static_cast<int>(B);
    int H_int = static_cast<int>(H);
    int D_int = static_cast<int>(D_val);
    
    dim3 grid(B_int * H_int);
    dim3 blk(my_min(S_int, 1024));
    
    // Shared memory: 2 * TILE_N * D * sizeof(bf16)
    size_t smem_bytes = 2ULL * TILE_N * D_int * sizeof(__nv_bfloat16);
    
    mha_causal_kernel<<<grid, blk, smem_bytes, stream>>>(
        d_Q, d_K, d_V, d_O, d_LSE, S_int, B_int, H_int, D_int);
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace mha_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);