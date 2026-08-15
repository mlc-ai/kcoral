#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define TILE_N 64
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

__global__ void mha_causal_kernel(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* O,
    float* LSE_out,
    int S, int B, int H) {

    const int bh_idx = blockIdx.x;
    if (bh_idx >= B * H) return;

    const int tid = threadIdx.x;
    const int row = blockIdx.y * 128 + tid / 4;  // 32 threads per warp, 4 warps = 128 rows
    const int lane = tid % 4;

    if (bh_idx >= (unsigned)B * H || row >= (unsigned)S) return;

    const long long bh_stride = (long long)S * D;
    const __nv_bfloat16* Q_bh = Q + bh_idx * bh_stride;
    const __nv_bfloat16* K_bh = K + bh_idx * bh_stride;
    const __nv_bfloat16* V_bh = V + bh_idx * bh_stride;
    __nv_bfloat16* O_bh = O + bh_idx * bh_stride;
    float* LSE_bh = LSE_out + bh_idx * S;

    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* smem_K = smem;
    __nv_bfloat16* smem_V = smem_K + TILE_N * D;

    const float inv_sqrt_d = 1.0f / sqrtf((float)D);
    
    // Load Q row into registers
    float q_reg[D];
    const __nv_bfloat16* q_row_ptr = Q_bh + (long long)row * D;
    for (int d = 0; d < D; ++d) {
        q_reg[d] = __bfloat162float(q_row_ptr[d]);
    }

    float acc_o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float max_s = -INF;
    float sum_e = 1.0f;

    for (int k_tile = 0; k_tile < S; k_tile += TILE_N) {
        const int k_end = my_min(k_tile + TILE_N, S);
        const int tile_len = k_end - k_tile;

        // Cooperative load K and V tiles
        for (int i = threadIdx.x; i < (tile_len * D); i += blockDim.x) {
            const int r = i / D;
            const int c = i % D;
            smem_K[i] = K_bh[(long long)(k_tile + r) * D + c];
            smem_V[i] = V_bh[(long long)(k_tile + r) * D + c];
        }
        __syncthreads();

        // Pass 1: Find tile maximum (only unmasked keys)
        float tile_max = -INF;
        for (int k_off = 0; k_off < tile_len; ++k_off) {
            if (k_tile + k_off <= row) {
                float s = 0.0f;
                const __nv_bfloat16* k_row = smem_K + k_off * D;
                for (int d = 0; d < D; ++d) {
                    s += q_reg[d] * __bfloat162float(k_row[d]);
                }
                s *= inv_sqrt_d;
                if (s > tile_max) tile_max = s;
            }
        }

        // Skip if entire tile masked out
        if (tile_max > -0.5f * INF) {
            float p_scale = expf(max_s - tile_max);
            
            // Rescale previous accumulators
            for (int l = 0; l < 4; ++l) acc_o[l] *= p_scale;
            sum_e *= p_scale;

            // Pass 2: Compute softmax and accumulate output for lanes
            for (int k_off = 0; k_off < tile_len; ++k_off) {
                float s = 0.0f;
                if (k_tile + k_off <= row) {
                    const __nv_bfloat16* k_row = smem_K + k_off * D;
                    for (int d = 0; d < D; ++d) {
                        s += q_reg[d] * __bfloat162float(k_row[d]);
                    }
                    s *= inv_sqrt_d;
                } else {
                    s = -INF;
                }
                
                float p = expf(s - tile_max);
                for (int l = 0; l < 4 && (lane + l) < D; ++l) {
                    acc_o[l] += p * __bfloat162float(smem_V[k_off * D + lane + l]);
                }
                sum_e += p;
            }
            max_s = tile_max;
        }
        __syncthreads();
    }

    // Final normalization and write results
    float final_inv = 1.0f / sum_e;
    for (int l = 0; l < 4 && (lane + l) < D; ++l) {
        O_bh[(long long)row * D + lane + l] = __float2bfloat16(acc_o[l] * final_inv);
    }
    LSE_bh[row] = max_s + logf(sum_e);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    const int64_t B = Q.size(0);
    const int64_t H = Q.size(1);
    const int64_t S = Q.size(2);
    
    const __nv_bfloat16* d_Q = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* d_K = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* d_V = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* d_O = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* d_LSE = static_cast<float*>(LSE.data_ptr());
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    const int S_int = static_cast<int>(S);
    const int B_int = static_cast<int>(B);
    const int H_int = static_cast<int>(H);
    
    // Grid: x=batch*head, y=query_tiles (ceil(S/128))
    dim3 grid(B_int * H_int, (S_int + 127) / 128);
    dim3 blk(128);
    
    // Shared memory: 2 * TILE_N * D * sizeof(bf16) = 2 * 64 * 128 * 2 = 32KB
    const size_t smem_bytes = 2ULL * TILE_N * D * sizeof(__nv_bfloat16);
    
    mha_causal_kernel<<<grid, blk, smem_bytes, stream>>>(
        d_Q, d_K, d_V, d_O, d_LSE, S_int, B_int, H_int);
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace mha_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);