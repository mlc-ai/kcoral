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

template<int BM, int BN, int BK, int HD>
__global__ void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, int D) 
{
    __shared__ float smem_Q[BM][HD];
    __shared__ float smem_K[BK][HD];
    __shared__ float smem_V[BK][BN];
    __shared__ float smem_O[BM][BN];

    int bh_idx = blockIdx.x;
    if (bh_idx >= B * H) return;
    int b = bh_idx / H;
    int h = bh_idx % H;

    const __nv_bfloat16* q_base = Q + ((size_t)b * H + h) * S * D;
    const __nv_bfloat16* k_base = K + ((size_t)b * H + h) * S * D;
    const __nv_bfloat16* v_base = V + ((size_t)b * H + h) * S * D;
    __nv_bfloat16* o_base = O + ((size_t)b * H + h) * S * D;
    float* lse_base = LSE + ((size_t)b * H + h) * S;

    int tid = threadIdx.x;
    int m = tid; 
    bool valid_m = (m < S);

    // Initialize output accumulator tile
    #pragma unroll
    for(int n = 0; n < BN; ++n) smem_O[m][n] = 0.0f;
    __syncthreads();

    // Load Q tile into shared memory
    #pragma unroll
    for(int d = 0; d < HD; ++d) {
        smem_Q[m][d] = valid_m ? __bfloat162float(q_base[m * HD + d]) : 0.0f;
    }
    __syncthreads();

    float inv_sqrt_D = 1.0f / sqrtf((float)HD);
    int num_kv_tiles = (S + BK - 1) / BK;
    
    float row_max = -1e20f;
    float row_sum = 0.0f;

    for(int kv_t = 0; kv_t < num_kv_tiles; ++kv_t) {
        int kv_start = kv_t * BK;
        int rem = S - kv_start;
        int cur_k_len = (BK < rem ? BK : rem);
        bool valid_k_row = (tid < cur_k_len);

        // Load K tile
        int k_row = tid;
        #pragma unroll
        for(int d = 0; d < HD; d+=2) {
            smem_K[k_row][d] = valid_k_row ? __bfloat162float(k_base[(kv_start + k_row) * HD + d]) : 0.0f;
            if(d+1 < HD) smem_K[k_row][d+1] = valid_k_row ? __bfloat162float(k_base[(kv_start + k_row) * HD + d + 1]) : 0.0f;
        }

        // Load V tile (each thread handles one column stride across all K rows)
        int v_col = tid;
        #pragma unroll
        for(int kr = 0; kr < BK; ++kr) {
            smem_V[kr][v_col] = (kr < cur_k_len) ? __bfloat162float(v_base[(kv_start + kr) * HD + v_col]) : 0.0f;
        }
        __syncthreads();

        // Compute attention scores S[m][j] = Q[m] @ K[j]^T / sqrt(D)
        float s_vals[BK] = {0.0f};
        #pragma unroll
        for(int j = 0; j < BK; ++j) {
            float dot = 0.0f;
            #pragma unroll
            for(int k = 0; k < HD; k+=4) {
                dot += smem_Q[m][k]   * smem_K[j][k];
                dot += smem_Q[m][k+1] * smem_K[j][k+1];
                dot += smem_Q[m][k+2] * smem_K[j][k+2];
                dot += smem_Q[m][k+3] * smem_K[j][k+3];
            }
            s_vals[j] = dot * inv_sqrt_D;
        }

        // Find maximum score in this KV tile for the row
        float tile_max = s_vals[0];
        #pragma unroll
        for(int j = 1; j < BK; ++j) tile_max = fmaxf(tile_max, s_vals[j]);

        // Online softmax update
        float old_max = row_max;
        row_max = fmaxf(row_max, tile_max);
        
        float alpha = expf(old_max - row_max);
        row_sum *= alpha;

        // Accumulate P @ V into smem_O
        #pragma unroll
        for(int j = 0; j < BK; ++j) {
            float p_val = expf(s_vals[j] - row_max);
            row_sum += p_val;
            #pragma unroll
            for(int n = 0; n < BN; ++n) {
                smem_O[m][n] = smem_O[m][n] * alpha + p_val * smem_V[j][n];
            }
        }
    }

    // Final normalization
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    
    if(valid_m) {
        #pragma unroll
        for(int n = 0; n < BN; ++n) {
            o_base[m * HD + n] = __float2bfloat16(smem_O[m][n] * inv_sum);
        }
        lse_base[m] = row_max + logf(row_sum);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    if (D != 128) { 
        fprintf(stderr, "Expected D=128, got %lld\n", D); 
        exit(1); 
    }
    
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    
    int num_blocks = (int)(B * H);
    int block_size = BM; 
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id)
    );
    
    mha_kernel<BM, BN, BK, 128><<<num_blocks, block_size, 0, stream>>>(
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