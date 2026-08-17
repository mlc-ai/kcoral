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

template<int BM, int BN, int BK, int BLOCK_SIZE>
__global__ void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, int D) 
{
    __shared__ float smem_Q[BM][BN];
    __shared__ float smem_K[BK][BN];
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

    // Initialize output accumulator tile via strided load
    int o_row = tid % BM;
    int o_col_start = tid / BM;
    int o_col_stride = BLOCK_SIZE / BM;
    for (int c = o_col_start; c < BN; c += o_col_stride) {
        smem_O[o_row][c] = 0.0f;
    }
    __syncthreads();

    // Load Q tile into shared memory via strided cooperative load
    int q_row = tid % BM;
    int q_col_start = tid / BM;
    int q_col_stride = BLOCK_SIZE / BM;
    for (int c = q_col_start; c < BN; c += q_col_stride) {
        smem_Q[q_row][c] = (q_row < S) ? __bfloat162float(q_base[q_row * D + c]) : 0.0f;
    }
    __syncthreads();

    // Only the canonical thread for each query row performs compute
    int m = tid % BM;
    bool valid_m = (tid < BM) && (m < S);

    float inv_sqrt_D = 1.0f / sqrtf((float)D);
    int num_kv_tiles = (S + BK - 1) / BK;

    float row_max = -1e20f;
    float row_sum = 0.0f;

    // Temporary registers for scores (BK=16, small footprint)
    float s_vals[BK];

    for (int kv_t = 0; kv_t < num_kv_tiles; ++kv_t) {
        int kv_start = kv_t * BK;
        int rem = S - kv_start;
        int cur_k_len = (BK < rem ? BK : rem);

        // Load K tile via strided cooperative load
        int k_row = tid % BK;
        int k_col_start = tid / BK;
        int k_col_stride = BLOCK_SIZE / BK;
        for (int c = k_col_start; c < BN; c += k_col_stride) {
            smem_K[k_row][c] = (k_row < cur_k_len) ? __bfloat162float(k_base[(kv_start + k_row) * D + c]) : 0.0f;
        }

        // Load V tile via strided cooperative load
        int v_row = tid % BK;
        int v_col_start = tid / BK;
        int v_col_stride = BLOCK_SIZE / BK;
        for (int c = v_col_start; c < BN; c += v_col_stride) {
            smem_V[v_row][c] = (v_row < cur_k_len) ? __bfloat162float(v_base[(kv_start + v_row) * D + c]) : 0.0f;
        }
        __syncthreads();

        // Compute attention scores S[m][j] = Q[m] @ K[j]^T / sqrt(D)
        float tile_max = -1e20f;
        for (int j = 0; j < BK; ++j) {
            float dot = 0.0f;
            for (int k = 0; k < BN; k += 4) {
                dot += smem_Q[m][k]     * smem_K[j][k];
                dot += smem_Q[m][k + 1] * smem_K[j][k + 1];
                dot += smem_Q[m][k + 2] * smem_K[j][k + 2];
                dot += smem_Q[m][k + 3] * smem_K[j][k + 3];
            }
            s_vals[j] = dot * inv_sqrt_D;
            if (s_vals[j] > tile_max) tile_max = s_vals[j];
        }

        // Online softmax state update
        float old_max = row_max;
        row_max = (tile_max > row_max) ? tile_max : row_max;

        float alpha = expf(old_max - row_max);
        row_sum *= alpha;

        // Accumulate P @ V into smem_O
        for (int j = 0; j < BK; ++j) {
            float p_val = expf(s_vals[j] - row_max);
            row_sum += p_val;
            for (int n = 0; n < BN; n += 4) {
                smem_O[m][n]     = smem_O[m][n]     * alpha + p_val * smem_V[j][n];
                smem_O[m][n + 1] = smem_O[m][n + 1] * alpha + p_val * smem_V[j][n + 1];
                smem_O[m][n + 2] = smem_O[m][n + 2] * alpha + p_val * smem_V[j][n + 2];
                smem_O[m][n + 3] = smem_O[m][n + 3] * alpha + p_val * smem_V[j][n + 3];
            }
        }
    }

    // Final normalization and writeback
    if (valid_m) {
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        for (int n = 0; n < BN; n += 4) {
            o_base[m * D + n]     = __float2bfloat16(smem_O[m][n]     * inv_sum);
            o_base[m * D + n + 1] = __float2bfloat16(smem_O[m][n + 1] * inv_sum);
            o_base[m * D + n + 2] = __float2bfloat16(smem_O[m][n + 2] * inv_sum);
            o_base[m * D + n + 3] = __float2bfloat16(smem_O[m][n + 3] * inv_sum);
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
    
    assert(D == 128);
    
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int BLOCK_SIZE = 128;
    
    int num_blocks = (int)(B * H);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id)
    );
    
    mha_kernel<BM, BN, BK, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
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