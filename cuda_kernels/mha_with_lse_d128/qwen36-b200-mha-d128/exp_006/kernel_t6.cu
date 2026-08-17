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
    static_assert(HD == 128, "HD must be 128");

    __shared__ float smem_Q[BM][HD];
    __shared__ float smem_K[BK][HD];
    __shared__ float smem_V[BK][HD];
    __shared__ float smem_O[BM][HD];

    int bh_idx = blockIdx.x;
    int num_bh = B * H;
    int total_tiles = (S + BM - 1) / BM;
    int tile_idx = bh_idx / num_bh;
    int bhtile_id = bh_idx % num_bh;
    
    if (tile_idx >= total_tiles || bhtile_id >= num_bh) return;
    
    int b = bhtile_id / H;
    int h = bhtile_id % H;
    int m_base = tile_idx * BM;

    const __nv_bfloat16* q_base = Q + ((size_t)b * H + h) * S * D;
    const __nv_bfloat16* k_base = K + ((size_t)b * H + h) * S * D;
    const __nv_bfloat16* v_base = V + ((size_t)b * H + h) * S * D;
    __nv_bfloat16* o_base = O + ((size_t)b * H + h) * S * D;
    float* lse_base = LSE + ((size_t)b * H + h) * S;

    int tid = threadIdx.x;
    int col = tid;

    // Phase 1: Initialize output accumulator to zero
    for (int r = 0; r < BM; ++r) {
        smem_O[r][col] = 0.0f;
    }
    __syncthreads();

    // Phase 2: Load Q tile into shared memory
    for (int r = 0; r < BM; ++r) {
        int gm = m_base + r;
        smem_Q[r][col] = (gm < S) ? __bfloat162float(q_base[gm * D + col]) : 0.0f;
    }
    __syncthreads();

    float inv_sqrt_D = rsqrtf((float)D);
    int num_kv_tiles = (S + BK - 1) / BK;

    float row_max = -1e20f;
    float row_sum = 0.0f;
    bool active_row = (tid < BM && (m_base + tid) < S);

    for (int kv_t = 0; kv_t < num_kv_tiles; ++kv_t) {
        int kv_start = kv_t * BK;
        int cur_k_len = min(BK, S - kv_start);

        // Zero-init ALL rows (including padding) then load valid rows
        for (int kr = 0; kr < BK; ++kr) {
            if (kr < cur_k_len) {
                int gidx = (kv_start + kr) * D + col;
                smem_K[kr][col] = __bfloat162float(k_base[gidx]);
                smem_V[kr][col] = __bfloat162float(v_base[gidx]);
            } else {
                smem_K[kr][col] = 0.0f;
                smem_V[kr][col] = 0.0f;
            }
        }
        __syncthreads();

        if (!active_row) continue;
        
        int m = tid;

        // Compute attention scores, only over valid KV range
        float tile_max = -1e20f;
        float s_vals[BK];
        
        #pragma unroll
        for (int j = 0; j < BK; ++j) {
            if (j < cur_k_len) {
                float dot = 0.0f;
                #pragma unroll
                for (int k = 0; k < HD; k += 4) {
                    dot += smem_Q[m][k]     * smem_K[j][k];
                    dot += smem_Q[m][k+1]   * smem_K[j][k+1];
                    dot += smem_Q[m][k+2]   * smem_K[j][k+2];
                    dot += smem_Q[m][k+3]   * smem_K[j][k+3];
                }
                s_vals[j] = dot * inv_sqrt_D;
            } else {
                s_vals[j] = -1e20f;  // exclude padded entries from softmax
            }
            if (s_vals[j] > tile_max) tile_max = s_vals[j];
        }

        // Online softmax state update
        float old_max = row_max;
        row_max = fmaxf(row_max, tile_max);
        float alpha = expf(old_max - row_max);
        row_sum *= alpha;

        // Accumulate P * V into smem_O
        #pragma unroll
        for (int j = 0; j < BK; ++j) {
            if (j >= cur_k_len) continue;
            float p_val = expf(s_vals[j] - row_max);
            row_sum += p_val;
            #pragma unroll
            for (int n = 0; n < HD; n += 4) {
                smem_O[m][n]     = smem_O[m][n]     * alpha + p_val * smem_V[j][n];
                smem_O[m][n+1]   = smem_O[m][n+1]   * alpha + p_val * smem_V[j][n+1];
                smem_O[m][n+2]   = smem_O[m][n+2]   * alpha + p_val * smem_V[j][n+2];
                smem_O[m][n+3]   = smem_O[m][n+3]   * alpha + p_val * smem_V[j][n+3];
            }
        }
    }

    // Final normalization and writeback
    if (active_row) {
        int gm = m_base + tid;
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        #pragma unroll
        for (int n = 0; n < HD; n += 4) {
            o_base[gm * D + n]     = __float2bfloat16(smem_O[tid][n]     * inv_sum);
            o_base[gm * D + n+1]   = __float2bfloat16(smem_O[tid][n+1]   * inv_sum);
            o_base[gm * D + n+2]   = __float2bfloat16(smem_O[tid][n+2]   * inv_sum);
            o_base[gm * D + n+3]   = __float2bfloat16(smem_O[tid][n+3]   * inv_sum);
        }
        lse_base[gm] = row_max + logf(row_sum);
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
    int block_size = BN;
    
    int num_bh = (int)(B * H);
    int num_q_tiles = (int)((S + BM - 1) / BM);
    int num_blocks = num_bh * num_q_tiles;
    
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