#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_blackwell {

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_M = 64;   // Query rows per block
constexpr int BLOCK_N = 64;   // Key/Value columns per tile
constexpr int BLOCK_D = 128;  // Head dimension
constexpr int THREADS_PER_BLOCK = 128;

// Each block handles BLOCK_M query rows
// THREADS_PER_BLOCK / BLOCK_M = 128/64 = 2 threads per query row
constexpr int THREADS_PER_ROW = THREADS_PER_BLOCK / BLOCK_M;  // = 2
// Each thread handles BLOCK_D / THREADS_PER_ROW = 64 elements along D
constexpr int ELEMENTS_PER_THREAD = BLOCK_D / THREADS_PER_ROW;  // = 64

__global__ void mha_causal_d128_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
          __nv_bfloat16* __restrict__ O,
          float*         __restrict__ LSE,
    int B, int H, int S, int D,
    float inv_sqrt_d)
{
    // Shared memory: K tile + V tile
    extern __shared__ char smem_char[];
    
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_char);
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_char + BLOCK_N * BLOCK_D * sizeof(__nv_bfloat16));
    
    // Strides
    const int stride_svd = S * D;
    const int stride_hd = H * S * D;
    const int stride_hsd = H * S;
    
    // Decode block index: blockIdx.x = batch*H + head, blockIdx.y = q_tile_index
    const int bh_idx = blockIdx.x;
    const int b = bh_idx / H;
    const int h = bh_idx % H;
    const int q_tile_start = blockIdx.y * BLOCK_M;
    
    const int tid = threadIdx.x;
    const int row_in_tile = tid / THREADS_PER_ROW;     // 0..63
    const int lane_in_row = tid % THREADS_PER_ROW;      // 0 or 1
    
    const int q_row_global = q_tile_start + row_in_tile;
    const bool valid_q = q_row_global < S;
    
    // Base pointers for this (batch, head)
    const __nv_bfloat16* Q_bh = Q + b * stride_hd + h * stride_svd;
    const __nv_bfloat16* K_bh = K + b * stride_hd + h * stride_svd;
    const __nv_bfloat16* V_bh = V + b * stride_hd + h * stride_svd;
    __nv_bfloat16* O_bh = O + b * stride_hd + h * stride_svd;
    float* LSE_bh = LSE + b * stride_hsd + h * S;
    
    // Load our portion of the query row into registers (expanded to fp32 * inv_sqrt_d)
    float q_regs[ELEMENTS_PER_THREAD];
    float acc_o[ELEMENTS_PER_THREAD];
    
    if (valid_q) {
        const __nv_bfloat16* q_row_ptr = Q_bh + q_row_global * D;
        const int d_base = lane_in_row * ELEMENTS_PER_THREAD;
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            q_regs[i] = __bfloat162float(q_row_ptr[d_base + i]) * inv_sqrt_d;
        }
    } else {
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            q_regs[i] = 0.0f;
        }
    }
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        acc_o[i] = 0.0f;
    }
    
    // Online softmax state
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    
    const int num_n_tiles = (S + BLOCK_N - 1) / BLOCK_N;
    
    for (int tn = 0; tn < num_n_tiles; ++tn) {
        int k_tile_start = tn * BLOCK_N;
        bool last_tile = (tn == num_n_tiles - 1);
        int n_act = last_tile ? (S - k_tile_start) : BLOCK_N;
        
        // ===== Cooperative K tile load =====
        for (int idx = tid; idx < BLOCK_N * BLOCK_D; idx += THREADS_PER_BLOCK) {
            int n_idx = idx / BLOCK_D + k_tile_start;
            int d_idx = idx % BLOCK_D;
            if (n_idx < S) {
                smem_k[idx] = K_bh[n_idx * D + d_idx];
            } else {
                smem_k[idx] = __float2bfloat16(0.0f);
            }
        }
        
        // ===== Cooperative V tile load =====
        for (int idx = tid; idx < BLOCK_N * BLOCK_D; idx += THREADS_PER_BLOCK) {
            int n_idx = idx / BLOCK_D + k_tile_start;
            int d_idx = idx % BLOCK_D;
            if (n_idx < S) {
                smem_v[idx] = V_bh[n_idx * D + d_idx];
            } else {
                smem_v[idx] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
        
        // ===== Compute Q @ K^T scores for this row =====
        // Each thread computes partial dot product over its ELEMENTS_PER_THREAD of D
        // Then pair-reduce within the row group (2 threads)
        float attn_scores[BLOCK_N];
        
        const unsigned full_mask = 0xFFFFFFFF;
        const int peer_tid = tid ^ 1;  // xor lowest bit to get partner thread
        
        for (int kc = 0; kc < BLOCK_N; ++kc) {
            float partial = 0.0f;
            const __nv_bfloat16* k_col = smem_k + kc * BLOCK_D;
            const int d_base = lane_in_row * ELEMENTS_PER_THREAD;
            
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; i += 4) {
                partial += q_regs[i+0] * __bfloat162float(k_col[d_base + i+0]);
                partial += q_regs[i+1] * __bfloat162float(k_col[d_base + i+1]);
                partial += q_regs[i+2] * __bfloat162float(k_col[d_base + i+2]);
                partial += q_regs[i+3] * __bfloat162float(k_col[d_base + i+3]);
            }
            
            // Reduce with partner thread using shuffle
            float other_partial = __shfl_down_sync(full_mask, partial, 1);
            float score = partial + other_partial;
            
            // Apply causal mask
            int k_col_global = k_tile_start + kc;
            if (!valid_q || k_col_global >= q_row_global) {
                attn_scores[kc] = -FLT_MAX;
            } else {
                attn_scores[kc] = score;
            }
        }
        
        // ===== Row-wise online softmax update =====
        float tile_max = -FLT_MAX;
        for (int kc = 0; kc < BLOCK_N; ++kc) {
            if (attn_scores[kc] > tile_max) tile_max = attn_scores[kc];
        }
        
        // Check if any valid (unmasked) keys exist in this tile for this row
        bool has_valid = false;
        for (int kc = 0; kc < n_act && !has_valid; ++kc) {
            if (k_tile_start + kc < q_row_global && valid_q) has_valid = true;
        }
        if (!has_valid) {
            __syncthreads();
            continue;
        }
        
        float old_max = row_max;
        if (tile_max > row_max) {
            row_max = tile_max;
            if (old_max > -FLT_MAX * 0.5f && row_sum > 0.0f) {
                float scale = expf(old_max - row_max);
                row_sum *= scale;
                for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    acc_o[i] *= scale;
                }
            }
        }
        
        // Compute probs and accumulate
        for (int kc = 0; kc < n_act; ++kc) {
            int k_col_global = k_tile_start + kc;
            if (k_col_global < q_row_global && valid_q) {
                float p = expf(attn_scores[kc] - row_max);
                row_sum += p;
                
                // Accumulate p * V[kc][:]
                const __nv_bfloat16* v_col = smem_v + kc * BLOCK_D;
                const int d_base = lane_in_row * ELEMENTS_PER_THREAD;
                for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    acc_o[i] += p * __bfloat162float(v_col[d_base + i]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // ===== Write-back =====
    if (valid_q) {
        float lse_val;
        if (row_sum > 0.0f) {
            float norm = 1.0f / row_sum;
            int d_base = lane_in_row * ELEMENTS_PER_THREAD;
            __nv_bfloat16* o_row = O_bh + q_row_global * D;
            for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                o_row[d_base + i] = __float2bfloat16(acc_o[i] * norm);
            }
            lse_val = row_max + logf(row_sum);
        } else {
            int d_base = lane_in_row * ELEMENTS_PER_THREAD;
            __nv_bfloat16* o_row = O_bh + q_row_global * D;
            for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                o_row[d_base + i] = __float2bfloat16(0.0f);
            }
            lse_val = -FLT_MAX;
        }
        LSE_bh[q_row_global] = lse_val;
    } else {
        int d_base = lane_in_row * ELEMENTS_PER_THREAD;
        __nv_bfloat16* o_row = O_bh + q_row_global * D;
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            o_row[d_base + i] = __float2bfloat16(0.0f);
        }
        LSE_bh[q_row_global] = 0.0f;
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());
    
    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(D));
    
    int64_t num_bh = B * H;
    int64_t num_q_tiles = (S + BLOCK_M - 1) / BLOCK_M;
    
    dim3 grid(static_cast<unsigned int>(num_bh), static_cast<unsigned int>(num_q_tiles));
    dim3 block(THREADS_PER_BLOCK);
    
    // Shared memory: K tile (BLOCK_N * BLOCK_D bf16) + V tile (BLOCK_N * BLOCK_D bf16)
    size_t smem_size = 2 * BLOCK_N * BLOCK_D * sizeof(__nv_bfloat16);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_causal_d128_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S),
        static_cast<int>(D), inv_sqrt_d);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_blackwell::run);

}  // namespace mha_blackwell