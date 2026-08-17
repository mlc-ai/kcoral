#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
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

// Constants
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_M = 16;   // Query tile size (per warp-group)
constexpr int BLOCK_N = 64;   // Key/Value tile size
constexpr int BLOCK_D = 128;  // Head dimension
constexpr int THREADS_PER_BLOCK = 128;

// Share 128 threads across 16 query rows -> 8 threads per query row
constexpr int THREADS_PER_ROW = THREADS_PER_BLOCK / BLOCK_M;  // = 8

__device__ inline float bf16_to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

__device__ inline __nv_bfloat16 float_to_bf16(float val) {
    return __float2bfloat16(val);
}

__global__ void mha_causal_d128_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
          __nv_bfloat16* __restrict__ O,
          float*         __restrict__ LSE,
    int B, int H, int S, int D,
    float inv_sqrt_d)
{
    // Shared memory buffers for K and V tiles
    // Using extern shared memory for flexibility
    extern __shared__ char smem_char[];
    
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_char);
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_char + BLOCK_M * BLOCK_D * sizeof(__nv_bfloat16));
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_char + BLOCK_N * BLOCK_D * sizeof(__nv_bfloat16));
    
    // Strides
    const int stride_svd = S * D;  // stride for (S, D) plane
    const int stride_hd = H * S * D;  // stride for one batch
    const int stride_hsd = H * S;  // stride for LSE output
    
    // Each block handles: one (batch, head) pair, one query tile of BLOCK_M rows
    // blockIdx.x encodes (batch, head), blockIdx.y encodes query tile
    const int bh_idx = blockIdx.x;
    const int b = bh_idx / H;
    const int h = bh_idx % H;
    const int q_tile_start = blockIdx.y * BLOCK_M;
    
    // Per-thread local state for online softmax
    // Each thread handles one query row (BLOCK_M threads total via cycling)
    const int tid = threadIdx.x;
    const int row_in_tile = tid / THREADS_PER_ROW;  // 0..15
    const int lane_in_row = tid % THREADS_PER_ROW;   // 0..7
    
    // Global query row index for this thread
    const int q_row_global = q_tile_start + row_in_tile;
    const bool valid_q = q_row_global < S;
    
    // Base pointers for this (batch, head)
    const __nv_bfloat16* Q_bh = Q + b * stride_hd + h * stride_svd;
    const __nv_bfloat16* K_bh = K + b * stride_hd + h * stride_svd;
    const __nv_bfloat16* V_bh = V + b * stride_hd + h * stride_svd;
    __nv_bfloat16* O_bh = O + b * stride_hd + h * stride_svd;
    float* LSE_bh = LSE + b * stride_hsd + h * S;
    
    // Load our query row(s) into registers from global memory
    // Each thread loads parts of its assigned query row
    // For D=128, each of 8 lanes in a row handles D/8 = 16 bf16 elements (=32 floats)
    constexpr int ELEMENTS_PER_LANE = BLOCK_D / THREADS_PER_ROW;  // 128/8 = 16
    
    float q_regs[ELEMENTS_PER_LANE];  // per thread's portion of q row, expanded to float
    float acc_o[ELEMENTS_PER_LANE];   // accumulated output
    
    if (valid_q) {
        const __nv_bfloat16* q_row_ptr = Q_bh + q_row_global * D;
        for (int i = 0; i < ELEMENTS_PER_LANE; ++i) {
            int d_idx = lane_in_row * ELEMENTS_PER_LANE + i;
            q_regs[i] = bf16_to_float(q_row_ptr[d_idx]) * inv_sqrt_d;
        }
        for (int i = 0; i < ELEMENTS_PER_LANE; ++i) {
            acc_o[i] = 0.0f;
        }
    } else {
        for (int i = 0; i < ELEMENTS_PER_LANE; ++i) {
            q_regs[i] = 0.0f;
            acc_o[i] = 0.0f;
        }
    }
    
    // Online softmax state
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    
    // Iterate over key/value tiles
    const int num_n_tiles = (S + BLOCK_N - 1) / BLOCK_N;
    
    for (int tn = 0; tn < num_n_tiles; ++tn) {
        int k_tile_start = tn * BLOCK_N;
        bool last_tile = (tn == num_n_tiles - 1);
        int n_tiles_actually = last_tile ? (S - k_tile_start) : BLOCK_N;
        
        // ========== Load K tile into shared memory ==========
        // All threads participate in loading K tile
        // K layout: [n][d], loaded cooperatively
        for (int idx = tid; idx < BLOCK_N * BLOCK_D; idx += THREADS_PER_BLOCK) {
            int n_idx = idx / BLOCK_D + k_tile_start;
            int d_idx = idx % BLOCK_D;
            if (n_idx < S) {
                smem_k[idx] = K_bh[n_idx * D + d_idx];
            } else {
                smem_k[idx] = __float2bfloat16(0.0f);
            }
        }
        
        // ========== Load V tile into shared memory ==========
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
        
        // ========== Compute Q @ K^T partial scores ==========
        // Each thread computes one (q_row, k_col) dot product contribution
        // Actually, for online softmax we need to compute the full QK^T tile
        // and do row-wise softmax on it
        
        // For efficiency, each thread accumulates its q row against all k cols in the tile
        // Thread's q row is in q_regs (ELEMENTS_PERLane floats)
        // We need to compute: s[q_row][k_col] = sum over d of q[d]*k[k_col][d]
        
        float attn_scores[BLOCK_N];  // scores for this row against all k_cols in tile
        #pragma unroll
        for (int kc = 0; kc < BLOCK_N; ++kc) {
            float s = 0.0f;
            // Dot product over D=128, using q_regs which holds ELEMENTS_PER_LANE
            // We need contributions from all lanes in the row
            // But actually q_regs only has part of the row... 
            // Need to collect from all 8 lanes
            
            // Redesign: each thread only knows its part. 
            // Do warp-level reduction for the dot product
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_LANE; ++i) {
                int d_idx = lane_in_row * ELEMENTS_PER_LANE + i;
                s += q_regs[i] * bf16_to_float(smem_k[kc * BLOCK_D + d_idx]);
            }
            
            // Warp-level reduce among lanes_in_row (lane 0..7 share row_in_tile)
            // These are threads with same row_in_tile but different lane_in_row
            // They form thread IDs: row_in_tile*8 + lane 0..7
            // Use shuffle to reduce
            for (int offset = 4; offset > 0; offset >>= 1) {
                float other = __shfl_down_sync(0xFF, s, offset);
                if (lane_in_row >= offset) {  // wrong sync mask, fix below
                    // Actually need to sync within the group of 8
                }
            }
            
            // Better approach: use shfl XOR or manual gathering
            // Gather all 8 partial sums from the row group
            float gathered[8];
            #pragma unroll
            for (int l = 0; l < THREADS_PER_ROW; ++l) {
                gathered[l] = __shfl_sync(0xFF, s, row_in_tile * THREADS_PER_ROW + l);
            }
            float total = 0.0f;
            #pragma unroll
            for (int l = 0; l < THREADS_PER_ROW; ++l) {
                total += gathered[l];
            }
            
            // Apply causal mask: zero out if k_col >= q_row (for future positions)
            int k_col_global = k_tile_start + kc;
            if (k_col_global >= q_row_global || !valid_q) {
                attn_scores[kc] = -FLT_MAX;
            } else {
                attn_scores[kc] = total;
            }
        }
        
        // ========== Row-wise softmax on this tile ==========
        // Find max in this tile's scores
        float tile_max = -FLT_MAX;
        #pragma unroll
        for (int kc = 0; kc < BLOCK_N; ++kc) {
            if (attn_scores[kc] > tile_max) {
                tile_max = attn_scores[kc];
            }
        }
        
        // Handle all masked case
        bool all_masked = true;
        #pragma unroll
        for (int kc = 0; kc < n_tiles_actually; ++kc) {
            if (k_tile_start + kc < q_row_global && valid_q) {
                all_masked = false;
                break;
            }
        }
        
        if (all_masked) {
            continue;  // No valid keys for this query row in this tile
        }
        
        // Online softmax update
        float old_max = row_max;
        if (tile_max > row_max) {
            row_max = tile_max;
            // Rescale previous sum
            if (old_max > -FLT_MAX && row_sum > 0.0f) {
                float scale = expf(old_max - row_max);
                row_sum *= scale;
                // Rescale accumulated output
                for (int i = 0; i < ELEMENTS_PER_LANE; ++i) {
                    acc_o[i] *= scale;
                }
            }
        }
        
        // Compute exp(score - row_max) and accumulate sum
        // Also compute weighted V contribution
        #pragma unroll
        for (int kc = 0; kc < n_tiles_actually; ++kc) {
            int k_col_global = k_tile_start + kc;
            if (k_col_global < q_row_global && valid_q) {
                float p = expf(attn_scores[kc] - row_max);
                row_sum += p;
                
                // Accumulate p * V[kc][:] into acc_o[:]
                // Gather V column from all lanes in row group
                #pragma unroll
                for (int i = 0; i < ELEMENTS_PER_LANE; ++i) {
                    int d_idx = lane_in_row * ELEMENTS_PER_LANE + i;
                    acc_o[i] += p * bf16_to_float(smem_v[kc * BLOCK_D + d_idx]);
                }
            }
        }
        
        __syncthreads();  // Needed before next K/V tile load
    }
    
    // ========== Final write-back ==========
    if (valid_q) {
        // Normalize by row_sum (LSE = row_max + log(row_sum))
        float lse = row_max;
        if (row_sum > 0.0f) {
            float norm = 1.0f / row_sum;
            for (int i = 0; i < ELEMENTS_PER_LANE; ++i) {
                float val = acc_o[i] * norm;
                const __nv_bfloat16* q_row_ptr = Q_bh + q_row_global * D;
                int d_idx = lane_in_row * ELEMENTS_PER_LANE + i;
                O_bh[q_row_global * D + d_idx] = float_to_bf16(val);
            }
            lse += logf(row_sum);
        } else {
            // No valid attention (all masked)
            for (int i = 0; i < ELEMENTS_PER_LANE; ++i) {
                int d_idx = lane_in_row * ELEMENTS_PER_LANE + i;
                O_bh[q_row_global * D + d_idx] = __float2bfloat16(0.0f);
            }
        }
        LSE_bh[q_row_global] = lse;
    } else {
        for (int i = 0; i < ELEMENTS_PER_LANE; ++i) {
            int d_idx = lane_in_row * ELEMENTS_PER_LANE + i;
            O_bh[q_row_global * D + d_idx] = __float2bfloat16(0.0f);
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
    
    // Grid: one block per (batch, head, query_tile)
    int64_t num_bh = B * H;
    int64_t num_q_tiles = (S + BLOCK_M - 1) / BLOCK_M;
    
    dim3 grid(num_bh, num_q_tiles);
    dim3 block(THREADS_PER_BLOCK);
    
    // Shared memory: smem_q + smem_k + smem_v
    // smem_q: BLOCK_M * BLOCK_D bf16
    // smem_k: BLOCK_N * BLOCK_D bf16
    // smem_v: BLOCK_N * BLOCK_D bf16
    size_t smem_size = (BLOCK_M * BLOCK_D + 2 * BLOCK_N * BLOCK_D) * sizeof(__nv_bfloat16);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_causal_d128_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S), static_cast<int>(D),
        inv_sqrt_d);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_blackwell::run);

}  // namespace mha_blackwell