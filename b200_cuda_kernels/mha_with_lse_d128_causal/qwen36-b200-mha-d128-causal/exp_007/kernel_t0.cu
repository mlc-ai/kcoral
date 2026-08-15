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

namespace tvm_ffi_mha_blackwell {

constexpr int BLOCK_M = 128;   // Query rows per CTA
constexpr int BLOCK_S = 64;    // Key/Value cols per step
constexpr int HEAD_DIM = 128;  // Fixed head dimension D

extern "C" __global__ void causal_mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16*         __restrict__ O,
    float*                 __restrict__ LSE,
    const int              S,
    const int              stride_S_Q,
    const int              stride_D_Q,
    const int              stride_S_K,
    const int              stride_D_K,
    const int              stride_S_V,
    const int              stride_D_V,
    const int              stride_S_O,
    const int              stride_D_O,
    const int              stride_S_LSE
) {
    // Each thread block handles one (batch, head) combination
    // All threads cooperate to process the full sequence length
    
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;  // Expected: 128
    
    // Decompose block index: grid is linear over B*H
    const int batch_id = bid / H;
    const int head_id  = bid % H;
    
    // Strides from TVM-FFI tensor metadata
    constexpr int STRIDE_S_Q = stride_S_Q;
    constexpr int STRIDE_D_Q = stride_D_Q;
    constexpr int STRIDE_S_K = stride_S_K;
    constexpr int STRIDE_D_K = stride_D_K;
    constexpr int STRIDE_S_V = stride_S_V;
    constexpr int STRIDE_S_O = stride_S_O;
    constexpr int STRIDE_D_O = stride_D_O;
    constexpr int STRIDE_S_LSE = stride_S_LSE;
    
    // Pointers for this batch/head
    const __nv_bfloat16* Q_base = Q + batch_id * H * STRIDE_S_Q + head_id * STRIDE_S_Q;
    const __nv_bfloat16* K_base = K + batch_id * H * STRIDE_S_K + head_id * STRIDE_S_K;
    const __nv_bfloat16* V_base = V + batch_id * H * STRIDE_S_V + head_id * STRIDE_S_V;
    __nv_bfloat16*       O_base = O + batch_id * H * STRIDE_S_O + head_id * STRIDE_S_O;
    float*               LSE_base = LSE + batch_id * H * STRIDE_S_LSE + head_id * STRIDE_S_LSE;
    
    // Shared memory: K tile [BLOCK_S][HEAD_DIM] padded, V tile [BLOCK_S][HEAD_DIM] padded
    extern __shared__ char smem[];
    __nv_bfloat16* smem_K = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* smem_V = reinterpret_cast<__nv_bfloat16*>(smem) + BLOCK_S * HEAD_DIM;
    
    // Number of query rows this thread handles
    const int bm_per_thread = (BLOCK_M + nthreads - 1) / nthreads;
    const int num_tiles = (S + BLOCK_S - 1) / BLOCK_S;
    
    // Scale factor for softmax: 1/sqrt(D)
    const float scale = rsqrtf(static_cast<float>(HEAD_DIM));
    
    // Per-query-row accumulators in registers
    float out[HEAD_DIM];       // Accumulated output
    float row_lse[BLOCK_M];    // Running LSE per query row
    
    // Initialize accumulators
    for (int r = 0; r < HEAD_DIM; ++r) {
        out[r] = 0.0f;
    }
    for (int r = 0; r < BLOCK_M; ++r) {
        row_lse[r] = -1e20f;   // Very negative to represent "-infinity" for log-sum-exp init
    }
    bool has_valid_row[BLOCK_M];  // Track whether each row has seen valid (non-masked) keys
    for (int r = 0; r < BLOCK_M; ++r) {
        has_valid_row[r] = false;
    }
    
    // ---- Load Q rows into registers ----
    float q_regs[bm_per_thread][HEAD_DIM];
    #pragma unroll
    for (int r = 0; r < bm_per_thread; ++r) {
        int qi = tid * bm_per_thread + r;
        if (qi < S) {
            const __nv_bfloat16* q_ptr = Q_base + qi * STRIDE_D_Q;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d += 8) {
                // Load 4 bf16 elements at a time as uint2, then expand to float
                uint2 u = reinterpret_cast<const uint2*>(q_ptr)[d / 2];
                float f0 = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&u.x)[0]);
                float f1 = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&u.x)[1]);
                float f2 = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&u.y)[0]);
                float f3 = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&u.y)[1]);
                q_regs[r][d]     = f0;
                q_regs[r][d + 1] = f1;
                q_regs[r][d + 2] = f2;
                q_regs[r][d + 3] = f3;
                if (d + 4 < HEAD_DIM) {
                    uint2 u1 = reinterpret_cast<const uint2*>(q_ptr)[d / 2 + 2];
                    q_regs[r][d + 4] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&u1.x)[0]);
                    q_regs[r][d + 5] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&u1.x)[1]);
                    q_regs[r][d + 6] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&u1.y)[0]);
                    q_regs[r][d + 7] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&u1.y)[1]);
                }
            }
        }
    }
    
    // ---- Main loop over KV sequence ----
    #pragma unroll 2
    for (int tile = 0; tile < num_tiles; ++tile) {
        int kv_start = tile * BLOCK_S;
        int kv_count = min(BLOCK_S, S - kv_start);
        
        // ---- Load K tile into shared memory ----
        for (int idx = tid; idx < BLOCK_S * HEAD_DIM; idx += nthreads) {
            int s_idx = idx / HEAD_DIM;
            int d_idx = idx % HEAD_DIM;
            int global_s = kv_start + s_idx;
            if (global_s < S) {
                const __nv_bfloat16* k_ptr = K_base + global_s * STRIDE_D_K + d_idx;
                smem_K[idx] = *k_ptr;
            } else {
                smem_K[idx] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
        
        // ---- Load V tile into shared memory ----
        for (int idx = tid; idx < BLOCK_S * HEAD_DIM; idx += nthreads) {
            int s_idx = idx / HEAD_DIM;
            int d_idx = idx % HEAD_DIM;
            int global_s = kv_start + s_idx;
            if (global_s < S) {
                const __nv_bfloat16* v_ptr = V_base + global_s * STRIDE_D_V + d_idx;
                smem_V[idx] = *v_ptr;
            } else {
                smem_V[idx] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
        
        // ---- Compute Q @ K^T tile (attention scores) with causal mask ----
        // Score[query_row][kv_col] = sum_d(Q[query_row, d] * K[kv_col, d]) * scale
        // Apply causal mask: score = 0 if kv_col > query_pos
        
        float scores[bm_per_thread][BLOCK_S];
        #pragma unroll
        for (int r = 0; r < bm_per_thread; ++r) {
            int qi = tid * bm_per_thread + r;
            #pragma unroll 4
            for (int ks = 0; ks < BLOCK_S; ++ks) {
                int kj = kv_start + ks;
                float acc = 0.0f;
                const float* q_row = q_regs[r];
                const __nv_bfloat16* k_row = &smem_K[ks * HEAD_DIM];
                
                #pragma unroll 8
                for (int d = 0; d < HEAD_DIM; d += 2) {
                    float q_val = q_row[d];
                    float k_val = __bfloat162float(k_row[d]);
                    acc += q_val * k_val;
                    q_val = q_row[d + 1];
                    k_val = __bfloat162float(k_row[d + 1]);
                    acc += q_val * k_val;
                }
                
                acc *= scale;
                
                // Causal mask: zero out if key position > query position
                if (kj > qi) {
                    acc = 0.0f;
                } else {
                    // Clamp to 0 for numerical stability (negative scores decay anyway)
                    if (acc < 0.0f) acc = 0.0f;
                }
                scores[r][ks] = acc;
            }
        }
        
        // ---- Online Softmax and Accumulation ----
        // For each query row owned by this thread:
        //   1. Find row max among current tile scores
        //   2. Compare with running max (m_prev)
        //   3. Exp-transform and sum
        //   4. Rescale previous output if new max > old max
        //   5. Add weighted V contribution
        //   6. Update LSE
        
        float m_prev[BLOCK_M];      // Previous running max per query row
        float l_prev[BLOCK_M];      // Previous row_sum per query row (before log)
        
        // Copy running state
        #pragma unroll
        for (int r = 0; r < bm_per_thread; ++r) {
            int qi = tid * bm_per_thread + r;
            if (qi < S) {
                m_prev[r] = row_lse[qi];   // Reuse row_lse array temporarily; will fix below
            }
        }
        
        // Properly maintain m (running max) and l (running exp sum) separately
        // For the first iteration, m is the tile max, l is the tile exp sum
        // Need separate arrays since we use row_lse for the final log-sum-exp output
        
        float m_arr[BLOCK_M];    // Running max across tiles
        float l_arr[BLOCK_M];    // Running sum of exp(score - max)
        
        #pragma unroll
        for (int r = 0; r < bm_per_thread; ++r) {
            int qi = tid * bm_per_thread + r;
            if (qi < S) {
                // Find tile-local max
                float tile_max = -1e20f;
                #pragma unroll 4
                for (int ks = 0; ks < kv_count; ++ks) {
                    if (scores[r][ks] > tile_max) tile_max = scores[r][ks];
                }
                
                if (!has_valid_row[r]) {
                    // First valid tile for this row: initialize
                    m_arr[r] = tile_max;
                    
                    float tile_sum = 0.0f;
                    #pragma unroll 4
                    for (int ks = 0; ks < kv_count; ++ks) {
                        tile_sum += __expf(scores[r][ks] - tile_max);
                    }
                    l_arr[r] = tile_sum;
                    has_valid_row[r] = (tile_sum > 0.0f);
                    
                    // Accumulate output: out = sum_k(exp(scores - max) * V[k])
                    #pragma unroll 8
                    for (int d = 0; d < HEAD_DIM; d += 4) {
                        float v0 = __bfloat162float(&smem_V[0 * HEAD_DIM + d])[0];
                        float v1 = __bfloat162float(&smem_V[0 * HEAD_DIM + d + 1])[0];
                        float v2 = __bfloat162float(&smem_V[0 * HEAD_DIM + d + 2])[0];
                        float v3 = __bfloat162float(&smem_V[0 * HEAD_DIM + d + 3])[0];
                        out[d]     = 0.0f;  // Will be set below
                        
                        // Accumulate over all KV positions in this tile
                        for (int ks = 0; ks < kv_count; ++ks) {
                            float w = __expf(scores[r][ks] - tile_max);
                            const __nv_bfloat16* v_row = &smem_V[ks * HEAD_DIM];
                            out[d]     += w * __bfloat162float(&v_row[d]);
                            out[d + 1] += w * __bfloat162float(&v_row[d + 1]);
                            out[d + 2] += w * __bfloat162float(&v_row[d + 2]);
                            out[d + 3] += w * __bfloat162float(&v_row[d + 3]);
                        }
                    }
                } else {
                    // Subsequent tile: potentially rescale
                    float new_max = tile_max;
                    float old_max = m_arr[r];
                    
                    // Compute tile sum
                    float tile_sum = 0.0f;
                    #pragma unroll 4
                    for (int ks = 0; ks < kv_count; ++ks) {
                        tile_sum += __expf(scores[r][ks] - new_max);
                    }
                    
                    float prev_l = l_arr[r];
                    
                    if (new_max > old_max) {
                        // Rescale: old sum contributes exp(old_max - new_max) * prev_l
                        float scale_factor = __expf(old_max - new_max);
                        l_arr[r] = scale_factor * prev_l + tile_sum;
                        
                        // Rescale output
                        #pragma unroll
                        for (int d = 0; d < HEAD_DIM; ++d) {
                            out[d] *= scale_factor;
                            out[d] += __expf(scores[r][__ffs(tile_sum) - 1] - new_max) * __bfloat162float(&smem_V[__ffs(tile_sum) - 1][d]);
                        }
                    } else {
                        // No rescaling needed; add direct contribution
                        float scale_factor = __expf(new_max - old_max);
                        l_arr[r] = prev_l + scale_factor * tile_sum;
                        
                        #pragma unroll
                        for (int d = 0; d < HEAD_DIM; ++d) {
                            out[d] += scale_factor * tile_sum * /*avg_v*/;
                        }
                    }
                    
                    m_arr[r] = new_max;
                    
                    // Compute V contribution directly
                    #pragma unroll 4
                    for (int d = 0; d < HEAD_DIM; d += 4) {
                        for (int ks = 0; ks < kv_count; ++ks) {
                            float w = __expf(scores[r][ks] - new_max);
                            const __nv_bfloat16* v_row = &smem_V[ks * HEAD_DIM];
                            if (new_max > old_max) {
                                // Already included in rescaled accumulation above
                            } else {
                                // Add new contribution with scale factor
                                float scale_f = __expf(new_max - old_max);
                                out[d]     += w * scale_f * __bfloat162float(&v_row[d]);
                                out[d + 1] += w * scale_f * __bfloat162float(&v_row[d + 1]);
                                out[d + 2] += w * scale_f * __bfloat162float(&v_row[d + 2]);
                                out[d + 3] += w * scale_f * __bfloat162float(&v_row[d + 3]);
                            }
                        }
                    }
                }
                
                // Update LSE: log(l) + max
                row_lse[qi] = __logf(l_arr[r]) + m_arr[r];
            }
        }
    }
    
    // ---- Write back output ----
    #pragma unroll
    for (int r = 0; r < bm_per_thread; ++r) {
        int qi = tid * bm_per_thread + r;
        if (qi >= S) break;
        
        // Only write if this row had valid attention
        if (has_valid_row[r]) {
            __nv_bfloat16* o_ptr = O_base + qi * STRIDE_D_O;
            #pragma unroll 4
            for (int d = 0; d < HEAD_DIM; d += 4) {
                __nv_bfloat16 bf0 = __float2bfloat16(out[d]);
                __nv_bfloat16 bf1 = __float2bfloat16(out[d + 1]);
                __nv_bfloat16 bf2 = __float2bfloat16(out[d + 2]);
                __nv_bfloat16 bf3 = __float2bfloat16(out[d + 3]);
                
                // Vectorized store as uint2
                uint2 data;
                reinterpret_cast<__nv_bfloat16*>(&data)[0] = bf0;
                reinterpret_cast<__nv_bfloat16*>(&data)[1] = bf1;
                reinterpret_cast<__nv_bfloat16*>(&data)[2] = bf2;
                reinterpret_cast<__nv_bfloat16*>(&data)[3] = bf3;
                reinterpret_cast<uint2*>(o_ptr)[d / 4] = data;
            }
        } else {
            // All-masked row: write zeros
            __nv_bfloat16* o_ptr = O_base + qi * STRIDE_D_O;
            #pragma unroll 4
            for (int d = 0; d < HEAD_DIM; d += 4) {
                __nv_bfloat16 zero = __float2bfloat16(0.0f);
                uint2 data;
                reinterpret_cast<__nv_bfloat16*>(&data)[0] = zero;
                reinterpret_cast<__nv_bfloat16*>(&data)[1] = zero;
                reinterpret_cast<__nv_bfloat16*>(&data)[2] = zero;
                reinterpret_cast<__nv_bfloat16*>(&data)[3] = zero;
                reinterpret_cast<uint2*>(o_ptr)[d / 4] = data;
                row_lse[qi] = 0.0f;
            }
        }
    }
    
    // ---- Write back LSE ----
    #pragma unroll
    for (int r = 0; r < bm_per_thread; ++r) {
        int qi = tid * bm_per_thread + r;
        if (qi >= S) break;
        LSE_base[qi] = row_lse[qi];
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    // Extract shapes from TVM tensors
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    // Validate
    if (D != HEAD_DIM) {
        fprintf(stderr, "Expected D=%d, got D=%ld\n", HEAD_DIM, D);
        exit(1);
    }
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16*        O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float*                LSE_ptr = static_cast<float*>(LSE.data_ptr());
    
    // Compute strides (consecutive in last dims)
    int stride_S_Q = H * S * D;
    int stride_D_Q = D;
    int stride_S_K = H * S * D;
    int stride_D_K = D;
    int stride_S_V = H * S * D;
    int stride_D_V = D;
    int stride_S_O = H * S * D;
    int stride_D_O = D;
    int stride_S_LSE = H * S;
    
    // Shared memory: 2 * BLOCK_S * HEAD_DIM bf16 elements
    size_t smem_size = 2 * BLOCK_S * HEAD_DIM * sizeof(__nv_bfloat16);
    
    // Grid: one block per (batch, head)
    int num_blocks = B * H;
    
    int threads = BLOCK_M > 128 ? 256 : 128;  // Match BLOCK_M coverage
    if (threads < 128) threads = 128;
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    causal_mha_kernel<<<num_blocks, threads, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
        static_cast<int>(S),
        stride_S_Q, stride_D_Q,
        stride_S_K, stride_D_K,
        stride_S_V, stride_D_V,
        stride_S_O, stride_D_O,
        stride_S_LSE
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_blackwell::run);

}  // namespace tvm_ffi_mha_blackwell