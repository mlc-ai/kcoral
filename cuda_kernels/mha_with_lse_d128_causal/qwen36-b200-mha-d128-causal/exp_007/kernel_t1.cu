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
    const int              B,
    const int              H,
    const int              S,
    const int              D,
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
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;  // Expected: 128
    
    // Decompose block index: grid is linear over B*H
    const int batch_id = bid / H;
    const int head_id  = bid % H;
    
    // Pointers for this batch/head
    const __nv_bfloat16* Q_base = Q + batch_id * H * stride_S_Q + head_id * stride_S_Q;
    const __nv_bfloat16* K_base = K + batch_id * H * stride_S_K + head_id * stride_S_K;
    const __nv_bfloat16* V_base = V + batch_id * H * stride_S_V + head_id * stride_S_V;
    __nv_bfloat16*       O_base = O + batch_id * H * stride_S_O + head_id * stride_S_O;
    float*               LSE_base = LSE + batch_id * H * stride_S_LSE + head_id * stride_S_LSE;
    
    // Shared memory: K tile [BLOCK_S][HEAD_DIM], V tile [BLOCK_S][HEAD_DIM]
    extern __shared__ char smem[];
    __nv_bfloat16* smem_K = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* smem_V = reinterpret_cast<__nv_bfloat16*>(smem) + BLOCK_S * HEAD_DIM;
    
    // Number of query rows this thread handles
    const int bm_per_thread = (BLOCK_M + nthreads - 1) / nthreads;
    const int num_tiles = (S + BLOCK_S - 1) / BLOCK_S;
    
    // Scale factor for softmax: 1/sqrt(D)
    const float scale = rsqrtf(static_cast<float>(D));
    
    // Per-query-row accumulators in registers
    // We only store values for rows owned by this thread
    float out[bm_per_thread][HEAD_DIM];       // Accumulated output
    float m_arr[bm_per_thread];               // Running max across tiles
    float l_arr[bm_per_thread];               // Running sum of exp(score - max)
    bool has_valid_row[bm_per_thread];        // Track whether each row has seen valid keys
    
    // Initialize accumulators
    #pragma unroll
    for (int r = 0; r < bm_per_thread; ++r) {
        m_arr[r] = -1e20f;
        l_arr[r] = 0.0f;
        has_valid_row[r] = false;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            out[r][d] = 0.0f;
        }
    }
    
    // ---- Load Q rows into registers ----
    float q_regs[bm_per_thread][HEAD_DIM];
    #pragma unroll
    for (int r = 0; r < bm_per_thread; ++r) {
        int qi = tid * bm_per_thread + r;
        if (qi < S) {
            const __nv_bfloat16* q_ptr = Q_base + qi * stride_D_Q;
            // Vectorized load: 4 bf16 -> uint2 -> 4 float
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d += 4) {
                uint2 u = reinterpret_cast<const uint2*>(q_ptr)[d / 2];
                __nv_bfloat16* tmp = reinterpret_cast<__nv_bfloat16*>(&u);
                q_regs[r][d]     = __bfloat162float(tmp[0]);
                q_regs[r][d + 1] = __bfloat162float(tmp[1]);
                q_regs[r][d + 2] = __bfloat162float(tmp[2]);
                q_regs[r][d + 3] = __bfloat162float(tmp[3]);
            }
        } else {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                q_regs[r][d] = 0.0f;
            }
        }
    }
    
    // ---- Main loop over KV sequence ----
    for (int tile = 0; tile < num_tiles; ++tile) {
        int kv_start = tile * BLOCK_S;
        int kv_count = min(BLOCK_S, S - kv_start);
        
        // ---- Load K tile into shared memory ----
        for (int idx = tid; idx < BLOCK_S * HEAD_DIM; idx += nthreads) {
            int s_idx = idx / HEAD_DIM;
            int d_idx = idx % HEAD_DIM;
            int global_s = kv_start + s_idx;
            if (global_s < S && s_idx < kv_count) {
                smem_K[idx] = K_base[global_s * stride_D_K + d_idx];
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
            if (global_s < S && s_idx < kv_count) {
                smem_V[idx] = V_base[global_s * stride_D_V + d_idx];
            } else {
                smem_V[idx] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
        
        // ---- Compute Q @ K^T tile (attention scores) with causal mask ----
        float scores[bm_per_thread][BLOCK_S];
        #pragma unroll
        for (int r = 0; r < bm_per_thread; ++r) {
            int qi = tid * bm_per_thread + r;
            #pragma unroll 2
            for (int ks = 0; ks < kv_count; ++ks) {
                int kj = kv_start + ks;
                float acc = 0.0f;
                
                // Dot product Q[qi] . K[kj]
                const float* q_row = q_regs[r];
                const __nv_bfloat16* k_row = &smem_K[ks * HEAD_DIM];
                
                #pragma unroll 16
                for (int d = 0; d < HEAD_DIM; d += 4) {
                    acc += q_row[d]     * __bfloat162float(k_row[d]);
                    acc += q_row[d + 1] * __bfloat162float(k_row[d + 1]);
                    acc += q_row[d + 2] * __bfloat162float(k_row[d + 2]);
                    acc += q_row[d + 3] * __bfloat162float(k_row[d + 3]);
                }
                
                acc *= scale;
                
                // Causal mask: zero out if key position > query position
                if (kj > qi || qi >= S) {
                    scores[r][ks] = -1e20f;
                } else {
                    scores[r][ks] = acc;
                }
            }
            // Zero out remaining scores beyond kv_count
            for (int ks = kv_count; ks < BLOCK_S; ++ks) {
                scores[r][ks] = -1e20f;
            }
        }
        
        // ---- Find per-row max within this tile ----
        float tile_max[bm_per_thread];
        #pragma unroll
        for (int r = 0; r < bm_per_thread; ++r) {
            float tm = -1e20f;
            #pragma unroll 4
            for (int ks = 0; ks < kv_count; ++ks) {
                if (scores[r][ks] > tm) tm = scores[r][ks];
            }
            tile_max[r] = tm;
        }
        
        // ---- Online Softmax and Accumulation ----
        #pragma unroll
        for (int r = 0; r < bm_per_thread; ++r) {
            int qi = tid * bm_per_thread + r;
            
            float new_max = tile_max[r];
            float old_max = m_arr[r];
            
            // Check if this row has any valid (non-masked) elements in this tile
            bool tile_has_valid = false;
            #pragma unroll
            for (int ks = 0; ks < kv_count; ++ks) {
                if (scores[r][ks] > -1e19f) {
                    tile_has_valid = true;
                    break;
                }
            }
            
            if (!tile_has_valid) continue;  // Skip if entire row is masked
            
            if (!has_valid_row[r]) {
                // First valid tile for this row: initialize
                m_arr[r] = new_max;
                
                // Compute exp(scores - new_max) and accumulate
                float tile_sum = 0.0f;
                float p[BLOCK_S];  // Attention weights for this tile
                
                #pragma unroll 4
                for (int ks = 0; ks < kv_count; ++ks) {
                    p[ks] = __expf(scores[r][ks] - new_max);
                    tile_sum += p[ks];
                }
                
                l_arr[r] = tile_sum;
                has_valid_row[r] = true;
                
                // Accumulate output: out = sum_k(p[k] * V[k])
                #pragma unroll 4
                for (int d = 0; d < HEAD_DIM; d += 4) {
                    out[r][d]     = 0.0f;
                    out[r][d + 1] = 0.0f;
                    out[r][d + 2] = 0.0f;
                    out[r][d + 3] = 0.0f;
                    
                    #pragma unroll 8
                    for (int ks = 0; ks < kv_count; ++ks) {
                        const __nv_bfloat16* v_row = &smem_V[ks * HEAD_DIM];
                        out[r][d]     += p[ks] * __bfloat162float(v_row[d]);
                        out[r][d + 1] += p[ks] * __bfloat162float(v_row[d + 1]);
                        out[r][d + 2] += p[ks] * __bfloat162float(v_row[d + 2]);
                        out[r][d + 3] += p[ks] * __bfloat162float(v_row[d + 3]);
                    }
                }
            } else {
                // Subsequent tile: potentially rescale previous output
                float prev_l = l_arr[r];
                
                if (new_max > old_max) {
                    // Rescale: old sum contributes exp(old_max - new_max) * prev_l
                    float scale_factor = __expf(old_max - new_max);
                    
                    // Scale previous output
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        out[r][d] *= scale_factor;
                    }
                    
                    m_arr[r] = new_max;
                    
                    // Compute new attention weights with new_max
                    float tile_sum = 0.0f;
                    #pragma unroll 4
                    for (int ks = 0; ks < kv_count; ++ks) {
                        float p_val = __expf(scores[r][ks] - new_max);
                        tile_sum += p_val;
                        
                        // Accumulate V contribution
                        const __nv_bfloat16* v_row = &smem_V[ks * HEAD_DIM];
                        #pragma unroll
                        for (int d = 0; d < HEAD_DIM; d += 4) {
                            out[r][d]     += p_val * __bfloat162float(v_row[d]);
                            out[r][d + 1] += p_val * __bfloat162float(v_row[d + 1]);
                            out[r][d + 2] += p_val * __bfloat162float(v_row[d + 2]);
                            out[r][d + 3] += p_val * __bfloat162float(v_row[d + 3]);
                        }
                    }
                    l_arr[r] = scale_factor * prev_l + tile_sum;
                } else {
                    // No rescaling needed for max, but scale V contribution
                    float scale_factor = __expf(new_max - old_max);
                    
                    #pragma unroll 4
                    for (int ks = 0; ks < kv_count; ++ks) {
                        float p_val = __expf(scores[r][ks] - old_max);
                        const __nv_bfloat16* v_row = &smem_V[ks * HEAD_DIM];
                        #pragma unroll
                        for (int d = 0; d < HEAD_DIM; d += 4) {
                            out[r][d]     += p_val * __bfloat162float(v_row[d]);
                            out[r][d + 1] += p_val * __bfloat162float(v_row[d + 1]);
                            out[r][d + 2] += p_val * __bfloat162float(v_row[d + 2]);
                            out[r][d + 3] += p_val * __bfloat162float(v_row[d + 3]);
                        }
                    }
                    
                    m_arr[r] = old_max;
                    float tile_sum_scaled = scale_factor * __expf(tile_max[r] - new_max);
                    // Actually compute tile_sum properly
                    float raw_tile_sum = 0.0f;
                    #pragma unroll 4
                    for (int ks = 0; ks < kv_count; ++ks) {
                        raw_tile_sum += __expf(scores[r][ks] - new_max);
                    }
                    l_arr[r] = prev_l + raw_tile_sum * __expf(new_max - old_max);
                }
            }
        }
    }
    
    // ---- Write back output and LSE ----
    #pragma unroll
    for (int r = 0; r < bm_per_thread; ++r) {
        int qi = tid * bm_per_thread + r;
        if (qi >= S) break;
        
        if (has_valid_row[r]) {
            // Compute final LSE: log(l) + max
            LSE_base[qi] = __logf(l_arr[r]) + m_arr[r];
            
            // Convert fp32 output to bf16 and write
            __nv_bfloat16* o_ptr = O_base + qi * stride_D_O;
            #pragma unroll 4
            for (int d = 0; d < HEAD_DIM; d += 4) {
                union {
                    uint32_t bits;
                    __nv_bfloat16 bf[2];
                } u;
                u.bf[0] = __float2bfloat16(out[r][d]);
                u.bf[1] = __float2bfloat16(out[r][d + 1]);
                uint2 data;
                ((uint32_t*)&data)[0] = u.bits;
                
                union {
                    uint32_t bits2;
                    __nv_bfloat16 bf2[2];
                } u2;
                u2.bf2[0] = __float2bfloat16(out[r][d + 2]);
                u2.bf2[1] = __float2bfloat16(out[r][d + 3]);
                ((uint32_t*)&data)[1] = u2.bits2;
                
                reinterpret_cast<uint2*>(o_ptr)[d / 4] = data;
            }
        } else {
            // All-masked row: write zeros, LSE=0
            LSE_base[qi] = 0.0f;
            __nv_bfloat16 zero = __float2bfloat16(0.0f);
            __nv_bfloat16* o_ptr = O_base + qi * stride_D_O;
            #pragma unroll 4
            for (int d = 0; d < HEAD_DIM; d += 4) {
                uint2 data = make_uint2(0, 0);
                reinterpret_cast<uint2*>(o_ptr)[d / 4] = data;
            }
        }
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
    
    // Compute strides (contiguous layout: [B, H, S, D])
    int stride_S_Q = static_cast<int>(S * D);
    int stride_D_Q = 1;
    int stride_S_K = static_cast<int>(S * D);
    int stride_D_K = 1;
    int stride_S_V = static_cast<int>(S * D);
    int stride_D_V = 1;
    int stride_S_O = static_cast<int>(S * D);
    int stride_D_O = 1;
    int stride_S_LSE = static_cast<int>(S);
    
    // Shared memory: 2 * BLOCK_S * HEAD_DIM bf16 elements
    size_t smem_size = 2 * BLOCK_S * HEAD_DIM * sizeof(__nv_bfloat16);
    
    // Grid: one block per (batch, head)
    int num_blocks = static_cast<int>(B * H);
    
    int threads = 128;  // Match BLOCK_M coverage
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    causal_mha_kernel<<<num_blocks, threads, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
        static_cast<int>(B),
        static_cast<int>(H),
        static_cast<int>(S),
        static_cast<int>(D),
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