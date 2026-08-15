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
constexpr int NUM_THREADS = 128; // Must match BLOCK_M

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
    
    // Each block handles one (batch, head) combination
    // Each thread handles exactly one query row
    const int batch_id = bid / H;
    const int head_id  = bid % H;
    const int qi = tid;  // Query position for this thread
    
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
    
    const int num_tiles = (S + BLOCK_S - 1) / BLOCK_S;
    
    // Scale factor for softmax: 1/sqrt(D)
    const float scale = rsqrtf(static_cast<float>(D));
    
    // Per-query-row accumulators (each thread has its own single row)
    float out[HEAD_DIM];        // Accumulated output
    float m_val = -1e20f;       // Running max across tiles
    float l_val = 0.0f;         // Running sum of exp(score - max)
    bool has_valid = false;     // Track whether row has seen valid keys
    
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        out[d] = 0.0f;
    }
    
    // ---- Load Q row into registers ----
    float q_row[HEAD_DIM];
    if (qi < S) {
        const __nv_bfloat16* q_ptr = Q_base + qi * stride_D_Q;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d += 4) {
            uint2 u = reinterpret_cast<const uint2*>(q_ptr)[d / 2];
            __nv_bfloat16* tmp = reinterpret_cast<__nv_bfloat16*>(&u);
            q_row[d]     = __bfloat162float(tmp[0]);
            q_row[d + 1] = __bfloat162float(tmp[1]);
            q_row[d + 2] = __bfloat162float(tmp[2]);
            q_row[d + 3] = __bfloat162float(tmp[3]);
        }
    } else {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            q_row[d] = 0.0f;
        }
    }
    
    // ---- Main loop over KV sequence ----
    for (int tile = 0; tile < num_tiles; ++tile) {
        int kv_start = tile * BLOCK_S;
        int kv_count = min(BLOCK_S, S - kv_start);
        
        // ---- Load K tile into shared memory ----
        for (int idx = tid; idx < BLOCK_S * HEAD_DIM; idx += NUM_THREADS) {
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
        for (int idx = tid; idx < BLOCK_S * HEAD_DIM; idx += NUM_THREADS) {
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
        
        // ---- Compute attention scores for this row ----
        float scores[BLOCK_S];
        #pragma unroll
        for (int ks = 0; ks < kv_count; ++ks) {
            int kj = kv_start + ks;
            float acc = 0.0f;
            
            // Dot product Q[qi] . K[kj]
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
                scores[ks] = -1e20f;
            } else {
                scores[ks] = acc;
            }
        }
        // Zero out remaining scores beyond kv_count
        for (int ks = kv_count; ks < BLOCK_S; ++ks) {
            scores[ks] = -1e20f;
        }
        
        // ---- Find per-row max within this tile ----
        float tile_max = -1e20f;
        #pragma unroll
        for (int ks = 0; ks < kv_count; ++ks) {
            if (scores[ks] > tile_max) tile_max = scores[ks];
        }
        
        // Check if this row has any valid (non-masked) elements in this tile
        bool tile_has_valid = (tile_max > -1e19f);
        
        if (!tile_has_valid) continue;  // Skip if entire row is masked
        
        if (!has_valid) {
            // First valid tile for this row: initialize
            m_val = tile_max;
            
            // Compute exp(scores - tile_max) and accumulate
            float p[BLOCK_S];
            float tile_sum = 0.0f;
            
            #pragma unroll 4
            for (int ks = 0; ks < kv_count; ++ks) {
                p[ks] = __expf(scores[ks] - tile_max);
                tile_sum += p[ks];
            }
            
            l_val = tile_sum;
            has_valid = true;
            
            // Accumulate output: out = sum_k(p[k] * V[k])
            #pragma unroll 4
            for (int d = 0; d < HEAD_DIM; d += 4) {
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
                
                #pragma unroll 8
                for (int ks = 0; ks < kv_count; ++ks) {
                    const __nv_bfloat16* v_row = &smem_V[ks * HEAD_DIM];
                    float w = p[ks];
                    acc0 += w * __bfloat162float(v_row[d]);
                    acc1 += w * __bfloat162float(v_row[d + 1]);
                    acc2 += w * __bfloat162float(v_row[d + 2]);
                    acc3 += w * __bfloat162float(v_row[d + 3]);
                }
                out[d]     = acc0;
                out[d + 1] = acc1;
                out[d + 2] = acc2;
                out[d + 3] = acc3;
            }
        } else {
            // Subsequent tile: potentially rescale previous output
            float old_m = m_val;
            float new_m = tile_max;
            float prev_l = l_val;
            
            if (new_m > old_m) {
                // Rescale: old values contribute exp(old_max - new_max)
                float scale_factor = __expf(old_m - new_m);
                
                // Scale previous output
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    out[d] *= scale_factor;
                }
                
                m_val = new_m;
                
                // Compute new attention weights with new_m
                float tile_sum = 0.0f;
                #pragma unroll 4
                for (int ks = 0; ks < kv_count; ++ks) {
                    float p_val = __expf(scores[ks] - new_m);
                    tile_sum += p_val;
                    
                    // Accumulate V contribution
                    const __nv_bfloat16* v_row = &smem_V[ks * HEAD_DIM];
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d += 4) {
                        out[d]     += p_val * __bfloat162float(v_row[d]);
                        out[d + 1] += p_val * __bfloat162float(v_row[d + 1]);
                        out[d + 2] += p_val * __bfloat162float(v_row[d + 2]);
                        out[d + 3] += p_val * __bfloat162float(v_row[d + 3]);
                    }
                }
                l_val = scale_factor * prev_l + tile_sum;
            } else {
                // No rescaling needed for max
                #pragma unroll 4
                for (int ks = 0; ks < kv_count; ++ks) {
                    float p_val = __expf(scores[ks] - old_m);
                    const __nv_bfloat16* v_row = &smem_V[ks * HEAD_DIM];
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d += 4) {
                        out[d]     += p_val * __bfloat162float(v_row[d]);
                        out[d + 1] += p_val * __bfloat162float(v_row[d + 1]);
                        out[d + 2] += p_val * __bfloat162float(v_row[d + 2]);
                        out[d + 3] += p_val * __bfloat162float(v_row[d + 3]);
                    }
                }
                
                // Update l_val without rescaling
                float raw_tile_sum = 0.0f;
                #pragma unroll 4
                for (int ks = 0; ks < kv_count; ++ks) {
                    raw_tile_sum += __expf(scores[ks] - new_m);
                }
                l_val = prev_l + raw_tile_sum * __expf(new_m - old_m);
            }
        }
    }
    
    // ---- Write back output and LSE ----
    if (qi < S) {
        if (has_valid) {
            // Compute final LSE: log(l) + max
            LSE_base[qi] = __logf(l_val) + m_val;
            
            // Convert fp32 output to bf16 and write
            __nv_bfloat16* o_ptr = O_base + qi * stride_D_O;
            #pragma unroll 4
            for (int d = 0; d < HEAD_DIM; d += 4) {
                union {
                    uint32_t bits;
                    __nv_bfloat16 bf[2];
                } u;
                u.bf[0] = __float2bfloat16(out[d]);
                u.bf[1] = __float2bfloat16(out[d + 1]);
                uint2 data;
                ((uint32_t*)&data)[0] = u.bits;
                
                union {
                    uint32_t bits2;
                    __nv_bfloat16 bf2[2];
                } u2;
                u2.bf2[0] = __float2bfloat16(out[d + 2]);
                u2.bf2[1] = __float2bfloat16(out[d + 3]);
                ((uint32_t*)&data)[1] = u2.bits2;
                
                reinterpret_cast<uint2*>(o_ptr)[d / 4] = data;
            }
        } else {
            // All-masked row: write zeros, LSE=0
            LSE_base[qi] = 0.0f;
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
    
    // Shared memory: 2 * BLOCK_S * HEAD_DIM bf16 elements = 2 * 64 * 128 * 2 = 32KB
    size_t smem_size = 2 * BLOCK_S * HEAD_DIM * sizeof(__nv_bfloat16);
    
    // Grid: one block per (batch, head)
    int num_blocks = static_cast<int>(B * H);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    causal_mha_kernel<<<num_blocks, NUM_THREADS, smem_size, stream>>>(
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