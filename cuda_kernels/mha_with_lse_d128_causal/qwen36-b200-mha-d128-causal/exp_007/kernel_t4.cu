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

namespace mha_kernels {

constexpr int BLOCK_M = 128;
constexpr int BLOCK_S = 64;
constexpr int HEAD_DIM = 128;
constexpr int NUM_THREADS = 128;

extern "C" __global__ void causal_mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16*         __restrict__ O,
    float*                 __restrict__ LSE,
    const int              S,
    const int              B,
    const int              H,
    const int              stride_B_H_Q,
    const int              stride_H_Q,
    const int              stride_S_Q,
    const int              stride_S_K,
    const int              stride_S_V,
    const int              stride_H_O,
    const int              stride_S_O,
    const int              stride_H_LSE,
    const int              stride_S_LSE
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Each block handles one (batch, head) combination
    const int batch_id = bid / H;
    const int head_id  = bid % H;
    const int qi = tid;  // Query position for this thread
    
    // Compute base offsets for this (batch, head) slice
    const __nv_bfloat16* Q_bh = Q + batch_id * stride_B_H_Q + head_id * stride_H_Q;
    const __nv_bfloat16* K_bh = K + batch_id * stride_B_H_Q + head_id * stride_S_K;
    const __nv_bfloat16* V_bh = V + batch_id * stride_B_H_Q + head_id * stride_S_V;
    __nv_bfloat16*       O_bh = O + batch_id * stride_B_H_Q + head_id * stride_H_O;
    float*               LSE_bh = LSE + batch_id * stride_B_H_Q + head_id * stride_H_LSE;
    
    // Shared memory: K tile [BLOCK_S][HEAD_DIM], V tile [BLOCK_S][HEAD_DIM]
    extern __shared__ char smem[];
    __nv_bfloat16* smem_K = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* smem_V = reinterpret_cast<__nv_bfloat16*>(smem) + BLOCK_S * HEAD_DIM;
    
    const int num_tiles = (S + BLOCK_S - 1) / BLOCK_S;
    const float scale = rsqrtf(static_cast<float>(HEAD_DIM));
    
    // Per-thread accumulators
    float out[HEAD_DIM];
    float m_val = -1e20f;
    float l_val = 0.0f;
    bool has_valid = false;
    
    for (int d = 0; d < HEAD_DIM; ++d) out[d] = 0.0f;
    
    // Load Q row into registers
    float q_row[HEAD_DIM];
    if (qi < S) {
        const __nv_bfloat16* q_ptr = Q_bh + qi * stride_S_Q;
        for (int d = 0; d < HEAD_DIM; d += 4) {
            uint2 u = reinterpret_cast<const uint2*>(q_ptr)[d >> 1];
            __nv_bfloat16* tmp = reinterpret_cast<__nv_bfloat16*>(&u);
            q_row[d]     = __bfloat162float(tmp[0]);
            q_row[d + 1] = __bfloat162float(tmp[1]);
            q_row[d + 2] = __bfloat162float(tmp[2]);
            q_row[d + 3] = __bfloat162float(tmp[3]);
        }
    } else {
        for (int d = 0; d < HEAD_DIM; ++d) q_row[d] = 0.0f;
    }
    
    // Main loop over KV sequence
    for (int tile = 0; tile < num_tiles; ++tile) {
        int kv_start = tile * BLOCK_S;
        int kv_count = min(BLOCK_S, S - kv_start);
        
        // Load K tile into shared memory
        for (int idx = tid; idx < BLOCK_S * HEAD_DIM; idx += NUM_THREADS) {
            int s_idx = idx / HEAD_DIM;
            int d_idx = idx % HEAD_DIM;
            if (s_idx < kv_count) {
                smem_K[idx] = K_bh[(kv_start + s_idx) * stride_S_K + d_idx];
            } else {
                smem_K[idx] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
        
        // Load V tile into shared memory
        for (int idx = tid; idx < BLOCK_S * HEAD_DIM; idx += NUM_THREADS) {
            int s_idx = idx / HEAD_DIM;
            int d_idx = idx % HEAD_DIM;
            if (s_idx < kv_count) {
                smem_V[idx] = V_bh[(kv_start + s_idx) * stride_S_V + d_idx];
            } else {
                smem_V[idx] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
        
        // Compute attention scores
        float scores[BLOCK_S];
        for (int ks = 0; ks < kv_count; ++ks) {
            int kj = kv_start + ks;
            float acc = 0.0f;
            const __nv_bfloat16* k_row = &smem_K[ks * HEAD_DIM];
            
            for (int d = 0; d < HEAD_DIM; d += 4) {
                acc += q_row[d]     * __bfloat162float(k_row[d]);
                acc += q_row[d + 1] * __bfloat162float(k_row[d + 1]);
                acc += q_row[d + 2] * __bfloat162float(k_row[d + 2]);
                acc += q_row[d + 3] * __bfloat162float(k_row[d + 3]);
            }
            acc *= scale;
            
            // Causal mask
            scores[ks] = (kj > qi || qi >= S) ? -1e20f : acc;
        }
        for (int ks = kv_count; ks < BLOCK_S; ++ks) {
            scores[ks] = -1e20f;
        }
        
        // Find per-row max within this tile
        float tile_max = -1e20f;
        for (int ks = 0; ks < kv_count; ++ks) {
            if (scores[ks] > tile_max) tile_max = scores[ks];
        }
        
        bool tile_has_valid = (tile_max > -1e19f);
        if (!tile_has_valid) continue;
        
        if (!has_valid) {
            // First valid tile
            m_val = tile_max;
            float p[BLOCK_S];
            float tile_sum = 0.0f;
            
            for (int ks = 0; ks < kv_count; ++ks) {
                p[ks] = __expf(scores[ks] - tile_max);
                tile_sum += p[ks];
            }
            
            l_val = tile_sum;
            has_valid = true;
            
            // Accumulate output
            for (int d = 0; d < HEAD_DIM; d += 4) {
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
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
            float old_m = m_val;
            float new_m = tile_max;
            float prev_l = l_val;
            
            if (new_m > old_m) {
                float sf = __expf(old_m - new_m);
                for (int d = 0; d < HEAD_DIM; ++d) out[d] *= sf;
                
                m_val = new_m;
                float tile_sum = 0.0f;
                for (int ks = 0; ks < kv_count; ++ks) {
                    float pv = __expf(scores[ks] - new_m);
                    tile_sum += pv;
                    const __nv_bfloat16* vr = &smem_V[ks * HEAD_DIM];
                    for (int d = 0; d < HEAD_DIM; d += 4) {
                        out[d]     += pv * __bfloat162float(vr[d]);
                        out[d + 1] += pv * __bfloat162float(vr[d + 1]);
                        out[d + 2] += pv * __bfloat162float(vr[d + 2]);
                        out[d + 3] += pv * __bfloat162float(vr[d + 3]);
                    }
                }
                l_val = sf * prev_l + tile_sum;
            } else {
                for (int ks = 0; ks < kv_count; ++ks) {
                    float pv = __expf(scores[ks] - old_m);
                    const __nv_bfloat16* vr = &smem_V[ks * HEAD_DIM];
                    for (int d = 0; d < HEAD_DIM; d += 4) {
                        out[d]     += pv * __bfloat162float(vr[d]);
                        out[d + 1] += pv * __bfloat162float(vr[d + 1]);
                        out[d + 2] += pv * __bfloat162float(vr[d + 2]);
                        out[d + 3] += pv * __bfloat162float(vr[d + 3]);
                    }
                }
                float rts = 0.0f;
                for (int ks = 0; ks < kv_count; ++ks) {
                    rts += __expf(scores[ks] - new_m);
                }
                l_val = prev_l + rts * __expf(new_m - old_m);
            }
        }
    }
    
    // Write back
    if (qi < S) {
        if (has_valid) {
            LSE_bh[qi * stride_S_LSE] = __logf(l_val) + m_val;
            
            // Convert fp32 output to bf16 using vectorized writes
            __nv_bfloat16* o_ptr = O_bh + qi * stride_S_O;
            for (int d = 0; d < HEAD_DIM; d += 4) {
                // Pack 4 bf16 values into 2 uint32 (uint2)
                __nv_bfloat16 tmp[4];
                tmp[0] = __float2bfloat16(out[d]);
                tmp[1] = __float2bfloat16(out[d + 1]);
                tmp[2] = __float2bfloat16(out[d + 2]);
                tmp[3] = __float2bfloat16(out[d + 3]);
                reinterpret_cast<uint2*>(o_ptr)[d >> 2] = *reinterpret_cast<const uint2*>(tmp);
            }
        } else {
            LSE_bh[qi * stride_S_LSE] = 0.0f;
            __nv_bfloat16* o_ptr = O_bh + qi * stride_S_O;
            for (int d = 0; d < HEAD_DIM; d += 4) {
                reinterpret_cast<uint2*>(o_ptr)[d >> 2] = make_uint2(0, 0);
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16*        O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float*                LSE_ptr = static_cast<float*>(LSE.data_ptr());
    
    // Use strides from TensorView for robustness
    int64_t stride_B_H_Q = Q.stride(0);  // stride between batches
    int64_t stride_H_Q   = Q.stride(1);  // stride between heads
    int64_t stride_S_Q   = Q.stride(2);  // stride between sequences (should be D)
    int64_t stride_S_K   = K.stride(2);
    int64_t stride_S_V   = V.stride(2);
    int64_t stride_H_O   = O.stride(1);
    int64_t stride_S_O   = O.stride(2);
    int64_t stride_H_LSE = LSE.stride(1);
    int64_t stride_S_LSE = LSE.stride(2);
    
    size_t smem_size = 2 * BLOCK_S * HEAD_DIM * sizeof(__nv_bfloat16);
    int num_blocks = static_cast<int>(B * H);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    causal_mha_kernel<<<num_blocks, NUM_THREADS, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
        static_cast<int>(S),
        static_cast<int>(B),
        static_cast<int>(H),
        static_cast<int>(stride_B_H_Q),
        static_cast<int>(stride_H_Q),
        static_cast<int>(stride_S_Q),
        static_cast<int>(stride_S_K),
        static_cast<int>(stride_S_V),
        static_cast<int>(stride_H_O),
        static_cast<int>(stride_S_O),
        static_cast<int>(stride_H_LSE),
        static_cast<int>(stride_S_LSE)
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernels::run);

}  // namespace mha_kernels