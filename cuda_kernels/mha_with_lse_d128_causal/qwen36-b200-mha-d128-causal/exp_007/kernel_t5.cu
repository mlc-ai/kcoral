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
    const int              elem_per_batch,     // H * S * D for Q/K/V/O
    const int              elem_per_head,       // S * D for Q/K/V/O
    const int              elem_per_seq_qkv,    // D for Q/K/V/O
    const int              lse_elem_per_batch,  // H * S for LSE
    const int              lse_elem_per_head    // S for LSE
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Each block handles one (batch, head) combination
    const int batch_id = bid / H;
    const int head_id  = bid % H;
    const int qi = tid;
    
    // Base pointers for this (batch, head) slice
    const __nv_bfloat16* Q_bh = Q + batch_id * elem_per_batch + head_id * elem_per_head;
    const __nv_bfloat16* K_bh = K + batch_id * elem_per_batch + head_id * elem_per_head;
    const __nv_bfloat16* V_bh = V + batch_id * elem_per_batch + head_id * elem_per_head;
    __nv_bfloat16*       O_bh = O + batch_id * elem_per_batch + head_id * elem_per_head;
    float*               LSE_bh = LSE + batch_id * lse_elem_per_batch + head_id * lse_elem_per_head;
    
    // Shared memory: K tile [BLOCK_S][HEAD_DIM], V tile [BLOCK_S][HEAD_DIM]
    extern __shared__ char smem[];
    __nv_bfloat16* smem_K = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* smem_V = reinterpret_cast<__nv_bfloat16*>(smem) + BLOCK_S * HEAD_DIM;
    
    const int num_tiles = (S + BLOCK_S - 1) / BLOCK_S;
    const float scale = rsqrtf(static_cast<float>(HEAD_DIM));
    
    // Per-thread accumulators - minimize register usage
    float out[HEAD_DIM];
    float m_val = -1e20f;
    float l_val = 0.0f;
    bool has_valid = false;
    
    for (int d = 0; d < HEAD_DIM; ++d) out[d] = 0.0f;
    
    // Load Q row into registers
    float q_row[HEAD_DIM];
    if (qi < S) {
        const __nv_bfloat16* q_ptr = Q_bh + qi * elem_per_seq_qkv;
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
                smem_K[idx] = K_bh[(kv_start + s_idx) * elem_per_seq_qkv + d_idx];
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
                smem_V[idx] = V_bh[(kv_start + s_idx) * elem_per_seq_qkv + d_idx];
            } else {
                smem_V[idx] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
        
        // Compute attention scores and online softmax/accumulate in a single pass
        float tile_max = -1e20f;
        bool tile_has_valid = false;
        
        // First pass: find tile max and check validity
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
            float score = (kj > qi || qi >= S) ? -1e20f : acc;
            if (score > tile_max) tile_max = score;
            if (score > -1e19f) tile_has_valid = true;
        }
        
        if (!tile_has_valid) continue;
        
        // Second pass: online softmax update
        if (!has_valid) {
            // First valid tile
            m_val = tile_max;
            float tile_sum = 0.0f;
            
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
                
                float score = (kj > qi || qi >= S) ? -1e20f : acc;
                float p = __expf(score - tile_max);
                tile_sum += p;
                
                // Accumulate V contribution
                const __nv_bfloat16* v_row = &smem_V[ks * HEAD_DIM];
                for (int d = 0; d < HEAD_DIM; d += 4) {
                    out[d]     += p * __bfloat162float(v_row[d]);
                    out[d + 1] += p * __bfloat162float(v_row[d + 1]);
                    out[d + 2] += p * __bfloat162float(v_row[d + 2]);
                    out[d + 3] += p * __bfloat162float(v_row[d + 3]);
                }
            }
            
            l_val = tile_sum;
            has_valid = true;
        } else {
            float old_m = m_val;
            float new_m = tile_max;
            float prev_l = l_val;
            
            if (new_m > old_m) {
                // Rescale previous output
                float sf = __expf(old_m - new_m);
                for (int d = 0; d < HEAD_DIM; ++d) out[d] *= sf;
                m_val = new_m;
                
                float tile_sum = 0.0f;
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
                    
                    float score = (kj > qi || qi >= S) ? -1e20f : acc;
                    float p = __expf(score - new_m);
                    tile_sum += p;
                    
                    const __nv_bfloat16* v_row = &smem_V[ks * HEAD_DIM];
                    for (int d = 0; d < HEAD_DIM; d += 4) {
                        out[d]     += p * __bfloat162float(v_row[d]);
                        out[d + 1] += p * __bfloat162float(v_row[d + 1]);
                        out[d + 2] += p * __bfloat162float(v_row[d + 2]);
                        out[d + 3] += p * __bfloat162float(v_row[d + 3]);
                    }
                }
                l_val = sf * prev_l + tile_sum;
            } else {
                float tile_sum_exp_new = 0.0f;
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
                    
                    float score = (kj > qi || qi >= S) ? -1e20f : acc;
                    float p = __expf(score - old_m);
                    
                    const __nv_bfloat16* v_row = &smem_V[ks * HEAD_DIM];
                    for (int d = 0; d < HEAD_DIM; d += 4) {
                        out[d]     += p * __bfloat162float(v_row[d]);
                        out[d + 1] += p * __bfloat162float(v_row[d + 1]);
                        out[d + 2] += p * __bfloat162float(v_row[d + 2]);
                        out[d + 3] += p * __bfloat162float(v_row[d + 3]);
                    }
                    
                    tile_sum_exp_new += __expf(score - new_m);
                }
                l_val = prev_l + tile_sum_exp_new * __expf(new_m - old_m);
            }
        }
    }
    
    // Write back
    if (qi < S) {
        if (has_valid) {
            LSE_bh[qi] = __logf(l_val) + m_val;
            
            // Convert fp32 output to bf16 using vectorized writes
            __nv_bfloat16* o_ptr = O_bh + qi * elem_per_seq_qkv;
            for (int d = 0; d < HEAD_DIM; d += 4) {
                __nv_bfloat16 tmp[4];
                tmp[0] = __float2bfloat16(out[d]);
                tmp[1] = __float2bfloat16(out[d + 1]);
                tmp[2] = __float2bfloat16(out[d + 2]);
                tmp[3] = __float2bfloat16(out[d + 3]);
                reinterpret_cast<uint2*>(o_ptr)[d >> 2] = *reinterpret_cast<const uint2*>(tmp);
            }
        } else {
            LSE_bh[qi] = 0.0f;
            __nv_bfloat16* o_ptr = O_bh + qi * elem_per_seq_qkv;
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
    
    // Compute element counts per dimension level for Q/K/V/O [B,H,S,D]
    int elem_per_batch   = static_cast<int>(H * S * D);
    int elem_per_head    = static_cast<int>(S * D);
    int elem_per_seq_qkv = static_cast<int>(D);  // stride along last dim
    
    // For LSE [B,H,S]
    int lse_elem_per_batch = static_cast<int>(H * S);
    int lse_elem_per_head  = static_cast<int>(S);
    
    size_t smem_size = 2 * BLOCK_S * HEAD_DIM * sizeof(__nv_bfloat16);
    int num_blocks = static_cast<int>(B * H);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    causal_mha_kernel<<<num_blocks, NUM_THREADS, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
        static_cast<int>(S),
        static_cast<int>(B),
        static_cast<int>(H),
        elem_per_batch,
        elem_per_head,
        elem_per_seq_qkv,
        lse_elem_per_batch,
        lse_elem_per_head
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernels::run);

}  // namespace mha_kernels