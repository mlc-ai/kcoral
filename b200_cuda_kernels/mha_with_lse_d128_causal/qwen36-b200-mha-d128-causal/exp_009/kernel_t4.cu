#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
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

namespace tvm_ffi_mha {

// Self-contained BF16 conversion helpers to avoid cuda_bf16.h dependency
__device__ __forceinline__ float bf16_to_float(uint16_t val) {
    union { float f; uint32_t i; } u;
    u.i = static_cast<uint32_t>(val) << 16;
    return u.f;
}

__device__ __forceinline__ uint16_t float_to_bf16(float val) {
    union { float f; uint32_t i; } u;
    u.f = val;
    uint32_t biased = u.i >> 16;
    // Round to nearest
    if (u.i & (1u << 15)) biased += 1;
    return static_cast<uint16_t>(biased);
}

template <int BM, int BN, int D>
__global__ void mha_attn_kernel(
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ K,
    const uint16_t* __restrict__ V,
    uint16_t* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S,
    float scale) {
    
    int bid = blockIdx.x;
    if (bid >= B * H) return;
    
    int b = bid / H;
    int h = bid % H;
    
    int q_row_start = blockIdx.y * BM;
    if (q_row_start >= S) return;
    
    int q_rows = (S - q_row_start < BM) ? (S - q_row_start) : BM;
    
    // Shared memory layout: Q(BMxD) + K(BNxD) + V(BNxD) in bf16, then O_acc(BMxD) in fp32
    extern __shared__ char smem[];
    uint16_t* smem_Q = reinterpret_cast<uint16_t*>(smem);
    uint16_t* smem_K = smem_Q + BM * D;
    uint16_t* smem_V = smem_K + BN * D;
    float* smem_O = reinterpret_cast<float*>(smem_V + BN * D);
    
    int tid = threadIdx.x;
    int NTHREADS = blockDim.x;
    
    // Base pointers for this (batch, head)
    const uint16_t* Q_g = Q + ((b * H + h) * S + q_row_start) * D;
    const uint16_t* K_g = K + ((b * H + h) * S) * D;
    const uint16_t* V_g = V + ((b * H + h) * S) * D;
    uint16_t* O_g = O + ((b * H + h) * S + q_row_start) * D;
    float* LSE_g = LSE + (b * H + h) * S + q_row_start;
    
    // Initialize O accumulator
    for (int i = tid; i < BM * D; i += NTHREADS) {
        smem_O[i] = 0.0f;
    }
    __syncthreads();
    
    // Load Q tile once
    for (int i = tid; i < q_rows * D; i += NTHREADS) {
        smem_Q[i] = Q_g[i];
    }
    for (int i = q_rows * D + tid; i < BM * D; i += NTHREADS) {
        smem_Q[i] = 0;
    }
    __syncthreads();
    
    // Each thread handles one query row's online softmax state
    float m_val[BM];
    float d_val[BM];
    float acc_scores[BN];
    
    #pragma unroll
    for (int r = 0; r < BM; ++r) {
        m_val[r] = -1e20f;
        d_val[r] = 1.0f;
    }
    
    // Main loop over KV sequence length
    for (int kv_start = 0; kv_start < S; kv_start += BN) {
        bool last_blk = (kv_start + BN >= S);
        int kv_len = last_blk ? (S - kv_start) : BN;
        
        // Load K tile
        for (int i = tid; i < kv_len * D; i += NTHREADS) {
            smem_K[i] = K_g[kv_start * D + i];
        }
        for (int i = kv_len * D + tid; i < BN * D; i += NTHREADS) {
            smem_K[i] = 0;
        }
        
        // Load V tile
        for (int i = tid; i < kv_len * D; i += NTHREADS) {
            smem_V[i] = V_g[kv_start * D + i];
        }
        for (int i = kv_len * D + tid; i < BN * D; i += NTHREADS) {
            smem_V[i] = 0;
        }
        __syncthreads();
        
        // Each thread processes its assigned query rows
        for (int r = tid; r < q_rows; r += NTHREADS) {
            const uint16_t* q_r = smem_Q + r * D;
            
            // Compute dot products for all k-cols in this block
            float local_max = -1e20f;
            for (int j = 0; j < kv_len; ++j) {
                float dot = 0.0f;
                const uint16_t* k_r = smem_K + j * D;
                
                // Manual FP32 dot product with unrolled 4-element steps
                #pragma unroll
                for (int d = 0; d < D; d += 4) {
                    float q0 = bf16_to_float(q_r[d]);
                    float q1 = bf16_to_float(q_r[d+1]);
                    float q2 = bf16_to_float(q_r[d+2]);
                    float q3 = bf16_to_float(q_r[d+3]);
                    float k0 = bf16_to_float(k_r[d]);
                    float k1 = bf16_to_float(k_r[d+1]);
                    float k2 = bf16_to_float(k_r[d+2]);
                    float k3 = bf16_to_float(k_r[d+3]);
                    dot += q0*k0 + q1*k1 + q2*k2 + q3*k3;
                }
                dot *= scale;
                
                // Causal mask: zero out future positions
                if (kv_start + j >= q_row_start + r) {
                    dot = -1e20f;
                }
                acc_scores[j] = dot;
                if (dot > local_max) local_max = dot;
            }
            // Zero-pad remaining
            for (int j = kv_len; j < BN; ++j) acc_scores[j] = -1e20f;
            
            // Online softmax update
            float cur_m = m_val[r];
            float new_m = (cur_m > local_max) ? cur_m : local_max;
            float alpha = expf(cur_m - new_m);
            float new_d = alpha * d_val[r];
            
            // Accumulate output
            float* o_r = smem_O + r * D;
            for (int j = 0; j < kv_len; ++j) {
                float p = expf(acc_scores[j] - new_m);
                new_d += p;
                const uint16_t* v_r = smem_V + j * D;
                #pragma unroll
                for (int d = 0; d < D; d += 4) {
                    float v0 = bf16_to_float(v_r[d]);
                    float v1 = bf16_to_float(v_r[d+1]);
                    float v2 = bf16_to_float(v_r[d+2]);
                    float v3 = bf16_to_float(v_r[d+3]);
                    
                    o_r[d]   = o_r[d]   * alpha + p * v0;
                    o_r[d+1] = o_r[d+1] * alpha + p * v1;
                    o_r[d+2] = o_r[d+2] * alpha + p * v2;
                    o_r[d+3] = o_r[d+3] * alpha + p * v3;
                }
            }
            
            m_val[r] = new_m;
            d_val[r] = new_d;
        }
        __syncthreads();
    }
    
    // Final normalization and write-back
    for (int r = tid; r < q_rows; r += NTHREADS) {
        float inv_d = (d_val[r] > 0.0f && m_val[r] > -5e9f) ? (1.0f / d_val[r]) : 0.0f;
        float lse = (d_val[r] > 0.0f && m_val[r] > -5e9f) ? (m_val[r] + logf(d_val[r])) : (-1e20f);
        LSE_g[r] = lse;
        
        float* o_r = smem_O + r * D;
        uint16_t* out_ptr = O_g + r * D;
        #pragma unroll
        for (int d = 0; d < D; d += 2) {
            float f0 = o_r[d] * inv_d;
            float f1 = o_r[d+1] * inv_d;
            out_ptr[d]   = float_to_bf16(f0);
            out_ptr[d+1] = float_to_bf16(f1);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D_int64 = Q.size(3);
    
    // D=128 is fixed per task spec
    constexpr int D = 128;
    if (D_int64 != D) {
        fprintf(stderr, "Unsupported D=%lld (expected %d)\n", (long long)D_int64, D);
        exit(1);
    }
    
    const uint16_t* dQ = static_cast<const uint16_t*>(Q.data_ptr());
    const uint16_t* dK = static_cast<const uint16_t*>(K.data_ptr());
    const uint16_t* dV = static_cast<const uint16_t*>(V.data_ptr());
    uint16_t* dO = static_cast<uint16_t*>(O.data_ptr());
    float* dLSE = static_cast<float*>(LSE.data_ptr());
    
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int THREADS = 128;
    
    int grid_y = (static_cast<int>(S) + BM - 1) / BM;
    dim3 grid(static_cast<int>(B) * static_cast<int>(H), grid_y, 1);
    dim3 block(THREADS, 1, 1);
    
    // Shared memory: Q(BMxD) + K(BNxD) + V(BNxD) in bf16, O(BMxD) in fp32
    size_t smem_size = (static_cast<size_t>(BM) + 2 * static_cast<size_t>(BN)) * D * sizeof(uint16_t) +
                       static_cast<size_t>(BM) * D * sizeof(float);
    
    if (smem_size > 200 * 1024) {
        fprintf(stderr, "Shared memory too large: %zu\n", smem_size);
        exit(1);
    }
    
    float scale = rsqrtf(static_cast<float>(D));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_attn_kernel<BM, BN, D><<<grid, block, smem_size, stream>>>(
        dQ, dK, dV, dO, dLSE, 
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S), 
        scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

}  // namespace tvm_ffi_mha