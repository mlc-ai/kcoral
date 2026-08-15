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

template <int BM, int BN, int D>
__global__ void mha_attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
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
    __nv_bfloat16* smem_Q = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* smem_K = smem_Q + BM * D;
    __nv_bfloat16* smem_V = smem_K + BN * D;
    float* smem_O = reinterpret_cast<float*>(smem_V + BN * D);
    
    int tid = threadIdx.x;
    int NTHREADS = blockDim.x;
    
    // Base pointers for this (batch, head)
    const __nv_bfloat16* Q_g = Q + ((b * H + h) * S + q_row_start) * D;
    const __nv_bfloat16* K_g = K + ((b * H + h) * S) * D;
    const __nv_bfloat16* V_g = V + ((b * H + h) * S) * D;
    __nv_bfloat16* O_g = O + ((b * H + h) * S + q_row_start) * D;
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
        smem_Q[i] = __float2bfloat16(0.0f);
    }
    __syncthreads();
    
    // Per-thread registers: each thread tracks multiple query rows' softmax state
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
            smem_K[i] = __float2bfloat16(0.0f);
        }
        
        // Load V tile
        for (int i = tid; i < kv_len * D; i += NTHREADS) {
            smem_V[i] = V_g[kv_start * D + i];
        }
        for (int i = kv_len * D + tid; i < BN * D; i += NTHREADS) {
            smem_V[i] = __float2bfloat16(0.0f);
        }
        __syncthreads();
        
        // Each thread processes its assigned query rows
        for (int r = tid; r < q_rows; r += NTHREADS) {
            const __nv_bfloat16* q_r = smem_Q + r * D;
            int q_pos = q_row_start + r;
            
            // Compute dot products for all k-cols in this block
            float local_max = -1e20f;
            for (int j = 0; j < kv_len; ++j) {
                float dot = 0.0f;
                const __nv_bfloat16* k_r = smem_K + j * D;
                
                // Manual FP32 dot product
                #pragma unroll
                for (int d = 0; d < D; d += 4) {
                    float q0 = __bfloat162float(q_r[d]);
                    float q1 = __bfloat162float(q_r[d+1]);
                    float q2 = __bfloat162float(q_r[d+2]);
                    float q3 = __bfloat162float(q_r[d+3]);
                    float k0 = __bfloat162float(k_r[d]);
                    float k1 = __bfloat162float(k_r[d+1]);
                    float k2 = __bfloat162float(k_r[d+2]);
                    float k3 = __bfloat162float(k_r[d+3]);
                    dot += q0*k0 + q1*k1 + q2*k2 + q3*k3;
                }
                dot *= scale;
                
                // Causal mask: key_pos > q_pos => mask out
                int k_pos = kv_start + j;
                if (k_pos > q_pos) {
                    dot = -1e20f;
                }
                acc_scores[j] = dot;
                if (dot > local_max) local_max = dot;
            }
            // Zero-pad remaining scores past kv_len
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
                const __nv_bfloat16* v_r = smem_V + j * D;
                #pragma unroll
                for (int d = 0; d < D; d += 4) {
                    float v0 = __bfloat162float(v_r[d]);
                    float v1 = __bfloat162float(v_r[d+1]);
                    float v2 = __bfloat162float(v_r[d+2]);
                    float v3 = __bfloat162float(v_r[d+3]);
                    
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
        __nv_bfloat16* out_ptr = O_g + r * D;
        #pragma unroll
        for (int d = 0; d < D; d += 2) {
            float f0 = o_r[d] * inv_d;
            float f1 = o_r[d+1] * inv_d;
            out_ptr[d]   = __float2bfloat16(f0);
            out_ptr[d+1] = __float2bfloat16(f1);
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
    
    const __nv_bfloat16* dQ = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* dK = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* dV = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* dO = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* dLSE = static_cast<float*>(LSE.data_ptr());
    
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int THREADS = 128;
    
    int grid_y = (static_cast<int>(S) + BM - 1) / BM;
    dim3 grid(static_cast<int>(B) * static_cast<int>(H), grid_y, 1);
    dim3 block(THREADS, 1, 1);
    
    // Shared memory: Q(BMxD) + K(BNxD) + V(BNxD) in bf16, O(BMxD) in fp32
    size_t smem_size = (static_cast<size_t>(BM) + 2 * static_cast<size_t>(BN)) * D * sizeof(__nv_bfloat16) +
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