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

namespace tvm_ffi_mha_causal_d128 {

// Tiled causal attention kernel for BF16 inputs
// Each block computes attention for one (batch, head) pair
// Processes all query positions in tiles of BM=64
template <int BM = 64, int BN = 64, int BK = 32, int NUM_THREADS = 128>
__global__ void causal_attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, int D,
    float inv_sqrt_d)
{
    extern __shared__ char shared_mem[];
    
    __nv_bfloat16* s_Q = reinterpret_cast<__nv_bfloat16*>(shared_mem);
    __nv_bfloat16* s_K = s_Q + BM * D;
    __nv_bfloat16* s_V = s_K + BN * D;
    
    int tid = threadIdx.x;
    int bid_bh = blockIdx.x;  // Flattened batch*head index
    
    int batch = bid_bh / H;
    int head = bid_bh % H;
    
    if (batch >= B || head >= H) return;
    
    // Pointers to this (batch, head) data
    const __nv_bfloat16* Q_base = Q + ((static_cast<int64_t>(batch) * H + head) * S * D);
    const __nv_bfloat16* K_base = K + ((static_cast<int64_t>(batch) * H + head) * S * D);
    const __nv_bfloat16* V_base = V + ((static_cast<int64_t>(batch) * H + head) * S * D);
    __nv_bfloat16* O_base = O + ((static_cast<int64_t>(batch) * H + head) * S * D);
    float* LSE_base = LSE + (static_cast<int64_t>(batch) * H + head) * S;
    
    // Number of query blocks and KV blocks
    int num_q_blocks = (S + BM - 1) / BM;
    int num_kv_blocks = (S + BN - 1) / BN;
    
    // Each thread handles BM elements along sequence dimension for Q load/store
    // Split work: threads collaborate on D-dimension loads via vectorized reads
    
    // Process each query block
    for (int b_m = 0; b_m < num_q_blocks; b_m++) {
        int q_start = b_m * BM;
        int q_end = min(q_start + BM, S);
        int bm_actual = q_end - q_start;
        
        // Load Q tile: BM x D -> shared memory
        // Each thread loads a row segment
        for (int i = tid; i < bm_actual * D; i += NUM_THREADS) {
            int row = i / D;
            int col = i % D;
            s_Q[row * D + col] = Q_base[(q_start + row) * D + col];
        }
        __syncthreads();
        
        // Initialize output accumulators and softmax state
        // Each thread owns some rows in the BM-block
        // Store FP32 partial sums per row owned by this thread
        float thread_o[NUM_THREADS > D ? NUM_THREADS/D : 1][D];
        float thread_max[NUM_THREADS > D ? NUM_THREADS/D : 1];
        float thread_sum[NUM_THREADS > D ? NUM_THREADS/D : 1];
        
        #pragma unroll
        for (int r = 0; r < (NUM_THREADS > D ? NUM_THREADS/D : 1); r++) {
            thread_max[r] = -FLT_MAX;
            thread_sum[r] = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; d++) {
                thread_o[r][d] = 0.0f;
            }
        }
        
        int my_row_start = (tid * BM) / NUM_THREADS;
        int my_row_count = ((tid + 1) * BM) / NUM_THREADS - my_row_start;
        
        // Iterate over KV blocks
        for (int b_n = 0; b_n < num_kv_blocks; b_n++) {
            int kv_start = b_n * BN;
            int kv_end = min(kv_start + BN, S);
            int bn_actual = kv_end - kv_start;
            
            // Load K tile: BN x D -> shared memory
            for (int i = tid; i < bn_actual * D; i += NUM_THREADS) {
                int row = i / D;
                int col = i % D;
                s_K[row * D + col] = K_base[(kv_start + row) * D + col];
            }
            
            // Load V tile: BN x D -> shared memory
            for (int i = tid; i < bn_actual * D; i += NUM_THREADS) {
                int row = i / D;
                int col = i % D;
                s_V[row * D + col] = V_base[(kv_start + row) * D + col];
            }
            __syncthreads();
            
            // Compute S = Q @ K^T for rows owned by this thread
            // Then apply causal mask, scale, and online softmax
            float new_max_val = -FLT_MAX;
            
            #pragma unroll
            for (int r = 0; r < my_row_count; r++) {
                int qr = my_row_start + r;
                if (qr >= bm_actual) continue;
                
                float p_max = -FLT_MAX;
                
                // Compute dot products Q[qr,:] @ K[:,:]
                #pragma unroll
                for (int k_col = 0; k_col < bn_actual; k_col++) {
                    float dot = 0.0f;
                    
                    // Manual dot product with unrolling for D=128
                    #pragma unroll
                    for (int d_idx = 0; d_idx < D; d_idx += 4) {
                        float q0 = __bfloat162float(s_Q[qr * D + d_idx + 0]);
                        float q1 = __bfloat162float(s_Q[qr * D + d_idx + 1]);
                        float q2 = __bfloat162float(s_Q[qr * D + d_idx + 2]);
                        float q3 = __bfloat162float(s_Q[qr * D + d_idx + 3]);
                        
                        float k0 = __bfloat162float(s_K[k_col * D + d_idx + 0]);
                        float k1 = __bfloat162float(s_K[k_col * D + d_idx + 1]);
                        float k2 = __bfloat162float(s_K[k_col * D + d_idx + 2]);
                        float k3 = __bfloat162float(s_K[k_col * D + d_idx + 3]);
                        
                        dot += q0*k0 + q1*k1 + q2*k2 + q3*k3;
                    }
                    
                    // Apply causal mask: only attend to keys <= queries
                    int qp = q_start + qr;
                    int kp = kv_start + k_col;
                    
                    if (kp > qp) {
                        dot = -FLT_MAX;  // Mask future tokens
                    } else {
                        dot *= inv_sqrt_d;
                    }
                    
                    if (dot > p_max) {
                        p_max = dot;
                    }
                }
                
                if (p_max > new_max_val) {
                    new_max_val = p_max;
                }
            }
            
            // Reduce max across threads within the block for this query block
            // Using warp-level reductions
            __shared__ float block_new_max;
            if (tid == 0) block_new_max = -FLT_MAX;
            
            // Warp reduce
            float warp_max = new_max_val;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, offset));
            }
            #pragma unroll
            for (int offset = 8; offset > 0; offset /= 2) {
                warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, offset));
            }
            #pragma unroll
            for (int offset = 4; offset > 0; offset /= 2) {
                warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, offset));
            }
            #pragma unroll
            for (int offset = 2; offset > 0; offset /= 2) {
                warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, offset));
            }
            #pragma unroll
            for (int offset = 1; offset > 0; offset /= 2) {
                warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, offset));
            }
            
            if (threadIdx.x % 32 == 0) {
                atomicMax(&__float_as_int(block_new_max), __float_as_int(warp_max));
            }
            __syncthreads();
            
            float max_diff = block_new_max - thread_max[tid % (NUM_THREADS > D ? NUM_THREADS/D : 1)];
            
            // Second pass: compute softmax and accumulate
            float new_sum_val = 0.0f;
            
            #pragma unroll
            for (int r = 0; r < my_row_count; r++) {
                int qr = my_row_start + r;
                if (qr >= bm_actual) continue;
                
                float alpha = expf(thread_max[r] - block_new_max);
                float local_sum = 0.0f;
                
                #pragma unroll
                for (int k_col = 0; k_col < bn_actual; k_col++) {
                    float dot = 0.0f;
                    
                    #pragma unroll
                    for (int d_idx = 0; d_idx < D; d_idx += 4) {
                        float q0 = __bfloat162float(s_Q[qr * D + d_idx + 0]);
                        float q1 = __bfloat162float(s_Q[qr * D + d_idx + 1]);
                        float q2 = __bfloat162float(s_Q[qr * D + d_idx + 2]);
                        float q3 = __bfloat162float(s_Q[qr * D + d_idx + 3]);
                        
                        float k0 = __bfloat162float(s_K[k_col * D + d_idx + 0]);
                        float k1 = __bfloat162float(s_K[k_col * D + d_idx + 1]);
                        float k2 = __bfloat162float(s_K[k_col * D + d_idx + 2]);
                        float k3 = __bfloat162float(s_K[k_col * D + d_idx + 3]);
                        
                        dot += q0*k0 + q1*k1 + q2*k2 + q3*k3;
                    }
                    
                    int qp = q_start + qr;
                    int kp = kv_start + k_col;
                    
                    float attn;
                    if (kp > qp) {
                        attn = 0.0f;
                    } else {
                        float scaled_dot = dot * inv_sqrt_d;
                        attn = expf(scaled_dot - block_new_max);
                        local_sum += attn;
                        
                        // Accumulate into output
                        #pragma unroll
                        for (int d_idx = 0; d_idx < D; d_idx++) {
                            float v = __bfloat162float(s_V[k_col * D + d_idx]);
                            thread_o[r][d_idx] += attn * v;
                        }
                    }
                }
                
                thread_sum[r] = thread_sum[r] * alpha + local_sum;
                new_sum_val += local_sum;
            }
            
            // Update thread_max
            #pragma unroll
            for (int r = 0; r < my_row_count; r++) {
                thread_max[r] = block_new_max;
            }
        }
        
        // Finalize: normalize output and write back
        // Store results back to O and LSE
        for (int r = 0; r < my_row_count; r++) {
            int qr = my_row_start + r;
            if (qr >= bm_actual) continue;
            
            float inv_sum = 1.0f / (thread_sum[r] + 1e-6f);
            int out_row = q_start + qr;
            
            for (int d = 0; d < D; d++) {
                O_base[out_row * D + d] = __float2bfloat16(thread_o[r][d] * inv_sum);
            }
            
            // Compute LSE = max + log(sum)
            float lse = thread_max[r] + logf(thread_sum[r] + 1e-6f);
            LSE_base[q_start + qr] = lse;
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
    
    const __nv_bfloat16* q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());
    
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int NUM_THREADS = 128;
    
    int total_bh = static_cast<int>(B * H);
    dim3 grid(total_bh);
    dim3 block(NUM_THREADS);
    
    // Shared memory: Q(BM*D) + K(BN*D) + V(BN*D) in bytes
    size_t smem_size = (BM * D + BN * D + BN * D) * sizeof(__nv_bfloat16);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    float inv_sqrt_d = rsqrtf(static_cast<float>(D));
    
    causal_attn_kernel<BM, BN, BK, NUM_THREADS><<<grid, block, smem_size, stream>>>(
        q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
        static_cast<int>(B), static_cast<int>(H),
        static_cast<int>(S), static_cast<int>(D),
        inv_sqrt_d
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_causal_d128::run);

}  // namespace tvm_ffi_mha_causal_d128