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

namespace mha_back_v2 {

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int NUM_THREADS = 256;

__global__ void memset_bf16_kernel(__nv_bfloat16* ptr, size_t n, __nv_bfloat16 val) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = val;
}

__global__ void mha_backward_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int d
) {
    extern __shared__ char smem[];
    
    // Shared memory layout (all in bf16):
    // [0 : BLOCK_M*d)         = sQ   (current Q tile)
    // [BLOCK_M*d : 2*BLOCK_M*d) = sdO (current dO tile)
    // [2*BLOCK_M*d : 3*BLOCK_M*d) = sO (current O tile)
    // [3*BLOCK_M*d : 3*BLOCK_M*d+BLOCK_N*d) = sK (current K tile)
    // [3*BLOCK_M*d+BLOCK_N*d : 3*BLOCK_M*d+2*BLOCK_N*d) = sV (current V tile)
    
    __nv_bfloat16* sQ   = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sdO  = sQ + BLOCK_M * d;
    __nv_bfloat16* sO   = sdO + BLOCK_M * d;
    __nv_bfloat16* sK   = sO + BLOCK_M * d;
    __nv_bfloat16* sV   = sK + BLOCK_N * d;
    
    int tid = threadIdx.x;
    int bh = blockIdx.x;
    if (bh >= B * H) return;
    
    int b = bh / H;
    int h = bh % H;
    
    size_t base_bh = (size_t)b * H * S * d + (size_t)h * S * d;
    size_t base_L = (size_t)b * H * S + (size_t)h * S;
    
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    
    // Each thread owns some columns of the d dimension for accumulation
    // We accumulate partial dQ[d_cols_per_thread] in registers
    int cols_per_thread = (d + NUM_THREADS - 1) / NUM_THREADS;
    int my_col_start = tid * cols_per_thread;
    
    // Process Q tiles
    for (int m_block = 0; m_block < S; m_block += BLOCK_M) {
        int sq_base = m_block;
        int cur_bm = min(BLOCK_M, S - sq_base);
        
        // Load Q tile
        for (int i = tid; i < cur_bm * d; i += NUM_THREADS) {
            int r = i / d;
            int c = i % d;
            sQ[r * d + c] = Q[base_bh + (size_t)(sq_base + r) * d + c];
        }
        
        // Load dO tile
        for (int i = tid; i < cur_bm * d; i += NUM_THREADS) {
            int r = i / d;
            int c = i % d;
            sdO[r * d + c] = dO[base_bh + (size_t)(sq_base + r) * d + c];
        }
        
        // Load O tile
        for (int i = tid; i < cur_bm * d; i += NUM_THREADS) {
            int r = i / d;
            int c = i % d;
            sO[r * d + c] = O[base_bh + (size_t)(sq_base + r) * d + c];
        }
        
        __syncthreads();
        
        // Initialize register accumulators for dQ
        float reg_dQ_accum[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // each thread tracks its cols
        
        // Process K/V tiles
        for (int n_block = 0; n_block < S; n_block += BLOCK_N) {
            int sk_base = n_block;
            int cur_bn = min(BLOCK_N, S - sk_base);
            
            // Causal early exit: if sk_base > sq_base + cur_bm - 1, no valid pairs exist
            if (sk_base >= sq_base + cur_bm) break;
            
            // Load K tile
            for (int i = tid; i < cur_bn * d; i += NUM_THREADS) {
                int r = i / d;
                int c = i % d;
                sK[r * d + c] = K[base_bh + (size_t)(sk_base + r) * d + c];
            }
            
            // Load V tile
            for (int i = tid; i < cur_bn * d; i += NUM_THREADS) {
                int r = i / d;
                int c = i % d;
                sV[r * d + c] = V[base_bh + (size_t)(sk_base + r) * d + c];
            }
            
            __syncthreads();
            
            // Each thread processes assigned Q rows against all K rows in tile
            for (int q_row = tid; q_row < cur_bm; q_row += NUM_THREADS) {
                int abs_sq = sq_base + q_row;
                
                float cur_lse = L[base_L + abs_sq];
                
                // Compute D[q_row] = dot(dO[q_row], O[q_row])
                float D_val = 0.0f;
                for (int j = 0; j < d; j += 4) {
                    D_val += __bfloat162float(sdO[q_row * d + j])   * __bfloat162float(sO[q_row * d + j]);
                    D_val += __bfloat162float(sdO[q_row * d + j+1]) * __bfloat162float(sO[q_row * d + j+1]);
                    D_val += __bfloat162float(sdO[q_row * d + j+2]) * __bfloat162float(sO[q_row * d + j+2]);
                    D_val += __bfloat162float(sdO[q_row * d + j+3]) * __bfloat162float(sO[q_row * d + j+3]);
                }
                
                // Iterate over K rows in this tile
                for (int k_row = 0; k_row < cur_bn; k_row++) {
                    int abs_sk = sk_base + k_row;
                    
                    // Causal mask check
                    if (abs_sk > abs_sq) continue;
                    
                    // Compute score = Q[abs_sq] . K[abs_sk] / sqrt(d)
                    float score = 0.0f;
                    for (int j = 0; j < d; j += 4) {
                        score += __bfloat162float(sQ[q_row * d + j])   * __bfloat162float(sK[k_row * d + j]);
                        score += __bfloat162float(sQ[q_row * d + j+1]) * __bfloat162float(sK[k_row * d + j+1]);
                        score += __bfloat162float(sQ[q_row * d + j+2]) * __bfloat162float(sK[k_row * d + j+2]);
                        score += __bfloat162float(sQ[q_row * d + j+3]) * __bfloat162float(sK[k_row * d + j+3]);
                    }
                    score *= inv_sqrt_d;
                    
                    float p = expf(score - cur_lse);
                    
                    // dP_partial = V[abs_sk] . dO[abs_sq]
                    float dp_partial = 0.0f;
                    for (int j = 0; j < d; j += 4) {
                        dp_partial += __bfloat162float(sV[k_row * d + j])   * __bfloat162float(sdO[q_row * d + j]);
                        dp_partial += __bfloat162float(sV[k_row * d + j+1]) * __bfloat162float(sdO[q_row * d + j+1]);
                        dp_partial += __bfloat162float(sV[k_row * d + j+2]) * __bfloat162float(sdO[q_row * d + j+2]);
                        dp_partial += __bfloat162float(sV[k_row * d + j+3]) * __bfloat162float(sdO[q_row * d + j+3]);
                    }
                    
                    float ds = p * (dp_partial - D_val);
                    
                    // Accumulate dQ: dQ[abs_sq, col] += ds * K[abs_sk, col]
                    // Each thread accumulates its column range
                    for (int col = my_col_start; col < my_col_start + cols_per_thread && col < d; col++) {
                        reg_dQ_accum[col & 3] += ds * __bfloat162float(sK[k_row * d + col]);
                    }
                    
                    // Atomic contribution to dV[abs_sk, :] 
                    // Since multiple Q positions contribute to same sk position, use atomics
                    float do_dim_val = 0.0f;
                    for (int col = my_col_start; col < my_col_start + cols_per_thread && col < d; col++) {
                        float dv_contrib = p * __bfloat162float(sdO[q_row * d + col]);
                        // Use atomicAdd on fp32 reinterpretation
                        unsigned long long* dV_addr = (unsigned long long*)&dV[base_bh + (size_t)abs_sk * d + col];
                        unsigned int bf16_old = *(unsigned int*)dV_addr;
                        float fv_old = __bfloat162float(*((__nv_bfloat16*)dV_addr));
                        float fv_new = fv_old + dv_contrib;
                        *(unsigned int*)dV_addr = *reinterpret_cast<unsigned int*>(&fv_new); // Not atomic!
                    }
                    
                    // Atomic contribution to dK[abs_sk, :]
                    // dK[abs_sk, col] += ds * Q[abs_sq, col]
                    for (int col = my_col_start; col < my_col_start + cols_per_thread && col < d; col++) {
                        float dk_contrib = ds * __bfloat162float(sQ[q_row * d + col]);
                        // Same issue - need atomics
                        float fk_old = __bfloat162float(dK[base_bh + (size_t)abs_sk * d + col]);
                        dK[base_bh + (size_t)abs_sk * d + col] = __float2bfloat16(fk_old + dk_contrib); // Race condition!
                    }
                }
            }
            
            __syncthreads();
        }
        
        // Write dQ results for this Q tile
        for (int q_row = tid; q_row < cur_bm; q_row += NUM_THREADS) {
            int abs_sq = sq_base + q_row;
            for (int col = my_col_start; col < my_col_start + cols_per_thread && col < d; col++) {
                dQ[base_bh + (size_t)abs_sq * d + col] = __float2bfloat16(reg_dQ_accum[col & 3]);
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int d = (int)Q.size(3);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    size_t total_elems = (size_t)B * H * S * d;
    int zero_threads = 512;
    int zero_blocks = (int)((total_elems + zero_threads - 1) / zero_threads);
    
    memset_bf16_kernel<<<zero_blocks, zero_threads, 0, stream>>>(
        dQ_ptr, total_elems, __float2bfloat16(0.0f));
    CUDA_CHECK(cudaGetLastError());
    memset_bf16_kernel<<<zero_blocks, zero_threads, 0, stream>>>(
        dK_ptr, total_elems, __float2bfloat16(0.0f));
    CUDA_CHECK(cudaGetLastError());
    memset_bf16_kernel<<<zero_blocks, zero_threads, 0, stream>>>(
        dV_ptr, total_elems, __float2bfloat16(0.0f));
    CUDA_CHECK(cudaGetLastError());
    
    // Shared memory size: (3*BLOCK_M + 2*BLOCK_N) * d * sizeof(bf16)
    size_t smem_size = ((size_t)3 * BLOCK_M + 2 * BLOCK_N) * d * sizeof(__nv_bfloat16);
    
    // Limit blocks to reasonable number
    int total_bh = B * H;
    int max_blocks = 1024;
    int num_blocks = min(total_bh, max_blocks);
    
    mha_backward_kernel<<<num_blocks, NUM_THREADS, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr, B, H, S, d);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_back_v2::run);

}  // namespace mha_back_v2