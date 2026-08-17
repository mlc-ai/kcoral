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

namespace mha_back_tiled {

constexpr int BLOCK_M = 64;   // Query tile size
constexpr int BLOCK_N = 64;   // Key tile size
constexpr int TMA_TILE_D = 32; // Stride-d chunk loaded per thread
constexpr int NUM_THREADS = 128; // Threads per block (for one bh pair)

// Zero out bf16 array
__global__ void memset_bf16_kernel(__nv_bfloat16* ptr, size_t n, __nv_bfloat16 val) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = val;
}

__global__ void mha_backward_tiled_kernel(
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
    
    // Shared memory layout:
    // Q_smem: BLOCK_M x d bytes as bf16
    // K_smem: BLOCK_N x d bytes as bf16
    // V_smem: BLOCK_N x d bytes as bf16
    // dO_smem: BLOCK_M x d bytes as bf16
    // O_smem: BLOCK_M x d bytes as bf16
    
    int stride_d_half = d / 2; // Each thread loads d/2 bf16 elements
    
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK = sQ + BLOCK_M * d;
    __nv_bfloat16* sV = sK + BLOCK_N * d;
    __nv_bfloat16* sdO = sV + BLOCK_N * d;
    __nv_bfloat16* sO = sdO + BLOCK_M * d;
    
    int tid = threadIdx.x;
    int bh = blockIdx.x;
    if (bh >= B * H) return;
    
    int b = bh / H;
    int h = bh % H;
    
    size_t base_bh = (size_t)b * H * S * d + (size_t)h * S * d;
    size_t base_L = (size_t)b * H * S + (size_t)h * S;
    
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    
    // Allocate register arrays for partial accumulators
    // Each thread owns (tid / (NUM_THREADS/BLOCK_M)) rows in the Q tile
    // and computes d/2 dimension elements
    
    int q_row_start = (tid / (NUM_THREADS / BLOCK_M)); // Which row(s) this thread owns
    int num_q_rows = BLOCK_M / (NUM_THREADS / BLOCK_M); // Rows per thread group
    
    // We'll accumulate dQ contributions in registers
    float reg_dQ[TMA_TILE_D]; // Partial dQ accumulators (each thread owns d/2 dims)
    for (int i = 0; i < TMA_TILE_D; i++) reg_dQ[i] = 0.0f;
    
    // Process tiles of Q dimension
    for (int m_block_idx = 0; m_block_idx < S; m_block_idx += BLOCK_M) {
        int sq_start = m_block_idx;
        int sq_end = min(sq_start + BLOCK_M, S);
        int cur_BLOCK_M = sq_end - sq_start;
        
        // Load Q tile into shared memory
        for (int row = sq_start + tid / (NUM_THREADS / BLOCK_M); row < sq_end; 
             row += NUM_THREADS / (NUM_THREADS / BLOCK_M)) {
            for (int col_off = tid % (NUM_THREADS / (NUM_THREADS / BLOCK_M)); 
                 col_off < d; col_off += NUM_THREADS / (NUM_THREADS / BLOCK_M)) {
                // This indexing is getting complicated. Simplify: each thread loads strided elements
            }
        }
        
        // Simpler loading: assign work based on total elements in tile
        int tile_size_Q = cur_BLOCK_M * d;
        for (int elem = tid; elem < tile_size_Q; elem += NUM_THREADS) {
            int r = elem / d;
            int c = elem % d;
            sQ[r * d + c] = Q[base_bh + (size_t)(sq_start + r) * d + c];
        }
        
        // Similarly load dO and O tiles
        int tile_size_dO = cur_BLOCK_M * d;
        for (int elem = tid; elem < tile_size_dO; elem += NUM_THREADS) {
            int r = elem / d;
            int c = elem % d;
            sdO[r * d + c] = dO[base_bh + (size_t)(sq_start + r) * d + c];
            sO[r * d + c] = O[base_bh + (size_t)(sq_start + r) * d + c];
        }
        
        __syncthreads();
        
        // Load LSE values for this Q tile
        float lse[BLOCK_M];
        for (int i = tid; i < cur_BLOCK_M; i += NUM_THREADS) {
            lse[i] = L[base_L + sq_start + i];
        }
        // Broadcast lse to all threads via shared mem or just recompute
        // Actually keep in shared mem
        __nv_bfloat16* sLSE = reinterpret_cast<__nv_bfloat16*>(&lse); // Not valid, use real shared
        
        // Compute D = row_sum(dO .* O) for each query position in tile
        float D_vals[BLOCK_M];
        #pragma unroll
        for (int i = 0; i < BLOCK_M; i++) D_vals[i] = 0.0f;
        
        for (int row = tid; row < cur_BLOCK_M; row += NUM_THREADS) {
            float D = 0.0f;
            for (int j = 0; j < d; j += 4) {
                D += __bfloat162float(sdO[row * d + j])   * __bfloat162float(sO[row * d + j]);
                D += __bfloat162float(sdO[row * d + j+1]) * __bfloat162float(sO[row * d + j+1]);
                D += __bfloat162float(sdO[row * d + j+2]) * __bfloat162float(sO[row * d + j+2]);
                D += __bfloat162float(sdO[row * d + j+3]) * __bfloat162float(sO[row * d + j+3]);
            }
            D_vals[row] = D;
        }
        
        __syncthreads();
        
        // Now iterate over K/V tiles
        for (int n_block_idx = 0; n_block_idx < S; n_block_idx += BLOCK_N) {
            int sk_start = n_block_idx;
            int sk_end = min(sk_start + BLOCK_N, S);
            int cur_BLOCK_N = sk_end - sk_start;
            
            // Check causal mask: only process when sk <= sq for some sq in current tile
            // Since causal: sk <= sq is required. If sk_start > sq_end - 1, skip entirely
            if (sk_start >= sq_end) break;
            
            // Load K and V tiles
            int tile_size_KV = cur_BLOCK_N * d;
            for (int elem = tid; elem < tile_size_KV; elem += NUM_THREADS) {
                int r = elem / d;
                int c = elem % d;
                sK[r * d + c] = K[base_bh + (size_t)(sk_start + r) * d + c];
                sV[r * d + c] = V[base_bh + (size_t)(sk_start + r) * d + c];
            }
            
            __syncthreads();
            
            // Each thread computes attention scores for its assigned Q row against all K rows
            // And accumulates dQ contributions
            
            for (int q_row = tid; q_row < cur_BLOCK_M; q_row += NUM_THREADS) {
                int abs_sq = sq_start + q_row;
                float cur_lse = L[base_L + abs_sq];
                float cur_D = 0.0f; // Will compute below
                
                // Actually compute D locally since it depends on q_row
                {
                    float D_local = 0.0f;
                    for (int j = 0; j < d; j += 4) {
                        D_local += __bfloat162float(sdO[q_row * d + j])   * __bfloat162float(sO[q_row * d + j]);
                        D_local += __bfloat162float(sdO[q_row * d + j+1]) * __bfloat162float(sO[q_row * d + j+1]);
                        D_local += __bfloat162float(sdO[q_row * d + j+2]) * __bfloat162float(sO[q_row * d + j+2]);
                        D_local += __bfloat162float(sdO[q_row * d + j+3]) * __bfloat162float(sO[q_row * d + j+3]);
                    }
                    cur_D = D_local;
                }
                
                // For each K row in tile (with causal mask check)
                for (int k_row = 0; k_row < cur_BLOCK_N; k_row++) {
                    int abs_sk = sk_start + k_row;
                    
                    // Causal check
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
                    
                    float ds = p * (dp_partial - cur_D);
                    
                    // Accumulate dQ[abs_sq, dim] += ds * K[abs_sk, dim]
                    // Each thread accumulates its portion
                    for (int dim_off = tid; dim_off < d; dim_off += NUM_THREADS) {
                        // Store as atomic-like in a temporary buffer since multiple threads may write same output
                        // Actually each (q_row, dim) is owned by exactly one thread across all n_blocks
                        // So we can accumulate in registers and write once at the end
                        // But reg_dQ array is small. Let's use a different approach.
                    }
                }
            }
        }
        
        // Write dQ results
        for (int q_row = tid; q_row < cur_BLOCK_M; q_row += NUM_THREADS) {
            int abs_sq = sq_start + q_row;
            for (int dim = tid; dim < d; dim += NUM_THREADS) {
                // Would need proper accumulator here
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
    
    // Shared memory needed: Q(BLOCK_M*d) + K(BLOCK_N*d) + V(BLOCK_N*d) + dO(BLOCK_M*d) + O(BLOCK_M*d)
    // In bytes: (3*BLOCK_M + 2*BLOCK_N) * d * sizeof(bf16) = (192 + 128) * 128 * 2 = 81920 bytes
    size_t smem_size = ((size_t)3 * BLOCK_M + 2 * BLOCK_N) * d * sizeof(__nv_bfloat16);
    
    int blocks_per_bh = 1; // One block per (b,h) pair for simplicity
    // But 4*48 = 192 blocks total, limited by max grid size
    
    mha_backward_tiled_kernel<<<B * H, NUM_THREADS, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr, B, H, S, d);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_back_tiled::run);

}  // namespace mha_back_tiled