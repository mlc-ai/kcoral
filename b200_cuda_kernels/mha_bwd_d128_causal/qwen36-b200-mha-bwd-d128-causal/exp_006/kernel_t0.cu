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

namespace mha_back {

// Zero out bf16 array
__global__ void memset_bf16_kernel(__nv_bfloat16* ptr, size_t n, __nv_bfloat16 val) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = val;
}

// Single kernel computing all gradients with proper tiling
// Each block handles one (b, h) pair and tiles over the S x S attention matrix
__global__ void mha_backward_full_kernel(
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
    // Thread index within block
    int tid = threadIdx.x;
    // Block handles one (b, h) pair
    int bh = blockIdx.x;
    if (bh >= B * H) return;
    
    int b = bh / H;
    int h = bh % H;
    
    // Base offset for this (b, h)
    size_t base = (size_t)b * H * S * d + (size_t)h * S * d;
    
    // Each thread processes one dimension across multiple sequence positions
    // Thread tid handles dimension tid % d and cycles through sequence positions
    
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    
    // Initialize local accumulators for dQ and dK
    // Each thread owns some (sq, dim) pairs for dQ
    // dK requires care due to multiple contributors
    
    // Simplest correct approach: each thread handles one (sq, dim) pair
    // and loops over sk positions internally
    
    // Number of (sq, dim) combinations per thread
    size_t num_pairs = (size_t)S * d;
    size_t total_threads_needed = num_pairs;
    
    // Use grid-stride loop pattern
    for (size_t pair_idx = (size_t)blockIdx.y * blockDim.y + tid;
         pair_idx < num_pairs;
         pair_idx += blockDim.y) {
        
        int sq = (int)(pair_idx / d);
        int dim = (int)(pair_idx % d);
        
        const __nv_bfloat16* q_vec = Q + base + (size_t)sq * d;
        const __nv_bfloat16* do_vec = dO + base + (size_t)sq * d;
        const __nv_bfloat16* o_vec = O + base + (size_t)sq * d;
        float lse = L[(size_t)b * H * S + (size_t)h * S + sq];
        
        // Compute D = sum(dO[sq] .* O[sq])
        float D = 0.0f;
        for (int i = 0; i < d; i += 4) {
            D += __bfloat162float(do_vec[i])   * __bfloat162float(o_vec[i]);
            D += __bfloat162float(do_vec[i+1]) * __bfloat162float(o_vec[i+1]);
            D += __bfloat162float(do_vec[i+2]) * __bfloat162float(o_vec[i+2]);
            D += __bfloat162float(do_vec[i+3]) * __bfloat162float(o_vec[i+3]);
        }
        
        float dq_acc = 0.0f;
        
        // Iterate over key positions (causal: sk <= sq)
        for (int sk = 0; sk <= sq; ++sk) {
            const __nv_bfloat16* k_vec = K + base + (size_t)sk * d;
            const __nv_bfloat16* v_vec = V + base + (size_t)sk * d;
            
            // Compute score = Q[sq].K[sk] / sqrt(d)
            float score = 0.0f;
            for (int i = 0; i < d; i += 4) {
                score += __bfloat162float(q_vec[i])   * __bfloat162float(k_vec[i]);
                score += __bfloat162float(q_vec[i+1]) * __bfloat162float(k_vec[i+1]);
                score += __bfloat162float(q_vec[i+2]) * __bfloat162float(k_vec[i+2]);
                score += __bfloat162float(q_vec[i+3]) * __bfloat162float(k_vec[i+3]);
            }
            score *= inv_sqrt_d;
            
            // P[sq, sk]
            float p = expf(score - lse);
            
            // dP partial = V[sk] . dO[sq]
            float dp_partial = 0.0f;
            for (int i = 0; i < d; i += 4) {
                dp_partial += __bfloat162float(v_vec[i])   * __bfloat162float(do_vec[i]);
                dp_partial += __bfloat162float(v_vec[i+1]) * __bfloat162float(do_vec[i+1]);
                dp_partial += __bfloat162float(v_vec[i+2]) * __bfloat162float(do_vec[i+2]);
                dp_partial += __bfloat162float(v_vec[i+3]) * __bfloat162float(do_vec[i+3]);
            }
            
            // dS = P * (dP - D)
            float ds = p * (dp_partial - D);
            
            // dQ[sq, dim] += ds * K[sk, dim]
            dq_acc += ds * __bfloat162float(k_vec[dim]);
        }
        
        dQ[base + (size_t)sq * d + dim] = __float2bfloat16(dq_acc);
    }
    
    __syncthreads();
    
    // Now compute dK: each thread handles (sk, dim) pairs
    // dK[sk, dim] = sum_{sq >= sk} dS[sq, sk] * Q[sq, dim]
    for (size_t pair_idx = (size_t)blockIdx.y * blockDim.y + tid;
         pair_idx < num_pairs;
         pair_idx += blockDim.y) {
        
        int sk = (int)(pair_idx / d);
        int dim = (int)(pair_idx % d);
        
        const __nv_bfloat16* k_vec = K + base + (size_t)sk * d;
        
        float dk_acc = 0.0f;
        
        // Iterate over query positions (causal: sq >= sk)
        for (int sq = sk; sq < S; ++sq) {
            const __nv_bfloat16* q_vec = Q + base + (size_t)sq * d;
            const __nv_bfloat16* do_vec = dO + base + (size_t)sq * d;
            const __nv_bfloat16* o_vec = O + base + (size_t)sq * d;
            const __nv_bfloat16* v_vec = V + base + (size_t)sk * d;
            float lse = L[(size_t)b * H * S + (size_t)h * S + sq];
            
            // Recompute D and score
            float D = 0.0f;
            for (int i = 0; i < d; i += 4) {
                D += __bfloat162float(do_vec[i])   * __bfloat162float(o_vec[i]);
                D += __bfloat162float(do_vec[i+1]) * __bfloat162float(o_vec[i+1]);
                D += __bfloat162float(do_vec[i+2]) * __bfloat162float(o_vec[i+2]);
                D += __bfloat162float(do_vec[i+3]) * __bfloat162float(o_vec[i+3]);
            }
            
            float score = 0.0f;
            for (int i = 0; i < d; i += 4) {
                score += __bfloat162float(q_vec[i])   * __bfloat162float(k_vec[i]);
                score += __bfloat162float(q_vec[i+1]) * __bfloat162float(k_vec[i+1]);
                score += __bfloat162float(q_vec[i+2]) * __bfloat162float(k_vec[i+2]);
                score += __bfloat162float(q_vec[i+3]) * __bfloat162float(k_vec[i+3]);
            }
            score *= inv_sqrt_d;
            
            float p = expf(score - lse);
            
            float dp_partial = 0.0f;
            for (int i = 0; i < d; i += 4) {
                dp_partial += __bfloat162float(v_vec[i])   * __bfloat162float(do_vec[i]);
                dp_partial += __bfloat162float(v_vec[i+1]) * __bfloat162float(do_vec[i+1]);
                dp_partial += __bfloat162float(v_vec[i+2]) * __bfloat162float(do_vec[i+2]);
                dp_partial += __bfloat162float(v_vec[i+3]) * __bfloat162float(do_vec[i+3]);
            }
            
            float ds = p * (dp_partial - D);
            
            dk_acc += ds * __bfloat162float(q_vec[dim]);
        }
        
        dK[base + (size_t)sk * d + dim] = __float2bfloat16(dk_acc);
    }
    
    __syncthreads();
    
    // Compute dV: dV[sk, dim] = sum_{sq >= sk} P[sq, sk] * dO[sq, dim]
    for (size_t pair_idx = (size_t)blockIdx.y * blockDim.y + tid;
         pair_idx < num_pairs;
         pair_idx += blockDim.y) {
        
        int sk = (int)(pair_idx / d);
        int dim = (int)(pair_idx % d);
        
        const __nv_bfloat16* k_vec = K + base + (size_t)sk * d;
        
        float dv_acc = 0.0f;
        
        for (int sq = sk; sq < S; ++sq) {
            const __nv_bfloat16* q_vec = Q + base + (size_t)sq * d;
            float lse = L[(size_t)b * H * S + (size_t)h * S + sq];
            
            float score = 0.0f;
            for (int i = 0; i < d; i += 4) {
                score += __bfloat162float(q_vec[i])   * __bfloat162float(k_vec[i]);
                score += __bfloat162float(q_vec[i+1]) * __bfloat162float(k_vec[i+1]);
                score += __bfloat162float(q_vec[i+2]) * __bfloat162float(k_vec[i+2]);
                score += __bfloat162float(q_vec[i+3]) * __bfloat162float(k_vec[i+3]);
            }
            score *= inv_sqrt_d;
            
            float p = expf(score - lse);
            
            dv_acc += p * __bfloat162float(dO[base + (size_t)sq * d + dim]);
        }
        
        dV[base + (size_t)sk * d + dim] = __float2bfloat16(dv_acc);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    const int64_t* shape = Q.shape();
    int B = (int)shape[0];
    int H = (int)shape[1];
    int S = (int)shape[2];
    int d = (int)shape[3];
    
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
    
    // Zero out outputs first
    size_t total_elems = (size_t)B * H * S * d;
    int zero_threads = 512;
    int zero_blocks_x = B * H;
    int zero_blocks_y = (int)((total_elems / (B*H) + zero_threads - 1) / zero_threads);
    
    memset_bf16_kernel<<<dim3(zero_blocks_x, zero_blocks_y), zero_threads, 0, stream>>>(
        dQ_ptr, total_elems, __float2bfloat16(0.0f));
    CUDA_CHECK(cudaGetLastError());
    memset_bf16_kernel<<<dim3(zero_blocks_x, zero_blocks_y), zero_threads, 0, stream>>>(
        dK_ptr, total_elems, __float2bfloat16(0.0f));
    CUDA_CHECK(cudaGetLastError());
    memset_bf16_kernel<<<dim3(zero_blocks_x, zero_blocks_y), zero_threads, 0, stream>>>(
        dV_ptr, total_elems, __float2bfloat16(0.0f));
    CUDA_CHECK(cudaGetLastError());
    
    // Launch main backward kernel
    // x-dimension: one block per (b, h) pair
    // y-dimension: enough threads to cover S*d work items per block
    int threads_per_block = 128;
    int blocks_y = (int)((S * d + threads_per_block - 1) / threads_per_block);
    
    mha_backward_full_kernel<<<dim3(B * H, blocks_y), threads_per_block, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr, B, H, S, d);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_back::run);

}  // namespace mha_back