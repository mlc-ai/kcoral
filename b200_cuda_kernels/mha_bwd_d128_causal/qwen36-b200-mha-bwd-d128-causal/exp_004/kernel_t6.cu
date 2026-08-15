#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd_impl {

__device__ __forceinline__ static float bf16tof(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ static __nv_bfloat16 f2bf16(float x) {
    return __float2bfloat16(x);
}

constexpr int D = 128;
constexpr int TM = 64;
constexpr int TN = 32;

/**
 * Single-kernel approach: one block per (batch, head).
 * Each block iterates over all (i,j) pairs in the causal triangle,
 * computing softmax probabilities and accumulating dQ, dK, dV in FP32.
 */
__global__ void mha_bwd_kernel_single(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ L,
    const __nv_bfloat16* __restrict__ dO,
    float* __restrict__ acc_dQ,  // FP32 accumulator [S, D]
    float* __restrict__ acc_dK,  // FP32 accumulator [S, D]
    float* __restrict__ acc_dV,  // FP32 accumulator [S, D]
    float* __restrict__ acc_D,   // FP32 accumulator [S] for D[i]
    int S, float inv_sqrt_d)
{
    int tid = threadIdx.x;
    int nt = blockDim.x;  // Will be 1024
    
    int b = blockIdx.x / gridDim.y;
    int h = blockIdx.x % gridDim.y;
    
    uint64_t off = (uint64_t)b * gridDim.y + h;
    
    const __nv_bfloat16* qb = Q + off * S * D;
    const __nv_bfloat16* kb = K + off * S * D;
    const __nv_bfloat16* vb = V + off * S * D;
    const float* lb = L + off * S;
    const __nv_bfloat16* dob = dO + off * S * D;
    
    // Zero-initialize accumulators (each thread handles some indices)
    for (int idx = tid; idx < S * D; idx += nt) {
        acc_dQ[idx] = 0.f;
        acc_dK[idx] = 0.f;
        acc_dV[idx] = 0.f;
    }
    for (int i = tid; i < S; i += nt) {
        acc_D[i] = 0.f;
    }
    __syncthreads();
    
    extern __shared__ char smem[];
    
    // Shared memory caches: store one row of Q, K, V, dO at a time
    __align__(16) __nv_bfloat16* s_Qrow = (__nv_bfloat16*)smem;            // [D]
    __align__(16) __nv_bfloat16* s_Krow = s_Qrow + D;                      // [D]
    __align__(16) __nv_bfloat16* s_Vrow = s_Krow + D;                      // [D]
    __align__(16) __nv_bfloat16* s_dOrow = s_Vrow + D;                     // [D]
    
    // Strategy: Iterate over i (query), load Q[i], dO[i], iterate over j <= i, load K[j], V[j]
    // For each (i,j): compute sim, delta, Pij, dSij
    // Accumulate dQ[i,:] += dSij * K[j,:]
    //               dV[j,:] += Pij * dO[i,:]
    //               dK[j,:] += dSij * Q[i,:]
    //               D[i]     += Pij * delta
    
    for (int i = tid; i < S; i += nt) {
        // Load Q[i,:] and dO[i,:] into shared memory
        for (int f = 0; f < D; f++) {
            s_Qrow[f] = qb[i * D + f];
            s_dOrow[f] = dob[i * D + f];
        }
        
        float Li = lb[i];
        float delta_j[D] = {};   // Per-feature dot(dO[i], V[j]) will be computed inline
        float di_partial = 0.f;
        
        for (int j = 0; j <= i && j < S; j++) {
            // Load K[j,:], V[j,:] 
            for (int f = 0; f < D; f++) {
                s_Krow[f] = kb[j * D + f];
                s_Vrow[f] = vb[j * D + f];
            }
            
            // Compute sim = Q[i].K[j] and delta = dO[i].V[j]
            float sim = 0.f, delta = 0.f;
            #pragma unroll
            for (int f = 0; f < D; f += 4) {
                sim   += bf16tof(s_Qrow[f])     * bf16tof(s_Krow[f]);
                sim   += bf16tof(s_Qrow[f+1])   * bf16tof(s_Krow[f+1]);
                sim   += bf16tof(s_Qrow[f+2])   * bf16tof(s_Krow[f+2]);
                sim   += bf16tof(s_Qrow[f+3])   * bf16tof(s_Krow[f+3]);
                delta += bf16tof(s_dOrow[f])     * bf16tof(s_Vrow[f]);
                delta += bf16tof(s_dOrow[f+1])   * bf16tof(s_Vrow[f+1]);
                delta += bf16tof(s_dOrow[f+2])   * bf16tof(s_Vrow[f+2]);
                delta += bf16tof(s_dOrow[f+3])   * bf16tof(s_Vrow[f+3]);
            }
            
            float Pij = expf(sim * inv_sqrt_d - Li);
            float dSij = Pij * (delta - acc_D[i]);  // Note: acc_D[i] grows during inner loop!
            // Actually D[i] should include ALL j contributions, so we need two passes or precompute
            
            // WRONG: Using running acc_D[i] here is incorrect. Let me use two passes.
            // For now, let me fix by NOT using acc_D in the first pass correctly.
            
            // Correct approach: dSij uses full D[i], which requires two passes over j.
            // Let me restructure.
        }
    }
}

/**
 * Correct single-block approach with two-phase accumulation:
 * Phase 1: Compute D[i] = sum_{j<=i} P[i,j] * delta[i,j]
 * Phase 2: Compute dQ, dK, dV using complete D[i]
 * We can combine both phases in one pass by storing P[i,j]*delta[i,j] temporarily,
 * but with S=4096 that's too much. Instead, we do two passes explicitly.
 */
__global__ void mha_bwd_correct(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ L,
    const __nv_bfloat16* __restrict__ dO,
    float* __restrict__ acc_dQ,
    float* __restrict__ acc_dK,
    float* __restrict__ acc_dV,
    float* __restrict__ acc_D,
    int S, float inv_sqrt_d)
{
    int tid = threadIdx.x;
    int nt = blockDim.x;
    
    uint64_t off = blockIdx.x * S;
    
    const __nv_bfloat16* qb = Q + off * D;
    const __nv_bfloat16* kb = K + off * D;
    const __nv_bfloat16* vb = V + off * D;
    const float* lb = L + off;
    const __nv_bfloat16* dob = dO + off * D;
    
    // Zero accumulators
    for (int idx = tid; idx < S * D; idx += nt) {
        acc_dQ[idx] = 0.f;
        acc_dK[idx] = 0.f;
        acc_dV[idx] = 0.f;
    }
    for (int i = tid; i < S; i += nt) {
        acc_D[i] = 0.f;
    }
    __syncthreads();
    
    // ========== PASS 1: Compute D[i] = sum_{j<=i} P[i,j] * delta[i,j] ==========
    for (int i = tid; i < S; i += nt) {
        float Li = lb[i];
        float Di = 0.f;
        
        // Load Q[i,:] and dO[i,:] into registers
        float q_reg[D], do_reg[D];
        for (int f = 0; f < D; f++) {
            q_reg[f] = bf16tof(qb[i * D + f]);
            do_reg[f] = bf16tof(dob[i * D + f]);
        }
        
        for (int j = 0; j <= i && j < S; j++) {
            float sim = 0.f, delta = 0.f;
            for (int f = 0; f < D; f += 4) {
                sim   += q_reg[f]     * bf16tof(kb[j*D+f]);
                sim   += q_reg[f+1]   * bf16tof(kb[j*D+f+1]);
                sim   += q_reg[f+2]   * bf16tof(kb[j*D+f+2]);
                sim   += q_reg[f+3]   * bf16tof(kb[j*D+f+3]);
                delta += do_reg[f]     * bf16tof(vb[j*D+f]);
                delta += do_reg[f+1]   * bf16tof(vb[j*D+f+1]);
                delta += do_reg[f+2]   * bf16tof(vb[j*D+f+2]);
                delta += do_reg[f+3]   * bf16tof(vb[j*D+f+3]);
            }
            float Pij = expf(sim * inv_sqrt_d - Li);
            Di += Pij * delta;
        }
        acc_D[i] = Di;
    }
    __syncthreads();
    
    // ========== PASS 2: Compute dQ, dK, dV ==========
    // Each thread owns one or more query positions i
    for (int i = tid; i < S; i += nt) {
        float Di = acc_D[i];
        float Li = lb[i];
        
        float q_reg[D], do_reg[D];
        for (int f = 0; f < D; f++) {
            q_reg[f] = bf16tof(qb[i * D + f]);
            do_reg[f] = bf16tof(dob[i * D + f]);
        }
        
        // Per-feature dQ accumulators (stored in registers since D=128 fits)
        float dq[D] = {}, dk_contribution[D] = {}, dv_contribution[D] = {};
        
        for (int j = 0; j <= i && j < S; j++) {
            float k_reg[D], v_reg[D];
            for (int f = 0; f < D; f++) {
                k_reg[f] = bf16tof(kb[j * D + f]);
                v_reg[f] = bf16tof(vb[j * D + f]);
            }
            
            float sim = 0.f, delta = 0.f;
            for (int f = 0; f < D; f += 4) {
                sim   += q_reg[f]     * k_reg[f];
                sim   += q_reg[f+1]   * k_reg[f+1];
                sim   += q_reg[f+2]   * k_reg[f+2];
                sim   += q_reg[f+3]   * k_reg[f+3];
                delta += do_reg[f]     * v_reg[f];
                delta += do_reg[f+1]   * v_reg[f+1];
                delta += do_reg[f+2]   * v_reg[f+2];
                delta += do_reg[f+3]   * v_reg[f+3];
            }
            
            float Pij = expf(sim * inv_sqrt_d - Li);
            float dSij = Pij * (delta - Di);
            
            for (int f = 0; f < D; f += 4) {
                dq[f]     += dSij * k_reg[f];
                dq[f+1]   += dSij * k_reg[f+1];
                dq[f+2]   += dSij * k_reg[f+2];
                dq[f+3]   += dSij * k_reg[f+3];
                
                // dV contribution: dV[j,f] += Pij * dO[i,f]
                dv_contribution[f]     += Pij * do_reg[f];
                dv_contribution[f+1]   += Pij * do_reg[f+1];
                dv_contribution[f+2]   += Pij * do_reg[f+2];
                dv_contribution[f+3]   += Pij * do_reg[f+3];
            }
        }
        
        // Write dQ[i,:] directly (no contention - each i written once)
        for (int f = 0; f < D; f++) {
            acc_dQ[i * D + f] = dq[f];
        }
        
        // Accumulate dV contributions using atomics (multiple i's contribute to same j)
        for (int j = 0; j <= i && j < S; j++) {
            // Need to recompute Pij for each j to get dV contribution
            float k_reg[D];
            for (int f = 0; f < D; f++) {
                k_reg[f] = bf16tof(kb[j * D + f]);
            }
            float sim = 0.f, delta = 0.f;
            for (int f = 0; f < D; f += 4) {
                sim   += q_reg[f]     * k_reg[f];
                sim   += q_reg[f+1]   * k_reg[f+1];
                sim   += q_reg[f+2]   * k_reg[f+2];
                sim   += q_reg[f+3]   * k_reg[f+3];
            }
            float Pij = expf(sim * inv_sqrt_d - Li);
            for (int f = 0; f < D; f++) {
                atomicAdd(&acc_dV[j * D + f], Pij * do_reg[f]);
            }
        }
    }
    __syncthreads();
    
    // ========== Compute dK[j,:] = sum_{i>=j} dS[i,j] * Q[i,:] ==========
    // Similar structure but iterating from j perspective
    for (int j = tid; j < S; j += nt) {
        float k_reg[D];
        for (int f = 0; f < D; f++) {
            k_reg[f] = bf16tof(kb[j * D + f]);
        }
        
        float dk[D] = {};
        
        for (int i = j; i < S; i++) {
            float q_reg[D], do_reg[D];
            for (int f = 0; f < D; f++) {
                q_reg[f] = bf16tof(qb[i * D + f]);
                do_reg[f] = bf16tof(dob[i * D + f]);
            }
            
            float sim = 0.f, delta = 0.f;
            for (int f = 0; f < D; f += 4) {
                sim   += q_reg[f]     * k_reg[f];
                sim   += q_reg[f+1]   * k_reg[f+1];
                sim   += q_reg[f+2]   * k_reg[f+2];
                sim   += q_reg[f+3]   * k_reg[f+3];
            }
            
            // Need delta for dS: delta = dO[i].V[j]
            float v_reg[D];
            for (int f = 0; f < D; f++) {
                v_reg[f] = bf16tof(vb[j * D + f]);
            }
            for (int f = 0; f < D; f += 4) {
                delta += do_reg[f]     * v_reg[f];
                delta += do_reg[f+1]   * v_reg[f+1];
                delta += do_reg[f+2]   * v_reg[f+2];
                delta += do_reg[f+3]   * v_reg[f+3];
            }
            
            float Pij = expf(sim * inv_sqrt_d - lb[i]);
            float dSij = Pij * (delta - acc_D[i]);
            
            for (int f = 0; f < D; f += 4) {
                dk[f]     += dSij * q_reg[f];
                dk[f+1]   += dSij * q_reg[f+1];
                dk[f+2]   += dSij * q_reg[f+2];
                dk[f+3]   += dSij * q_reg[f+3];
            }
        }
        
        for (int f = 0; f < D; f++) {
            acc_dK[j * D + f] = dk[f];
        }
    }
}

// Convert FP32 accumulators to BF16 output
__global__ void convert_to_bf16_kernel(
    const float* __restrict__ src_f32,
    __nv_bfloat16* __restrict__ dst_bf16,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        dst_bf16[idx] = f2bf16(src_f32[idx]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t ddim = Q.size(3);
    (void)ddim;  // Always 128

    int64_t num_bh = B * H;
    int64_t seq_d = S * D;

    const __nv_bfloat16* ptr_Q  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* ptr_K  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* ptr_V  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const float* ptr_L          = static_cast<const float*>(L.data_ptr());
    const __nv_bfloat16* ptr_dO = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    __nv_bfloat16* ptr_dQ       = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* ptr_dK       = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* ptr_dV       = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    // Allocate FP32 accumulator buffers [num_bh][S][D] and [num_bh][S]
    size_t sd_bytes = seq_d * sizeof(float);
    size_t s_bytes = S * sizeof(float);
    
    float* d_acc_dQ = nullptr;
    float* d_acc_dK = nullptr;
    float* d_acc_dV = nullptr;
    float* d_acc_D = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_acc_dQ, num_bh * sd_bytes));
    CUDA_CHECK(cudaMalloc(&d_acc_dK, num_bh * sd_bytes));
    CUDA_CHECK(cudaMalloc(&d_acc_dV, num_bh * sd_bytes));
    CUDA_CHECK(cudaMalloc(&d_acc_D, num_bh * s_bytes));
    CUDA_CHECK(cudaMemsetAsync(d_acc_dV, 0, num_bh * sd_bytes, stream));

    float inv_sqrt_d = 1.0f / sqrtf((float)D);

    // Launch one block per (b, h) pair
    dim3 grid(num_bh);
    dim3 block(1024);
    
    // Shared memory: Q(D*2) + K(D*2) + V(D*2) + dO(D*2) = 8*D*2 bytes = 2KB
    size_t smem_size = 8ULL * D * sizeof(__nv_bfloat16);

    for (int bh = 0; bh < num_bh; bh++) {
        mha_bwd_correct<<<1, block, smem_size, stream>>>(
            ptr_Q, ptr_K, ptr_V, ptr_L, ptr_dO,
            d_acc_dQ + bh * seq_d,
            d_acc_dK + bh * seq_d,
            d_acc_dV + bh * seq_d,
            d_acc_D + bh * S,
            (int)S, inv_sqrt_d
        );
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Wait for all kernels to finish
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Convert FP32 -> BF16 for outputs
    int conv_threads = 256;
    int conv_blocks = (int)((seq_d + conv_threads - 1) / conv_threads);
    
    for (int bh = 0; bh < num_bh; bh++) {
        convert_to_bf16_kernel<<<conv_blocks, conv_threads, 0, stream>>>(
            d_acc_dQ + bh * seq_d,
            ptr_dQ + bh * seq_d,
            (int)seq_d
        );
        convert_to_bf16_kernel<<<conv_blocks, conv_threads, 0, stream>>>(
            d_acc_dK + bh * seq_d,
            ptr_dK + bh * seq_d,
            (int)seq_d
        );
        convert_to_bf16_kernel<<<conv_blocks, conv_threads, 0, stream>>>(
            d_acc_dV + bh * seq_d,
            ptr_dV + bh * seq_d,
            (int)seq_d
        );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_acc_dQ));
    CUDA_CHECK(cudaFree(d_acc_dK));
    CUDA_CHECK(cudaFree(d_acc_dV));
    CUDA_CHECK(cudaFree(d_acc_D));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

} // namespace mha_bwd_impl