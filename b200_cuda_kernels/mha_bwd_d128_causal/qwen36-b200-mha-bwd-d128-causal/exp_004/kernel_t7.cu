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
 * Tile-based kernel: one block per (batch, head).
 * Shared memory holds Q tile [TM][D], K row [D], V row [D], dO tile [TM][D].
 * We process in (M-tile, N-tile) iterations, accumulating into FP32 global buffers.
 */
__global__ void mha_bwd_kernel(
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
    
    extern __shared__ char smem[];
    
    // Layout: s_Q[TM][D], s_dO[TM][D], s_K_row[D], s_V_row[D]
    __align__(16) float* sf_Q = reinterpret_cast<float*>(smem);           // [TM*D] fp32
    __align__(16) float* sf_dO = sf_Q + TM * D;                          // [TM*D] fp32
    __align__(16) float* sf_K = sf_dO + TM * D;                          // [D] fp32
    __align__(16) float* sf_V = sf_K + D;                                // [D] fp32
    
    // ========== PHASE 1: Compute D[i] for all i ==========
    for (int bm = 0; bm < S; bm += TM) {
        int bm_clamped = min(bm + TM, S);
        
        // Load Q[bm:bm+TM, :] and dO[bm:bm+TM, :] into shared mem as fp32
        for (int idx = tid; idx < (bm_clamped - bm) * D; idx += nt) {
            int i = idx / D;
            int f = idx % D;
            sf_Q[idx] = bf16tof(qb[(bm + i) * D + f]);
            sf_dO[idx] = bf16tof(dob[(bm + i) * D + f]);
        }
        // Zero-pad if partial tile
        for (int idx = (bm_clamped - bm) * D; idx < TM * D; idx += nt) {
            sf_Q[idx] = 0.f;
            sf_dO[idx] = 0.f;
        }
        __syncthreads();
        
        // Each thread computes D[i] for its assigned i in this tile
        for (int i_local = tid; i_local < (bm_clamped - bm); i_local += nt) {
            int gi = bm + i_local;
            if (gi >= S) continue;
            
            float Li = lb[gi];
            float Di = 0.f;
            
            // Iterate over all j <= gi
            for (int j = 0; j <= gi && j < S; j++) {
                // Dot products directly from global mem for K[j,:], V[j,:]
                float sim = 0.f, delta = 0.f;
                #pragma unroll
                for (int f = 0; f < D; f += 4) {
                    sim   += sf_Q[i_local * D + f]     * bf16tof(kb[j * D + f]);
                    sim   += sf_Q[i_local * D + f + 1] * bf16tof(kb[j * D + f + 1]);
                    sim   += sf_Q[i_local * D + f + 2] * bf16tof(kb[j * D + f + 2]);
                    sim   += sf_Q[i_local * D + f + 3] * bf16tof(kb[j * D + f + 3]);
                    delta += sf_dO[i_local * D + f]     * bf16tof(vb[j * D + f]);
                    delta += sf_dO[i_local * D + f + 1] * bf16tof(vb[j * D + f + 1]);
                    delta += sf_dO[i_local * D + f + 2] * bf16tof(vb[j * D + f + 2]);
                    delta += sf_dO[i_local * D + f + 3] * bf16tof(vb[j * D + f + 3]);
                }
                Di += expf(sim * inv_sqrt_d - Li) * delta;
            }
            acc_D[gi] = Di;
        }
        __syncthreads();
    }
    
    // ========== PHASE 2: Compute dQ, dK, dV ==========
    for (int bm = 0; bm < S; bm += TM) {
        int bm_end = min(bm + TM, S);
        
        // Load Q and dO tile into shared mem as fp32
        for (int idx = tid; idx < (bm_end - bm) * D; idx += nt) {
            int i = idx / D;
            int f = idx % D;
            sf_Q[idx] = bf16tof(qb[(bm + i) * D + f]);
            sf_dO[idx] = bf16tof(dob[(bm + i) * D + f]);
        }
        for (int idx = (bm_end - bm) * D; idx < TM * D; idx += nt) {
            sf_Q[idx] = 0.f;
            sf_dO[idx] = 0.f;
        }
        __syncthreads();
        
        // Each thread computes dQ for its assigned query rows
        for (int i_local = tid; i_local < (bm_end - bm); i_local += nt) {
            int gi = bm + i_local;
            if (gi >= S) continue;
            
            float Di = acc_D[gi];
            float Li = lb[gi];
            
            // Accumulate dQ[gi,:] = sum_{j<=gi} dSij * K[j,:]
            float dq_accum[D] = {};
            
            for (int j = 0; j <= gi && j < S; j++) {
                float sim = 0.f, delta = 0.f;
                #pragma unroll
                for (int f = 0; f < D; f += 4) {
                    sim   += sf_Q[i_local * D + f]     * bf16tof(kb[j * D + f]);
                    sim   += sf_Q[i_local * D + f + 1] * bf16tof(kb[j * D + f + 1]);
                    sim   += sf_Q[i_local * D + f + 2] * bf16tof(kb[j * D + f + 2]);
                    sim   += sf_Q[i_local * D + f + 3] * bf16tof(kb[j * D + f + 3]);
                    delta += sf_dO[i_local * D + f]     * bf16tof(vb[j * D + f]);
                    delta += sf_dO[i_local * D + f + 1] * bf16tof(vb[j * D + f + 1]);
                    delta += sf_dO[i_local * D + f + 2] * bf16tof(vb[j * D + f + 2]);
                    delta += sf_dO[i_local * D + f + 3] * bf16tof(vb[j * D + f + 3]);
                }
                
                float Pij = expf(sim * inv_sqrt_d - Li);
                float dSij = Pij * (delta - Di);
                
                #pragma unroll
                for (int f = 0; f < D; f += 4) {
                    dq_accum[f]     += dSij * bf16tof(kb[j * D + f]);
                    dq_accum[f + 1] += dSij * bf16tof(kb[j * D + f + 1]);
                    dq_accum[f + 2] += dSij * bf16tof(kb[j * D + f + 2]);
                    dq_accum[f + 3] += dSij * bf16tof(kb[j * D + f + 3]);
                    
                    // Also accumulate dV[j,f] via atomicAdd
                    atomicAdd(&acc_dV[j * D + f],     Pij * sf_dO[i_local * D + f]);
                    atomicAdd(&acc_dV[j * D + f + 1], Pij * sf_dO[i_local * D + f + 1]);
                    atomicAdd(&acc_dV[j * D + f + 2], Pij * sf_dO[i_local * D + f + 2]);
                    atomicAdd(&acc_dV[j * D + f + 3], Pij * sf_dO[i_local * D + f + 3]);
                }
            }
            
            // Write dQ (each gi written exactly once, no contention)
            for (int f = 0; f < D; f++) {
                acc_dQ[gi * D + f] = dq_accum[f];
            }
        }
        __syncthreads();
    }
    
    // ========== PHASE 3: Compute dK ==========
    // dK[j,:] = sum_{i>=j} dS[i,j] * Q[i,:]
    // Process j in batches
    for (int bn = 0; bn < S; bn += TN) {
        int bn_end = min(bn + TN, S);
        
        // Load K tile into shared mem as fp32
        for (int idx = tid; idx < (bn_end - bn) * D; idx += nt) {
            int j = idx / D;
            int f = idx % D;
            sf_K[idx] = bf16tof(kb[(bn + j) * D + f]);
        }
        for (int idx = (bn_end - bn) * D; idx < TN * D; idx += nt) {
            sf_K[idx] = 0.f;
        }
        __syncthreads();
        
        for (int j_local = tid; j_local < (bn_end - bn); j_local += nt) {
            int gj = bn + j_local;
            if (gj >= S) continue;
            
            float dk_accum[D] = {};
            
            for (int i = gj; i < S; i++) {
                float sim = 0.f, delta = 0.f;
                #pragma unroll
                for (int f = 0; f < D; f += 4) {
                    sim   += bf16tof(qb[i * D + f])     * sf_K[j_local * D + f];
                    sim   += bf16tof(qb[i * D + f + 1]) * sf_K[j_local * D + f + 1];
                    sim   += bf16tof(qb[i * D + f + 2]) * sf_K[j_local * D + f + 2];
                    sim   += bf16tof(qb[i * D + f + 3]) * sf_K[j_local * D + f + 3];
                    delta += bf16tof(dob[i * D + f])     * bf16tof(vb[gj * D + f]);
                    delta += bf16tof(dob[i * D + f + 1]) * bf16tof(vb[gj * D + f + 1]);
                    delta += bf16tof(dob[i * D + f + 2]) * bf16tof(vb[gj * D + f + 2]);
                    delta += bf16tof(dob[i * D + f + 3]) * bf16tof(vb[gj * D + f + 3]);
                }
                
                float Pij = expf(sim * inv_sqrt_d - lb[i]);
                float dSij = Pij * (delta - acc_D[i]);
                
                #pragma unroll
                for (int f = 0; f < D; f += 4) {
                    dk_accum[f]     += dSij * bf16tof(qb[i * D + f]);
                    dk_accum[f + 1] += dSij * bf16tof(qb[i * D + f + 1]);
                    dk_accum[f + 2] += dSij * bf16tof(qb[i * D + f + 2]);
                    dk_accum[f + 3] += dSij * bf16tof(qb[i * D + f + 3]);
                }
            }
            
            for (int f = 0; f < D; f++) {
                acc_dK[gj * D + f] = dk_accum[f];
            }
        }
        __syncthreads();
    }
}

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
    (void)Q.size(3); // Always 128

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

    // Allocate FP32 accumulator buffers
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
    
    float inv_sqrt_d = 1.0f / sqrtf((float)D);

    dim3 grid(num_bh);
    dim3 block(1024);
    
    // Shared memory layout:
    // sf_Q: TM*D floats = 64*128*4 = 32KB
    // sf_dO: TM*D floats = 32KB  
    // sf_K: max(TM,TN)*D floats = 64*128*4 = 32KB
    // sf_V: unused now but allocated = 32KB
    // Total = 128KB which is within limits
    size_t smem_size = (2ULL * TM + 2ULL * TN) * D * sizeof(float);

    mha_bwd_kernel<<<grid, block, smem_size, stream>>>(
        ptr_Q, ptr_K, ptr_V, ptr_L, ptr_dO,
        d_acc_dQ, d_acc_dK, d_acc_dV, d_acc_D,
        (int)S, inv_sqrt_d
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Convert FP32 -> BF16
    int conv_threads = 256;
    int conv_blocks = (int)((seq_d + conv_threads - 1) / conv_threads);
    int total_conv = num_bh * conv_blocks;
    
    // Launch one large grid converting all blocks
    convert_to_bf16_kernel<<<conv_blocks * num_bh, conv_threads, 0, stream>>>(
        d_acc_dQ, ptr_dQ, (int)(num_bh * seq_d)
    );
    CUDA_CHECK(cudaGetLastError());
    
    convert_to_bf16_kernel<<<conv_blocks * num_bh, conv_threads, 0, stream>>>(
        d_acc_dK, ptr_dK, (int)(num_bh * seq_d)
    );
    CUDA_CHECK(cudaGetLastError());
    
    convert_to_bf16_kernel<<<conv_blocks * num_bh, conv_threads, 0, stream>>>(
        d_acc_dV, ptr_dV, (int)(num_bh * seq_d)
    );
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_acc_dQ));
    CUDA_CHECK(cudaFree(d_acc_dK));
    CUDA_CHECK(cudaFree(d_acc_dV));
    CUDA_CHECK(cudaFree(d_acc_D));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

} // namespace mha_bwd_impl