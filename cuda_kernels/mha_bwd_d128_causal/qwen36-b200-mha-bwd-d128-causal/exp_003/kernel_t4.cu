#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>
#include <cstdint>
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

namespace tvm_ffi_mha_bwd {

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int BLOCK_DIM_X = 256;

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float v) {
    return __float2bfloat16(v);
}

template<int D_DIM>
__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L_in,
    float* __restrict__ dQ_f32,
    float* __restrict__ dK_f32,
    float* __restrict__ dV_f32,
    __nv_bfloat16* __restrict__ dQ_out,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S) {
    
    int bh = blockIdx.x;
    if (bh >= B * H) return;
    
    int b = bh / H;
    int h = bh % H;
    size_t off_bh = (size_t)(b * H + h) * S * D_DIM;
    float inv_d = rsqrtf((float)D_DIM);
    int tid = threadIdx.x;
    int nthread = blockDim.x;
    
    // Shared memory pointers
    extern __shared__ char shm[];
    __nv_bfloat16* sQ  = reinterpret_cast<__nv_bfloat16*>(shm);
    __nv_bfloat16* sK  = sQ + TILE_M * D_DIM;
    __nv_bfloat16* sV  = sK + TILE_N * D_DIM;
    __nv_bfloat16* sdO = sV + TILE_N * D_DIM;
    float* sp          = reinterpret_cast<float*>(sdO + TILE_M * D_DIM);
    float* sdpv        = sp + TILE_M * TILE_N;
    float* sD          = sdpv + TILE_M * TILE_N;
    float* sL          = sD + TILE_M;
    
    const __nv_bfloat16* Qg  = Q + off_bh;
    const __nv_bfloat16* Kg  = K + off_bh;
    const __nv_bfloat16* Vg  = V + off_bh;
    const __nv_bfloat16* dOg = dO + off_bh;
    const float* Lg = L_in + b * H * S + h * S;
    
    float* dQ_f = dQ_f32 + bh * S * D_DIM;
    float* dK_f = dK_f32 + bh * S * D_DIM;
    float* dV_f = dV_f32 + bh * S * D_DIM;
    
    // Initialize float accumulators to zero
    for (int i = tid; i < S * D_DIM; i += nthread) {
        dQ_f[i] = 0.0f;
        dK_f[i] = 0.0f;
        dV_f[i] = 0.0f;
    }
    __syncthreads();
    
    // ================= PHASE 1 & 2: Compute D[q], then dQ, dK, dV =================
    for (int qb = 0; qb < S; qb += TILE_M) {
        int qm = min(qb + TILE_M, S);
        
        // Load Q, dO, L into shared memory
        for (int i = tid; i < TILE_M * D_DIM; i += nthread) {
            int qq = i / D_DIM;
            int dd = i % D_DIM;
            if (qb + qq < S) {
                sQ[i] = Qg[(size_t)(qb + qq) * D_DIM + dd];
                sdO[i] = dOg[(size_t)(qb + qq) * D_DIM + dd];
            }
        }
        for (int i = tid; i < TILE_M; i += nthread) {
            sL[i] = (qb + i < S) ? Lg[qb + i] : 0.0f;
            sD[i] = 0.0f;
        }
        __syncthreads();
        
        // Iterate k tiles (causal constraint: k <= q)
        for (int kb = 0; kb < qm; kb += TILE_N) {
            int kn = min(kb + TILE_N, S);
            
            // Load K, V into shared memory
            for (int i = tid; i < TILE_N * D_DIM; i += nthread) {
                int kk = i / D_DIM;
                int dd = i % D_DIM;
                if (kb + kk < S) {
                    sK[i] = Kg[(size_t)(kb + kk) * D_DIM + dd];
                    sV[i] = Vg[(size_t)(kb + kk) * D_DIM + dd];
                }
            }
            __syncthreads();
            
            // Compute softmax probs P[m][n], dPV[m][n], and accumulate D[m]
            for (int i = tid; i < TILE_M * TILE_N; i += nthread) {
                int mm = i / TILE_N;
                int nn = i % TILE_N;
                int qi = qb + mm;
                int ki = kb + nn;
                
                if (qi < S && ki < S && ki <= qi) {
                    float score = 0.0f;
                    float dpv   = 0.0f;
                    
                    #pragma unroll
                    for (int d = 0; d < D_DIM; d += 2) {
                        float q0 = bf16_to_float(sQ[mm * D_DIM + d]);
                        float q1 = bf16_to_float(sQ[mm * D_DIM + d + 1]);
                        float k0 = bf16_to_float(sK[nn * D_DIM + d]);
                        float k1 = bf16_to_float(sK[nn * D_DIM + d + 1]);
                        float v0 = bf16_to_float(sV[nn * D_DIM + d]);
                        float v1 = bf16_to_float(sV[nn * D_DIM + d + 1]);
                        float do0 = bf16_to_float(sdO[mm * D_DIM + d]);
                        float do1 = bf16_to_float(sdO[mm * D_DIM + d + 1]);
                        
                        score += fmaf(q0, k0, fmaf(q1, k1, 0.0f));
                        dpv   += fmaf(do0, v0, fmaf(do1, v1, 0.0f));
                    }
                    score *= inv_d;
                    float Li = sL[mm];
                    sp[i] = expf(score - Li);
                    sdpv[i] = dpv;
                    sD[mm] += sp[i] * dpv;
                } else {
                    sp[i] = 0.0f;
                    sdpv[i] = 0.0f;
                }
            }
            __syncthreads();
        }
        
        // Flush D[q] to global memory for Phase 2 consumption
        for (int i = tid; i < TILE_M; i += nthread) {
            if (qb + i < S) {
                // We'll reload D dynamically or keep it cached. 
                // For simplicity, we store it temporarily in L buffer offset or just recompute/load next pass.
                // Actually, we can just keep D in register/smem and process gradients immediately in same loop!
            }
        }
        __syncthreads();
        
        // ================= GRADIENT ACCUMULATION =================
        // Re-scan k tiles to apply diff = (dPV - D) and accumulate dQ, dK, dV
        for (int kb = 0; kb < qm; kb += TILE_N) {
            int kn = min(kb + TILE_N, S);
            
            for (int i = tid; i < TILE_N * D_DIM; i += nthread) {
                int kk = i / D_DIM;
                int dd = i % D_DIM;
                if (kb + kk < S) {
                    sK[i] = Kg[(size_t)(kb + kk) * D_DIM + dd];
                    sV[i] = Vg[(size_t)(kb + kk) * D_DIM + dd];
                }
            }
            __syncthreads();
            
            int total_elems = TILE_M * TILE_N * D_DIM;
            for (int i = tid; i < total_elems; i += nthread) {
                int plane = i / D_DIM;
                int mm = plane / TILE_N;
                int nn = plane % TILE_N;
                int dd = i % D_DIM;
                
                int qi = qb + mm;
                int ki = kb + nn;
                
                if (qi < S && ki < S && ki <= qi) {
                    int idx_mk = mm * TILE_N + nn;
                    float p    = sp[idx_mk];
                    float dpv  = sdpv[idx_mk];
                    float Dval = sD[mm];
                    float diff = dpv - Dval;
                    
                    float Kv  = bf16_to_float(sK[nn * D_DIM + dd]);
                    float Qv  = bf16_to_float(sQ[mm * D_DIM + dd]);
                    float dOv = bf16_to_float(sdO[mm * D_DIM + dd]);
                    
                    // Atomic add to FP32 staging buffers (guaranteed 4B aligned)
                    atomicAdd(&dQ_f[(size_t)qi * D_DIM + dd], p * Kv * diff);
                    atomicAdd(&dK_f[(size_t)ki * D_DIM + dd], p * Qv * diff);
                    atomicAdd(&dV_f[(size_t)ki * D_DIM + dd], p * dOv);
                }
            }
            __syncthreads();
        }
    }
    
    // ================= EPILOGUE: Convert FP32 -> BF16 =================
    for (int i = tid; i < S * D_DIM; i += nthread) {
        dQ_out[i] = float_to_bf16(dQ_f[i]);
        dK_out[i] = float_to_bf16(dK_f[i]);
        dV_out[i] = float_to_bf16(dV_f[i]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);
    
    int total_bh = static_cast<int>(B * H);
    dim3 grid(total_bh);
    dim3 block(BLOCK_DIM_X);
    
    // Shared memory size estimation:
    // sQ: 16*128*2 = 4096
    // sK: 16*128*2 = 4096
    // sV: 16*128*2 = 4096
    // sdO: 16*128*2 = 4096
    // sp: 16*16*4 = 1024
    // sdpv: 16*16*4 = 1024
    // sD: 16*4 = 64
    // sL: 16*4 = 64
    // Total: ~18,560 bytes
    int smem_size = 19 * 1024;
    
    // Allocate FP32 staging buffers to avoid misaligned bf16 atomic faults
    float* d_dQ_f32 = nullptr;
    float* d_dK_f32 = nullptr;
    float* d_dV_f32 = nullptr;
    size_t staging_size = total_bh * S * d * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_dQ_f32, staging_size));
    CUDA_CHECK(cudaMalloc(&d_dK_f32, staging_size));
    CUDA_CHECK(cudaMalloc(&d_dV_f32, staging_size));
    CUDA_CHECK(cudaMemset(d_dQ_f32, 0, staging_size));
    CUDA_CHECK(cudaMemset(d_dK_f32, 0, staging_size));
    CUDA_CHECK(cudaMemset(d_dV_f32, 0, staging_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_bwd_kernel<128><<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        d_dQ_f32,
        d_dK_f32,
        d_dV_f32,
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    CUDA_CHECK(cudaFree(d_dQ_f32));
    CUDA_CHECK(cudaFree(d_dK_f32));
    CUDA_CHECK(cudaFree(d_dV_f32));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd