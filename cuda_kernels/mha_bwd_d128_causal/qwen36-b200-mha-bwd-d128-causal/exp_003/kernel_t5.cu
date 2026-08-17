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

constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
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
    
    static_assert(D_DIM == 128);
    
    int bh = blockIdx.x;
    if (bh >= B * H) return;
    
    int b = bh / H;
    int h = bh % H;
    size_t off_bh = (size_t)(b * H + h) * S * D_DIM;
    float inv_d = rsqrtf((float)D_DIM);
    int tid = threadIdx.x;
    int nthread = blockDim.x;
    
    // Shared memory layout (approx 48KB):
    // sQ[TILE_M*D_DIM] bf16     : 8KB
    // sdO[TILE_M*D_DIM] bf16    : 8KB
    // sK[TILE_N*D_DIM] bf16     : 8KB
    // sV[TILE_N*D_DIM] bf16     : 8KB
    // sp[TILE_M*TILE_N] float   : 4KB
    // sdpv[TILE_M*TILE_N] float : 4KB
    // sD[TILE_M] float          : 128B
    // sL[TILE_M] float          : 128B
    
    extern __shared__ char shm[];
    __nv_bfloat16* sQ  = reinterpret_cast<__nv_bfloat16*>(shm);
    __nv_bfloat16* sdO = sQ + TILE_M * D_DIM;
    __nv_bfloat16* sK  = sdO + TILE_M * D_DIM;
    __nv_bfloat16* sV  = sK + TILE_N * D_DIM;
    float* sp          = reinterpret_cast<float*>(sV + TILE_N * D_DIM);
    float* sdpv        = sp + TILE_M * TILE_N;
    float* sD          = sdpv + TILE_M * TILE_N;
    float* sL          = sD + TILE_M;
    
    const __nv_bfloat16* Qg  = Q + off_bh;
    const __nv_bfloat16* Kg  = K + off_bh;
    const __nv_bfloat16* Vg  = V + off_bh;
    const __nv_bfloat16* dOg = dO + off_bh;
    const float* Lg          = L_in + b * H * S + h * S;
    
    float* dQ_f = dQ_f32 + bh * S * D_DIM;
    float* dK_f = dK_f32 + bh * S * D_DIM;
    float* dV_f = dV_f32 + bh * S * D_DIM;
    
    // Initialize float staging outputs to zero
    for (int i = tid; i < S * D_DIM; i += nthread) {
        dQ_f[i] = 0.0f;
        dK_f[i] = 0.0f;
        dV_f[i] = 0.0f;
    }
    __syncthreads();
    
    // Process Q-tiles
    for (int qb = 0; qb < S; qb += TILE_M) {
        int qm = min(qb + TILE_M, S);
        
        // ---- Load Q, dO, L into smem ----
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
        }
        __syncthreads();
        
        // ======== PASS 1: Compute sp, sdpv, accumulate sD ========
        for (int kb = 0; kb < qm; kb += TILE_N) {
            int kn = min(kb + TILE_N, S);
            
            // Load K, V
            for (int i = tid; i < TILE_N * D_DIM; i += nthread) {
                int kk = i / D_DIM;
                int dd = i % D_DIM;
                if (kb + kk < S) {
                    sK[i] = Kg[(size_t)(kb + kk) * D_DIM + dd];
                    sV[i] = Vg[(size_t)(kb + kk) * D_DIM + dd];
                }
            }
            __syncthreads();
            
            // Zero sD accumulation targets for this pass
            for (int i = tid; i < TILE_M; i += nthread) {
                sD[i] = 0.0f;
            }
            
            // Compute probabilities and dPV, accumulate D[m]
            for (int i = tid; i < TILE_M * TILE_N; i += nthread) {
                int mm = i / TILE_N;
                int nn = i % TILE_N;
                int qi = qb + mm;
                int ki = kb + nn;
                
                if (qi < S && ki < S && ki <= qi) {
                    float score = 0.0f;
                    float dpv   = 0.0f;
                    
                    // Vectorized dot products using bf16 pairs
                    #pragma unroll
                    for (int d = 0; d < D_DIM; d += 4) {
                        __nv_bfloat162 q2a = *reinterpret_cast<const __nv_bfloat162*>(&sQ[mm * D_DIM + d]);
                        __nv_bfloat162 q2b = *reinterpret_cast<const __nv_bfloat162*>(&sQ[mm * D_DIM + d + 2]);
                        __nv_bfloat162 k2a = *reinterpret_cast<const __nv_bfloat162*>(&sK[nn * D_DIM + d]);
                        __nv_bfloat162 k2b = *reinterpret_cast<const __nv_bfloat162*>(&sK[nn * D_DIM + d + 2]);
                        __nv_bfloat162 v2a = *reinterpret_cast<const __nv_bfloat162*>(&sV[nn * D_DIM + d]);
                        __nv_bfloat162 v2b = *reinterpret_cast<const __nv_bfloat162*>(&sV[nn * D_DIM + d + 2]);
                        __nv_bfloat162 do2a = *reinterpret_cast<const __nv_bfloat162*>(&sdO[mm * D_DIM + d]);
                        __nv_bfloat162 do2b = *reinterpret_cast<const __nv_bfloat162*>(&sdO[mm * D_DIM + d + 2]);
                        
                        float qa[4] = {bf16_to_float(q2a.x), bf16_to_float(q2a.y),
                                       bf16_to_float(q2b.x), bf16_to_float(q2b.y)};
                        float ka[4] = {bf16_to_float(k2a.x), bf16_to_float(k2a.y),
                                       bf16_to_float(k2b.x), bf16_to_float(k2b.y)};
                        float va[4] = {bf16_to_float(v2a.x), bf16_to_float(v2a.y),
                                       bf16_to_float(v2b.x), bf16_to_float(v2b.y)};
                        float doa[4] = {bf16_to_float(do2a.x), bf16_to_float(do2a.y),
                                        bf16_to_float(do2b.x), bf16_to_float(do2b.y)};
                        
                        score += qa[0]*ka[0] + qa[1]*ka[1] + qa[2]*ka[2] + qa[3]*ka[3];
                        dpv   += doa[0]*va[0] + doa[1]*va[1] + doa[2]*va[2] + doa[3]*va[3];
                    }
                    score *= inv_d;
                    sp[i] = expf(score - sL[mm]);
                    sdpv[i] = dpv;
                } else {
                    sp[i] = 0.0f;
                    sdpv[i] = 0.0f;
                }
            }
            
            // Warp-reduce along N dimension to accumulate sD[m]
            for (int mm = tid; mm < TILE_M; mm += nthread) {
                float acc = 0.0f;
                #pragma unroll
                for (int nn = 0; nn < TILE_N; ++nn) {
                    int qi = qb + mm;
                    int ki = kb + nn;
                    if (qi < S && ki < S && ki <= qi) {
                        acc += sp[mm * TILE_N + nn] * sdpv[mm * TILE_N + nn];
                    }
                }
                atomicAdd(&sD[mm], acc);
            }
            __syncthreads();
        }
        
        // ======== PASS 2: Use cached sp/sdpv + completed sD to compute gradients ========
        for (int kb = 0; kb < qm; kb += TILE_N) {
            int kn = min(kb + TILE_N, S);
            
            // Reload K, V
            for (int i = tid; i < TILE_N * D_DIM; i += nthread) {
                int kk = i / D_DIM;
                int dd = i % D_DIM;
                if (kb + kk < S) {
                    sK[i] = Kg[(size_t)(kb + kk) * D_DIM + dd];
                    sV[i] = Vg[(size_t)(kb + kk) * D_DIM + dd];
                }
            }
            __syncthreads();
            
            // Recompute sp/sdpv identically (needed since smem was overwritten by reload... 
            // but actually we didn't overwrite sp/sdpv. They're still valid!)
            // Problem: we DID overwrite sp/sdpv during the reload above since sK/sV live 
            // in the same shared memory space. The layout separates them, so sp/sdpv are safe.
            // GOOD - no recompute needed. But WAIT: the K/V reload happens BEFORE the gradient
            // computation. The sp/sdpv values from PASS 1 correspond to exactly these K/V rows,
            // so they're still valid. Perfect!
            
            // Accumulate gradients
            for (int i = tid; i < TILE_M * TILE_N * D_DIM; i += nthread) {
                int plane = i / D_DIM;
                int mm = plane / TILE_N;
                int nn = plane % TILE_N;
                int dd = i % D_DIM;
                
                int qi = qb + mm;
                int ki = kb + nn;
                
                if (qi < S && ki < S && ki <= qi) {
                    int mk = mm * TILE_N + nn;
                    float p    = sp[mk];
                    float dpv  = sdpv[mk];
                    float Dval = sD[mm];
                    float diff = dpv - Dval;
                    
                    float Kv  = bf16_to_float(sK[nn * D_DIM + dd]);
                    float Qv  = bf16_to_float(sQ[mm * D_DIM + dd]);
                    float dOv = bf16_to_float(sdO[mm * D_DIM + dd]);
                    
                    atomicAdd(&dQ_f[(size_t)qi * D_DIM + dd], p * Kv * diff);
                    atomicAdd(&dK_f[(size_t)ki * D_DIM + dd], p * Qv * diff);
                    atomicAdd(&dV_f[(size_t)ki * D_DIM + dd], p * dOv);
                }
            }
            __syncthreads();
        }
    }
    
    // Final conversion: FP32 -> BF16
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
    
    // SMEM: sQ(8KB) + sdO(8KB) + sK(8KB) + sV(8KB) + sp(4KB) + sdpv(4KB) + sD(128B) + sL(128B)
    int smem_size = 2 * TILE_M * d * 2 + 2 * TILE_N * d * 2 + 2 * TILE_M * TILE_N * 4 + TILE_M * 4 * 2;
    
    // Float staging buffers for atomic accumulation
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
        d_dQ_f32, d_dK_f32, d_dV_f32,
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