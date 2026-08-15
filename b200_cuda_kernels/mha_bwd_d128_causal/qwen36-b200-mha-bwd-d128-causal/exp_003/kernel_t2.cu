#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>
#include <cstdint>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/runtime.h>

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
constexpr int BLOCK_DIM_X = 128;

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float v) {
    return __float2bfloat16(v);
}

// Shared memory layout for tiled computation
struct SMEMLayout {
    __nv_bfloat16 sQ[TILE_M][128];       // Q tile: [TILE_M][d_dim]
    __nv_bfloat16 sK[TILE_N][128];       // K tile: [TILE_N][d_dim]
    __nv_bfloat16 sV[TILE_N][128];       // V tile: [TILE_N][d_dim]
    __nv_bfloat16 sdO[TILE_M][128];      // dO tile: [TILE_M][d_dim]
    float sL[TILE_M];                    // logsumexp for Q rows
    float sD[TILE_M];                    // D[q] accumulators
    float sp[TILE_M][TILE_N];            // softmax probs p[m][n]
    float sdpv[TILE_M][TILE_N];          // dPV = dO dot V values
};

template<int D_DIM>
__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L_in,
    __nv_bfloat16* __restrict__ dQ_out,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S) {
    
    static_assert(D_DIM == 128, "Only D=128 supported in this template");
    
    int bh = blockIdx.x;
    if (bh >= B * H) return;
    
    int b = bh / H;
    int h = bh % H;
    size_t off_bh = (size_t)(b * H + h) * S * D_DIM;
    float inv_d = rsqrtf((float)D_DIM);
    
    extern __shared__ char shm[];
    SMEMLayout& sm = *reinterpret_cast<SMEMLayout*>(shm);
    
    const __nv_bfloat16* Qg = Q + off_bh;
    const __nv_bfloat16* Kg = K + off_bh;
    const __nv_bfloat16* Vg = V + off_bh;
    const __nv_bfloat16* dOg = dO + off_bh;
    const float* Lg = L_in + b * H * S + h * S;
    __nv_bfloat16* dQg = dQ_out + off_bh;
    __nv_bfloat16* dKg = dK_out + off_bh;
    __nv_bfloat16* dVg = dV_out + off_bh;
    
    int tid = threadIdx.x;
    int nthread = blockDim.x;
    
    // Initialize outputs to zero
    for (int i = tid; i < S * D_DIM; i += nthread) {
        dQg[i] = __nv_bfloat16();
        dKg[i] = __nv_bfloat16();
        dVg[i] = __nv_bfloat16();
    }
    __syncthreads();
    
    // Phase 1: Compute dV (doesn't depend on D[q])
    // dV[ki][dd] += sum_q p[qi][ki] * dO[qi][dd]
    // Process in q-tiles
    for (int qb = 0; qb < S; qb += TILE_M) {
        int qm = min(qb + TILE_M, S);
        
        // Load Q, dO, L into shared memory
        for (int i = tid; i < TILE_M * D_DIM; i += nthread) {
            int qq = i / D_DIM;
            int dd = i % D_DIM;
            if (qb + qq < S) {
                sm.sQ[qq][dd] = Qg[(size_t)(qb + qq) * D_DIM + dd];
                sm.sdO[qq][dd] = dOg[(size_t)(qb + qq) * D_DIM + dd];
            }
        }
        for (int i = tid; i < TILE_M; i += nthread) {
            if (qb + i < S) sm.sL[i] = Lg[qb + i];
        }
        __syncthreads();
        
        // Process k-tiles for causal region
        for (int kb = 0; kb < qm; kb += TILE_N) {
            int kn = min(kb + TILE_N, S);
            
            // Load K, V
            for (int i = tid; i < TILE_N * D_DIM; i += nthread) {
                int kk = i / D_DIM;
                int dd = i % D_DIM;
                if (kb + kk < S) {
                    sm.sK[kk][dd] = Kg[(size_t)(kb + kk) * D_DIM + dd];
                    sm.sV[kk][dd] = Vg[(size_t)(kb + kk) * D_DIM + dd];
                }
            }
            __syncthreads();
            
            // Compute softmax probs p[m][n] and dPV[m][n] = dO.dot(V)
            for (int i = tid; i < TILE_M * TILE_N; i += nthread) {
                int mm = i / TILE_N;
                int nn = i % TILE_N;
                int qi = qb + mm;
                int ki = kb + nn;
                
                if (qi < S && ki < S && ki <= qi) {
                    // Dot product Q[mm].K[nn]
                    float score = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < D_DIM; d += 2) {
                        score += bf16_to_float(sm.sQ[mm][d])     * bf16_to_float(sm.sK[nn][d]);
                        score += bf16_to_float(sm.sQ[mm][d + 1]) * bf16_to_float(sm.sK[nn][d + 1]);
                    }
                    score *= inv_d;
                    
                    float Li = sm.sL[mm];
                    sm.sp[mm][nn] = expf(score - Li);
                    
                    // dO[mm].dot.V[nn]
                    float dpv = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < D_DIM; d += 2) {
                        dpv += bf16_to_float(sm.sdO[mm][d])     * bf16_to_float(sm.sV[nn][d]);
                        dpv += bf16_to_float(sm.sdO[mm][d + 1]) * bf16_to_float(sm.sV[nn][d + 1]);
                    }
                    sm.sdpv[mm][nn] = dpv;
                } else {
                    sm.sp[mm][nn] = 0.0f;
                    sm.sdpv[mm][nn] = 0.0f;
                }
            }
            __syncthreads();
            
            // Accumulate dV[ki][dd] += p[mm][nn] * dO[mm][dd]
            for (int i = tid; i < TILE_M * TILE_N * D_DIM; i += nthread) {
                int plane = i / D_DIM;
                int mm = plane / TILE_N;
                int nn = plane % TILE_N;
                int dd = i % D_DIM;
                
                int qi = qb + mm;
                int ki = kb + nn;
                
                if (qi < S && ki < S && ki <= qi) {
                    float contrib = sm.sp[mm][nn] * bf16_to_float(sm.sdO[mm][dd]);
                    // Atomic add to global bf16 buffer via float reinterpretation
                    atomicAdd((float*)&dVg[(size_t)ki * D_DIM + dd], contrib);
                }
            }
            __syncthreads();
        }
    }
    __syncthreads();
    
    // Phase 2: Compute dQ and dK
    // dQ[qi][dd] = sum_{ki<=qi} p[qi][ki] * (dPV[qi][ki] - D[qi]) * K[ki][dd]
    // dK[ki][dd] = sum_{qi>=ki} p[qi][ki] * (dPV[qi][ki] - D[qi]) * Q[qi][dd]
    // D[qi] = sum_{ki<=qi} p[qi][ki] * dPV[qi][ki]
    
    // Process q-tiles
    for (int qb = 0; qb < S; qb += TILE_M) {
        int qm = min(qb + TILE_M, S);
        
        // Load Q, dO, L
        for (int i = tid; i < TILE_M * D_DIM; i += nthread) {
            int qq = i / D_DIM;
            int dd = i % D_DIM;
            if (qb + qq < S) {
                sm.sQ[qq][dd] = Qg[(size_t)(qb + qq) * D_DIM + dd];
                sm.sdO[qq][dd] = dOg[(size_t)(qb + qq) * D_DIM + dd];
            }
        }
        for (int i = tid; i < TILE_M; i += nthread) {
            if (qb + i < S) sm.sL[i] = Lg[qb + i];
        }
        __syncthreads();
        
        // Compute full D[mm] for each row in this q-tile
        // D[qi] = sum_{ki=0}^{qi} p[qi][ki] * dPV[qi][ki]
        for (int mm = tid; mm < TILE_M; mm += nthread) {
            int qi = qb + mm;
            if (qi >= S) {
                sm.sD[mm] = 0.0f;
                continue;
            }
            
            float Dval = 0.0f;
            float Li = sm.sL[mm];
            
            // Iterate all k <= qi
            for (int ki = 0; ki <= qi; ++ki) {
                float score = 0.0f;
                float dpv = 0.0f;
                
                #pragma unroll
                for (int d = 0; d < D_DIM; d += 2) {
                    float q0 = bf16_to_float(sm.sQ[mm][d]);
                    float q1 = bf16_to_float(sm.sQ[mm][d + 1]);
                    float k0 = bf16_to_float(Kg[(size_t)ki * D_DIM + d]);
                    float k1 = bf16_to_float(Kg[(size_t)ki * D_DIM + d + 1]);
                    float v0 = bf16_to_float(Vg[(size_t)ki * D_DIM + d]);
                    float v1 = bf16_to_float(Vg[(size_t)ki * D_DIM + d + 1]);
                    float do0 = bf16_to_float(sm.sdO[mm][d]);
                    float do1 = bf16_to_float(sm.sdO[mm][d + 1]);
                    
                    score += fmaf(q0, k0, fmaf(q1, k1, 0.0f));
                    dpv   += fmaf(do0, v0, fmaf(do1, v1, 0.0f));
                }
                score *= inv_d;
                Dval += expf(score - Li) * dpv;
            }
            sm.sD[mm] = Dval;
        }
        __syncthreads();
        
        // Now compute dQ and dK contributions for this q-tile
        for (int mm = tid; mm < TILE_M; mm += nthread) {
            int qi = qb + mm;
            if (qi >= S) continue;
            
            float Dval = sm.sD[mm];
            float Li = sm.sL[mm];
            
            // Iterate all k <= qi, compute diff = p * (dPV - D)
            // Then accumulate into dQ[qi] and dK[ki]
            float dQ_acc[D_DIM];
            for (int d = 0; d < D_DIM; d++) dQ_acc[d] = 0.0f;
            
            for (int ki = 0; ki <= qi; ++ki) {
                float score = 0.0f;
                float dpv = 0.0f;
                
                #pragma unroll
                for (int d = 0; d < D_DIM; d += 2) {
                    float q0 = bf16_to_float(sm.sQ[mm][d]);
                    float q1 = bf16_to_float(sm.sQ[mm][d + 1]);
                    float k0 = bf16_to_float(Kg[(size_t)ki * D_DIM + d]);
                    float k1 = bf16_to_float(Kg[(size_t)ki * D_DIM + d + 1]);
                    float v0 = bf16_to_float(Vg[(size_t)ki * D_DIM + d]);
                    float v1 = bf16_to_float(Vg[(size_t)ki * D_DIM + d + 1]);
                    float do0 = bf16_to_float(sm.sdO[mm][d]);
                    float do1 = bf16_to_float(sm.sdO[mm][d + 1]);
                    
                    score += fmaf(q0, k0, fmaf(q1, k1, 0.0f));
                    dpv   += fmaf(do0, v0, fmaf(do1, v1, 0.0f));
                }
                score *= inv_d;
                float p = expf(score - Li);
                float diff = p * (dpv - Dval);
                
                // Accumulate dQ[qi][d] += diff * K[ki][d]
                #pragma unroll
                for (int d = 0; d < D_DIM; d += 2) {
                    float k0 = bf16_to_float(Kg[(size_t)ki * D_DIM + d]);
                    float k1 = bf16_to_float(Kg[(size_t)ki * D_DIM + d + 1]);
                    dQ_acc[d]     += diff * k0;
                    dQ_acc[d + 1] += diff * k1;
                    
                    // dK[ki][d] += diff * Q[qi][d]
                    float q0 = bf16_to_float(sm.sQ[mm][d]);
                    float q1 = bf16_to_float(sm.sQ[mm][d + 1]);
                    atomicAdd((float*)&dKg[(size_t)ki * D_DIM + d],     diff * q0);
                    atomicAdd((float*)&dKg[(size_t)ki * D_DIM + d + 1], diff * q1);
                }
            }
            
            // Write dQ for this row
            #pragma unroll
            for (int d = 0; d < D_DIM; d += 2) {
                dQg[(size_t)qi * D_DIM + d]     = float_to_bf16(dQ_acc[d]);
                dQg[(size_t)qi * D_DIM + d + 1] = float_to_bf16(dQ_acc[d + 1]);
            }
        }
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
    
    // Shared memory: SMEMLayout
    // sQ: 16*128*2 = 4096, sK: 16*128*2 = 4096, sV: same, sdO: same,
    // sL: 16*4 = 64, sD: 64, sp: 16*16*4 = 1024, sdpv: 1024
    // Total ~ 14 KB
    int smem_size = sizeof(SMEMLayout);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_bwd_kernel<128><<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd