#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <algorithm>
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

// Tiled MHA backward using shared memory staging
// BLOCK_DIM threads per (batch,head), tile over S dimension
constexpr int TQ = 8;   // query tile height
constexpr int TK = 32;  // key tile width
constexpr int BLOCK_DIM = 128;

__global__ void mha_backward_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ L,
    const __nv_bfloat16* __restrict__ dO,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int d) 
{
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    const int bi = bh / H;
    const int hi = bh % H;
    
    const float scale = rsqrtf(static_cast<float>(d));
    
    const uint64_t bh_off = static_cast<uint64_t>(bh) * S * d;
    const uint64_t l_off = static_cast<uint64_t>(bi) * H * S + static_cast<uint64_t>(hi) * S;
    
    const __nv_bfloat16* Q_bh = Q + bh_off;
    const __nv_bfloat16* K_bh = K + bh_off;
    const __nv_bfloat16* V_bh = V + bh_off;
    const __nv_bfloat16* dO_bh = dO + bh_off;
    const float* L_bh = L + l_off;
    __nv_bfloat16* dQ_bh = dQ + bh_off;
    __nv_bfloat16* dK_bh = dK + bh_off;
    __nv_bfloat16* dV_bh = dV + bh_off;
    
    const int di = tid;  // This thread's d-element index
    
    // ===== Shared Memory Layout =====
    // K_tile[TK][d], V_tile[TK][d] staged in shared memory
    // But d=128 -> TK*d*2 bytes each = 8KB each. Total ~16KB for K,V tiles.
    extern __shared__ char smem[];
    __nv_bfloat16* K_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* V_smem = K_smem + TK * d;
    
    // ===================== Phase 1: Compute dQ and dV =====================
    // Process S in chunks of TQ queries at a time
    for (int qb = 0; qb < S; qb += TQ) {
        // Load Q, dO, L for TQ rows into registers
        float Q_r[TQ];
        float dO_r[TQ];
        float L_r[TQ];
        
        #pragma unroll
        for (int tt = 0; tt < TQ; ++tt) {
            const int qq = qb + tt;
            if (qq < S) {
                Q_r[tt] = __bfloat162float(Q_bh[qq * d + di]);
                dO_r[tt] = __bfloat162float(dO_bh[qq * d + di]);
                L_r[tt] = L_bh[qq];
            } else {
                Q_r[tt] = 0.f;
                dO_r[tt] = 0.f;
                L_r[tt] = 0.f;
            }
        }
        
        // Accumulators
        float corr[TQ] = {};
        float dq_a[TQ] = {};
        float dq_b[TQ] = {};
        
        // Iterate over KV tiles using shared memory staging
        for (int kt = 0; kt < S; kt += TK) {
            const int k_end = min(kt + TK, S);
            const int tk = k_end - kt;
            
            // Cooperative load: each thread loads (tk / BLOCK_DIM) elements along d-dimension
            // Actually since d == BLOCK_DIM, each thread loads tk elements sequentially
            // Thread di loads K_smem[kk][di] for kk in [kt, kt+tk)
            {
                int s_offset = tid;
                for (int vv = 0; vv < tk; ++vv) {
                    const int kk = kt + vv;
                    if (kk < S) {
                        K_smem[vv * d + tid] = K_bh[kk * d + tid];
                        V_smem[vv * d + tid] = V_bh[kk * d + tid];
                    }
                }
            }
            __syncthreads();
            
            // Now read from shared memory
            #pragma unroll
            for (int tt = 0; tt < TQ; ++tt) {
                if (qb + tt >= S) continue;
                
                const float qv = Q_r[tt];
                const float dov = dO_r[tt];
                const float lv = L_r[tt];
                
                float c_add = 0.f;
                float da_add = 0.f;
                float db_add = 0.f;
                
                #pragma unroll
                for (int vv = 0; vv < tk; ++vv) {
                    const float kv = __bfloat162float(K_smem[vv * d + di]);
                    const float vval = __bfloat162float(V_smem[vv * d + di]);
                    const float score = qv * kv * scale;
                    const float p_val = expf(score - lv);
                    const float ds = dov * vval;
                    
                    c_add += p_val * ds;
                    da_add += p_val * ds * kv;
                    db_add += p_val * kv;
                }
                corr[tt] += c_add;
                dq_a[tt] += da_add;
                dq_b[tt] += db_add;
            }
            __syncthreads();
        }
        
        // Write dQ
        #pragma unroll
        for (int tt = 0; tt < TQ; ++tt) {
            if (qb + tt >= S) continue;
            const float dq = dq_a[tt] - corr[tt] * dq_b[tt];
            dQ_bh[(qb + tt) * d + di] = __float2bfloat16(dq);
        }
    }
    
    // ===================== Phase 2: Compute dV =====================
    // dV[k][di] = sum_q P[q][k] * dO[q][di]
    for (int kt = 0; kt < S; kt += TK) {
        const int k_end = min(kt + TK, S);
        const int tk = k_end - kt;
        
        // Initialize dV accumulators in local regs for this tile
        float dv_acc_local[TK] = {};
        
        // Process all query positions
        for (int qb = 0; qb < S; qb += TQ) {
            float Q_r[TQ];
            float dO_r[TQ];
            float L_r[TQ];
            
            #pragma unroll
            for (int tt = 0; tt < TQ; ++tt) {
                const int qq = qb + tt;
                if (qq < S) {
                    Q_r[tt] = __bfloat162float(Q_bh[qq * d + di]);
                    dO_r[tt] = __bfloat162float(dO_bh[qq * d + di]);
                    L_r[tt] = L_bh[qq];
                } else {
                    Q_r[tt] = 0.f;
                    dO_r[tt] = 0.f;
                    L_r[tt] = 0.f;
                }
            }
            
            // Load K tile once more (could optimize but keeping simple)
            for (int vv = 0; vv < tk; ++vv) {
                const int kk = kt + vv;
                if (kk < S) {
                    K_smem[vv * d + tid] = K_bh[kk * d + tid];
                }
            }
            __syncthreads();
            
            // Accumulate dV contributions
            #pragma unroll
            for (int vv = 0; vv < tk; ++vv) {
                const float kv = __bfloat162float(K_smem[vv * d + di]);
                float dv_partial = 0.f;
                
                #pragma unroll
                for (int tt = 0; tt < TQ; ++tt) {
                    if (qb + tt >= S) continue;
                    const float qv = Q_r[tt];
                    const float dov = dO_r[tt];
                    const float lv = L_r[tt];
                    const float score = qv * kv * scale;
                    const float p_val = expf(score - lv);
                    dv_partial += p_val * dov;
                }
                dv_acc_local[vv] += dv_partial;
            }
            __syncthreads();
        }
        
        // Write dV to global memory
        #pragma unroll
        for (int vv = 0; vv < tk; ++vv) {
            const int kk = kt + vv;
            dV_bh[kk * d + di] = __float2bfloat16(dv_acc_local[vv]);
        }
    }
    
    // ===================== Phase 3: Compute dK =====================
    // Two sub-phases: first cache corr[q], then compute dK
    // corr[q] = sum_k P[q][k] * dO[q][d] * V[k][d] summed over all d by ALL threads
    // But each thread only sees its own d-element! So we need cross-thread reduction.
    // For correctness with single-block-per-head: use shared memory for corr cache.
    
    // Allocate corr_cache[S] in additional shared memory
    // Since we already used some smem for K,V tiles, let's use register-based approach
    // OR restructure: process corr[q] in query-tile fashion like dQ
    
    // Cache corr[q] values in an array allocated at start
    float* corr_cache = reinterpret_cast<float*>(V_smem + TK * d);
    
    // Zero the corr cache
    for (int i = tid; i < S; i += BLOCK_DIM) {
        corr_cache[i] = 0.f;
    }
    __syncthreads();
    
    // Compute corr[q] for all q using atomicAdd to accumulate across threads' d-parts
    for (int qb = 0; qb < S; qb += TQ) {
        float Q_r[TQ];
        float dO_r[TQ];
        float L_r[TQ];
        
        #pragma unroll
        for (int tt = 0; tt < TQ; ++tt) {
            const int qq = qb + tt;
            if (qq < S) {
                Q_r[tt] = __bfloat162float(Q_bh[qq * d + di]);
                dO_r[tt] = __bfloat162float(dO_bh[qq * d + di]);
                L_r[tt] = L_bh[qq];
            } else {
                Q_r[tt] = 0.f;
                dO_r[tt] = 0.f;
                L_r[tt] = 0.f;
            }
        }
        
        for (int kt = 0; kt < S; kt += TK) {
            const int k_end = min(kt + TK, S);
            const int tk = k_end - kt;
            
            // Load K, V tiles
            for (int vv = 0; vv < tk; ++vv) {
                const int kk = kt + vv;
                if (kk < S) {
                    K_smem[vv * d + tid] = K_bh[kk * d + tid];
                    V_smem[vv * d + tid] = V_bh[kk * d + tid];
                }
            }
            __syncthreads();
            
            #pragma unroll
            for (int tt = 0; tt < TQ; ++tt) {
                if (qb + tt >= S) continue;
                const float qv = Q_r[tt];
                const float dov = dO_r[tt];
                const float lv = L_r[tt];
                float c_add = 0.f;
                
                #pragma unroll
                for (int vv = 0; vv < tk; ++vv) {
                    const float kv = __bfloat162float(K_smem[vv * d + di]);
                    const float vval = __bfloat162float(V_smem[vv * d + di]);
                    const float score = qv * kv * scale;
                    const float p_val = expf(score - lv);
                    c_add += p_val * dov * vval;
                }
                atomicAdd(&corr_cache[qb + tt], c_add);
            }
            __syncthreads();
        }
    }
    
    // Now compute dK using cached corr
    for (int kt = 0; kt < S; kt += TK) {
        const int k_end = min(kt + TK, S);
        const int tk = k_end - kt;
        
        float dk_accum[TK] = {};
        
        for (int qb = 0; qb < S; qb += TQ) {
            float Q_r[TQ];
            float dO_r[TQ];
            float L_r[TQ];
            
            #pragma unroll
            for (int tt = 0; tt < TQ; ++tt) {
                const int qq = qb + tt;
                if (qq < S) {
                    Q_r[tt] = __bfloat162float(Q_bh[qq * d + di]);
                    dO_r[tt] = __bfloat162float(dO_bh[qq * d + di]);
                    L_r[tt] = L_bh[qq];
                } else {
                    Q_r[tt] = 0.f;
                    dO_r[tt] = 0.f;
                    L_r[tt] = 0.f;
                }
            }
            
            // Load K, V tiles
            for (int vv = 0; vv < tk; ++vv) {
                const int kk = kt + vv;
                if (kk < S) {
                    K_smem[vv * d + tid] = K_bh[kk * d + tid];
                    V_smem[vv * d + tid] = V_bh[kk * d + tid];
                }
            }
            __syncthreads();
            
            #pragma unroll
            for (int vv = 0; vv < tk; ++vv) {
                const float kv = __bfloat162float(K_smem[vv * d + di]);
                const float vval = __bfloat162float(V_smem[vv * d + di]);
                float dk_partial = 0.f;
                
                #pragma unroll
                for (int tt = 0; tt < TQ; ++tt) {
                    if (qb + tt >= S) continue;
                    const float qv = Q_r[tt];
                    const float dov = dO_r[tt];
                    const float lv = L_r[tt];
                    const float cc = corr_cache[qb + tt];
                    const float score = qv * kv * scale;
                    const float p_val = expf(score - lv);
                    const float ds = dov * vval;
                    dk_partial += p_val * (ds - cc) * qv;
                }
                dk_accum[vv] += dk_partial;
            }
            __syncthreads();
        }
        
        // Write dK
        #pragma unroll
        for (int vv = 0; vv < tk; ++vv) {
            const int kk = kt + vv;
            dK_bh[kk * d + di] = __float2bfloat16(dk_accum[vv]);
        }
    }
}

}  // namespace mha_bwd_impl

extern "C" {

void run(tvm::ffi::TensorView Q,
         tvm::ffi::TensorView K,
         tvm::ffi::TensorView V,
         tvm::ffi::TensorView O,
         tvm::ffi::TensorView dO,
         tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ,
         tvm::ffi::TensorView dK,
         tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    const int64_t B = Q.size(0);
    const int64_t H = Q.size(1);
    const int64_t S = Q.size(2);
    const int64_t d = Q.size(3);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    dim3 grid(static_cast<int>(B * H));
    dim3 block(mha_bwd_impl::BLOCK_DIM);
    
    // Shared memory: K tile + V tile + corr cache
    // K: TK*d*2, V: TK*d*2, corr: S*4
    const int smem_bytes = 2 * mha_bwd_impl::TK * static_cast<int>(d) * sizeof(__nv_bfloat16) 
                         + static_cast<int>(S) * sizeof(float);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_bwd_impl::mha_backward_kernel<<<grid, block, smem_bytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, L_ptr, dO_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        static_cast<int>(B), static_cast<int>(H),
        static_cast<int>(S), static_cast<int>(d));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // extern "C"