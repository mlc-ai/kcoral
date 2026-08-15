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

// Tile sizes for flash attention backward
constexpr int TQ = 8;  // query tile rows loaded per block step
constexpr int TK = 32; // key tile columns loaded per block step

// Each thread owns one d-element
template <int TQ, int TK>
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
    
    const uint64_t bh_off = static_cast<uint64_t>(bh) * S * d;
    const int bi = bh / H;
    const int hi = bh % H;
    const uint64_t l_off = static_cast<uint64_t>(bi) * H * S + static_cast<uint64_t>(hi) * S;
    
    const float scale = rsqrtf(static_cast<float>(d));
    
    const __nv_bfloat16* Q_bh = Q + bh_off;
    const __nv_bfloat16* K_bh = K + bh_off;
    const __nv_bfloat16* V_bh = V + bh_off;
    const __nv_bfloat16* dO_bh = dO + bh_off;
    const float* L_bh = L + l_off;
    __nv_bfloat16* dQ_bh = dQ + bh_off;
    __nv_bfloat16* dK_bh = dK + bh_off;
    __nv_bfloat16* dV_bh = dV + bh_off;
    
    const int di = tid;
    
    // ===================== Compute dQ =====================
    // dQ[q][di] = sum_k P[q][k]*(dS[q][k]-corr[q])*K[k][di]
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
        
        float corr[TQ] = {};
        float dq_a[TQ] = {};
        float dq_b[TQ] = {};
        
        for (int kt = 0; kt < S; kt += TK) {
            const int k_end = min(kt + TK, S);
            const int tk = k_end - kt;
            
            float K_r[TK];
            float V_r[TK];
            #pragma unroll
            for (int vv = 0; vv < tk; ++vv) {
                const int kk = kt + vv;
                K_r[vv] = __bfloat162float(K_bh[kk * d + di]);
                V_r[vv] = __bfloat162float(V_bh[kk * d + di]);
            }
            
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
                    const float kv = K_r[vv];
                    const float vval = V_r[vv];
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
            
            // Write dV contributions incrementally (unique per thread -> no races)
            #pragma unroll
            for (int tt = 0; tt < TQ; ++tt) {
                if (qb + tt >= S) continue;
                const float dov = dO_r[tt];
                const float qv = Q_r[tt];
                const float lv = L_r[tt];
                
                #pragma unroll
                for (int vv = 0; vv < tk; ++vv) {
                    const int kk = kt + vv;
                    const float kv = K_r[vv];
                    const float score = qv * kv * scale;
                    const float p_val = expf(score - lv);
                    
                    const int dv_idx = kk * d + di;
                    float dv_cur = __bfloat162float(dV_bh[dv_idx]);
                    dV_bh[dv_idx] = __float2bfloat16(dv_cur + p_val * dov);
                }
            }
        }
        
        // Write dQ after all KV tiles
        #pragma unroll
        for (int tt = 0; tt < TQ; ++tt) {
            if (qb + tt >= S) continue;
            const float dq = dq_a[tt] - corr[tt] * dq_b[tt];
            dQ_bh[(qb + tt) * d + di] = __float2bfloat16(dq);
        }
    }
    
    // ===================== Compute dK (two-pass) =====================
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
        
        // Pass 1: compute full corr[q]
        float corr_full[TQ] = {};
        for (int kt = 0; kt < S; kt += TK) {
            const int k_end = min(kt + TK, S);
            const int tk = k_end - kt;
            float K_r[TK];
            float V_r[TK];
            #pragma unroll
            for (int vv = 0; vv < tk; ++vv) {
                const int kk = kt + vv;
                K_r[vv] = __bfloat162float(K_bh[kk * d + di]);
                V_r[vv] = __bfloat162float(V_bh[kk * d + di]);
            }
            #pragma unroll
            for (int tt = 0; tt < TQ; ++tt) {
                if (qb + tt >= S) continue;
                const float qv = Q_r[tt];
                const float dov = dO_r[tt];
                const float lv = L_r[tt];
                float c_add = 0.f;
                #pragma unroll
                for (int vv = 0; vv < tk; ++vv) {
                    const float kv = K_r[vv];
                    const float vval = V_r[vv];
                    const float score = qv * kv * scale;
                    const float p_val = expf(score - lv);
                    c_add += p_val * dov * vval;
                }
                corr_full[tt] += c_add;
            }
        }
        
        // Pass 2: compute dK using corr_full
        for (int kt = 0; kt < S; kt += TK) {
            const int k_end = min(kt + TK, S);
            const int tk = k_end - kt;
            float K_r[TK];
            float V_r[TK];
            #pragma unroll
            for (int vv = 0; vv < tk; ++vv) {
                const int kk = kt + vv;
                K_r[vv] = __bfloat162float(K_bh[kk * d + di]);
                V_r[vv] = __bfloat162float(V_bh[kk * d + di]);
            }
            
            #pragma unroll
            for (int vv = 0; vv < tk; ++vv) {
                const int kk = kt + vv;
                const float kv = K_r[vv];
                const float vval = V_r[vv];
                float dk_val = 0.f;
                
                #pragma unroll
                for (int tt = 0; tt < TQ; ++tt) {
                    if (qb + tt >= S) continue;
                    const float qv = Q_r[tt];
                    const float dov = dO_r[tt];
                    const float lv = L_r[tt];
                    const float score = qv * kv * scale;
                    const float p_val = expf(score - lv);
                    const float ds = dov * vval;
                    dk_val += p_val * (ds - corr_full[tt]) * qv;
                }
                
                float dk_cur = __bfloat162float(dK_bh[kk * d + di]);
                dK_bh[kk * d + di] = __float2bfloat16(dk_cur + dk_val);
            }
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
    dim3 block(static_cast<int>(d));
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_bwd_impl::mha_backward_kernel<mha_bwd_impl::TQ, mha_bwd_impl::TK><<<grid, block, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, L_ptr, dO_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        static_cast<int>(B), static_cast<int>(H),
        static_cast<int>(S), static_cast<int>(d));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // extern "C"