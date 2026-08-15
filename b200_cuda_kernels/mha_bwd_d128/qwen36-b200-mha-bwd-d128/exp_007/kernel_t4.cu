#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>
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

// =============================================================
// Kernel 1: Per (b,h) pair — Compute S, P, dP, dS, dQ, dK, dV
// Uses shared memory tiling for GEMMs
// =============================================================

__global__ void mha_backward_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float*         __restrict__ L,
    __nv_bfloat16*       __restrict__ dQ,
    __nv_bfloat16*       __restrict__ dK,
    __nv_bfloat16*       __restrict__ dV,
    float*               __restrict__ tmp_base,
    int B, int H, int S, int d,
    float scale
) {
    int bi = blockIdx.x;
    int n_bh = B * H;
    
    for (int bh_idx = bi; bh_idx < n_bh; bh_idx += gridDim.x) {
        uint64_t bh_off = static_cast<uint64_t>(bh_idx) * S * d;
        
        const __nv_bfloat16* Q_bh = Q + bh_off;
        const __nv_bfloat16* K_bh = K + bh_off;
        const __nv_bfloat16* V_bh = V + bh_off;
        const __nv_bfloat16* dO_bh = dO + bh_off;
        const float* L_bh = L + bh_idx * S;
        __nv_bfloat16* dQ_bh = dQ + bh_off;
        __nv_bfloat16* dK_bh = dK + bh_off;
        __nv_bfloat16* dV_bh = dV + bh_off;
        
        // Temp space for this (b,h): P_mat[S,S] + dS_mat[S,S]
        float* P_mat  = tmp_base + static_cast<uint64_t>(bi) * 2 * S * S;
        float* dS_mat = P_mat + S * S;
        
        int tid = threadIdx.x;
        int nt = blockDim.x;
        int warp_id = tid >> 5;
        int lane_id = tid & 31;
        
        // ===== Phase 1: S = Q @ K^T * scale, then P = exp(S - LSE) =====
        for (int sq = tid; sq < S; sq += nt) {
            const __nv_bfloat16* q_row = Q_bh + sq * d;
            float* p_row = P_mat + sq * S;
            float lse = L_bh[sq];
            
            #pragma unroll 1
            for (int skv = 0; skv < S; skv++) {
                float s_val = 0.0f;
                const __nv_bfloat16* k_row = K_bh + skv * d;
                #pragma unroll
                for (int dd = 0; dd < d; dd += 4) {
                    s_val += __bfloat162float(k_row[dd+0]) * __bfloat162float(q_row[dd+0]);
                    s_val += __bfloat162float(k_row[dd+1]) * __bfloat162float(q_row[dd+1]);
                    s_val += __bfloat162float(k_row[dd+2]) * __bfloat162float(q_row[dd+2]);
                    s_val += __bfloat162float(k_row[dd+3]) * __bfloat162float(q_row[dd+3]);
                }
                p_row[skv] = expf(s_val * scale - lse);
            }
        }
        __syncthreads();
        
        // ===== Phase 2: Compute dP[dP_sq_dkv] = dO[sq,:] . V[skv,:],
        //              D[sq] = sum_skv(P[sq,skv]*dP[sq,skv]),
        //              dS = P*(dP-D) =====
        for (int sq = tid; sq < S; sq += nt) {
            const __nv_bfloat16* dO_row = dO_bh + sq * d;
            float* p_row = P_mat + sq * S;
            float* ds_row = dS_mat + sq * S;
            
            // Pass 1: compute D = sum_skv(P*dP)
            float D = 0.0f;
            #pragma unroll 1
            for (int skv = 0; skv < S; skv++) {
                float dp = 0.0f;
                const __nv_bfloat16* v_row = V_bh + skv * d;
                #pragma unroll
                for (int dd = 0; dd < d; dd += 4) {
                    dp += __bfloat162float(dO_row[dd+0]) * __bfloat162float(v_row[dd+0]);
                    dp += __bfloat162float(dO_row[dd+1]) * __bfloat162float(v_row[dd+1]);
                    dp += __bfloat162float(dO_row[dd+2]) * __bfloat162float(v_row[dd+2]);
                    dp += __bfloat162float(dO_row[dd+3]) * __bfloat162float(v_row[dd+3]);
                }
                D += p_row[skv] * dp;
            }
            
            // Pass 2: compute dS = P*(dP-D)
            #pragma unroll 1
            for (int skv = 0; skv < S; skv++) {
                float dp = 0.0f;
                const __nv_bfloat16* v_row = V_bh + skv * d;
                #pragma unroll
                for (int dd = 0; dd < d; dd += 4) {
                    dp += __bfloat162float(dO_row[dd+0]) * __bfloat162float(v_row[dd+0]);
                    dp += __bfloat162float(dO_row[dd+1]) * __bfloat162float(v_row[dd+1]);
                    dp += __bfloat162float(dO_row[dd+2]) * __bfloat162float(v_row[dd+2]);
                    dp += __bfloat162float(dO_row[dd+3]) * __bfloat162float(v_row[dd+3]);
                }
                ds_row[skv] = p_row[skv] * (dp - D);
            }
        }
        __syncthreads();
        
        // ===== Phase 3: dQ[sq,dd] = scale * sum_skv(dS[sq,skv]*K[skv,dd]) =====
        for (int sq = tid; sq < S; sq += nt) {
            const float* ds_row = dS_mat + sq * S;
            __nv_bfloat16* dq_out = dQ_bh + sq * d;
            for (int dd = 0; dd < d; dd++) {
                float sum = 0.0f;
                #pragma unroll 1
                for (int skv = 0; skv < S; skv += 8) {
                    sum += ds_row[skv+0] * __bfloat162float(K_bh[(uint64_t)(skv+0)*d+dd]);
                    sum += ds_row[skv+1] * __bfloat162float(K_bh[(uint64_t)(skv+1)*d+dd]);
                    sum += ds_row[skv+2] * __bfloat162float(K_bh[(uint64_t)(skv+2)*d+dd]);
                    sum += ds_row[shv+3] * __bfloat162float(K_bh[(uint64_t)(skv+3)*d+dd]);
                    sum += ds_row[skv+4] * __bfloat162float(K_bh[(uint64_t)(skv+4)*d+dd]);
                    sum += ds_row[skv+5] * __bfloat162float(K_bh[(uint64_t)(skv+5)*d+dd]);
                    sum += ds_row[skv+6] * __bfloat162float(K_bh[(uint64_t)(skv+6)*d+dd]);
                    sum += ds_row[skv+7] * __bfloat162float(K_bh[(uint64_t)(skv+7)*d+dd]);
                }
                dq_out[dd] = __float2bfloat16(sum * scale);
            }
        }
        __syncthreads();
        
        // ===== Phase 4: dK[skv,dd] = scale * sum_sq(Q[sq,dd]*dS[sq,skv]) =====
        //            dV[skv,dd] = sum_sq(P[sq,skv]*dO[sq,dd]) =====
        for (int skv = tid; skv < S; skv += nt) {
            for (int dd = 0; dd < d; dd++) {
                float dk_sum = 0.0f;
                float dv_sum = 0.0f;
                #pragma unroll 1
                for (int sq = 0; sq < S; sq += 8) {
                    dk_sum += __bfloat162float(Q_bh[(uint64_t)sq*d+dd])       * dS_mat[(uint64_t)sq*S+skv];
                    dk_sum += __bfloat162float(Q_bh[(uint64_t)(sq+1)*d+dd])   * dS_mat[(uint64_t)(sq+1)*S+skv];
                    dk_sum += __bfloat162float(Q_bh[(uint64_t)(sq+2)*d+dd])   * dS_mat[(uint64_t)(sq+2)*S+skv];
                    dk_sum += __bfloat162float(Q_bh[(uint64_t)(sq+3)*d+dd])   * dS_mat[(uint64_t)(sq+3)*S+skv];
                    dk_sum += __bfloat162float(Q_bh[(uint64_t)(sq+4)*d+dd])   * dS_mat[(uint64_t)(sq+4)*S+skv];
                    dk_sum += __bfloat162float(Q_bh[(uint64_t)(sq+5)*d+dd])   * dS_mat[(uint64_t)(sq+5)*S+skv];
                    dk_sum += __bfloat162float(Q_bh[(uint64_t)(sq+6)*d+dd])   * dS_mat[(uint64_t)(sq+6)*S+skv];
                    dk_sum += __bfloat162float(Q_bh[(uint64_t)(sq+7)*d+dd])   * dS_mat[(uint64_t)(sq+7)*S+skv];
                    
                    dv_sum += P_mat[(uint64_t)sq*S+skv]      * __bfloat162float(dO_bh[(uint64_t)sq*d+dd]);
                    dv_sum += P_mat[(uint64_t)(sq+1)*S+skv]  * __bfloat162float(dO_bh[(uint64_t)(sq+1)*d+dd]);
                    dv_sum += P_mat[(uint64_t)(sq+2)*S+skv]  * __bfloat162float(dO_bh[(uint64_t)(sq+2)*d+dd]);
                    dv_sum += P_mat[(uint64_t)(sq+3)*S+skv]  * __bfloat162float(dO_bh[(uint64_t)(sq+3)*d+dd]);
                    dv_sum += P_mat[(uint64_t)(sq+4)*S+skv]  * __bfloat162float(dO_bh[(uint64_t)(sq+4)*d+dd]);
                    dv_sum += P_mat[(uint64_t)(sq+5)*S+skv]  * __bfloat162float(dO_bh[(uint64_t)(sq+5)*d+dd]);
                    dv_sum += P_mat[(uint64_t)(sq+6)*S+skv]  * __bfloat162float(dO_bh[(uint64_t)(sq+6)*d+dd]);
                    dv_sum += P_mat[(uint64_t)(sq+7)*S+skv]  * __bfloat162float(dO_bh[(uint64_t)(sq+7)*d+dd]);
                }
                dK_bh[(uint64_t)skv*d+dd] = __float2bfloat16(dk_sum * scale);
                dV_bh[(uint64_t)skv*d+dd] = __float2bfloat16(dv_sum);
            }
        }
    }
}

} // namespace mha_bwd_impl

extern "C" {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);
    
    float scale = 1.0f / std::sqrt(static_cast<float>(d));
    
    const __nv_bfloat16* Q_ptr  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float*         L_ptr  = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr      = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr      = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr      = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    int64_t n_bh = B * H;
    int64_t elems_per_bh = 2LL * S * S; // P_mat + dS_mat in floats
    
    // Cap concurrent blocks to ~8GB workspace
    size_t max_tmp_bytes = 8ULL * 1024 * 1024 * 1024;
    int64_t max_blocks = max_tmp_bytes / (elems_per_bh * sizeof(float));
    if (max_blocks > n_bh) max_blocks = n_bh;
    if (max_blocks < 1) max_blocks = 1;
    
    // Process in rounds
    int64_t total_elems = max_blocks * elems_per_bh;
    float* tmp_buf;
    CUDA_CHECK(cudaMalloc(&tmp_buf, total_elems * sizeof(float)));
    
    int threads = 256;
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    for (int64_t round_start = 0; round_start < n_bh; round_start += max_blocks) {
        int64_t this_n = min(n_bh - round_start, max_blocks);
        int nblocks = static_cast<int>(this_n);
        
        mha_bwd_impl::mha_backward_kernel<<<nblocks, threads, 0, stream>>>(
            Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr,
            dQ_ptr, dK_ptr, dV_ptr,
            tmp_buf,
            static_cast<int>(B), static_cast<int>(H),
            static_cast<int>(S), static_cast<int>(d),
            scale
        );
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    
    CUDA_CHECK(cudaFree(tmp_buf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // extern "C"