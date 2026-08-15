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
    float scale, int elems_per_bh, int n_rounds
) {
    // Each block handles ONE (b,h) pair per round
    int bi = blockIdx.x;
    if (bi >= B * H) return;
    
    uint64_t bh_off = static_cast<uint64_t>(bi) * S * d;
    
    const __nv_bfloat16* Q_bh = Q + bh_off;
    const __nv_bfloat16* K_bh = K + bh_off;
    const __nv_bfloat16* V_bh = V + bh_off;
    const __nv_bfloat16* dO_bh = dO + bh_off;
    const float* L_bh = L + bi * S;
    __nv_bfloat16* dQ_bh = dQ + bh_off;
    __nv_bfloat16* dK_bh = dK + bh_off;
    __nv_bfloat16* dV_bh = dV + bh_off;
    
    float* P_mat  = tmp_base + bi * elems_per_bh;
    float* dS_mat = P_mat + S * S;
    
    int tid = threadIdx.x;
    int nt = blockDim.x;
    
    // ===== Phase 1: Compute P = exp(Q@K^T*scale - LSE) =====
    // Write P[S,S] row-major: P[sq*S + skv]
    for (int sq = tid; sq < S; sq += nt) {
        const __nv_bfloat16* q_row = Q_bh + static_cast<uint64_t>(sq) * d;
        float lse = L_bh[sq];
        float* p_row = P_mat + static_cast<uint64_t>(sq) * S;
        
        for (int skv = 0; skv < S; skv++) {
            float s_val = 0.0f;
            const __nv_bfloat16* k_row = K_bh + static_cast<uint64_t>(skv) * d;
            
            // Vectorized BF16 dot product
            int i = 0;
            for (; i <= d - 4; i += 4) {
                s_val += __bfloat162float(k_row[i])   * __bfloat162float(q_row[i]);
                s_val += __bfloat162float(k_row[i+1]) * __bfloat162float(q_row[i+1]);
                s_val += __bfloat162float(k_row[i+2]) * __bfloat162float(q_row[i+2]);
                s_val += __bfloat162float(k_row[i+3]) * __bfloat162float(q_row[i+3]);
            }
            for (; i < d; i++) {
                s_val += __bfloat162float(k_row[i]) * __bfloat162float(q_row[i]);
            }
            p_row[skv] = expf(s_val * scale - lse);
        }
    }
    __syncthreads();
    
    // ===== Phase 2: Compute dS = P*(dP - D) =====
    // dP[sq,skv] = dO[sq,:] . V[skv,:]
    // D[sq] = sum_k(P[sq,k] * dP[sq,k])
    // dS[sq,skv] = P[sq,skv] * (dP[sq,skv] - D[sq])
    for (int sq = tid; sq < S; sq += nt) {
        const __nv_bfloat16* dO_row = dO_bh + static_cast<uint64_t>(sq) * d;
        float* p_row = P_mat + static_cast<uint64_t>(sq) * S;
        float* ds_row = dS_mat + static_cast<uint64_t>(sq) * S;
        
        // Pass 1: compute D
        float D = 0.0f;
        for (int skv = 0; skv < S; skv++) {
            float dp = 0.0f;
            const __nv_bfloat16* v_row = V_bh + static_cast<uint64_t>(skv) * d;
            int i = 0;
            for (; i <= d - 4; i += 4) {
                dp += __bfloat162float(dO_row[i])   * __bfloat162float(v_row[i]);
                dp += __bfloat162float(dO_row[i+1]) * __bfloat162float(v_row[i+1]);
                dp += __bfloat162float(dO_row[i+2]) * __bfloat162float(v_row[i+2]);
                dp += __bfloat162float(dO_row[i+3]) * __bfloat162float(v_row[i+3]);
            }
            for (; i < d; i++) {
                dp += __bfloat162float(dO_row[i]) * __bfloat162float(v_row[i]);
            }
            D += p_row[skv] * dp;
        }
        
        // Pass 2: compute dS = P*(dP-D)
        for (int skv = 0; skv < S; skv++) {
            float dp = 0.0f;
            const __nv_bfloat16* v_row = V_bh + static_cast<uint64_t>(skv) * d;
            int i = 0;
            for (; i <= d - 4; i += 4) {
                dp += __bfloat162float(dO_row[i])   * __bfloat162float(v_row[i]);
                dp += __bfloat162float(dO_row[i+1]) * __bfloat162float(v_row[i+1]);
                dp += __bfloat162float(dO_row[i+2]) * __bfloat162float(v_row[i+2]);
                dp += __bfloat162float(dO_row[i+3]) * __bfloat162float(v_row[i+3]);
            }
            for (; i < d; i++) {
                dp += __bfloat162float(dO_row[i]) * __bfloat162float(v_row[i]);
            }
            ds_row[skv] = p_row[skv] * (dp - D);
        }
    }
    __syncthreads();
    
    // ===== Phase 3: dQ[sq,dd] = scale * sum_skv(dS[sq,skv] * K[skv,dd]) =====
    for (int sq = tid; sq < S; sq += nt) {
        const float* ds_row = dS_mat + static_cast<uint64_t>(sq) * S;
        __nv_bfloat16* dq_out = dQ_bh + static_cast<uint64_t>(sq) * d;
        for (int dd = 0; dd < d; dd++) {
            float sum = 0.0f;
            for (int skv = 0; skv < S; skv += 4) {
                sum += ds_row[skv+0] * __bfloat162float(K_bh[static_cast<uint64_t>(skv+0)*d+dd]);
                sum += ds_row[skv+1] * __bfloat162float(K_bh[static_cast<uint64_t>(skv+1)*d+dd]);
                sum += ds_row[skv+2] * __bfloat162float(K_bh[static_cast<uint64_t>(skv+2)*d+dd]);
                sum += ds_row[skv+3] * __bfloat162float(K_bh[static_cast<uint64_t>(skv+3)*d+dd]);
            }
            dq_out[dd] = __float2bfloat16(sum * scale);
        }
    }
    __syncthreads();
    
    // ===== Phase 4: dK[skv,dd] = scale * sum_sq(Q[sq,dd]*dS[sq,skv]) =====
    //            dV[skv,dd]     = sum_sq(P[sq,skv]*dO[sq,dd]) =====
    for (int skv = tid; skv < S; skv += nt) {
        for (int dd = 0; dd < d; dd++) {
            float dk_sum = 0.0f;
            float dv_sum = 0.0f;
            for (int sq = 0; sq < S; sq += 4) {
                dk_sum += __bfloat162float(Q_bh[static_cast<uint64_t>sq*d+dd])      * dS_mat[static_cast<uint64_t>sq*S+skv];
                dk_sum += __bfloat162float(Q_bh[static_cast<uint64_t>(sq+1)*d+dd])  * dS_mat[static_cast<uint64_t>(sq+1)*S+skv];
                dk_sum += __bfloat162float(Q_bh[static_cast<uint64_t>(sq+2)*d+dd])  * dS_mat[static_cast<uint64_t>(sq+2)*S+skv];
                dk_sum += __bfloat162float(Q_bh[static_cast<uint64_t>(sq+3)*d+dd])  * dS_mat[static_cast<uint64_t>(sq+3)*S+skv];
                
                dv_sum += P_mat[static_cast<uint64_t>sq*S+skv]      * __bfloat162float(dO_bh[static_cast<uint64_t>sq*d+dd]);
                dv_sum += P_mat[static_cast<uint64_t>(sq+1)*S+skv]  * __bfloat162float(dO_bh[static_cast<uint64_t>(sq+1)*d+dd]);
                dv_sum += P_mat[static_cast<uint64_t>(sq+2)*S+skv]  * __bfloat162float(dO_bh[static_cast<uint64_t>(sq+2)*d+dd]);
                dv_sum += P_mat[static_cast<uint64_t>(sq+3)*S+skv]  * __bfloat162float(dO_bh[static_cast<uint64_t>(sq+3)*d+dd]);
            }
            dK_bh[static_cast<uint64_t>skv*d+dd] = __float2bfloat16(dk_sum * scale);
            dV_bh[static_cast<uint64_t>skv*d+dd] = __float2bfloat16(dv_sum);
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
    int64_t elems_pm = S * S;
    int64_t elems_per_bh = 2 * elems_pm; // P_mat + dS_mat
    
    // Use very small number of concurrent blocks to avoid OOM
    // 16 blocks = 16 * 2 * S*S * 4B = 16 * 2 * 16MB = ~512MB workspace
    int64_t max_blocks = 16;
    if (max_blocks > n_bh) max_blocks = n_bh;
    
    int64_t total_tmp = max_blocks * elems_per_bh;
    float* tmp_buf;
    CUDA_CHECK(cudaMalloc(&tmp_buf, total_tmp * sizeof(float)));
    
    int threads = 256;
    int n_rounds = (n_bh + max_blocks - 1) / max_blocks;
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    for (int r = 0; r < n_rounds; r++) {
        int64_t start = r * max_blocks;
        int64_t end = min(start + max_blocks, n_bh);
        int nb = static_cast<int>(end - start);
        
        // Clear workspace for this round
        CUDA_CHECK(cudaMemsetAsync(tmp_buf, 0, nb * elems_per_bh * sizeof(float), stream));
        
        mha_bwd_impl::mha_backward_kernel<<<nb, threads, 0, stream>>>(
            Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr,
            dQ_ptr, dK_ptr, dV_ptr,
            tmp_buf,
            static_cast<int>(B), static_cast<int>(H),
            static_cast<int>(S), static_cast<int>(d),
            scale, static_cast<int>(elems_per_bh), n_rounds
        );
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    
    CUDA_CHECK(cudaFree(tmp_buf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // extern "C"