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

// Temporary layout per (b,h):
// Offsets in elements (float4-size):
// S_mat:   S*S
// P_mat:   S*S
// dP_mat:  S*S  
// dS_mat:  S*S
// Total: 4 * S * S floats per (b,h)

__global__ void mha_backward_kernel(
    const __nv_bfloat16* __restrict__ Q,    // [B, H, S, d]
    const __nv_bfloat16* __restrict__ K,    // [B, H, S, d]
    const __nv_bfloat16* __restrict__ V,    // [B, H, S, d]
    const __nv_bfloat16* __restrict__ dO,   // [B, H, S, d]
    const __nv_bfloat16* __restrict__ O,    // [B, H, S, d] - forward output (unused but kept for API compat)
    const float*         __restrict__ L,    // [B, H, S] - logsumexp per query position
    __nv_bfloat16*       __restrict__ dQ,   // [B, H, S, d]
    __nv_bfloat16*       __restrict__ dK,   // [B, H, S, d]
    __nv_bfloat16*       __restrict__ dV,   // [B, H, S, d]
    float*               __restrict__ tmp,  // workspace
    int B, int H, int S, int d,
    float scale, int tmp_elems_per_bh
) {
    // Each block handles one (b, h) pair
    int bh_idx = blockIdx.x;
    int n_bh = B * H;
    if (bh_idx >= n_bh) return;
    
    int b = bh_idx / H;
    int h = bh_idx % H;
    
    // Base offsets for this (b, h) pair
    uint64_t bh_base = static_cast<uint64_t>(b * H + h) * S * d;
    
    const __nv_bfloat16* Q_bh = Q + bh_base;
    const __nv_bfloat16* K_bh = K + bh_base;
    const __nv_bfloat16* V_bh = V + bh_base;
    const __nv_bfloat16* dO_bh = dO + bh_base;
    const float* L_bh = L + bh_idx * S;
    __nv_bfloat16* dQ_bh = dQ + bh_base;
    __nv_bfloat16* dK_bh = dK + bh_base;
    __nv_bfloat16* dV_bh = dV + bh_base;
    
    // Temp space for this (b,h): 4 segments of S*S floats
    float* S_mat  = tmp + bh_idx * tmp_elems_per_bh;
    float* P_mat  = S_mat  + S * S;
    float* dP_mat = P_mat  + S * S;
    float* dS_mat = dP_mat + S * S;
    
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    
    // ============================================================
    // Phase 1: S = Q @ K^T * scale  (S x S matrix, FP32)
    // Each thread processes rows of S_mat
    // ============================================================
    for (int sq = tid; sq < S; sq += nthreads) {
        const float* q_row_f = reinterpret_cast<const float*>(Q_bh + sq * d);
        for (int skv = 0; skv < S; skv++) {
            float sum = 0.0f;
            const __nv_bfloat16* k_row = K_bh + skv * d;
            // Unrolled inner product over dimension d
            int i = 0;
            for (; i + 4 <= d; i += 4) {
                sum += __bfloat162float(k_row[i])   * __bfloat162float(q_row_f[i]);
                sum += __bfloat162float(k_row[i+1]) * __bfloat162float(q_row_f[i+1]);
                sum += __bfloat162float(k_row[i+2]) * __bfloat162float(q_row_f[i+2]);
                sum += __bfloat162float(k_row[i+3]) * __bfloat162float(q_row_f[i+3]);
            }
            for (; i < d; i++) {
                sum += __bfloat162float(k_row[i]) * __bfloat162float(q_row_f[i]);
            }
            S_mat[sq * S + skv] = sum * scale;
        }
    }
    __syncthreads();
    
    // ============================================================
    // Phase 2: P = exp(S - L)  (recover attention probs from LSE)
    // L[bh, sq] is the logsumexp for query position sq
    // ============================================================
    for (int sq = tid; sq < S; sq += nthreads) {
        float lse = L_bh[sq];
        float* p_row = P_mat + sq * S;
        const float* s_row = S_mat + sq * S;
        for (int skv = 0; skv < S; skv++) {
            p_row[skv] = expf(s_row[skv] - lse);
        }
    }
    __syncthreads();
    
    // ============================================================
    // Phase 3: dP = dO @ V^T  (S x S matrix, FP32)
    // dP[sq, skv] = sum_dd(dO[sq,dd] * V[skv,dd])
    // ============================================================
    for (int sq = tid; sq < S; sq += nthreads) {
        const float* do_row_f = reinterpret_cast<const float*>(dO_bh + sq * d);
        for (int skv = 0; skv < S; skv++) {
            float sum = 0.0f;
            const __nv_bfloat16* v_row = V_bh + skv * d;
            int i = 0;
            for (; i + 4 <= d; i += 4) {
                sum += __bfloat162float(do_row_f[i])   * __bfloat162float(v_row[i]);
                sum += __bfloat162float(do_row_f[i+1]) * __bfloat162float(v_row[i+1]);
                sum += __bfloat162float(do_row_f[i+2]) * __bfloat162float(v_row[i+2]);
                sum += __bfloat162float(do_row_f[i+3]) * __bfloat162float(v_row[i+3]);
            }
            for (; i < d; i++) {
                sum += __bfloat162float(do_row_f[i]) * __bfloat162float(v_row[i]);
            }
            dP_mat[sq * S + skv] = sum;
        }
    }
    __syncthreads();
    
    // ============================================================
    // Phase 4: dS = P * (dP - D)
    // D[sq] = sum_skv(P[sq,skv] * dP[sq,skv])  -- scalar per row
    // dS[sq,skv] = P[sq,skv] * (dP[sq,skv] - D[sq])
    // ============================================================
    for (int sq = tid; sq < S; sq += nthreads) {
        float D = 0.0f;
        float* p_row = P_mat + sq * S;
        float* dp_row = dP_mat + sq * S;
        for (int skv = 0; skv < S; skv++) {
            D += p_row[skv] * dp_row[skv];
        }
        float* ds_row = dS_mat + sq * S;
        for (int skv = 0; skv < S; skv++) {
            ds_row[skv] = p_row[skv] * (dp_row[skv] - D);
        }
    }
    __syncthreads();
    
    // ============================================================
    // Phase 5: dQ = (dS * scale) @ K  -> store as BF16
    // dQ[sq, dd] = sum_skv(dS[sq, skv] * K[skv, dd]) * scale
    // ============================================================
    for (int sq = tid; sq < S; sq += nthreads) {
        const float* ds_row = dS_mat + sq * S;
        __nv_bfloat16* dq_out = dQ_bh + sq * d;
        for (int dd = 0; dd < d; dd++) {
            float sum = 0.0f;
            for (int skv = 0; skv < S; skv++) {
                sum += ds_row[skv] * __bfloat162float(K_bh[skv * d + dd]);
            }
            dq_out[dd] = __float2bfloat16(sum * scale);
        }
    }
    __syncthreads();
    
    // ============================================================
    // Phase 6: dK = K^T @ (dS * scale) = ((dS*scale)^T @ Q)^T
    // Equivalently: dK[skv, dd] = sum_sq(Q[sq, dd] * dS[sq, skv]) * scale
    // ============================================================
    for (int skv = tid; skv < S; skv += nthreads) {
        __nv_bfloat16* dk_out = dK_bh + skv * d;
        for (int dd = 0; dd < d; dd++) {
            float sum = 0.0f;
            for (int sq = 0; sq < S; sq++) {
                sum += __bfloat162float(Q_bh[sq * d + dd]) * dS_mat[sq * S + skv];
            }
            dk_out[dd] = __float2bfloat16(sum * scale);
        }
    }
    __syncthreads();
    
    // ============================================================
    // Phase 7: dV = P^T @ dO  -> store as BF16
    // dV[skv, dd] = sum_sq(P[sq, skv] * dO[sq, dd])
    // ============================================================
    for (int skv = tid; skv < S; skv += nthreads) {
        __nv_bfloat16* dv_out = dV_bh + skv * d;
        for (int dd = 0; dd < d; dd++) {
            float sum = 0.0f;
            for (int sq = 0; sq < S; sq++) {
                sum += P_mat[sq * S + skv] * __bfloat162float(dO_bh[sq * d + dd]);
            }
            dv_out[dd] = __float2bfloat16(sum);
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
    
    // L may be [B,H,S] (3D) or [B,H,S,1] (4D); extract scalar S_dim
    int64_t L_S = L.size(L.ndim() - 1);
    (void)L_S; // Should equal S
    
    float scale = 1.0f / std::sqrt(static_cast<float>(d));
    
    const __nv_bfloat16* Q_ptr  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const __nv_bfloat16* O_ptr  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const float*         L_ptr  = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr      = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr      = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr      = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    // Workspace: 4 * S * S floats per (b,h) pair
    int64_t tmp_elems_per_bh = 4LL * S * S;
    size_t tmp_bytes = static_cast<size_t>(B * H) * tmp_elems_per_bh * sizeof(float);
    
    float* tmp_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&tmp_buf, tmp_bytes));
    
    int64_t n_bh = B * H;
    int threads = 256;
    int blocks = static_cast<int>(n_bh);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_bwd_impl::mha_backward_kernel<<<blocks, threads, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        tmp_buf,
        static_cast<int>(B), static_cast<int>(H),
        static_cast<int>(S), static_cast<int>(d),
        scale, static_cast<int>(tmp_elems_per_bh)
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    CUDA_CHECK(cudaFree(tmp_buf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // extern "C"