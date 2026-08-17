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

template <int TDIM>
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
    float scale, int elems_per_matrix, int nblocks
) {
    // Each block processes multiple (b,h) pairs sequentially
    // For each (b,h), temp space: P[S,S] + dS[S,S] = 2 matrices
    // tmp_offset per block = blockIdx.x * 2 * S * S
    // tmp_offset per (b,h) within block = 2 * S * S
    
    int elems_pm = S * S;  // elements per S×S matrix
    int bh_stride = (B * H + nblocks - 1) / nblocks;
    
    for (int bi = blockIdx.x; bi < B * H; bi += blockDim.x * gridDim.x) {
        int b = bi / H;
        int h = bi % H;
        
        uint64_t bh_off = static_cast<uint64_t>(bi) * d * S;
        
        const __nv_bfloat16* Q_bh = Q + bh_off;
        const __nv_bfloat16* K_bh = K + bh_off;
        const __nv_bfloat16* V_bh = V + bh_off;
        const __nv_bfloat16* dO_bh = dO + bh_off;
        const float* L_bh = L + bi * S;
        __nv_bfloat16* dQ_bh = dQ + bh_off;
        __nv_bfloat16* dK_bh = dK + bh_off;
        __nv_bfloat16* dV_bh = dV + bh_off;
        
        // Each block has its own temp region
        float* tmp = tmp_base + blockIdx.x * 2 * elems_pm;
        float* P_mat  = tmp;
        float* dS_mat = tmp + elems_pm;
        
        int tid = threadIdx.x;
        int nt = blockDim.x;
        
        // =============================
        // Phase 1: S = Q @ K^T * scale
        // P = exp(S - LSE), store P in P_mat
        // =============================
        #pragma unroll 1
        for (int sq = tid; sq < S; sq += nt) {
            const __nv_bfloat16* q_row = Q_bh + sq * d;
            float* p_row = P_mat + sq * S;
            float lse = L_bh[sq];
            
            #pragma unroll 1
            for (int skv = 0; skv < S; skv++) {
                float s_val = 0.0f;
                const __nv_bfloat16* k_row = K_bh + skv * d;
                
                // Unrolled inner product over d
                int i = 0;
                #pragma unroll
                for (; i + TDIM <= d; i += TDIM) {
                    #pragma unroll
                    for (int j = 0; j < TDIM; j++) {
                        s_val += __bfloat162float(k_row[i+j]) * __bfloat162float(q_row[i+j]);
                    }
                }
                for (; i < d; i++) {
                    s_val += __bfloat162float(k_row[i]) * __bfloat162float(q_row[i]);
                }
                p_row[skv] = expf(s_val * scale - lse);
            }
        }
        __syncthreads();
        
        // =============================
        // Phase 2: dP = dO @ V^T
        // Then: dS = P * (dP - D), store in dS_mat, D = sum(P*dP) per row
        // =============================
        #pragma unroll 1
        for (int sq = tid; sq < S; sq += nt) {
            const __nv_bfloat16* dO_row = dO_bh + sq * d;
            float* p_row = P_mat + sq * S;
            float* ds_row = dS_mat + sq * S;
            
            // First compute dP values and accumulate D simultaneously
            // We need dP temporarily; compute and apply on the fly
            
            // Step 2a: compute dP values into temp registers
            // dP[skv] = dO[sq,:] . V[skv,:]
            float dP_vals[TDIM > S ? S : TDIM]; // stack is limited, don't do this for large S
            
            // Actually for S=4096 we can't keep dP in registers. Need two sub-steps.
            // Compute dP row first, then combine.
            
            // Since we don't have a separate dP buffer, compute dP on-the-fly per skv:
            // For each skv, compute dP_skv = dO[sq,:] . V[skv,:]
            // Then compute D = sum(P * dP) across all skv
            // Then dS[skv] = P[skv] * (dP_skv - D)
            
            // This requires two passes over skv or storing dP temporarily.
            // Pass 1: compute D
            float D = 0.0f;
            #pragma unroll 1
            for (int skv = 0; skv < S; skv++) {
                float dp = 0.0f;
                const __nv_bfloat16* v_row = V_bh + skv * d;
                int i = 0;
                #pragma unroll
                for (; i + TDIM <= d; i += TDIM) {
                    #pragma unroll
                    for (int j = 0; j < TDIM; j++) {
                        dp += __bfloat162float(dO_row[i+j]) * __bfloat162float(v_row[i+j]);
                    }
                }
                for (; i < d; i++) {
                    dp += __bfloat162float(dO_row[i]) * __bfloat162float(v_row[i]);
                }
                D += p_row[skv] * dp;
            }
            
            // Pass 2: compute dS = P * (dP - D)
            #pragma unroll 1
            for (int skv = 0; skv < S; skv++) {
                float dp = 0.0f;
                const __nv_bfloat16* v_row = V_bh + skv * d;
                int i = 0;
                #pragma unroll
                for (; i + TDIM <= d; i += TDIM) {
                    #pragma unroll
                    for (int j = 0; j < TDIM; j++) {
                        dp += __bfloat162float(dO_row[i+j]) * __bfloat162float(v_row[i+j]);
                    }
                }
                for (; i < d; i++) {
                    dp += __bfloat162float(dO_row[i]) * __bfloat162float(v_row[i]);
                }
                ds_row[skv] = p_row[skv] * (dp - D);
            }
        }
        __syncthreads();
        
        // =============================
        // Phase 3: dQ[sq,dd] = scale * sum_skv(dS[sq,skv] * K[skv,dd])
        // Write directly to dQ output
        // =============================
        #pragma unroll 1
        for (int sq = tid; sq < S; sq += nt) {
            const float* ds_row = dS_mat + sq * S;
            __nv_bfloat16* dq_out = dQ_bh + sq * d;
            #pragma unroll 1
            for (int dd = 0; dd < d; dd++) {
                float sum = 0.0f;
                #pragma unroll 1
                for (int skv = 0; skv < S; skv++) {
                    sum += ds_row[skv] * __bfloat162float(K_bh[static_cast<uint64_t>(skv) * d + dd]);
                }
                dq_out[dd] = __float2bfloat16(sum * scale);
            }
        }
        __syncthreads();
        
        // =============================
        // Phase 4: dK[skv,dd] = scale * sum_sq(Q[sq,dd] * dS[sq,skv])
        // dV[skv,dd] = sum_sq(P[sq,skv] * dO[sq,dd])
        // These accumulate across sq — use atomics
        // =============================
        #pragma unroll 1
        for (int skv = tid; skv < S; skv += nt) {
            #pragma unroll 1
            for (int dd = 0; dd < d; dd++) {
                float dk_sum = 0.0f;
                float dv_sum = 0.0f;
                #pragma unroll 1
                for (int sq = 0; sq < S; sq++) {
                    dk_sum += __bfloat162float(Q_bh[static_cast<uint64_t>(sq) * d + dd]) * dS_mat[static_cast<uint64_t>(sq) * S + skv];
                    dv_sum += P_mat[static_cast<uint64_t>(sq) * S + skv] * __bfloat162float(dO_bh[static_cast<uint64_t>(sq) * d + dd]);
                }
                dK_bh[static_cast<uint64_t>(skv) * d + dd] = __float2bfloat16(dk_sum * scale);
                dV_bh[static_cast<uint64_t>(skv) * d + dd] = __float2bfloat16(dv_sum);
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
    
    // Choose number of blocks to limit peak memory
    int64_t elems_pm = S * S;
    // Limit concurrent blocks based on available memory
    // Each block needs 2 * S * S floats = 2 * elems_pm * 4 bytes
    // Cap at ~6 GB peak temp to be safe
    int64_t mem_per_block = 2 * elems_pm * 4; // bytes
    int64_t max_blocks = 6LL * 1024 * 1024 * 1024 / mem_per_block;
    if (max_blocks < 1) max_blocks = 1;
    
    int64_t nblocks = B * H;
    if (nblocks > max_blocks) nblocks = max_blocks;
    
    int64_t tmp_elems = nblocks * 2 * elems_pm;
    size_t tmp_bytes = tmp_elems * sizeof(float);
    
    float* tmp_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&tmp_buf, tmp_bytes));
    
    int threads = 256;
    int blocks = static_cast<int>(nblocks);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_bwd_impl::mha_backward_kernel<8><<<blocks, threads, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        tmp_buf,
        static_cast<int>(B), static_cast<int>(H),
        static_cast<int>(S), static_cast<int>(d),
        scale, static_cast<int>(elems_pm), static_cast<int>(nblocks)
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    CUDA_CHECK(cudaFree(tmp_buf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // extern "C"