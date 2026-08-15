#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_bwd {

__forceinline__ __device__ float bf162f(__nv_bfloat16 h) {
    return __bfloat162float(h);
}

__forceinline__ __device__ __nv_bfloat16 f2bf16(float f) {
    return __float2bfloat16(f);
}

__forceinline__ __device__ void atomic_add_bf16(__nv_bfloat16* address, __nv_bfloat16 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    atomicAdd(address, val);
#else
    unsigned int* address_as_ui = reinterpret_cast<unsigned int*>(
        reinterpret_cast<char*>(address));
    unsigned int old = *address_as_ui, assumed;
    do {
        assumed = old;
        __nv_bfloat16 new_val = val + *reinterpret_cast<__nv_bfloat16*>(&assumed);
        old = atomicCAS(address_as_ui, assumed, *(unsigned int*)&new_val);
    } while (old != assumed);
#endif
}

// Shared memory layout constants
static constexpr int BDIM = 128;
static constexpr int BLOCK_M = 16;
static constexpr int BLOCK_N = 16;
static constexpr int NT = 256;
static constexpr int TPR = NT / BLOCK_M; // 16 threads per row
static constexpr int FPT = BDIM / TPR;   // 8 features per thread

template<int D>
__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S)
{
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;

    if (b >= B || h >= H) return;

    size_t bh_off = (size_t)(b * H + h) * S * D;
    size_t l_off  = (size_t)(b * H + h) * S;

    const __nv_bfloat16* Qbh = Q + bh_off;
    const __nv_bfloat16* Khb = K + bh_off;
    const __nv_bfloat16* Vbh = V + bh_off;
    const __nv_bfloat16* dObh = dO + bh_off;
    const float* Lbh = L + l_off;
    __nv_bfloat16* dQbh = dQ + bh_off;
    __nv_bfloat16* dKbh = dK + bh_off;
    __nv_bfloat16* dVbh = dV + bh_off;

    float inv_sd = rsqrtf((float)D);

    // Thread mapping: which m-row and which feature range this thread owns
    int my_r = tid % BLOCK_M;           // 0..15
    int feat_base = (tid / BLOCK_M) * FPT; // 0,8,16,...,120

    int num_mtiles = (S + BLOCK_M - 1) / BLOCK_M;
    int num_ntiles = (S + BLOCK_N - 1) / BLOCK_N;

    // Init output buffers to zero
    for (int i = tid; i < S * D; i += NT) {
        dQbh[i] = f2bf16(0.f);
        dKbh[i] = f2bf16(0.f);
        dVbh[i] = f2bf16(0.f);
    }
    __syncthreads();

    // Shared memory pointers
    extern __shared__ char smem[];
    // Layout: s_K[BN][D], s_V[BN][D], s_corr[BLOCK_M]
    int smem_offset = 0;
    float* s_K    = reinterpret_cast<float*>(smem + smem_offset);
    smem_offset += BLOCK_N * D * sizeof(float);
    float* s_V    = reinterpret_cast<float*>(smem + smem_offset);
    smem_offset += BLOCK_N * D * sizeof(float);
    float* s_corr = reinterpret_cast<float*>(smem + smem_offset);

    // =====================
    // PHASE 1: Compute correction[c] for each m-row
    // corr[m] = sum_n attn[m,n] * (dO[m] · V[n])
    // =====================
    for (int tm = 0; tm < num_mtiles; tm++) {
        int mg_start = tm * BLOCK_M;
        int m_global = mg_start + my_r;

        // Load Q[m_local] and dO[m_local] into registers for our thread
        float q_vals[FPT];
        float do_vals[FPT];
        if (m_global < S) {
            #pragma unroll
            for (int fi = 0; fi < FPT; fi++) {
                q_vals[fi]  = bf162f(Qbh[m_global * D + feat_base + fi]);
                do_vals[fi] = bf162f(dObh[m_global * D + feat_base + fi]);
            }
        } else {
            #pragma unroll
            for (int fi = 0; fi < FPT; fi++) {
                q_vals[fi] = 0.f;
                do_vals[fi] = 0.f;
            }
        }

        float lm = (m_global < S) ? Lbh[m_global] : 0.f;
        float corr_val = 0.f;

        // Iterate over all n-tiles
        for (int tn = 0; tn < num_ntiles; tn++) {
            int ng_start = tn * BLOCK_N;

            // Cooperative load K and V tiles
            for (int idx = tid; idx < BLOCK_N * D; idx += NT) {
                int kp = idx / D;
                int fc = idx % D;
                int ng = ng_start + kp;
                if (ng < S) {
                    s_K[kp * D + fc] = bf162f(Khb[ng * D + fc]);
                    s_V[kp * D + fc] = bf162f(Vbh[ng * D + fc]);
                } else {
                    s_K[kp * D + fc] = 0.f;
                    s_V[kp * D + fc] = 0.f;
                }
            }
            __syncthreads();

            if (m_global >= S) continue;

            for (int kp = 0; kp < BLOCK_N; kp++) {
                int n_global = ng_start + kp;
                if (n_global >= S) break;

                // Partial dot products for this thread's feature slice
                float qk_p = 0.f;
                float dv_p = 0.f;
                #pragma unroll
                for (int fi = 0; fi < FPT; fi++) {
                    qk_p += q_vals[fi] * s_K[kp * D + feat_base + fi];
                    dv_p += do_vals[fi] * s_V[kp * D + feat_base + fi];
                }

                // Warp reduce within the 16 threads that share an m-row
                float qk_full = qk_p;
                float dv_full = dv_p;
                for (int mask = 8; mask > 0; mask >>= 1) {
                    qk_full += __shfl_xor_sync(0x000F, qk_full, mask);
                    dv_full += __shfl_xor_sync(0x000F, dv_full, mask);
                }
                qk_full = __shfl_sync(0x000F, qk_full, 0);
                dv_full = __shfl_sync(0x000F, dv_full, 0);

                float score = qk_full * inv_sd;
                float attn = expf(score - lm);
                corr_val += attn * dv_full;
            }
        }

        s_corr[my_r] = corr_val;
    }
    __syncthreads();

    // Read back all corr values into register array for fast access
    float corrs[BLOCK_M];
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++) {
        corrs[i] = s_corr[i];
    }
    __syncthreads();

    // =====================
    // PHASE 2: Compute dQ, dK, dV using corrections
    // =====================
    // Each thread handles one m-row across all n-tiles, accumulating results
    // For dQ[m]: sum_n [attn*(dov-corr)*K[n]/sqrt(d)] -> write at end
    // For dK[n]: sum_m [attn*(dov-corr)*Q[m]/sqrt(d)] -> atomic add per kp
    // For dV[n]: sum_m [attn*dO[m]]                  -> atomic add per kp
    
    // We'll accumulate dQ locally per m-tile and flush once done
    // dK and dV use atomics since they aggregate from multiple m rows
    
    for (int tm = 0; tm < num_mtiles; tm++) {
        int mg_start = tm * BLOCK_M;
        int m_global = mg_start + my_r;

        // Load Q[m_local] and dO[m_local] for our row
        float q_vals[FPT];
        float do_vals[FPT];
        if (m_global < S) {
            #pragma unroll
            for (int fi = 0; fi < FPT; fi++) {
                q_vals[fi]  = bf162f(Qbh[m_global * D + feat_base + fi]);
                do_vals[fi] = bf162f(dObh[m_global * D + feat_base + fi]);
            }
        } else {
            #pragma unroll
            for (int fi = 0; fi < FPT; fi++) {
                q_vals[fi] = 0.f;
                do_vals[fi] = 0.f;
            }
        }

        float lm = (m_global < S) ? Lbh[m_global] : 0.f;
        float my_corr = corrs[my_r];

        // Accumulators for dQ: split into two terms
        // term1 = sum_n(attn * dov * K/sqrt(d))
        // term2 = sum_n(attn * K/sqrt(d)), final = term1 - corr*term2
        float dq_t1[FPT] = {};
        float dq_t2[FPT] = {};
        #pragma unroll
        for (int fi = 0; fi < FPT; fi++) {
            dq_t1[fi] = 0.f;
            dq_t2[fi] = 0.f;
        }

        // Accumulator for dV contribution from our m_row
        // We need to flush this per-kp since multiple rows contribute
        // Actually let's just use atomics for simplicity
        // dV_acc would store partial per n... too much state. Use atomics.

        for (int tn = 0; tn < num_ntiles; tn++) {
            int ng_start = tn * BLOCK_N;

            // Cooperative load K and V tiles
            for (int idx = tid; idx < BLOCK_N * D; idx += NT) {
                int kp = idx / D;
                int fc = idx % D;
                int ng = ng_start + kp;
                if (ng < S) {
                    s_K[kp * D + fc] = bf162f(Khb[ng * D + fc]);
                    s_V[kp * D + fc] = bf162f(Vbh[ng * D + fc]);
                } else {
                    s_K[kp * D + fc] = 0.f;
                    s_V[kp * D + fc] = 0.f;
                }
            }
            __syncthreads();

            if (m_global >= S) continue;

            for (int kp = 0; kp < BLOCK_N; kp++) {
                int n_global = ng_start + kp;
                if (n_global >= S) break;

                // Partial dot products
                float qk_p = 0.f;
                float dv_p = 0.f;
                #pragma unroll
                for (int fi = 0; fi < FPT; fi++) {
                    qk_p += q_vals[fi] * s_K[kp * D + feat_base + fi];
                    dv_p += do_vals[fi] * s_V[kp * D + feat_base + fi];
                }

                // Warp reduce
                float qk_full = qk_p;
                float dv_full = dv_p;
                for (int mask = 8; mask > 0; mask >>= 1) {
                    qk_full += __shfl_xor_sync(0x000F, qk_full, mask);
                    dv_full += __shfl_xor_sync(0x000F, dv_full, mask);
                }
                qk_full = __shfl_sync(0x000F, qk_full, 0);
                dv_full = __shfl_sync(0x000F, dv_full, 0);

                float score = qk_full * inv_sd;
                float attn = expf(score - lm);
                float dscore = attn * (dv_full - my_corr);

                // Accumulate dV[n] += attn * dO[m] (per feature slice)
                #pragma unroll
                for (int fi = 0; fi < FPT; fi++) {
                    int fd = feat_base + fi;
                    if (fd < D) {
                        float d_v = attn * do_vals[fi];
                        atomic_add_bf16(&dVbh[n_global * D + fd], f2bf16(d_v));
                    }
                }

                // Accumulate dK[n] += dscore * Q[m] / sqrt(d) (per feature slice)
                #pragma unroll
                for (int fi = 0; fi < FPT; fi++) {
                    int fd = feat_base + fi;
                    if (fd < D) {
                        float d_k = dscore * q_vals[fi] * inv_sd;
                        atomic_add_bf16(&dKbh[n_global * D + fd], f2bf16(d_k));
                    }
                }

                // Accumulate dQ terms (local register accumulators)
                float k_vals[FPT];
                #pragma unroll
                for (int fi = 0; fi < FPT; fi++) {
                    k_vals[fi] = s_K[kp * D + feat_base + fi];
                }
                #pragma unroll
                for (int fi = 0; fi < FPT; fi++) {
                    dq_t1[fi] += attn * dv_full * k_vals[fi] * inv_sd;
                    dq_t2[fi] += attn * k_vals[fi] * inv_sd;
                }
            }
        }

        // Flush dQ for our m_row after processing all n-tiles
        if (m_global < S) {
            #pragma unroll
            for (int fi = 0; fi < FPT; fi++) {
                int fd = feat_base + fi;
                if (fd < D) {
                    float final_dq = dq_t1[fi] - my_corr * dq_t2[fi];
                    // Direct write (only one writer per element)
                    dQbh[m_global * D + fd] = f2bf16(final_dq);
                }
            }
        }
        __syncthreads();
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    constexpr int B = 4;
    constexpr int H = 48;
    int S = (int)Q.size(2);

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* dO_data = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_data = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_data = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_data = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_data = static_cast<__nv_bfloat16*>(dV.data_ptr());

    int num_blocks = B * H;
    int smem_size = (BLOCK_N * BDIM * 2) * sizeof(float) + BLOCK_M * sizeof(float);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_bwd_kernel<BDIM><<<num_blocks, NT, smem_size, stream>>>(
        Q_data, K_data, V_data, dO_data, L_data,
        dQ_data, dK_data, dV_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

}  // namespace tvm_ffi_mha_bwd