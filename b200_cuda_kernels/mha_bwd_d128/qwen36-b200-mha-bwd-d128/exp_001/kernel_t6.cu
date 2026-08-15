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

__forceinline__ __device__ void atomic_add_bf16(__nv_bfloat16* addr, __nv_bfloat16 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    atomicAdd(addr, val);
#else
    unsigned int* p = reinterpret_cast<unsigned int*>(reinterpret_cast<char*>(addr));
    unsigned int old = *p, assumed;
    do {
        assumed = old;
        __nv_bfloat16 nv = val + *reinterpret_cast<__nv_bfloat16*>(&assumed);
        old = atomicCAS(p, assumed, *(unsigned int*)&nv);
    } while (old != assumed);
#endif
}

static constexpr int BDIM_VAL = 128;
static constexpr int BM = 16;
static constexpr int BN = 16;
static constexpr int NT = 256;
static constexpr int TPR = NT / BM;     // 16 threads per row
static constexpr int FPT = BDIM_VAL / TPR;  // 8 features per thread

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
    int mtile_idx = blockIdx.y;
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

    int m_gs = mtile_idx * BM;
    int m_global = m_gs + (tid % BM);
    int feat_base = (tid / BM) * FPT;

    int num_ntiles = (S + BN - 1) / BN;

    // Shared memory: s_Q[BM][D], s_dO[BM][D], s_K[BN][D], s_V[BN][D], s_corr[BM]
    extern __shared__ char smem[];
    float* s_Q    = reinterpret_cast<float*>(smem);
    float* s_dO   = s_Q + BM * D;
    float* s_K    = s_dO + BM * D;
    float* s_V    = s_K + BN * D;
    float* s_corr = s_V + BN * D;

    // ============================
    // Load Q and dO tiles ONCE
    // ============================
    for (int elem = tid; elem < BM * D; elem += NT) {
        int r = elem / D;
        int c = elem % D;
        int mg = m_gs + r;
        if (mg < S) {
            s_Q[r * D + c]     = bf162f(Qbh[mg * D + c]);
            s_dO[r * D + c]    = bf162f(dObh[mg * D + c]);
        } else {
            s_Q[r * D + c]     = 0.f;
            s_dO[r * D + c]    = 0.f;
        }
    }
    __syncthreads();

    // ============================
    // PHASE 1: Compute corr[m_local] for our m-tile
    // corr[m] = sum_n attn[m,n] * (dO[m]·V[n])
    // ============================
    {
        float corr_val = 0.f;
        float lm = (m_global < S) ? Lbh[m_global] : 0.f;

        // Load Q[m_local] and dO[m_local] into regs for speed
        float qv[FPT];
        float dv[FPT];
        if (m_global < S) {
            #pragma unroll
            for (int fi = 0; fi < FPT; fi++) {
                qv[fi] = s_Q[(tid % BM) * D + feat_base + fi];
                dv[fi] = s_dO[(tid % BM) * D + feat_base + fi];
            }
        } else {
            #pragma unroll
            for (int fi = 0; fi < FPT; fi++) {
                qv[fi] = 0.f;
                dv[fi] = 0.f;
            }
        }

        for (int tn = 0; tn < num_ntiles; tn++) {
            int n_gs = tn * BN;

            // Cooperative load K and V tiles
            for (int idx = tid; idx < BN * D; idx += NT) {
                int kr = idx / D;
                int kc = idx % D;
                int ng = n_gs + kr;
                if (ng < S) {
                    s_K[kr * D + kc] = bf162f(Khb[ng * D + kc]);
                    s_V[kr * D + kc] = bf162f(Vbh[ng * D + kc]);
                } else {
                    s_K[kr * D + kc] = 0.f;
                    s_V[kr * D + kc] = 0.f;
                }
            }
            __syncthreads();

            if (m_global >= S) continue;

            int mr = tid % BM;
            for (int kp = 0; kp < BN; kp++) {
                int n_global = n_gs + kp;
                if (n_global >= S) break;

                float qk_p = 0.f;
                float dov_p = 0.f;
                #pragma unroll
                for (int fi = 0; fi < FPT; fi++) {
                    qk_p += qv[fi] * s_K[kp * D + feat_base + fi];
                    dov_p += dv[fi] * s_V[kp * D + feat_base + fi];
                }

                // Warp reduce: 16 threads per row, use XOR shuffle with offset 8
                float qk_f = qk_p;
                float dov_f = dov_p;
                for (int mask = 8; mask > 0; mask >>= 1) {
                    qk_f += __shfl_xor_sync(0xFFFF, qk_f, mask);
                    dov_f += __shfl_xor_sync(0xFFFF, dov_f, mask);
                }

                float attn = expf(qk_f * inv_sd - lm);
                corr_val += attn * dov_f;
            }
        }
        s_corr[tid % BM] = corr_val;
    }
    __syncthreads();

    // Read corr values for all rows in this tile
    float corrs[BM];
    #pragma unroll
    for (int i = 0; i < BM; i++) corrs[i] = s_corr[i];
    __syncthreads();

    // ============================
    // PHASE 2: Compute dQ[m], dK[n], dV[n]
    // ============================
    {
        int mr = tid % BM;
        float my_corr = corrs[mr];
        float lm = (m_global < S) ? Lbh[m_global] : 0.f;

        float qv[FPT];
        float dv_reg[FPT];
        if (m_global < S) {
            #pragma unroll
            for (int fi = 0; fi < FPT; fi++) {
                qv[fi]  = s_Q[mr * D + feat_base + fi];
                dv_reg[fi] = s_dO[mr * D + feat_base + fi];
            }
        } else {
            #pragma unroll
            for (int fi = 0; fi < FPT; fi++) {
                qv[fi] = 0.f;
                dv_reg[fi] = 0.f;
            }
        }

        // Local accumulators for dQ (split formula: term1 - corr*term2)
        float dq_t1[FPT] = {};
        float dq_t2[FPT] = {};
        #pragma unroll
        for (int fi = 0; fi < FPT; fi++) {
            dq_t1[fi] = 0.f;
            dq_t2[fi] = 0.f;
        }

        for (int tn = 0; tn < num_ntiles; tn++) {
            int n_gs = tn * BN;

            // Load K and V tiles
            for (int idx = tid; idx < BN * D; idx += NT) {
                int kr = idx / D;
                int kc = idx % D;
                int ng = n_gs + kr;
                if (ng < S) {
                    s_K[kr * D + kc] = bf162f(Khb[ng * D + kc]);
                    s_V[kr * D + kc] = bf162f(Vbh[ng * D + kc]);
                } else {
                    s_K[kr * D + kc] = 0.f;
                    s_V[kr * D + kc] = 0.f;
                }
            }
            __syncthreads();

            if (m_global >= S) continue;

            for (int kp = 0; kp < BN; kp++) {
                int n_global = n_gs + kp;
                if (n_global >= S) break;

                float qk_p = 0.f;
                float dov_p = 0.f;
                #pragma unroll
                for (int fi = 0; fi < FPT; fi++) {
                    qk_p += qv[fi] * s_K[kp * D + feat_base + fi];
                    dov_p += dv_reg[fi] * s_V[kp * D + feat_base + fi];
                }

                float qk_f = qk_p;
                float dov_f = dov_p;
                for (int mask = 8; mask > 0; mask >>= 1) {
                    qk_f += __shfl_xor_sync(0xFFFF, qk_f, mask);
                    dov_f += __shfl_xor_sync(0xFFFF, dov_f, mask);
                }

                float score = qk_f * inv_sd;
                float attn = expf(score - lm);
                float dscore = attn * (dov_f - my_corr);

                // dV[n] += attn * dO[m] (atomic since multiple m-tile blocks contribute)
                #pragma unroll
                for (int fi = 0; fi < FPT; fi++) {
                    int fd = feat_base + fi;
                    if (fd < D) {
                        atomic_add_bf16(&dVbh[n_global * D + fd], f2bf16(attn * dv_reg[fi]));
                    }
                }

                // dK[n] += dscore * Q[m] / sqrt(d) (atomic)
                #pragma unroll
                for (int fi = 0; fi < FPT; fi++) {
                    int fd = feat_base + fi;
                    if (fd < D) {
                        atomic_add_bf16(&dKbh[n_global * D + fd], f2bf16(dscore * qv[fi] * inv_sd));
                    }
                }

                // dQ accumulators (local registers, no atomics needed)
                #pragma unroll
                for (int fi = 0; fi < FPT; fi++) {
                    float kval = s_K[kp * D + feat_base + fi];
                    dq_t1[fi] += attn * dov_f * kval * inv_sd;
                    dq_t2[fi] += attn * kval * inv_sd;
                }
            }
        }

        // Finalize and write dQ (coalesced - each block owns unique m positions)
        if (m_global < S) {
            #pragma unroll
            for (int fi = 0; fi < FPT; fi++) {
                int fd = feat_base + fi;
                if (fd < D) {
                    dQbh[m_global * D + fd] = f2bf16(dq_t1[fi] - my_corr * dq_t2[fi]);
                }
            }
        }
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

    int num_bh = B * H;
    int num_mtiles = (S + BM - 1) / BM;

    // Shared memory: Q(BM*D) + dO(BM*D) + K(BN*D) + V(BN*D) + corr(BM)
    int smem_bytes = (BM * D + BM * D + BN * D + BN * D) * sizeof(float) + BM * sizeof(float);
    // = (2*16*128 + 2*16*128)*4 + 16*4 = (4096+4096)*4 + 64 = 32768 + 64 = 32832

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    // Initialize dK and dV to zero using memset
    size_t out_size = (size_t)B * H * S * BDIM_VAL * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMemsetAsync(dK_data, 0, out_size, stream));
    CUDA_CHECK(cudaMemsetAsync(dV_data, 0, out_size, stream));

    dim3 grid(num_bh, num_mtiles, 1);
    dim3 block(NT);

    mha_bwd_kernel<BDIM_VAL><<<grid, block, smem_bytes, stream>>>(
        Q_data, K_data, V_data, dO_data, L_data,
        dQ_data, dK_data, dV_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

}  // namespace tvm_ffi_mha_bwd