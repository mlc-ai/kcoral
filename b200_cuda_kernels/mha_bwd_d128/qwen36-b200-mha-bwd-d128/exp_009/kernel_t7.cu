#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
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

namespace mha_bwd_impl {

constexpr int BM = 32;
constexpr int BN = 32;
constexpr int NF = 8;
constexpr int NT = 256;

__device__ __forceinline__ float to_float(const __nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <int BM_, int BN_, int NF_, int NT_>
__global__ void mha_backward_dqdk_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ O,
    const float* __restrict__ L,
    float* __restrict__ dQ_acc,
    float* __restrict__ dK_acc,
    int B, int H, int S, int d)
{
    unsigned int bh = blockIdx.x;
    if (bh >= (unsigned int)(B * H)) return;

    int tid = threadIdx.x;

    const __nv_bfloat16* Q_bh  = Q  + bh * S * d;
    const __nv_bfloat16* K_bh  = K  + bh * S * d;
    const __nv_bfloat16* V_bh  = V  + bh * S * d;
    const __nv_bfloat16* dO_bh = dO + bh * S * d;
    const __nv_bfloat16* O_bh  = O  + bh * S * d;
    const float* L_bh          = L  + bh * S;
    float* dQ_acc_bh           = dQ_acc + bh * S * d;
    float* dK_acc_bh           = dK_acc + bh * S * d;

    // Shared memory: Q[BM*d] + K[BN*d] + V[BN*d] + dO[BM*d] + P[BM*BN]
    extern __shared__ __align__(4) char smem_char[];
    float* Q_smem  = reinterpret_cast<float*>(smem_char);
    float* K_smem  = Q_smem  + BM_ * d;
    float* V_smem  = K_smem  + BN_ * d;
    float* dO_smem = V_smem  + BN_ * d;
    float* P_smem  = dO_smem + BM_ * d;

    float inv_sqrt_d = rsqrtf((float)d);

    int num_q_blocks = (S + BM_ - 1) / BM_;
    int num_k_blocks = (S + BN_ - 1) / BN_;

    // Init accumulators
    for (int idx = tid; idx < S * d; idx += NT_) {
        dQ_acc_bh[idx] = 0.0f;
        dK_acc_bh[idx] = 0.0f;
    }
    __syncthreads();

    for (int qb = 0; qb < num_q_blocks; ++qb) {
        int q_off = qb * BM_;
        if (q_off >= S) break;

        // Load Q[BM*d] and dO[BM*d] to shared
        for (int idx = tid; idx < BM_ * d; idx += NT_) {
            int qi = idx / d;
            int fi = idx % d;
            int gr = q_off + qi;
            uint64_t row = (uint64_t)gr * d;
            Q_smem[idx]   = (gr < S) ? to_float(Q_bh[row + fi]) : 0.0f;
            dO_smem[idx]  = (gr < S) ? to_float(dO_bh[row + fi]) : 0.0f;
        }

        // Compute D[q] = sum_f(dO[q,f] * O[q,f])
        float D_val[BM_];
        for (int qi = tid; qi < BM_; qi += NT_) {
            int gr = q_off + qi;
            float sum = 0.0f;
            if (gr < S) {
                uint64_t row = (uint64_t)gr * d;
                for (int f = 0; f < d; f++) {
                    sum += to_float(dO_bh[row + f]) * to_float(O_bh[row + f]);
                }
            }
            D_val[qi] = sum;
        }
        __syncthreads();

        for (int kb = 0; kb < num_k_blocks; ++kb) {
            int k_off = kb * BN_;
            if (k_off >= S) break;

            // Load K[BN*d] and V[BN*d] to shared
            for (int idx = tid; idx < BN_ * d; idx += NT_) {
                int ki = idx / d;
                int fi = idx % d;
                int gr = k_off + ki;
                uint64_t row = (uint64_t)gr * d;
                K_smem[idx] = (gr < S) ? to_float(K_bh[row + fi]) : 0.0f;
                V_smem[idx] = (gr < S) ? to_float(V_bh[row + fi]) : 0.0f;
            }
            __syncthreads();

            // Step 1: S[q,k] = Q @ K^T, apply softmax => P[q,k]
            for (int idx = tid; idx < BM_ * BN_; idx += NT_) {
                P_smem[idx] = 0.0f;
            }
            for (int fc = 0; fc < d; fc += NF_) {
                for (int idx = tid; idx < BM_ * BN_; idx += NT_) {
                    int qi = idx / BN_;
                    int ki = idx % BN_;
                    float acc = 0.0f;
                    for (int u = 0; u < NF_; ++u) {
                        acc += Q_smem[qi * d + fc + u] * K_smem[ki * d + fc + u];
                    }
                    P_smem[idx] += acc;
                }
            }
            __syncthreads();

            // Apply softmax: P[q,k] = exp(S/sqrt(d) - L[q])
            for (int idx = tid; idx < BM_ * BN_; idx += NT_) {
                int qi = idx / BN_;
                int gr = q_off + qi;
                float s = P_smem[idx] * inv_sqrt_d;
                P_smem[idx] = (gr < S) ? expf(s - L_bh[gr]) : 0.0f;
            }
            __syncthreads();

            // Step 2 & 3: For each (q,k), compute dOV, then dS = P*(dOV-D)
            // Reuse P_smem to store dS after computing dOV inline
            for (int idx = tid; idx < BM_ * BN_; idx += NT_) {
                int qi = idx / BN_;
                int ki = idx % BN_;
                int gr_q = q_off + qi;
                int gr_k = k_off + ki;
                
                if (gr_q >= S || gr_k >= S) {
                    P_smem[idx] = 0.0f;
                    continue;
                }
                
                // dOV[q,k] = dO[q] . V[k]
                float dOV = 0.0f;
                uint64_t row_q = (uint64_t)gr_q * d;
                uint64_t row_k = (uint64_t)gr_k * d;
                for (int f = 0; f < d; f += 4) {
                    for (int u = 0; u < 4; ++u) {
                        dOV += to_float(dO_bh[row_q + f + u]) * to_float(V_bh[row_k + f + u]);
                    }
                }
                
                float P = P_smem[idx];
                P_smem[idx] = P * (dOV - D_val[qi]);
            }
            __syncthreads();

            // Now P_smem holds dS[q,k]. Compute dQ and dK.
            
            // dQ[q,f] += SUM_k dS[q,k] * K[k,f]
            for (int fi = tid; fi < d; fi += NT_) {
                for (int qi = 0; qi < BM_; ++qi) {
                    int gr_q = q_off + qi;
                    if (gr_q >= S) continue;
                    float dq_sum = 0.0f;
                    for (int ki = 0; ki < BN_; ++ki) {
                        dq_sum += P_smem[qi * BN_ + ki] * K_smem[ki * d + fi];
                    }
                    atomicAdd(&dQ_acc_bh[(uint64_t)gr_q * d + fi], dq_sum);
                }
            }
            __syncthreads();

            // dK[k,f] += SUM_q dS[q,k] * Q[q,f]
            for (int fi = tid; fi < d; fi += NT_) {
                for (int ki = 0; ki < BN_; ++ki) {
                    int gr_k = k_off + ki;
                    if (gr_k >= S) continue;
                    float dk_sum = 0.0f;
                    for (int qi = 0; qi < BM_; ++qi) {
                        dk_sum += P_smem[qi * BN_ + ki] * Q_smem[qi * d + fi];
                    }
                    atomicAdd(&dK_acc_bh[(uint64_t)gr_k * d + fi], dk_sum);
                }
            }
            __syncthreads();
        }
    }
}

template <int BM_, int BN_, int NF_, int NT_>
__global__ void mha_backward_dv_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    float* __restrict__ dV_acc,
    int B, int H, int S, int d)
{
    unsigned int bh = blockIdx.x;
    if (bh >= (unsigned int)(B * H)) return;

    int tid = threadIdx.x;

    const __nv_bfloat16* Q_bh  = Q  + bh * S * d;
    const __nv_bfloat16* K_bh  = K  + bh * S * d;
    const __nv_bfloat16* dO_bh = dO + bh * S * d;
    const float* L_bh          = L  + bh * S;
    float* dV_acc_bh           = dV_acc + bh * S * d;

    // SMEM: Q[BM*d] + K[BN*d] + P[BM*BN]  (dO loaded on-the-fly from global)
    extern __shared__ __align__(4) char smem_char[];
    float* Q_smem = reinterpret_cast<float*>(smem_char);
    float* K_smem = Q_smem + BM_ * d;
    float* P_smem = K_smem + BN_ * d;

    float inv_sqrt_d = rsqrtf((float)d);

    int num_v_blocks = (S + BN_ - 1) / BN_;
    int num_q_blocks = (S + BM_ - 1) / BM_;

    for (int idx = tid; idx < S * d; idx += NT_) {
        dV_acc_bh[idx] = 0.0f;
    }
    __syncthreads();

    for (int vb = 0; vb < num_v_blocks; ++vb) {
        int v_off = vb * BN_;
        if (v_off >= S) break;

        // Load K[BN*d] to shared
        for (int idx = tid; idx < BN_ * d; idx += NT_) {
            int vi = idx / d;
            int fi = idx % d;
            int gr = v_off + vi;
            K_smem[idx] = (gr < S) ? to_float(K_bh[(uint64_t)gr * d + fi]) : 0.0f;
        }
        __syncthreads();

        for (int qb = 0; qb < num_q_blocks; ++qb) {
            int q_off = qb * BM_;
            if (q_off >= S) break;

            // Load Q[BM*d] to shared
            for (int idx = tid; idx < BM_ * d; idx += NT_) {
                int qi = idx / d;
                int fi = idx % d;
                int gr = q_off + qi;
                Q_smem[idx] = (gr < S) ? to_float(Q_bh[(uint64_t)gr * d + fi]) : 0.0f;
            }
            __syncthreads();

            // P[v,q] = exp(Q[q] @ K[v]^T / sqrt(d) - L[v])
            for (int idx = tid; idx < BM_ * BN_; idx += NT_) {
                P_smem[idx] = 0.0f;
            }
            for (int fc = 0; fc < d; fc += NF_) {
                for (int idx = tid; idx < BM_ * BN_; idx += NT_) {
                    int vi = idx / BM_;
                    int qi = idx % BM_;
                    float acc = 0.0f;
                    for (int u = 0; u < NF_; ++u) {
                        acc += Q_smem[qi * d + fc + u] * K_smem[vi * d + fc + u];
                    }
                    P_smem[idx] += acc;
                }
            }
            __syncthreads();

            // Apply softmax
            for (int idx = tid; idx < BM_ * BN_; idx += NT_) {
                int vi = idx / BM_;
                int gr_v = v_off + vi;
                float s = P_smem[idx] * inv_sqrt_d;
                P_smem[idx] = (gr_v < S) ? expf(s - L_bh[gr_v]) : 0.0f;
            }
            __syncthreads();

            // dV[v,f] += SUM_q P[v,q] * dO[q,f]  (read dO from global)
            for (int fi = tid; fi < d; fi += NT_) {
                for (int vi = 0; vi < BN_; ++vi) {
                    int gr_v = v_off + vi;
                    if (gr_v >= S) continue;
                    float dv_sum = 0.0f;
                    for (int qi = 0; qi < BM_; ++qi) {
                        int gr_q = q_off + qi;
                        if (gr_q >= S) continue;
                        dv_sum += P_smem[vi * BM_ + qi] * to_float(dO_bh[(uint64_t)gr_q * d + fi]);
                    }
                    atomicAdd(&dV_acc_bh[(uint64_t)gr_v * d + fi], dv_sum);
                }
            }
            __syncthreads();
        }
    }
}

__global__ void convert_to_bf16_kernel(const float* src, __nv_bfloat16* dst, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
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

    const __nv_bfloat16* Q_ptr  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr          = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr      = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr      = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr      = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int64_t out_size = B * H * S * d;
    int64_t acc_bytes = out_size * sizeof(float);
    float* dQ_acc = nullptr, *dK_acc = nullptr, *dV_acc = nullptr;
    CUDA_CHECK(cudaMalloc(&dQ_acc, acc_bytes));
    CUDA_CHECK(cudaMalloc(&dK_acc, acc_bytes));
    CUDA_CHECK(cudaMalloc(&dV_acc, acc_bytes));

    dim3 grid(B * H);
    dim3 block(NT);

    // SMEM dqdk: Q[BM*d]+K[BN*d]+V[BN*d]+dO[BM*d]+P[BM*BN] = (2*BM+2*BN)*d + BM*BN floats
    int smem_dqdk = ((2*BM + 2*BN) * d + BM * BN) * sizeof(float);
    
    // SMEM dV: Q[BM*d]+K[BN*d]+P[BM*BN] = (BM+BN)*d + BM*BN floats
    int smem_dv = ((BM + BN) * d + BM * BN) * sizeof(float);

    mha_backward_dqdk_kernel<BM, BN, NF, NT><<<grid, block, smem_dqdk, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dQ_acc, dK_acc,
        (int)B, (int)H, (int)S, (int)d);
    CUDA_CHECK(cudaGetLastError());

    mha_backward_dv_kernel<BM, BN, NF, NT><<<grid, block, smem_dv, stream>>>(
        Q_ptr, K_ptr, dO_ptr, L_ptr, dV_acc,
        (int)B, (int)H, (int)S, (int)d);
    CUDA_CHECK(cudaGetLastError());

    dim3 gconv((out_size + NT - 1) / NT);
    dim3 bconv(NT);
    convert_to_bf16_kernel<<<gconv, bconv, 0, stream>>>(dQ_acc, dQ_ptr, out_size);
    CUDA_CHECK(cudaGetLastError());
    convert_to_bf16_kernel<<<gconv, bconv, 0, stream>>>(dK_acc, dK_ptr, out_size);
    CUDA_CHECK(cudaGetLastError());
    convert_to_bf16_kernel<<<gconv, bconv, 0, stream>>>(dV_acc, dV_ptr, out_size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(dQ_acc));
    CUDA_CHECK(cudaFree(dK_acc));
    CUDA_CHECK(cudaFree(dV_acc));
}

}  // namespace mha_bwd_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);