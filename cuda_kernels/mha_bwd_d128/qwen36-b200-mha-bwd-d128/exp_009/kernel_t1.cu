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

constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 32;
constexpr int TILE_F  = 8;
constexpr int NT = 256;

__device__ __forceinline__ float to_float(const __nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 to_bf16(float x) {
    return __float2bfloat16(x);
}

template <int BM, int BN, int NF, int NT>
__global__ void mha_backward_dqdk_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ O,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    int B, int H, int S, int d)
{
    unsigned int bh = blockIdx.x;
    if (bh >= (unsigned int)(B * H)) return;

    int tid = threadIdx.x;

    const __nv_bfloat16* Q_bh = Q + bh * S * d;
    const __nv_bfloat16* K_bh = K + bh * S * d;
    const __nv_bfloat16* V_bh = V + bh * S * d;
    const __nv_bfloat16* dO_bh = dO + bh * S * d;
    const __nv_bfloat16* O_bh  = O  + bh * S * d;
    const float* L_bh = L + bh * S;
    __nv_bfloat16* dQ_bh = dQ + bh * S * d;
    __nv_bfloat16* dK_bh = dK + bh * S * d;

    extern __shared__ __align__(4) char smem_char[];
    float* Q_smem = reinterpret_cast<float*>(smem_char);
    float* K_smem = Q_smem + BM * d;

    float inv_sqrt_d = rsqrtf((float)d);

    int num_q_blocks = (S + BM - 1) / BM;
    int num_k_blocks = (S + BN - 1) / BN;

    // Initialize dQ[bh] to zero
    for (int i = tid; i < S * d; i += NT) {
        dQ_bh[i] = __float2bfloat16(0.0f);
    }
    __syncthreads();

    for (int qb = 0; qb < num_q_blocks; ++qb) {
        int q_off = qb * BM;
        if (q_off >= S) break;

        // Load Q tile [BM x d] to shared
        for (int idx = tid; idx < BM * d; idx += NT) {
            int qi = idx / d;
            int fi = idx % d;
            int gr = q_off + qi;
            Q_smem[idx] = (gr < S) ? to_float(Q_bh[(uint64_t)gr * d + fi]) : 0.0f;
        }

        // Precompute D[q] = sum_f(dO[q,f] * O[q,f]) for this q tile
        float D_tile[BM];
        for (int qi = tid; qi < BM; qi += NT) {
            int gr = q_off + qi;
            float sum = 0.0f;
            if (gr < S) {
                for (int f = 0; f < d; f++) {
                    sum += to_float(dO_bh[(uint64_t)gr * d + f]) * to_float(O_bh[(uint64_t)gr * d + f]);
                }
            }
            D_tile[qi] = sum;
        }
        __syncthreads();

        // Accumulators for dQ: each thread owns some (qi, fi)
        float dQ_acc[BM * d / NT];
        #pragma unroll
        for (int ii = 0; ii < BM * d / NT; ++ii) dQ_acc[ii] = 0.0f;

        // Accumulators for dK partials per k_block
        // We'll write dK to global at end of each k_block
        float dK_partial[BN * d / NT];
        #pragma unroll
        for (int ii = 0; ii < BN * d / NT; ++ii) dK_partial[ii] = 0.0f;

        for (int kb = 0; kb < num_k_blocks; ++kb) {
            int k_off = kb * BN;
            if (k_off >= S) break;

            // Reset dK partials
            #pragma unroll
            for (int ii = 0; ii < BN * d / NT; ++ii) dK_partial[ii] = 0.0f;

            // Load K tile [BN x d] to shared
            for (int idx = tid; idx < BN * d; idx += NT) {
                int ki = idx / d;
                int fi = idx % d;
                int gr = k_off + ki;
                K_smem[idx] = (gr < S) ? to_float(K_bh[(uint64_t)gr * d + fi]) : 0.0f;
            }
            __syncthreads();

            // Compute S[BM x BN] = Q @ K^T / sqrt(d), then P = exp(S/sqrt(d) - L)
            float S_tile[BM * BN];
            float P_tile[BM * BN];

            // Init S_tile
            for (int idx = tid; idx < BM * BN; idx += NT) {
                S_tile[idx] = 0.0f;
            }

            // Gemm: Q_smem[BM*d] x K_smem[BN*d]^T -> S_tile[BM*BN]
            for (int fc = 0; fc < d; fc += NF) {
                for (int idx = tid; idx < BM * BN; idx += NT) {
                    int qi = idx / BN;
                    int ki = idx % BN;
                    float acc = 0.0f;
                    for (int u = 0; u < NF && fc + u < d; ++u) {
                        acc += Q_smem[qi * d + fc + u] * K_smem[ki * d + fc + u];
                    }
                    S_tile[idx] += acc;
                }
            }
            __syncthreads();

            // Apply softmax: P = exp(S/sqrt(d) - L[q])
            for (int idx = tid; idx < BM * BN; idx += NT) {
                int qi = idx / BN;
                int gr = q_off + qi;
                float s = S_tile[idx] * inv_sqrt_d;
                P_tile[idx] = (gr < S) ? expf(s - L_bh[gr]) : 0.0f;
            }
            __syncthreads();

            // Load V and dO for this k tile to compute dS = P * (dOV - D)
            // dOV[q,k] = dO[q] @ V[k]^T
            float dOV[BM];
            for (int qi = tid; qi < BM; qi += NT) {
                int gr_q = q_off + qi;
                dOV[qi] = 0.0f;
            }
            // Actually compute dOV per (q, k) pair below inline

            // For each (qi, ki): compute dS, then update dQ_acc and dK_partial
            for (int idx = tid; idx < BM * BN; idx += NT) {
                int qi = idx / BN;
                int ki = idx % BN;
                int gr_q = q_off + qi;
                int gr_k = k_off + ki;

                if (gr_q >= S || gr_k >= S) continue;

                // Compute dOV[q,k] = dO[q] . V[k]
                float dOV_val = 0.0f;
                for (int f = 0; f < d; f += 4) {
                    for (int u = 0; u < 4 && f + u < d; ++u) {
                        dOV_val += to_float(dO_bh[(uint64_t)gr_q * d + f + u])
                                 * to_float(V_bh[(uint64_t)gr_k * d + f + u]);
                    }
                }

                float P = P_tile[idx];
                float dS = P * (dOV_val - D_tile[qi]);

                // dQ[gr_q, f] += dS * K[gr_k, f]
                for (int f = tid; f < d; f += NT) {
                    dQ_acc[qi * d / NT + f / NT * NT / NT] = 0.0f; // Will be handled below
                }
                
                // dK_partial[ki, f] += dS * Q[gr_q, f]  
                for (int f_base = tid; f_base < d; f_base += NT) {
                    float k_val = K_smem[ki * d + f_base];
                    float q_val = Q_smem[qi * d + f_base];
                    // Use register accumulation - we'll aggregate later
                }
            }

            __syncthreads();

            // Write dK partials: accumulate into global memory using atomics
            for (int idx = tid; idx < BN * d; idx += NT) {
                int ki = idx / d;
                int fi = idx % d;
                int gr_k = k_off + ki;
                if (gr_k >= S) continue;

                float dk_sum = 0.0f;
                for (int qi = 0; qi < BM; ++qi) {
                    int gr_q = q_off + qi;
                    if (gr_q >= S) continue;
                    int pidx = qi * BN + ki;
                    
                    // Recompute dS for this (qi, ki)
                    int g_q_row = q_off + qi;
                    float dOV_val = 0.0f;
                    for (int f = 0; f < d; f += 4) {
                        for (int u = 0; u < 4 && f + u < d; ++u) {
                            dOV_val += to_float(dO_bh[(uint64_t)g_q_row * d + f + u])
                                     * to_float(V_bh[(uint64_t)gr_k * d + f + u]);
                        }
                    }
                    float P = P_tile[pidx];
                    float dS = P * (dOV_val - D_tile[qi]);
                    dk_sum += dS * Q_smem[qi * d + fi];
                }

                // Atomic add to dK
                atomicAdd((float*)&dK_bh[(uint64_t)gr_k * d + fi], dk_sum);
            }

            __syncthreads();

            // Write dQ partials: accumulate into global memory
            for (int idx = tid; idx < BM * d; idx += NT) {
                int qi = idx / d;
                int fi = idx % d;
                int gr_q = q_off + qi;
                if (gr_q >= S) continue;

                float dq_sum = 0.0f;
                for (int ki = 0; ki < BN; ++ki) {
                    int gr_k = k_off + ki;
                    if (gr_k >= S) continue;
                    int pidx = qi * BN + ki;

                    int g_q_row = q_off + qi;
                    float dOV_val = 0.0f;
                    for (int f = 0; f < d; f += 4) {
                        for (int u = 0; u < 4 && f + u < d; ++u) {
                            dOV_val += to_float(dO_bh[(uint64_t)g_q_row * d + f + u])
                                     * to_float(V_bh[(uint64_t)gr_k * d + f + u]);
                        }
                    }
                    float P = P_tile[pidx];
                    float dS = P * (dOV_val - D_tile[qi]);
                    dq_sum += dS * K_smem[ki * d + fi];
                }

                atomicAdd((float*)&dQ_bh[(uint64_t)gr_q * d + fi], dq_sum);
            }

            __syncthreads();
        }
    }
}

template <int BM, int BN, int NF, int NT>
__global__ void mha_backward_dv_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int d)
{
    unsigned int bh = blockIdx.x;
    if (bh >= (unsigned int)(B * H)) return;

    int tid = threadIdx.x;

    const __nv_bfloat16* Q_bh = Q + bh * S * d;
    const __nv_bfloat16* K_bh = K + bh * S * d;
    const __nv_bfloat16* dO_bh = dO + bh * S * d;
    const float* L_bh = L + bh * S;
    __nv_bfloat16* dV_bh = dV + bh * S * d;

    extern __shared__ __align__(4) char smem_char[];
    float* Q_smem = reinterpret_cast<float*>(smem_char);
    float* K_smem = Q_smem + BM * d;
    float* dO_smem = K_smem + BN * d;

    float inv_sqrt_d = rsqrtf((float)d);

    int num_v_blocks = (S + BN - 1) / BN;
    int num_q_blocks = (S + BM - 1) / BM;

    // Zero dV
    for (int i = tid; i < S * d; i += NT) {
        dV_bh[i] = __float2bfloat16(0.0f);
    }
    __syncthreads();

    for (int vb = 0; vb < num_v_blocks; ++vb) {
        int v_off = vb * BN;
        if (v_off >= S) break;

        // Load K[v_off:v_off+BN] to shared
        for (int idx = tid; idx < BN * d; idx += NT) {
            int vi = idx / d;
            int fi = idx % d;
            int gr = v_off + vi;
            K_smem[idx] = (gr < S) ? to_float(K_bh[(uint64_t)gr * d + fi]) : 0.0f;
        }
        __syncthreads();

        // Sweep over all q blocks
        for (int qb = 0; qb < num_q_blocks; ++qb) {
            int q_off = qb * BM;
            if (q_off >= S) break;

            // Load Q[q_off:q_off+BM] to shared
            for (int idx = tid; idx < BM * d; idx += NT) {
                int qi = idx / d;
                int fi = idx % d;
                int gr = q_off + qi;
                Q_smem[idx] = (gr < S) ? to_float(Q_bh[(uint64_t)gr * d + fi]) : 0.0f;
            }
            __syncthreads();

            // Compute P[v,q] = exp(Q[q] @ K[v]^T / sqrt(d) - L[v])
            float P_tile[BM * BN];
            for (int idx = tid; idx < BM * BN; idx += NT) {
                P_tile[idx] = 0.0f;
            }

            for (int fc = 0; fc < d; fc += NF) {
                for (int idx = tid; idx < BM * BN; idx += NT) {
                    int vi = idx / BM;
                    int qi = idx % BM;
                    float acc = 0.0f;
                    for (int u = 0; u < NF && fc + u < d; ++u) {
                        acc += Q_smem[qi * d + fc + u] * K_smem[vi * d + fc + u];
                    }
                    P_tile[idx] += acc;
                }
            }
            __syncthreads();

            // Apply softmax
            for (int idx = tid; idx < BM * BN; idx += NT) {
                int vi = idx / BM;
                int gr_v = v_off + vi;
                float s = P_tile[idx] * inv_sqrt_d;
                P_tile[idx] = (gr_v < S) ? expf(s - L_bh[gr_v]) : 0.0f;
            }
            __syncthreads();

            // Load dO[q_off:q_off+BM]
            for (int idx = tid; idx < BM * d; idx += NT) {
                int qi = idx / d;
                int fi = idx % d;
                int gr = q_off + qi;
                dO_smem[idx] = (gr < S) ? to_float(dO_bh[(uint64_t)gr * d + fi]) : 0.0f;
            }
            __syncthreads();

            // dV[v,f] += SUM_q P[v,q] * dO[q,f]
            for (int idx = tid; idx < BN * d; idx += NT) {
                int vi = idx / d;
                int fi = idx % d;
                int gr_v = v_off + vi;
                if (gr_v >= S) continue;

                float dv_sum = 0.0f;
                for (int qi = 0; qi < BM; ++qi) {
                    int gr_q = q_off + qi;
                    if (gr_q >= S) continue;
                    float P = P_tile[vi * BM + qi];
                    dv_sum += P * dO_smem[qi * d + fi];
                }

                atomicAdd((float*)&dV_bh[(uint64_t)gr_v * d + fi], dv_sum);
            }
            __syncthreads();
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

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    dim3 grid(B * H);
    dim3 block(NT);

    // Shared memory: Q_smem[BM*d] + K_smem[BN*d] + dO_smem[BN*d] = (32*128 + 32*128 + 32*128)*4 = 46080 bytes
    int smem_bytes = (BLOCK_M * d + BLOCK_N * d + BLOCK_N * d) * sizeof(float);

    // Pass 1: dQ and dK
    mha_backward_dqdk_kernel<BLOCK_M, BLOCK_N, TILE_F, NT><<<grid, block, smem_bytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dQ_ptr, dK_ptr,
        (int)B, (int)H, (int)S, (int)d);
    CUDA_CHECK(cudaGetLastError());

    // Pass 2: dV
    mha_backward_dv_kernel<BLOCK_M, BLOCK_N, TILE_F, NT><<<grid, block, smem_bytes, stream>>>(
        Q_ptr, K_ptr, dO_ptr, L_ptr, dV_ptr,
        (int)B, (int)H, (int)S, (int)d);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace mha_bwd_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);