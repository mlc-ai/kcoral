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

template <int BM, int BN, int NF, int NT>
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

    const __nv_bfloat16* Q_bh = Q + bh * S * d;
    const __nv_bfloat16* K_bh = K + bh * S * d;
    const __nv_bfloat16* V_bh = V + bh * S * d;
    const __nv_bfloat16* dO_bh = dO + bh * S * d;
    const __nv_bfloat16* O_bh  = O  + bh * S * d;
    const float* L_bh = L + bh * S;
    float* dQ_acc_bh = dQ_acc + bh * S * d;
    float* dK_acc_bh = dK_acc + bh * S * d;

    extern __shared__ __align__(4) char smem_char[];
    float* Q_smem   = reinterpret_cast<float*>(smem_char);
    float* K_smem   = Q_smem + BM * d;
    float* S_P_smem = K_smem + BN * d;

    float inv_sqrt_d = rsqrtf((float)d);

    int num_q_blocks = (S + BM - 1) / BM;
    int num_k_blocks = (S + BN - 1) / BN;

    float D_val[BM];

    // Initialize accumulators to zero
    for (int idx = tid; idx < S * d; idx += NT) {
        dQ_acc_bh[idx] = 0.0f;
        dK_acc_bh[idx] = 0.0f;
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

        // Compute D[q] = sum_f(dO[q,f] * O[q,f]) for this q tile
        for (int qi = tid; qi < BM; qi += NT) {
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
            int k_off = kb * BN;
            if (k_off >= S) break;

            // Load K tile [BN x d] to shared
            for (int idx = tid; idx < BN * d; idx += NT) {
                int ki = idx / d;
                int fi = idx % d;
                int gr = k_off + ki;
                K_smem[idx] = (gr < S) ? to_float(K_bh[(uint64_t)gr * d + fi]) : 0.0f;
            }
            __syncthreads();

            // Init S_P_smem[BM*BN] = 0
            for (int idx = tid; idx < BM * BN; idx += NT) {
                S_P_smem[idx] = 0.0f;
            }

            // Gemm: Q_smem[BM*d] x K_smem[BN*d]^T -> S_P_smem[BM*BN]
            for (int fc = 0; fc < d; fc += NF) {
                for (int idx = tid; idx < BM * BN; idx += NT) {
                    int qi = idx / BN;
                    int ki = idx % BN;
                    float acc = 0.0f;
                    int q_base = qi * d + fc;
                    int k_base = ki * d + fc;
                    for (int u = 0; u < NF && fc + u < d; ++u) {
                        acc += Q_smem[q_base + u] * K_smem[k_base + u];
                    }
                    S_P_smem[idx] += acc;
                }
            }
            __syncthreads();

            // Apply softmax: P = exp(S/sqrt(d) - L[q])
            for (int idx = tid; idx < BM * BN; idx += NT) {
                int qi = idx / BN;
                int gr = q_off + qi;
                float s = S_P_smem[idx] * inv_sqrt_d;
                S_P_smem[idx] = (gr < S) ? expf(s - L_bh[gr]) : 0.0f;
            }
            __syncthreads();

            // Precompute dOV[q,k] for all pairs and store for reuse
            // We can't easily cache all of them, so compute inline
            
            // dQ contribution: dQ[gr_q, fi] += SUM_k dS[q,k] * K[gr_k, fi]
            for (int fi = tid; fi < d; fi += NT) {
                for (int qi = 0; qi < BM; ++qi) {
                    int gr_q = q_off + qi;
                    if (gr_q >= S) continue;
                    
                    float dq_sum = 0.0f;
                    uint64_t gq_row = (uint64_t)gr_q * d;
                    
                    for (int ki = 0; ki < BN; ++ki) {
                        int gr_k = k_off + ki;
                        if (gr_k >= S) continue;
                        
                        // Compute dOV[q,k] = dO[gr_q] . V[gr_k]
                        float dOV = 0.0f;
                        uint64_t gk_row = (uint64_t)gr_k * d;
                        for (int f2 = 0; f2 < d; f2 += 4) {
                            for (int u2 = 0; u2 < 4 && f2 + u2 < d; ++u2) {
                                dOV += to_float(dO_bh[gq_row + f2 + u2]) * to_float(V_bh[gk_row + f2 + u2]);
                            }
                        }
                        
                        float P = S_P_smem[qi * BN + ki];
                        float dS = P * (dOV - D_val[qi]);
                        dq_sum += dS * K_smem[ki * d + fi];
                    }
                    
                    atomicAdd(&dQ_acc_bh[gq_row + fi], dq_sum);
                }
            }
            __syncthreads();

            // dK contribution: dK[gr_k, fi] += SUM_q dS[q,k] * Q[gr_q, fi]
            for (int fi = tid; fi < d; fi += NT) {
                for (int ki = 0; ki < BN; ++ki) {
                    int gr_k = k_off + ki;
                    if (gr_k >= S) continue;
                    
                    float dk_sum = 0.0f;
                    uint64_t gk_row = (uint64_t)gr_k * d;
                    
                    for (int qi = 0; qi < BM; ++qi) {
                        int gr_q = q_off + qi;
                        if (gr_q >= S) continue;
                        
                        float dOV = 0.0f;
                        uint64_t gq_row = (uint64_t)gr_q * d;
                        uint64_t vk_row = (uint64_t)gr_k * d;
                        for (int f2 = 0; f2 < d; f2 += 4) {
                            for (int u2 = 0; u2 < 4 && f2 + u2 < d; ++u2) {
                                dOV += to_float(dO_bh[gq_row + f2 + u2]) * to_float(V_bh[vk_row + f2 + u2]);
                            }
                        }
                        
                        float P = S_P_smem[qi * BN + ki];
                        float dS = P * (dOV - D_val[qi]);
                        dk_sum += dS * Q_smem[qi * d + fi];
                    }
                    
                    atomicAdd(&dK_acc_bh[gk_row + fi], dk_sum);
                }
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
    float* __restrict__ dV_acc,
    int B, int H, int S, int d)
{
    unsigned int bh = blockIdx.x;
    if (bh >= (unsigned int)(B * H)) return;

    int tid = threadIdx.x;

    const __nv_bfloat16* Q_bh = Q + bh * S * d;
    const __nv_bfloat16* K_bh = K + bh * S * d;
    const __nv_bfloat16* dO_bh = dO + bh * S * d;
    const float* L_bh = L + bh * S;
    float* dV_acc_bh = dV_acc + bh * S * d;

    extern __shared__ __align__(4) char smem_char[];
    float* Q_smem = reinterpret_cast<float*>(smem_char);
    float* K_smem = Q_smem + BM * d;
    float* work   = K_smem + BN * d;  // Used for P_tile or dO_smem

    float inv_sqrt_d = rsqrtf((float)d);

    int num_v_blocks = (S + BN - 1) / BN;
    int num_q_blocks = (S + BM - 1) / BM;

    // Initialize dV accumulator to zero
    for (int idx = tid; idx < S * d; idx += NT) {
        dV_acc_bh[idx] = 0.0f;
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

            // Phase 1: Compute P_tile[vi*BM + qi] = exp(Q[qi] @ K[vi]^T / sqrt(d) - L[v])
            // Store in work[] as P_smem (BM*BN elements, fits in BM*d since d>=BN=32)
            float* P_smem = work;
            for (int idx = tid; idx < BM * BN; idx += NT) {
                P_smem[idx] = 0.0f;
            }

            for (int fc = 0; fc < d; fc += NF) {
                for (int idx = tid; idx < BM * BN; idx += NT) {
                    int vi = idx / BM;
                    int qi = idx % BM;
                    float acc = 0.0f;
                    for (int u = 0; u < NF && fc + u < d; ++u) {
                        acc += Q_smem[qi * d + fc + u] * K_smem[vi * d + fc + u];
                    }
                    P_smem[idx] += acc;
                }
            }
            __syncthreads();

            // Apply softmax to get P
            for (int idx = tid; idx < BM * BN; idx += NT) {
                int vi = idx / BM;
                int gr_v = v_off + vi;
                float s = P_smem[idx] * inv_sqrt_d;
                P_smem[idx] = (gr_v < S) ? expf(s - L_bh[gr_v]) : 0.0f;
            }
            __syncthreads();

            // Phase 2: Accumulate dV[v,f] += SUM_q P[v,q] * dO[q,f]
            for (int fi = tid; fi < d; fi += NT) {
                for (int vi = 0; vi < BN; ++vi) {
                    int gr_v = v_off + vi;
                    if (gr_v >= S) continue;

                    float dv_sum = 0.0f;
                    for (int qi = 0; qi < BM; ++qi) {
                        int gr_q = q_off + qi;
                        if (gr_q >= S) continue;
                        float P = P_smem[vi * BM + qi];
                        float dO_f = to_float(dO_bh[(uint64_t)gr_q * d + fi]);
                        dv_sum += P * dO_f;
                    }

                    atomicAdd(&dV_acc_bh[(uint64_t)gr_v * d + fi], dv_sum);
                }
            }
            __syncthreads();
        }
    }
}

// Kernel to convert float32 accumulator to bf16 output
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

    // Allocate float32 accumulator buffers
    int64_t acc_size = B * H * S * d * sizeof(float);
    float* dQ_acc = nullptr;
    float* dK_acc = nullptr;
    float* dV_acc = nullptr;
    CUDA_CHECK(cudaMalloc(&dQ_acc, acc_size));
    CUDA_CHECK(cudaMalloc(&dK_acc, acc_size));
    CUDA_CHECK(cudaMalloc(&dV_acc, acc_size));

    dim3 grid(B * H);
    dim3 block(NT);

    // Shared memory: Q_smem[BM*d] + K_smem[BN*d] + S_P_smem[BM*BN]
    int smem_bytes = (BLOCK_M * d + BLOCK_N * d + BLOCK_M * BLOCK_N) * sizeof(float);
    
    // For dV kernel: Q_smem[BM*d] + K_smem[BN*d] + work[BM*d]
    int smem_bytes_dv = (BLOCK_M * d + BLOCK_N * d + BLOCK_M * d) * sizeof(float);

    // Pass 1: dQ and dK (accumulate into float32 buffers)
    mha_backward_dqdk_kernel<BLOCK_M, BLOCK_N, TILE_F, NT><<<grid, block, smem_bytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dQ_acc, dK_acc,
        (int)B, (int)H, (int)S, (int)d);
    CUDA_CHECK(cudaGetLastError());

    // Pass 2: dV (accumulate into float32 buffer)
    mha_backward_dv_kernel<BLOCK_M, BLOCK_N, TILE_F, NT><<<grid, block, smem_bytes_dv, stream>>>(
        Q_ptr, K_ptr, dO_ptr, L_ptr, dV_acc,
        (int)B, (int)H, (int)S, (int)d);
    CUDA_CHECK(cudaGetLastError());

    // Convert float32 accumulators to bf16 outputs
    int64_t out_size = B * H * S * d;
    dim3 grid_conv((out_size + NT - 1) / NT);
    dim3 block_conv(NT);
    convert_to_bf16_kernel<<<grid_conv, block_conv, 0, stream>>>(dQ_acc, dQ_ptr, out_size);
    CUDA_CHECK(cudaGetLastError());
    convert_to_bf16_kernel<<<grid_conv, block_conv, 0, stream>>>(dK_acc, dK_ptr, out_size);
    CUDA_CHECK(cudaGetLastError());
    convert_to_bf16_kernel<<<grid_conv, block_conv, 0, stream>>>(dV_acc, dV_ptr, out_size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(dQ_acc));
    CUDA_CHECK(cudaFree(dK_acc));
    CUDA_CHECK(cudaFree(dV_acc));
}

}  // namespace mha_bwd_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);