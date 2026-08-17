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
constexpr int NUM_THREADS = 256;

// Helper: convert bf16 pointer to float for accumulation
__device__ inline float bf16_to_float(const __nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ inline __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

template <int BM, int BN, int NF, int NT>
__global__ void mha_backward_dqdV_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int d)
{
    // One block per (batch, head)
    unsigned int bh = blockIdx.x;
    if (bh >= (unsigned int)(B * H)) return;
    int b = bh / H;
    int h = bh % H;
    
    int tid = threadIdx.x;
    int num_t = blockDim.x;
    
    // Base pointers for this (b,h)
    const __nv_bfloat16* Q_bh = Q + bh * S * d;
    const __nv_bfloat16* K_bh = K + bh * S * d;
    const __nv_bfloat16* V_bh = V + bh * S * d;
    const __nv_bfloat16* dO_bh = dO + bh * S * d;
    const float* L_bh = L + bh * S;
    const float* D_bh = D + bh * S;
    __nv_bfloat16* dQ_bh = dQ + bh * S * d;
    __nv_bfloat16* dK_bh = dK + bh * S * d;
    __nv_bfloat16* dV_bh = dV + bh * S * d;
    
    // External shared memory: Q_smem[BM*d], K_smem[BN*d], V_smem[BN*d], dO_smem[BN*d]
    // Plus dS_accum[BM*BN]
    extern __shared__ __align__(4) char smem_char[];
    float* Q_smem   = reinterpret_cast<float*>(smem_char);
    float* K_smem   = Q_smem   + BM * d;
    float* V_smem   = K_smem   + BN * d;
    float* dO_smem  = V_smem   + BN * d;
    float* dS_acc   = dO_smem  + BN * d;
    
    float inv_sqrt_d = rsqrtf((float)d);
    
    // ==================== Pass 1: Compute dQ, dK ====================
    // For each query tile, sweep all key tiles
    int num_q_blocks = (S + BM - 1) / BM;
    int num_k_blocks = (S + BN - 1) / BN;
    
    for (int qb = 0; qb < num_q_blocks; ++qb) {
        int q_off = qb * BM;
        if (q_off >= S) break;
        
        // Load Q tile to shared memory
        for (int idx = tid; idx < BM * d; idx += num_t) {
            int qi = idx / d;
            int fi = idx % d;
            int g_row = q_off + qi;
            Q_smem[qi * d + fi] = (g_row < S) ? bf16_to_float(Q_bh[g_row * d + fi]) : 0.0f;
        }
        __syncthreads();
        
        // Sweep over key tiles
        for (int kb = 0; kb < num_k_blocks; ++kb) {
            int k_off = kb * BN;
            if (k_off >= S) break;
            
            // Load K tile to shared memory
            for (int idx = tid; idx < BN * d; idx += num_t) {
                int ki = idx / d;
                int fi = idx % d;
                int g_row = k_off + ki;
                K_smem[ki * d + fi] = (g_row < S) ? bf16_to_float(K_bh[g_row * d + fi]) : 0.0f;
            }
            __syncthreads();
            
            // Compute S = Q @ K^T / sqrt(d) and accumulate into dS_acc
            // dS_acc[i*BN+j] accumulates s_ij across F tiles
            for (int idx = tid; idx < BM * BN; idx += num_t) {
                dS_acc[idx] = 0.0f;
            }
            
            // Tiled matmul: Q[BM*d] @ K[BN*d]^T -> S[BM*BN]
            for (int fc = 0; fc < d; fc += NF) {
                int end_fc = min(fc + NF, d);
                int nf = end_fc - fc;
                for (int idx = tid; idx < BM * BN; idx += num_t) {
                    int qi = idx / BN;
                    int ki = idx % BN;
                    float acc = 0.0f;
                    #pragma unroll
                    for (int u = 0; u < NF; ++u) {
                        if (fc + u < d) {
                            acc += Q_smem[qi * d + fc + u] * K_smem[ki * d + fc + u];
                        }
                    }
                    dS_acc[idx] += acc;
                }
            }
            __syncthreads();
            
            // Divide by sqrt(d) and apply softmax: P = exp(S/sqrt(d) - L)
            for (int idx = tid; idx < BM * BN; idx += num_t) {
                int qi = idx / BN;
                int ki = idx % BN;
                int g_row = q_off + qi;
                float s = dS_acc[idx] * inv_sqrt_d;
                dS_acc[idx] = (g_row < S) ? expf(s - L_bh[g_row]) : 0.0f;
            }
            __syncthreads();
            
            // Load V and dO for this k block to compute dOV and finalize dS
            for (int idx = tid; idx < BN * d; idx += num_t) {
                int ki = idx / d;
                int fi = idx % d;
                int g_row = k_off + ki;
                V_smem[ki * d + fi] = (g_row < S) ? bf16_to_float(V_bh[g_row * d + fi]) : 0.0f;
                dO_smem[ki * d + fi] = (g_row < S) ? bf16_to_float(dO_bh[g_row * d + fi]) : 0.0f;
            }
            __syncthreads();
            
            // Compute dS[q][k] *= (dO @ V^T[q][k] - D[q])
            // And accumulate dQ, dK
            for (int qi = tid; qi < BM; qi += num_t) {
                int g_row_q = q_off + qi;
                if (g_row_q >= S) continue;
                float D_q = D_bh[g_row_q];
                for (int ki = 0; ki < BN; ++ki) {
                    int g_row_k = k_off + ki;
                    if (g_row_k >= S) continue;
                    
                    // Compute dOV = dO[g_row_q] @ V[g_row_k]^T
                    float dOV = 0.0f;
                    #pragma unroll
                    for (int fc = 0; fc < d; fc += 4) {
                        #pragma unroll
                        for (int u = 0; u < 4 && fc + u < d; ++u) {
                            dOV += dO_smem[qi * d + fc + u] * V_smem[ki * d + fc + u];
                        }
                    }
                    
                    float P = dS_acc[qi * BN + ki];
                    float dS_qk = P * (dOV - D_q);
                    
                    // dQ[g_row_q][f] += dS_qk * K[g_row_k][f]
                    // Use atomicAdd for float32 accumulation then convert
                    for (int fc = 0; fc < d; fc += 4) {
                        int base = fc;
                        #pragma unroll
                        for (int u = 0; u < 4 && fc + u < d; ++u) {
                            float dq_val = dS_qk * K_smem[ki * d + base + u];
                            float dk_val = dS_qk * Q_smem[qi * d + base + u];
                            
                            // Atomic add to global memory
                            float* dq_ptr = reinterpret_cast<float*>(&dQ_bh[g_row_q * d + base + u]);
                            float* dk_ptr = reinterpret_cast<float*>(&dK_bh[g_row_k * d + base + u]);
                            atomicAdd(dq_ptr, dq_val);
                            atomicAdd(dk_ptr, dk_val);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
    
    // ==================== Pass 2: Compute dV ====================
    // dV[v][f] = SUM_q P[v][q] * dO[q][f]
    // Reuse same shared memory layout
    
    for (int vb = 0; vb < num_q_blocks; ++vb) {
        int v_off = vb * BM;
        if (v_off >= S) break;
        
        // Load V tile for dV destination later (we need it for indexing, not for computation)
        // Actually dV accumulates into global, so we just need P and dO
        
        // We need to recompute P[v][q] = exp(Q[v] @ K[q]^T / sqrt(d) - L[v])
        // This is same as S^T computation
        for (int qb = 0; qb < num_q_blocks; ++qb) {
            int q_off = qb * BM;
            if (q_off >= S) break;
            
            // Load Q[q_off:q_off+BM] to shared
            for (int idx = tid; idx < BM * d; idx += num_t) {
                int qi = idx / d;
                int fi = idx % d;
                int g_row = q_off + qi;
                Q_smem[qi * d + fi] = (g_row < S) ? bf16_to_float(Q_bh[g_row * d + fi]) : 0.0f;
            }
            __syncthreads();
            
            // Need K[v_off:v_off+BM] to compute P
            for (int idx = tid; idx < BM * d; idx += num_t) {
                int vi = idx / d;
                int fi = idx % d;
                int g_row = v_off + vi;
                K_smem[vi * d + fi] = (g_row < S) ? bf16_to_float(K_bh[g_row * d + fi]) : 0.0f;
            }
            __syncthreads();
            
            // Compute P[v][q] = exp(Q[q] @ K[v]^T / sqrt(d) - L[v]) for v in [v_off, v_off+BM), q in [q_off, q_off+BM)
            for (int idx = tid; idx < BM * BM; idx += num_t) {
                dS_acc[idx] = 0.0f;
            }
            for (int fc = 0; fc < d; fc += NF) {
                for (int idx = tid; idx < BM * BM; idx += num_t) {
                    int vi = idx / BM;  // v index
                    int qi = idx % BM;  // q index
                    float acc = 0.0f;
                    #pragma unroll
                    for (int u = 0; u < NF; ++u) {
                        if (fc + u < d) {
                            acc += Q_smem[qi * d + fc + u] * K_smem[vi * d + fc + u];
                        }
                    }
                    dS_acc[idx] += acc;
                }
            }
            __syncthreads();
            
            // Apply softmax
            for (int idx = tid; idx < BM * BM; idx += num_t) {
                int vi = idx / BM;
                int g_v = v_off + vi;
                float s = dS_acc[idx] * inv_sqrt_d;
                dS_acc[idx] = (g_v < S) ? expf(s - L_bh[g_v]) : 0.0f;
            }
            __syncthreads();
            
            // Load dO[q_off:q_off+BM] for dV accumulation
            for (int idx = tid; idx < BM * d; idx += num_t) {
                int qi = idx / d;
                int fi = idx % d;
                int g_row = q_off + qi;
                dO_smem[qi * d + fi] = (g_row < S) ? bf16_to_float(dO_bh[g_row * d + fi]) : 0.0f;
            }
            __syncthreads();
            
            // dV[v][f] += SUM_q P[v][q] * dO[q][f]
            for (int vi = tid; vi < BM; vi += num_t) {
                int g_v = v_off + vi;
                if (g_v >= S) continue;
                for (int qi = 0; qi < BM; ++qi) {
                    int g_q = q_off + qi;
                    if (g_q >= S) continue;
                    float P = dS_acc[vi * BM + qi];
                    for (int fc = 0; fc < d; fc += 4) {
                        int base = fc;
                        #pragma unroll
                        for (int u = 0; u < 4 && fc + u < d; ++u) {
                            float dv_val = P * dO_smem[qi * d + base + u];
                            float* dv_ptr = reinterpret_cast<float*>(&dV_bh[g_v * d + base + u]);
                            atomicAdd(dv_ptr, dv_val);
                        }
                    }
                }
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
    
    // Compute D = rowsum(dO * O) for each (b, h, s)
    // D has shape [B, H, S] in float32
    int64_t D_size = B * H * S;
    float* D_host = new float[D_size];
    
    // Precompute D on CPU for simplicity (could be moved to GPU)
    for (int64_t bh = 0; bh < B * H; ++bh) {
        for (int64_t s = 0; s < S; ++s) {
            float sum = 0.0f;
            for (int64_t f = 0; f < d; ++f) {
                float do_val = __bfloat162float(dO_ptr[(bh * S + s) * d + f]);
                float o_val = __bfloat162float(O_ptr[(bh * S + s) * d + f]);
                sum += do_val * o_val;
            }
            D_host[bh * S + s] = sum;
        }
    }
    
    float* D_dev;
    CUDA_CHECK(cudaMalloc(&D_dev, D_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(D_dev, D_host, D_size * sizeof(float), cudaMemcpyHostToDevice));
    delete[] D_host;
    
    // Allocate workspace for initializing dQ, dK, dV to zero
    int64_t workspace_size = B * H * S * d * sizeof(__nv_bfloat16) * 3;
    __nv_bfloat16* workspace;
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    
    // Copy dQ, dK, dV to workspace, zero, copy back
    CUDA_CHECK(cudaMemcpy(workspace, dQ_ptr, B * H * S * d * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(workspace, 0, B * H * S * d * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(dQ_ptr, workspace, B * H * S * d * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
    
    CUDA_CHECK(cudaMemcpy(workspace + B * H * S * d, dK_ptr, B * H * S * d * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(workspace + B * H * S * d, 0, B * H * S * d * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(dK_ptr, workspace + B * H * S * d, B * H * S * d * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
    
    CUDA_CHECK(cudaMemcpy(workspace + 2 * B * H * S * d, dV_ptr, B * H * S * d * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(workspace + 2 * B * H * S * d, 0, B * H * S * d * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(dV_ptr, workspace + 2 * B * H * S * d, B * H * S * d * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
    
    CUDA_CHECK(cudaFree(workspace));
    
    // Shared memory size: Q_smem + K_smem + V_smem + dO_smem + dS_acc
    // Each is float32: BM*d + BN*d + BN*d + BN*d + BM*BN
    int smem_bytes = (BLOCK_M * d + 2 * BLOCK_N * d + BLOCK_N * d + BLOCK_M * BLOCK_N) * sizeof(float);
    smem_bytes += BLOCK_M * d * sizeof(float); // Extra for pass 2
    
    dim3 grid(B * H);
    dim3 block(NUM_THREADS);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_backward_dqdV_kernel<BLOCK_M, BLOCK_N, TILE_F, NUM_THREADS><<<grid, block, smem_bytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_dev, dQ_ptr, dK_ptr, dV_ptr,
        (int)B, (int)H, (int)S, (int)d);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    CUDA_CHECK(cudaFree(D_dev));
}

}  // namespace mha_bwd_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);