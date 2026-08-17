#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); exit(1); } \
} while(0)

#define BLOCK_M 32
#define BLOCK_N 32
#define THREADS 128

namespace mha_bwd_impl {

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
    int B, int H, int S, float scale) {

    int bh = blockIdx.x;
    if (bh >= B * H) return;
    int b = bh / H;
    int h = bh % H;

    int64_t base = (int64_t)(b * H + h) * S * D;
    const __nv_bfloat16* q_base = Q + base;
    const __nv_bfloat16* k_base = K + base;
    const __nv_bfloat16* v_base = V + base;
    const __nv_bfloat16* do_base = dO + base;
    const float* l_base = L + (b * H + h) * S;
    __nv_bfloat16* dq_base = dQ + base;
    __nv_bfloat16* dk_base = dK + base;
    __nv_bfloat16* dv_base = dV + base;

    extern __shared__ char smem[];
    
    __nv_bfloat16* s_Q  = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* s_K  = s_Q  + BLOCK_M * D;
    __nv_bfloat16* s_V  = s_K  + BLOCK_N * D;
    __nv_bfloat16* s_dO = s_V  + BLOCK_N * D;
    
    float* a_dQ = reinterpret_cast<float*>(s_dO + BLOCK_M * D);
    float* a_dK = a_dQ + BLOCK_M * D;
    float* a_dV = a_dK + BLOCK_N * D;

    int tid = threadIdx.x;

    for (int ns = 0; ns < S; ns += BLOCK_N) {
        int n_valid = min(BLOCK_N, S - ns);
        
        for (int i = tid; i < BLOCK_N * D; i += THREADS) {
            a_dK[i] = 0.0f;
            a_dV[i] = 0.0f;
        }
        __syncthreads();

        for (int i = tid; i < BLOCK_N * D; i += THREADS) {
            int r = i / D;
            int c = i % D;
            int g_r = ns + r;
            s_K[i] = (g_r < S) ? k_base[(int64_t)g_r * D + c] : __float2bfloat16(0.0f);
        }
        for (int i = tid; i < BLOCK_N * D; i += THREADS) {
            int r = i / D;
            int c = i % D;
            int g_r = ns + r;
            s_V[i] = (g_r < S) ? v_base[(int64_t)g_r * D + c] : __float2bfloat16(0.0f);
        }
        __syncthreads();

        for (int ms = 0; ms < S; ms += BLOCK_M) {
            int m_valid = min(BLOCK_M, S - ms);
            
            for (int i = tid; i < BLOCK_M * D; i += THREADS) {
                a_dQ[i] = 0.0f;
            }
            __syncthreads();

            for (int i = tid; i < BLOCK_M * D; i += THREADS) {
                int r = i / D;
                int c = i % D;
                int g_r = ms + r;
                s_Q[i] = (g_r < S) ? q_base[(int64_t)g_r * D + c] : __float2bfloat16(0.0f);
            }
            for (int i = tid; i < BLOCK_M * D; i += THREADS) {
                int r = i / D;
                int c = i % D;
                int g_r = ms + r;
                s_dO[i] = (g_r < S) ? do_base[(int64_t)g_r * D + c] : __float2bfloat16(0.0f);
            }
            __syncthreads();

            int pairs = BLOCK_M * BLOCK_N;
            for (int idx = tid; idx < pairs; idx += THREADS) {
                int mr = idx / BLOCK_N;
                int nr = idx % BLOCK_N;
                
                int q_g = ms + mr;
                int k_g = ns + nr;
                
                if (q_g < S && k_g < S && k_g <= q_g) {
                    float qk_dot = 0.0f;
                    float dov_dot = 0.0f;
                    
                    const __nv_bfloat16* q_row = s_Q + mr * D;
                    const __nv_bfloat16* k_row = s_K + nr * D;
                    const __nv_bfloat16* v_row = s_V + nr * D;
                    const __nv_bfloat16* do_row = s_dO + mr * D;
                    
                    #pragma unroll
                    for (int f = 0; f < D; f += 4) {
                        float q0 = __bfloat162float(q_row[f]);
                        float q1 = __bfloat162float(q_row[f+1]);
                        float q2 = __bfloat162float(q_row[f+2]);
                        float q3 = __bfloat162float(q_row[f+3]);
                        
                        float k0 = __bfloat162float(k_row[f]);
                        float k1 = __bfloat162float(k_row[f+1]);
                        float k2 = __bfloat162float(k_row[f+2]);
                        float k3 = __bfloat162float(k_row[f+3]);
                        
                        float v0 = __bfloat162float(v_row[f]);
                        float v1 = __bfloat162float(v_row[f+1]);
                        float v2 = __bfloat162float(v_row[f+2]);
                        float v3 = __bfloat162float(v_row[f+3]);
                        
                        float d0 = __bfloat162float(do_row[f]);
                        float d1 = __bfloat162float(do_row[f+1]);
                        float d2 = __bfloat162float(do_row[f+2]);
                        float d3 = __bfloat162float(do_row[f+3]);
                        
                        qk_dot += q0*k0 + q1*k1 + q2*k2 + q3*k3;
                        dov_dot += d0*v0 + d1*v1 + d2*v2 + d3*v3;
                    }
                    
                    float log_lse = l_base[q_g];
                    float p = expf(qk_dot * scale - log_lse);
                    float dp = p * dov_dot;
                    
                    float* adq_ptr = a_dQ + mr * D;
                    float* adk_ptr = a_dK + nr * D;
                    float* adv_ptr = a_dV + nr * D;
                    
                    #pragma unroll
                    for (int f = 0; f < D; f += 4) {
                        atomicAdd(&adq_ptr[f],   dp * __bfloat162float(k_row[f]));
                        atomicAdd(&adq_ptr[f+1], dp * __bfloat162float(k_row[f+1]));
                        atomicAdd(&adq_ptr[f+2], dp * __bfloat162float(k_row[f+2]));
                        atomicAdd(&adq_ptr[f+3], dp * __bfloat162float(k_row[f+3]));
                        
                        atomicAdd(&adk_ptr[f],   dp * __bfloat162float(q_row[f]));
                        atomicAdd(&adk_ptr[f+1], dp * __bfloat162float(q_row[f+1]));
                        atomicAdd(&adk_ptr[f+2], dp * __bfloat162float(q_row[f+2]));
                        atomicAdd(&adk_ptr[f+3], dp * __bfloat162float(q_row[f+3]));
                        
                        atomicAdd(&adv_ptr[f],   p * __bfloat162float(do_row[f]));
                        atomicAdd(&adv_ptr[f+1], p * __bfloat162float(do_row[f+1]));
                        atomicAdd(&adv_ptr[f+2], p * __bfloat162float(do_row[f+2]));
                        atomicAdd(&adv_ptr[f+3], p * __bfloat162float(do_row[f+3]));
                    }
                }
            }
            __syncthreads();
            
            for (int i = tid; i < BLOCK_M * D; i += THREADS) {
                int r = i / D;
                int c = i % D;
                int q_g = ms + r;
                if (q_g < S) {
                    gpuAtomicAdd(&dq_base[(int64_t)q_g * D + c], __float2bfloat16(a_dQ[i]));
                }
            }
            __syncthreads();
        }
        
        for (int i = tid; i < BLOCK_N * D; i += THREADS) {
            int r = i / D;
            int c = i % D;
            int k_g = ns + r;
            if (k_g < S) {
                gpuAtomicAdd(&dk_base[(int64_t)k_g * D + c], __float2bfloat16(a_dK[i]));
                gpuAtomicAdd(&dv_base[(int64_t)k_g * D + c], __float2bfloat16(a_dV[i]));
            }
        }
        __syncthreads();
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
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    size_t elem_count = (size_t)B * H * S * d;
    CUDA_CHECK(cudaMemset(dQ_ptr, 0, elem_count * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(dK_ptr, 0, elem_count * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(dV_ptr, 0, elem_count * sizeof(__nv_bfloat16)));
    
    dim3 grid(B * H);
    dim3 block(THREADS);
    
    size_t smem_bytes = (BLOCK_M * d + BLOCK_N * d + BLOCK_N * d + BLOCK_M * d) * sizeof(__nv_bfloat16) +
                        (BLOCK_M * d + BLOCK_N * d + BLOCK_N * d) * sizeof(float);
    
    float scale = 1.0f / sqrtf((float)d);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
        
    mha_bwd_kernel<128><<<grid, block, smem_bytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr, dK_ptr, dV_ptr, (int)B, (int)H, (int)S, scale);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace mha_bwd_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);