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

#define BLOCK_S 16
#define THREADS 128
#define HEAD_DIM 128
#define PAD_D 132

namespace mha_bwd_impl {

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

    int64_t base = (int64_t)(b * H + h) * S * HEAD_DIM;
    const __nv_bfloat16* q_base = Q + base;
    const __nv_bfloat16* k_base = K + base;
    const __nv_bfloat16* v_base = V + base;
    const __nv_bfloat16* do_base = dO + base;
    const float* l_base = L + (b * H + h) * S;
    __nv_bfloat16* dq_base = dQ + base;
    __nv_bfloat16* dk_base = dK + base;
    __nv_bfloat16* dv_base = dV + base;

    extern __shared__ char smem[];
    
    // Shared memory layout with padding to avoid bank conflicts
    __nv_bfloat16* s_Q = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* s_K = s_Q + BLOCK_S * PAD_D;
    __nv_bfloat16* s_V = s_K + BLOCK_S * PAD_D;
    __nv_bfloat16* s_dO = s_V + BLOCK_S * PAD_D;
    
    float* a_dQ = reinterpret_cast<float*>(s_dO + BLOCK_S * PAD_D);
    float* a_dK = a_dQ + BLOCK_S * HEAD_DIM;
    float* a_dV = a_dK + BLOCK_S * HEAD_DIM;

    int tid = threadIdx.x;

    // Iterate over Key sequence in tiles
    for (int ks = 0; ks < S; ks += BLOCK_S) {
        int k_valid = min(BLOCK_S, S - ks);

        // Clear dK, dV accumulators
        for (int i = tid; i < BLOCK_S * HEAD_DIM; i += THREADS) {
            a_dK[i] = 0.0f;
            a_dV[i] = 0.0f;
        }
        __syncthreads();

        // Load K tile
        for (int r = tid; r < BLOCK_S; r += THREADS) {
            int g_r = ks + r;
            __nv_bfloat16* srow = s_K + r * PAD_D;
            const __nv_bfloat16* grow = k_base + (int64_t)g_r * HEAD_DIM;
            for (int c = 0; c < HEAD_DIM; c += 4) {
                if (g_r < S) {
                    ((float4*)srow)[c>>2] = reinterpret_cast<const float4*>(&grow[c])[0];
                } else {
                    ((float4*)srow)[c>>2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }
        // Load V tile
        for (int r = tid; r < BLOCK_S; r += THREADS) {
            int g_r = ks + r;
            __nv_bfloat16* srow = s_V + r * PAD_D;
            const __nv_bfloat16* grow = v_base + (int64_t)g_r * HEAD_DIM;
            for (int c = 0; c < HEAD_DIM; c += 4) {
                if (g_r < S) {
                    ((float4*)srow)[c>>2] = reinterpret_cast<const float4*>(&grow[c])[0];
                } else {
                    ((float4*)srow)[c>>2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }
        __syncthreads();

        // Iterate over Query sequence in tiles
        for (int qs = 0; qs < S; qs += BLOCK_S) {
            int q_valid = min(BLOCK_S, S - qs);

            // Clear dQ accumulator
            for (int i = tid; i < BLOCK_S * HEAD_DIM; i += THREADS) {
                a_dQ[i] = 0.0f;
            }
            __syncthreads();

            // Load Q tile
            for (int r = tid; r < BLOCK_S; r += THREADS) {
                int g_r = qs + r;
                __nv_bfloat16* srow = s_Q + r * PAD_D;
                const __nv_bfloat16* grow = q_base + (int64_t)g_r * HEAD_DIM;
                for (int c = 0; c < HEAD_DIM; c += 4) {
                    if (g_r < S) {
                        ((float4*)srow)[c>>2] = reinterpret_cast<const float4*>(&grow[c])[0];
                    } else {
                        ((float4*)srow)[c>>2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    }
                }
            }
            // Load dO tile
            for (int r = tid; r < BLOCK_S; r += THREADS) {
                int g_r = qs + r;
                __nv_bfloat16* srow = s_dO + r * PAD_D;
                const __nv_bfloat16* grow = do_base + (int64_t)g_r * HEAD_DIM;
                for (int c = 0; c < HEAD_DIM; c += 4) {
                    if (g_r < S) {
                        ((float4*)srow)[c>>2] = reinterpret_cast<const float4*>(&grow[c])[0];
                    } else {
                        ((float4*)srow)[c>>2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    }
                }
            }
            __syncthreads();

            // Compute gradients pairwise
            int pairs = BLOCK_S * BLOCK_S;
            for (int assign = tid; assign < pairs; assign += THREADS) {
                int qr = assign / BLOCK_S;
                int kr = assign % BLOCK_S;

                int q_g = qs + qr;
                int k_g = ks + kr;

                // Causal mask & boundary
                if (q_g < S && k_g < S && k_g <= q_g) {
                    float qk_dot = 0.0f;
                    float dov_dot = 0.0f;

                    const __nv_bfloat16* q_row = s_Q + qr * PAD_D;
                    const __nv_bfloat16* k_row = s_K + kr * PAD_D;
                    const __nv_bfloat16* v_row = s_V + kr * PAD_D;
                    const __nv_bfloat16* do_row = s_dO + qr * PAD_D;

                    #pragma unroll
                    for (int f = 0; f < HEAD_DIM; f += 4) {
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

                    float* adq_ptr = a_dQ + qr * HEAD_DIM;
                    float* adk_ptr = a_dK + kr * HEAD_DIM;
                    float* adv_ptr = a_dV + kr * HEAD_DIM;

                    #pragma unroll
                    for (int f = 0; f < HEAD_DIM; f += 4) {
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

            // Flush accumulated dQ for this query tile to global memory
            for (int i = tid; i < BLOCK_S * HEAD_DIM; i += THREADS) {
                int r = i / HEAD_DIM;
                int c = i % HEAD_DIM;
                int q_g = qs + r;
                if (q_g < S) {
                    dq_base[(int64_t)q_g * HEAD_DIM + c] = __float2bfloat16(a_dQ[i]);
                }
            }
            __syncthreads();
        }

        // Flush accumulated dK and dV for this key tile to global memory
        for (int i = tid; i < BLOCK_S * HEAD_DIM; i += THREADS) {
            int r = i / HEAD_DIM;
            int c = i % HEAD_DIM;
            int k_g = ks + r;
            if (k_g < S) {
                dk_base[(int64_t)k_g * HEAD_DIM + c] = __float2bfloat16(a_dK[i]);
                dv_base[(int64_t)k_g * HEAD_DIM + c] = __float2bfloat16(a_dV[i]);
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
    
    // Initialize outputs to zero
    size_t elem_count = (size_t)B * H * S * d;
    CUDA_CHECK(cudaMemset(dQ_ptr, 0, elem_count * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(dK_ptr, 0, elem_count * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(dV_ptr, 0, elem_count * sizeof(__nv_bfloat16)));
    
    dim3 grid(B * H);
    dim3 block(THREADS);
    
    // Shared memory calculation: 4 bf16 tiles + 3 fp32 accumulators
    size_t smem_bytes = BLOCK_S * PAD_D * sizeof(__nv_bfloat16) * 4 + 
                        BLOCK_S * HEAD_DIM * sizeof(float) * 3;
    
    float scale = 1.0f / sqrtf((float)d);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
        
    mha_bwd_kernel<<<grid, block, smem_bytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr, dK_ptr, dV_ptr, (int)B, (int)H, (int)S, scale);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace mha_bwd_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);