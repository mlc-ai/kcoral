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

namespace mha_with_lse_d128 {

#define bf16x uint32_t

__device__ __forceinline__ float __low(float2 x) {
    asm volatile("cvt.f32.b16 %0, %1;" : "=f"(x) : "h"(*(uint16_t*)&x));
    return x;
}

__device__ __forceinline__ float __high(float2 x) {
    asm volatile("cvt.f32.b16 %0, %1;" : "=f"(x) : "h"(*(uint16_t*)&((uint32_t&)x)[1]));
    return x;
}

__device__ __forceinline__ float __exp2f(float x) {
    float y;
    int n = floorf(x);
    float f = x - n;
    float p = 0.0771f * f + 0.2276f;
    p = p * f + 0.6951f;
    p = p * f + 1.0f;
    int exponent = (n + 127) << 23;
    float bn = __int_as_float(exponent);
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(f));
    return __float_as_int(y + bn - __float_as_int(1<<23));
}

__device__ __forceinline__ float __exp2f_to_lse(float x) {
    float log2x = __logf(x);
    return log2x * 1.44269504089f;
}

__device__ __forceinline__ void load_swizzled_128B_async(__nv_bfloat16* smem, int s_row, const __nv_bfloat16* gmem, int S, int tid, int s_col, int stride, int chunk_y_offset, int chunk_y_step, int chunk_x_step, int elem_step) {
    if (s_row < S) {
        *(uint2*)(&smem[s_col]) = *(uint2*)(&gmem[s_row * stride + (s_col % 128)]);
    } else {
        *(uint2*)(&smem[s_col]) = make_uint2(0, 0);
    }
}

__global__ __launch_bounds__(128, 1) void mha_kernel(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int S) {
    int bh = blockIdx.x;
    int q_tile = blockIdx.y;
    int tid = threadIdx.x;
    int q_start = q_tile * 128;
    
    if (q_start >= S) return;
    
    const __nv_bfloat16* g_Q = Q + bh * S * 128 + q_start * 128;
    const __nv_bfloat16* g_K = K + bh * S * 128;
    const __nv_bfloat16* g_V = V + bh * S * 128;
    __nv_bfloat16* g_O = O + bh * S * 128 + q_start * 128;
    
    extern __shared__ __align__(1024) char smem[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)(smem + 0);
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem + 32768);
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem + 65536);
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem + 98304);
    __nv_bfloat16* smem_O = (__nv_bfloat16*)(smem + 131072);
    __nv_bfloat16* smem_S = (__nv_bfloat16*)(smem + 163840); 
    
    uint64_t* bar_Q = (uint64_t*)(smem + 180224);
    uint64_t* bar_K = (uint64_t*)(smem + 180232);
    uint64_t* bar_V = (uint64_t*)(smem + 180240);

    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(bar_Q)), "r"(1));
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(bar_K)), "r"(1));
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(bar_V)), "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    
    if (tid == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(bar_Q)), "r"(32768));
        for (int i = 0; i < 128; i += 4) {
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                :: "r"((uint32_t)__cvta_generic_to_shared(smem_Q + i)), "l"(g_Q + i), "r"(8), "r"((uint32_t)__cvta_generic_to_shared(bar_Q)) : "memory");
        }
    }
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_Q_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_Q_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar_Q)), "r"(0));
    
    float O_acc[128] = {0};
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    
    int phase_k = 0;
    int phase_v = 0;
    int num_k_tiles = (S + 127) / 128;
    
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        int k_start = k_tile * 128;
        
        if (tid == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(bar_K)), "r"(32768));
            for (int i = 0; i < 128; i += 4) {
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                    :: "r"((uint32_t)__cvta_generic_to_shared(smem_K + i)), "l"(g_K + k_start * 128 + i), "r"(8), "r"((uint32_t)__cvta_generic_to_shared(bar_K)) : "memory");
            }
            
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(bar_V)), "r"(32768));
            for (int i = 0; i < 128; i += 4) {
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                    :: "r"((uint32_t)__cvta_generic_to_shared(smem_V + i)), "l"(g_V + k_start * 128 + i), "r"(8), "r"((uint32_t)__cvta_generic_to_shared(bar_V)) : "memory");
            }
        }
        
        asm volatile(
            "{\n"
            ".reg .pred P;\n"
            "WAIT_K_%=:\n"
            "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
            "@!P bra WAIT_K_%=;\n"
            "}\n"
            :: "r"((uint32_t)__cvta_generic_to_shared(bar_K)), "r"(phase_k));
            
        asm volatile(
            "{\n"
            ".reg .pred P;\n"
            "WAIT_V_%=:\n"
            "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
            "@!P bra WAIT_V_%=;\n"
            "}\n"
            :: "r"((uint32_t)__cvta_generic_to_shared(bar_V)), "r"(phase_v));
            
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
        
        { 
            float S_acc[128] = {0};
            
            for (int n_block = 0; n_block < 128; n_block += 16) {
                float S_acc_n[2] = {0, 0};
                int swizzled_n_chunk_base_x = ((n_block) / 8) ^ (tid % 8);
                for (int d_block = 0; d_block < 128; d_block += 8) {
                    int swizzled_d_chunk_x = (d_block / 8) ^ (tid % 8);
                    
                    uint32_t K_val = *(uint32_t*)&smem_K[n_block * 128 + swizzled_n_chunk_base_x * 8 + (d_block % 8)];
                    uint32_t Q_val = *(uint32_t*)&smem_Q[tid * 128 + swizzled_d_chunk_x * 8 + (d_block % 8)];
                    S_acc_n[0] += __low(K_val) * __low(Q_val);
                    S_acc_n[1] += __high(K_val) * __high(Q_val);
                    
                    K_val = *(uint32_t*)&smem_K[n_block * 128 + swizzled_n_chunk_base_x * 8 + (d_block % 8) + 4];
                    Q_val = *(uint32_t*)&smem_Q[tid * 128 + swizzled_d_chunk_x * 8 + (d_block % 8) + 4];
                    S_acc_n[0] += __low(K_val) * __low(Q_val);
                    S_acc_n[1] += __high(K_val) * __high(Q_val);
                }
                
                int chunk_x = (n_block / 8) ^ (tid % 8);
                S_acc[chunk_x * 8 + (n_block % 8)] = S_acc_n[0];
                S_acc[chunk_x * 8 + (n_block % 8) + 1] = S_acc_n[1];
            }
            
            float m_curr = -INFINITY;
            for(int i = 0; i < 128; ++i) {
                if (k_start + i >= S) {
                    S_acc[i] = -INFINITY;
                } else {
                    S_acc[i] *= 0.08838834764f; // 1/sqrt(128)
                }
                m_curr = fmaxf(m_curr, S_acc[i]);
            }
            
            float m_new = fmaxf(m_prev, m_curr);
            float l_new = 0.0f;
            for(int i = 0; i < 128; ++i) {
                float p = 0.0f;
                if (k_start + i < S) {
                    p = __exp2f((S_acc[i] - m_new) * 1.44269504089f);
                }
                l_new += p;
                S_acc[i] = p;
            }
            l_new += l_prev * __exp2f((m_prev - m_new) * 1.44269504089f);
            float scale_o = __exp2f((m_prev - m_new) * 1.44269504089f);
            l_prev = l_new;
            m_prev = m_new;
            
            __syncthreads(); 
            
            for(int i = 0; i < 128; ++i) {
                int chunk_x = i / 8;
                int swizzled_chunk_x = chunk_x ^ (tid % 8);
                smem_P[tid * 128 + swizzled_chunk_x * 8 + (i % 8)] = __float2bfloat16(S_acc[i]);
            }
            __syncthreads();
            
            if (scale_o > 0.0f) {
                for(int i = 0; i < 128; ++i) {
                    O_acc[i] *= scale_o;
                }
            }
            
            for (int n_block = 0; n_block < 128; n_block += 16) {
                float O_acc_n[2] = {0, 0};
                int swizzled_n_chunk_x = (n_block / 8) ^ (tid % 8);
                for (int d_block = 0; d_block < 128; d_block += 8) {
                    int swizzled_d_chunk_x = (d_block / 8) ^ (tid % 8);
                    
                    uint32_t P_val = *(uint32_t*)&smem_P[tid * 128 + swizzled_n_chunk_x * 8 + (n_block % 8)];
                    uint32_t V_val = *(uint32_t*)&smem_V[n_block * 128 + swizzled_d_chunk_x * 8 + (d_block % 8)];
                    
                    O_acc_n[0] += __low(P_val) * __low(V_val);
                    O_acc_n[1] += __high(P_val) * __high(V_val);
                    
                    P_val = *(uint32_t*)&smem_P[tid * 128 + swizzled_n_chunk_x * 8 + (n_block % 8) + 4];
                    V_val = *(uint32_t*)&smem_V[n_block * 128 + swizzled_d_chunk_x * 8 + (d_block % 8) + 4];
                    
                    O_acc_n[0] += __low(P_val) * __low(V_val);
                    O_acc_n[1] += __high(P_val) * __high(V_val);
                }
                
                int chunk_x = (n_block / 8) ^ (tid % 8);
                O_acc[chunk_x * 8 + (n_block % 8)] += O_acc_n[0];
                O_acc[chunk_x * 8 + (n_block % 8) + 1] += O_acc_n[1];
            }
        }
        
        __syncthreads();
        phase_k ^= 1;
        phase_v ^= 1;
    }
    
    __syncthreads();
    for(int i = 0; i < 128; ++i) {
        int chunk_x = i / 8;
        int swizzled_chunk_x = chunk_x ^ (tid % 8);
        smem_O[tid * 128 + swizzled_chunk_x * 8 + (i % 8)] = __float2bfloat16(O_acc[i] / l_prev);
    }
    
    __syncthreads();
    
    int s_row_out = q_start + tid;
    if (s_row_out < S) {
        for (int chunk_x = 0; chunk_x < 16; ++chunk_x) {
            int swizzled_chunk_x = chunk_x ^ (tid % 8);
            int byte_offset = tid * 128 + swizzled_chunk_x * 8;
            *(uint2*)(&g_O[chunk_x * 8]) = *(uint2*)(&smem_O[byte_offset]);
        }
    }
    
    if (q_start + tid < S) {
        LSE[bh * S + q_start + tid] = m_prev + logf(__exp2f_to_lse(l_prev));
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());
    
    int num_q_tiles = (S + 127) / 128;
    dim3 grid(B * H, num_q_tiles);
    dim3 block(128);
    
    int smem_size = 180248;
    CUDA_CHECK(cudaFuncSetAttribute(
        mha_with_lse_d128::mha_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));
        
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    mha_kernel<<<grid, block, smem_size, stream>>>(Q_data, K_data, V_data, O_data, LSE_data, S);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace mha_with_lse_d128