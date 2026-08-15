#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <math.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__global__ void causal_mha_lse_kernel(
    __nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    uint32_t S, uint32_t H, uint32_t B) 
{
    uint32_t tiles_per_block = (S + 127) / 128;
    uint32_t bh_idx = blockIdx.x / tiles_per_block;
    uint32_t q_tile_idx = blockIdx.x % tiles_per_block;
    uint32_t q_start = q_tile_idx * 128;
    int head_idx = bh_idx % H;
    int batch_idx = bh_idx / H;
    int row = threadIdx.x;
    uint32_t global_q_idx = q_start + row;
    
    if (q_start >= S) return;
    
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_K0 = (__nv_bfloat16*)smem_pool;           
    __nv_bfloat16* smem_K1 = (__nv_bfloat16*)(smem_pool + 16384); 
    __nv_bfloat16* smem_V0 = (__nv_bfloat16*)(smem_pool + 32768); 
    __nv_bfloat16* smem_V1 = (__nv_bfloat16*)(smem_pool + 49152); 
    __nv_bfloat16* smem_P  = (__nv_bfloat16*)(smem_pool + 65536); // 32KB allocated perfectly for P storage
    
    __nv_bfloat16 my_Q[128];
    if (global_q_idx < S) {
        const __nv_bfloat16* q_ptr = Q + (batch_idx * H + head_idx) * S * 128 + global_q_idx * 128;
        for (int i = 0; i < 128 / 8; ++i) {
            *(uint4*)&my_Q[i * 8] = *(uint4*)&q_ptr[i * 8];
        }
    } else {
        for (int i = 0; i < 128; ++i) my_Q[i] = __float2bfloat16(0.0f);
    }
    
    float m_old = -INFINITY;
    float l_old = 0.0f;
    
    __nv_bfloat16 my_O[128];
    for (int i = 0; i < 128; ++i) my_O[i] = __float2bfloat16(0.0f);
    
    uint32_t q_end = min(q_start + 128, S);
    uint32_t max_k = min(q_end - 1, (uint32_t)(q_start + 127));
    
    int warp_id = row / 32;
    int lane_idx = row % 32;
    
    for (uint32_t k_start = 0; k_start <= max_k; k_start += 128) {
        for (int i = lane_idx; i < 32 * 8; i += 32) {
            int load_row = warp_id * 32 + i / 8;
            int chunk = i % 8;
            uint32_t gk = k_start + load_row;
            if (gk < S) {
                *(uint4*)&smem_K0[load_row * 64 + chunk * 8] = *(uint4*)&K[bh_idx * S * 128 + gk * 128 + chunk * 8];
                *(uint4*)&smem_V0[load_row * 64 + chunk * 8] = *(uint4*)&V[bh_idx * S * 128 + gk * 128 + chunk * 8];
                
                *(uint4*)&smem_K1[load_row * 64 + chunk * 8] = *(uint4*)&K[bh_idx * S * 128 + gk * 128 + 64 + chunk * 8];
                *(uint4*)&smem_V1[load_row * 64 + chunk * 8] = *(uint4*)&V[bh_idx * S * 128 + gk * 128 + 64 + chunk * 8];
            } else {
                *(uint4*)&smem_K0[load_row * 64 + chunk * 8] = {0,0,0,0};
                *(uint4*)&smem_V0[load_row * 64 + chunk * 8] = {0,0,0,0};
                
                *(uint4*)&smem_K1[load_row * 64 + chunk * 8] = {0,0,0,0};
                *(uint4*)&smem_V1[load_row * 64 + chunk * 8] = {0,0,0,0};
            }
        }
        __syncthreads();
        
        float s_reg[128];
        for (int k = 0; k < 128; ++k) {
            float sum = 0;
            for (int d = 0; d < 64; d += 2) {
                sum += __bfloat162float(my_Q[d]) * __bfloat162float(smem_K0[k * 64 + d]);
                sum += __bfloat162float(my_Q[d+1]) * __bfloat162float(smem_K0[k * 64 + d + 1]);
            }
            for (int d = 0; d < 64; d += 2) {
                sum += __bfloat162float(my_Q[64 + d]) * __bfloat162float(smem_K1[k * 64 + d]);
                sum += __bfloat162float(my_Q[64 + d + 1]) * __bfloat162float(smem_K1[k * 64 + d + 1]);
            }
            s_reg[k] = sum * 0.08838834764f; 
            
            uint32_t gk = k_start + k;
            if (global_q_idx >= S || gk > global_q_idx || gk >= S) {
                s_reg[k] = -INFINITY;
            }
        }
        
        float m_row = -INFINITY;
        for (int k = 0; k < 128; ++k) {
            m_row = fmaxf(m_row, s_reg[k]);
        }
        
        float l_row = 0.0f;
        for (int k = 0; k < 128; ++k) {
            if (m_row > -INFINITY) {
                float diff = m_row - s_reg[k];
                float e = fast_exp2f_fn(diff * 1.4426950408889634f);
                s_reg[k] = e;
                l_row += e;
            }
        }
        
        float m_new = fmaxf(m_old, m_row);
        float l_new = l_old * fast_exp2f_fn((m_old - m_new) * 1.4426950408889634f) + 
                      l_row * fast_exp2f_fn((m_row - m_new) * 1.4426950408889634f);
        
        float exp_m_old_m_new = fast_exp2f_fn((m_old - m_new) * 1.4426950408889634f);
        for(int d = 0; d < 128; d++) {
            float temp_o = __bfloat162float(my_O[d]);
            temp_o *= exp_m_old_m_new;
            my_O[d] = __float2bfloat16(temp_o);
        }
        
        for (int k = 0; k < 128; k += 4) {
            __nv_bfloat16 bf[4];
            if (m_new > -INFINITY) {
                float e0 = fast_exp2f_fn((m_row - s_reg[k]) * 1.4426950408889634f) * fast_exp2f_fn((m_row - m_new) * 1.4426950408889634f);
                float e1 = fast_exp2f_fn((m_row - s_reg[k+1]) * 1.4426950408889634f) * fast_exp2f_fn((m_row - m_new) * 1.4426950408889634f);
                float e2 = fast_exp2f_fn((m_row - s_reg[k+2]) * 1.4426950408889634f) * fast_exp2f_fn((m_row - m_new) * 1.4426950408889634f);
                float e3 = fast_exp2f_fn((m_row - s_reg[k+3]) * 1.4426950408889634f) * fast_exp2f_fn((m_row - m_new) * 1.4426950408889634f);
                bf[0] = __float2bfloat16(e0);
                bf[1] = __float2bfloat16(e1);
                bf[2] = __float2bfloat16(e2);
                bf[3] = __float2bfloat16(e3);
            } else {
                bf[0] = __float2bfloat16(0.0f);
                bf[1] = __float2bfloat16(0.0f);
                bf[2] = __float2bfloat16(0.0f);
                bf[3] = __float2bfloat16(0.0f);
            }
            *(uint32_t*)&smem_P[row * 128 + k] = *(uint32_t*)&bf[0];
        }
        __syncthreads();
        
        for (int d = 0; d < 64; ++d) {
            float sum0 = 0, sum1 = 0;
            for (int k = 0; k < 128; k += 2) {
                float p0 = __bfloat162float(smem_P[row * 128 + k]);
                float p1 = __bfloat162float(smem_P[row * 128 + k + 1]);
                
                sum0 += p0 * __bfloat162float(smem_V0[k * 64 + d]) + p1 * __bfloat162float(smem_V0[(k+1) * 64 + d]);
                sum1 += p0 * __bfloat162float(smem_V1[k * 64 + d]) + p1 * __bfloat162float(smem_V1[(k+1) * 64 + d]);
            }
            float temp_o0 = __bfloat162float(my_O[d]);
            temp_o0 += sum0;
            my_O[d] = __float2bfloat16(temp_o0);
            
            float temp_o1 = __bfloat162float(my_O[64 + d]);
            temp_o1 += sum1;
            my_O[64 + d] = __float2bfloat16(temp_o1);
        }
        
        m_old = m_new;
        l_old = l_new;
        __syncthreads();
    }
    
    if (global_q_idx < S) {
        for (int d = 0; d < 128; ++d) {
            float temp_o = __bfloat162float(my_O[d]);
            temp_o /= l_old;
            my_O[d] = __float2bfloat16(temp_o);
        }
        
        __nv_bfloat16* out_O = O + (batch_idx * H + head_idx) * S * 128 + global_q_idx * 128;
        for (int d = 0; d < 128 / 8; ++d) {
            *(uint4*)&out_O[d * 8] = *(uint4*)&my_O[d * 8];
        }
        
        LSE[(batch_idx * H + head_idx) * S + global_q_idx] = m_old + logf(l_old);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t D = Q.size(3);

    uint32_t* o_mem = (uint32_t*)O.data_ptr();
    cudaMemsetAsync(o_mem, 0, B * H * S * D * sizeof(__nv_bfloat16), static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id)));

    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    uint32_t tiles_per_block = (S + 127) / 128;
    dim3 grid(tiles_per_block * B * H);
    dim3 block(128);

    CUDA_CHECK(cudaFuncSetAttribute(
        causal_mha_lse_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        100000
    ));

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    causal_mha_lse_kernel<<<grid, block, 100000, stream>>>(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, S, H, B);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda