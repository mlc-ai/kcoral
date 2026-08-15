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

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ uint16_t swizzled_col_128B(uint16_t row, uint16_t col) {
    return (((col >> 3) ^ (row & 7)) << 3) + (col & 7);
}

__device__ __forceinline__ void load_tile(const __nv_bfloat16* gmem_ptr, uint16_t* smem, uint64_t base_row, uint64_t S_len_scaled, uint32_t tid) {
    for (uint32_t i = tid; i < 8192; i += 128) {
        uint32_t row = i / 64;
        uint32_t col = (i % 64) * 2;
        if (base_row + row * 128 + col < S_len_scaled) {
            float2 val = *(const float2*)&gmem_ptr[base_row + row * 128 + col];
            uint16_t sw_col = swizzled_col_128B(row, col);
            smem[row * 128 + sw_col] = __low2bfloat16(val);
            smem[row * 128 + sw_col + 1] = __high2bfloat16(val);
        } else {
            uint16_t sw_col = swizzled_col_128B(row, col);
            smem[row * 128 + sw_col] = __float2bfloat16(0.0f);
            smem[row * 128 + sw_col + 1] = __float2bfloat16(0.0f);
        }
    }
}

__device__ __forceinline__ void transpose(uint16_t* smem_K, uint16_t* smem_tmp, uint32_t tid) {
    for (uint32_t i = tid; i < 8192; i += 128) {
        uint32_t row = i / 64;
        uint32_t col = i % 64;
        uint16_t val = smem_K[row * 128 + swizzled_col_128B(row, col * 2)];
        uint16_t sw_col_T = swizzled_col_128B(col, row);
        smem_tmp[col * 128 + sw_col_T] = val;
    }
}

__device__ __forceinline__ void clear_smem_S(float* smem_S, uint32_t tid) {
    for (uint32_t i = tid; i < 16384; i += 128) {
        smem_S[i] = 0.0f;
    }
}

template<uint32_t S_len>
__device__ __forceinline__ void compute_matmul(uint16_t* smem_Q_sw, uint16_t* smem_K_T_sw, float* smem_S_fp32, uint32_t q_start, uint32_t k_start, uint32_t tid) {
    for (int k = 0; k < 128; ++k) {
        float q = __bfloat162float(smem_Q_sw[tid * 128 + swizzled_col_128B(tid, k)]);
        for (int n = 0; n < 128; ++n) {
            float k_val = __bfloat162float(smem_K_T_sw[k * 128 + swizzled_col_128B(k, n)]);
            smem_S_fp32[tid * 128 + n] += q * k_val;
        }
    }
}

__device__ __forceinline__ bool include_in_softmax(uint32_t q_pos, uint32_t k_pos, uint32_t S_len) {
    return q_pos < S_len && q_pos >= k_pos;
}

template<uint32_t S_len>
__device__ __forceinline__ void compute_softmax_with_causal_mask(float* smem_S_fp32, uint16_t* smem_P_sw, 
                                                                  float& m_val, float& l_val, 
                                                                  uint32_t q_start, uint32_t k_start, float denom, 
                                                                  uint32_t tid, float (&out_reg_0)[128], float (&out_reg_1)[128]) {
    float row_max = -1e20f;
    for (int n = 0; n < 128; ++n) {
        float val = smem_S_fp32[tid * 128 + n];
        if (include_in_softmax(q_start + tid, k_start + n, S_len)) {
            val /= denom;
            row_max = fmaxf(row_max, val);
        }
        smem_S_fp32[tid * 128 + n] = val;
    }
    
    float m_new = fmaxf(m_val, row_max);
    float exp_old = expf(m_val - m_new);
    
    if (m_new > m_val) {
        for (int i = 0; i < 128; ++i) {
            out_reg_0[i] *= exp_old;
            out_reg_1[i] *= exp_old;
        }
    }
    
    float row_sum = 0.0f;
    for (int n = 0; n < 128; ++n) {
        float val = smem_S_fp32[tid * 128 + n];
        if (include_in_softmax(q_start + tid, k_start + n, S_len)) {
            float p = expf(val - m_new);
            row_sum += p;
            smem_P_sw[tid * 128 + swizzled_col_128B(tid, n)] = __float2bfloat16(p);
        } else {
            smem_P_sw[tid * 128 + swizzled_col_128B(tid, n)] = __float2bfloat16(0.0f);
        }
    }
    
    l_val = l_val * exp_old + row_sum;
    m_val = m_new;
}

template<uint32_t S_len>
__device__ __forceinline__ void compute_matmul_onto_out(float (&out_reg_0)[128], float (&out_reg_1)[128], uint16_t* smem_P_sw, uint16_t* smem_V_sw, uint32_t q_start, uint32_t k_start, uint32_t tid) {
    for (int k = 0; k < 128; ++k) {
        float p = __bfloat162float(smem_P_sw[k][tid]);
        for (int n = 0; n < 128; ++n) {
            float v = __bfloat162float(smem_V_sw[k * 128 + swizzled_col_128B(k, n)]);
            if (n < 64) out_reg_0[n] += p * v;
            else         out_reg_1[n - 64] += p * v;
        }
    }
}

__device__ __forceinline__ void store_out_half_vec(const __nv_bfloat16* O, uint32_t base_idx, float (&out_reg)[128], uint32_t S_len, uint32_t tid) {
    for (uint32_t i = tid; i < 64; i += 128) {
        uint32_t idx = base_idx + i;
        if (idx < S_len) {
            __nv_bfloat16 val = __float2bfloat16(out_reg[i]);
            *reinterpret_cast<__nv_bfloat16*>(O + idx * 128) = val;
        }
    }
}

__device__ __forceinline__ void store_lse(const float* m_val, const float* l_val, float* LSE, uint32_t base_idx, uint32_t S_len, uint32_t tid) {
    if (tid < 128) {
        uint32_t idx = base_idx + tid;
        if (idx < S_len) {
            if (l_val[tid] > 0.0f) {
                *reinterpret_cast<float*>(LSE + idx) = m_val[tid] + logf(l_val[tid]);
            } else {
                *reinterpret_cast<float*>(LSE + idx) = -1e20f;
            }
        }
    }
}

__global__ void __launch_bounds__(128, 2) attention_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    uint32_t S_len, uint32_t num_heads)
{
    uint32_t h = blockIdx.x;
    uint32_t b = blockIdx.y;
    uint64_t offset_bh = (static_cast<uint64_t>(b) * num_heads + h) * S_len * 128;
    uint32_t tid = threadIdx.x;

    extern __shared__ __align__(128) uint8_t smem_buf_raw[];
    uint16_t* smem_buf = (uint16_t*)smem_buf_raw;
    
    uint16_t* smem_Q = smem_buf;                   // [0 .. 16383]
    uint16_t* smem_K = smem_buf + 16384;            // [16384 .. 32767]
    uint16_t* smem_V = smem_buf + 32768;            // [32768 .. 49151]
    float* smem_S_fp32 = (float*)(smem_buf + 49152);// [49152 .. 81919] (98304 bytes)
    uint16_t* smem_P = smem_buf + 81920;            // [81920 .. 98303]
    uint16_t* smem_tmp = smem_buf + 98304;          // [98304 .. 114687]
    uint16_t* smem_K_T = smem_tmp;

    uint32_t D = 128;
    float denom = 1.0f / sqrtf((float)D);

    for (uint32_t q_start = 0; q_start < S_len; q_start += 128) {
        load_tile(Q, smem_Q, offset_bh + q_start * 128, S_len * 128, tid);
        __syncthreads();
        
        float m_val[128];
        float l_val[128];
        if (tid < 128) {
            m_val[tid] = -1e20f;
            l_val[tid] = 0.0f;
        }
        
        float out_reg_0[128] = {0};
        float out_reg_1[128] = {0};
        
        uint32_t S_and_q_start = (q_start + 128 < S_len) ? (q_start + 128) : S_len;
        
        for (uint32_t k_start = 0; k_start < S_and_q_start; k_start += 128) {
            load_tile(K, smem_K, offset_bh + k_start * 128, S_len * 128, tid);
            load_tile(V, smem_V, offset_bh + k_start * 128, S_len * 128, tid);
            __syncthreads();
            
            transpose(smem_K, smem_tmp, tid);
            __syncthreads();
            
            clear_smem_S(smem_S_fp32, tid);
            __syncthreads();
            
            compute_matmul<S_len>(smem_Q, smem_K_T, smem_S_fp32, q_start, k_start, tid);
            __syncthreads();
            
            compute_softmax_with_causal_mask<S_len>(smem_S_fp32, smem_P, m_val[tid], l_val[tid], q_start, k_start, denom, tid, out_reg_0, out_reg_1);
            __syncthreads();
            
            compute_matmul_onto_out<S_len>(out_reg_0, out_reg_1, smem_P, smem_V, q_start, k_start, tid);
        }
        
        if (l_val[tid] > 0.0f) {
            for(int i=0; i<128; ++i) {
                out_reg_0[i] /= l_val[tid];
                out_reg_1[i] /= l_val[tid];
            }
        }
        
        store_out_half_vec(O, offset_bh + q_start * 128, S_len * 128, S_len, out_reg_0, tid);
        store_out_half_vec(O, offset_bh + q_start * 128 + 64, S_len * 128, S_len, out_reg_1, tid);
        
        store_lse(m_val, l_val, LSE, (offset_bh / 128) + q_start, S_len, tid);
        __syncthreads();
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  
  uint32_t B = Q.size(0);
  uint32_t H = Q.size(1);
  uint32_t S = Q.size(2);
  uint32_t D = Q.size(3); 
  
  const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* LSE_ptr = static_cast<float*>(LSE.data_ptr());
  
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  
  dim3 grid(H, B);
  dim3 block(128);
  
  // Request 114688 + 32768 = 147456 bytes of dynamic shared memory
  // 114688 bytes for the 6 x 16384 uint16_t buffers
  // 32768 bytes for the 16384 float buffer
  CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 229376));
  
  attention_kernel<<<grid, block, 229376, stream>>>(Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, S, H);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda