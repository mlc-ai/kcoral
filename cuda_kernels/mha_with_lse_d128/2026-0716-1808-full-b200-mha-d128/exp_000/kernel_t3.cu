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

namespace tvm_ffi_example_cuda {

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ int swizzle_128B(int row, int col) {
    int chunk_x = col / 8;
    int swizzled_chunk = (row % 8) ^ chunk_x;
    return row * 128 + swizzled_chunk * 8 + (col % 8);
}

__device__ __forceinline__ int swizzle_128B_vec_idx(int row, int col_vec) {
    int swizzled_chunk = (row % 8) ^ col_vec;
    return row * 128 + swizzled_chunk * 8;
}

extern __shared__ __align__(128) uint8_t smem_pool[];

__global__ void __launch_bounds__(128) attention_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE, int S)
{
    setmaxnreg_inc_sync_fn<256>();
    
    int q_start = blockIdx.x * 128;
    int bh = blockIdx.y;
    
    if (q_start >= S) return;
    
    int tid = threadIdx.x; 
    
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)(smem_pool + 0);               
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem_pool + 32768);              
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem_pool + 65536);              
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem_pool + 98304);              
    __nv_bfloat16* smem_out = (__nv_bfloat16*)(smem_pool + 131072);           
    float* smem_O = (float*)(smem_pool + 163840);                           
    
    float* smem_S = (float*)(smem_pool + 98304);                            
    float* smem_m_prev = (float*)(smem_pool + 164352);                       
    float* smem_l_prev = (float*)(smem_pool + 164864);                       
    
    __device__ __forceinline__ void load_gmem_to_smem_vec(
        __nv_bfloat16* smem, const __nv_bfloat16* gmem, 
        int num_rows, int num_cols, int stride, int max_rows) 
    {
        for (int i = threadIdx.x; i < num_rows * (num_cols / 8); i += blockDim.x) {
            int r = i / (num_cols / 8);
            int c_vec = i % (num_cols / 8);
            int swizzled_offset = swizzle_128B_vec_idx(r, c_vec);
            if (r < max_rows) {
                *(float2*)&smem[swizzled_offset] = *(float2*)&gmem[r * stride + c_vec * 8];
            } else {
                *(float2*)&smem[swizzled_offset] = make_float2(0, 0);
            }
        }
    }
    
    int max_q_rows = min(128, S - q_start);
    load_gmem_to_smem_vec(smem_Q, Q + bh * S * 128 + q_start * 128, 128, 128, 128, max_q_rows);
    
    if (tid < 128) {
        smem_m_prev[tid] = -1e20f;
        smem_l_prev[tid] = 0.0f;
        for (int col = 0; col < 128; col++) {
            smem_O[tid * 128 + col] = 0.0f;
        }
    }
    __syncthreads();
    
    float inv_sqrt_D = 1.0f / sqrtf(128.0f);
    
    uint64_t num_kv_blocks = (S + 127) / 128;
    
    for (uint64_t kv_start = 0; kv_start < num_kv_blocks * 128; kv_start += 128) {
        int max_kv_rows = min(128, S - kv_start);
        
        load_gmem_to_smem_vec(smem_K, K + bh * S * 128 + kv_start * 128, 128, 128, 128, max_kv_rows);
        load_gmem_to_smem_vec(smem_V, V + bh * S * 128 + kv_start * 128, 128, 128, 128, max_kv_rows);
        __syncthreads();
        
        float temp_S[128] = {0};
        for (int i = 0; i < 128; i += 8) {
            float2 q0 = *(float2*)&smem_Q[swizzle_128B_vec_idx(tid, i/8)];
            float2 q1 = *(float2*)&smem_Q[swizzle_128B_vec_idx(tid, i/8 + 1)];
            float2 q2 = *(float2*)&smem_Q[swizzle_128B_vec_idx(tid, i/8 + 2)];
            float2 q3 = *(float2*)&smem_Q[swizzle_128B_vec_idx(tid, i/8 + 3)];
            
            for (int j = 0; j < 128; j++) {
                float k0 = __bfloat162float(smem_K[swizzle_128B(j, i)]);
                float k1 = __bfloat162float(smem_K[swizzle_128B(j, i+1)]);
                float k2 = __bfloat162float(smem_K[swizzle_128B(j, i+2)]);
                float k3 = __bfloat162float(smem_K[swizzle_128B(j, i+3)]);
                float k4 = __bfloat162float(smem_K[swizzle_128B(j, i+4)]);
                float k5 = __bfloat162float(smem_K[swizzle_128B(j, i+5)]);
                float k6 = __bfloat162float(smem_K[swizzle_128B(j, i+6)]);
                float k7 = __bfloat162float(smem_K[swizzle_128B(j, i+7)]);
                
                temp_S[j] += (q0.x * k0 + q0.y * k1 + q1.x * k2 + q1.y * k3 + 
                              q2.x * k4 + q2.y * k5 + q3.x * k6 + q3.y * k7) * inv_sqrt_D;
            }
        }
        
        float max_val = -1e20f;
        for (int j = 0; j < 128; j++) {
            int kv_idx = kv_start + j;
            if (kv_idx >= S || temp_S[j] > max_val) {
                max_val = temp_S[j];
            }
        }
        
        float m_prev = smem_m_prev[tid];
        float m_new = fmaxf(m_prev, max_val);
        
        float rescale = 0.0f;
        if (m_new > -1e19f) {
            rescale = fast_exp2f_fn((m_prev - m_new) * 1.44269504f);
        }
        
        float l_prev = smem_l_prev[tid];
        smem_l_prev[tid] = l_prev * rescale;
        
        float sum_p = 0;
        for (int j = 0; j < 128; j++) {
            int kv_idx = kv_start + j;
            float val = temp_S[j];
            if (kv_idx >= S) val = -1e20f;
            float p = (val > -1e19f) ? fast_exp2f_fn((val - m_new) * 1.44269504f) : 0;
            sum_p += p;
            smem_l_prev[tid] += p; 
            
            temp_S[j] = p * fast_exp2f_fn((max_val - m_new) * 1.44269504f);
        }
        
        for (int col = 0; col < 128; col++) {
            smem_O[tid * 128 + col] *= rescale;
        }
        
        for (int j = 0; j < 128; j++) {
            smem_P[swizzle_128B(tid, j)] = __float2bfloat16(temp_S[j]);
        }
        
        smem_m_prev[tid] = m_new;
        
        __syncthreads();
        
        float temp_O[128];
        for (int i = 0; i < 128; i += 2) {
            temp_O[i] = smem_O[tid * 128 + i];
            temp_O[i+1] = smem_O[tid * 128 + i+1];
        }
        
        for (int j = 0; j < 128; j += 8) {
            float2 v0 = *(float2*)&smem_V[swizzle_128B_vec_idx(j, 0)];
            float2 v1 = *(float2*)&smem_V[swizzle_128B_vec_idx(j, 1)];
            float2 v2 = *(float2*)&smem_V[swizzle_128B_vec_idx(j, 2)];
            float2 v3 = *(float2*)&smem_V[swizzle_128B_vec_idx(j, 3)];
            
            for (int i = 0; i < 128; i += 8) {
                float p0 = __bfloat162float(smem_P[swizzle_128B(tid, i)]);
                float p1 = __bfloat162float(smem_P[swizzle_128B(tid, i+1)]);
                float p2 = __bfloat162float(smem_P[swizzle_128B(tid, i+2)]);
                float p3 = __bfloat162float(smem_P[swizzle_128B(tid, i+3)]);
                float p4 = __bfloat162float(smem_P[swizzle_128B(tid, i+4)]);
                float p5 = __bfloat162float(smem_P[swizzle_128B(tid, i+5)]);
                float p6 = __bfloat162float(smem_P[swizzle_128B(tid, i+6)]);
                float p7 = __bfloat162float(smem_P[swizzle_128B(tid, i+7)]);
                
                temp_O[i]     += p0 * v0.x;
                temp_O[i + 1] += p0 * v0.y;
                temp_O[i + 2] += p1 * v1.x;
                temp_O[i + 3] += p1 * v1.y;
                temp_O[i + 4] += p2 * v2.x;
                temp_O[i + 5] += p2 * v2.y;
                temp_O[i + 6] += p3 * v3.x;
                temp_O[i + 7] += p3 * v3.y;
            }
        }
        
        for (int col = 0; col < 128; col += 2) {
            smem_O[tid * 128 + col] = temp_O[col];
            smem_O[tid * 128 + col + 1] = temp_O[col + 1];
        }
        
        __syncthreads();
    }
    
    for (int i = tid; i < 128 * 16; i += blockDim.x) {
        int r = i / 16;
        int c = (i % 16) * 8;
        float val = smem_O[r * 128 + c];
        
        if (smem_l_prev[r] > 0) {
            val /= smem_l_prev[r];
        }
        
        *(float2*)&smem_out[swizzle_128B_vec_idx(r, i % 16)] = make_float2(__float2bfloat16(val), __float2bfloat16(smem_O[r * 128 + c + 1] / smem_l_prev[r]));
    }
    __syncthreads();
    
    uint64_t offset_O = (bh * S + q_start) * 128;
    
    for (int i = tid; i < 128 * 16; i += blockDim.x) {
        int r = i / 16;
        int c = (i % 16) * 8;
        
        if (q_start + r < S) {
            *(float2*)&O[offset_O + r * 128 + c] = *(float2*)&smem_out[swizzle_128B_vec_idx(r, i % 16)];
        }
    }
    
    if (q_start + tid < S) {
        LSE[bh * S + q_start + tid] = smem_m_prev[tid] + __logf(smem_l_prev[tid]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  
  int64_t B = Q.size(0);
  int64_t H = Q.size(1);
  int64_t S = Q.size(2);
  int64_t D = Q.size(3);
  
  if (D != 128) {
    fprintf(stderr, "Expected D=128, got D=%ld\n", D);
    exit(1);
  }
  
  const __nv_bfloat16* g_Q = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* g_K = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* g_V = static_cast<const __nv_bfloat16*>(V.data_ptr());
  
  __nv_bfloat16* g_O = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* g_LSE = static_cast<float*>(LSE.data_ptr());
  
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  
  dim3 grid((S + 127) / 128, B * H);
  dim3 block(128);
  
  int smem_size = 228 * 1024;
  CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  
  attention_kernel<<<grid, block, smem_size, stream>>>(g_Q, g_K, g_V, g_O, g_LSE, S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda