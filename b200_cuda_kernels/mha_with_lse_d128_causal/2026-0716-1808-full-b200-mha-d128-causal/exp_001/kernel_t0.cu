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

__device__ __forceinline__ void load_q_tile(const __nv_bfloat16* Q, __nv_bfloat16* q_shared, int abs_m_start, int S_len, int bh_idx) {
    for (int i = 0; i < 128 * 128 / 4; i += 128) {
        int flat_idx = i + threadIdx.x;
        int elem_idx = flat_idx * 4;
        int row = elem_idx / 128;
        int col = elem_idx % 128;
        int abs_row = abs_m_start + row;
        
        uint2 val = {0, 0};
        if (abs_row < S_len && col + 3 < 128) {
            val = *reinterpret_cast<const uint2*>(&Q[bh_idx * S_len * 128 + abs_row * 128 + col]);
        }
        *reinterpret_cast<uint2*>(&q_shared[elem_idx]) = val;
    }
}

__device__ __forceinline__ void load_kv_tile(const __nv_bfloat16* K, const __nv_bfloat16* V, __nv_bfloat16* k_shared, __nv_bfloat16* v_shared, int abs_n_start, int S_len, int bh_idx) {
    for (int i = 0; i < 128 * 128 / 4; i += 128) {
        int flat_idx = i + threadIdx.x;
        int elem_idx = flat_idx * 4;
        int row = elem_idx / 128;
        int col = elem_idx % 128;
        int abs_row = abs_n_start + row;
        
        uint2 k_val = {0, 0};
        uint2 v_val = {0, 0};
        if (abs_row < S_len && col + 3 < 128) {
            k_val = *reinterpret_cast<const uint2*>(&K[bh_idx * S_len * 128 + abs_row * 128 + col]);
            v_val = *reinterpret_cast<const uint2*>(&V[bh_idx * S_len * 128 + abs_row * 128 + col]);
        }
        *reinterpret_cast<uint2*>(&k_shared[elem_idx]) = k_val;
        *reinterpret_cast<uint2*>(&v_shared[elem_idx]) = v_val;
    }
}

__device__ __forceinline__ void compute_qkt(const __nv_bfloat16* q_shared, const __nv_bfloat16* k_shared, float* s_shared, int tid) {
    int row = tid;
    for (int col = 0; col < 128; col++) {
        float sum = 0;
        for (int k = 0; k < 128; k++) {
            sum += __bfloat162float(q_shared[row * 128 + k]) * __bfloat162float(k_shared[col * 128 + k]);
        }
        s_shared[row * 128 + col] = sum;
    }
}

__device__ __forceinline__ void softmax_step(const float* s_shared, __nv_bfloat16* p_shared, float* o_acc, float* m_val, float* l_val, int abs_m_start, int abs_n_start, int S_len, float scale, int tid) {
    int row = tid;
    int abs_row = abs_m_start + row;
    
    float row_max = -INFINITY;
    for (int col = 0; col < 128; col++) {
        int abs_col = abs_n_start + col;
        if (abs_row >= S_len || abs_col > abs_row || abs_col >= S_len) {
            row_max = fmaxf(row_max, -INFINITY);
        } else {
            row_max = fmaxf(row_max, s_shared[row * 128 + col] * scale);
        }
    }
    
    float new_m = fmaxf(*m_val, row_max);
    float o_s = __expf(*m_val - new_m);
    
    float row_sum = 0;
    for (int col = 0; col < 128; col++) {
        int abs_col = abs_n_start + col;
        if (abs_row >= S_len || abs_col > abs_row || abs_col >= S_len) {
            p_shared[row * 128 + col] = __float2bfloat16(0.0f);
        } else {
            float p = __expf(s_shared[row * 128 + col] * scale - new_m);
            p_shared[row * 128 + col] = __float2bfloat16(p);
            row_sum += p;
        }
    }
    
    *l_val = (*l_val) * o_s + row_sum * __expf(row_max - new_m);
    *m_val = new_m;
    
    for (int d = 0; d < 128; d++) {
        o_acc[d] *= o_s;
    }
}

__device__ __forceinline__ void compute_pv(const __nv_bfloat16* p_shared, const __nv_bfloat16* v_shared, float* o_acc, int tid) {
    int row = tid;
    for (int d = 0; d < 128; d++) {
        float sum = 0;
        for (int j = 0; j < 128; j++) {
            sum += __bfloat162float(p_shared[row * 128 + j]) * __bfloat162float(v_shared[j * 128 + d]);
        }
        o_acc[d] += sum;
    }
}

__global__ void __launch_bounds__(128, 1) attention_forward_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE, int S_len, float scale)
{
    int m_block = blockIdx.x;
    int bh_idx = blockIdx.y;
    int abs_m_start = m_block * 128;
    
    if (abs_m_start >= S_len) return;
    
    extern __shared__ __align__(128) char smem_pool[];
    char* smem_ptr = smem_pool;
    
    __nv_bfloat16* q_shared = (__nv_bfloat16*)smem_ptr;
    smem_ptr += 128 * 128 * sizeof(__nv_bfloat16);
    
    __nv_bfloat16* k_shared = (__nv_bfloat16*)smem_ptr;
    smem_ptr += 128 * 128 * sizeof(__nv_bfloat16);
    
    __nv_bfloat16* v_shared = (__nv_bfloat16*)smem_ptr;
    smem_ptr += 128 * 128 * sizeof(__nv_bfloat16);
    
    __nv_bfloat16* p_shared = (__nv_bfloat16*)smem_ptr;
    smem_ptr += 128 * 128 * sizeof(__nv_bfloat16);
    
    float* s_shared = (float*)smem_ptr;
    
    int tid = threadIdx.x;
    
    load_q_tile(Q, q_shared, abs_m_start, S_len, bh_idx);
    __syncthreads();
    
    float o_acc[128] = {0};
    float m_val = -INFINITY;
    float l_val = 0;
    
    for (int n_block = 0; n_block <= m_block; n_block++) {
        load_kv_tile(K, V, k_shared, v_shared, n_block * 128, S_len, bh_idx);
        __syncthreads();
        
        compute_qkt(q_shared, k_shared, s_shared, tid);
        __syncthreads();
        
        softmax_step(s_shared, p_shared, o_acc, &m_val, &l_val, abs_m_start, n_block * 128, S_len, scale, tid);
        __syncthreads();
        
        compute_pv(p_shared, v_shared, o_acc, tid);
    }
    
    int abs_row = abs_m_start + tid;
    if (abs_row < S_len) {
        for (int d = 0; d < 128; d += 4) {
            __nv_bfloat16 out[4];
            out[0] = __float2bfloat16(o_acc[d] / l_val);
            out[1] = __float2bfloat16(o_acc[d+1] / l_val);
            out[2] = __float2bfloat16(o_acc[d+2] / l_val);
            out[3] = __float2bfloat16(o_acc[d+3] / l_val);
            *reinterpret_cast<uint2*>(&O[bh_idx * S_len * 128 + abs_row * 128 + d]) = *reinterpret_cast<uint2*>(out);
        }
        
        LSE[bh_idx * S_len + abs_row] = m_val + __logf(l_val);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, 
         tvm::ffi::TensorView V, tvm::ffi::TensorView O, 
         tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());
    
    float scale = 1.0f / std::sqrt((float)D);
    
    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    
    int smem_size = 128 * 128 * 2 * 4 + 128 * 128 * 4; 
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(
        attention_forward_kernel, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        smem_size));
    
    attention_forward_kernel<<<grid, block, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, S, scale);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda