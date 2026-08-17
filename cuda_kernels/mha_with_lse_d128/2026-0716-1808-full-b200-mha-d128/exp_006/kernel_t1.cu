#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <mma.h>
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

using namespace nvcuda;

__device__ void load_smem_padded(__nv_bfloat16 (*dst)[128], const __nv_bfloat16* src, int rows, int max_rows, int valid_rows, int tid) {
    for (int i = tid; i < max_rows * 128 / 2; i += 128) {
        int r = i / 2;
        int c = (i % 2) * 2;
        if (r < valid_rows) {
            uint2 val = *(reinterpret_cast<const uint2*>(src + r * 128 + c));
            *(reinterpret_cast<uint2*>(&dst[r][c])) = val;
        } else {
            *(reinterpret_cast<uint2*>(&dst[r][c])) = {0, 0};
        }
    }
}

__global__ void flash_attention_kernel(
    const __nv_bfloat16* __restrict__ Q, 
    const __nv_bfloat16* __restrict__ K, 
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O, 
    float* __restrict__ LSE,
    int S_len)
{
    int batch_head = blockIdx.x; 
    int query_start = blockIdx.y * 64;
    int tid = threadIdx.x;
    
    if (query_start >= S_len) return;
    
    int m = min(64, S_len - query_start);
    const __nv_bfloat16* Q_bh = Q + batch_head * S_len * 128;
    const __nv_bfloat16* K_bh = K + batch_head * S_len * 128;
    const __nv_bfloat16* V_bh = V + batch_head * S_len * 128;
    
    extern __shared__ __align__(128) char smem[];
    
    __nv_bfloat16 (*smem_K)[128][128] = reinterpret_cast<__nv_bfloat16 (*)[128][128]>(smem);
    __nv_bfloat16 (*smem_V)[128][128] = smem_K + 2;
    __nv_bfloat16 (*smem_Q)[64][128] = reinterpret_cast<__nv_bfloat16 (*)[64][128]>(smem_V + 2);
    float (*smem_S)[64][128] = reinterpret_cast<float (*)[64][128]>(smem_Q + 1);
    __nv_bfloat16 (*smem_P)[64][128] = reinterpret_cast<__nv_bfloat16 (*)[64][128]>(smem_S + 1);
    float (*smem_O)[64][128] = reinterpret_cast<float (*)[64][128]>(smem_P + 1);
    
    uint4 reg_O[16][2] = {0}; 
    
    float local_max[2] = {-1e20f, -1e20f};
    float local_sum[2] = {0, 0};
    
    load_smem_padded(smem_Q[0], Q_bh + query_start * 128, 64, S_len, m, tid);
    
    for (int j = 0; j < S_len; j += 128) {
        int ping = (j / 128) % 2;
        
        load_smem_padded(smem_K[ping], K_bh + j * 128, 128, S_len, min(128, S_len - j), tid);
        load_smem_padded(smem_V[ping], V_bh + j * 128, 128, S_len, min(128, S_len - j), tid);
        
        __syncthreads(); 
        
        uint4 reg_S[2][4] = {0}; 
        
        for (int k_step = 0; k_step < 128 / 16; ++k_step) { 
            const __nv_bfloat16* Q_ptr = &smem_Q[0][(tid/32)*16][k_step * 16]; 
            for (int n_step = 0; n_step < 128 / 8; ++n_step) { 
                const __nv_bfloat16* K_ptr = &smem_K[ping][n_step * 8][k_step * 16]; 
                for (int i = 0; i < 2; ++i) { 
                    wmma::mma_sync(reg_S[n_step % 4][i], Q_ptr, K_ptr, 1); 
                } 
            } 
        } 
        
        for (int n_step = 0; n_step < 128 / 8; ++n_step) {
            for (int i = 0; i < 2; ++i) {
                wmma::store_matrix_sync(&smem_S[0][(tid/32)*16][n_step * 8], reg_S[n_step % 4][i], 128, wmma::mem_row_major);
            }
        }
        __syncthreads(); 
        
        float my_max[2] = {-1e20f, -1e20f};
        for(int i = tid; i < 64 * 128 / 4; i += 128) {
            int r = i / 2;
            int c = (i % 2) * 4;
            float4 s0 = *(reinterpret_cast<float4*>(&smem_S[0][r][c]));
            if (j + c < S_len) {
                my_max[0] = max(my_max[0], max(s0.x, max(s0.y, max(s0.z, s0.w))));
            }
            float4 s1 = *(reinterpret_cast<float4*>(&smem_S[0][r+1][c]));
            if (j + c < S_len) {
                my_max[1] = max(my_max[1], max(s1.x, max(s1.y, max(s1.z, s1.w))));
            }
        }
        
        for (int offset = 1; offset < 32; offset *= 2) {
            my_max[0] = max(my_max[0], __shfl_xor_sync(0xffffffff, my_max[0], offset));
            my_max[1] = max(my_max[1], __shfl_xor_sync(0xffffffff, my_max[1], offset));
        }
        
        float new_max[2] = {max(local_max[0], my_max[0]), max(local_max[1], my_max[1])};
        float factor[2] = {expf(local_max[0] - new_max[0]), expf(local_max[1] - new_max[1])};
        
        local_sum[0] *= factor[0];
        local_sum[1] *= factor[1];
        
        for(int i = tid; i < 64 * 128 / 4; i += 128) {
            int r = i / 2;
            int c = (i % 2) * 4;
            float4 o0 = *(reinterpret_cast<float4*>(&smem_O[0][r][c]));
            *(reinterpret_cast<float4*>(&smem_O[0][r][c])) = {o0.x * factor[0], o0.y * factor[0], o0.z * factor[0], o0.w * factor[0]};
            
            float4 o1 = *(reinterpret_cast<float4*>(&smem_O[0][r+1][c]));
            *(reinterpret_cast<float4*>(&smem_O[0][r+1][c])) = {o1.x * factor[1], o1.y * factor[1], o1.z * factor[1], o1.w * factor[1]};
        }
        
        float my_sum[2] = {0, 0};
        for (int c = 0; c < 128; c += 4) {
            float4 s0_raw = *(reinterpret_cast<float4*>(&smem_S[0][tid][c]));
            float v0x = (j + c < S_len) ? expf(s0_raw.x * (1.0f / sqrtf(128.0f)) - new_max[0]) : 0;
            float v0y = (j + c + 1 < S_len) ? expf(s0_raw.y * (1.0f / sqrtf(128.0f)) - new_max[0]) : 0;
            float v0z = (j + c + 2 < S_len) ? expf(s0_raw.z * (1.0f / sqrtf(128.0f)) - new_max[0]) : 0;
            float v0w = (j + c + 3 < S_len) ? expf(s0_raw.w * (1.0f / sqrtf(128.0f)) - new_max[0]) : 0;
            
            my_sum[0] += v0x + v0y + v0z + v0w;
            
            uint2 p0 = {(uint16_t)__float2bfloat16(v0x), (uint16_t)__float2bfloat16(v0y)};
            uint2 p1 = {(uint16_t)__float2bfloat16(v0z), (uint16_t)__float2bfloat16(v0w)};
            *(reinterpret_cast<uint2*>(&smem_P[0][tid][c])) = p0;
            *(reinterpret_cast<uint2*>(&smem_P[0][tid][c+2])) = p1;
            
            float4 s1_raw = *(reinterpret_cast<float4*>(&smem_S[0][tid+32][c]));
            float v1x = (j + c < S_len) ? expf(s1_raw.x * (1.0f / sqrtf(128.0f)) - new_max[1]) : 0;
            float v1y = (j + c + 1 < S_len) ? expf(s1_raw.y * (1.0f / sqrtf(128.0f)) - new_max[1]) : 0;
            float v1z = (j + c + 2 < S_len) ? expf(s1_raw.z * (1.0f / sqrtf(128.0f)) - new_max[1]) : 0;
            float v1w = (j + c + 3 < S_len) ? expf(s1_raw.w * (1.0f / sqrtf(128.0f)) - new_max[1]) : 0;
            
            my_sum[1] += v1x + v1y + v1z + v1w;
            
            uint2 p2 = {(uint16_t)__float2bfloat16(v1x), (uint16_t)__float2bfloat16(v1y)};
            uint2 p3 = {(uint16_t)__float2bfloat16(v1z), (uint16_t)__float2bfloat16(v1w)};
            *(reinterpret_cast<uint2*>(&smem_P[0][tid+32][c])) = p2;
            *(reinterpret_cast<uint2*>(&smem_P[0][tid+32][c+2])) = p3;
        }
        
        for (int offset = 1; offset < 32; offset *= 2) {
            my_sum[0] += __shfl_xor_sync(0xffffffff, my_sum[0], offset);
            my_sum[1] += __shfl_xor_sync(0xffffffff, my_sum[1], offset);
        }
        
        local_sum[0] += my_sum[0];
        local_sum[1] += my_sum[1];
        local_max[0] = new_max[0];
        local_max[1] = new_max[1];
        
        __syncthreads(); 
        
        for (int k_step = 0; k_step < 128 / 16; ++k_step) { 
            for (int n_step = 0; n_step < 128 / 8; ++n_step) { 
                const __nv_bfloat16* P_ptr = &smem_P[0][(tid/32)*16][k_step * 16]; 
                const __nv_bfloat16* V_ptr = &smem_V[ping][k_step * 16][n_step * 8]; 
                for (int i = 0; i < 2; ++i) { 
                    wmma::mma_sync(reg_O[n_step][i], P_ptr, V_ptr, 1); 
                } 
            } 
        }
        
        __syncthreads();
    }
    
    for (int n_step = 0; n_step < 128 / 8; ++n_step) {
        for (int i = 0; i < 2; ++i) {
            wmma::store_matrix_sync(&smem_O[0][(tid/32)*16][n_step * 8], reg_O[n_step][i], 128, wmma::mem_row_major);
        }
    }
    __syncthreads();
    
    for(int i = tid; i < 64 * 128 / 4; i += 128) {
        int r = i / 2;
        int c = (i % 2) * 4;
        float inv_sum_0 = 1.0f / local_sum[r / 32];
        float4 o0 = *(reinterpret_cast<float4*>(&smem_O[0][r][c]));
        
        __nv_bfloat16* out_0 = &((__nv_bfloat16*)O)[batch_head * S_len * 128 + (query_start + r) * 128 + c];
        if (r < m) {
            *(reinterpret_cast<uint2*>(&out_0[0])) = {(uint16_t)__float2bfloat16(o0.x * inv_sum_0), (uint16_t)__float2bfloat16(o0.y * inv_sum_0)};
            *(reinterpret_cast<uint2*>(&out_0[2])) = {(uint16_t)__float2bfloat16(o0.z * inv_sum_0), (uint16_t)__float2bfloat16(o0.w * inv_sum_0)};
        }
        
        float inv_sum_1 = 1.0f / local_sum[(r + 1) / 32];
        float4 o1 = *(reinterpret_cast<float4*>(&smem_O[0][r+1][c]));
        
        __nv_bfloat16* out_1 = &((__nv_bfloat16*)O)[batch_head * S_len * 128 + (query_start + r + 1) * 128 + c];
        if (r + 1 < m) {
            *(reinterpret_cast<uint2*>(&out_1[0])) = {(uint16_t)__float2bfloat16(o1.x * inv_sum_1), (uint16_t)__float2bfloat16(o1.y * inv_sum_1)};
            *(reinterpret_cast<uint2*>(&out_1[2])) = {(uint16_t)__float2bfloat16(o1.z * inv_sum_1), (uint16_t)__float2bfloat16(o1.w * inv_sum_1)};
        }
    }
    
    if (tid < 64) {
        int r = tid;
        if (r < m) {
            LSE[batch_head * S_len + query_start + r] = local_max[r / 32] + logf(local_sum[r / 32]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    dim3 grid(B * H, (S + 63) / 64); 
    dim3 block(128); 
    
    int smem_size = 229376; 
    CUDA_CHECK(cudaFuncSetAttribute(flash_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    flash_attention_kernel<<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()), 
        static_cast<const __nv_bfloat16*>(K.data_ptr()), 
        static_cast<const __nv_bfloat16*>(V.data_ptr()), 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda