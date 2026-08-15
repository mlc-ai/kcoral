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

__device__ void load_smem_padded(__nv_bfloat16 (*dst)[128], const __nv_bfloat16* src, int rows, int max_rows, int valid_rows, int tid) {
    for (int i = tid; i < rows * 128 / 2; i += 32) {
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

__device__ void swizzle_64x128(__nv_bfloat16 (*dst)[128], const __nv_bfloat16 (*src)[128], int tid) {
    for(int i = tid; i < 64 * 128; i += 32) {
        int r = i / 128;
        int c = i % 128;
        int swizzled_c = ((r % 8) ^ (c / 8)) * 8 + (c % 8);
        dst[r][swizzled_c] = src[r][c];
    }
}

__device__ void matmul_transpose_B(__nv_bfloat16 (*A)[128], __nv_bfloat16 (*B)[128], float (*C)[128], int M, int N, int K_dim) {
    int row_0 = tid;
    int row_1 = tid + 32;
    
    for (int k = 0; k < K_dim; ++k) {
        float a0 = (row_0 < M) ? __bfloat162float(A[row_0][(k % 8) ^ ((k / 8) % 8)]) : 0; // Swizzle decode inline matching swizzle_64x128
        float a1 = (row_1 < M) ? __bfloat162float(A[row_1][(k % 8) ^ ((k / 8) % 8)]) : 0;
        
        float4 b_vec = {0, 0, 0, 0};
        if (k < K_dim) {
            b_vec = *(reinterpret_cast<float4*>(&B[k]));
        }
        
        for(int n = 0; n < N; n += 4) {
             float4 b = b_vec; // b holds K^T row
             C[row_0][n] += a0 * b.x;
             C[row_0][n+1] += a0 * b.y;
             C[row_0][n+2] += a0 * b.z;
             C[row_0][n+3] += a0 * b.w;
             
             C[row_1][n] += a1 * b.x;
             C[row_1][n+1] += a1 * b.y;
             C[row_1][n+2] += a1 * b.z;
             C[row_1][n+3] += a1 * b.w;
        }
    }
}

__device__ void matmul_no_transpose(float (*A)[128], __nv_bfloat16 (*B)[128], float (*C)[128], int M, int N, int K_dim) {
    int row_0 = tid;
    int row_1 = tid + 32;
    
    for (int k = 0; k < K_dim; ++k) {
        float a0 = (row_0 < M) ? A[row_0][k] : 0;
        float a1 = (row_1 < M) ? A[row_1][k] : 0;
        
        for(int n = 0; n < N; n += 4) {
            float4 b = *(reinterpret_cast<float4*>(&B[k][n]));
            C[row_0][n] += a0 * b.x;
            C[row_0][n+1] += a0 * b.y;
            C[row_0][n+2] += a0 * b.z;
            C[row_0][n+3] += a0 * b.w;
            
            C[row_1][n] += a1 * b.x;
            C[row_1][n+1] += a1 * b.y;
            C[row_1][n+2] += a1 * b.z;
            C[row_1][n+3] += a1 * b.w;
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
    
    extern __shared__ char smem[];
    char* aligned_smem = smem + 256; 
    
    __nv_bfloat16 (*temp_Q)[128] = (__nv_bfloat16 (*)[128])aligned_smem;               // offset 0, 16KB
    __nv_bfloat16 (*swizzled_Q)[128] = (__nv_bfloat16 (*)[128])(aligned_smem + 16384);  // offset 16KB, 16KB
    __nv_bfloat16 (*smem_K)[128] = (__nv_bfloat16 (*)[128])(aligned_smem + 32768);      // offset 32KB, 32KB
    __nv_bfloat16 (*smem_V)[128] = (__nv_bfloat16 (*)[128])(aligned_smem + 65536);      // offset 64KB, 32KB
    float (*smem_S)[128] = (float (*)[128])(aligned_smem + 98304);                      // offset 96KB, 32KB
    float (*smem_S_transposed_fp32)[128] = (float (*)[128])(aligned_smem + 131072);     // offset 128KB, 32KB
    float (*smem_temp)[128] = (float (*)[128])(aligned_smem + 163840);                  // offset 160KB, 32KB
    
    for (int i = tid; i < 64 * 128; i += 32) {
        smem_temp[i / 128][i % 128] = 0; 
    }
    __syncthreads();
    
    float local_max[2] = {-1e20f, -1e20f};
    float local_sum[2] = {0, 0};
    
    for (int j = 0; j < S_len; j += 64) {
        load_smem_padded(temp_Q, Q_bh + query_start * 128, 64, S_len, m, tid);
        __syncthreads();
        swizzle_64x128(swizzled_Q, temp_Q, tid);
        __syncthreads();
        
        load_smem_padded(smem_K, K_bh + j * 128, 128, S_len, min(128, S_len - j), tid);
        __syncthreads();
        
        *(reinterpret_cast<float4*>(&smem_S[tid * 2]) ) = {0,0,0,0};
        *(reinterpret_cast<float4*>(&smem_S[tid * 2 + 1]) ) = {0,0,0,0};
        
        matmul_transpose_B(swizzled_Q, smem_K, smem_S, 64, 128, 128); 
        
        for(int i = tid; i < 64 * 128 / 4; i += 32) {
            int r = i / 2;
            int c = (i % 2) * 4;
            float4 s_val = *(reinterpret_cast<float4*>(&smem_S[r][c]));
            float scale = 1.0f / sqrtf(128.0f);
            *(reinterpret_cast<float4*>(&smem_S[r][c])) = {s_val.x * scale, s_val.y * scale, s_val.z * scale, s_val.w * scale};
        }
        
        float max_val[2] = {-1e20f, -1e20f};
        for (int c = 0; c < 128; ++c) {
            if (j + c < S_len) {
                max_val[0] = max(max_val[0], smem_S[tid][c]);
                max_val[1] = max(max_val[1], smem_S[tid + 32][c]);
            }
        }
        
        float new_max[2] = {max(local_max[0], max_val[0]), max(local_max[1], max_val[1])};
        float factor[2] = {expf(local_max[0] - new_max[0]), expf(local_max[1] - new_max[1])};
        
        local_sum[0] *= factor[0];
        local_sum[1] *= factor[1];
        
        for (int c = 0; c < 128; c += 4) {
            float4 o_val_0 = *(reinterpret_cast<float4*>(&smem_temp[tid][c]));
            *(reinterpret_cast<float4*>(&smem_temp[tid][c])) = {o_val_0.x * factor[0], o_val_0.y * factor[0], o_val_0.z * factor[0], o_val_0.w * factor[0]};
            
            float4 o_val_1 = *(reinterpret_cast<float4*>(&smem_temp[tid + 32][c]));
            *(reinterpret_cast<float4*>(&smem_temp[tid + 32][c])) = {o_val_1.x * factor[1], o_val_1.y * factor[1], o_val_1.z * factor[1], o_val_1.w * factor[1]};
        }
        
        float sum_val[2] = {0, 0};
        for (int c = 0; c < 128; ++c) {
            float v0 = (j + c < S_len) ? expf(smem_S[tid][c] - new_max[0]) : 0;
            float v1 = (j + c < S_len) ? expf(smem_S[tid + 32][c] - new_max[1]) : 0;
            smem_S[tid][c] = v0;
            smem_S[tid + 32][c] = v1;
            sum_val[0] += v0;
            sum_val[1] += v1;
        }
        local_sum[0] += sum_val[0];
        local_sum[1] += sum_val[1];
        local_max[0] = new_max[0];
        local_max[1] = new_max[1];
        
        load_smem_padded(smem_V, V_bh + j * 128, 128, S_len, min(128, S_len - j), tid);
        __syncthreads();
        
        for(int i = tid; i < 128 * 128 / 2; i += 32) {
            int r = i / 2;
            int c = (i % 2) * 2;
            uint2 val = *(reinterpret_cast<uint2*>(&smem_V[r][c]));
            *(reinterpret_cast<uint2*>(&smem_V[c][r])) = val; 
        }
        __syncthreads();
        
        matmul_transpose_B((__nv_bfloat16 (*)[128])smem_S, smem_V, smem_temp, 64, 128, 128);
        __syncthreads();
    }
    
    for(int i = tid; i < 64 * 128 / 2; i += 32) {
        int r = i / 2;
        int c = (i % 2) * 2;
        float out_0 = smem_temp[r][c] / local_sum[r / 32];
        float out_1 = smem_temp[r][c+1] / local_sum[r / 32];
        *(reinterpret_cast<uint2*>(&((__nv_bfloat16*)O)[batch_head * S_len * 128 + (query_start + r) * 128 + c])) = {__float2bfloat16(out_0), __float2bfloat16(out_1)};
    }
    
    if (tid < m) {
        LSE[batch_head * S_len + query_start + tid] = local_max[tid / 32] + logf(local_sum[tid / 32]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    dim3 grid(B * H, (S + 63) / 64); 
    dim3 block(32); 
    
    int smem_size = 192512 + 256; 
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