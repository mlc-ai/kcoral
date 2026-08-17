#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <mma.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_attention_bwd {

// ---------------------- Hardware Helper Functions ----------------------

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void gemm_128x128(
    wmma::matrixFragment<wmma::mma_sync, 16, 128, 16, float> &D,
    const __nv_bfloat16* A_SMem, int row_offset_A, int col_offset_A,
    const __nv_bfloat16* B_SMem, int row_offset_B, int col_offset_B,
    bool A_Is_Cols, bool B_Is_Cols) 
{
    wmma::fillFragment(D, 0.0f);
    
    wmma::matrixFragment<wmma::mma_sync, 16, 16, 16, __nv_bfloat16, A_Is_Cols ? wmma::memColMajor : wmma::memRowMajor> A;
    wmma::matrixFragment<wmma::mma_sync, 16, 128, 16, __nv_bfloat16, B_Is_Cols ? wmma::memColMajor : wmma::memRowMajor> B;
    
    for (int k_step = 0; k_step < 8; ++k_step) {
        wmma::loadRegisterSync(A, A_SMem + (row_offset_A << 7) + (col_offset_A + k_step << 4), 128);
        wmma::loadRegisterSync(B, B_SMem + (row_offset_B << 7) + (col_offset_B + k_step << 4), 128);
        wmma::mma_sync(D, A, B, D);
    }
}

__device__ __forceinline__ void storeMatrix(
    const wmma::matrixFragment<wmma::mma_sync, 16, 128, 16, float>& D,
    __nv_bfloat16* D_SMem, int row_offset, int col_offset) 
{
    wmma::storeRegisterSync(D_SMem + (row_offset << 7) + col_offset, D, 128, wmma::memRowMajor);
}

extern __shared__ __align__(1024) uint8_t smem_pool[];

__global__ void __launch_bounds__(128, 1) bwd_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* L,
    float* dK_fp32, float* dV_fp32, __nv_bfloat16* dQ, int S_len, float scale)
{
    int q_tile = blockIdx.x;
    int head_idx = blockIdx.y;
    int num_tiles = (S_len + 127) / 128;
    if (q_tile >= num_tiles) return;
    
    int q_base = q_tile * 128;
    uint64_t head_offset = head_idx * S_len * 128;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int my_m = warp_id * 16;
    
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem_pool;
    __nv_bfloat16* smem_K = smem_Q + 128*128;
    __nv_bfloat16* smem_V = smem_K + 128*128;
    __nv_bfloat16* smem_O = smem_V + 128*128;
    __nv_bfloat16* smem_dO = smem_O + 128*128;
    __nv_bfloat16* smem_P = smem_dO + 128*128;
    __nv_bfloat16* smem_dS = smem_P + 128*128;
    float* smem_D = (float*)(smem_dS + 128*128);
    float* smem_LSE = smem_D + 128;
    
    uint64_t q_h_offset = head_offset + q_base * 128;
    
    for (int idx = tid; idx < 128*128/8; idx += 128) {
        *reinterpret_cast<float4*>(&smem_Q[idx*8]) = *reinterpret_cast<const float4*>(&Q[q_h_offset + idx*8]);
        *reinterpret_cast<float4*>(&smem_O[idx*8]) = *reinterpret_cast<const float4*>(&O[q_h_offset + idx*8]);
        *reinterpret_cast<float4*>(&smem_dO[idx*8]) = *reinterpret_cast<const float4*>(&dO[q_h_offset + idx*8]);
    }
    __syncthreads();
    
    if (tid < 128) {
        float sum = 0;
        if (q_base + tid < S_len) {
            for (int d = 0; d < 128; d++) {
                sum += __bfloat162float(smem_dO[tid*128+d]) * __bfloat162float(smem_O[tid*128+d]);
            }
        }
        smem_D[tid] = sum;
        smem_LSE[tid] = (q_base + tid < S_len) ? L[head_idx * S_len + q_base + tid] : 0;
    }
    __syncthreads();
    
    wmma::matrixFragment<wmma::mma_sync, 16, 128, 16, float> dQ_frag[2];
    wmma::fillFragment(dQ_frag[0], 0.0f);
    wmma::fillFragment(dQ_frag[1], 0.0f);
    
    for (int k_tile = 0; k_tile <= q_tile && k_tile < num_tiles; k_tile++) {
        int k_base = k_tile * 128;
        uint64_t k_h_offset = head_offset + k_base * 128;
        
        for (int idx = tid; idx < 128*128/8; idx += 128) {
            *reinterpret_cast<float4*>(&smem_K[idx*8]) = *reinterpret_cast<const float4*>(&K[k_h_offset + idx*8]);
            *reinterpret_cast<float4*>(&smem_V[idx*8]) = *reinterpret_cast<const float4*>(&V[k_h_offset + idx*8]);
        }
        __syncthreads();
        
        wmma::matrixFragment<wmma::mma_sync, 16, 128, 16, float> S_frag[2];
        gemm_128x128(S_frag[0], smem_Q, my_m, 0, smem_K, 0, 0, true);
        gemm_128x128(S_frag[1], smem_Q, 64 + my_m, 0, smem_K, 0, 0, true);
        
        storeMatrix(S_frag[0], smem_P, my_m, 0);
        storeMatrix(S_frag[1], smem_P, 64 + my_m, 0);
        __syncthreads();
        
        for (int i = tid; i < 128; i++) {
            for (int j = 0; j < 128; j++) {
                float s_val = __bfloat162float(smem_P[i*128+j]) * scale;
                float p_val = fast_exp2f_fn((s_val - smem_LSE[i]) * 1.4426950408889634f);
                if (q_base + i < k_base + j || q_base + i >= S_len || k_base + j >= S_len) {
                    p_val = 0;
                }
                smem_P[i*128+j] = __float2bfloat16(p_val);
            }
        }
        __syncthreads();
        
        wmma::matrixFragment<wmma::mma_sync, 16, 128, 16, float> dP_frag[2];
        gemm_128x128(dP_frag[0], smem_dO, my_m, 0, smem_V, 0, 0, true);
        gemm_128x128(dP_frag[1], smem_dO, 64 + my_m, 0, smem_V, 0, 0, true);
        
        storeMatrix(dP_frag[0], smem_dS, my_m, 0);
        storeMatrix(dP_frag[1], smem_dS, 64 + my_m, 0);
        __syncthreads();
        
        for (int i = tid; i < 128; i++) {
            for (int j = 0; j < 128; j++) {
                float p_val = __bfloat162float(smem_P[i*128+j]);
                float dp_val = __bfloat162float(smem_dS[i*128+j]);
                float ds_val = p_val * (dp_val - smem_D[i]);
                if (q_base + i < k_base + j || q_base + i >= S_len || k_base + j >= S_len) {
                    ds_val = 0;
                }
                smem_dS[i*128+j] = __float2bfloat16(ds_val);
            }
        }
        __syncthreads();
        
        wmma::matrixFragment<wmma::mma_sync, 16, 128, 16, float> dV_frag[2];
        gemm_128x128(dV_frag[0], smem_P, 0, my_m, smem_dO, 0, 0, true, false);
        gemm_128x128(dV_frag[1], smem_P, 0, 64 + my_m, smem_dO, 0, 0, true, false);
        
        storeMatrix(dV_frag[0], smem_K, my_m, 0);
        storeMatrix(dV_frag[1], smem_K, 64 + my_m, 0);
        __syncthreads();
        
        for (int i = tid; i < 128; i++) {
            if (k_base + i < S_len) {
                for (int d = 0; d < 128; d++) {
                    atomicAdd(&dV_fp32[head_offset + (k_base + i) * 128 + d], __bfloat162float(smem_K[i*128+d]));
                }
            }
        }
        __syncthreads();
        
        wmma::matrixFragment<wmma::mma_sync, 16, 128, 16, float> dK_frag[2];
        gemm_128x128(dK_frag[0], smem_dS, 0, my_m, smem_Q, 0, 0, true, false);
        gemm_128x128(dK_frag[1], smem_dS, 0, 64 + my_m, smem_Q, 0, 0, true, false);
        
        storeMatrix(dK_frag[0], smem_V, my_m, 0);
        storeMatrix(dK_frag[1], smem_V, 64 + my_m, 0);
        __syncthreads();
        
        for (int i = tid; i < 128; i++) {
            if (k_base + i < S_len) {
                for (int d = 0; d < 128; d++) {
                    atomicAdd(&dK_fp32[head_offset + (k_base + i) * 128 + d], __bfloat162float(smem_V[i*128+d]));
                }
            }
        }
        __syncthreads();
        
        wmma::matrixFragment<wmma::mma_sync, 16, 128, 16, float> dQ_step[2];
        gemm_128x128(dQ_step[0], smem_dS, my_m, 0, smem_K, 0, 0, false, true);
        gemm_128x128(dQ_step[1], smem_dS, 64 + my_m, 0, smem_K, 0, 0, false, true);
        
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < 128; c += 2) {
                dQ_frag[0](r, c) += dQ_step[0](r, c);
                dQ_frag[0](r, c+1) += dQ_step[0](r, c+1);
                dQ_frag[1](r, c) += dQ_step[1](r, c);
                dQ_frag[1](r, c+1) += dQ_step[1](r, c+1);
            }
        }
    }
    
    storeMatrix(dQ_frag[0], smem_O, my_m, 0);
    storeMatrix(dQ_frag[1], smem_O, 64 + my_m, 0);
    __syncthreads();
    
    for (int idx = tid; idx < 128*128/8; idx += 128) {
        int row = (idx * 8) / 128;
        if (q_base + row < S_len) {
            int col = (idx * 8) % 128;
            *reinterpret_cast<float4*>(&dQ[head_offset + (q_base + row) * 128 + col]) = *reinterpret_cast<float4*>(&smem_O[row * 128 + col]);
        }
    }
}

__global__ void fp32_to_bf16_kernel(const float* in, __nv_bfloat16* out, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(in[idx]);
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
    
    float scale = 1.0f / sqrtf((float)d);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    size_t num_elements = B * H * S * d;
    float* dV_fp32 = nullptr;
    float* dK_fp32 = nullptr;
    
    CUDA_CHECK(cudaMallocAsync(&dV_fp32, num_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&dK_fp32, num_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_fp32, 0, num_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dK_fp32, 0, num_elements * sizeof(float), stream));
    
    CUDA_CHECK(cudaFuncSetAttribute(bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 230400));
    
    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    
    bwd_kernel<<<grid, block, 230400, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        dV_fp32, dK_fp32,
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        S, scale);
    
    CUDA_CHECK(cudaGetLastError());
    
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    fp32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(dV_fp32, static_cast<__nv_bfloat16*>(dV.data_ptr()), num_elements);
    fp32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(dK_fp32, static_cast<__nv_bfloat16*>(dK.data_ptr()), num_elements);
    
    CUDA_CHECK(cudaFreeAsync(dV_fp32, stream));
    CUDA_CHECK(cudaFreeAsync(dK_fp32, stream));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_attention_bwd::run);

}  // namespace tvm_ffi_attention_bwd