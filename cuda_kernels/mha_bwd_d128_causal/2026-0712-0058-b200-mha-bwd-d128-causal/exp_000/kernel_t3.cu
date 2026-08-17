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

namespace tvm_ffi_kernel {

__device__ __forceinline__ void load_tile_64x128(const __nv_bfloat16* gmem, __nv_bfloat16* smem) {
    int tid = threadIdx.x;
    // 64 rows * 128 cols * 2 bytes = 16384 bytes. 128 threads * 16 bytes = 2048 bytes per iteration -> 8 iterations
    for (int i = 0; i < 8; i++) {
        int byte_offset = i * 2048 + tid * 16;
        float4 val = *(const float4*)(gmem + byte_offset);
        *(float4*)(smem + byte_offset) = val;
    }
}

__device__ __forceinline__ void load_tile_128x128(const __nv_bfloat16* gmem, __nv_bfloat16* smem) {
    int tid = threadIdx.x;
    // 128 rows * 128 cols * 2 bytes = 32768 bytes. 16 iterations.
    for (int i = 0; i < 16; i++) {
        int byte_offset = i * 2048 + tid * 16;
        float4 val = *(const float4*)(gmem + byte_offset);
        *(float4*)(smem + byte_offset) = val;
    }
}

__device__ __forceinline__ void store_tile_64x128_fp32(const float* smem_acc, __nv_bfloat16* gmem) {
    int tid = threadIdx.x;
    for (int i = 0; i < 8; i++) {
        int elem_offset = i * 1024 + tid * 8;
        float4 in_f4 = *(const float4*)&smem_acc[elem_offset];
        __nv_bfloat16 out_bf[8];
        for(int k=0; k<8; k++) {
            out_bf[k] = __float2bfloat16(in_f4.x + k * 0.0f); // Placeholder logic, fixed below
        }
        // Fixed packing logic inline
        float* in_ptr = (float*)&in_f4;
        __nv_bfloat16* out_ptr = (__nv_bfloat16*)&gmem[elem_offset];
        for(int k=0; k<8; k++) {
            out_ptr[k] = __float2bfloat16(in_ptr[k]);
        }
    }
}

__device__ __forceinline__ void store_tile_128x128_bf16(const __nv_bfloat16* smem_acc, __nv_bfloat16* gmem) {
    int tid = threadIdx.x;
    for (int i = 0; i < 16; i++) {
        int byte_offset = i * 2048 + tid * 16;
        float4 val = *(const float4*)(smem_acc + byte_offset);
        *(float4*)(gmem + byte_offset) = val;
    }
}

// GEMM D[64, 128] = A[64, 128] x B[128, 128]^T
__device__ __forceinline__ void gemm_64x128x128_AT(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* D, int tid) {
    if (tid < 64) {
        float acc[128];
        for(int n=0; n<128; n++) acc[n] = 0.0f;
        
        for (int k = 0; k < 128; k++) {
            float a = __bfloat162float(A[tid * 128 + k]);
            for (int n = 0; n < 128; n++) {
                acc[n] += a * __bfloat162float(B[n * 128 + k]);
            }
        }
        for(int n=0; n<128; n++) {
            D[tid * 128 + n] = __float2bfloat16(acc[n]);
        }
    }
}

// GEMM D[64, 128] = A[64, 128] x B[128, 128]
__device__ __forceinline__ void gemm_64x128x128_AB(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* D, int tid) {
    if (tid < 64) {
        float acc[128];
        for(int n=0; n<128; n++) acc[n] = 0.0f;
        
        for (int k = 0; k < 128; k++) {
            float a = __bfloat162float(A[tid * 128 + k]);
            for (int n = 0; n < 128; n++) {
                acc[n] += a * __bfloat162float(B[k * 128 + n]);
            }
        }
        for(int n=0; n<128; n++) {
            D[tid * 128 + n] = __float2bfloat16(acc[n]);
        }
    }
}

// GEMM D[128, 128] = A[64, 128]^T x B[64, 128]
__device__ __forceinline__ void gemm_64x128x128_AT_B(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* D, int tid) {
    float acc[128];
    for(int n=0; n<128; n++) acc[n] = 0.0f;
    
    for (int k = 0; k < 64; k++) {
        float a = __bfloat162float(A[k * 128 + tid]);
        for (int n = 0; n < 128; n++) {
            acc[n] += a * __bfloat162float(B[k * 128 + n]);
        }
    }
    for(int n=0; n<128; n++) {
        D[tid * 128 + n] = __float2bfloat16(acc[n]);
    }
}


__global__ void __launch_bounds__(128) bwd_dq_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, 
    const __nv_bfloat16* V, const __nv_bfloat16* dO, 
    const __nv_bfloat16* O, const float* L, 
    __nv_bfloat16* dQ, int S) 
{
    int bh = blockIdx.y;
    int num_q_blocks = S / 128;
    int q_blk = blockIdx.x / 2;
    int m_offset = (blockIdx.x % 2) * 64;

    if (q_blk >= num_q_blocks) return;

    extern __shared__ __align__(128) uint8_t smem[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem;                     // 0
    __nv_bfloat16* smem_dO = (__nv_bfloat16*)(smem + 16384);          // 16KB
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem + 32768);           // 32KB
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem + 65536);           // 64KB
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem + 98304);           // 96KB
    __nv_bfloat16* smem_dS = (__nv_bfloat16*)(smem + 114688);         // 112KB
    __nv_bfloat16* smem_S = (__nv_bfloat16*)(smem + 131072);          // 128KB
    __nv_bfloat16* smem_dP = smem_S;                                  // 128KB
    float* smem_dQ_acc = (float*)(smem + 147456);                     // 144KB
    float* smem_D = (float*)(smem + 179200);                          // 176KB
    float* smem_sum_L = (float*)(smem + 179456);                      // 176.25KB

    const __nv_bfloat16* Q_bh = Q + bh * S * 128;
    const __nv_bfloat16* K_bh = K + bh * S * 128;
    const __nv_bfloat16* V_bh = V + bh * S * 128;
    const __nv_bfloat16* dO_bh =