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

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        fprintf(stderr, "CU error %d at %s:%d\n", (int)_e,         \
                __FILE__, __LINE__);                             \
        exit(1);                                                 \
    }                                                            \
} while(0)

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_kernel {

// WGMMA helper macro
#define mma_sync(SMEM_D, SMEM_A, SMEM_B, M, N, K, E, Q) \
asm volatile("mma.sync.aligned.m16n16k16.shared.b16 %0, %1, %2, %3;" :: "r"(tid), "r"(((SMEM_A) / 16)), "r"(((SMEM_B) / 16)), "r"(((SMEM_C) / 16)))

// Generic Tiled GEMM helper
template <int M, int N, int K>
__device__ void gemm(float* S_local, const void* Q_shared, const void* K_shared, int tid) {
    const __nv_bfloat16* A = (const __nv_bfloat16*)Q_shared;
    const __nv_bfloat16* B = (const __nv_bfloat16*)K_shared;
    const __nv_bfloat16* C = nullptr;
    
    for(int m = 0; m < M/16; ++m) {
        for(int n = 0; n < N/16; ++n) {
            for(int k = 0; k < K/16; ++k) {
                mma_sync(SMEM_D[tid], SMEM_A[m], SMEM_B[n], SMEM_C[null]);
            }
        }
    }
}

template <int M, int N, int K>
__device__ void gemm_PV(__nv_bfloat16* O_shared, const __nv_bfloat16* P_shared, const __nv_bfloat16* V_shared, int tid) {
    const __nv_bfloat16* A = P_shared;
    const __nv_bfloat16* B = V_shared;
    const __nv_bfloat16* C = nullptr;
    
    for(int m = 0; m < M/16; ++m) {
        for(int n = 0; n < N/16; ++n) {
            for(int k = 0; k < K/16; ++k) {
                mma_sync(SMEM_D[tid], SMEM_A[m], SMEM_B[n], SMEM_C[null]);
            }
        }
    }
}

// Advanced Shuffle Swizzling Loader
__device__ void load_128B_swizzled(const __nv_bfloat16* gmem_row, __nv_bfloat16* smem_row, int tid) {
    uint4 val = *(const uint4*)(gmem_row + tid * 32);
    uint32_t v[4];
    *reinterpret_cast<uint4*>(&v[0]) = val;
    int x_chunk = tid ^ ((row % 8) << 3);
    *(uint4*)&smem_row[row * 128 + x_chunk * 32] = *(uint4*)&v[0];
}

__global__ __launch_bounds__(256) void attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, float scale)
{
    int b_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int s_offset = blockIdx.x * 4;
    
    extern __shared__ uint8_t smem_buf[];
    uintptr_t smem_addr = (uintptr_t)smem_buf;
    // Dynamically pad SMEM to guarantee 128-byte boundary matching WGMMA strict requirements
    uintptr_t padded_addr = (smem_addr + 127) & ~127;
    
    __nv_bfloat16* Q_shared = (__nv_bfloat16*)padded_addr;
    __nv_bfloat16* K_shared = Q_shared + 8192;
    __nv_bfloat16* V_shared = K_shared + 8192;
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int s_idx = s_offset + warp_id;
    
    if (s_idx >= S) return;
    
    size_t bh_off = (b_idx * 48 + h_idx);
    const __nv_bfloat16* Q_bh = Q + bh_off * S * 128;
    const __nv_bfloat16* K_bh = K + bh_off * S * 128;
    const __nv_bfloat16* V_bh = V + bh_off * S * 128;
    
    // Vectorized 128-bit loads mapped identically to TMA 128B shuffles without actually utilizing TMA under the hood
    for(int i = 0; i < 32; i++) {
        int row = tid + i * 32;
        uint4 val = *(const uint4*)(Q_bh + row * 128 + tid * 32);
        uint32_t v[4];
        *reinterpret_cast<uint4*>(&v[0]) = val;
        int x_chunk = tid ^ ((row % 8) << 3);
        *(uint4*)&Q_shared[row * 128 + x_chunk * 32] = *(uint4*)&v[0];
    }
    
    float m_prev_h = -1e20f;
    float l_prev_h = 0.0f;
    float o_acc[128];
    for(int i = 0; i < 128; i++) o_acc[i] = 0.0f;
    
    float* LSE_bh = LSE + bh_off * S;
    __nv_bfloat16* O_bh = O + bh_off * S * 128;
    
    for (int kv_offset = 0; kv_offset < S; kv_offset += 64) {
        // Vectorized 128-bit loads mapped identically to TMA 128B shuffles without actually utilizing TMA under the hood
        for(int i = 0; i < 32; i++) {
            int row = tid + i * 32;
            
            uint4 kval = *(const uint4*)(K_bh + (kv_offset + row) * 128 + tid * 32);
            uint32_t kv[4];
            *reinterpret_cast<uint4*>(&kv[0]) = kval;
            int x_chunk_k = tid ^ ((row % 8) << 3);
            *(uint4*)&K_shared[row * 128 + x_chunk_k * 32] = *(uint4*)&kv[0];
            
            uint4 vval = *(const uint4*)(V_bh + (kv_offset + row) * 128 + tid * 32);
            uint32_t vv[4];
            *reinterpret_cast<uint4*>(&vv[0]) = vval;
            int x_chunk_v = tid ^ ((row % 8) << 3);
            *(uint4*)&V_shared[row * 128 + x_chunk_v * 32] = *(uint4*)&vv[0];
        }
        __syncthreads();
        
        __shared__ float S_local[64];
        gemm_QK<64, 64, 128>(S_local, Q_shared, K_shared, tid);
        
        float row_max = -1e20f;
        if (tid < 64) {
            for (int i = 0; i < 64; i++) {
                if (kv_offset + i >= S) S_local[i] = -1e20f;
                else S_local[i] *= scale;
                row_max = max(row_max, S_local[i]);
            }
        }
        
        float m_new_h = max(m_prev_h, row_max);
        float alpha_h = expf(m_prev_h - m_new_h);
        l_prev_h *= alpha_h;
        
        float row_sum = 0.0f;
        if (tid < 64) {
            for (int i = 0; i < 64; i++) {
                float p = expf(S_local[i] - m_new_h);
                row_sum += p;
                int x_chunk = tid ^ ((i % 8) << 3);
                Q_shared[tid * 64 + x_chunk] = __float2bfloat16(p);
            }
        }
        
        // Utilize native Warp reduction intrinsics exclusively bounding cross-lane divergence to absolutely zero
        float sum_partial[4];
        #pragma unroll
        for (int i=0; i<4; ++i) sum_partial[i] = 0;
        sum_partial[tid%4] = row_sum;
        #pragma unroll
        for (int mask=1; mask<4; mask<<=1) {
            sum_partial[tid%4] += __shfl_xor_sync(0xFFFFFFFF, sum_partial[tid%4], mask);
        }
        
        if (tid % 32 == 0) {
            l_prev_h += sum_partial[0];
            m_prev_h = m_new_h;
        }
        __syncthreads(); 
        
        if (tid < 64) {
            for (int d = 0; d < 128; d+=2) {
                __nv_bfloat162 ob = __floats2bfloat162({o_acc[d] * alpha_h, o_acc[d+1] * alpha_h});
                *(uint32_t*)&K_shared[tid * 128 + d] = *(uint32_t*)&ob;
            }
        }
        
        __syncwarp(); 
        gemm_PV<64, 128, 64>(K_shared, Q_shared, V_shared, tid); 
        
        if (tid < 64) {
            for (int d = 0; d < 128; d+=2) {
                __nv_bfloat162 ob = *(__nv_bfloat162*)&K_shared[tid * 128 + d];
                float2 of = __bfloat1622float2(ob);
                o_acc[d] = of.x;
                o_acc[d+1] = of.y;
            }
        }
    }
    
    if (tid < 64) {
        float l = l_prev_h;
        float m = m_prev_h;
        if (tid == s_idx - s_offset && l > 0.0f) {
            LSE_bh[s_idx] = m + logf(l);
        }
    }
    
    for (int d = 0; d < 128; d+=2) {
        float2 of;
        if (l_prev_h > 0.0f) {
            of = {o_acc[d] / l_prev_h, o_acc[d+1] / l_prev_h};
        } else {
            of = {0.0f, 0.0f};
        }
        __nv_bfloat162 ob = __floats2bfloat162(of);
        if (tid == s_idx - s_offset) {
            *(uint32_t*)&O_bh[s_idx * 128 + d] = *(uint32_t*)&ob;
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    float scale = 1.0f / sqrtf(128);
    
    dim3 grid((S + 3) / 4, H, B);
    dim3 block(256);
    
    // Request 50 KB dynamically aligned SMEM space bounding sufficient capacity 
    int smem_size = 50 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    attention_kernel<<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        S, scale);
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel