#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

// Helper to pack two fp32 values as a single uint32_t containing two bf16s
__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

#define SMEM_STRIDE_VEC 17  // 17 float4s = 136 bf16 elements, avoids bank conflicts

__global__ void mha_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S) 
{
    int batch_head_idx = blockIdx.y;
    int q_start = blockIdx.x * 64;
    int tid = threadIdx.x;
    int q_idx = q_start + tid;

    int64_t batch_head_offset = (int64_t)batch_head_idx * S * 128;

    __shared__ alignas(16) __nv_bfloat16 smem_K[64 * 136];
    __shared__ alignas(16) __nv_bfloat16 smem_V[64 * 136];

    // Reusing K's shared memory for Q and O to save total SMEM footprint
    __nv_bfloat16* smem_Q = smem_K;
    __nv_bfloat16* smem_O = smem_K;

    // Phase 1: Load Q into SMEM collectively
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int idx = i * 64 + tid;
        int row = idx / 16;
        int col_vec = idx % 16;
        int q_idx_mem = q_start + row;
        if (q_idx_mem < S) {
            float4 val = ((const float4*)&Q[batch_head_offset + q_idx_mem * 128])[col_vec];
            ((float4*)smem_Q)[row * SMEM_STRIDE_VEC + col_vec] = val;
        }
    }
    __syncthreads();

    // Read Q into thread-local registers
    __nv_bfloat16 Q_reg[128];
    if (q_idx < S) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            ((float4*)Q_reg)[i] = ((float4*)smem_Q)[tid * SMEM_STRIDE_VEC + i];
        }
    }
    __syncthreads(); // Safe to overwrite smem_K now

    float O_reg[128];
    #pragma unroll
    for(int i=0; i<128; ++i) O_reg[i] = 0.0f;
    
    float m = -1e20f;
    float l = 0.0f;
    const float scale = 0.08838834764831843f; // 1 / sqrt(128)

    // Phase 2: Iterate over Key/Value blocks
    for (int k_start = 0; k_start <= q_start; k_start += 64) {
        // Load K and V collectively
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int idx = i * 64 + tid;
            int row = idx / 16;
            int col_vec = idx % 16;
            int k_idx_mem = k_start + row;
            if (k_idx_mem < S) {
                ((float4*)smem_K)[row * SMEM_STRIDE_VEC + col_vec] = ((const float4*)&K[batch_head_offset + k_idx_mem * 128])[col_vec];
                ((float4*)smem_V)[row * SMEM_STRIDE_VEC + col_vec] = ((const float4*)&V[batch_head_offset + k_idx_mem * 128])[col_vec];
            }
        }
        __syncthreads();

        if (q_idx < S) {
            int k_end = k_start + 64;
            if (k_end > S) k_end = S;
            
            for (int k = 0; k < k_end - k_start; ++k) {
                int k_idx_mem = k_start + k;
                if (k_idx_mem > q_idx) break; // Causal mask
                
                float S_val = 0.0f;
                // Reading K from SMEM. Since all threads access the same row at a time, it acts as a conflict-free broadcast.
                float4* K_row = (float4*)&smem_K[k * 136];
                
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    float4 k_val = K_row[i];
                    __nv_bfloat16* q_ptr = (__nv_bfloat16*)&Q_reg[i * 8];
                    __nv_bfloat16* k_ptr = (__nv_bfloat16*)&k_val;
                    
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        S_val += __bfloat162float(q_ptr[j]) * __bfloat162float(k_ptr[j]);
                    }
                }
                
                S_val *= scale;
                
                float m_new = fmaxf(m, S_val);
                float exp_diff = expf(m - m_new);
                float exp_val = expf(S_val - m_new);
                
                l = l * exp_diff + exp_val;
                m = m_new;
                
                float4* V_row = (float4*)&smem_V[k * 136];
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    float4 v_val = V_row[i];
                    __nv_bfloat16* v_ptr = (__nv_bfloat16*)&v_val;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        O_reg[i * 8 + j] = O_reg[i * 8 + j] * exp_diff + exp_val * __bfloat162float(v_ptr[j]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // Phase 3: Finalize O and write to GMEM
    if (q_idx < S) {
        float l_inv = 1.0f / l;
        #pragma unroll
        for (int i = 0; i < 128; ++i) {
            O_reg[i] *= l_inv;
        }
        
        int64_t lse_idx = (int64_t)batch_head_idx * S + q_idx;
        LSE[lse_idx] = m + logf(l);
        
        // Pack back to bf16 and store in O_smem collectively
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint32_t val02 = pack_bf16_fn(*(uint32_t*)&O_reg[i*8+0], *(uint32_t*)&O_reg[i*8+1]);
            uint32_t val24 = pack_bf16_fn(*(uint32_t*)&O_reg[i*8+2], *(uint32_t*)&O_reg[i*8+3]);
            uint32_t val46 = pack_bf16_fn(*(uint32_t*)&O_reg[i*8+4], *(uint32_t*)&O_reg[i*8+5]);
            uint32_t val68 = pack_bf16_fn(*(uint32_t*)&O_reg[i*8+6], *(uint32_t*)&O_reg[i*8+7]);
            
            float4 out;
            out.x = *(float*)&val02;
            out.y = *(float*)&val24;
            out.z = *(float*)&val46;
            out.w = *(float*)&val68;
            
            ((float4*)smem_O)[tid * SMEM_STRIDE_VEC + i] = out;
        }
    }
    __syncthreads();

    // Write O to GMEM coalesced
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int idx = i * 64 + tid;
        int row = idx / 16;
        int col_vec = idx % 16;
        int q_idx_mem = q_start + row;
        if (q_idx_mem < S) {
            float4 val = ((float4*)smem_O)[row * SMEM_STRIDE_VEC + col_vec];
            ((float4*)&O[batch_head_offset + q_idx_mem * 128])[col_vec] = val;
        }
    }
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) 
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = 4;
    int64_t H = 48;
    int64_t S = Q.size(2);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());
    
    // We launch 64 threads per block; each handles exactly one sequence step for the batch_head coordinate
    int threads = 64;
    int grid_x = (S + 63) / 64;
    int grid_y = B * H;
    dim3 blocks(grid_x, grid_y, 1);
    
    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
        
    mha_fwd_kernel<<<blocks, threads, 0, stream>>>(Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, S);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda