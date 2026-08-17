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
        fprintf(stderr, "CU error %d at %s:%d\n",               \
                _e, __FILE__, __LINE__);                         \
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

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ float read_bf16_swizzled(__nv_bfloat16* smem, int row, int col) {
    int chunk_x = col / 8;
    int chunk_y = row % 8;
    int chunk_x_swizzled = chunk_y ^ chunk_x;
    int col_swizzled = chunk_x_swizzled * 8 + (col % 8);
    return __bfloat162float(smem[row * 64 + col_swizzled]);
}

__global__ __launch_bounds__(128)
void causal_attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE, int S)
{
    int b_blk = blockIdx.x;
    int bh = b_blk % 192;
    int q_blk = b_blk / 192;
    int q_start = q_blk * 64;
    
    extern __shared__ __align__(1024) char smem_buf[];
    __nv_bfloat16* s_Q0 = (__nv_bfloat16*)smem_buf;                
    __nv_bfloat16* s_Q1 = s_Q0 + 64*64;                              
    __nv_bfloat16* s_K0 = s_Q1 + 64*64;                              
    __nv_bfloat16* s_K1 = s_K0 + 64*64;                              
    __nv_bfloat16* s_V0 = s_K1 + 64*64;                              
    __nv_bfloat16* s_V1 = s_V0 + 64*64;                              
    __nv_bfloat16* s_P_bf16 = s_V1 + 64*64;                           
    float* s_P = (float*)(s_P_bf16 + 64*64);                        
    __nv_bfloat16* s_O0 = (__nv_bfloat16*)(s_P + 64*64);             
    __nv_bfloat16* s_O1 = s_O0 + 64*64;                              
    uint64_t* mbar = (uint64_t*)(s_O1 + 64*64);                       
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar, 1);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 16384);
        tma_load_2d_fn(&tma_Q, mbar, s_Q0, 0, bh * S + q_start);
        tma_load_2d_fn(&tma_Q, mbar, s_Q1, 64, bh * S + q_start);
    }
    mbarrier_wait_fn(mbar, 0);
    __syncthreads();
    
    int phase = 1;
    float scale = 1.0f / sqrtf(128.0f);
    
    float running_max = -1e20f;
    float running_sum = 0.0f;
    
    float running_O0[64];
    float running_O1[64];
    #pragma unroll
    for (int i=0; i<64; i++) {
        running_O0[i] = 0;
        running_O1[i] = 0;
    }
    
    int my_row = threadIdx.x % 64;
    int global_q = q_start + my_row;
    
    for (int block_idx = 0; block_idx <= q_blk && block_idx * 64 < S; block_idx++) {
        int k_start = block_idx * 64;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 32768);
            tma_load_2d_fn(&tma_K, mbar, s_K0, 0, bh * S + k_start);
            tma_load_2d_fn(&tma_K, mbar, s_K1, 64, bh * S + k_start);
            tma_load_2d_fn(&tma_V, mbar, s_V0, 0, bh * S + k_start);
            tma_load_2d_fn(&tma_V, mbar, s_V1, 64, bh * S + k_start);
        }
        mbarrier_wait_fn(mbar, phase);
        __syncthreads();
        phase ^= 1;
        
        float block_max = -1e20f;
        for (int col_chunk = 0; col_chunk < 64; col_chunk++) {
            float p_val = 0;
            for(int d=0; d<64; d++) {
                p_val += read_bf16_swizzled(s_Q0, my_row, d) * read_bf16_swizzled(s_K0, col_chunk, d);
                p_val += read_bf16_swizzled(s_Q1, my_row, d) * read_bf16_swizzled(s_K1, col_chunk, d);
            }
            
            int global_k = k_start + col_chunk;
            if (global_q >= global_k && global_q < S && global_k < S) {
                s_P[my_row * 64 + col_chunk] = p_val * scale;
                if (p_val * scale > block_max) block_max = p_val * scale;
            } else {
                s_P[my_row * 64 + col_chunk] = -1e20f;
            }
        }
        
        float new_max = fmaxf(running_max, block_max);
        float alpha = expf(running_max - new_max);
        
        float block_sum = 0;
        for (int col_chunk = 0; col_chunk < 64; col_chunk++) {
            float p = s_P[my_row * 64 + col_chunk];
            float e = expf(p - new_max);
            s_P_bf16[my_row * 64 + col_chunk] = __float2bfloat16(e);
            block_sum += e;
        }
        
        float new_sum = running_sum * alpha + block_sum;
        
        #pragma unroll
        for (int i=0; i<64; i++) {
            running_O0[i] *= alpha;
            running_O1[i] *= alpha;
        }
        
        running_sum = new_sum;
        running_max = new_max;
        
        #pragma unroll
        for(int col = 0; col < 64; col++) {
            float sum_x = 0;
            float sum_y = 0;
            for(int k=0; k<64; k++) {
                float e = __bfloat162float(s_P_bf16[my_row * 64 + k]);
                sum_x += e * read_bf16_swizzled(s_V0, k, col);
                sum_y += e * read_bf16_swizzled(s_V1, k, col);
            }
            running_O0[col] += sum_x;
            running_O1[col] += sum_y;
        }
        
        __syncthreads();
    }
    
    float rs = running_sum;
    
    #pragma unroll
    for (int i=0; i<64; i++) {
        s_O0[my_row * 64 + i] = __float2bfloat16(running_O0[i] / rs);
        s_O1[my_row * 64 + i] = __float2bfloat16(running_O1[i] / rs);
    }
    
    __syncthreads();
    
    for (int idx = threadIdx.x; idx < 64 * 64; idx += 128) {
        int row = idx / 64;
        int col = idx % 64;
        int g_row = bh * S + q_start + row;
        if (g_row < S) {
            *(uint32_t*)((char*)&O[g_row * 128 + col]) = *(uint32_t*)((char*)&s_O0[row * 64 + col]);
            *(uint32_t*)((char*)&O[g_row * 128 + 64 + col]) = *(uint32_t*)((char*)&s_O1[row * 64 + col]);
        }
    }
    
    if (threadIdx.x < 64) {
        int g_q = q_start + threadIdx.x;
        if (g_q < S) {
            LSE[bh * S + g_q] = running_max + logf(running_sum);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    int64_t BH = B * H;
    int64_t num_q_blocks = (S + 63) / 64;
    int64_t total_blocks = BH * num_q_blocks;
    
    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), D, BH * S, 64, 64, 
                                CU_TENSOR_MAP_SWIZZLE_128B, 
                                CU_TENSOR_MAP_L2_PROMOTION_NONE, 
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), D, BH * S, 64, 64, 
                                CU_TENSOR_MAP_SWIZZLE_128B, 
                                CU_TENSOR_MAP_L2_PROMOTION_NONE, 
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), D, BH * S, 64, 64, 
                                CU_TENSOR_MAP_SWIZZLE_128B, 
                                CU_TENSOR_MAP_L2_PROMOTION_NONE, 
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
                                
    int smem_size = 95000;
    CUDA_CHECK(cudaFuncSetAttribute(causal_attention_kernel, 
                        cudaFuncAttributeMaxDynamicSharedMemorySize, 
                        smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    causal_attention_kernel<<<total_blocks, 128, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel