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

__device__ __forceinline__ float2 unpack(uint32_t x) {
    uint16_t lo = x & 0xFFFF;
    uint16_t hi = x >> 16;
    __nv_bfloat16 bfa, bfb;
    *reinterpret_cast<uint16_t*>(&bfa) = lo;
    *reinterpret_cast<uint16_t*>(&bfb) = hi;
    return {__bfloat162float(bfa), __bfloat162float(bfb)};
}

__device__ __forceinline__ int32_t load_swizzled(__nv_bfloat16* smem, int32_t row, int32_t col) {
    int32_t chunk_x = col / 8;
    int32_t chunk_y = row % 8;
    int32_t col_swizzled = (chunk_y ^ chunk_x) * 8 + (col % 8);
    return smem[row * 64 + col_swizzled];
}

__device__ __forceinline__ uint32_t load_u32(__nv_bfloat16* smem, int32_t row, int32_t col) {
    int32_t chunk_x = col / 8;
    int32_t chunk_y = row % 8;
    int32_t col_swizzled = (chunk_y ^ chunk_x) * 8 + (col % 8);
    return *(uint32_t*)&smem[row * 64 + col_swizzled];
}

__device__ __forceinline__ int32_t swizzled_col(int32_t row, int32_t col) {
    int32_t chunk_x = col / 8;
    int32_t chunk_y = row % 8;
    return (chunk_y ^ chunk_x) * 8 + (col % 8);
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
    
    int wg_id = (threadIdx.x < 64) ? 0 : 1;
    int row = threadIdx.x % 64;
    
    int global_q = q_start + row;
    
    extern __shared__ __align__(1024) char smem_buf[];
    __nv_bfloat16* s_Q0 = (__nv_bfloat16*)smem_buf;                
    __nv_bfloat16* s_Q1 = s_Q0 + 64*64;                              
    __nv_bfloat16* s_K0 = s_Q1 + 64*64;                              
    __nv_bfloat16* s_K1 = s_K0 + 64*64;                              
    __nv_bfloat16* s_V0 = s_K1 + 64*64;                              
    __nv_bfloat16* s_V1 = s_V0 + 64*64;                              
    __nv_bfloat16* s_P = s_V1 + 64*64;                               
    uint64_t* mbar = (uint64_t*)(s_P + 64*64);                       
    
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
    
    float O0 = 0.0f, O1 = 0.0f;
    
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
        __nv_bfloat16* my_Q = (wg_id == 0) ? s_Q0 : s_Q1;
        __nv_bfloat16* my_K0 = (wg_id == 0) ? s_K0 : s_K1;
        
        float P_my[64];
        for (int col = 0; col < 64; col++) {
            float sum = 0;
            for(int i=0; i<64; i+=2) {
                uint32_t q = load_u32(my_Q, row, i);
                uint32_t k = load_u32(my_K0, col, i);
                float2 qv = unpack(q);
                float2 kv = unpack(k);
                sum += qv.x * kv.x + qv.y * kv.y;
            }
            int global_k = k_start + col;
            if (global_q >= global_k && global_q < S && global_k < S) {
                float p = sum * scale;
                P_my[col] = p;
                if (p > block_max) block_max = p;
            } else {
                P_my[col] = -1e20f;
            }
        }
        
        float new_max = fmaxf(running_max, block_max);
        float alpha = expf(running_max - new_max);
        float beta = expf(block_max - new_max);
        
        float block_sum = 0;
        for (int col = 0; col < 64; col++) {
            float e = expf(P_my[col] - new_max);
            block_sum += e;
            
            int col_swizzled = swizzled_col(row, col);
            s_P[row * 64 + col_swizzled] = __float2bfloat16(e);
        }
        
        float new_sum = running_sum * alpha + block_sum;
        
        O0 *= alpha;
        O1 *= alpha;
        
        running_sum = new_sum;
        running_max = new_max;
        
        float sum_half = 0;
        __nv_bfloat16* my_V0 = (wg_id == 0) ? s_V0 : s_V1;
        for(int col = 0; col < 64; col++) {
            float e = __bfloat162float(s_P[row * 64 + swizzled_col(row, col)]);
            for(int i=0; i<64; i+=2) {
                uint32_t v = load_u32(my_V0, col, i);
                float2 vv = unpack(v);
                sum_half += e * vv.x + e * vv.y;
            }
        }
        
        if (wg_id == 0) {
            O0 += sum_half;
        } else {
            O1 += sum_half;
        }
        
        __syncthreads();
    }
    
    float rs = (running_sum > 0) ? running_sum : 1.0f;
    int g_row = bh * S + global_q;
    
    if (global_q < S) {
        if (wg_id == 0) {
            for(int i=0; i<64; i++) {
                O[g_row * 128 + i] = __float2bfloat16(O0 / rs);
            }
        } else {
            for(int i=0; i<64; i++) {
                O[g_row * 128 + 64 + i] = __float2bfloat16(O1 / rs);
            }
        }
    }
    
    if (threadIdx.x < 64) {
        int g_q = q_start + threadIdx.x;
        if (g_q < S) {
            LSE[bh * S + g_q] = (running_sum > 0) ? (running_max + logf(running_sum)) : 0.0f;
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
                                
    int smem_size = 58000;
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