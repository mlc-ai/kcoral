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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)


namespace tvm_ffi_optimized_cuda {

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_cta_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ __nv_bfloat16* get_swizzled_ptr_128B(__nv_bfloat16* base, int row, int col) {
    int x_int4 = col >> 3;
    int swizzled_x = (row & 7) ^ x_int4;
    int swizzled_col = (swizzled_x << 3) | (col & 7);
    return &base[(row << 6) | swizzled_col];
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        l2Promotion,
        oobFill
    );
}

__global__ void __launch_bounds__(128) mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* gmem_O,
    float* gmem_LSE,
    uint32_t S)
{
    extern __shared__ __align__(1024) char smem_pool[];
    
    // Allocate 8192 bytes (64x64 bf16) per sub-tile
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem_pool;                
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)(smem_pool + 8192);      
    __nv_bfloat16* smem_K0 = (__nv_bfloat16*)(smem_pool + 16384);      
    __nv_bfloat16* smem_K1 = (__nv_bfloat16*)(smem_pool + 24576);      
    __nv_bfloat16* smem_V0 = (__nv_bfloat16*)(smem_pool + 32768);      
    __nv_bfloat16* smem_V1 = (__nv_bfloat16*)(smem_pool + 40960);      
    float* smem_P = (float*)(smem_pool + 49152);     
    __nv_bfloat16* smem_S = (__nv_bfloat16*)(smem_pool + 49152 + 256);     
    
    // Aligned to 16 bytes
    float (*smem_P_partial)[64] = (float (*)[64])(smem_pool + 49152 + 256 + 128);
    float* tmp_smem = (float*)(smem_pool + 49152 + 256 + 128 + 1024);
    
    uint64_t* mbar_Q = (uint64_t*)(smem_pool + 49152 + 256 + 128 + 1024 + 512);                  
    uint64_t* mbar_KV = (uint64_t*)(smem_pool + 49152 + 256 + 128 + 1024 + 512 + 8);                 

    uint32_t idx = threadIdx.x;
    uint32_t row_base = blockIdx.y;
    uint32_t bh = blockIdx.x;

    if (idx == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_KV, 1);
    }
    __syncthreads();
    
    if (idx == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 16384);
        tma_load_2d_cta_fn(&tma_Q, mbar_Q, smem_Q0, 0, bh * S + row_base);
        tma_load_2d_cta_fn(&tma_Q, mbar_Q, smem_Q1, 64, bh * S + row_base);
    }
    mbarrier_wait_fn(mbar_Q, 0);
    fence_proxy_async_fn();

    float global_max = -INFINITY;
    float global_sum = 0.0f;
    float o_acc[128] = {0.0f};
    tmp_smem[idx] = o_acc[idx];
    
    uint32_t phase = 0;

    for (uint32_t k_blk = 0; k_blk < S/64; ++k_blk) {
        if (idx == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_KV, 32768);
            tma_load_2d_cta_fn(&tma_K, mbar_KV, smem_K0, 0, bh * S + k_blk * 64);
            tma_load_2d_cta_fn(&tma_K, mbar_KV, smem_K1, 64, bh * S + k_blk * 64);
            tma_load_2d_cta_fn(&tma_V, mbar_KV, smem_V0, 0, bh * S + k_blk * 64);
            tma_load_2d_cta_fn(&tma_V, mbar_KV, smem_V1, 64, bh * S + k_blk * 64);
        }
        mbarrier_wait_fn(mbar_KV, phase);
        fence_proxy_async_fn();
        phase ^= 1;
        
        __syncthreads(); 
        
        float my_max = -INFINITY;
        uint4 q_reg;
        if (idx < 64) {
            q_reg = *reinterpret_cast<uint4*>(get_swizzled_ptr_128B(smem_Q0, 0, idx));
        } else {
            q_reg = *reinterpret_cast<uint4*>(get_swizzled_ptr_128B(smem_Q1, 0, idx - 64));
        }
        __nv_bfloat16* q_ptr = reinterpret_cast<__nv_bfloat16*>(&q_reg);
        
        // Calculate P values for CURRENT block
        float my_partial = 0;
        for (int k = 0; k < 64; ++k) {
            uint4 k_reg;
            if (idx < 64) {
                k_reg = *reinterpret_cast<uint4*>(get_swizzled_ptr_128B(smem_K0, k, idx));
            } else {
                k_reg = *reinterpret_cast<uint4*>(get_swizzled_ptr_128B(smem_K1, k, idx - 64));
            }
            __nv_bfloat16* k_ptr = reinterpret_cast<__nv_bfloat16*>(&k_reg);
            
            for(int i = 0; i < 4; ++i) {
                my_partial += __bfloat162float(q_ptr[i]) * __bfloat162float(k_ptr[i]);
            }
            
            __syncwarp();
            my_partial += __shfl_down_sync(0xFFFFFFFF, my_partial, 16);
            my_partial += __shfl_down_sync(0xFFFFFFFF, my_partial, 8);
            my_partial += __shfl_down_sync(0xFFFFFFFF, my_partial, 4);
            my_partial += __shfl_down_sync(0xFFFFFFFF, my_partial, 2);
            my_partial += __shfl_down_sync(0xFFFFFFFF, my_partial, 1);
            
            if (idx % 32 == 0) {
                smem_P_partial[(idx / 32)][k] = my_partial;
            }
        }
        __syncwarp();
        __syncthreads();
        
        if (idx < 64) {
            float sum = 0;
            for(int i = 0; i < 4; ++i) {
                sum += smem_P_partial[i][idx];
            }
            smem_P[idx] = sum;
        }
        __syncthreads();
        
        if (idx < 64) {
            float max = -INFINITY;
            for (int k = 0; k < 64; ++k) {
                max = fmaxf(max, smem_P[k]);
            }
            
            float rmax = max;
            rmax = fmaxf(rmax, __shfl_xor_sync(0xFFFFFFFF, rmax, 1));
            rmax = fmaxf(rmax, __shfl_xor_sync(0xFFFFFFFF, rmax, 2));
            rmax = fmaxf(rmax, __shfl_xor_sync(0xFFFFFFFF, rmax, 4));
            
            float prev_max = global_max;
            float new_max = fmaxf(prev_max, rmax);
            float scale = __expf(prev_max - new_max);
            
            global_sum *= scale;
            global_max = new_max;
            
            float rsum = 0;
            for (int k = 0; k < 64; ++k) {
                float exp_val = __expf(smem_P[k] - rmax);
                smem_S[idx] = __float2bfloat16(exp_val);
                rsum += exp_val;
            }
            rsum += __shfl_xor_sync(0xFFFFFFFF, rsum, 1);
            rsum += __shfl_xor_sync(0xFFFFFFFF, rsum, 2);
            rsum += __shfl_xor_sync(0xFFFFFFFF, rsum, 4);
            
            if (idx == 0) {
                global_sum += rsum * __expf(rmax - new_max);
            }
            
            // Scale CURRENT o_acc
            for(int i = 0; i < 4; ++i) {
                q_ptr[i] = __float2bfloat16(__bfloat162float(q_ptr[i]) * scale);
            }
        }
        __syncthreads();
        
        for(int i = 0; i < 4; ++i) {
            tmp_smem[idx * 4 + i] = __bfloat162float(q_ptr[i]);
        }
        
        for (int k = 0; k < 64; ++k) {
            __nv_bfloat16 s_val = smem_S[k];
            uint4 v_reg;
            if (idx < 64) {
                v_reg = *reinterpret_cast<uint4*>(get_swizzled_ptr_128B(smem_V0, k, idx));
            } else {
                v_reg = *reinterpret_cast<uint4*>(get_swizzled_ptr_128B(smem_V1, k, idx - 64));
            }
            __nv_bfloat16* v_ptr = reinterpret_cast<__nv_bfloat16*>(&v_reg);
            
            for(int i = 0; i < 4; ++i) {
                tmp_smem[idx * 4 + i] += __bfloat162float(s_val) * __bfloat162float(v_ptr[i]);
            }
        }
        
        for(int i = 0; i < 4; ++i) {
            o_acc[idx * 4 + i] = tmp_smem[idx * 4 + i];
        }
        
        __syncthreads();
    }
    
    for(int i = 0; i < 4; ++i) {
        tmp_smem[idx * 4 + i] = o_acc[idx * 4 + i] / global_sum;
        __nv_bfloat16 out_val = __float2bfloat16(tmp_smem[idx * 4 + i]);
        if (row_base < S) {
            gmem_O[bh * S * 128 + row_base * 128 + idx * 4 + i] = out_val;
        }
    }
    
    if (idx == 0 && row_base < S) {
        *(gmem_LSE + bh * S + row_base) = global_max + __logf(global_sum);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t D = Q.size(3); 

    CUtensorMap tma_Q, tma_K, tma_V;
    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, q_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, k_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, v_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    __nv_bfloat16* o_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid(B * H, S);
    dim3 block(128);

    uint32_t smem_bytes = 128 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel, tma_Q, tma_K, tma_V, o_data, lse_data, S));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_optimized_cuda::run);

} // namespace tvm_ffi_optimized_cuda