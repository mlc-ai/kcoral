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

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* smem_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    return ((uint64_t)(addr >> 4)) | (1024ULL << 16) | (1024ULL << 32) | (1ULL << 46) | (2ULL << 61);
}

__device__ __forceinline__ uint64_t make_smem_desc_n_major(void* smem_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    return ((uint64_t)(addr >> 4)) | (8192ULL << 16) | (1024ULL << 32) | (1ULL << 46) | (2ULL << 61);
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (0u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_trans_b(uint32_t M, uint32_t N) {
    uint32_t d = make_instr_desc_fn(M, N);
    d |= (1u << 16);   
    return d;
}

__device__ __forceinline__ void umma_f16_sm100(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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
    int q_start = q_blk * 128;
    
    int q_start_wg0 = q_start;
    int q_start_wg1 = q_start + 64;
    
    int wg_id = (threadIdx.x < 64) ? 0 : 1;
    int wg_offset = wg_id * 64;
    int lane_id = threadIdx.x % 32;
    int warp_id_within_wg = (threadIdx.x - wg_offset) / 32;
    int my_row = warp_id_within_wg * 32 + lane_id;
    int my_wg_offset = (wg_id == 0) ? 0 : 128;
    
    int global_q = (wg_id == 0) ? (q_start_wg0 + my_row) : (q_start_wg1 + my_row);
    
    extern __shared__ __align__(1024) char smem_buf[];
    __nv_bfloat16* s_Q0_wg0 = (__nv_bfloat16*)smem_buf;                
    __nv_bfloat16* s_Q1_wg0 = s_Q0_wg0 + 64*64;                              
    __nv_bfloat16* s_Q0_wg1 = s_Q1_wg0 + 64*64;                              
    __nv_bfloat16* s_Q1_wg1 = s_Q0_wg1 + 64*64;                              
    __nv_bfloat16* s_K0 = s_Q1_wg1 + 64*64;                                  
    __nv_bfloat16* s_K1 = s_K0 + 64*64;                                      
    __nv_bfloat16* s_V0 = s_K1 + 64*64;                                      
    __nv_bfloat16* s_V1 = s_V0 + 64*64;                                      
    __nv_bfloat16* s_P_bf16_wg0 = s_V1 + 64*64;                               
    __nv_bfloat16* s_P_bf16_wg1 = s_P_bf16_wg0 + 64*64;                       
    uint64_t* mbar = (uint64_t*)(s_P_bf16_wg1 + 64*64);                       
    
    __shared__ uint32_t tmem_addr_wg0;
    __shared__ uint32_t tmem_addr_wg1;
    
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_addr_wg0, 128);
        tmem_alloc_fn(&tmem_addr_wg1, 128);
    }
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar, 1);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 32768);
        tma_load_2d_fn(&tma_Q, mbar, s_Q0_wg0, 0, bh * S + q_start_wg0);
        tma_load_2d_fn(&tma_Q, mbar, s_Q1_wg0, 64, bh * S + q_start_wg0);
        tma_load_2d_fn(&tma_Q, mbar, s_Q0_wg1, 0, bh * S + q_start_wg1);
        tma_load_2d_fn(&tma_Q, mbar, s_Q1_wg1, 64, bh * S + q_start_wg1);
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
        running_O0[i] = 0.0f;
        running_O1[i] = 0.0f;
    }
    
    uint32_t tmem_P_base_wg0 = tmem_addr_wg0;
    uint32_t tmem_O1_base_wg0 = tmem_addr_wg0 + 64;
    
    uint32_t tmem_P_base_wg1 = tmem_addr_wg1;
    uint32_t tmem_O1_base_wg1 = tmem_addr_wg1 + 64;
    
    uint32_t tmem_P_base = (wg_id == 0) ? tmem_P_base_wg0 : tmem_P_base_wg1;
    uint32_t tmem_O0_base = tmem_P_base; 
    uint32_t tmem_O1_base = (wg_id == 0) ? tmem_O1_base_wg0 : tmem_O1_base_wg1;
    
    uint32_t idesc_64x64 = make_instr_desc_fn(64, 64);
    uint32_t idesc_64x64_n_major = make_instr_desc_fn_trans_b(64, 64);
    
    __nv_bfloat16* s_P_bf16 = (wg_id == 0) ? s_P_bf16_wg0 : s_P_bf16_wg1;
    __nv_bfloat16* s_Q0 = (wg_id == 0) ? s_Q0_wg0 : s_Q0_wg1;
    __nv_bfloat16* s_Q1 = (wg_id == 0) ? s_Q1_wg0 : s_Q1_wg1;
    
    int q_blk_curr = (wg_id == 0) ? (q_start_wg0 / 64) : (q_start_wg1 / 64);
    
    uint32_t my_col_base = (my_row % 64) << 16;

    for (int block_idx = 0; block_idx <= q_blk_curr && block_idx * 64 < S; block_idx++) {
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
        
        if (threadIdx.x == 0) {
            bool first = true;
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_Q0 = make_smem_desc_k_major((char*)s_Q0 + k * 2);
                uint64_t desc_K0 = make_smem_desc_k_major((char*)s_K0 + k * 2);
                if (first) {
                    umma_f16_sm100(tmem_P_base, desc_Q0, desc_K0, idesc_64x64, false);
                    first = false;
                } else {
                    umma_f16_sm100(tmem_P_base, desc_Q0, desc_K0, idesc_64x64, true);
                }
                
                uint64_t desc_Q1 = make_smem_desc_k_major((char*)s_Q1 + k * 2);
                uint64_t desc_K1 = make_smem_desc_k_major((char*)s_K1 + k * 2);
                umma_f16_sm100(tmem_P_base, desc_Q1, desc_K1, idesc_64x64, true);
            }
        }
        
        float P_vals[64];
        for (int col = 0; col < 64; col += 4) {
            uint32_t tmem_P_addr = tmem_P_base + my_col_base + col;
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_P_addr, &r0, &r1, &r2, &r3);
            P_vals[col] = __uint_as_float(r0);
            P_vals[col+1] = __uint_as_float(r1);
            P_vals[col+2] = __uint_as_float(r2);
            P_vals[col+3] = __uint_as_float(r3);
        }
        tmem_load_fence_fn();
        
        float thread_max = -1e20f;
        for(int col = 0; col < 64; col++) {
            float p = P_vals[col];
            int global_k = k_start + col;
            if (global_q >= global_k && global_q < S && global_k < S) {
                P_vals[col] = p * scale;
                if (p * scale > thread_max) thread_max = p * scale;
            } else {
                P_vals[col] = -1e20f;
            }
        }
        
        float max_val = thread_max;
        for(int offset = 16; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
        }
        
        float new_max = fmaxf(running_max, max_val);
        float alpha = expf(running_max - new_max);
        
        #pragma unroll
        for(int i = 0; i < 64; i++) {
            running_O0[i] *= alpha;
            running_O1[i] *= alpha;
        }
        
        float thread_sum = 0;
        for(int col = 0; col < 64; col++) {
            float p = P_vals[col];
            float e = expf(p - new_max);
            thread_sum += e;
            
            int chunk_x = col / 8;
            int chunk_y = my_row % 8;
            int col_swizzled = (chunk_y ^ chunk_x) * 8 + (col % 8);
            s_P_bf16[my_row * 64 + col_swizzled] = __float2bfloat16(e);
        }
        
        for(int offset = 16; offset > 0; offset /= 2) {
            thread_sum += __shfl_xor_sync(0xFFFFFFFF, thread_sum, offset);
        }
        
        float new_sum = running_sum * alpha + thread_sum;
        running_sum = new_sum;
        running_max = new_max;
        
        fence_async_shared_fn();
        
        if (threadIdx.x == 0) {
            bool first_v = true;
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_p = make_smem_desc_k_major((char*)s_P_bf16 + k * 2);
                uint64_t desc_v0 = make_smem_desc_n_major((char*)s_V0 + k * 128);
                uint64_t desc_v1 = make_smem_desc_n_major((char*)s_V1 + k * 128);
                
                if (first_v) {
                    umma_f16_sm100(tmem_O0_base, desc_p, desc_v0, idesc_64x64_n_major, false);
                    umma_f16_sm100(tmem_O1_base, desc_p, desc_v1, idesc_64x64_n_major, false);
                    first_v = false;
                } else {
                    umma_f16_sm100(tmem_O0_base, desc_p, desc_v0, idesc_64x64_n_major, true);
                    umma_f16_sm100(tmem_O1_base, desc_p, desc_v1, idesc_64x64_n_major, true);
                }
            }
        }
        
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_O0_base + my_col_base + col, &r0, &r1, &r2, &r3);
            running_O0[col] += __uint_as_float(r0);
            running_O0[col+1] += __uint_as_float(r1);
            running_O0[col+2] += __uint_as_float(r2);
            running_O0[col+3] += __uint_as_float(r3);
            
            tmem_load_4x_fn(tmem_O1_base + my_col_base + col, &r0, &r1, &r2, &r3);
            running_O1[col] += __uint_as_float(r0);
            running_O1[col+1] += __uint_as_float(r1);
            running_O1[col+2] += __uint_as_float(r2);
            running_O1[col+3] += __uint_as_float(r3);
        }
        tmem_load_fence_fn();
        
        __syncthreads();
    }
    
    float rs = running_sum;
    if (rs == 0.0f) rs = 1.0f;
    
    #pragma unroll
    for (int i=0; i<64; i++) {
        int col = i;
        int chunk_x = col / 8;
        int chunk_y = my_row % 8;
        int col_swizzled = (chunk_y ^ chunk_x) * 8 + (col % 8);
        
        if (wg_id == 0) {
            s_Q0[my_row * 64 + col_swizzled] = __float2bfloat16(running_O0[i] / rs);
            s_Q1[my_row * 64 + col_swizzled] = __float2bfloat16(running_O1[i] / rs);
        } else {
            s_Q0[my_row * 64 + col_swizzled] = __float2bfloat16(running_O0[i] / rs);
            s_Q1[my_row * 64 + col_swizzled] = __float2bfloat16(running_O1[i] / rs);
        }
    }
    
    __syncthreads();
    
    uint32_t* O_u32 = (uint32_t*)O;
    for (int step = 0; step < 16; step++) {
        int idx = step * 128 + threadIdx.x; 
        if (idx < 512) {
            int row = idx / 4;
            int col_u32 = idx % 4;
            int col = col_u32 * 2;
            
            int wg = row / 64;
            int row_in_wg = row % 64;
            
            int g_row = bh * S + ((wg == 0) ? q_start_wg0 : q_start_wg1) + row_in_wg;
            if (g_row < S && col < 64) {
                int chunk_x = col / 8;
                int chunk_y = row_in_wg % 8;
                int col_swizzled = (chunk_y ^ chunk_x) * 8 + (col % 8);
                
                __nv_bfloat16* src = (wg == 0) ? s_Q0 : s_Q0;
                uint32_t val0 = *(uint32_t*)&src[row_in_wg * 64 + col_swizzled];
                O_u32[(g_row * 128 + wg * 64 + col) / 2] = val0;
                
                src = (wg == 0) ? s_Q1 : s_Q1;
                uint32_t val1 = *(uint32_t*)&src[row_in_wg * 64 + col_swizzled];
                O_u32[(g_row * 128 + wg * 64 + col + 64) / 2] = val1;
            }
        }
    }
    
    if (my_row < 64) {
        int g_q = (wg_id == 0) ? (q_start_wg0 + my_row) : (q_start_wg1 + my_row);
        if (g_q < S) {
            if (running_sum > 0) {
                LSE[bh * S + g_q] = running_max + logf(running_sum);
            } else {
                LSE[bh * S + g_q] = 0.0f;
            }
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
    
    int64_t num_q_blocks = (S + 127) / 128;
    int64_t total_blocks = B * H * num_q_blocks;
    
    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), D, B * H * S, 64, 64, 
                                CU_TENSOR_MAP_SWIZZLE_128B, 
                                CU_TENSOR_MAP_L2_PROMOTION_NONE, 
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), D, B * H * S, 64, 64, 
                                CU_TENSOR_MAP_SWIZZLE_128B, 
                                CU_TENSOR_MAP_L2_PROMOTION_NONE, 
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), D, B * H * S, 64, 64, 
                                CU_TENSOR_MAP_SWIZZLE_128B, 
                                CU_TENSOR_MAP_L2_PROMOTION_NONE, 
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
                                
    int smem_size = 95000;
    CUDA_CHECK(cudaFuncSetAttribute(causal_attention_kernel, 
                        cudaFuncAttributeMaxDynamicSharedMemorySize, 
                        smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(total_blocks);
    config.blockDim = dim3(128);
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, causal_attention_kernel,
        tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel