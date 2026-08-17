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

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_dim0, uint64_t gmem_dim1, uint64_t gmem_dim2, uint64_t gmem_dim3, uint32_t smem_dim0, uint32_t smem_dim1, uint32_t smem_dim2, uint32_t smem_dim3, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[4] = {gmem_dim0, gmem_dim1, gmem_dim2, gmem_dim3};
    cuuint64_t globalStrides[3] = {gmem_dim0 * 2, gmem_dim0 * gmem_dim1 * 2, gmem_dim0 * gmem_dim1 * gmem_dim2 * 2};
    cuuint32_t boxDim[4] = {smem_dim0, smem_dim1, smem_dim2, smem_dim3};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim, globalStrides,
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

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
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

__global__ __launch_bounds__(256)
void causal_attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE, int S)
{
    int b_blk = blockIdx.x;
    int bh = b_blk % 192;
    int q_blk = b_blk / 192;
    int b_idx = bh % 4;
    int h_idx = bh / 4;
    
    int q_blk_wg0 = q_blk / 2;
    int q_blk_wg1 = q_blk - q_blk_wg0;
    
    int q_start_wg0 = q_blk_wg0 * 64;
    int q_start_wg1 = (q_blk_wg0 + q_blk_wg1) * 64;
    
    int wg_offset = (threadIdx.x < 128) ? 0 : 128;
    int q_start = (wg_offset == 0) ? q_start_wg0 : q_start_wg1;
    int q_blk_curr = (wg_offset == 0) ? q_blk_wg0 : q_blk_wg1;
    
    extern __shared__ __align__(1024) char smem_buf[];
    __nv_bfloat16* s_Q0 = (__nv_bfloat16*)smem_buf;                
    __nv_bfloat16* s_Q1 = s_Q0 + 64*64;                              
    __nv_bfloat16* s_K0 = s_Q1 + 64*64;                              
    __nv_bfloat16* s_K1 = s_K0 + 64*64;                              
    __nv_bfloat16* s_V0 = s_K1 + 64*64;                              
    __nv_bfloat16* s_V1 = s_V0 + 64*64;                              
    __nv_bfloat16* s_P_bf16_wg0 = s_V1 + 64*64;                       
    __nv_bfloat16* s_P_bf16_wg1 = s_P_bf16_wg0 + 64*64;               
    uint64_t* mbar = (uint64_t*)(s_P_bf16_wg1 + 64*64);               
    
    __shared__ uint32_t tmem_addr_wg0;
    __shared__ uint32_t tmem_addr_wg1;
    
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_addr_wg0, 512);
        tmem_alloc_fn(&tmem_addr_wg1, 512);
    }
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar, 1);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 16384);
        tma_load_4d_fn(&tma_Q, mbar, s_Q0, 0, q_start, h_idx, b_idx);
        tma_load_4d_fn(&tma_Q, mbar, s_Q1, 64, q_start, h_idx, b_idx);
    }
    mbarrier_wait_fn(mbar, 0);
    __syncthreads();
    
    int phase = 1;
    float scale = 1.0f / sqrtf(128.0f);
    
    float running_max = -1e20f;
    float running_sum = 0.0f;
    
    uint32_t tmem_P_base_wg0 = tmem_addr_wg0;
    uint32_t tmem_O_base_wg0 = tmem_addr_wg0 + 64;
    
    uint32_t tmem_P_base_wg1 = tmem_addr_wg1;
    uint32_t tmem_O_base_wg1 = tmem_addr_wg1 + 64;
    
    uint32_t tmem_P_base = (wg_offset == 0) ? tmem_P_base_wg0 : tmem_P_base_wg1;
    uint32_t tmem_O_base = (wg_offset == 0) ? tmem_O_base_wg0 : tmem_O_base_wg1;
    
    __nv_bfloat16* s_P_bf16 = (wg_offset == 0) ? s_P_bf16_wg0 : s_P_bf16_wg1;
    
    uint32_t idesc_64x64 = make_instr_desc_fn(64, 64);
    uint32_t idesc_64x64_k_n = make_instr_desc_fn_trans_b(64, 64);
    
    int warp_id_within_wg = (threadIdx.x - wg_offset) / 32;
    int lane_id = (threadIdx.x - wg_offset) % 32;
    int my_row = warp_id_within_wg * 32 + lane_id;
    int global_q = q_start + my_row;
    
    for (int block_idx = 0; block_idx <= q_blk_curr && block_idx * 64 < S; block_idx++) {
        int k_start = block_idx * 64;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 32768);
            tma_load_4d_fn(&tma_K, mbar, s_K0, 0, k_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_K, mbar, s_K1, 64, k_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, mbar, s_V0, 0, k_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, mbar, s_V1, 64, k_start, h_idx, b_idx);
        }
        mbarrier_wait_fn(mbar, phase);
        __syncthreads();
        phase ^= 1;
        
        if (threadIdx.x == 0) {
            bool first = true;
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_Q0 = make_smem_desc_sm100_fn((char*)s_Q0 + k * 2, 0, 1024);
                uint64_t desc_K0 = make_smem_desc_sm100_fn((char*)s_K0 + k * 128, 8192, 1024);
                if (first) {
                    umma_f16_sm100(tmem_P_base, desc_Q0, desc_K0, idesc_64x64_k_n, false);
                    first = false;
                } else {
                    umma_f16_sm100(tmem_P_base, desc_Q0, desc_K0, idesc_64x64_k_n, true);
                }
                
                uint64_t desc_Q1 = make_smem_desc_sm100_fn((char*)s_Q1 + k * 2, 0, 1024);
                uint64_t desc_K1 = make_smem_desc_sm100_fn((char*)s_K1 + k * 128, 8192, 1024);
                umma_f16_sm100(tmem_P_base, desc_Q1, desc_K1, idesc_64x64_k_n, true);
            }
        }
        
        float P_vals[8]; 
        for (int col = 0; col < 64; col += 8) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_P_base + col, &r0, &r1, &r2, &r3);
            P_vals[(col/8)*4 + 0] = __uint_as_float(r0);
            P_vals[(col/8)*4 + 1] = __uint_as_float(r1);
            P_vals[(col/8)*4 + 2] = __uint_as_float(r2);
            P_vals[(col/8)*4 + 3] = __uint_as_float(r3);
        }
        tmem_load_fence_fn();
        
        float thread_max = -1e20f;
        for(int i = 0; i < 8; i++) {
            float p = P_vals[i];
            int col = (i / 4) * 8 + (i % 4);
            int global_k = k_start + col;
            if (global_q >= global_k && global_q < S && global_k < S) {
                P_vals[i] = p * scale;
                if (p * scale > thread_max) thread_max = p * scale;
            } else {
                P_vals[i] = -1e20f;
            }
        }
        
        float max_val = thread_max;
        for (int offset = 2; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
        }
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, 1));
        
        float new_max = fmaxf(running_max, max_val);
        float alpha = expf(running_max - new_max);
        
        float thread_sum = 0;
        for(int i = 0; i < 8; i++) {
            float p = P_vals[i];
            float e = expf(p - new_max);
            thread_sum += e;
            
            int col = (i / 4) * 8 + (i % 4);
            int chunk_x = col / 8;
            int chunk_y = my_row % 8;
            int col_swizzled = (chunk_y ^ chunk_x) * 8 + (col % 8);
            s_P_bf16[my_row * 64 + col_swizzled] = __float2bfloat16(e);
        }
        
        for (int offset = 2; offset > 0; offset /= 2) {
            thread_sum += __shfl_xor_sync(0xFFFFFFFF, thread_sum, offset);
        }
        thread_sum += __shfl_xor_sync(0xFFFFFFFF, thread_sum, 1);
        
        float new_sum = running_sum * alpha + thread_sum;
        
        running_sum = new_sum;
        running_max = new_max;
        
        if (threadIdx.x == 0) {
            bool first_v = true;
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_p = make_smem_desc_sm100_fn((char*)s_P_bf16 + k * 2, 0, 1024);
                uint64_t desc_v0 = make_smem_desc_sm100_fn((char*)s_V0 + k * 128, 0, 1024);
                uint64_t desc_v1 = make_smem_desc_sm100_fn((char*)s_V1 + k * 128, 0, 1024);
                
                if (first_v) {
                    umma_f16_sm100(tmem_O_base, desc_p, desc_v0, idesc_64x64, false);
                    first_v = false;
                } else {
                    umma_f16_sm100(tmem_O_base, desc_p, desc_v0, idesc_64x64, true);
                }
            }
        }
        
        __syncthreads();
    }
    
    float O_vals[8];
    for (int col = 0; col < 64; col += 8) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_O_base + col, &r0, &r1, &r2, &r3);
        O_vals[(col/8)*4 + 0] = __uint_as_float(r0);
        O_vals[(col/8)*4 + 1] = __uint_as_float(r1);
        O_vals[(col/8)*4 + 2] = __uint_as_float(r2);
        O_vals[(col/8)*4 + 3] = __uint_as_float(r3);
    }
    tmem_load_fence_fn();
    
    float rs = running_sum;
    if (rs == 0.0f) rs = 1.0f;
    
    for(int i = 0; i < 8; i++) {
        float out_val = O_vals[i] / rs;
        int col = (i / 4) * 8 + (i % 4);
        int chunk_x = col / 8;
        int chunk_y = my_row % 8;
        int col_swizzled = (chunk_y ^ chunk_x) * 8 + (col % 8);
        
        if (wg_offset == 0) {
            s_Q0[my_row * 64 + col_swizzled] = __float2bfloat16(out_val);
        } else {
            s_Q1[my_row * 64 + col_swizzled] = __float2bfloat16(out_val);
        }
    }
    
    __syncthreads();
    
    uint32_t* O_u32 = (uint32_t*)O;
    for (int idx = threadIdx.x; idx < 64 * 64 / 2; idx += 256) {
        int tile = idx / (64 * 32);
        int row = (idx % (64 * 32)) / 32;
        int col_vec = (idx % (64 * 32)) % 32;
        int col = col_vec * 2;
        
        int wg = tile % 2;
        row += wg * 64; 
        
        __nv_bfloat16* src = (wg == 0) ? s_Q0 : s_Q1;
        int chunk_x = col / 8;
        int chunk_y = row % 8;
        int col_swizzled = (chunk_y ^ chunk_x) * 8 + (col % 8);
        
        uint32_t val = *(uint32_t*)&src[row * 64 + col_swizzled];
        
        int g_row = bh * S + q_start + row;
        if (g_row < S && col < 64) {
            O_u32[(g_row * 128 + wg * 64 + col) / 2] = val;
        }
    }
    
    if (my_row < 64) {
        int g_q = q_start + my_row;
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
    
    int64_t num_q_blocks = (S + 63) / 64;
    int64_t total_blocks = B * H * num_q_blocks;
    
    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, H, B, 64, 64, 1, 1, 
                                CU_TENSOR_MAP_SWIZZLE_128B, 
                                CU_TENSOR_MAP_L2_PROMOTION_NONE, 
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), D, S, H, B, 64, 64, 1, 1, 
                                CU_TENSOR_MAP_SWIZZLE_128B, 
                                CU_TENSOR_MAP_L2_PROMOTION_NONE, 
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), D, S, H, B, 64, 64, 1, 1, 
                                CU_TENSOR_MAP_SWIZZLE_128B, 
                                CU_TENSOR_MAP_L2_PROMOTION_NONE, 
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
                                
    int smem_size = 78000;
    CUDA_CHECK(cudaFuncSetAttribute(causal_attention_kernel, 
                        cudaFuncAttributeMaxDynamicSharedMemorySize, 
                        smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(total_blocks);
    config.blockDim = dim3(256);
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
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