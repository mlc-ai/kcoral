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

__device__ __forceinline__ uint32_t pack(float a, float b) {
    __nv_bfloat16 ba = __float2bfloat16(a);
    __nv_bfloat16 bb = __float2bfloat16(b);
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&ba)),
          "h"(*reinterpret_cast<uint16_t*>(&bb)));
    return result;
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
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (A is K-Major)
    d |= (0u << 16);   // b_major = 0 (B is K-Major)
    d |= ((N / 8) << 17);     // n_dim
    d |= ((M / 16) << 24);    // m_dim
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_trans_b(uint32_t M, uint32_t N) {
    uint32_t d = make_instr_desc_fn(M, N);
    d |= (1u << 16);   // b_major = 1 (B is MN-Major / Transposed)
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

__global__ __launch_bounds__(64, 2)
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
    uint64_t* mbar = (uint64_t*)(s_P_bf16 + 64*64);                   
    
    __shared__ uint32_t tmem_addr;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_addr, 256);
    }
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
    
    float running_max = -1e20f;
    float running_sum = 0.0f;
    float running_O0 = 0.0f;
    float running_O1 = 0.0f;
    
    int phase = 1;
    float scale = 1.0f / sqrtf(128.0f);
    
    uint32_t tmem_P_base = tmem_addr;
    uint32_t tmem_O0_base = tmem_addr + 64 * 128;
    uint32_t tmem_O1_base = tmem_addr + 128 * 128;
    
    uint32_t idesc_64x64 = make_instr_desc_fn(64, 64);
    uint32_t idesc_64x64_trans_b = make_instr_desc_fn_trans_b(64, 64);
    
    bool first_kv = true;
    
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
        
        if (threadIdx.x == 0) {
            bool first = true;
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_Q0 = make_smem_desc_sm100_fn((char*)s_Q0 + k * 2, 0, 1024);
                uint64_t desc_K0 = make_smem_desc_sm100_fn((char*)s_K0 + k * 2, 0, 1024);
                if (first) {
                    umma_f16_sm100(tmem_P_base, desc_Q0, desc_K0, idesc_64x64, false);
                    first = false;
                } else {
                    umma_f16_sm100(tmem_P_base, desc_Q0, desc_K0, idesc_64x64, true);
                }
                
                uint64_t desc_Q1 = make_smem_desc_sm100_fn((char*)s_Q1 + k * 2, 0, 1024);
                uint64_t desc_K1 = make_smem_desc_sm100_fn((char*)s_K1 + k * 2, 0, 1024);
                umma_f16_sm100(tmem_P_base, desc_Q1, desc_K1, idesc_64x64, true);
            }
        }
        
        float P_vals[64];
        int global_q = q_start + threadIdx.x;
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_P_base + col, &r0, &r1, &r2, &r3);
            P_vals[col+0] = __uint_as_float(r0);
            P_vals[col+1] = __uint_as_float(r1);
            P_vals[col+2] = __uint_as_float(r2);
            P_vals[col+3] = __uint_as_float(r3);
        }
        tmem_load_fence_fn();
        
        float block_max = -1e20f;
        for (int col = 0; col < 64; col++) {
            float p = P_vals[col];
            int global_k = k_start + col;
            if (global_q >= global_k && global_q < S && global_k < S) {
                P_vals[col] = p * scale;
                block_max = fmaxf(block_max, p * scale);
            } else {
                P_vals[col] = -1e20f;
            }
        }
        
        if (block_max > running_max) running_max = block_max;
        float alpha = expf(running_max - block_max);
        
        float block_sum = 0;
        for (int col = 0; col < 64; col++) {
            float p = P_vals[col];
            float e = expf(p - running_max);
            P_vals[col] = e; 
            block_sum += e;
        }
        
        float new_sum = running_sum * alpha + block_sum;
        
        running_O0 *= alpha;
        running_O1 *= alpha;
        
        running_sum = new_sum;
        
        for(int col = 0; col < 64; col++) {
            int chunk_x = col / 8;
            int chunk_y = threadIdx.x % 8;
            int col_swizzled = (chunk_y ^ chunk_x) * 8 + (col % 8);
            s_P_bf16[threadIdx.x * 64 + col_swizzled] = __float2bfloat16(P_vals[col]);
        }
        
        if (threadIdx.x == 0) {
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_P = make_smem_desc_sm100_fn((char*)s_P_bf16 + k * 2, 0, 1024);
                uint64_t desc_V0 = make_smem_desc_sm100_fn((char*)s_V0 + k * 128, 8192, 1024);
                uint64_t desc_V1 = make_smem_desc_sm100_fn((char*)s_V1 + k * 128, 8192, 1024);
                
                if (first_kv && k == 0) {
                    umma_f16_sm100(tmem_O0_base, desc_P, desc_V0, idesc_64x64_trans_b, false);
                    umma_f16_sm100(tmem_O1_base, desc_P, desc_V1, idesc_64x64_trans_b, false);
                } else {
                    umma_f16_sm100(tmem_O0_base, desc_P, desc_V0, idesc_64x64_trans_b, true);
                    umma_f16_sm100(tmem_O1_base, desc_P, desc_V1, idesc_64x64_trans_b, true);
                }
            }
            first_kv = false;
        }
        __syncthreads();
    }
    
    float O0_vals[64], O1_vals[64];
    for (int col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_O0_base + col, &r0, &r1, &r2, &r3);
        O0_vals[col+0] = __uint_as_float(r0);
        O0_vals[col+1] = __uint_as_float(r1);
        O0_vals[col+2] = __uint_as_float(r2);
        O0_vals[col+3] = __uint_as_float(r3);
        
        tmem_load_4x_fn(tmem_O1_base + col, &r0, &r1, &r2, &r3);
        O1_vals[col+0] = __uint_as_float(r0);
        O1_vals[col+1] = __uint_as_float(r1);
        O1_vals[col+2] = __uint_as_float(r2);
        O1_vals[col+3] = __uint_as_float(r3);
    }
    tmem_load_fence_fn();
    
    int global_q = q_start + threadIdx.x;
    if (global_q < S) {
        float rs = running_sum;
        for (int i = 0; i < 64; i++) {
            *(uint32_t*)((char*)&O[(bh * S + global_q) * 128 + i]) = pack(O0_vals[i] / rs, 0);
            *(uint32_t*)((char*)&O[(bh * S + global_q) * 128 + i + 64]) = pack(O1_vals[i] / rs, 0);
        }
        if (rs > 0) {
            LSE[bh * S + global_q] = running_max + logf(rs);
        } else {
            LSE[bh * S + global_q] = 0.0f;
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
                                
    int smem_size = 74000;
    CUDA_CHECK(cudaFuncSetAttribute(causal_attention_kernel, 
                        cudaFuncAttributeMaxDynamicSharedMemorySize, 
                        smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    causal_attention_kernel<<<total_blocks, 64, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel