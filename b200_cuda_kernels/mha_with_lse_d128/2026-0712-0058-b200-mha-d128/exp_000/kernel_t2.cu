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

// -------------------------------------------------------------------------
// Minimal inline PTX wrappers optimized for SM100 execution topology
// -------------------------------------------------------------------------

__device__ __forceinline__ float fast_exp2f_fn(float x, float log2e) {
    x *= log2e;
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

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

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
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

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
       :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_f16_cg1_scaled_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, float scale) {
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1, %4;\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "f"(scale));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzled(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzled_k_slice(void* smem_ptr, uint32_t slice) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) + slice * 32;
    uint64_t d = (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((1 & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((1024 & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzled_v(void* smem_ptr) {
    return make_smem_desc_swizzled(smem_ptr, 8192, 1024);
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

__device__ __forceinline__ uint32_t make_instr_desc_transposed(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);     
    d |= (1u << 7);     
    d |= (1u << 10);    
    d |= (0u << 15);    
    d |= (1u << 16);    
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3, uint32_t box0, uint32_t box1, uint32_t box2, uint32_t box3, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {box0, box1, box2, box3};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim,
        globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

// -------------------------------------------------------------------------
// Advanced Software-Pipelined Attention Kernel targeting SM100 Architecture
// -------------------------------------------------------------------------

__global__ void __launch_bounds__(128) attention_kernel_swizzled(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    uint32_t S, uint32_t H, uint32_t B, float* LSE) 
{
    uint32_t bh = blockIdx.y;         
    uint32_t sq_block = blockIdx.x;   
    
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    uint8_t* s_Q0 = smem_pool;              // 8192 bytes
    uint8_t* s_Q1 = smem_pool + 8192;       // 8192 bytes
    uint8_t* s_K0 = smem_pool + 16384;      // 8192 bytes
    uint8_t* s_K1 = smem_pool + 24576;      // 8192 bytes
    uint8_t* s_V0 = smem_pool + 32768;      // 8192 bytes
    uint8_t* s_V1 = smem_pool + 40960;      // 8192 bytes
    uint8_t* s_P  = smem_pool + 49152;      // 8192 bytes
    
    uint64_t* bar_Q = (uint64_t*)(smem_pool + 57344);
    uint64_t* bar_K = (uint64_t*)(smem_pool + 57352);
    uint64_t* bar_V = (uint64_t*)(smem_pool + 57360);
    uint32_t* tmem_S = (uint32_t*)(smem_pool + 57368);
    uint32_t* tmem_P = (uint32_t*)(smem_pool + 57372);
    uint32_t* tmem_O_0 = (uint32_t*)(smem_pool + 57376);
    uint32_t* tmem_O_1 = (uint32_t*)(smem_pool + 57380);

    const float log2e = 1.4426950408889634f;
    const uint32_t D = 128;
    const uint32_t M = 64;
    const uint32_t N = 64;
    const uint32_t HEAD_SCALE = 1.0f / sqrtf(128.0f);

    if (threadIdx.x == 0) {
        tmem_alloc_fn(tmem_S, 64);
        tmem_alloc_fn(tmem_P, 64);
        tmem_alloc_fn(tmem_O_0, 64);
        tmem_alloc_fn(tmem_O_1, 64);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_K, 1);
        init_smem_barrier_fn(bar_V, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t tmem_S_base = *tmem_S;
    uint32_t tmem_P_base = *tmem_P;
    uint32_t tmem_O_0_base = *tmem_O_0;
    uint32_t tmem_O_1_base = *tmem_O_1;

    int h_idx = bh % H;
    int b_idx = bh / H;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_Q, 16384);
        tma_load_4d_fn(&tma_Q, bar_Q, s_Q0, 0, sq_block * 64, h_idx, b_idx);
        tma_load_4d_fn(&tma_Q, bar_Q, s_Q1, 64, sq_block * 64, h_idx, b_idx);
    }
    
    mbarrier_wait_fn(bar_Q, 0);

    float l_sum = 0;
    float m_global = -1e20f;
    uint32_t phase_K = 0, phase_V = 0;

    for (uint32_t kv_iter = 0; kv_iter < (S + 63) / 64; ++kv_iter) {
        uint32_t kv_block = kv_iter * 64;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_K, 16384);
            tma_load_4d_fn(&tma_K, bar_K, s_K0, 0, kv_block, h_idx, b_idx);
            tma_load_4d_fn(&tma_K, bar_K, s_K1, 64, kv_block, h_idx, b_idx);

            mbarrier_arrive_and_expect_tx_fn(bar_V, 16384);
            tma_load_4d_fn(&tma_V, bar_V, s_V0, 0, kv_block, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, bar_V, s_V1, 64, kv_block, h_idx, b_idx);
        }
        
        mbarrier_wait_fn(bar_K, phase_K);
        mbarrier_wait_fn(bar_V, phase_V);
        
        float row_max = -1e20f;
        float regs_p[64];
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_S_base + col));
            tmem_load_fence_fn();
            
            for(int i = 0; i < 4; i++) {
                float val = __uint_as_float(((col << 8) | (i << 6)) + (threadIdx.x & 63));
                uint32_t global_sq_idx = sq_block * 64 + threadIdx.x;
                uint32_t global_kv_idx = kv_block + col + i;
                if (global_kv_idx < S && global_sq_idx < S) {
                    val *= HEAD_SCALE;
                } else {
                    val = -1e20f;
                }
                if (val > row_max) row_max = val;
            }
        }
        
        for (uint32_t offset = 1; offset < 32; offset *= 2) {
            row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, offset));
        }
        
        float row_sum = 0;
        for (uint32_t col = 0; col < 64; col += 4) {
            for(int i = 0; i < 4; i++) {
                float val = __uint_as_float(((col << 8) | (i << 6)) + (threadIdx.x & 63));
                uint32_t global_sq_idx = sq_block * 64 + threadIdx.x;
                uint32_t global_kv_idx = kv_block + col + i;
                float val_f = 0;
                if (global_kv_idx < S && global_sq_idx < S) {
                    val *= HEAD_SCALE; 
                    val_f = fast_exp2f_fn(val - row_max, log2e);
                    row_sum += val_f;
                }
                regs_p[col + i] = val_f;
                
                uint32_t row = threadIdx.x;
                uint32_t chunk = col / 8;
                uint32_t rem = col % 8;
                uint32_t swizzled_chunk = chunk ^ (row % 8);
                uint32_t swizzled_col = swizzled_chunk * 8 + rem;
                ((__nv_bfloat16*)s_P)[row * 64 + swizzled_col] = __float2bfloat16(val_f);
            }
        }
        
        for (uint32_t offset = 1; offset < 32; offset *= 2) {
            row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, offset);
        }
        
        float m_new = fmaxf(m_global, row_max);
        float m_scale = fast_exp2f_fn(m_global - m_new, log2e);
        l_sum = l_sum * m_scale + row_sum;
        m_global = m_new;

        __syncthreads(); 
        fence_proxy_async_fn(); 
        
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t p0 = pack_bf16_fn(__float_as_uint(regs_p[col]), __float_as_uint(regs_p[col+1]));
            uint32_t p1 = pack_bf16_fn(__float_as_uint(regs_p[col+2]), __float_as_uint(regs_p[col+3]));
            uint32_t p2 = pack_bf16_fn(__float_as_uint(regs_p[col+4]), __float_as_uint(regs_p[col+5]));
            uint32_t p3 = pack_bf16_fn(__float_as_uint(regs_p[col+6]), __float_as_uint(regs_p[col+7]));
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                : : "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(tmem_P_base + col));
        }
        
        for (uint32_t i = 0; i < 4; ++i) {
            uint64_t desc_A = make_smem_desc_swizzled_k_slice(s_P, i);
            uint64_t desc_B0 = make_smem_desc_v(s_V0 + i * 16 * 128);
            uint64_t desc_B1 = make_smem_desc_v(s_V1 + i * 16 * 128);
            
            if (i == 0) {
                umma_f16_cg1_scaled_fn(tmem_O_0_base, desc_A, desc_B0, make_instr_desc_transposed(M, 64), m_scale);
                umma_f16_cg1_scaled_fn(tmem_O_1_base, desc_A, desc_B1, make_instr_desc_transposed(M, 64), m_scale);
            } else {
                umma_f16_cg1_fn(tmem_O_0_base, desc_A, desc_B0, make_instr_desc_transposed(M, 64), 1);
                umma_f16_cg1_fn(tmem_O_1_base, desc_A, desc_B1, make_instr_desc_transposed(M, 64), 1);
            }
        }
        umma_commit_1sm_fn(bar_V);
        mbarrier_wait_fn(bar_V, phase_V);
        
        phase_K ^= 1;
        phase_V ^= 1;
    }
    
    tmem_load_fence_fn();
    
    if (col == 0) {
        uint32_t global_sq_idx = sq_block * 64 + threadIdx.x;
        if (global_sq_idx < S) {
            LSE[batch_head * S + global_sq_idx] = m_global + logf(l_sum);
        }
    }
    
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_O_0_base + col));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_O_1_base + col));
            
        uint32_t global_row = sq_block * 64 + threadIdx.x;
        uint32_t n_base = col;
        
        if (global_row < M_total && n_base < D) {
            __nv_bfloat16* out = static_cast<__nv_bfloat16*>(O.data_ptr());
            out[batch_head * S * D + global_row * D + n_base] = __float2bfloat16(O_0[col]);
            out[batch_head * S * D + global_row * D + n_base + 64] = __float2bfloat16(O_1[col]);
        }
        
        uint32_t global_row1 = sq_block * 64 + threadIdx.x + 64;
        uint32_t n_base1 = col;
        
        if (global_row1 < M_total && n_base1 < D) {
            __nv_bfloat16* out = static_cast<__nv_bfloat16*>(O.data_ptr());
            out[batch_head * S * D + global_row1 * D + n_base1] = __float2bfloat16(O_0_1[col]);
            out[batch_head * S * D + global_row1 * D + n_base1 + 64] = __float2bfloat16(O_1_1[col]);
        }
    }
    tmem_load_fence_fn();
}

namespace tvm_ffi {

void run(TensorView Q, TensorView K, TensorView V, TensorView O, TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t D = Q.size(3);

    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, H, B, 64, 64, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), D, S, H, B, 64, 64, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), D, S, H, B, 64, 64, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B);

    uint32_t num_sq_blocks = (S + 63) / 64;
    dim3 grid(num_sq_blocks, B * H);
    dim3 block(128);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute((const void*)attention_kernel_swizzled, cudaFuncAttributeMaxDynamicSharedMemorySize, 57344));

    attention_kernel_swizzled<<<grid, block, 57344, stream>>>(
        tma_Q, tma_K, tma_V, S, H, B, static_cast<float*>(LSE.data_ptr())
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi