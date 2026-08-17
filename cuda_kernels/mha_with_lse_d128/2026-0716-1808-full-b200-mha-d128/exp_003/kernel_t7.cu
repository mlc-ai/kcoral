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

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ void tma_load_2d(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_2d(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1) : "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void commit_cp_async_bulk(uint64_t* bar) {
    (void)bar;
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
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

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major_128B(void* smem_ptr) {
    return make_smem_desc(smem_ptr, 1, 1024);
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major_128B(void* smem_ptr) {
    return make_smem_desc(smem_ptr, 8192, 1024);
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N, int trans_b) {
    uint32_t d = 0;
    d |= (1u << 4);      // c_format = FP32
    d |= (1u << 7);      // a_format = BF16
    d |= (1u << 10);     // b_format = BF16
    d |= (0u << 15);     // a_major = 0 (K-Major)
    d |= ((trans_b & 1) << 16); // b_major (Transpose B Matrix)
    d |= ((N / 8) << 17);      // n_dim
    d |= ((M / 16) << 24);     // m_dim
    return d;
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

__global__ void __launch_bounds__(128) attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* LSE, int S)
{
    __shared__ __align__(4) uint32_t tmem_pool[2];
    
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)(smem_pool + 0);
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)(smem_pool + 16384);
    __nv_bfloat16* smem_K_0 = (__nv_bfloat16*)(smem_pool + 32768);
    __nv_bfloat16* smem_K_1 = (__nv_bfloat16*)(smem_pool + 49152);
    __nv_bfloat16* smem_V_0 = (__nv_bfloat16*)(smem_pool + 65536);
    __nv_bfloat16* smem_V_1 = (__nv_bfloat16*)(smem_pool + 81920);
    __nv_bfloat16* smem_P   = (__nv_bfloat16*)(smem_pool + 98304);
    __nv_bfloat16* smem_out = (__nv_bfloat16*)(smem_pool + 114688);
    
    uint64_t* bar = (uint64_t*)(smem_pool + 147456);
    float* m_val = (float*)(smem_pool + 147472);
    float* l_val = (float*)(smem_pool + 147984);
    float* scale_ptr = (float*)(smem_pool + 148496);
    
    if (threadIdx.x == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;\n"
            :: "r"((uint32_t)__cvta_generic_to_shared(&tmem_pool[0])));
    }
    __syncthreads();
    
    uint32_t tmem_addr = tmem_pool[0];
    uint32_t tmem_S_base = tmem_addr;
    uint32_t tmem_O_base = tmem_addr + 128; 
    
    int bh = blockIdx.x;
    int m_start = blockIdx.y * 128;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar, 1);
        init_smem_barrier_fn(bar + 1, 1);
        init_smem_barrier_fn(bar + 2, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    if (threadIdx.x < 128) {
        m_val[threadIdx.x] = -1e20f;
        l_val[threadIdx.x] = 0.0f;
    }
    if (threadIdx.x == 0) {
        *scale_ptr = 1.0f / sqrtf(128.0f);
    }
    
    if (threadIdx.x < 128) {
        for (int i = 0; i < 128; i += 4) {
            uint32_t tmem_o_addr = tmem_O_base + (threadIdx.x << 16) + i;
            *(float*)(tmem_addr + tmem_o_addr) = 0.0f;
        }
    }
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar, 32768);
        tma_load_2d(&tma_Q, bar, smem_Q0, 0, bh * S + m_start);
        tma_load_2d(&tma_Q, bar, smem_Q1, 64, bh * S + m_start);
    }
    mbarrier_wait_fn(bar, 0);
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar + 1, 65536);
        tma_load_2d(&tma_K, bar + 1, smem_K_0, 0, bh * S + 0);
        tma_load_2d(&tma_K, bar + 1, smem_K_0 + 4096, 64, bh * S + 0);
        tma_load_2d(&tma_V, bar + 1, smem_V_0, 0, bh * S + 0);
        tma_load_2d(&tma_V, bar + 1, smem_V_0 + 4096, 64, bh * S + 0);
    }
    
    uint32_t idesc_QK = make_instr_desc(128, 64, 1);
    uint32_t idesc_PV = make_instr_desc(128, 64, 1);
    
    uint64_t desc_Q0 = make_smem_desc_k_major_128B(smem_Q0);
    uint64_t desc_Q1 = make_smem_desc_k_major_128B(smem_Q1);
    uint64_t desc_P0 = make_smem_desc_k_major_128B(smem_P);
    
    auto swizzle_128B = [](int r, int c) {
        int chunk = c / 8;
        int swizzled_chunk = chunk ^ (r & 7);
        return swizzled_chunk * 8 + (c & 7);
    };
    
    int phase = 0;
    for (int chunk = 0; chunk < S; chunk += 64) {
        mbarrier_wait_fn(bar + 1, phase);
        
        if (chunk + 64 < S) {
            __nv_bfloat16* next_K = (phase == 0) ? smem_K_1 : smem_K_0;
            __nv_bfloat16* next_V = (phase == 0) ? smem_V_1 : smem_V_0;
            
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(bar + 1, 65536);
                tma_load_2d(&tma_K, bar + 1, next_K, 0, bh * S + chunk + 64);
                tma_load_2d(&tma_K, bar + 1, next_K + 4096, 64, bh * S + chunk + 64);
                tma_load_2d(&tma_V, bar + 1, next_V, 0, bh * S + chunk + 64);
                tma_load_2d(&tma_V, bar + 1, next_V + 4096, 64, bh * S + chunk + 64);
            }
        }
        
        __nv_bfloat16* cur_K = (phase == 0) ? smem_K_0 : smem_K_1;
        __nv_bfloat16* cur_V0 = (phase == 0) ? smem_V_0 : smem_V_1;
        __nv_bfloat16* cur_V1 = (phase == 0) ? smem_V_1 : smem_V_0;
        
        uint64_t cK0 = make_smem_desc_k_major_128B(cur_K);
        
        uint32_t tmem_s_addr = tmem_S_base;
        tcgen05_fence_before_fn();
        
        if (threadIdx.x == 0) {
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_s_addr, desc_Q0 + k * 2, cK0 + k * 2, idesc_QK, k==0 ? 0 : 1);
            }
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_s_addr, desc_Q1 + k * 2, cK0 + k * 2 + 4, idesc_QK, 1);
            }
        }
        tcgen05_fence_after_fn();
        tmem_load_fence_fn();
        
        int row_idx = threadIdx.x;
        float my_max = -1e20f;
        
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S_base + (row_idx << 16) + col, &r0, &r1, &r2, &r3);
            float f0 = __uint_as_float(r0) * *scale_ptr;
            float f1 = __uint_as_float(r1) * *scale_ptr;
            float f2 = __uint_as_float(r2) * *scale_ptr;
            float f3 = __uint_as_float(r3) * *scale_ptr;
            
            if (chunk + col >= S) f0 = -1e20f;
            if (chunk + col + 1 >= S) f1 = -1e20f;
            if (chunk + col + 2 >= S) f2 = -1e20f;
            if (chunk + col + 3 >= S) f3 = -1e20f;
            
            my_max = fmaxf(fmaxf(fmaxf(fmaxf(my_max, f0), f1), f2), f3);
        }
        
        float new_m = fmaxf(m_val[row_idx], my_max);
        bool needs_rescale = (new_m != m_val[row_idx]);
        
        float my_sum = 0.0f;
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S_base + (row_idx << 16) + col, &r0, &r1, &r2, &r3);
            float f0 = __uint_as_float(r0) * *scale_ptr;
            float f1 = __uint_as_float(r1) * *scale_ptr;
            float f2 = __uint_as_float(r2) * *scale_ptr;
            float f3 = __uint_as_float(r3) * *scale_ptr;
            
            if (chunk + col >= S) f0 = -1e20f;
            if (chunk + col + 1 >= S) f1 = -1e20f;
            if (chunk + col + 2 >= S) f2 = -1e20f;
            if (chunk + col + 3 >= S) f3 = -1e20f;
            
            float p0 = expf(f0 - new_m);
            float p1 = expf(f1 - new_m);
            float p2 = expf(f2 - new_m);
            float p3 = expf(f3 - new_m);
            my_sum += p0 + p1 + p2 + p3;
            
            if (needs_rescale) {
                float sf = expf(m_val[row_idx] - new_m);
                p0 *= sf;
                p1 *= sf;
                p2 *= sf;
                p3 *= sf;
            }
            
            __nv_bfloat16 bf_p0 = __float2bfloat16(p0);
            __nv_bfloat16 bf_p1 = __float2bfloat16(p1);
            __nv_bfloat16 bf_p2 = __float2bfloat16(p2);
            __nv_bfloat16 bf_p3 = __float2bfloat16(p3);
            
            uint32_t packed0 = ((uint32_t)*reinterpret_cast<uint16_t*>(&bf_p1) << 16) | *reinterpret_cast<uint16_t*>(&bf_p0);
            uint32_t packed1 = ((uint32_t)*reinterpret_cast<uint16_t*>(&bf_p3) << 16) | *reinterpret_cast<uint16_t*>(&bf_p2);
            
            int sc0 = swizzle_128B(row_idx, col);
            int sc1 = swizzle_128B(row_idx, col + 2);
            
            *(uint32_t*)&smem_P[row_idx * 64 + sc0] = packed0;
            *(uint32_t*)&smem_P[row_idx * 64 + sc1] = packed1;
        }
        
        if (needs_rescale) {
            float sf = expf(m_val[row_idx] - new_m);
            for (int i = 0; i < 128; i += 4) {
                uint32_t tmem_o_addr = tmem_O_base + (row_idx << 16) + i;
                float* o_ptr = (float*)(tmem_addr + tmem_o_addr);
                *o_ptr *= sf;
            }
            l_val[row_idx] *= sf;
            m_val[row_idx] = new_m;
        }
        l_val[row_idx] += my_sum;
        
        named_barrier_sync_fn(1, 128); 
        
        uint64_t desc_V0 = make_smem_desc_mn_major_128B(cur_V0);
        uint64_t desc_V1 = make_smem_desc_mn_major_128B(cur_V1);
        
        tcgen05_fence_before_fn();
        if (threadIdx.x == 0) {
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_O_base, desc_P0 + k * 2, desc_V0 + k * 128, idesc_PV, 1);
            }
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_O_base + 64, desc_P0 + k * 2, desc_V1 + k * 128, idesc_PV, 1);
            }
        }
        tcgen05_fence_after_fn();
        
        phase ^= 1;
    }
    
    if (threadIdx.x < 128) {
        int row_idx = threadIdx.x;
        for (int i = 0; i < 128; i += 4) {
            uint32_t tmem_o_addr = tmem_O_base + (row_idx << 16) + i;
            float* o_ptr = (float*)(tmem_addr + tmem_o_addr);
            *o_ptr /= l_val[row_idx];
        }
    }
    __syncthreads();
    
    if (threadIdx.x < 128) {
        int row_idx = threadIdx.x;
        for (int c = 0; c < 128; c += 4) {
            uint32_t tmem_o_addr = tmem_O_base + (row_idx << 16) + c;
            float f0 = *(float*)(tmem_addr + tmem_o_addr);
            float f1 = *(float*)(tmem_addr + tmem_o_addr + 1);
            float f2 = *(float*)(tmem_addr + tmem_o_addr + 2);
            float f3 = *(float*)(tmem_addr + tmem_o_addr + 3);
            
            __nv_bfloat16 bf_f0 = __float2bfloat16(f0);
            __nv_bfloat16 bf_f1 = __float2bfloat16(f1);
            __nv_bfloat16 bf_f2 = __float2bfloat16(f2);
            __nv_bfloat16 bf_f3 = __float2bfloat16(f3);
            
            uint32_t val0 = ((uint32_t)*reinterpret_cast<uint16_t*>(&bf_f1) << 16) | *reinterpret_cast<uint16_t*>(&bf_f0);
            uint32_t val1 = ((uint32_t)*reinterpret_cast<uint16_t*>(&bf_f3) << 16) | *reinterpret_cast<uint16_t*>(&bf_f2);
            
            int sc0 = swizzle_128B(row_idx, c);
            int sc1 = swizzle_128B(row_idx, c + 2);
            
            *(uint32_t*)&smem_out[row_idx * 64 + sc0] = val0;
            *(uint32_t*)&smem_out[row_idx * 64 + sc1] = val1;
            
            *(uint32_t*)&smem_out[8192 + row_idx * 64 + sc0] = val0;
            *(uint32_t*)&smem_out[8192 + row_idx * 64 + sc1] = val1;
        }
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        tma_store_2d(&tma_O, smem_out, 0, bh * S + m_start);
        tma_store_2d(&tma_O, smem_out + 8192, 64, bh * S + m_start);
        commit_cp_async_bulk(bar + 2);
        tma_store_wait<0>();
    }
    
    if (threadIdx.x < 128) {
        if (m_start + threadIdx.x < S) {
            LSE[bh * S + m_start + threadIdx.x] = m_val[threadIdx.x] + logf(l_val[threadIdx.x]);
        }
    }
    
    if (threadIdx.x == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;\n" :: "r"(tmem_addr));
    }
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

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    if (create_tma_2d_descriptor_2B(&tma_Q, (void*)Q_ptr, 128, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != 0) exit(1);
    if (create_tma_2d_descriptor_2B(&tma_K, (void*)K_ptr, 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != 0) exit(1);
    if (create_tma_2d_descriptor_2B(&tma_V, (void*)V_ptr, 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != 0) exit(1);
    if (create_tma_2d_descriptor_2B(&tma_O, (void*)O_ptr, 128, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != 0) exit(1);
    
    dim3 grid(B * H, (S + 127) / 128, 1);
    dim3 block(128);
    int smem_size = 155648; // 152 KB
    
    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    attention_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, tma_O, LSE_ptr, S);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda