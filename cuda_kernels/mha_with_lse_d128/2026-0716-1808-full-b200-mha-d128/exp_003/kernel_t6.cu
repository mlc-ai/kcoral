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

__device__ __forceinline__ void tma_load_3d_cta(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_3d_cta(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void commit_cp_async_bulk(uint64_t* bar) {
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
    d |= (1u << 4);      
    d |= (1u << 7);      
    d |= (1u << 10);     
    d |= (0u << 15);     
    d |= ((trans_b & 1) << 16); 
    d |= ((N / 8) << 17);      
    d |= ((M / 16) << 24);     
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

__global__ void __launch_bounds__(64) attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* LSE, int S)
{
    __shared__ __align__(4) uint32_t tmem_pool[2];
    
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_Q   = (__nv_bfloat16*)(smem_pool + 0);
    __nv_bfloat16* smem_K_0 = (__nv_bfloat16*)(smem_pool + 32768);
    __nv_bfloat16* smem_K_1 = (__nv_bfloat16*)(smem_pool + 65536);
    __nv_bfloat16* smem_V_0 = (__nv_bfloat16*)(smem_pool + 98304);
    __nv_bfloat16* smem_V_1 = (__nv_bfloat16*)(smem_pool + 131072);
    __nv_bfloat16* smem_P   = (__nv_bfloat16*)(smem_pool + 163840);
    __nv_bfloat16* smem_out = (__nv_bfloat16*)(smem_pool + 180224);
    
    uint64_t* bar = (uint64_t*)(smem_pool + 212992);
    float* m_val = (float*)(smem_pool + 213056);
    float* l_val = (float*)(smem_pool + 213312);
    float* scale_ptr = (float*)(smem_pool + 213568);
    
    if (threadIdx.x < 32) {
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
    }
    fence_smem_barrier_init_fn();
    
    if (threadIdx.x < 64) {
        m_val[threadIdx.x] = -1e20f;
        l_val[threadIdx.x] = 0.0f;
    }
    if (threadIdx.x == 0) {
        *scale_ptr = 1.0f / sqrtf(128.0f);
    }
    __syncthreads();
    
    if (threadIdx.x < 64) {
        for (int i = 0; i < 128; i += 4) {
            uint32_t tmem_o_addr = tmem_O_base + ((m_start + threadIdx.x) << 16) + i;
            *(float*)(tmem_addr + tmem_o_addr) = 0.0f;
        }
    }
    
    uint32_t idesc_QK = make_instr_desc(128, 64, 0);
    uint32_t idesc_PV = make_instr_desc(128, 64, 1);
    
    uint64_t desc_Q0 = make_smem_desc_k_major_128B(smem_Q);
    uint64_t desc_Q1 = make_smem_desc_k_major_128B(smem_Q + 8192); // Offset by 16384 bytes
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar, 32768);
        tma_load_3d_cta(&tma_Q, bar, smem_Q, 0, m_start, bh);
        tma_load_3d_cta(&tma_Q, bar, smem_Q + 8192, 64, m_start, bh);
    }
    mbarrier_wait_fn(bar, 0);
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar + 1, 32768);
        tma_load_3d_cta(&tma_K, bar + 1, smem_K_0, 0, 0, bh);
        tma_load_3d_cta(&tma_K, bar + 1, smem_K_0 + 4096, 64, 0, bh);
        tma_load_3d_cta(&tma_V, bar + 1, smem_V_0, 0, 0, bh);
        tma_load_3d_cta(&tma_V, bar + 1, smem_V_0 + 4096, 64, 0, bh);
    }
    
    auto swizzle_128B = [](int r, int c) {
        int chunk = c / 8;
        int swizzled_chunk = chunk ^ (r & 7);
        return swizzled_chunk * 8 + (c & 7);
    };
    
    int phase = 0;
    for (int chunk = 0; chunk < S; chunk += 64) {
        mbarrier_wait_fn(bar + 1, phase);
        
        if (chunk + 64 < S) {
            int next_phase = phase ^ 1;
            __nv_bfloat16* next_K = (next_phase == 0) ? smem_K_0 : smem_K_1;
            __nv_bfloat16* next_V = (next_phase == 0) ? smem_V_0 : smem_V_1;
            
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(bar + 1, 32768);
                tma_load_3d_cta(&tma_K, bar + 1, next_K, 0, chunk + 64, bh);
                tma_load_3d_cta(&tma_K, bar + 1, next_K + 4096, 64, chunk + 64, bh);
                tma_load_3d_cta(&tma_V, bar + 1, next_V, 0, chunk + 64, bh);
                tma_load_3d_cta(&tma_V, bar + 1, next_V + 4096, 64, chunk + 64, bh);
            }
        }
        
        __nv_bfloat16* cur_K = (phase == 0) ? smem_K_0 : smem_K_1;
        __nv_bfloat16* cur_V0 = (phase == 0) ? smem_V_0 : smem_V_1;
        __nv_bfloat16* cur_V1 = (phase == 0) ? smem_V_1 : smem_V_0;
        
        uint32_t tmem_s_addr = tmem_S_base;
        tcgen05_fence_before_fn();
        
        if (threadIdx.x == 0) {
            uint64_t cK0 = make_smem_desc_k_major_128B(cur_K);
            uint64_t cK1 = make_smem_desc_k_major_128B(cur_K + 4096);
            
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_s_addr, desc_Q0 + k * 2, cK0 + k * 2, idesc_QK, k==0 ? 0 : 1);
            }
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_s_addr, desc_Q1 + k * 2, cK1 + k * 2, idesc_QK, 1);
            }
        }
        tcgen05_fence_after_fn();
        tmem_load_fence_fn();
        
        int row_idx = threadIdx.x;
        float my_max[2] = {-1e20f, -1e20f};
        
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0[2], r1[2], r2[2], r3[2];
            tmem_load_4x_fn(tmem_S_base + ((m_start + row_idx) << 16) + col, &r0[0], &r1[0], &r2[0], &r3[0]);
            tmem_load_4x_fn(tmem_S_base + ((m_start + row_idx + 64) << 16) + col, &r0[1], &r1[1], &r2[1], &r3[1]);
            
            float f0_0 = __uint_as_float(r0[0]) * *scale_ptr;
            float f1_0 = __uint_as_float(r1[0]) * *scale_ptr;
            float f2_0 = __uint_as_float(r2[0]) * *scale_ptr;
            float f3_0 = __uint_as_float(r3[0]) * *scale_ptr;
            
            float f0_1 = __uint_as_float(r0[1]) * *scale_ptr;
            float f1_1 = __uint_as_float(r1[1]) * *scale_ptr;
            float f2_1 = __uint_as_float(r2[1]) * *scale_ptr;
            float f3_1 = __uint_as_float(r3[1]) * *scale_ptr;
            
            if (chunk + col >= S) f0_0 = -1e20f;
            if (chunk + col + 1 >= S) f1_0 = -1e20f;
            if (chunk + col + 2 >= S) f2_0 = -1e20f;
            if (chunk + col + 3 >= S) f3_0 = -1e20f;
            
            if (chunk + col >= S) f0_1 = -1e20f;
            if (chunk + col + 1 >= S) f1_1 = -1e20f;
            if (chunk + col + 2 >= S) f2_1 = -1e20f;
            if (chunk + col + 3 >= S) f3_1 = -1e20f;
            
            my_max[0] = fmaxf(fmaxf(fmaxf(fmaxf(my_max[0], f0_0), f1_0), f2_0), f3_0);
            my_max[1] = fmaxf(fmaxf(fmaxf(fmaxf(my_max[1], f0_1), f1_1), f2_1), f3_1);
        }
        
        float new_m[2];
        bool needs_rescale[2] = {false, false};
        new_m[0] = fmaxf(m_val[row_idx], my_max[0]);
        new_m[1] = fmaxf(m_val[row_idx + 64], my_max[1]);
        if (new_m[0] != m_val[row_idx]) needs_rescale[0] = true;
        if (new_m[1] != m_val[row_idx + 64]) needs_rescale[1] = true;
        
        float my_sum[2] = {0.0f, 0.0f};
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0[2], r1[2], r2[2], r3[2];
            tmem_load_4x_fn(tmem_S_base + ((m_start + row_idx) << 16) + col, &r0[0], &r1[0], &r2[0], &r3[0]);
            tmem_load_4x_fn(tmem_S_base + ((m_start + row_idx + 64) << 16) + col, &r0[1], &r1[1], &r2[1], &r3[1]);
            
            float f0_0 = __uint_as_float(r0[0]) * *scale_ptr;
            float f1_0 = __uint_as_float(r1[0]) * *scale_ptr;
            float f2_0 = __uint_as_float(r2[0]) * *scale_ptr;
            float f3_0 = __uint_as_float(r3[0]) * *scale_ptr;
            
            float f0_1 = __uint_as_float(r0[1]) * *scale_ptr;
            float f1_1 = __uint_as_float(r1[1]) * *scale_ptr;
            float f2_1 = __uint_as_float(r2[1]) * *scale_ptr;
            float f3_1 = __uint_as_float(r3[1]) * *scale_ptr;
            
            if (chunk + col >= S) f0_0 = -1e20f;
            if (chunk + col + 1 >= S) f1_0 = -1e20f;
            if (chunk + col + 2 >= S) f2_0 = -1e20f;
            if (chunk + col + 3 >= S) f3_0 = -1e20f;
            
            if (chunk + col >= S) f0_1 = -1e20f;
            if (chunk + col + 1 >= S) f1_1 = -1e20f;
            if (chunk + col + 2 >= S) f2_1 = -1e20f;
            if (chunk + col + 3 >= S) f3_1 = -1e20f;
            
            float p0_0 = expf(f0_0 - new_m[0]);
            float p1_0 = expf(f1_0 - new_m[0]);
            float p2_0 = expf(f2_0 - new_m[0]);
            float p3_0 = expf(f3_0 - new_m[0]);
            my_sum[0] += p0_0 + p1_0 + p2_0 + p3_0;
            
            float p0_1 = expf(f0_1 - new_m[1]);
            float p1_1 = expf(f1_1 - new_m[1]);
            float p2_1 = expf(f2_1 - new_m[1]);
            float p3_1 = expf(f3_1 - new_m[1]);
            my_sum[1] += p0_1 + p1_1 + p2_1 + p3_1;
            
            if (needs_rescale[0]) {
                float sf = expf(m_val[row_idx] - new_m[0]);
                p0_0 *= sf; p1_0 *= sf; p2_0 *= sf; p3_0 *= sf;
            }
            if (needs_rescale[1]) {
                float sf = expf(m_val[row_idx + 64] - new_m[1]);
                p0_1 *= sf; p1_1 *= sf; p2_1 *= sf; p3_1 *= sf;
            }
            
            __nv_bfloat16 bf_p0_0 = __float2bfloat16(p0_0);
            __nv_bfloat16 bf_p1_0 = __float2bfloat16(p1_0);
            __nv_bfloat16 bf_p2_0 = __float2bfloat16(p2_0);
            __nv_bfloat16 bf_p3_0 = __float2bfloat16(p3_0);
            
            __nv_bfloat16 bf_p0_1 = __float2bfloat16(p0_1);
            __nv_bfloat16 bf_p1_1 = __float2bfloat16(p1_1);
            __nv_bfloat16 bf_p2_1 = __float2bfloat16(p2_1);
            __nv_bfloat16 bf_p3_1 = __float2bfloat16(p3_1);
            
            uint32_t p0_0 = ((uint32_t)*reinterpret_cast<uint16_t*>(&bf_p1_0) << 16) | *reinterpret_cast<uint16_t*>(&bf_p0_0);
            uint32_t p1_0 = ((uint32_t)*reinterpret_cast<uint16_t*>(&bf_p3_0) << 16) | *reinterpret_cast<uint16_t*>(&bf_p2_0);
            
            uint32_t p0_1 = ((uint32_t)*reinterpret_cast<uint16_t*>(&bf_p1_1) << 16) | *reinterpret_cast<uint16_t*>(&bf_p0_1);
            uint32_t p1_1 = ((uint32_t)*reinterpret_cast<uint16_t*>(&bf_p3_1) << 16) | *reinterpret_cast<uint16_t*>(&bf_p2_1);
            
            int sc0_0 = swizzle_128B(row_idx, col);
            int sc1_0 = swizzle_128B(row_idx, col + 2);
            
            int sc0_1 = swizzle_128B(row_idx + 64, col);
            int sc1_1 = swizzle_128B(row_idx + 64, col + 2);
            
            *(uint32_t*)&smem_P[row_idx * 64 + sc0_0] = p0_0;
            *(uint32_t*)&smem_P[row_idx * 64 + sc1_0] = p1_0;
            
            *(uint32_t*)&smem_P[(row_idx + 64) * 64 + sc0_1] = p0_1;
            *(uint32_t*)&smem_P[(row_idx + 64) * 64 + sc1_1] = p1_1;
        }
        
        if (needs_rescale[0]) {
            float sf = expf(m_val[row_idx] - new_m[0]);
            for (int i = 0; i < 128; i += 4) {
                uint32_t tmem_o_addr = tmem_O_base + ((m_start + row_idx) << 16) + i;
                float* o_ptr = (float*)(tmem_addr + tmem_o_addr);
                *o_ptr *= sf;
            }
            l_val[row_idx] *= sf;
            m_val[row_idx] = new_m[0];
        }
        if (needs_rescale[1]) {
            float sf = expf(m_val[row_idx + 64] - new_m[1]);
            for (int i = 0; i < 128; i += 4) {
                uint32_t tmem_o_addr = tmem_O_base + ((m_start + row_idx + 64) << 16) + i;
                float* o_ptr = (float*)(tmem_addr + tmem_o_addr);
                *o_ptr *= sf;
            }
            l_val[row_idx + 64] *= sf;
            m_val[row_idx + 64] = new_m[1];
        }
        l_val[row_idx] += my_sum[0];
        l_val[row_idx + 64] += my_sum[1];
        
        named_barrier_sync_fn(1, 64); 
        
        uint32_t tmem_o_addr = tmem_O_base;
        tcgen05_fence_before_fn();
        if (threadIdx.x == 0) {
            uint64_t desc_P0 = make_smem_desc_k_major_128B(smem_P);
            uint64_t cV0 = make_smem_desc_mn_major_128B(cur_V0);
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_o_addr, desc_P0 + k * 2, cV0 + k * 128, idesc_PV, 1);
            }
            
            uint64_t cV1 = make_smem_desc_mn_major_128B(cur_V1);
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_o_addr + 64, desc_P0 + k * 2, cV1 + k * 128, idesc_PV, 1);
            }
        }
        tcgen05_fence_after_fn();
        
        phase ^= 1;
    }
    
    if (threadIdx.x < 64) {
        int row_idx = threadIdx.x;
        for (int i = 0; i < 128; i += 4) {
            uint32_t tmem_o_addr = tmem_O_base + ((m_start + row_idx) << 16) + i;
            float* o_ptr = (float*)(tmem_addr + tmem_o_addr);
            *o_ptr /= l_val[row_idx];
            
            uint32_t tmem_o_addr_1 = tmem_O_base + ((m_start + row_idx + 64) << 16) + i;
            float* o_ptr_1 = (float*)(tmem_addr + tmem_o_addr_1);
            *o_ptr_1 /= l_val[row_idx + 64];
        }
    }
    __syncthreads(); 
    
    if (threadIdx.x < 64) {
        int row_idx = threadIdx.x;
        for (int c = 0; c < 128; c += 4) {
            uint32_t tmem_o_addr = tmem_O_base + ((m_start + row_idx) << 16) + c;
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
            
            *(uint32_t*)&smem_out[row_idx * 128 + sc0] = val0;
            *(uint32_t*)&smem_out[row_idx * 128 + sc1] = val1;
        }
        
        for (int c = 0; c < 128; c += 4) {
            uint32_t tmem_o_addr = tmem_O_base + ((m_start + row_idx + 64) << 16) + c;
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
            
            int sc0 = swizzle_128B(row_idx + 64, c);
            int sc1 = swizzle_128B(row_idx + 64, c + 2);
            
            *(uint32_t*)&smem_out[(row_idx + 64) * 128 + sc0] = val0;
            *(uint32_t*)&smem_out[(row_idx + 64) * 128 + sc1] = val1;
        }
    }
    __syncthreads(); 
    
    if (threadIdx.x == 0) {
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        tma_store_3d_cta(&tma_O, smem_out, 0, m_start, bh);
        tma_store_3d_cta(&tma_O, smem_out + 8192, 64, m_start, bh);
        commit_cp_async_bulk(bar + 2);
        tma_store_wait<0>();
    }
    
    if (threadIdx.x < 64) {
        if (m_start + threadIdx.x < S) {
            LSE[bh * S + m_start + threadIdx.x] = m_val[threadIdx.x] + logf(l_val[threadIdx.x]);
        }
        if (m_start + threadIdx.x + 64 < S) {
            LSE[bh * S + m_start + threadIdx.x + 64] = m_val[threadIdx.x + 64] + logf(l_val[threadIdx.x + 64]);
        }
    }
    
    if (threadIdx.x == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;\n" :: "r"(tmem_addr));
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_dim0, uint64_t gmem_dim1, uint64_t gmem_dim2, uint32_t box0, uint32_t box1, uint32_t box2, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {gmem_dim0, gmem_dim1, gmem_dim2};
    cuuint64_t globalStrides[2] = {gmem_dim0 * 2, gmem_dim1 * gmem_dim0 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3, 
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
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
    if (create_tma_3d_descriptor_2B(&tma_Q, (void*)Q_ptr, 128, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B) != 0) exit(1);
    if (create_tma_3d_descriptor_2B(&tma_K, (void*)K_ptr, 128, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B) != 0) exit(1);
    if (create_tma_3d_descriptor_2B(&tma_V, (void*)V_ptr, 128, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B) != 0) exit(1);
    if (create_tma_3d_descriptor_2B(&tma_O, (void*)O_ptr, 128, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B) != 0) exit(1);
    
    dim3 grid(B * H, (S + 127) / 128, 1);
    dim3 block(64);
    int smem_size = 213584; // ~170 KB
    
    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    attention_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, tma_O, LSE_ptr, S);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda