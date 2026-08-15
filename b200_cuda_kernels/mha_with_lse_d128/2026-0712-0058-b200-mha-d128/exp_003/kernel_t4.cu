#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <math.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        fprintf(stderr, "CU error %d at %s:%d\n", (int)_e,       \
                __FILE__, __LINE__);                               \
        exit(1);                                                   \
    }                                                              \
} while(0)


__device__ __forceinline__ uint32_t to_bits(float f) {
    return *reinterpret_cast<uint32_t*>(&f);
}

__device__ __forceinline__ float tmem_to_float(uint32_t val) {
    return __uint_as_float(val);
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

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
       :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
       :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF;
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
}

__device__ __forceinline__ void tma_load_multicast_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, uint16_t mask) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF;
    uint64_t cache_hint = 0;
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5, %6;"
        :: "r"(sa), "l"((uint64_t)d), "r"(ba), "r"(c0), "r"(c1), "h"(mask), "l"(cache_hint) : "memory");
}

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}

template <typename T>
__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t lbo = 16; 
    uint32_t sbo = 1024;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

template <typename T>
__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t sbo = 1024;
    uint32_t lbo = (128 / 8) * sbo;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

template <typename T>
__device__ __forceinline__ uint64_t make_smem_desc_k_major_cg2(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t lbo = 16; 
    uint32_t sbo = 1024;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    d |= (uint64_t)(cluster_rank_fn() & 1) << 49;
    return d;
}

template <typename T>
__device__ __forceinline__ uint64_t make_smem_desc_mn_major_cg2(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t sbo = 1024;
    uint32_t lbo = (128 / 8) * sbo;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    d |= (uint64_t)(cluster_rank_fn() & 1) << 49;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_cg2(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (0u << 16);   
    d |= ((N >> 3) << 17);     
    d |= ((M >> 4) << 24);    
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_mn_major_b_cg2(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (0u << 15);   
    d |= (1u << 16);   
    d |= ((N >> 3) << 17);     
    d |= ((M >> 4) << 24);    
    return d;
}

__global__ __launch_bounds__(128, 2)
void fa_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE,
    uint32_t S, float scale_factor, uint32_t H, uint32_t B) 
{
    int q_block = blockIdx.x;
    int h_idx = blockIdx.y % H;
    int b_idx = blockIdx.z;
    
    uint32_t m_block = q_block * 128;
    if (m_block >= S) return;

    extern __shared__ char smem_pool_raw[];
    uintptr_t pool_addr = (uintptr_t)smem_pool_raw;
    uintptr_t aligned_addr = (pool_addr + 1023) & ~1023;
    char* smem_pool = (char*)aligned_addr;

    __nv_bfloat16* smem_Q_0 = (__nv_bfloat16*)smem_pool;                                
    __nv_bfloat16* smem_Q_1 = (__nv_bfloat16*)(smem_pool + 32768);                      
    __nv_bfloat16* smem_K_0 = (__nv_bfloat16*)(smem_pool + 65536);                      
    __nv_bfloat16* smem_K_1 = (__nv_bfloat16*)(smem_pool + 98304);                      
    __nv_bfloat16* smem_V_0 = (__nv_bfloat16*)(smem_pool + 131072);                     
    __nv_bfloat16* smem_V_1 = (__nv_bfloat16*)(smem_pool + 163840);                     
    __nv_bfloat16* smem_P_0 = (__nv_bfloat16*)(smem_pool + 163840);                       
    __nv_bfloat16* smem_P_1 = (__nv_bfloat16*)(smem_pool + 176128);                       
    
    uint32_t* p_tmem_S = (uint32_t*)(smem_pool + 196608);
    uint32_t* p_tmem_P = (uint32_t*)(smem_pool + 196616);
    uint32_t* p_tmem_O = (uint32_t*)(smem_pool + 196624);
    uint64_t* mbar_Q = (uint64_t*)(smem_pool + 196640);
    uint64_t* mbar_KV_0 = (uint64_t*)(smem_pool + 196648);
    uint64_t* mbar_KV_1 = (uint64_t*)(smem_pool + 196656);
    uint64_t* mbar_UMA_0 = (uint64_t*)(smem_pool + 196664);
    uint64_t* mbar_UMA_1 = (uint64_t*)(smem_pool + 196672);

    uint64_t* mbar_KV[2] = {mbar_KV_0, mbar_KV_1};
    __nv_bfloat16* smem_K[2] = {smem_K_0, smem_K_1};
    __nv_bfloat16* smem_V[2] = {smem_V_0, smem_V_1};

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_KV_0, 1);
        init_smem_barrier_fn(mbar_KV_1, 1);
        init_smem_barrier_fn(mbar_UMA_0, 1);
        init_smem_barrier_fn(mbar_UMA_1, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    if (threadIdx.x < 32) {
        tmem_alloc_fn(p_tmem_S, 128);
        tmem_alloc_fn(p_tmem_P, 128);
        tmem_alloc_fn(p_tmem_O, 128);
    }
    __syncthreads();
    uint32_t tmem_S = *p_tmem_S;
    uint32_t tmem_P = *p_tmem_P;
    uint32_t tmem_O = *p_tmem_O;

    uint32_t expect_tx_Q = 32768; 
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, expect_tx_Q);
        
        uint16_t cta_mask = 0x3;
        uint32_t outer_offset = b_idx * H * S + h_idx * S;
        
        tma_load_multicast_2d_cg2_fn(&tma_Q, mbar_Q, smem_Q_0, 0, outer_offset + m_block, cta_mask);
        tma_load_multicast_2d_cg2_fn(&tma_Q, mbar_Q, smem_Q_1, 64, outer_offset + m_block, cta_mask);
    }

    int num_steps = (S + 127) / 128;
    
    if (num_steps > 0) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_KV_0, expect_tx_Q);
            
            uint32_t outer_offset = b_idx * H * S + h_idx * S;
            
            tma_load_2d_cg2_fn(&tma_K, mbar_KV_0, smem_K_0, 0, outer_offset + 0 * 128);
            tma_load_2d_cg2_fn(&tma_K, mbar_KV_0, smem_K_1, 64, outer_offset + 0 * 128);
            
            tma_load_2d_cg2_fn(&tma_V, mbar_KV_0, smem_V_0, 0, outer_offset + 0 * 128);
            tma_load_2d_cg2_fn(&tma_V, mbar_KV_0, smem_V_1, 64, outer_offset + 0 * 128);
        }
    }

    mbarrier_wait_fn(mbar_Q, 0);

    uint32_t phase_KV[2] = {0};

    float my_rowmax_prev0 = -1e20f;
    float my_rowsum_prev0 = 0.0f;
    float my_rowmax_prev1 = -1e20f;
    float my_rowsum_prev1 = 0.0f;

    uint32_t idesc_QK = make_instr_desc_fn_cg2(128, 128);
    uint32_t idesc_PV = make_instr_desc_fn_mn_major_b_cg2(128, 128);

    uint32_t phase_UMA_0 = 0;
    uint32_t phase_UMA_1 = 0;

    cluster_sync_fn();

    for (int step = 0; step < num_steps; step++) {
        int kv = step % 2;
        int next_kv = (step + 1) % 2;

        if (step < num_steps - 1) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(mbar_KV[next_kv], expect_tx_Q);
                
                uint32_t outer_offset = b_idx * H * S + h_idx * S;
                uint32_t next_kv_start = (step + 1) * 128;
                
                tma_load_2d_cg2_fn(&tma_K, mbar_KV[next_kv], smem_K[next_kv], 0, outer_offset + next_kv_start);
                tma_load_2d_cg2_fn(&tma_K, mbar_KV[next_kv], smem_K[next_kv] + 8192, 64, outer_offset + next_kv_start);
                
                tma_load_2d_cg2_fn(&tma_V, mbar_KV[next_kv], smem_V[next_kv], 0, outer_offset + next_kv_start);
                tma_load_2d_cg2_fn(&tma_V, mbar_KV[next_kv], smem_V[next_kv] + 8192, 64, outer_offset + next_kv_start);
            }
        }

        mbarrier_wait_fn(mbar_KV[kv], phase_KV[kv]);
        phase_KV[kv] ^= 1;

        __nv_bfloat16* curr_smem_K_0 = (kv == 0) ? smem_K_0 : smem_K_1;
        __nv_bfloat16* curr_smem_K_1 = (kv == 0) ? smem_K_1 : smem_K_0; 
        
        __nv_bfloat16* curr_smem_V_0 = (kv == 0) ? smem_V_0 : smem_V_1;
        __nv_bfloat16* curr_smem_V_1 = (kv == 0) ? smem_V_1 : smem_V_0;

        __nv_bfloat16* curr_smem_Q_0 = smem_Q_0;
        __nv_bfloat16* curr_smem_Q_1 = smem_Q_1;
        
        __nv_bfloat16* curr_smem_P_0 = smem_P_0;
        __nv_bfloat16* curr_smem_P_1 = smem_P_1;

        __syncthreads(); 
        
        if (threadIdx.x == 0) {
            for (int k = 0; k < 4; k++) {
                uint64_t desc_Q_k = make_smem_desc_k_major<__nv_bfloat16>(curr_smem_Q_0 + k * 32);
                uint64_t desc_K_k = make_smem_desc_k_major<__nv_bfloat16>(curr_smem_K_0 + k * 32);
                umma_f16_cg2_fn(tmem_S, desc_Q_k, desc_K_k, idesc_QK, k == 0 ? 0 : 1);
            }
            for (int k = 0; k < 4; k++) {
                uint64_t desc_Q_k = make_smem_desc_k_major<__nv_bfloat16>(curr_smem_Q_1 + k * 32);
                uint64_t desc_K_k = make_smem_desc_k_major<__nv_bfloat16>(curr_smem_K_1 + k * 32);
                umma_f16_cg2_fn(tmem_S, desc_Q_k, desc_K_k, idesc_QK, 1);
            }
            umma_commit_2sm_fn(mbar_UMA_0);
        }
        mbarrier_wait_fn(mbar_UMA_0, phase_UMA_0);
        phase_UMA_0 ^= 1;

        tmem_load_fence_fn();

        uint32_t my_row0 = (threadIdx.x / 2) % 64;
        uint32_t my_row1 = my_row0 + 64;

        float my_rowmax0 = -1e20f;
        float my_rowmax1 = -1e20f;

        for (int col_chunk = 0; col_chunk < 64; col_chunk += 4) {
            uint32_t r0[4], r1[4];
            uint32_t tmem_addr0 = (my_row0 << 16) + col_chunk;
            uint32_t tmem_addr1 = (my_row1 << 16) + col_chunk;
            asm volatile("tcgen05.ld.sync.aligned." "32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0[0]), "=r"(r0[1]), "=r"(r0[2]), "=r"(r0[3]) : "r"(tmem_addr0));
            asm volatile("tcgen05.ld.sync.aligned." "32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r1[0]), "=r"(r1[1]), "=r"(r1[2]), "=r"(r1[3]) : "r"(tmem_addr1));
            
            for(int k = 0; k < 4; k++) {
                float val0 = tmem_to_float(r0[k]) * scale_factor;
                if (val0 > my_rowmax0) my_rowmax0 = val0;
                
                float val1 = tmem_to_float(r1[k]) * scale_factor;
                if (val1 > my_rowmax1) my_rowmax1 = val1;
            }
        }

        float my_rowsum0 = 0.0f;
        float my_rowsum1 = 0.0f;
        float p_val0[64];
        float p_val1[64];
        uint32_t p_val_bits0[64];
        uint32_t p_val_bits1[64];

        for (int col_chunk = 0; col_chunk < 64; col_chunk += 4) {
            uint32_t r0[4], r1[4];
            uint32_t tmem_addr0 = (my_row0 << 16) + col_chunk;
            uint32_t tmem_addr1 = (my_row1 << 16) + col_chunk;
            asm volatile("tcgen05.ld.sync.aligned." "32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0[0]), "=r"(r0[1]), "=r"(r0[2]), "=r"(r0[3]) : "r"(tmem_addr0));
            asm volatile("tcgen05.ld.sync.aligned." "32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r1[0]), "=r"(r1[1]), "=r"(r1[2]), "=r"(r1[3]) : "r"(tmem_addr1));
            
            for(int k = 0; k < 4; k++) {
                float val0 = tmem_to_float(r0[k]) * scale_factor;
                p_val0[col_chunk + k] = fast_exp2f_fn((val0 - my_rowmax0) * 1.4426950408889634f);
                my_rowsum0 += p_val0[col_chunk + k];
                p_val_bits0[col_chunk + k] = to_bits(p_val0[col_chunk + k]);
                
                float val1 = tmem_to_float(r1[k]) * scale_factor;
                p_val1[col_chunk + k] = fast_exp2f_fn((val1 - my_rowmax1) * 1.4426950408889634f);
                my_rowsum1 += p_val1[col_chunk + k];
                p_val_bits1[col_chunk + k] = to_bits(p_val1[col_chunk + k]);
            }
        }

        float corr0 = fast_exp2f_fn((my_rowmax_prev0 - my_rowmax0) * 1.4426950408889634f);
        float corr1 = fast_exp2f_fn((my_rowmax_prev1 - my_rowmax1) * 1.4426950408889634f);

        for (int col_chunk = 0; col_chunk < 64; col_chunk += 4) {
            uint32_t r0[4], r1[4];
            uint32_t tmem_addr0 = (my_row0 << 16) + col_chunk;
            uint32_t tmem_addr1 = (my_row1 << 16) + col_chunk;
            asm volatile("tcgen05.ld.sync.aligned." "32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0[0]), "=r"(r0[1]), "=r"(r0[2]), "=r"(r0[3]) : "r"(tmem_addr0));
            asm volatile("tcgen05.ld.sync.aligned." "32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r1[0]), "=r"(r1[1]), "=r"(r1[2]), "=r"(r1[3]) : "r"(tmem_addr1));
            
            for(int j = 0; j < 4; j++) {
                r0[j] = to_bits(tmem_to_float(r0[j]) * corr0);
                r1[j] = to_bits(tmem_to_float(r1[j]) * corr1);
            }
            asm volatile("tcgen05.st.sync.aligned." "32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                :: "r"(r0[0]), "r"(r0[1]), "r"(r0[2]), "r"(r0[3]) : "r"(tmem_addr0));
            asm volatile("tcgen05.st.sync.aligned." "32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                :: "r"(r1[0]), "r"(r1[1]), "r"(r1[2]), "r"(r1[3]) : "r"(tmem_addr1));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        __syncthreads();
        fence_proxy_async_fn();

        for (int k = 0; k < 64; k+=2) {
            float next_p_val = (k + 1 < 64) ? p_val0[k+1] : 0.0f;
            uint32_t x_swizzled = ((my_row0 % 8) ^ (k / 8)) * 8 + (k % 8);
            uint32_t packed = pack_bf16_fn(p_val_bits0[k], to_bits(next_p_val));
            *reinterpret_cast<uint32_t*>(&curr_smem_P_0[my_row0 * 64 + x_swizzled]) = packed;
        }
        for (int k = 0; k < 64; k+=2) {
            float next_p_val = (k + 1 < 64) ? p_val1[k+1] : 0.0f;
            uint32_t x_swizzled = ((my_row1 % 8) ^ (k / 8)) * 8 + (k % 8);
            uint32_t packed = pack_bf16_fn(p_val_bits1[k], to_bits(next_p_val));
            *reinterpret_cast<uint32_t*>(&curr_smem_P_1[my_row1 * 64 + x_swizzled]) = packed;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            for (int k = 0; k < 4; k++) {
                uint64_t desc_P_k = make_smem_desc_k_major<__nv_bfloat16>(curr_smem_P_0 + k * 32);
                uint64_t desc_V_k = make_smem_desc_mn_major<__nv_bfloat16>(curr_smem_V_0 + k * 2048); 
                umma_f16_cg2_fn(tmem_O, desc_P_k, desc_V_k, idesc_PV, k == 0 ? 0 : 1);
            }
            for (int k = 0; k < 4; k++) {
                uint64_t desc_P_k = make_smem_desc_k_major<__nv_bfloat16>(curr_smem_P_1 + k * 32);
                uint64_t desc_V_k = make_smem_desc_mn_major<__nv_bfloat16>(curr_smem_V_1 + k * 2048); 
                umma_f16_cg2_fn(tmem_O, desc_P_k, desc_V_k, idesc_PV, 1);
            }
            umma_commit_2sm_fn(mbar_UMA_1);
        }
        mbarrier_wait_fn(mbar_UMA_1, phase_UMA_1);
        phase_UMA_1 ^= 1;

        __syncthreads();

        my_rowsum_prev0 *= corr0;
        my_rowsum_prev0 += my_rowsum0;
        my_rowmax_prev0 = my_rowmax0;

        my_rowsum_prev1 *= corr1;
        my_rowsum_prev1 += my_rowsum1;
        my_rowmax_prev1 = my_rowmax1;
    }

    tmem_load_fence_fn();
    __syncthreads();

    for (int col_chunk = 0; col_chunk < 64; col_chunk += 4) {
        uint32_t r0[4], r1[4];
        uint32_t tmem_addr0 = (my_row0 << 16) + col_chunk;
        uint32_t tmem_addr1 = (my_row1 << 16) + col_chunk;
        asm volatile("tcgen05.ld.sync.aligned." "32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0[0]), "=r"(r0[1]), "=r"(r0[2]), "=r"(r0[3]) : "r"(tmem_addr0));
        asm volatile("tcgen05.ld.sync.aligned." "32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r1[0]), "=r"(r1[1]), "=r"(r1[2]), "=r"(r1[3]) : "r"(tmem_addr1));
        
        for(int j = 0; j < 4; j++) {
            float val0 = tmem_to_float(r0[j]);
            if (my_rowsum_prev0 > 0.0f) val0 /= my_rowsum_prev0;
            r0[j] = to_bits(val0);
            
            float val1 = tmem_to_float(r1[j]);
            if (my_rowsum_prev1 > 0.0f) val1 /= my_rowsum_prev1;
            r1[j] = to_bits(val1);
        }
        
        for(int j = 0; j < 4; j += 2) {
            uint32_t x_swizzled0 = ((my_row0 % 8) ^ (col_chunk / 8)) * 8 + (col_chunk % 8);
            uint32_t packed0 = pack_bf16_fn(r0[j], r0[j+1]);
            *reinterpret_cast<uint32_t*>(&curr_smem_P_0[my_row0 * 64 + x_swizzled0]) = packed0;

            uint32_t x_swizzled1 = ((my_row1 % 8) ^ (col_chunk / 8)) * 8 + (col_chunk % 8);
            uint32_t packed1 = pack_bf16_fn(r1[j], r1[j+1]);
            *reinterpret_cast<uint32_t*>(&curr_smem_P_1[my_row1 * 64 + x_swizzled1]) = packed1;
        }
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    for (uint32_t step = 0; step < 4; ++step) {
        uint32_t col_start = step * 32;
        uint32_t row0 = (step * 8 + warp_id) % 64;
        uint32_t global_row0 = m_block + row0;
        if (global_row0 < S && col_start + 15 < 64) {
            uint4 data = *reinterpret_cast<uint4*>(&curr_smem_P_0[row0 * 64 + col_start]);
            uint32_t outer_offset_flat = b_idx * H * S + h_idx * S;
            *reinterpret_cast<uint4*>(O + (uint64_t)(outer_offset_flat + global_row0) * 128 + col_start) = data;
        }
        
        uint32_t row1 = row0 + 64;
        uint32_t global_row1 = m_block + row1;
        if (global_row1 < S && col_start + 15 < 64) {
            uint4 data = *reinterpret_cast<uint4*>(&curr_smem_P_1[row1 * 64 + col_start]);
            uint32_t outer_offset_flat = b_idx * H * S + h_idx * S;
            *reinterpret_cast<uint4*>(O + (uint64_t)(outer_offset_flat + global_row1) * 128 + col_start) = data;
        }
    }

    float global_rowmax = (cluster_rank_fn() == 0) ? my_rowmax_prev0 : my_rowmax_prev1;
    float global_rowsum = (cluster_rank_fn() == 0) ? my_rowsum_prev0 : my_rowsum_prev1;
    uint32_t global_my_row = (cluster_rank_fn() == 0) ? my_row0 : my_row1;

    if (lane_id == 0) {
        uint32_t global_row = m_block + global_my_row;
        if (global_row < S) {
            uint32_t outer_offset_flat = b_idx * H * S + h_idx * S;
            LSE[outer_offset_flat + global_row] = global_rowmax + __logf(global_rowsum);
        }
    }
    
    __syncthreads();

    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_S, 128);
        tmem_dealloc_fn(tmem_P, 128);
        tmem_dealloc_fn(tmem_O, 128);
    }
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 2, globalAddress, globalDim, globalStrides, boxDim,
        elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

namespace tvm_ffi {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 
    
    if (D != 128) {
        fprintf(stderr, "Error: Expected head dimension 128, got %ld\n", D);
        exit(1);
    }

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, B * H * S, 64, 128, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, B * H * S, 64, 128, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, B * H * S, 64, 128, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    dim3 grid((S + 127) / 128, H, B);
    dim3 block(128, 1, 1);
    int smem_size = 192 * 1024 + 256;

    CUDA_CHECK(cudaFuncSetAttribute(fa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    float scale_factor = 1.0f / sqrtf((float)D);
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, fa_kernel,
        tma_Q, tma_K, tma_V,
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()),
        S, scale_factor, H, B
    ));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi