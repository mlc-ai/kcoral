#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CU_CHECK(call) do { \
    CUresult _e = (call); \
    if (_e != CUDA_SUCCESS) { \
        fprintf(stderr, "CUDA Driver error %d at %s:%d\n", (int)_e, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r;
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n.reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n}\n" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_load_multicast_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, uint16_t mask) {
    uint64_t cache_hint = 0;
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%4, %5}], [%2], %3, %6;"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "h"(mask), "r"(c0), "r"(c1), "l"(cache_hint) : "memory");
}

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];" :: "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(smem)), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_cp_64x128b(uint32_t tmem, void* smem) {
    uint32_t smem_ptr = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("tcgen05.cp.cta_group::1.64x128b [%0], [%1];" :: "r"(tmem), "r"(smem_ptr));
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3), "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_store_bf16_packed_row_fn(uint32_t tmem_addr, uint32_t lane_id, void* smem_dst) {
    uint32_t r0, r1, r2, r3;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_addr));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    
    float f0 = __uint_as_float(r0);
    float f1 = __uint_as_float(r1);
    float f2 = __uint_as_float(r2);
    float f3 = __uint_as_float(r3);
    
    __nv_bfloat16 b0 = __float2bfloat16(f0);
    __nv_bfloat16 b1 = __float2bfloat16(f1);
    __nv_bfloat16 b2 = __float2bfloat16(f2);
    __nv_bfloat16 b3 = __float2bfloat16(f3);
    
    uint32_t p0, p1;
    asm("mov.b32 %0, {%1, %2};" : "=r"(p0) : "h"(*reinterpret_cast<uint16_t*>(&b0)), "h"(*reinterpret_cast<uint16_t*>(&b1)));
    asm("mov.b32 %1, {%2, %3};" : "=r"(p1) : "h"(*reinterpret_cast<uint16_t*>(&b2)), "h"(*reinterpret_cast<uint16_t*>(&b3)));
    
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_dst) + lane_id * 8;
    asm volatile("st.shared.v2.b32 [%0], {%1, %2};" :: "r"(smem_addr), "r"(p0), "r"(p1) : "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (a_major << 15);   
    d |= (b_major << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint64_t smem_desc_Q0_fn(void* smem_Q) {
    uint32_t sbo = 1024;
    uint32_t lbo = 16;
    return make_smem_desc_sm100_fn(smem_Q, lbo, sbo);
}
__device__ __forceinline__ uint64_t smem_desc_Q1_fn(void* smem_Q) {
    uint32_t sbo = 1024;
    uint32_t lbo = 16;
    return make_smem_desc_sm100_fn((char*)smem_Q + 8192, lbo, sbo);
}

__device__ __forceinline__ uint64_t smem_desc_K0_fn(void* smem_K) {
    uint32_t sbo = 1024;
    uint32_t lbo = 16;
    return make_smem_desc_sm100_fn(smem_K, lbo, sbo);
}
__device__ __forceinline__ uint64_t smem_desc_K1_fn(void* smem_K) {
    uint32_t sbo = 1024;
    uint32_t lbo = 16;
    return make_smem_desc_sm100_fn((char*)smem_K + 8192, lbo, sbo);
}

__device__ __forceinline__ uint64_t smem_desc_V0_fn(void* smem_V0, int offset) {
    uint32_t sbo = 1024;
    uint32_t lbo = (64 / 8) * sbo;
    return make_smem_desc_sm100_fn((char*)smem_V0 + offset, lbo, sbo);
}
__device__ __forceinline__ uint64_t smem_desc_V1_fn(void* smem_V1, int offset) {
    uint32_t sbo = 1024;
    uint32_t lbo = (64 / 8) * sbo;
    return make_smem_desc_sm100_fn((char*)smem_V1 + offset, lbo, sbo);
}

__device__ __forceinline__ uint64_t smem_desc_P_fn(void* smem_P) {
    uint32_t sbo = 1024;
    uint32_t lbo = 16;
    return make_smem_desc_sm100_fn(smem_P, lbo, sbo);
}

__global__ void attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* __restrict__ LSE,
    int B, int H, int S, float scale) 
{
    int b_h = blockIdx.y;
    int cluster_idx = blockIdx.x / 2;
    int q_block = cluster_idx * 2 + (cluster_rank_fn() % 2);

    int safe_guard = (q_block >= 0 && q_block < (S + 63)/64);
    
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem_pool;                
    __nv_bfloat16* smem_K[2] = {(__nv_bfloat16*)(smem_pool + 16384), (__nv_bfloat16*)(smem_pool + 32768)};       
    __nv_bfloat16* smem_V0[2] = {(__nv_bfloat16*)(smem_pool + 49152), (__nv_bfloat16*)(smem_pool + 65536)};     
    __nv_bfloat16* smem_V1[2] = {(__nv_bfloat16*)(smem_pool + 81920), (__nv_bfloat16*)(smem_pool + 98304)};
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem_pool + 114688);
    
    __shared__ __align__(128) uint64_t mbar_Q[1];
    __shared__ __align__(128) uint64_t mbar_K[2];
    __shared__ __align__(128) uint64_t mbar_V0[2];
    __shared__ __align__(128) uint64_t mbar_V1[2];
    
    int phase_Q[1] = {0};
    int phase_K[2] = {0, 0};
    int phase_V0[2] = {0, 0};
    int phase_V1[2] = {0, 0};
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_Q[0], 1);
        init_smem_barrier_fn(&mbar_K[0], 1);
        init_smem_barrier_fn(&mbar_K[1], 1);
        init_smem_barrier_fn(&mbar_V0[0], 1);
        init_smem_barrier_fn(&mbar_V0[1], 1);
        init_smem_barrier_fn(&mbar_V1[0], 1);
        init_smem_barrier_fn(&mbar_V1[1], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (safe_guard) {
        if (threadIdx.x == 0) {
            uint16_t ctaMask = (1 << cluster_rank_fn());
            mbarrier_arrive_and_expect_tx_fn(&mbar_Q[0], 16384);
            tma_load_multicast_2d_fn(&tma_Q, &mbar_Q[0], smem_Q, 0, b_h * S + q_block * 64, ctaMask);
            tma_load_multicast_2d_fn(&tma_Q, &mbar_Q[0], (char*)smem_Q + 8192, 64, b_h * S + q_block * 64, ctaMask);
        }
    }
    
    int num_q_blocks = (S + 63)/64;
    int max_kv_steps = safe_guard ? q_block + 1 : 0;
    
    if (threadIdx.x == 0 && max_kv_steps > 0) {
        uint16_t ctaMask = (1 << cluster_rank_fn());
        mbarrier_arrive_and_expect_tx_fn(&mbar_K[0], 16384);
        tma_load_multicast_2d_fn(&tma_K, &mbar_K[0], smem_K[0], 0, b_h * S + 0 * 64, ctaMask);
        tma_load_multicast_2d_fn(&tma_K, &mbar_K[0], (char*)smem_K[0] + 8192, 64, b_h * S + 0 * 64, ctaMask);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_V0[0], 16384);
        tma_load_multicast_2d_fn(&tma_V, &mbar_V0[0], smem_V0[0], 0, b_h * S + 0 * 64, ctaMask);
        tma_load_multicast_2d_fn(&tma_V, &mbar_V0[0], (char*)smem_V0[0] + 8192, 64, b_h * S + 0 * 64, ctaMask);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_V1[0], 16384);
        tma_load_multicast_2d_fn(&tma_V, &mbar_V1[0], smem_V1[0], 0, b_h * S + 0 * 64, ctaMask);
        tma_load_multicast_2d_fn(&tma_V, &mbar_V1[0], (char*)smem_V1[0] + 8192, 64, b_h * S + 0 * 64, ctaMask);
    }
    mbarrier_wait_fn(&mbar_Q[0], phase_Q[0]);
    phase_Q[0] ^= 1;
    
    uint32_t tmem_Q0, tmem_Q1;
    uint32_t tmem_K0, tmem_K1;
    uint32_t tmem_V0, tmem_V1;
    uint32_t tmem_S, tmem_P, tmem_O0, tmem_O1;
    
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_Q0, 64);
        tmem_alloc_fn(&tmem_Q1, 64);
        tmem_alloc_fn(&tmem_K0, 64);
        tmem_alloc_fn(&tmem_K1, 64);
        tmem_alloc_fn(&tmem_V0, 64);
        tmem_alloc_fn(&tmem_V1, 64);
        tmem_alloc_fn(&tmem_S, 64);
        tmem_P = tmem_S;
        tmem_alloc_fn(&tmem_O0, 64);
        tmem_alloc_fn(&tmem_O1, 64);
    }
    __syncthreads();
    
    if (safe_guard) {
        if (threadIdx.x == 0) {
            tmem_cp_64x128b(tmem_Q0, smem_Q);
            tmem_cp_64x128b(tmem_Q1, (char*)smem_Q + 8192);
        }
    }

    uint32_t my_m_base = (threadIdx.x % 2) * 8;
    int warp_id = threadIdx.x / 32;
    int tid = threadIdx.x % 32;
    float global_max = -INFINITY;
    float global_sum = 0;
    
    for (int query_step = 0; query_step < num_q_blocks; ++query_step) {
        if (threadIdx.x == 0) {
            tmem_dealloc_fn(tmem_O0, 64);
            tmem_dealloc_fn(tmem_O1, 64);
            tmem_dealloc_fn(tmem_S, 64);
            tmem_dealloc_fn(tmem_P, 64);
            
            tmem_alloc_fn(&tmem_O0, 64);
            tmem_alloc_fn(&tmem_O1, 64);
            tmem_alloc_fn(&tmem_S, 64);
            tmem_P = tmem_S;
            tmem_alloc_fn(&tmem_P, 64);
        }
        __syncthreads();
        
        float rescale_o = 1.0f;
        if (global_sum > 0) {
            rescale_o = __expf(global_max);
        }
        
        float current_max = -INFINITY;
        float current_sum = 0;
        
        int q_step = cluster_rank_fn() == 0 ? query_step - 1 : query_step;
        
        for (int step = 0; step <= q_step && step < num_q_blocks; step++) {
            if (step + 1 <= q_step && threadIdx.x == 0) {
                uint16_t ctaMask = (1 << cluster_rank_fn());
                mbarrier_arrive_and_expect_tx_fn(&mbar_K[(step + 1) % 2], 16384);
                tma_load_multicast_2d_fn(&tma_K, &mbar_K[(step + 1) % 2], smem_K[(step + 1) % 2], 0, b_h * S + (step + 1) * 64, ctaMask);
                tma_load_multicast_2d_fn(&tma_K, &mbar_K[(step + 1) % 2], (char*)smem_K[(step + 1) % 2] + 8192, 64, b_h * S + (step + 1) * 64, ctaMask);
                
                mbarrier_arrive_and_expect_tx_fn(&mbar_V0[(step + 1) % 2], 16384);
                tma_load_multicast_2d_fn(&tma_V, &mbar_V0[(step + 1) % 2], smem_V0[(step + 1) % 2], 0, b_h * S + (step + 1) * 64, ctaMask);
                tma_load_multicast_2d_fn(&tma_V, &mbar_V0[(step + 1) % 2], (char*)smem_V0[(step + 1) % 2] + 8192, 64, b_h * S + (step + 1) * 64, ctaMask);
                
                mbarrier_arrive_and_expect_tx_fn(&mbar_V1[(step + 1) % 2], 16384);
                tma_load_multicast_2d_fn(&tma_V, &mbar_V1[(step + 1) % 2], smem_V1[(step + 1) % 2], 0, b_h * S + (step + 1) * 64, ctaMask);
                tma_load_multicast_2d_fn(&tma_V, &mbar_V1[(step + 1) % 2], (char*)smem_V1[(step + 1) % 2] + 8192, 64, b_h * S + (step + 1) * 64, ctaMask);
            }
            
            mbarrier_wait_fn(&mbar_K[step % 2], phase_K[step % 2]);
            phase_K[step % 2] ^= 1;
            mbarrier_wait_fn(&mbar_V0[step % 2], phase_V0[step % 2]);
            phase_V0[step % 2] ^= 1;
            mbarrier_wait_fn(&mbar_V1[step % 2], phase_V1[step % 2]);
            phase_V1[step % 2] ^= 1;
            
            if (threadIdx.x == 0) {
                tmem_dealloc_fn(tmem_K0, 64); tmem_dealloc_fn(tmem_K1, 64);
                tmem_dealloc_fn(tmem_V0, 64); tmem_dealloc_fn(tmem_V1, 64);
                tmem_alloc_fn(&tmem_K0, 64); tmem_alloc_fn(&tmem_K1, 64);
                tmem_alloc_fn(&tmem_V0, 64); tmem_alloc_fn(&tmem_V1, 64);
                
                tmem_cp_64x128b(tmem_K0, smem_K[step % 2]);
                tmem_cp_64x128b(tmem_K1, (char*)smem_K[step % 2] + 8192);
                tmem_cp_64x128b(tmem_V0, smem_V0[step % 2]);
                tmem_cp_64x128b(tmem_V1, smem_V1[step % 2]);
            }
            
            uint64_t desc_a0 = smem_desc_Q0_fn(smem_Q);
            uint64_t desc_b0 = smem_desc_K0_fn(smem_K[step % 2]);
            uint32_t idesc0 = make_instr_desc_fn(64, 64, 0, 1); 
            uint32_t accum0 = (step == 0) ? 0 : 1;
            
            if (threadIdx.x == 0) {
                umma_f16_cg2_fn(tmem_S, desc_a0, desc_b0, idesc0, accum0);
                umma_commit_2sm_fn(&mbar_K[step % 2]);
            }
            
            uint64_t desc_a1 = smem_desc_Q1_fn(smem_Q);
            uint64_t desc_b1 = smem_desc_K1_fn(smem_K[step % 2]);
            uint32_t idesc1 = make_instr_desc_fn(64, 64, 0, 1);
            uint32_t accum1 = (step == 0) ? 0 : 1;
            
            if (threadIdx.x == 0) {
                umma_f16_cg2_fn(tmem_S, desc_a1, desc_b1, idesc1, accum1);
                umma_commit_2sm_fn(&mbar_K[step % 2]);
            }
            
            mbarrier_wait_fn(&mbar_K[step % 2], phase_K[step % 2]);
            phase_K[step % 2] ^= 1;
            
            uint32_t r0[8], r1[8];
            tmem_load_8x_fn(tmem_S + 0, &r0[0], &r0[1], &r0[2], &r0[3], &r0[4], &r0[5], &r0[6], &r0[7]);
            tmem_load_8x_fn(tmem_S + 32, &r1[0], &r1[1], &r1[2], &r1[3], &r1[4], &r1[5], &r1[6], &r1[7]);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float thread_max = -INFINITY;
            for(int i = 0; i < 8; i++) {
                float val = __uint_as_float(r0[i]);
                int col = (i / 4) * 8 + (i % 4);
                int global_row = q_step * 64 + my_m_base + warp_id * 16 + (tid % 4);
                int global_col = step * 64 + col;
                if (global_col > global_row) {
                    val = -INFINITY;
                }
                val *= scale;
                r0[i] = __float_as_uint(val);
                thread_max = fmaxf(thread_max, val);
            }
            for(int i = 0; i < 8; i++) {
                float val = __uint_as_float(r1[i]);
                int col = (i / 4) * 8 + (i % 4);
                int global_row = q_step * 64 + my_m_base + warp_id * 16 + (tid % 4) + 16;
                int global_col = step * 64 + col;
                if (global_col > global_row) {
                    val = -INFINITY;
                }
                val *= scale;
                r1[i] = __float_as_uint(val);
                thread_max = fmaxf(thread_max, val);
            }
            
            float row_max[2] = {-INFINITY, -INFINITY};
            row_max[0] = thread_max;
            for(int i=1; i<4; i++) row_max[0] = fmaxf(row_max[0], __shfl_xor_sync(0xffffffff, row_max[0], i));
            
            row_max[1] = thread_max;
            for(int i=1; i<4; i++) row_max[1] = fmaxf(row_max[1], __shfl_xor_sync(0xffffffff, row_max[1], i));
            
            float new_max[2];
            float r_o[2] = {1.0f, 1.0f};
            float r_s[2] = {1.0f, 1.0f};
            
            new_max[0] = fmaxf(global_max, row_max[0]);
            r_o[0] = __expf(global_max - new_max[0]);
            r_s[0] = r_o[0];
            current_sum *= r_s[0];
            
            new_max[1] = fmaxf(global_max, row_max[1]);
            r_o[1] = __expf(global_max - new_max[1]);
            r_s[1] = r_o[1];
            current_sum *= r_s[1];
            
            float thread_sum = 0;
            for(int i = 0; i < 8; i++) {
                int row_idx = (i / 4);
                float val = __uint_as_float(row_idx == 0 ? r0[i] : r1[i]);
                val = __expf(val - (row_idx == 0 ? new_max[0] : new_max[1]));
                thread_sum += val;
                
                uint32_t tmem_addr = (tid << 4) + ((tid / 4) * 8) + (i % 4);
                asm volatile("tcgen05.st.sync.aligned.b32 [%0], %1;" :: "r"(tmem_S + tmem_addr), "r"(__float_as_uint(val)) : "memory");
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            
            float row_sum[2] = {thread_sum, thread_sum};
            for(int i=1; i<4; i++) {
                row_sum[0] += __shfl_xor_sync(0xffffffff, row_sum[0], i);
                row_sum[1] += __shfl_xor_sync(0xffffffff, row_sum[1], i);
            }
            
            current_sum += row_sum[0] * r_s[0] + row_sum[1] * r_s[1];
            
            global_max = new_max[0]; // Assuming uniform max for simplicity across the warp rows
            
            uint64_t desc_p = smem_desc_P_fn(smem_P);
            uint64_t desc_v0 = smem_desc_V0_fn(smem_V0[step % 2], 0);
            uint32_t idesc_pv0 = make_instr_desc_fn(64, 64, 0, 0);
            
            if (threadIdx.x == 0) {
                umma_f16_cg2_fn(tmem_O0, desc_p, desc_v0, idesc_pv0, 1); 
                umma_commit_2sm_fn(&mbar_V0[step % 2]);
            }
            
            uint64_t desc_v1 = smem_desc_V1_fn(smem_V1[step % 2], 0);
            uint32_t idesc_pv1 = make_instr_desc_fn(64, 64, 0, 0);
            
            if (threadIdx.x == 0) {
                umma_f16_cg2_fn(tmem_O1, desc_p, desc_v1, idesc_pv1, 1); 
                umma_commit_2sm_fn(&mbar_V1[step % 2]);
            }
            
            mbarrier_wait_fn(&mbar_V0[step % 2], phase_V0[step % 2]);
            phase_V0[step % 2] ^= 1;
            mbarrier_wait_fn(&mbar_V1[step % 2], phase_V1[step % 2]);
            phase_V1[step % 2] ^= 1;
        }
        
        float final_sum = current_sum;
        
        for (int i = 0; i < 4; i++) {
            uint32_t tmem_addr = (tid << 4) + ((tid / 4) * 8) + (i % 4);
            asm volatile("tcgen05.st.sync.aligned.b32 [%0], %1;" :: "r"(tmem_O0 + tmem_addr), "r"(__float_as_uint(0.0f)) : "memory");
            asm volatile("tcgen05.st.sync.aligned.b32 [%0], %1;" :: "r"(tmem_O1 + tmem_addr), "r"(__float_as_uint(0.0f)) : "memory");
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        
        tmem_store_bf16_packed_row_fn(tmem_O0, tid, smem_P);
        if (safe_guard) {
            if (tid == 0) {
                tma_store_2d_fn(&tma_O, smem_P, 0, b_h * S + q_step * 64);
                tma_store_commit_fn();
            }
        }
        
        tmem_store_bf16_packed_row_fn(tmem_O1, tid, smem_P);
        if (safe_guard) {
            if (tid == 0) {
                tma_store_2d_fn(&tma_O, smem_P, 64, b_h * S + q_step * 64);
                tma_store_commit_fn();
            }
        }
        
        tma_store_wait_fn<0>();
        
        if (safe_guard) {
            if (tid == 0) {
                uint32_t row = q_step * 64 + warp_id * 16 + tid;
                if (row < S) {
                    float sum_val = __expf(current_max - global_max) + __logf(final_sum.y);
                    LSE_bh[row] = sum_val;
                }
            }
        }
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_Q0, 64);
        tmem_dealloc_fn(tmem_Q1, 64);
        tmem_dealloc_fn(tmem_K0, 64);
        tmem_dealloc_fn(tmem_K1, 64);
        tmem_dealloc_fn(tmem_V0, 64);
        tmem_dealloc_fn(tmem_V1, 64);
        tmem_dealloc_fn(tmem_O0, 64);
        tmem_dealloc_fn(tmem_O1, 64);
        tmem_dealloc_fn(tmem_S, 64);
        tmem_dealloc_fn(tmem_P, 64);
    }
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 
    
    if (S == 0) return;
    
    float scale = 1.0f / sqrtf((float)D);
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    uint64_t gmem_dim[2] = {(uint64_t)D, (uint64_t)(B * H * S)};
    uint32_t smem_dim[2] = {64, 64}; 
    
    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_Q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)Q_data, gmem_dim, (const cuuint64_t*)gmem_dim, smem_dim, (const cuuint32_t[]){{1, 1}}, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_K, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)K_data, gmem_dim, (const cuuint64_t*)gmem_dim, smem_dim, (const cuuint32_t[]){{1, 1}}, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_V, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)V_data, gmem_dim, (const cuuint64_t*)gmem_dim, smem_dim, (const cuuint32_t[]){{1, 1}}, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_O, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)O_data, gmem_dim, (const cuuint64_t*)gmem_dim, smem_dim, (const cuuint32_t[]){{1, 1}}, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    int num_q_blocks = (S + 63) / 64;
    dim3 grid(num_q_blocks * 2, B * H); 
    dim3 block(128); 
    
    int smem_size = 128 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(
        attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2; 
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, attention_kernel,
        tma_Q, tma_K, tma_V, tma_O,
        LSE_data, B, H, S, scale
    ));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(config.stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_example_cuda