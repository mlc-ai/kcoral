#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
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
                (int)_e, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)


CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t dim0, uint64_t dim1, uint64_t dim2, 
                                     uint32_t box0, uint32_t box1, uint32_t box2,
                                     CUtensorMapSwizzle swizzle, 
                                     CUtensorMapL2promotion l2Promotion, 
                                     CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
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

__device__ __forceinline__ void fence_proxy_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
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

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_nmaj(void* smem_ptr, int k_dim, int n_dim) {
    uint32_t sbo = 1024; // 8 spans * 128B = 1024B
    uint32_t lbo = (k_dim / 8) * sbo; 
    return make_smem_desc_sm100_fn(smem_ptr, lbo, sbo);
}

__device__ __forceinline__ uint64_t modify_base_offset(uint64_t desc, void* new_smem_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(new_smem_ptr);
    uint64_t new_d = (desc & ~0x3FFF) | ((addr & 0x3FFFF) >> 4);
    uint32_t base_offset = (addr >> 7) & 0x7;
    new_d &= ~(0x7ULL << 49);
    new_d |= ((uint64_t)base_offset << 49);
    return new_d;
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

__device__ __forceinline__ uint32_t make_instr_desc_mixed_major_fn(uint32_t M, uint32_t N) {
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

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ void named_barrier_sync_fn_warp() {
    asm volatile("barrier.sync.aligned 2, 32;" ::: "memory");
}

__global__ void attention_forward_cluster(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    uint64_t S,
    uint64_t D,
    uint64_t B,
    uint64_t H) 
{
    if (threadIdx.x < 32) {
        setmaxnreg_inc_sync_fn<256>();
    }

    extern __shared__ __align__(128) uint8_t smem_pool[];
    uintptr_t pool_addr = (uintptr_t)smem_pool;
    pool_addr = (pool_addr + 127) & ~127; 
    
    uint64_t* mbar_Q   = reinterpret_cast<uint64_t*>(pool_addr);
    uint64_t* mbar_K0  = mbar_Q + 1;
    uint64_t* mbar_K1  = mbar_K0 + 1;
    uint64_t* mbar_V0  = mbar_K1 + 1;
    uint64_t* mbar_V1  = mbar_V0 + 1;
    uint64_t* mbar_comp = mbar_V1 + 1;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1); 
        init_smem_barrier_fn(mbar_K0, 1); 
        init_smem_barrier_fn(mbar_K1, 1); 
        init_smem_barrier_fn(mbar_V0, 1); 
        init_smem_barrier_fn(mbar_V1, 1); 
        init_smem_barrier_fn(mbar_comp, 1);
    }
    fence_smem_barrier_init_fn();
    
    uint32_t* tmem_S = reinterpret_cast<uint32_t*>(pool_addr + 48);
    uint32_t* tmem_O = reinterpret_cast<uint32_t*>(pool_addr + 52);
    
    if (threadIdx.x < 32) {
        tmem_alloc_fn(tmem_S, 64);
        tmem_alloc_fn(tmem_O, 128);
    }

    char* smem_Q = reinterpret_cast<char*>(((uintptr_t)(pool_addr + 48) + 1023) & ~1023);
    
    uint32_t row_base = (uint32_t)(blockIdx.x * 64);
    uint32_t head_idx = (blockIdx.x * 64) / S;
    uint32_t myRow = row_base % S;
    uint32_t myRow_c1 = myRow;
    uint32_t myRow_c2 = head_idx;
    
    char* s_Q0_0 = smem_Q;                
    char* s_Q0_1 = smem_Q + 8192;         
    char* s_Q1_0 = smem_Q + 16384;        
    char* s_Q1_1 = smem_Q + 24576;        

    char* s_K0 = smem_Q + 32768;          
    char* s_K1 = smem_Q + 40960;          
    char* s_K2 = smem_Q + 49152;          
    char* s_K3 = smem_Q + 57344;          

    char* s_P  = smem_Q + 65536;          
    char* s_V0 = smem_Q + 73728;          
    char* s_V1 = smem_Q + 81920;          
    char* s_V2 = smem_Q + 90112;          
    char* s_V3 = smem_Q + 98304;          

    int32_t c0_Q0 = 0;
    int32_t c0_Q1 = 64;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 8192 * 4);
        tma_load_3d_fn(&tma_Q, mbar_Q, s_Q0_0, c0_Q0, myRow_c1, myRow_c2);
        tma_load_3d_fn(&tma_Q, mbar_Q, s_Q0_1, c0_Q1, myRow_c1, myRow_c2);
        tma_load_3d_fn(&tma_Q, mbar_Q, s_Q1_0, c0_Q0, (myRow_c1 + 64) % S, myRow_c2);
        tma_load_3d_fn(&tma_Q, mbar_Q, s_Q1_1, c0_Q1, (myRow_c1 + 64) % S, myRow_c2);
    }
    
    mbarrier_wait_fn(mbar_Q, 0);
    
    uint64_t* mbar_K_local[2] = {mbar_K0, mbar_K1};
    uint64_t* mbar_V_local[2] = {mbar_V0, mbar_V1};
    
    int cur_k = 0;
    int cur_v = 0;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_K_local[cur_k], 8192 * 2);
        tma_load_3d_fn(&tma_K, &mbar_K_local[cur_k], s_K0, 0, 0, head_idx);
        tma_load_3d_fn(&tma_K, &mbar_K_local[cur_k], s_K1, 64, 0, head_idx);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_V_local[cur_v], 8192 * 2);
        tma_load_3d_fn(&tma_V, &mbar_V_local[cur_v], s_V0, 0, 0, head_idx);
        tma_load_3d_fn(&tma_V, &mbar_V_local[cur_v], s_V1, 64, 0, head_idx);
    }
    
    uint64_t desc_Q0_0_base = make_smem_desc_sm100_fn(s_Q0_0, 1, 1024);
    uint64_t desc_Q0_1_base = make_smem_desc_sm100_fn(s_Q0_1, 1, 1024);
    uint64_t desc_Q1_0_base = make_smem_desc_sm100_fn(s_Q1_0, 1, 1024);
    uint64_t desc_Q1_1_base = make_smem_desc_sm100_fn(s_Q1_1, 1, 1024);
    uint64_t desc_K0_base = make_smem_desc_sm100_fn(s_K0, 1, 1024);
    uint64_t desc_K1_base = make_smem_desc_sm100_fn(s_K1, 1, 1024);
    uint64_t desc_K2_base = make_smem_desc_sm100_fn(s_K2, 1, 1024);
    uint64_t desc_K3_base = make_smem_desc_sm100_fn(s_K3, 1, 1024);
    uint64_t desc_P_base = make_smem_desc_sm100_fn(s_P, 1, 1024);
    uint64_t desc_V0_base = make_smem_desc_nmaj(s_V0, 64, 64);
    uint64_t desc_V1_base = make_smem_desc_nmaj(s_V1, 64, 64);
    
    uint32_t idesc_QK = make_instr_desc_fn(64, 64);
    uint32_t idesc_PV = make_instr_desc_mixed_major_fn(64, 128);
    
    float row_global_max = -1e20f;
    float row_global_sum = 0.0f;
    bool first_iter = true;
    int phase_K[2] = {0, 0};
    int phase_V[2] = {0, 0};
    int phase_comp = 0;
    
    const float SQRT_D = 1.0f / sqrtf((float)D);
    const float LOG2E = 1.4426950408889634f;
    const float LN2 = 0.6931471805599453f;
    
    int num_iters = (S + 63) / 64;

    for (int j_blk = 0; j_blk < num_iters; j_blk++) {
        if (threadIdx.x < 32) {
            mbarrier_wait_fn(&mbar_K_local[cur_k], phase_K[cur_k]);
        }
        
        int n_valid[128];
        if (threadIdx.x < 128) {
            n_valid[threadIdx.x] = (j_blk * 64 + threadIdx.x < S);
        }
        __syncthreads();
        
        uint32_t accum = first_iter ? 0 : 1;
        int q_idx = (j_blk / 2) % 2;
        for (int k_idx = 0; k_idx < 8; k_idx++) {
            uint64_t desc_Q_curr0, desc_Q_curr1;
            if (q_idx == 0) {
                desc_Q_curr0 = modify_base_offset(desc_Q0_0_base, (uint64_t)s_Q0_0 + k_idx * 8);
                desc_Q_curr1 = modify_base_offset(desc_Q0_1_base, (uint64_t)s_Q0_1 + k_idx * 8);
            } else {
                desc_Q_curr0 = modify_base_offset(desc_Q1_0_base, (uint64_t)s_Q1_0 + k_idx * 8);
                desc_Q_curr1 = modify_base_offset(desc_Q1_1_base, (uint64_t)s_Q1_1 + k_idx * 8);
            }
            
            uint64_t desc_K_curr0, desc_K_curr1;
            if (cur_k == 0) {
                desc_K_curr0 = modify_base_offset(desc_K0_base, (uint64_t)s_K0 + k_idx * 8);
                desc_K_curr1 = modify_base_offset(desc_K1_base, (uint64_t)s_K1 + k_idx * 8);
            } else {
                desc_K_curr0 = modify_base_offset(desc_K2_base, (uint64_t)s_K2 + k_idx * 8);
                desc_K_curr1 = modify_base_offset(desc_K3_base, (uint64_t)s_K3 + k_idx * 8);
            }
            
            if (threadIdx.x == 0) {
                if (accum == 0) {
                    umma_f16_cg2_fn(*tmem_S, desc_Q_curr0, desc_K_curr0, idesc_QK, accum);
                    umma_f16_cg2_fn(*tmem_S, desc_Q_curr1, desc_K_curr1, idesc_QK, accum);
                } else {
                    umma_f16_cg2_fn(*tmem_S, desc_Q_curr0, desc_K_curr0, idesc_QK, accum);
                    umma_f16_cg2_fn(*tmem_S, desc_Q_curr1, desc_K_curr1, idesc_QK, accum);
                }
            }
            accum = 1;
        }
        first_iter = false;
        
        if (threadIdx.x == 0) {
            umma_commit_2sm_fn(mbar_comp); 
        }
        mbarrier_wait_fn(mbar_comp, phase_comp);
        
        float local_max = -1e20f;
        float local_sum = 0.0f;
        
        if (threadIdx.x >= 32 && threadIdx.x < 96) {
            uint32_t r0, r1, r2, r3;
            float vals[8];
            for (int col_group = 0; col_group < 2; col_group++) {
                int col_start = col_group * 32 + lane_id * 4;
                tmem_load_4x_fn(col_start, &r0, &r1, &r2, &r3);
                vals[col_group * 4 + 0] = __uint_as_float(r0);
                vals[col_group * 4 + 1] = __uint_as_float(r1);
                vals[col_group * 4 + 2] = __uint_as_float(r2);
                vals[col_group * 4 + 3] = __uint_as_float(r3);
            }
            tmem_load_fence_fn();
            
            int warp_id = (threadIdx.x - 32) / 32;
            int lane_id = (threadIdx.x - 32) % 32;
            int m_idx = warp_id * 32 + lane_id;
            
            float my_max = -1e20f;
            for(int i = 0; i < 8; i++) {
                int n_idx = (i / 4) * 32 + (i % 4) * 8 + lane_id;
                if (n_valid[n_idx]) {
                    float val = vals[i] * SQRT_D;
                    if (val > my_max) my_max = val;
                }
            }
            
            my_max = fmaxf(my_max, __shfl_xor_sync(0xFFFFFFFF, my_max, 1));
            my_max = fmaxf(my_max, __shfl_xor_sync(0xFFFFFFFF, my_max, 2));
            my_max = fmaxf(my_max, __shfl_xor_sync(0xFFFFFFFF, my_max, 4));
            my_max = fmaxf(my_max, __shfl_xor_sync(0xFFFFFFFF, my_max, 8));
            my_max = fmaxf(my_max, __shfl_xor_sync(0xFFFFFFFF, my_max, 16));
            
            local_max = my_max;
            
            float my_sum = 0.0f;
            for(int i = 0; i < 8; i++) {
                int n_idx = (i / 4) * 32 + (i % 4) * 8 + lane_id;
                if (n_valid[n_idx]) {
                    float val = vals[i] * SQRT_D;
                    float p = exp2f((val - my_max) * LOG2E);
                    my_sum += p;
                    vals[i] = p;
                } else {
                    vals[i] = 0.0f;
                }
            }
            
            my_sum += __shfl_xor_sync(0xFFFFFFFF, my_sum, 1);
            my_sum += __shfl_xor_sync(0xFFFFFFFF, my_sum, 2);
            my_sum += __shfl_xor_sync(0xFFFFFFFF, my_sum, 4);
            my_sum += __shfl_xor_sync(0xFFFFFFFF, my_sum, 8);
            my_sum += __shfl_xor_sync(0xFFFFFFFF, my_sum, 16);
            
            local_sum = my_sum;
            
            float new_max = fmaxf(row_global_max, local_max);
            float correction = exp2f((row_global_max - new_max) * LOG2E);
            float new_sum = row_global_sum * correction + local_sum * exp2f((local_max - new_max) * LOG2E);
            
            row_global_max = new_max;
            row_global_sum = new_sum;
            
            for(int i = 0; i < 8; i++) {
                int n_idx = (i / 4) * 32 + (i % 4) * 8 + lane_id;
                float p = vals[i] * exp2f((local_max - new_max) * LOG2E);
                int x_chunk = n_idx / 8;
                int x_rem = n_idx % 8;
                int swizzled_x_chunk = (m_idx % 8) ^ x_chunk;
                int phys_k = swizzled_x_chunk * 8 + x_rem;
                __nv_bfloat16 pb = __float2bfloat16(p);
                s_P[m_idx * 64 + phys_k] = pb;
            }
        }
        
        named_barrier_sync_fn_warp();
        
        int next_k = cur_k ^ 1;
        int next_v = cur_v ^ 1;
        
        if (threadIdx.x < 32) {
            if (j_blk + 1 < num_iters) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_K_local[next_k], 8192 * 2);
                if (next_k == 0) {
                    tma_load_3d_fn(&tma_K, &mbar_K_local[next_k], s_K0, 0, (j_blk + 1) * 64 % S, head_idx);
                    tma_load_3d_fn(&tma_K, &mbar_K_local[next_k], s_K1, 64, (j_blk + 1) * 64 % S, head_idx);
                } else {
                    tma_load_3d_fn(&tma_K, &mbar_K_local[next_k], s_K2, 0, (j_blk + 1) * 64 % S, head_idx);
                    tma_load_3d_fn(&tma_K, &mbar_K_local[next_k], s_K3, 64, (j_blk + 1) * 64 % S, head_idx);
                }
                
                mbarrier_arrive_and_expect_tx_fn(&mbar_V_local[next_v], 8192 * 2);
                if (next_v == 0) {
                    tma_load_3d_fn(&tma_V, &mbar_V_local[next_v], s_V0, 0, (j_blk + 1) * 64 % S, head_idx);
                    tma_load_3d_fn(&tma_V, &mbar_V_local[next_v], s_V1, 64, (j_blk + 1) * 64 % S, head_idx);
                } else {
                    tma_load_3d_fn(&tma_V, &mbar_V_local[next_v], s_V2, 0, (j_blk + 1) * 64 % S, head_idx);
                    tma_load_3d_fn(&tma_V, &mbar_V_local[next_v], s_V3, 64, (j_blk + 1) * 64 % S, head_idx);
                }
            }
            named_barrier_sync_fn_warp();
            mbarrier_wait_fn(&mbar_V_local[cur_v], phase_V[cur_v]);
        }
        
        if (threadIdx.x >= 32 && threadIdx.x < 96) {
            named_barrier_sync_fn_warp();
            if (threadIdx.x == 32) { 
                accum = 1;
                for (int k_idx = 0; k_idx < 8; k_idx++) {
                    int n_idx = k_idx / 4;
                    int n_offset = n_idx * 8 * 16;
                    uint64_t desc_P_curr = modify_base_offset(desc_P_base, (uint64_t)s_P + k_idx * 8);
                    
                    uint64_t desc_V_curr0 = modify_base_offset(desc_V0_base, (uint64_t)((cur_v == 0 ? s_V0 : s_V2) + n_offset + k_idx * 8 * 128));
                    uint64_t desc_V_curr1 = modify_base_offset(desc_V1_base, (uint64_t)((cur_v == 0 ? s_V1 : s_V3) + n_offset + k_idx * 8 * 128));
                    
                    umma_f16_cg2_fn(*tmem_O, desc_P_curr, desc_V_curr0, idesc_PV, accum);
                    umma_f16_cg2_fn(*tmem_O, desc_P_curr, desc_V_curr1, idesc_PV, accum);
                }
                umma_commit_2sm_fn(mbar_comp); 
            }
            mbarrier_wait_fn(mbar_comp, phase_comp);
        }
        
        if (threadIdx.x >= 96) {
            if (threadIdx.x == 96) { 
                accum = 1;
                for (int k_idx = 0; k_idx < 8; k_idx++) {
                    int n_idx = k_idx / 4;
                    int n_offset = n_idx * 8 * 16;
                    uint64_t desc_P_curr = modify_base_offset(desc_P_base, (uint64_t)s_P + k_idx * 8);
                    
                    uint64_t desc_V_curr0 = modify_base_offset(desc_V0_base, (uint64_t)((cur_v == 0 ? s_V0 : s_V2) + n_offset + k_idx * 8 * 128));
                    uint64_t desc_V_curr1 = modify_base_offset(desc_V1_base, (uint64_t)((cur_v == 0 ? s_V1 : s_V3) + n_offset + k_idx * 8 * 128));
                    
                    umma_f16_cg2_fn(*tmem_O, desc_P_curr, desc_V_curr0, idesc_PV, accum);
                    umma_f16_cg2_fn(*tmem_O, desc_P_curr, desc_V_curr1, idesc_PV, accum);
                }
                umma_commit_2sm_fn(mbar_comp); 
            }
            mbarrier_wait_fn(mbar_comp, phase_comp);
        }
        
        cur_k = next_k;
        cur_v = next_v;
        phase_K[cur_k] ^= 1;
        phase_V[cur_v] ^= 1;
        phase_comp ^= 1;
    }
    
    uint32_t myRow_tmem = (myRow < 64) ? myRow : myRow - 64;
    
    for (int col_idx = 0; col_idx < 4; col_idx++) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(col_idx * 8 + (threadIdx.x % 2) * 64, &r0, &r1, &r2, &r3);
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        int c_base = col_idx * 8 + (threadIdx.x % 2) * 64;
        int c0 = (c_base >> 3) ^ (myRow_tmem & 7);
        int c1 = (c_base & 7);
        int col = (c0 << 3) + c1;
        
        int row_idx = head_idx * S + myRow_tmem;
        
        if (row_idx < (int)(B*H*S)) {
            O[row_idx * D + col] = __float2bfloat16(f0 / row_global_sum);
            O[row_idx * D + col + 1] = __float2bfloat16(f1 / row_global_sum);
            O[row_idx * D + col + 2] = __float2bfloat16(f2 / row_global_sum);
            O[row_idx * D + col + 3] = __float2bfloat16(f3 / row_global_sum);
        }
    }
    
    for (int col_idx = 0; col_idx < 4; col_idx++) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(col_idx * 8 + (threadIdx.x % 2) * 64 + 64, &r0, &r1, &r2, &r3);
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        int c_base = col_idx * 8 + (threadIdx.x % 2) * 64;
        int c0 = (c_base >> 3) ^ (myRow_tmem & 7);
        int c1 = (c_base & 7);
        int col = (c0 << 3) + c1 + 64;
        
        int row_idx = head_idx * S + myRow_tmem;
        
        if (row_idx < (int)(B*H*S)) {
            O[row_idx * D + col] = __float2bfloat16(f0 / row_global_sum);
            O[row_idx * D + col + 1] = __float2bfloat16(f1 / row_global_sum);
            O[row_idx * D + col + 2] = __float2bfloat16(f2 / row_global_sum);
            O[row_idx * D + col + 3] = __float2bfloat16(f3 / row_global_sum);
        }
    }
    
    if (threadIdx.x >= 32 && threadIdx.x < 96) {
        int warp_id = (threadIdx.x - 32) / 32;
        int lane_id = (threadIdx.x - 32) % 32;
        int m_idx = warp_id * 32 + lane_id;
        
        uint32_t myRow = m_idx;
        int row_idx = head_idx * S + myRow;
        
        if (row_idx < (int)(B*H*S)) {
            float lse = row_global_max + log2f(row_global_sum) * LN2;
            LSE[row_idx] = lse;
        }
    }
    
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(*tmem_S, 64);
        tmem_dealloc_fn(*tmem_O, 128);
    }
}

namespace tvm_ffi {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    uint64_t B = Q.size(0);
    uint64_t H = Q.size(1);
    uint64_t S_val = Q.size(2);
    uint64_t D = Q.size(3);

    void* Q_ptr = Q.data_ptr();
    void* K_ptr = K.data_ptr();
    void* V_ptr = V.data_ptr();
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V;

    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, Q_ptr, 128, S_val, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, K_ptr, 128, S_val, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, V_ptr, 128, S_val, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    dim3 grid((S_val + 63) / 64, 1, 1);
    dim3 block(128, 1, 1);

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 128 * 1024; 
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaFuncSetAttribute((const void*)attention_forward_cluster, cudaFuncAttributeMaxDynamicSharedMemorySize, 128 * 1024));
    CUDA_CHECK(cudaLaunchKernelEx(&config, attention_forward_cluster, tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S_val, D, B, H));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(config.stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi