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
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n",                 \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx_cluster_fn(uint64_t* bar, uint32_t tx, uint32_t target_cta) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    uint32_t remote_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_a) : "r"(a), "r"(target_cta));
    asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;" :: "r"(remote_a), "r"(tx) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

__device__ __forceinline__ void tma_load_3d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba_local = (uint32_t)__cvta_generic_to_shared(bar);
    uint32_t ba;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(ba) : "r"(ba_local));
    asm volatile(
        "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(c2), "r"(ba) : "memory");
}

__device__ __forceinline__ void tma_store_3d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_wait_st_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_load_32x_fn(uint32_t col, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                 "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                 : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                   "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
                   "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
                   "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
                 : "r"(col));
}

__device__ __forceinline__ void tmem_store_32x_fn(uint32_t col, const uint32_t* r) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                 "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
                 :: "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),
                    "r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),"r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]),
                    "r"(r[16]),"r"(r[17]),"r"(r[18]),"r"(r[19]),"r"(r[20]),"r"(r[21]),"r"(r[22]),"r"(r[23]),
                    "r"(r[24]),"r"(r[25]),"r"(r[26]),"r"(r[27]),"r"(r[28]),"r"(r[29]),"r"(r[30]),"r"(r[31]),
                    "r"(col));
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t lbo = 1;
    uint32_t sbo = 1024;
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 49;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major_v(void* ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t lbo = 32768; 
    uint32_t sbo = 1024;
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 49;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N, int a_major, int b_major) {
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

__device__ __forceinline__ void umma_f16(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, int accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t pack_bf16_fast(float a, float b) {
    __nv_bfloat162 res = __floats2bfloat162_rn(a, b);
    return *(uint32_t*)&res;
}

__global__ void mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_q,
    const __grid_constant__ CUtensorMap tma_k,
    const __grid_constant__ CUtensorMap tma_v,
    const __grid_constant__ CUtensorMap tma_o,
    float* lse_out,
    int S
) {
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    __nv_bfloat16* q_smem_0 = (__nv_bfloat16*)(smem_pool + 0 * 16384);
    __nv_bfloat16* q_smem_1 = (__nv_bfloat16*)(smem_pool + 1 * 16384);
    
    __nv_bfloat16* k_smem_0_0 = (__nv_bfloat16*)(smem_pool + 2 * 16384);
    __nv_bfloat16* k_smem_0_1 = (__nv_bfloat16*)(smem_pool + 3 * 16384);
    __nv_bfloat16* k_smem_1_0 = (__nv_bfloat16*)(smem_pool + 4 * 16384);
    __nv_bfloat16* k_smem_1_1 = (__nv_bfloat16*)(smem_pool + 5 * 16384);
    
    __nv_bfloat16* v_smem_0 = (__nv_bfloat16*)(smem_pool + 6 * 16384);
    __nv_bfloat16* v_smem_1 = (__nv_bfloat16*)(smem_pool + 8 * 16384);
    
    __nv_bfloat16* p_smem_0 = (__nv_bfloat16*)(smem_pool + 10 * 16384);
    
    uint64_t* mbar_q = (uint64_t*)(smem_pool + 14 * 16384);
    uint64_t* mbar_k_0 = mbar_q + 1;
    uint64_t* mbar_k_1 = mbar_k_0 + 1;
    uint64_t* mbar_v_0 = mbar_k_1 + 1;
    uint64_t* mbar_v_1 = mbar_v_0 + 1;
    uint64_t* mbar_umma_qk = mbar_v_1 + 1;
    uint64_t* mbar_umma_pv = mbar_umma_qk + 1;
    uint32_t* tmem_base_ptr = (uint32_t*)(mbar_umma_pv + 1);

    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int rank = cluster_rank_fn();
    int q_start = (blockIdx.x / 2) * 256 + rank * 128;
    int c2 = batch_idx * gridDim.y + head_idx;
    
    int phase_q = 0;
    int phase_k[2] = {0, 0};
    int phase_v[2] = {0, 0};
    int phase_umma_qk = 0;
    int phase_umma_pv = 0;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_umma_qk, 1);
        init_smem_barrier_fn(mbar_umma_pv, 1);
        if (rank == 0) {
            init_smem_barrier_fn(mbar_q, 2);
            init_smem_barrier_fn(mbar_k_0, 2);
            init_smem_barrier_fn(mbar_k_1, 2);
            init_smem_barrier_fn(mbar_v_0, 2);
            init_smem_barrier_fn(mbar_v_1, 2);
        }
    }
    cluster_sync_fn();

    if (threadIdx.x < 32) {
        uint32_t addr = (uint32_t)__cvta_generic_to_shared(tmem_base_ptr);
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(addr), "r"(512));
    }
    __syncthreads();
    
    uint32_t tmem_S = *tmem_base_ptr;
    uint32_t tmem_O = tmem_S + 256;

    #pragma unroll
    for (int i = 0; i < 128; i += 32) {
        uint32_t zero[32] = {0};
        tmem_store_32x_fn(tmem_O + i, zero);
    }
    tcgen05_wait_st_fn();

    if (threadIdx.x == 0) {
        if (rank == 0) mbarrier_arrive_and_expect_tx_fn(mbar_q, 32768);
        else mbarrier_arrive_expect_tx_cluster_fn(mbar_q, 32768, 0);
        
        tma_load_3d_cg2_fn(&tma_q, mbar_q, q_smem_0, 0, q_start, c2);
        tma_load_3d_cg2_fn(&tma_q, mbar_q, q_smem_1, 64, q_start, c2);
    }
    
    int num_steps = (S + 255) / 256;
    
    if (num_steps > 0) {
        if (threadIdx.x == 0) {
            if (rank == 0) mbarrier_arrive_and_expect_tx_fn(mbar_k_0, 32768);
            else mbarrier_arrive_expect_tx_cluster_fn(mbar_k_0, 32768, 0);
            
            int k_start = 0 + rank * 128;
            tma_load_3d_cg2_fn(&tma_k, mbar_k_0, k_smem_0_0, 0, k_start, c2);
            tma_load_3d_cg2_fn(&tma_k, mbar_k_0, k_smem_0_1, 64, k_start, c2);
            
            if (rank == 0) mbarrier_arrive_and_expect_tx_fn(mbar_v_0, 32768);
            else mbarrier_arrive_expect_tx_cluster_fn(mbar_v_0, 32768, 0);
            
            int v_d_start = rank * 64;
            tma_load_3d_cg2_fn(&tma_v, mbar_v_0, v_smem_0, v_d_start, 0, c2);
        }
    }
    
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    
    for (int step = 0; step < num_steps; ++step) {
        int buf = step % 2;
        
        uint64_t* mbar_k_buf = (buf == 0) ? mbar_k_0 : mbar_k_1;
        uint64_t* mbar_v_buf = (buf == 0) ? mbar_v_0 : mbar_v_1;
        
        __nv_bfloat16* k_s_0 = (buf == 0) ? k_smem_0_0 : k_smem_1_0;
        __nv_bfloat16* k_s_1 = (buf == 0) ? k_smem_0_1 : k_smem_1_1;
        __nv_bfloat16* v_s = (buf == 0) ? v_smem_0 : v_smem_1;
        
        if (step + 1 < num_steps) {
            if (threadIdx.x == 0) {
                uint64_t* mbar_k_next = (buf == 0) ? mbar_k_1 : mbar_k_0;
                uint64_t* mbar_v_next = (buf == 0) ? mbar_v_1 : mbar_v_0;
                
                __nv_bfloat16* k_s_0_next = (buf == 0) ? k_smem_1_0 : k_smem_0_0;
                __nv_bfloat16* k_s_1_next = (buf == 0) ? k_smem_1_1 : k_smem_0_1;
                __nv_bfloat16* v_s_next = (buf == 0) ? v_smem_1 : v_smem_0;
                
                if (rank == 0) mbarrier_arrive_and_expect_tx_fn(mbar_k_next, 32768);
                else mbarrier_arrive_expect_tx_cluster_fn(mbar_k_next, 32768, 0);
                
                int k_start = (step + 1) * 256 + rank * 128;
                tma_load_3d_cg2_fn(&tma_k, mbar_k_next, k_s_0_next, 0, k_start, c2);
                tma_load_3d_cg2_fn(&tma_k, mbar_k_next, k_s_1_next, 64, k_start, c2);
                
                if (rank == 0) mbarrier_arrive_and_expect_tx_fn(mbar_v_next, 32768);
                else mbarrier_arrive_expect_tx_cluster_fn(mbar_v_next, 32768, 0);
                
                int v_d_start = rank * 64;
                tma_load_3d_cg2_fn(&tma_v, mbar_v_next, v_s_next, v_d_start, (step + 1) * 256, c2);
            }
        }
        
        if (step == 0 && rank == 0) {
            mbarrier_wait_fn(mbar_q, phase_q);
            phase_q ^= 1;
        }
        
        if (rank == 0) {
            mbarrier_wait_fn(mbar_k_buf, phase_k[buf]);
        }
        
        if (rank == 0 && threadIdx.x == 0) {
            #pragma unroll
            for (int k_idx = 0; k_idx < 4; ++k_idx) {
                uint64_t q_desc = make_smem_desc_k_major(q_smem_0 + k_idx * 16);
                uint64_t k_desc = make_smem_desc_k_major(k_s_0 + k_idx * 16);
                uint32_t idesc = make_instr_desc(256, 256, 0, 0);
                int accum = (k_idx == 0) ? 0 : 1;
                umma_f16(tmem_S, q_desc, k_desc, idesc, accum);
            }
            #pragma unroll
            for (int k_idx = 0; k_idx < 4; ++k_idx) {
                uint64_t q_desc = make_smem_desc_k_major(q_smem_1 + k_idx * 16);
                uint64_t k_desc = make_smem_desc_k_major(k_s_1 + k_idx * 16);
                uint32_t idesc = make_instr_desc(256, 256, 0, 0);
                umma_f16(tmem_S, q_desc, k_desc, idesc, 1);
            }
            tcgen05_fence_after_fn();
            uint32_t a = (uint32_t)__cvta_generic_to_shared(mbar_umma_qk);
            asm volatile(
                "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                :: "r"(a), "h"((uint16_t)0x3));
        }
        mbarrier_wait_fn(mbar_umma_qk, phase_umma_qk);
        phase_umma_qk ^= 1;
        
        if (step > 0) {
            mbarrier_wait_fn(mbar_umma_pv, phase_umma_pv);
            phase_umma_pv ^= 1;
        }
        
        float m_curr = -INFINITY;
        #pragma unroll
        for (int col = 0; col < 256; col += 32) {
            uint32_t chunk[32];
            tmem_load_32x_fn(tmem_S + col, chunk);
            tmem_load_fence_fn();
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                float val = __uint_as_float(chunk[j]);
                if (step == num_steps - 1 && step * 256 + col + j >= S) {
                    val = -INFINITY;
                } else {
                    val *= 0.0883883476f;
                }
                m_curr = fmaxf(m_curr, val);
            }
        }
        
        float m_new = fmaxf(m_prev, m_curr);
        float exp_diff = fast_exp2f_fn((m_prev - m_new) * 1.44269504f);
        
        float l_curr = 0.0f;
        int y = threadIdx.x;
        int y_mod_8 = y % 8;
        int4* p_base = (int4*)p_smem_0;
        
        #pragma unroll
        for (int col = 0; col < 256; col += 32) {
            uint32_t chunk[32];
            tmem_load_32x_fn(tmem_S + col, chunk);
            tmem_load_fence_fn();
            
            float p_vals[32];
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                float val = __uint_as_float(chunk[j]);
                if (step == num_steps - 1 && step * 256 + col + j >= S) {
                    val = -INFINITY;
                } else {
                    val *= 0.0883883476f;
                }
                float p = fast_exp2f_fn((val - m_new) * 1.44269504f);
                p_vals[j] = p;
                l_curr += p;
            }
            
            #pragma unroll
            for (int j = 0; j < 32; j += 8) {
                int x = (col + j) / 8;
                int span_idx = x / 8;
                int in_span_x = x % 8;
                int swizzled_x = span_idx * 8 + (in_span_x ^ y_mod_8);
                
                uint32_t p0 = pack_bf16_fast(p_vals[j + 0], p_vals[j + 1]);
                uint32_t p1 = pack_bf16_fast(p_vals[j + 2], p_vals[j + 3]);
                uint32_t p2 = pack_bf16_fast(p_vals[j + 4], p_vals[j + 5]);
                uint32_t p3 = pack_bf16_fast(p_vals[j + 6], p_vals[j + 7]);
                
                p_base[y * 32 + swizzled_x] = make_int4(p0, p1, p2, p3);
            }
        }
        float l_new = exp_diff * l_prev + l_curr;
        
        #pragma unroll
        for (int col = 0; col < 128; col += 32) {
            uint32_t chunk[32];
            tmem_load_32x_fn(tmem_O + col, chunk);
            tmem_load_fence_fn();
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                float val = __uint_as_float(chunk[j]);
                val *= exp_diff;
                chunk[j] = __float_as_uint(val);
            }
            tmem_store_32x_fn(tmem_O + col, chunk);
        }
        tcgen05_wait_st_fn();
        
        cluster_sync_fn();
        
        if (rank == 0) {
            mbarrier_wait_fn(mbar_v_buf, phase_v[buf]);
        }
        
        if (rank == 0 && threadIdx.x == 0) {
            asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
            #pragma unroll
            for (int k_idx = 0; k_idx < 16; ++k_idx) {
                __nv_bfloat16* p_ptr = p_smem_0 + k_idx * 16;
                uint64_t p_desc = make_smem_desc_k_major(p_ptr);
                uint64_t v_desc = make_smem_desc_mn_major_v(v_s + k_idx * 16 * 64);
                uint32_t idesc = make_instr_desc(256, 128, 0, 1);
                int accum = (step == 0 && k_idx == 0) ? 0 : 1;
                umma_f16(tmem_O, p_desc, v_desc, idesc, accum);
            }
            tcgen05_fence_after_fn();
            uint32_t a = (uint32_t)__cvta_generic_to_shared(mbar_umma_pv);
            asm volatile(
                "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                :: "r"(a), "h"((uint16_t)0x3));
        }
        
        m_prev = m_new;
        l_prev = l_new;
        
        phase_k[buf] ^= 1;
        phase_v[buf] ^= 1;
    }
    
    if (num_steps > 0) {
        mbarrier_wait_fn(mbar_umma_pv, phase_umma_pv);
        phase_umma_pv ^= 1;
    }
    
    #pragma unroll
    for (int col = 0; col < 128; col += 32) {
        uint32_t chunk[32];
        tmem_load_32x_fn(tmem_O + col, chunk);
        tmem_load_fence_fn();
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            float val = __uint_as_float(chunk[j]);
            val /= l_prev;
            chunk[j] = __float_as_uint(val);
        }
        
        int y = threadIdx.x;
        int y_mod_8 = y % 8;
        #pragma unroll
        for (int j = 0; j < 32; j += 8) {
            int x = (col + j) / 8;
            int is_second_half = x >= 8;
            int local_x = x % 8;
            int swizzled_x = local_x ^ y_mod_8;
            
            uint32_t p0 = pack_bf16_fast(__uint_as_float(chunk[j+0]), __uint_as_float(chunk[j+1]));
            uint32_t p1 = pack_bf16_fast(__uint_as_float(chunk[j+2]), __uint_as_float(chunk[j+3]));
            uint32_t p2 = pack_bf16_fast(__uint_as_float(chunk[j+4]), __uint_as_float(chunk[j+5]));
            uint32_t p3 = pack_bf16_fast(__uint_as_float(chunk[j+6]), __uint_as_float(chunk[j+7]));
            
            if (!is_second_half) {
                ((int4*)q_smem_0)[y * 8 + swizzled_x] = make_int4(p0, p1, p2, p3);
            } else {
                ((int4*)q_smem_1)[y * 8 + swizzled_x] = make_int4(p0, p1, p2, p3);
            }
        }
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        tma_store_3d_fn(&tma_o, q_smem_0, 0, q_start, c2);
        tma_store_3d_fn(&tma_o, q_smem_1, 64, q_start, c2);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
    }
    
    int seq_idx = q_start + threadIdx.x;
    if (seq_idx < S) {
        lse_out[batch_idx * gridDim.y * S + head_idx * S + seq_idx] = m_prev + logf(l_prev);
    }
    
    cluster_sync_fn();
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(*tmem_base_ptr), "r"(512));
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, 1};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    void* q_ptr = Q.data_ptr();
    void* k_ptr = K.data_ptr();
    void* v_ptr = V.data_ptr();
    void* o_ptr = O.data_ptr();
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_q, tma_k, tma_v, tma_o;
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_q, q_ptr, 128, S, B * H, 64, 128));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_k, k_ptr, 128, S, B * H, 64, 128));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_v, v_ptr, 128, S, B * H, 64, 256));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_o, o_ptr, 128, S, B * H, 64, 128));
    
    int64_t threads = 128;
    int64_t blocks_x = (S + 255) / 256;
    dim3 blocks(blocks_x * 2, H, B);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int smem_bytes = 14 * 16384 + 2048;
    CUDA_CHECK(cudaFuncSetAttribute((const void*)mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = blocks;
    config.blockDim = threads;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_fwd_kernel, tma_q, tma_k, tma_v, tma_o, lse_ptr, static_cast<int>(S)));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}