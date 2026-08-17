#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

#define K_MAJOR 0
#define MN_MAJOR 1

namespace tvm_ffi_mha {

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_3d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
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

__device__ __forceinline__ void umma_f16_tmemA_cg1_fn(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void tcgen05_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, int a_major, int b_major) {
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

__device__ __forceinline__ uint64_t advance_desc_k_major(uint64_t desc, int k_elements) {
    return desc + 2; 
}

__device__ __forceinline__ uint64_t advance_desc_mn_major(uint64_t desc) {
    return desc + 128; // 2048 bytes
}

__device__ __forceinline__ void store_swizzled_128B(void* base, int row, int col_elements, uint32_t val) {
    uint32_t col_bytes = col_elements * 2;
    uint32_t chunk_x = col_bytes / 16;
    uint32_t chunk_y = row % 8;
    uint32_t sx = chunk_y ^ chunk_x;
    uint32_t swizzled_offset = (row * 128) + (sx * 16) + (col_bytes % 16);
    *(uint32_t*)((char*)base + swizzled_offset) = val;
}

__global__ void __launch_bounds__(128) mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* __restrict__ LSE,
    int B, int H, int S
) {
    setmaxnreg_inc_sync_fn<248>();

    int q_offset = blockIdx.x * 128;
    int batch_head_idx = blockIdx.y;
    if (q_offset >= S) return;

    extern __shared__ __align__(1024) char smem[];

    __nv_bfloat16* Q_L = (__nv_bfloat16*)(smem + 0);
    __nv_bfloat16* Q_R = (__nv_bfloat16*)(smem + 16384);
    
    __nv_bfloat16* K_L[2];
    K_L[0] = (__nv_bfloat16*)(smem + 32768);
    K_L[1] = (__nv_bfloat16*)(smem + 65536);

    __nv_bfloat16* K_R[2];
    K_R[0] = (__nv_bfloat16*)(smem + 49152);
    K_R[1] = (__nv_bfloat16*)(smem + 81920);

    __nv_bfloat16* V_L[2];
    V_L[0] = (__nv_bfloat16*)(smem + 98304);
    V_L[1] = (__nv_bfloat16*)(smem + 131072);

    __nv_bfloat16* V_R[2];
    V_R[0] = (__nv_bfloat16*)(smem + 114688);
    V_R[1] = (__nv_bfloat16*)(smem + 147456);

    uint32_t* smem_tmem_O = (uint32_t*)(smem + 163840);
    uint32_t* smem_tmem_P = (uint32_t*)(smem + 163844);

    uint64_t* mbar_Q = (uint64_t*)(smem + 163848);
    uint64_t* mbar_KV = (uint64_t*)(smem + 163856);
    uint64_t* mbar_commit = (uint64_t*)(smem + 163872);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(&mbar_KV[0], 1);
        init_smem_barrier_fn(&mbar_KV[1], 1);
        init_smem_barrier_fn(mbar_commit, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (threadIdx.y == 0) {
        tmem_alloc_cg1_fn(smem_tmem_O, 128);
        tmem_alloc_cg1_fn(smem_tmem_P, 128);
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
        tma_load_3d_fn(&tma_Q, mbar_Q, Q_L, 0, q_offset, batch_head_idx);
        tma_load_3d_fn(&tma_Q, mbar_Q, Q_R, 64, q_offset, batch_head_idx);

        mbarrier_arrive_and_expect_tx_fn(&mbar_KV[0], 65536);
        tma_load_3d_fn(&tma_K, &mbar_KV[0], K_L[0], 0, 0, batch_head_idx);
        tma_load_3d_fn(&tma_K, &mbar_KV[0], K_R[0], 64, 0, batch_head_idx);
        tma_load_3d_fn(&tma_V, &mbar_KV[0], V_L[0], 0, 0, batch_head_idx);
        tma_load_3d_fn(&tma_V, &mbar_KV[0], V_R[0], 64, 0, batch_head_idx);
    }

    uint32_t tmem_O = *smem_tmem_O;
    uint32_t tmem_P = *smem_tmem_P;

    float thread_m = -1e38f;
    float thread_l = 0.0f;
    int kv_phase[2] = {0, 0};
    int commit_phase = 0;

    int q_valid = min(128, S - q_offset);
    int row = threadIdx.y * 32 + threadIdx.x;
    float scale = 0.0883883476f; 

    uint32_t idesc_P = make_instr_desc_fn(128, 128, K_MAJOR, K_MAJOR);
    uint32_t idesc_O = make_instr_desc_fn(128, 64, K_MAJOR, MN_MAJOR);

    uint32_t r[128];

    for (int kv_seq = 0; kv_seq < S; kv_seq += 128) {
        int buf = (kv_seq / 128) % 2;
        int next_buf = 1 - buf;

        if (kv_seq + 128 < S) {
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_KV[next_buf], 65536);
                tma_load_3d_fn(&tma_K, &mbar_KV[next_buf], K_L[next_buf], 0, kv_seq + 128, batch_head_idx);
                tma_load_3d_fn(&tma_K, &mbar_KV[next_buf], K_R[next_buf], 64, kv_seq + 128, batch_head_idx);
                tma_load_3d_fn(&tma_V, &mbar_KV[next_buf], V_L[next_buf], 0, kv_seq + 128, batch_head_idx);
                tma_load_3d_fn(&tma_V, &mbar_KV[next_buf], V_R[next_buf], 64, kv_seq + 128, batch_head_idx);
            }
        }

        if (kv_seq == 0) {
            mbarrier_wait_fn(mbar_Q, 0);
        }
        mbarrier_wait_fn(&mbar_KV[buf], kv_phase[buf]);

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            tcgen05_fence_after_fn();
            uint64_t desc_Q_L = make_smem_desc_sm100_fn(Q_L, 1, 1024);
            uint64_t desc_K_L = make_smem_desc_sm100_fn(K_L[buf], 1, 1024);
            
            #pragma unroll
            for (int k = 0; k < 64; k += 16) {
                int accum = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_P, desc_Q_L, desc_K_L, idesc_P, accum);
                desc_Q_L = advance_desc_k_major(desc_Q_L, 16);
                desc_K_L = advance_desc_k_major(desc_K_L, 16);
            }

            uint64_t desc_Q_R = make_smem_desc_sm100_fn(Q_R, 1, 1024);
            uint64_t desc_K_R = make_smem_desc_sm100_fn(K_R[buf], 1, 1024);
            
            #pragma unroll
            for (int k = 0; k < 64; k += 16) {
                umma_f16_cg1_fn(tmem_P, desc_Q_R, desc_K_R, idesc_P, 1);
                desc_Q_R = advance_desc_k_major(desc_Q_R, 16);
                desc_K_R = advance_desc_k_major(desc_K_R, 16);
            }
            tcgen05_commit_cg1_fn(mbar_commit);
        }
        mbarrier_wait_fn(mbar_commit, commit_phase);
        commit_phase ^= 1;
        tcgen05_fence_after_fn();

        #pragma unroll
        for (int col = 0; col < 128; col += 4) {
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r[col]),"=r"(r[col+1]),"=r"(r[col+2]),"=r"(r[col+3]) : "r"(tmem_P + col));
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

        float row_max = -1e38f;
        int kv_valid = min(128, S - kv_seq);

        #pragma unroll
        for (int col = 0; col < 128; col += 4) {
            float f0 = __uint_as_float(r[col]) * scale;
            float f1 = __uint_as_float(r[col+1]) * scale;
            float f2 = __uint_as_float(r[col+2]) * scale;
            float f3 = __uint_as_float(r[col+3]) * scale;
            
            if (row >= q_valid) {
                f0 = -1e38f; f1 = -1e38f; f2 = -1e38f; f3 = -1e38f;
            } else {
                if (col + 0 >= kv_valid) f0 = -1e38f;
                if (col + 1 >= kv_valid) f1 = -1e38f;
                if (col + 2 >= kv_valid) f2 = -1e38f;
                if (col + 3 >= kv_valid) f3 = -1e38f;
            }
            
            r[col+0] = *(uint32_t*)&f0;
            r[col+1] = *(uint32_t*)&f1;
            r[col+2] = *(uint32_t*)&f2;
            r[col+3] = *(uint32_t*)&f3;
            
            row_max = max(row_max, f0); row_max = max(row_max, f1);
            row_max = max(row_max, f2); row_max = max(row_max, f3);
        }

        float m_new = max(thread_m, row_max);
        float exp_diff = exp2f((thread_m - m_new) * 1.44269504f);

        float row_sum = 0.0f;
        
        #pragma unroll
        for (int col = 0; col < 128; col++) {
            float p = __uint_as_float(r[col]);
            float p_exp = exp2f((p - m_new) * 1.44269504f);
            if (col >= kv_valid || row >= q_valid) p_exp = 0.0f;
            r[col] = *(uint32_t*)&p_exp;
            row_sum += p_exp;
        }

        float l_new = thread_l * exp_diff + row_sum;
        thread_m = m_new;
        thread_l = l_new;

        #pragma unroll
        for (int col = 0; col < 64; col += 2) {
            uint32_t packed_L = pack_bf16_fn(r[col*2], r[col*2+1]);
            uint32_t packed_R = pack_bf16_fn(r[col*2+2], r[col*2+3]);
            asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32 [%2], {%0,%1};"
                         :: "r"(packed_L), "r"(packed_R), "r"(tmem_P + col));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");

        if (kv_seq > 0) {
            #pragma unroll
            for (int col = 0; col < 128; col += 4) {
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(r[col]),"=r"(r[col+1]),"=r"(r[col+2]),"=r"(r[col+3]) : "r"(tmem_O + col));
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            #pragma unroll
            for (int col = 0; col < 128; col += 4) {
                float o0 = __uint_as_float(r[col]) * exp_diff;
                float o1 = __uint_as_float(r[col+1]) * exp_diff;
                float o2 = __uint_as_float(r[col+2]) * exp_diff;
                float o3 = __uint_as_float(r[col+3]) * exp_diff;
                r[col] = *(uint32_t*)&o0;
                r[col+1] = *(uint32_t*)&o1;
                r[col+2] = *(uint32_t*)&o2;
                r[col+3] = *(uint32_t*)&o3;
            }
            
            #pragma unroll
            for (int col = 0; col < 128; col += 4) {
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                             :: "r"(r[col]), "r"(r[col+1]), "r"(r[col+2]), "r"(r[col+3]), "r"(tmem_O + col));
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }

        tcgen05_fence_before_fn();
        fence_async_shared_fn();
        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            tcgen05_fence_after_fn();
            
            uint32_t tmem_a_L = tmem_P;
            uint64_t desc_V_L = make_smem_desc_sm100_fn(V_L[buf], 2048, 1024);
            
            #pragma unroll
            for (int k = 0; k < 64; k += 16) {
                int accum = (kv_seq == 0 && k == 0) ? 0 : 1;
                umma_f16_tmemA_cg1_fn(tmem_O, tmem_a_L, desc_V_L, idesc_O, accum);
                tmem_a_L += 8;
                desc_V_L = advance_desc_mn_major(desc_V_L);
            }
            
            uint32_t tmem_a_R = tmem_P + 32;
            
            #pragma unroll
            for (int k = 64; k < 128; k += 16) {
                umma_f16_tmemA_cg1_fn(tmem_O, tmem_a_R, desc_V_L, idesc_O, 1);
                tmem_a_R += 8;
                desc_V_L = advance_desc_mn_major(desc_V_L);
            }

            uint32_t tmem_a_L2 = tmem_P;
            uint64_t desc_V_R = make_smem_desc_sm100_fn(V_R[buf], 2048, 1024);
            
            #pragma unroll
            for (int k = 0; k < 64; k += 16) {
                int accum = (kv_seq == 0 && k == 0) ? 0 : 1;
                umma_f16_tmemA_cg1_fn(tmem_O + 64, tmem_a_L2, desc_V_R, idesc_O, accum);
                tmem_a_L2 += 8;
                desc_V_R = advance_desc_mn_major(desc_V_R);
            }
            
            uint32_t tmem_a_R2 = tmem_P + 32;
            
            #pragma unroll
            for (int k = 64; k < 128; k += 16) {
                umma_f16_tmemA_cg1_fn(tmem_O + 64, tmem_a_R2, desc_V_R, idesc_O, 1);
                tmem_a_R2 += 8;
                desc_V_R = advance_desc_mn_major(desc_V_R);
            }
            
            tcgen05_commit_cg1_fn(mbar_commit);
        }
        mbarrier_wait_fn(mbar_commit, commit_phase);
        commit_phase ^= 1;

        kv_phase[buf] ^= 1;
    }

    tcgen05_fence_after_fn();
    
    #pragma unroll
    for (int col = 0; col < 128; col += 4) {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r[col]),"=r"(r[col+1]),"=r"(r[col+2]),"=r"(r[col+3]) : "r"(tmem_O + col));
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

    float l_div = thread_l > 0.0f ? (1.0f / thread_l) : 0.0f;
    
    #pragma unroll
    for (int col = 0; col < 64; col += 2) {
        float o0 = __uint_as_float(r[col]) * l_div;
        float o1 = __uint_as_float(r[col+1]) * l_div;
        uint32_t packed_L = pack_bf16_fn(*(uint32_t*)&o0, *(uint32_t*)&o1);
        store_swizzled_128B(Q_L, row, col, packed_L);
        
        float o2 = __uint_as_float(r[col+64]) * l_div;
        float o3 = __uint_as_float(r[col+65]) * l_div;
        uint32_t packed_R = pack_bf16_fn(*(uint32_t*)&o2, *(uint32_t*)&o3);
        store_swizzled_128B(Q_R, row, col, packed_R);
    }

    tcgen05_fence_before_fn();
    fence_async_shared_fn();
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        tma_store_3d_fn(&tma_O, Q_L, 0, q_offset, batch_head_idx);
        tma_store_3d_fn(&tma_O, Q_R, 64, q_offset, batch_head_idx);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    __syncthreads();

    if (row < q_valid) {
        LSE[batch_head_idx * S + q_offset + row] = thread_m + logf(thread_l);
    }

    if (threadIdx.y == 0) {
        tmem_dealloc_cg1_fn(tmem_O, 128);
        tmem_dealloc_cg1_fn(tmem_P, 128);
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_mid_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_mid_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[3] = {gmem_inner_dim, gmem_mid_dim, gmem_outer_dim};
    cuuint64_t globalStrides[2] = {gmem_inner_dim * 2, gmem_inner_dim * gmem_mid_dim * 2};
    cuuint32_t boxDim[3] = {smem_inner_dim, smem_mid_dim, smem_outer_dim};
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
        l2Promotion,
        oobFill
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    create_tma_3d_descriptor_2B(&tma_Q, q_ptr, D, S, B*H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_3d_descriptor_2B(&tma_K, k_ptr, D, S, B*H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_3d_descriptor_2B(&tma_V, v_ptr, D, S, B*H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_3d_descriptor_2B(&tma_O, o_ptr, D, S, B*H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    dim3 grid((S + 127) / 128, B * H, 1);
    dim3 block(32, 4, 1);
    size_t smem_size = 163880; 

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    mha_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, tma_O, lse_ptr, B, H, S);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

}  // namespace tvm_ffi_mha