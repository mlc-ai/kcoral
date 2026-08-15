#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

namespace tvm_ffi_mha {

struct SharedStorage {
    __align__(1024) __nv_bfloat16 Q[2][128][64]; // [block][S][D]
    __align__(1024) __nv_bfloat16 K[2][2][128][64]; // [db][block][S][D]
    __align__(1024) __nv_bfloat16 V[2][2][128][64];
    __align__(1024) __nv_bfloat16 P[2][128][64];
    uint64_t bar_Q;
    uint64_t bar_K[2];
    uint64_t bar_V[2];
    uint64_t mbar_mma_S;
    uint64_t mbar_mma_O;
    uint32_t tmem_addr;
};

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
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

__device__ __forceinline__ void tma_store_fence_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (a_major << 15);
    d |= (b_major << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t base_offset = 0) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1
    d |= (uint64_t)(base_offset & 0x7) << 49;
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
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

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a) : "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};" : "=r"(result) : "h"(*reinterpret_cast<uint16_t*>(&a)), "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

#define LOG2E 1.4426950408889634f

__global__ __launch_bounds__(128, 2) void mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* LSE, int S, int D)
{
    int bh_idx = blockIdx.x;
    int q_coord = blockIdx.y * 128; 
    
    if (q_coord >= S) return;

    extern __shared__ char smem_buf[];
    SharedStorage& smem = *(SharedStorage*)smem_buf;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&smem.bar_Q, 1);
        init_smem_barrier_fn(&smem.bar_K[0], 1);
        init_smem_barrier_fn(&smem.bar_K[1], 1);
        init_smem_barrier_fn(&smem.bar_V[0], 1);
        init_smem_barrier_fn(&smem.bar_V[1], 1);
        init_smem_barrier_fn(&smem.mbar_mma_S, 1);
        init_smem_barrier_fn(&smem.mbar_mma_O, 1);
    }
    __syncthreads();

    asm volatile("setmaxnreg.inc.sync.aligned.u32 248;" ::: "memory");

    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&smem.tmem_addr, 256);
    }
    __syncthreads();
    
    uint32_t S_tmem = smem.tmem_addr;
    uint32_t O_tmem = smem.tmem_addr + 128;

    uint32_t phase_Q = 0;
    uint32_t phase_S = 0;
    uint32_t phase_O = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&smem.bar_Q, 32768);
        tma_load_3d_fn(&tma_Q, &smem.bar_Q, smem.Q[0], 0, q_coord, bh_idx);
        tma_load_3d_fn(&tma_Q, &smem.bar_Q, smem.Q[1], 64, q_coord, bh_idx);
        
        mbarrier_arrive_and_expect_tx_fn(&smem.bar_K[0], 32768);
        tma_load_3d_fn(&tma_K, &smem.bar_K[0], smem.K[0][0], 0, 0, bh_idx);
        tma_load_3d_fn(&tma_K, &smem.bar_K[0], smem.K[0][1], 64, 0, bh_idx);

        mbarrier_arrive_and_expect_tx_fn(&smem.bar_V[0], 32768);
        tma_load_3d_fn(&tma_V, &smem.bar_V[0], smem.V[0][0], 0, 0, bh_idx);
        tma_load_3d_fn(&tma_V, &smem.bar_V[0], smem.V[0][1], 64, 0, bh_idx);
    }
    mbarrier_wait_fn(&smem.bar_Q, phase_Q);
    phase_Q ^= 1;

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float scale = 1.0f / sqrtf((float)D);

    uint32_t phase_K[2] = {0, 0};
    uint32_t phase_V[2] = {0, 0};
    int num_kv_blocks = (S + 127) / 128;

    for (int kv_idx = 0; kv_idx < num_kv_blocks; ++kv_idx) {
        int cur_db = kv_idx % 2;
        int next_db = (kv_idx + 1) % 2;
        int k_coord = kv_idx * 128;
        int next_k_coord = (kv_idx + 1) * 128;

        if (threadIdx.x == 0 && kv_idx + 1 < num_kv_blocks) {
            mbarrier_arrive_and_expect_tx_fn(&smem.bar_K[next_db], 32768);
            tma_load_3d_fn(&tma_K, &smem.bar_K[next_db], smem.K[next_db][0], 0, next_k_coord, bh_idx);
            tma_load_3d_fn(&tma_K, &smem.bar_K[next_db], smem.K[next_db][1], 64, next_k_coord, bh_idx);

            mbarrier_arrive_and_expect_tx_fn(&smem.bar_V[next_db], 32768);
            tma_load_3d_fn(&tma_V, &smem.bar_V[next_db], smem.V[next_db][0], 0, next_k_coord, bh_idx);
            tma_load_3d_fn(&tma_V, &smem.bar_V[next_db], smem.V[next_db][1], 64, next_k_coord, bh_idx);
        }

        mbarrier_wait_fn(&smem.bar_K[cur_db], phase_K[cur_db]);
        phase_K[cur_db] ^= 1;

        if (threadIdx.x == 0) {
            tcgen05_fence_after_fn();
            uint32_t idesc_S = make_instr_desc_fn(128, 128, 0, 0); 
            for (int k_step = 0; k_step < 8; ++k_step) {
                int block = k_step / 4;
                int local_k = k_step % 4;
                uint64_t dq = make_smem_desc_sm100_fn(&smem.Q[block][0][local_k * 16], 1, 1024, 0);
                uint64_t dk = make_smem_desc_sm100_fn(&smem.K[cur_db][block][0][local_k * 16], 1, 1024, 0);
                uint32_t accum = (k_step == 0) ? 0 : 1;
                umma_f16_cg1_fn(S_tmem, dq, dk, idesc_S, accum);
            }
            umma_commit_cg1_fn(&smem.mbar_mma_S);
            tcgen05_fence_before_fn();
        }
        
        mbarrier_wait_fn(&smem.mbar_mma_S, phase_S);
        phase_S ^= 1;
        __syncthreads();

        float max_val = -INFINITY;
        int tid = threadIdx.x; 

        #pragma unroll
        for (int chunk = 0; chunk < 8; ++chunk) {
            uint32_t r[16];
            uint32_t c0 = S_tmem + chunk * 16 + 0;
            uint32_t c1 = S_tmem + chunk * 16 + 4;
            uint32_t c2 = S_tmem + chunk * 16 + 8;
            uint32_t c3 = S_tmem + chunk * 16 + 12;

            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(c0));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(c1));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]) : "r"(c2));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]) : "r"(c3));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                float val = __uint_as_float(r[i]) * scale;
                if (k_coord + chunk * 16 + i >= S) val = -INFINITY;
                max_val = fmaxf(max_val, val);
            }
        }

        float m_new = fmaxf(m_prev, max_val);
        float rescale = fast_exp2f_fn((m_prev - m_new) * LOG2E);

        if (kv_idx > 0) {
            #pragma unroll
            for (int chunk = 0; chunk < 8; ++chunk) {
                uint32_t r[16];
                uint32_t c0 = O_tmem + chunk * 16 + 0;
                uint32_t c1 = O_tmem + chunk * 16 + 4;
                uint32_t c2 = O_tmem + chunk * 16 + 8;
                uint32_t c3 = O_tmem + chunk * 16 + 12;

                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(c0));
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(c1));
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]) : "r"(c2));
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]) : "r"(c3));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    r[i] = __float_as_uint(__uint_as_float(r[i]) * rescale);
                }
                
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                    :: "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(c0) : "memory");
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                    :: "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(c1) : "memory");
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                    :: "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(c2) : "memory");
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                    :: "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(c3) : "memory");
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            }
        } else {
            uint32_t zero = 0;
            #pragma unroll
            for (int chunk = 0; chunk < 8; ++chunk) {
                uint32_t c0 = O_tmem + chunk * 16 + 0;
                uint32_t c1 = O_tmem + chunk * 16 + 4;
                uint32_t c2 = O_tmem + chunk * 16 + 8;
                uint32_t c3 = O_tmem + chunk * 16 + 12;

                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                    :: "r"(zero), "r"(zero), "r"(zero), "r"(zero), "r"(c0) : "memory");
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                    :: "r"(zero), "r"(zero), "r"(zero), "r"(zero), "r"(c1) : "memory");
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                    :: "r"(zero), "r"(zero), "r"(zero), "r"(zero), "r"(c2) : "memory");
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                    :: "r"(zero), "r"(zero), "r"(zero), "r"(zero), "r"(c3) : "memory");
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            }
        }

        float sum_val = 0.0f;
        
        #pragma unroll
        for (int chunk = 0; chunk < 8; ++chunk) {
            uint32_t r[16];
            uint32_t c0 = S_tmem + chunk * 16 + 0;
            uint32_t c1 = S_tmem + chunk * 16 + 4;
            uint32_t c2 = S_tmem + chunk * 16 + 8;
            uint32_t c3 = S_tmem + chunk * 16 + 12;

            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(c0));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(c1));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]) : "r"(c2));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]) : "r"(c3));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                float val = __uint_as_float(r[i]) * scale;
                if (k_coord + chunk * 16 + i >= S) val = -INFINITY;
                float p = fast_exp2f_fn((val - m_new) * LOG2E);
                sum_val += p;
                r[i] = __float_as_uint(p);
            }
            
            int block = chunk / 4;
            int local_col = (chunk % 4) * 16;
            
            uint32_t p0 = pack_bf16_fn(r[0], r[1]);
            uint32_t p1 = pack_bf16_fn(r[2], r[3]);
            uint32_t p2 = pack_bf16_fn(r[4], r[5]);
            uint32_t p3 = pack_bf16_fn(r[6], r[7]);
            uint32_t p4 = pack_bf16_fn(r[8], r[9]);
            uint32_t p5 = pack_bf16_fn(r[10], r[11]);
            uint32_t p6 = pack_bf16_fn(r[12], r[13]);
            uint32_t p7 = pack_bf16_fn(r[14], r[15]);
            
            uint32_t base_addr = (uint32_t)__cvta_generic_to_shared(smem.P[block]);
            
            uint32_t x_bytes0 = local_col * 2;
            uint32_t x_chunk0 = x_bytes0 / 16;
            uint32_t x_swizzled0 = ((tid % 8) ^ x_chunk0) * 16;
            uint32_t final_addr0 = base_addr + tid * 128 + x_swizzled0;

            uint32_t x_bytes1 = local_col * 2 + 16;
            uint32_t x_chunk1 = x_bytes1 / 16;
            uint32_t x_swizzled1 = ((tid % 8) ^ x_chunk1) * 16;
            uint32_t final_addr1 = base_addr + tid * 128 + x_swizzled1;
            
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                :: "r"(final_addr0), "r"(p0), "r"(p1), "r"(p2), "r"(p3) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                :: "r"(final_addr1), "r"(p4), "r"(p5), "r"(p6), "r"(p7) : "memory");
        }
        
        float l_new = l_prev * rescale + sum_val;
        m_prev = m_new;
        l_prev = l_new;

        fence_async_shared_fn();
        __syncthreads();

        mbarrier_wait_fn(&smem.bar_V[cur_db], phase_V[cur_db]);
        phase_V[cur_db] ^= 1;

        if (threadIdx.x == 0) {
            tcgen05_fence_after_fn();
            for (int n_idx = 0; n_idx < 2; ++n_idx) {
                uint32_t idesc_P = make_instr_desc_fn(128, 64, 0, 1);
                for (int k_step = 0; k_step < 8; ++k_step) {
                    int block = k_step / 4;
                    int local_k = k_step % 4;
                    uint64_t dp = make_smem_desc_sm100_fn(&smem.P[block][0][local_k * 16], 1, 1024, 0);
                    uint64_t dv = make_smem_desc_sm100_fn(&smem.V[cur_db][n_idx][k_step * 16][0], 16384, 1024, 0);
                    uint32_t accum = 1; 
                    uint32_t d_tmem = O_tmem + n_idx * 64;
                    umma_f16_cg1_fn(d_tmem, dp, dv, idesc_P, accum);
                }
            }
            umma_commit_cg1_fn(&smem.mbar_mma_O);
            tcgen05_fence_before_fn();
        }
        
        mbarrier_wait_fn(&smem.mbar_mma_O, phase_O);
        phase_O ^= 1;
        
        __syncthreads();
    }

    float norm = 1.0f / l_prev;
    int col_tid = threadIdx.x;
    #pragma unroll
    for (int chunk = 0; chunk < 8; ++chunk) {
        uint32_t r[16];
        uint32_t c0 = O_tmem + chunk * 16 + 0;
        uint32_t c1 = O_tmem + chunk * 16 + 4;
        uint32_t c2 = O_tmem + chunk * 16 + 8;
        uint32_t c3 = O_tmem + chunk * 16 + 12;

        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(c0));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(c1));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]) : "r"(c2));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]) : "r"(c3));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float o0 = __uint_as_float(r[i*4+0]) * norm;
            float o1 = __uint_as_float(r[i*4+1]) * norm;
            float o2 = __uint_as_float(r[i*4+2]) * norm;
            float o3 = __uint_as_float(r[i*4+3]) * norm;
            
            int col_offset = (chunk * 4 + i) * 4;
            int block = col_offset / 64;
            int local_col = col_offset % 64;
            
            uint32_t p0 = pack_bf16_fn(__float_as_uint(o0), __float_as_uint(o1));
            uint32_t p1 = pack_bf16_fn(__float_as_uint(o2), __float_as_uint(o3));
            
            uint32_t base_addr = (uint32_t)__cvta_generic_to_shared(smem.P[block]);
            uint32_t x_bytes = local_col * 2;
            uint32_t x_chunk = x_bytes / 16;
            uint32_t x_rem = x_bytes % 16;
            uint32_t x_swizzled = ((col_tid % 8) ^ x_chunk) * 16 + x_rem;
            uint32_t final_addr = base_addr + col_tid * 128 + x_swizzled;
            
            asm volatile("st.shared.v2.b32 [%0], {%1, %2};"
                :: "r"(final_addr), "r"(p0), "r"(p1) : "memory");
        }
    }
    
    tma_store_fence_fn();
    __syncthreads();

    if (threadIdx.x == 0) {
        tma_store_3d_fn(&tma_O, smem.P[0], 0, q_coord, bh_idx);
        tma_store_3d_fn(&tma_O, smem.P[1], 64, q_coord, bh_idx);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }

    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(smem.tmem_addr, 256);
    }

    if (threadIdx.x < 128) {
        int s_idx = q_coord + threadIdx.x;
        if (s_idx < S) {
            LSE[bh_idx * S + s_idx] = m_prev + logf(l_prev);
        }
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
    uint64_t dim0, uint64_t dim1, uint64_t dim2, 
    uint32_t box0, uint32_t box1, uint32_t box2) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3); 

    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    create_tma_3d_descriptor_2B(&tma_Q, q_ptr, 128, S, B * H, 64, 128, 1);
    create_tma_3d_descriptor_2B(&tma_K, k_ptr, 128, S, B * H, 64, 128, 1);
    create_tma_3d_descriptor_2B(&tma_V, v_ptr, 128, S, B * H, 64, 128, 1);
    create_tma_3d_descriptor_2B(&tma_O, o_ptr, 128, S, B * H, 64, 128, 1);

    int64_t threads = 128;
    dim3 grid(B * H, (S + 127) / 128);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage)));
    
    mha_fwd_kernel<<<grid, threads, sizeof(SharedStorage), stream>>>(tma_Q, tma_K, tma_V, tma_O, lse_ptr, S, D);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

} // namespace tvm_ffi_mha