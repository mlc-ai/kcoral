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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_name;                                      \
        cuGetErrorName(_e, &err_name);                             \
        fprintf(stderr, "CU error %s at %s:%d\n",                  \
                err_name, __FILE__, __LINE__);                     \
        exit(1);                                                   \
    }                                                              \
} while(0)

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

__device__ __forceinline__ void tmem_load_16x_fn(uint32_t col, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                 : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
                   "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                   "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),
                   "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)((addr >> 4) & 0x3FFF);
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)(base_offset) << 49;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major_fn(void* smem_ptr, uint32_t sbo, uint32_t lbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)((addr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)(base_offset) << 49;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, int trans_A, int trans_B) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    if (trans_A) d |= (1u << 15);
    if (trans_B) d |= (1u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
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

struct alignas(128) SharedMemory {
    uint16_t Q_left[128 * 64];
    uint16_t Q_right[128 * 64];
    uint16_t P_left[128 * 64];
    uint16_t P_right[128 * 64];
    uint16_t K_left[2][128 * 64];
    uint16_t K_right[2][128 * 64];
    uint16_t V_left[2][128 * 64];
    uint16_t V_right[2][128 * 64];
    uint64_t bar_Q[1];
    uint64_t bar_K[2];
    uint64_t bar_V[2];
    uint64_t bar_UMMA_S[1];
    uint64_t bar_UMMA_O[1];
    uint32_t tmem_base;
};

__global__ void attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* LSE,
    int S
) {
    extern __shared__ __align__(128) uint8_t smem_buf[];
    SharedMemory* smem = reinterpret_cast<SharedMemory*>(smem_buf);

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;

    if (bx * 128 >= S) return;

    if (tid < 32) {
        tmem_alloc_cg1_fn(&smem->tmem_base, 256);
    }
    __syncthreads();

    uint32_t TMEM_S = smem->tmem_base;
    uint32_t TMEM_O = smem->tmem_base + 128;

    int commit_count_K[2] = {0, 0};
    int commit_count_V[2] = {0, 0};
    int wait_parity_K[2] = {0, 0};
    int wait_parity_V[2] = {0, 0};

    int commit_count_UMMA_S = 0;
    int commit_count_UMMA_O = 0;
    int expected_parity_UMMA_O = 0;

    if (tid == 0) {
        init_smem_barrier_fn(&smem->bar_Q[0], 1);
        init_smem_barrier_fn(&smem->bar_K[0], 1);
        init_smem_barrier_fn(&smem->bar_K[1], 1);
        init_smem_barrier_fn(&smem->bar_V[0], 1);
        init_smem_barrier_fn(&smem->bar_V[1], 1);
        init_smem_barrier_fn(&smem->bar_UMMA_S[0], 1);
        init_smem_barrier_fn(&smem->bar_UMMA_O[0], 1);
    }
    __syncthreads();
    if (tid == 0) fence_smem_barrier_init_fn();
    __syncthreads();

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(&smem->bar_Q[0], 16384 * 2);
        tma_load_3d_fn(&tma_Q, &smem->bar_Q[0], smem->Q_left, 0, bx * 128, by);
        tma_load_3d_fn(&tma_Q, &smem->bar_Q[0], smem->Q_right, 64, bx * 128, by);

        mbarrier_arrive_and_expect_tx_fn(&smem->bar_K[0], 16384 * 2);
        tma_load_3d_fn(&tma_K, &smem->bar_K[0], smem->K_left[0], 0, 0, by);
        tma_load_3d_fn(&tma_K, &smem->bar_K[0], smem->K_right[0], 64, 0, by);

        mbarrier_arrive_and_expect_tx_fn(&smem->bar_V[0], 16384 * 2);
        tma_load_3d_fn(&tma_V, &smem->bar_V[0], smem->V_left[0], 0, 0, by);
        tma_load_3d_fn(&tma_V, &smem->bar_V[0], smem->V_right[0], 64, 0, by);
    }
    wait_parity_K[0] = commit_count_K[0] & 1; commit_count_K[0]++;
    wait_parity_V[0] = commit_count_V[0] & 1; commit_count_V[0]++;

    int num_k_steps = (S + 127) / 128;
    if (num_k_steps > 1) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem->bar_K[1], 16384 * 2);
            tma_load_3d_fn(&tma_K, &smem->bar_K[1], smem->K_left[1], 0, 128, by);
            tma_load_3d_fn(&tma_K, &smem->bar_K[1], smem->K_right[1], 64, 128, by);

            mbarrier_arrive_and_expect_tx_fn(&smem->bar_V[1], 16384 * 2);
            tma_load_3d_fn(&tma_V, &smem->bar_V[1], smem->V_left[1], 0, 128, by);
            tma_load_3d_fn(&tma_V, &smem->bar_V[1], smem->V_right[1], 64, 128, by);
        }
        wait_parity_K[1] = commit_count_K[1] & 1; commit_count_K[1]++;
        wait_parity_V[1] = commit_count_V[1] & 1; commit_count_V[1]++;
    }

    mbarrier_wait_fn(&smem->bar_Q[0], 0);

    float O_reg[128];
    #pragma unroll
    for (int i = 0; i < 128; i++) O_reg[i] = 0.0f;
    float m_prev = -INFINITY;
    float l_prev = 0.0f;

    uint32_t idesc_QK = make_instr_desc_fn(128, 128, 0, 0);
    uint32_t idesc_PV = make_instr_desc_fn(128, 64, 0, 1);

    int r_mod_8 = tid & 7;
    int r_times_128 = tid << 7;

    for (int k = 0; k < num_k_steps; k++) {
        int stage = k % 2;
        int next_stage = (k + 1) % 2;

        if (k > 0) {
            mbarrier_wait_fn(&smem->bar_UMMA_O[0], expected_parity_UMMA_O);
            fence_proxy_async_fn();
            
            #pragma unroll
            for (int c = 0; c < 64; c += 16) {
                uint32_t R0[16], R1[16];
                tmem_load_16x_fn(TMEM_O + c, R0);
                tmem_load_16x_fn(TMEM_O + 64 + c, R1);
                tmem_load_fence_fn();
                #pragma unroll
                for(int i = 0; i < 16; i++) O_reg[c + i] += __uint_as_float(R0[i]);
                #pragma unroll
                for(int i = 0; i < 16; i++) O_reg[64 + c + i] += __uint_as_float(R1[i]);
            }
        }

        mbarrier_wait_fn(&smem->bar_K[stage], wait_parity_K[stage]);
        fence_proxy_async_fn();

        if (tid == 0) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint64_t a_desc = make_smem_desc_sm100_fn(smem->Q_left + i * 16, 1024);
                uint64_t b_desc = make_smem_desc_sm100_fn(smem->K_left[stage] + i * 16, 1024);
                umma_f16_cg1_fn(TMEM_S, a_desc, b_desc, idesc_QK, (i > 0));
            }
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint64_t a_desc = make_smem_desc_sm100_fn(smem->Q_right + i * 16, 1024);
                uint64_t b_desc = make_smem_desc_sm100_fn(smem->K_right[stage] + i * 16, 1024);
                umma_f16_cg1_fn(TMEM_S, a_desc, b_desc, idesc_QK, 1);
            }
            umma_commit_cg1_fn(&smem->bar_UMMA_S[0]);
        }
        int expected_parity_UMMA_S = commit_count_UMMA_S & 1;
        commit_count_UMMA_S++;

        mbarrier_wait_fn(&smem->bar_UMMA_S[0], expected_parity_UMMA_S);
        fence_proxy_async_fn();

        float m_curr = m_prev;
        int k_base = k * 128;

        #pragma unroll
        for (int c = 0; c < 128; c += 32) {
            uint32_t R[32];
            tmem_load_16x_fn(TMEM_S + c, &R[0]);
            tmem_load_16x_fn(TMEM_S + c + 16, &R[16]);
            tmem_load_fence_fn();
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                float s = (k_base + c + i < S) ? __uint_as_float(R[i]) * 0.088388347f : -INFINITY;
                m_curr = fmaxf(m_curr, s);
            }
        }

        float exp_scale = fast_exp2f_fn((m_prev - m_curr) * 1.44269504089f);
        float l_curr = l_prev * exp_scale;

        #pragma unroll
        for (int i = 0; i < 128; i++) O_reg[i] *= exp_scale;

        #pragma unroll
        for (int c = 0; c < 128; c += 32) {
            uint32_t R[32];
            tmem_load_16x_fn(TMEM_S + c, &R[0]);
            tmem_load_16x_fn(TMEM_S + c + 16, &R[16]);
            tmem_load_fence_fn();

            float p[32];
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                float s = (k_base + c + i < S) ? __uint_as_float(R[i]) * 0.088388347f : -INFINITY;
                p[i] = fast_exp2f_fn((s - m_curr) * 1.44269504089f);
                if (k_base + c + i >= S) p[i] = 0.0f;
                l_curr += p[i];
            }

            #pragma unroll
            for (int i = 0; i < 32; i += 8) {
                uint32_t bf01 = pack_bf16_fn(__float_as_uint(p[i+0]), __float_as_uint(p[i+1]));
                uint32_t bf23 = pack_bf16_fn(__float_as_uint(p[i+2]), __float_as_uint(p[i+3]));
                uint32_t bf45 = pack_bf16_fn(__float_as_uint(p[i+4]), __float_as_uint(p[i+5]));
                uint32_t bf67 = pack_bf16_fn(__float_as_uint(p[i+6]), __float_as_uint(p[i+7]));
                int c_idx = c + i;
                if (c_idx < 64) {
                    int swizzled_chunk = r_mod_8 ^ (c_idx >> 3);
                    int byte_offset = r_times_128 + (swizzled_chunk << 4);
                    *(uint4*)((char*)smem->P_left + byte_offset) = make_uint4(bf01, bf23, bf45, bf67);
                } else {
                    int swizzled_chunk = r_mod_8 ^ ((c_idx - 64) >> 3);
                    int byte_offset = r_times_128 + (swizzled_chunk << 4);
                    *(uint4*)((char*)smem->P_right + byte_offset) = make_uint4(bf01, bf23, bf45, bf67);
                }
            }
        }

        __syncthreads();
        fence_proxy_async_fn();
        mbarrier_wait_fn(&smem->bar_V[stage], wait_parity_V[stage]);
        fence_proxy_async_fn();

        if (tid == 0) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint64_t a_desc = make_smem_desc_sm100_fn(smem->P_left + i * 16, 1024);
                uint64_t b_desc = make_smem_desc_mn_major_fn(smem->V_left[stage] + i * 16 * 64, 1024, 16384);
                umma_f16_cg1_fn(TMEM_O, a_desc, b_desc, idesc_PV, (i > 0));
            }
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint64_t a_desc = make_smem_desc_sm100_fn(smem->P_right + i * 16, 1024);
                uint64_t b_desc = make_smem_desc_mn_major_fn(smem->V_left[stage] + (64 + i * 16) * 64, 1024, 16384);
                umma_f16_cg1_fn(TMEM_O, a_desc, b_desc, idesc_PV, 1);
            }
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint64_t a_desc = make_smem_desc_sm100_fn(smem->P_left + i * 16, 1024);
                uint64_t b_desc = make_smem_desc_mn_major_fn(smem->V_right[stage] + i * 16 * 64, 1024, 16384);
                umma_f16_cg1_fn(TMEM_O + 64, a_desc, b_desc, idesc_PV, (i > 0));
            }
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint64_t a_desc = make_smem_desc_sm100_fn(smem->P_right + i * 16, 1024);
                uint64_t b_desc = make_smem_desc_mn_major_fn(smem->V_right[stage] + (64 + i * 16) * 64, 1024, 16384);
                umma_f16_cg1_fn(TMEM_O + 64, a_desc, b_desc, idesc_PV, 1);
            }
            umma_commit_cg1_fn(&smem->bar_UMMA_O[0]);
        }
        expected_parity_UMMA_O = commit_count_UMMA_O & 1;
        commit_count_UMMA_O++;

        if (k + 2 < num_k_steps) {
            if (tid == 0) {
                mbarrier_arrive_and_expect_tx_fn(&smem->bar_K[stage], 16384 * 2);
                tma_load_3d_fn(&tma_K, &smem->bar_K[stage], smem->K_left[stage], 0, (k + 2) * 128, by);
                tma_load_3d_fn(&tma_K, &smem->bar_K[stage], smem->K_right[stage], 64, (k + 2) * 128, by);

                mbarrier_arrive_and_expect_tx_fn(&smem->bar_V[stage], 16384 * 2);
                tma_load_3d_fn(&tma_V, &smem->bar_V[stage], smem->V_left[stage], 0, (k + 2) * 128, by);
                tma_load_3d_fn(&tma_V, &smem->bar_V[stage], smem->V_right[stage], 64, (k + 2) * 128, by);
            }
            wait_parity_K[stage] = commit_count_K[stage] & 1; commit_count_K[stage]++;
            wait_parity_V[stage] = commit_count_V[stage] & 1; commit_count_V[stage]++;
        }

        m_prev = m_curr;
        l_prev = l_curr;
    }

    if (num_k_steps > 0) {
        mbarrier_wait_fn(&smem->bar_UMMA_O[0], expected_parity_UMMA_O);
        fence_proxy_async_fn();
        
        #pragma unroll
        for (int c = 0; c < 64; c += 16) {
            uint32_t R0[16], R1[16];
            tmem_load_16x_fn(TMEM_O + c, R0);
            tmem_load_16x_fn(TMEM_O + 64 + c, R1);
            tmem_load_fence_fn();
            #pragma unroll
            for (int i = 0; i < 16; i++) O_reg[c + i] += __uint_as_float(R0[i]);
            #pragma unroll
            for (int i = 0; i < 16; i++) O_reg[64 + c + i] += __uint_as_float(R1[i]);
        }
    }

    #pragma unroll
    for (int i = 0; i < 128; i++) O_reg[i] /= l_prev;

    #pragma unroll
    for (int c = 0; c < 64; c += 8) {
        uint32_t bf01 = pack_bf16_fn(__float_as_uint(O_reg[c+0]), __float_as_uint(O_reg[c+1]));
        uint32_t bf23 = pack_bf16_fn(__float_as_uint(O_reg[c+2]), __float_as_uint(O_reg[c+3]));
        uint32_t bf45 = pack_bf16_fn(__float_as_uint(O_reg[c+4]), __float_as_uint(O_reg[c+5]));
        uint32_t bf67 = pack_bf16_fn(__float_as_uint(O_reg[c+6]), __float_as_uint(O_reg[c+7]));
        int swizzled_chunk = r_mod_8 ^ (c >> 3);
        int byte_offset = r_times_128 + (swizzled_chunk << 4);
        *(uint4*)((char*)smem->P_left + byte_offset) = make_uint4(bf01, bf23, bf45, bf67);
    }
    #pragma unroll
    for (int c = 0; c < 64; c += 8) {
        uint32_t bf01 = pack_bf16_fn(__float_as_uint(O_reg[64+c+0]), __float_as_uint(O_reg[64+c+1]));
        uint32_t bf23 = pack_bf16_fn(__float_as_uint(O_reg[64+c+2]), __float_as_uint(O_reg[64+c+3]));
        uint32_t bf45 = pack_bf16_fn(__float_as_uint(O_reg[64+c+4]), __float_as_uint(O_reg[64+c+5]));
        uint32_t bf67 = pack_bf16_fn(__float_as_uint(O_reg[64+c+6]), __float_as_uint(O_reg[64+c+7]));
        int swizzled_chunk = r_mod_8 ^ (c >> 3);
        int byte_offset = r_times_128 + (swizzled_chunk << 4);
        *(uint4*)((char*)smem->P_right + byte_offset) = make_uint4(bf01, bf23, bf45, bf67);
    }
    __syncthreads();
    fence_proxy_async_fn();

    if (tid == 0) {
        tma_store_3d_fn(&tma_O, smem->P_left, 0, bx * 128, by);
        tma_store_3d_fn(&tma_O, smem->P_right, 64, bx * 128, by);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    __syncthreads();

    if (bx * 128 + tid < S) {
        LSE[by * S + bx * 128 + tid] = m_prev + logf(l_prev);
    }

    if (tid < 32) {
        tmem_dealloc_cg1_fn(smem->tmem_base, 256);
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, uint32_t box2, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

CUresult create_tma_store_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, uint32_t box2, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    void* q_ptr = Q.data_ptr();
    void* k_ptr = K.data_ptr();
    void* v_ptr = V.data_ptr();
    void* o_ptr = O.data_ptr();
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, q_ptr, 128, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, k_ptr, 128, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, v_ptr, 128, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_store_3d_descriptor_2B(&tma_O, o_ptr, 128, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B));

    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedMemory)));

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = sizeof(SharedMemory);
    config.stream = stream;

    CUDA_CHECK(cudaLaunchKernelEx(&config, attention_kernel, tma_Q, tma_K, tma_V, tma_O, lse_ptr, S));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}