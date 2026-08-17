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

// ----------------------------------------------------------------------
// PTX Wrappers
// ----------------------------------------------------------------------

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void write_to_tmem_packed(const uint32_t* regs, uint32_t tmem_addr) {
    asm volatile("tcgen05.st.sync.aligned.16x32b.b32 [%0], {%1, %2, %3, %4};" 
        :: "r"(tmem_addr), "r"(regs[0]), "r"(regs[1]), "r"(regs[2]), "r"(regs[3]));
}

__device__ __forceinline__ void read_from_tmem_packed(uint32_t* regs, uint32_t tmem_addr) {
    asm volatile("tcgen05.ld.sync.aligned.16x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3]) : "r"(tmem_addr));
}

__device__ __forceinline__ void tmem_commit_cp_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void tmem_wait_cp_fn(uint64_t* bar, uint32_t phase) {
    mbarrier_wait_fn(bar, phase);
}

__device__ __forceinline__ void tmem_commit_mma_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void tmem_wait_mma_fn(uint64_t* bar, uint32_t phase) {
    mbarrier_wait_fn(bar, phase);
}

__device__ __forceinline__ void umma_f16_cta1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_f16_cta1_with_tmem_a(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* smem_ptr, void* base_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    uint64_t lbo = 16; // Encodes to 1, effectively leaving it unused in swizzled mode
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    uint64_t sbo = 1024;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* smem_ptr, void* base_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    uint64_t lbo = 8192; // (64 / 8) * 1024
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    uint64_t sbo = 1024;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// ----------------------------------------------------------------------
// Shared Storage & Kernels
// ----------------------------------------------------------------------

struct __align__(128) SharedStorage {
    __nv_bfloat16 s_Q0[128 * 64];
    __nv_bfloat16 s_Q1[128 * 64];
    __nv_bfloat16 s_K0[128 * 64];
    __nv_bfloat16 s_K1[128 * 64];
    __nv_bfloat16 s_V0[128 * 64];
    __nv_bfloat16 s_V1[128 * 64];
    __nv_bfloat16 s_O0[128 * 64];
    __nv_bfloat16 s_O1[128 * 64];
    __nv_bfloat16 s_dO0[128 * 64];
    __nv_bfloat16 s_dO1[128 * 64];
    __nv_bfloat16 s_PT[128 * 128];
    __nv_bfloat16 s_dPT[128 * 128];
    float s_L[128];
    uint64_t s_bar;
};

__global__ void bwd_dq_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L, __nv_bfloat16* dQ,
    uint32_t S, float scale)
{
    uint32_t m_block = blockIdx.x * 128;
    uint32_t n_block = blockIdx.y * 128;
    uint32_t bh = blockIdx.z;
    uint32_t offset_bh = bh * S;

    if (m_block >= S || n_block >= S) return;

    extern __shared__ __align__(128) char smem_buf[];
    SharedStorage* s = (SharedStorage*)smem_buf;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&s->s_bar, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t phase = 0;

    if (threadIdx.x == 0) {
        uint32_t tx_bytes = 8 * (128 * 64 * sizeof(__nv_bfloat16));
        mbarrier_arrive_and_expect_tx_fn(&s->s_bar, tx_bytes);

        tma_load_2d_fn(&tma_Q, &s->s_bar, s->s_Q0, 0, offset_bh + m_block);
        tma_load_2d_fn(&tma_Q, &s->s_bar, s->s_Q1, 64, offset_bh + m_block);
        tma_load_2d_fn(&tma_K, &s->s_bar, s->s_K0, 0, offset_bh + n_block);
        tma_load_2d_fn(&tma_K, &s->s_bar, s->s_K1, 64, offset_bh + n_block);
        tma_load_2d_fn(&tma_V, &s->s_bar, s->s_V0, 0, offset_bh + n_block);
        tma_load_2d_fn(&tma_V, &s->s_bar, s->s_V1, 64, offset_bh + n_block);
        tma_load_2d_fn(&tma_dO, &s->s_bar, s->s_dO0, 0, offset_bh + m_block);
        tma_load_2d_fn(&tma_dO, &s->s_bar, s->s_dO1, 64, offset_bh + m_block);
    }

    uint32_t tid = threadIdx.x;
    if (tid < 128) {
        s->s_L[tid] = (m_block + tid < S) ? L[offset_bh + m_block + tid] : 0.0f;
    }
    mbarrier_wait_fn(&s->s_bar, phase);
    phase ^= 1;
    __syncthreads();
    fence_proxy_async_fn();

    uint32_t tmem_S, tmem_dP, tmem_Q0, tmem_Q1, tmem_K0, tmem_K1;
    uint32_t tmem_V0, tmem_V1, tmem_O0, tmem_O1, tmem_dO0, tmem_dO1;
    uint32_t tmem_PT, tmem_dPT, tmem_dQ0, tmem_dQ1;

    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S, 128);
        tmem_alloc_fn(&tmem_dP, 128);
        tmem_alloc_fn(&tmem_Q0, 64);
        tmem_alloc_fn(&tmem_Q1, 64);
        tmem_alloc_fn(&tmem_K0, 64);
        tmem_alloc_fn(&tmem_K1, 64);
        tmem_alloc_fn(&tmem_V0, 64);
        tmem_alloc_fn(&tmem_V1, 64);
        tmem_alloc_fn(&tmem_O0, 64);
        tmem_alloc_fn(&tmem_O1, 64);
        tmem_alloc_fn(&tmem_dO0, 64);
        tmem_alloc_fn(&tmem_dO1, 64);
        tmem_alloc_fn(&tmem_PT, 128);
        tmem_alloc_fn(&tmem_dPT, 128);
        tmem_alloc_fn(&tmem_dQ0, 64);
        tmem_alloc_fn(&tmem_dQ1, 64);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&s->s_bar, 4 * (128 * 64 * 2));
        tma_load_2d_fn(&tma_Q, &s->s_bar, s->s_Q0, 0, offset_bh + m_block);
        tma_load_2d_fn(&tma_Q, &s->s_bar, s->s_Q1, 64, offset_bh + m_block);
        tma_load_2d_fn(&tma_K, &s->s_bar, s->s_K0, 0, offset_bh + n_block);
        tma_load_2d_fn(&tma_K, &s->s_bar, s->s_K1, 64, offset_bh + n_block);
    }
    mbarrier_wait_fn(&s->s_bar, phase);
    phase ^= 1;

    // S = Q * K^T
    for (uint32_t step = 0; step < 64; step += 16) {
        uint64_t desc_A_Q0 = make_smem_desc_k_major(s->s_Q0 + step, s->s_Q0);
        uint64_t desc_B_K0 = make_smem_desc_mn_major(s->s_K0 + step * 64, s->s_K0);
        uint32_t idesc = make_instr_desc_fn(128, 64);
        if (threadIdx.x == 0) {
            umma_f16_cta1_fn(tmem_S, desc_A_Q0, desc_B_K0, idesc, 1);
        }
        uint64_t desc_A_Q1 = make_smem_desc_k_major(s->s_Q1 + step, s->s_Q1);
        uint64_t desc_B_K1 = make_smem_desc_mn_major(s->s_K1 + step * 64, s->s_K1);
        if (threadIdx.x == 0) {
            umma_f16_cta1_fn(tmem_S, desc_A_Q1, desc_B_K1, idesc, 1);
        }
    }
    
    tmem_commit_mma_fn(&s->s_bar);
    tmem_wait_mma_fn(&s->s_bar, phase);
    phase ^= 1;

    float S_local[128] = {0};
    for (uint32_t i = 0; i < 4; i++) {
        uint32_t r_S[4];
        read_from_tmem_packed(r_S, tmem_S + (tid * 4 + i * 16));
        uint32_t r_idx[4];
        r_idx[0] = (tid * 4 + i * 16) / 2 % 128;
        r_idx[1] = (tid * 4 + i * 16 + 1) / 2 % 128;
        r_idx[2] = (tid * 4 + i * 16 + 2) / 2 % 128;
        r_idx[3] = (tid * 4 + i * 16 + 3) / 2 % 128;
        
        S_local[r_idx[0]] = __uint_as_float(r_S[0]);
        S_local[r_idx[1]] = __uint_as_float(r_S[1]);
        S_local[r_idx[2]] = __uint_as_float(r_S[2]);
        S_local[r_idx[3]] = __uint_as_float(r_S[3]);
    }

    __nv_bfloat16 P_T_local[128];
    for (uint32_t i = 0; i < 128; i++) {
        float s_val = S_local[i] * scale;
        float l_val = s->s_L[i];
        float p = fast_exp2f_fn((s_val - l_val) * 1.44269504f);
        P_T_local[i] = __float2bfloat16(p);
    }

    for (uint32_t j = 0; j < 128; j += 8) {
        uint32_t swizzled_j = ((tid % 8) ^ (j / 8)) * 8 + (j % 8);
        uint32_t addr = tid * 128 + swizzled_j;
        __nv_bfloat16* ptr = &s->s_PT[addr];
        *(uint4*)ptr = *(uint4*)&P_T_local[j];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&s->s_bar, 4 * (128 * 64 * 2));
        tma_load_2d_fn(&tma_V, &s->s_bar, s->s_V0, 0, offset_bh + n_block);
        tma_load_2d_fn(&tma_V, &s->s_bar, s->s_V1, 64, offset_bh + n_block);
        tma_load_2d_fn(&tma_dO, &s->s_bar, s->s_dO0, 0, offset_bh + m_block);
        tma_load_2d_fn(&tma_dO, &s->s_bar, s->s_dO1, 64, offset_bh + m_block);
    }
    mbarrier_wait_fn(&s->s_bar, phase);
    phase ^= 1;

    // dP = dO * V^T
    for (uint32_t step = 0; step < 64; step += 16) {
        uint64_t desc_A_dO0 = make_smem_desc_k_major(s->s_dO0 + step, s->s_dO0);
        uint64_t desc_B_V0 = make_smem_desc_mn_major(s->s_V0 + step * 64, s->s_V0);
        uint32_t idesc = make_instr_desc_fn(128, 64);
        if (threadIdx.x == 0) {
            umma_f16_cta1_fn(tmem_dP, desc_A_dO0, desc_B_V0, idesc, 1);
        }
        uint64_t desc_A_dO1 = make_smem_desc_k_major(s->s_dO1 + step, s->s_dO1);
        uint64_t desc_B_V1 = make_smem_desc_mn_major(s->s_V1 + step * 64, s->s_V1);
        if (threadIdx.x == 0) {
            umma_f16_cta1_fn(tmem_dP, desc_A_dO1, desc_B_V1, idesc, 1);
        }
    }
    tmem_commit_mma_fn(&s->s_bar);
    tmem_wait_mma_fn(&s->s_bar, phase);
    phase ^= 1;

    float dP_local[128] = {0};
    for (uint32_t i = 0; i < 4; i++) {
        uint32_t r_dP[4];
        read_from_tmem_packed(r_dP, tmem_dP + (tid * 4 + i * 16));
        uint32_t r_idx[4];
        r_idx[0] = (tid * 4 + i * 16) / 2 % 128;
        r_idx[1] = (tid * 4 + i * 16 + 1) / 2 % 128;
        r_idx[2] = (tid * 4 + i * 16 + 2) / 2 % 128;
        r_idx[3] = (tid * 4 + i * 16 + 3) / 2 % 128;

        dP_local[r_idx[0]] = __uint_as_float(r_dP[0]);
        dP_local[r_idx[1]] = __uint_as_float(r_dP[1]);
        dP_local[r_idx[2]] = __uint_as_float(r_dP[2]);
        dP_local[r_idx[3]] = __uint_as_float(r_dP[3]);
    }

    float D[128] = {0};
    for (uint32_t j = 0; j < 128; j += 8) {
        uint32_t swizzled_j = ((tid % 8) ^ (j / 8)) * 8 + (j % 8);
        uint32_t addr = tid * 128 + swizzled_j;
        float p_ij = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&s->s_PT[addr]));
        float dp_ij = dP_local[swizzled_j];
        D[tid] += p_ij * dp_ij;
    }
    __syncthreads();

    __nv_bfloat16 dPT_local[128];
    for (uint32_t i = 0; i < 128; i++) {
        float p_val = fast_exp2f_fn((S_local[i] * scale - s->s_L[tid]) * 1.44269504f);
        float dp_val = dP_local[i];
        float ds = (dp_val - D[tid]) * p_val * scale;
        dPT_local[i] = __float2bfloat16(ds);
    }

    for (uint32_t j = 0; j < 128; j += 8) {
        uint32_t swizzled_j = ((tid % 8) ^ (j / 8)) * 8 + (j % 8);
        uint32_t addr = tid * 128 + swizzled_j;
        __nv_bfloat16* ptr = &s->s_dPT[addr];
        *(uint4*)ptr = *(uint4*)&dPT_local[j];
    }
    __syncthreads();

    // dQ += dS @ V
    for (uint32_t step = 0; step < 64; step += 16) {
        uint64_t desc_A_dST = make_smem_desc_k_major(s->s_dPT + step, s->s_dPT);
        uint64_t desc_B_V0 = make_smem_desc_mn_major(s->s_V0 + step * 64, s->s_V0);
        uint32_t idesc = make_instr_desc_fn(128, 64);
        if (threadIdx.x == 0) {
            umma_f16_cta1_fn(tmem_dQ0, desc_A_dST, desc_B_V0, idesc, 1);
        }
    }
    for (uint32_t step = 0; step < 64; step += 16) {
        uint64_t desc_A_dST1 = make_smem_desc_k_major(s->s_dPT + 64 + step, s->s_dPT);
        uint64_t desc_B_V1 = make_smem_desc_mn_major(s->s_V1 + step * 64, s->s_V1);
        uint32_t idesc = make_instr_desc_fn(128, 64);
        if (threadIdx.x == 0) {
            umma_f16_cta1_fn(tmem_dQ1, desc_A_dST1, desc_B_V1, idesc, 1);
        }
    }
    
    tmem_commit_mma_fn(&s->s_bar);
    tmem_wait_mma_fn(&s->s_bar, phase);
    phase ^= 1;

    float dQ0_out[8] = {0}, dQ1_out[8] = {0};
    for (uint32_t i = 0; i < 4; i++) {
        uint32_t r_dQ0[4], r_dQ1[4];
        read_from_tmem_packed(r_dQ0, tmem_dQ0 + (tid * 4 + i * 16));
        read_from_tmem_packed(r_dQ1, tmem_dQ1 + (tid * 4 + i * 16));
        
        uint32_t r_idx0 = (tid * 4 + i * 16) / 2 % 64;
        uint32_t r_idx1 = (tid * 4 + i * 16 + 1) / 2 % 64;
        
        dQ0_out[r_idx0] += __uint_as_float(r_dQ0[0]);
        dQ0_out[r_idx1] += __uint_as_float(r_dQ0[1]);

        dQ1_out[r_idx0] += __uint_as_float(r_dQ1[0]);
        dQ1_out[r_idx1] += __uint_as_float(r_dQ1[1]);
    }

    for (uint32_t i = 0; i < 8; i+=2) {
        if (m_block + tid < S) {
            uint32_t global_offset0 = (uint64_t)(m_block + tid) * 128 + i;
            uint32_t global_offset1 = (uint64_t)(m_block + tid) * 128 + 64 + i;
            atomicAdd(&dQ[global_offset0], __float2bfloat16(dQ0_out[i]));
            atomicAdd(&dQ[global_offset1], __float2bfloat16(dQ1_out[i]));
        }
    }

    tmem_dealloc_fn(tmem_S, 128);
    tmem_dealloc_fn(tmem_dP, 128);
    tmem_dealloc_fn(tmem_Q0, 64);
    tmem_dealloc_fn(tmem_Q1, 64);
    tmem_dealloc_fn(tmem_K0, 64);
    tmem_dealloc_fn(tmem_K1, 64);
    tmem_dealloc_fn(tmem_V0, 64);
    tmem_dealloc_fn(tmem_V1, 64);
    tmem_dealloc_fn(tmem_O0, 64);
    tmem_dealloc_fn(tmem_O1, 64);
    tmem_dealloc_fn(tmem_dO0, 64);
    tmem_dealloc_fn(tmem_dO1, 64);
    tmem_dealloc_fn(tmem_PT, 128);
    tmem_dealloc_fn(tmem_dPT, 128);
    tmem_dealloc_fn(tmem_dQ0, 64);
    tmem_dealloc_fn(tmem_dQ1, 64);
}

__global__ void bwd_dkv_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L, __nv_bfloat16* dK, __nv_bfloat16* dV,
    uint32_t S, float scale)
{
    uint32_t n_block = blockIdx.x * 128;
    uint32_t bh = blockIdx.z;
    uint32_t offset_bh = bh * S;

    if (n_block >= S) return;

    extern __shared__ __align__(128) char smem_buf[];
    SharedStorage* s = (SharedStorage*)smem_buf;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&s->s_bar, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t phase = 0;
    uint32_t tid = threadIdx.x;

    if (threadIdx.x == 0) {
        uint32_t tx_bytes = 4 * (128 * 64 * 2);
        mbarrier_arrive_and_expect_tx_fn(&s->s_bar, tx_bytes);
        tma_load_2d_fn(&tma_K, &s->s_bar, s->s_K0, 0, offset_bh + n_block);
        tma_load_2d_fn(&tma_K, &s->s_bar, s->s_K1, 64, offset_bh + n_block);
        tma_load_2d_fn(&tma_V, &s->s_bar, s->s_V0, 0, offset_bh + n_block);
        tma_load_2d_fn(&tma_V, &s->s_bar, s->s_V1, 64, offset_bh + n_block);
    }
    mbarrier_wait_fn(&s->s_bar, phase);
    phase ^= 1;
    __syncthreads();

    uint32_t tmem_S, tmem_dP, tmem_Q0, tmem_Q1, tmem_K0, tmem_K1;
    uint32_t tmem_V0, tmem_V1, tmem_O0, tmem_O1, tmem_dO0, tmem_dO1;
    uint32_t tmem_PT, tmem_dPT, tmem_dK0, tmem_dK1, tmem_dV0, tmem_dV1;

    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S, 128);
        tmem_alloc_fn(&tmem_dP, 128);
        tmem_alloc_fn(&tmem_Q0, 64);
        tmem_alloc_fn(&tmem_Q1, 64);
        tmem_alloc_fn(&tmem_K0, 64);
        tmem_alloc_fn(&tmem_K1, 64);
        tmem_alloc_fn(&tmem_V0, 64);
        tmem_alloc_fn(&tmem_V1, 64);
        tmem_alloc_fn(&tmem_O0, 64);
        tmem_alloc_fn(&tmem_O1, 64);
        tmem_alloc_fn(&tmem_dO0, 64);
        tmem_alloc_fn(&tmem_dO1, 64);
        tmem_alloc_fn(&tmem_PT, 128);
        tmem_alloc_fn(&tmem_dPT, 128);
        tmem_alloc_fn(&tmem_dK0, 64);
        tmem_alloc_fn(&tmem_dK1, 64);
        tmem_alloc_fn(&tmem_dV0, 64);
        tmem_alloc_fn(&tmem_dV1, 64);
    }
    __syncthreads();

    float dK0_acc[8] = {0}, dK1_acc[8] = {0};
    float dV0_acc[8] = {0}, dV1_acc[8] = {0};

    for (uint32_t m_block = 0; m_block < S; m_block += 128) {
        if (threadIdx.x == 0) {
            uint32_t tx_bytes = 3 * (128 * 64 * 2);
            mbarrier_arrive_and_expect_tx_fn(&s->s_bar, tx_bytes);
            tma_load_2d_fn(&tma_Q, &s->s_bar, s->s_Q0, 0, offset_bh + m_block);
            tma_load_2d_fn(&tma_Q, &s->s_bar, s->s_Q1, 64, offset_bh + m_block);
            tma_load_2d_fn(&tma_O, &s->s_bar, s->s_O0, 0, offset_bh + m_block);
            tma_load_2d_fn(&tma_O, &s->s_bar, s->s_O1, 64, offset_bh + m_block);
            tma_load_2d_fn(&tma_dO, &s->s_bar, s->s_dO0, 0, offset_bh + m_block);
            tma_load_2d_fn(&tma_dO, &s->s_bar, s->s_dO1, 64, offset_bh + m_block);
        }
        
        if (tid < 128) {
            s->s_L[tid] = (m_block + tid < S) ? L[offset_bh + m_block + tid] : 0.0f;
        }
        mbarrier_wait_fn(&s->s_bar, phase);
        phase ^= 1;
        __syncthreads();
        fence_proxy_async_fn();

        // S = Q * K^T
        for (uint32_t step = 0; step < 64; step += 16) {
            uint64_t desc_A_Q0 = make_smem_desc_k_major(s->s_Q0 + step, s->s_Q0);
            uint64_t desc_B_K0 = make_smem_desc_mn_major(s->s_K0 + step * 64, s->s_K0);
            uint32_t idesc = make_instr_desc_fn(128, 64);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_S, desc_A_Q0, desc_B_K0, idesc, 1);
            }
            uint64_t desc_A_Q1 = make_smem_desc_k_major(s->s_Q1 + step, s->s_Q1);
            uint64_t desc_B_K1 = make_smem_desc_mn_major(s->s_K1 + step * 64, s->s_K1);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_S, desc_A_Q1, desc_B_K1, idesc, 1);
            }
        }
        
        tmem_commit_mma_fn(&s->s_bar);
        tmem_wait_mma_fn(&s->s_bar, phase);
        phase ^= 1;

        float S_local[128] = {0};
        for (uint32_t i = 0; i < 4; i++) {
            uint32_t r_S[4];
            read_from_tmem_packed(r_S, tmem_S + (tid * 4 + i * 16));
            uint32_t r_idx[4];
            r_idx[0] = (tid * 4 + i * 16) / 2 % 128;
            r_idx[1] = (tid * 4 + i * 16 + 1) / 2 % 128;
            r_idx[2] = (tid * 4 + i * 16 + 2) / 2 % 128;
            r_idx[3] = (tid * 4 + i * 16 + 3) / 2 % 128;
            
            S_local[r_idx[0]] = __uint_as_float(r_S[0]);
            S_local[r_idx[1]] = __uint_as_float(r_S[1]);
            S_local[r_idx[2]] = __uint_as_float(r_S[2]);
            S_local[r_idx[3]] = __uint_as_float(r_S[3]);
        }

        __nv_bfloat16 P_T_local[128];
        for (uint32_t i = 0; i < 128; i++) {
            float s_val = S_local[i] * scale;
            float l_val = s->s_L[i];
            float p = fast_exp2f_fn((s_val - l_val) * 1.44269504f);
            P_T_local[i] = __float2bfloat16(p);
        }

        for (uint32_t j = 0; j < 128; j += 8) {
            uint32_t swizzled_j = ((tid % 8) ^ (j / 8)) * 8 + (j % 8);
            uint32_t addr = tid * 128 + swizzled_j;
            __nv_bfloat16* ptr = &s->s_PT[addr];
            *(uint4*)ptr = *(uint4*)&P_T_local[j];
        }
        __syncthreads();

        // dP = dO * V^T
        for (uint32_t step = 0; step < 64; step += 16) {
            uint64_t desc_A_dO0 = make_smem_desc_k_major(s->s_dO0 + step, s->s_dO0);
            uint64_t desc_B_V0 = make_smem_desc_mn_major(s->s_V0 + step * 64, s->s_V0);
            uint32_t idesc = make_instr_desc_fn(128, 64);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_dP, desc_A_dO0, desc_B_V0, idesc, 1);
            }
            uint64_t desc_A_dO1 = make_smem_desc_k_major(s->s_dO1 + step, s->s_dO1);
            uint64_t desc_B_V1 = make_smem_desc_mn_major(s->s_V1 + step * 64, s->s_V1);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_dP, desc_A_dO1, desc_B_V1, idesc, 1);
            }
        }
        tmem_commit_mma_fn(&s->s_bar);
        tmem_wait_mma_fn(&s->s_bar, phase);
        phase ^= 1;

        float dP_local[128] = {0};
        for (uint32_t i = 0; i < 4; i++) {
            uint32_t r_dP[4];
            read_from_tmem_packed(r_dP, tmem_dP + (tid * 4 + i * 16));
            uint32_t r_idx[4];
            r_idx[0] = (tid * 4 + i * 16) / 2 % 128;
            r_idx[1] = (tid * 4 + i * 16 + 1) / 2 % 128;
            r_idx[2] = (tid * 4 + i * 16 + 2) / 2 % 128;
            r_idx[3] = (tid * 4 + i * 16 + 3) / 2 % 128;

            dP_local[r_idx[0]] = __uint_as_float(r_dP[0]);
            dP_local[r_idx[1]] = __uint_as_float(r_dP[1]);
            dP_local[r_idx[2]] = __uint_as_float(r_dP[2]);
            dP_local[r_idx[3]] = __uint_as_float(r_dP[3]);
        }

        float D[128] = {0};
        for (uint32_t j = 0; j < 128; j += 8) {
            uint32_t swizzled_j = ((tid % 8) ^ (j / 8)) * 8 + (j % 8);
            uint32_t addr = tid * 128 + swizzled_j;
            float p_ij = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&s->s_PT[addr]));
            float dp_ij = dP_local[swizzled_j];
            D[tid] += p_ij * dp_ij;
        }
        __syncthreads();

        __nv_bfloat16 dPT_local[128];
        for (uint32_t i = 0; i < 128; i++) {
            float p_val = fast_exp2f_fn((S_local[i] * scale - s->s_L[tid]) * 1.44269504f);
            float dp_val = dP_local[i];
            float ds = (dp_val - D[tid]) * p_val * scale;
            dPT_local[i] = __float2bfloat16(ds);
        }

        for (uint32_t j = 0; j < 128; j += 8) {
            uint32_t swizzled_j = ((tid % 8) ^ (j / 8)) * 8 + (j % 8);
            uint32_t addr = tid * 128 + swizzled_j;
            __nv_bfloat16* ptr = &s->s_dPT[addr];
            *(uint4*)ptr = *(uint4*)&dPT_local[j];
        }
        __syncthreads();

        // dK += dS^T * Q
        for (uint32_t step = 0; step < 64; step += 16) {
            uint64_t desc_A_dST = make_smem_desc_mn_major(s->s_dPT + step * 128, s->s_dPT);
            uint64_t desc_B_Q0 = make_smem_desc_mn_major(s->s_Q0 + step * 64, s->s_Q0);
            uint32_t idesc = make_instr_desc_fn(128, 64);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_dK0, desc_A_dST, desc_B_Q0, idesc, 1);
            }
        }
        for (uint32_t step = 0; step < 64; step += 16) {
            uint64_t desc_A_dST1 = make_smem_desc_mn_major(s->s_dPT + 64 + step * 128, s->s_dPT);
            uint64_t desc_B_Q1 = make_smem_desc_mn_major(s->s_Q1 + step * 64, s->s_Q1);
            uint32_t idesc = make_instr_desc_fn(128, 64);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_dK1, desc_A_dST1, desc_B_Q1, idesc, 1);
            }
        }

        // dV += P_T * dO
        for (uint32_t step = 0; step < 64; step += 16) {
            uint64_t desc_A_PT = make_smem_desc_mn_major(s->s_PT + step * 128, s->s_PT);
            uint64_t desc_B_dO0 = make_smem_desc_mn_major(s->s_dO0 + step * 64, s->s_dO0);
            uint32_t idesc = make_instr_desc_fn(128, 64);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_dV0, desc_A_PT, desc_B_dO0, idesc, 1);
            }
        }
        for (uint32_t step = 0; step < 64; step += 16) {
            uint64_t desc_A_PT1 = make_smem_desc_mn_major(s->s_PT + 64 + step * 128, s->s_PT);
            uint64_t desc_B_dO1 = make_smem_desc_mn_major(s->s_dO1 + step * 64, s->s_dO1);
            uint32_t idesc = make_instr_desc_fn(128, 64);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_dV1, desc_A_PT1, desc_B_dO1, idesc, 1);
            }
        }
        
        tmem_commit_mma_fn(&s->s_bar);
        tmem_wait_mma_fn(&s->s_bar, phase);
        phase ^= 1;

        float dK0_out[8] = {0}, dK1_out[8] = {0};
        float dV0_out[8] = {0}, dV1_out[8] = {0};
        
        for (uint32_t i = 0; i < 4; i++) {
            uint32_t r_dK0[4], r_dK1[4], r_dV0[4], r_dV1[4];
            read_from_tmem_packed(r_dK0, tmem_dK0 + (tid * 4 + i * 16));
            read_from_tmem_packed(r_dK1, tmem_dK1 + (tid * 4 + i * 16));
            read_from_tmem_packed(r_dV0, tmem_dV0 + (tid * 4 + i * 16));
            read_from_tmem_packed(r_dV1, tmem_dV1 + (tid * 4 + i * 16));
            
            uint32_t r_idx0 = (tid * 4 + i * 16) / 2 % 64;
            uint32_t r_idx1 = (tid * 4 + i * 16 + 1) / 2 % 64;
            
            dK0_acc[r_idx0] += __uint_as_float(r_dK0[0]);
            dK0_acc[r_idx1] += __uint_as_float(r_dK0[1]);

            dK1_acc[r_idx0] += __uint_as_float(r_dK1[0]);
            dK1_acc[r_idx1] += __uint_as_float(r_dK1[1]);

            dV0_acc[r_idx0] += __uint_as_float(r_dV0[0]);
            dV0_acc[r_idx1] += __uint_as_float(r_dV0[1]);

            dV1_acc[r_idx0] += __uint_as_float(r_dV1[0]);
            dV1_acc[r_idx1] += __uint_as_float(r_dV1[1]);
        }
    }

    for (uint32_t i = 0; i < 8; i+=2) {
        if (n_block + tid < S) {
            uint32_t global_offset_k0 = (uint64_t)(n_block + tid) * 128 + i;
            dK[global_offset_k0] = __float2bfloat16(dK0_acc[i]);
            
            uint32_t global_offset_k1 = (uint64_t)(n_block + tid) * 128 + 64 + i;
            dK[global_offset_k1] = __float2bfloat16(dK1_acc[i]);

            uint32_t global_offset_v0 = (uint64_t)(n_block + tid) * 128 + i;
            dV[global_offset_v0] = __float2bfloat16(dV0_acc[i]);

            uint32_t global_offset_v1 = (uint64_t)(n_block + tid) * 128 + 64 + i;
            dV[global_offset_v1] = __float2bfloat16(dV1_acc[i]);
        }
    }

    tmem_dealloc_fn(tmem_S, 128);
    tmem_dealloc_fn(tmem_dP, 128);
    tmem_dealloc_fn(tmem_Q0, 64);
    tmem_dealloc_fn(tmem_Q1, 64);
    tmem_dealloc_fn(tmem_K0, 64);
    tmem_dealloc_fn(tmem_K1, 64);
    tmem_dealloc_fn(tmem_V0, 64);
    tmem_dealloc_fn(tmem_V1, 64);
    tmem_dealloc_fn(tmem_O0, 64);
    tmem_dealloc_fn(tmem_O1, 64);
    tmem_dealloc_fn(tmem_dO0, 64);
    tmem_dealloc_fn(tmem_dO1, 64);
    tmem_dealloc_fn(tmem_PT, 128);
    tmem_dealloc_fn(tmem_dPT, 128);
    tmem_dealloc_fn(tmem_dK0, 64);
    tmem_dealloc_fn(tmem_dK1, 64);
    tmem_dealloc_fn(tmem_dV0, 64);
    tmem_dealloc_fn(tmem_dV1, 64);
}

namespace tvm_ffi_mha_bwd_d128 {

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
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
         
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3); 

    const __nv_bfloat16* q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* o_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* do_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* l_ptr = static_cast<const float*>(L.data_ptr());

    __nv_bfloat16* dq_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dk_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dv_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    create_tma_2d_descriptor_2B(&tma_Q, (void*)q_ptr, d, B*H*S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, (void*)k_ptr, d, B*H*S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, (void*)v_ptr, d, B*H*S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_O, (void*)o_ptr, d, B*H*S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dO, (void*)do_ptr, d, B*H*S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    float scale = 1.0f / sqrtf((float)d);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int smem_size = sizeof(SharedStorage);
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    bwd_dq_kernel<<<dim3((S + 127) / 128, (S + 127) / 128, B * H), 128, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO, l_ptr, dq_ptr, S, scale
    );
    bwd_dkv_kernel<<<dim3((S + 127) / 128, (S + 127) / 128, B * H), 128, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO, l_ptr, dk_ptr, dv_ptr, S, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd_d128::run);

} // namespace tvm_ffi_mha_bwd_d128