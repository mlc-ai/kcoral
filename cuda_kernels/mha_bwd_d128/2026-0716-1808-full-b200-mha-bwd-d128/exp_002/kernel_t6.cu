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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
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

__device__ __forceinline__ void read_from_tmem_packed(uint32_t* regs, uint32_t tmem_addr) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3]) : "r"(tmem_addr));
}

__device__ __forceinline__ void tmem_commit_cp_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void tmem_wait_cp_fn(uint64_t* bar, uint32_t phase) {
    mbarrier_wait_fn(bar, phase);
}

__device__ __forceinline__ void tmem_commit_mma_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(a));
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

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* smem_ptr, void* base_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t base_addr = (uint32_t)__cvta_generic_to_shared(base_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    uint64_t lbo = 16; 
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    uint64_t sbo = 1024;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    uint64_t base = (base_addr >> 7) & 7;
    d |= (base << 49);
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* smem_ptr, void* base_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t base_addr = (uint32_t)__cvta_generic_to_shared(base_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    uint64_t lbo = 8192; 
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    uint64_t sbo = 1024;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    uint64_t base = (base_addr >> 7) & 7;
    d |= (base << 49);
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t a_transpose, uint32_t b_transpose) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    d |= (a_transpose << 15);
    d |= (b_transpose << 16);
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

struct __align__(1024) SharedStorage {
    __nv_bfloat16 s_Q0[128 * 64];
    __nv_bfloat16 s_Q1[128 * 64];
    __nv_bfloat16 s_K0[128 * 64];
    __nv_bfloat16 s_K1[128 * 64];
    __nv_bfloat16 s_V0[128 * 64];
    __nv_bfloat16 s_V1[128 * 64];
    
    __nv_bfloat16 s_PT[128 * 128];  
    __nv_bfloat16 s_dPT[128 * 128]; 
    
    __nv_bfloat16 s_dO0[128 * 64];
    __nv_bfloat16 s_dO1[128 * 64];
    
    float s_L[128];
    uint64_t s_bar;
};

__global__ void convert_fp32_to_bf16(__nv_bfloat16* out, const float* in, uint64_t num_elements) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        out[idx] = __float2bfloat16(in[idx]);
    }
}

__global__ void bwd_dq_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L, float* dQ_workspace,
    uint32_t S, float scale)
{
    uint32_t m_block = blockIdx.x * 128;
    uint32_t bh = blockIdx.z;
    uint32_t offset_bh = bh * S;

    if (m_block >= S) return;

    extern __shared__ __align__(1024) char smem_buf[];
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

        tma_load_3d_fn(&tma_Q, &s->s_bar, s->s_Q0, 0, offset_bh + m_block, bh);
        tma_load_3d_fn(&tma_Q, &s->s_bar, s->s_Q1, 64, offset_bh + m_block, bh);
        tma_load_3d_fn(&tma_dO, &s->s_bar, s->s_dO0, 0, offset_bh + m_block, bh);
        tma_load_3d_fn(&tma_dO, &s->s_bar, s->s_dO1, 64, offset_bh + m_block, bh);
    }

    if (tid < 128) {
        s->s_L[tid] = (m_block + tid < S) ? L[offset_bh + m_block + tid] : 0.0f;
    }
    mbarrier_wait_fn(&s->s_bar, phase);
    phase ^= 1;
    __syncthreads();
    fence_proxy_async_fn();

    uint32_t tmem_S, tmem_dP, tmem_dQ0, tmem_dQ1;

    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S, 128);
        tmem_alloc_fn(&tmem_dP, 128);
        tmem_alloc_fn(&tmem_dQ0, 64);
        tmem_alloc_fn(&tmem_dQ1, 64);
    }
    __syncthreads();

    float dQ_acc0[64] = {0}, dQ_acc1[64] = {0};

    for (uint32_t n_block = 0; n_block < S; n_block += 128) {
        if (threadIdx.x == 0) {
            uint32_t tx_bytes = 4 * (128 * 64 * 2);
            mbarrier_arrive_and_expect_tx_fn(&s->s_bar, tx_bytes);

            tma_load_3d_fn(&tma_K, &s->s_bar, s->s_K0, 0, offset_bh + n_block, bh);
            tma_load_3d_fn(&tma_K, &s->s_bar, s->s_K1, 64, offset_bh + n_block, bh);
            tma_load_3d_fn(&tma_V, &s->s_bar, s->s_V0, 0, offset_bh + n_block, bh);
            tma_load_3d_fn(&tma_V, &s->s_bar, s->s_V1, 64, offset_bh + n_block, bh);
        }
        
        mbarrier_wait_fn(&s->s_bar, phase);
        phase ^= 1;
        __syncthreads();
        fence_proxy_async_fn();

        uint32_t accum = 0;
        // S = Q @ K^T
        for (uint32_t step = 0; step < 64; step += 16) {
            uint64_t desc_A = make_smem_desc_k_major(&s->s_Q0[tid * 64 + step], s->s_Q0);
            uint64_t desc_B = make_smem_desc_k_major(&s->s_K0[tid * 64 + step], s->s_K0);
            uint32_t idesc = make_instr_desc_fn(128, 64, 0, 0);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_S, desc_A, desc_B, idesc, accum);
            }
            
            uint64_t desc_A1 = make_smem_desc_k_major(&s->s_Q1[tid * 64 + step], s->s_Q1);
            uint64_t desc_B1 = make_smem_desc_k_major(&s->s_K1[tid * 64 + step], s->s_K1);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_S + (64 << 2), desc_A1, desc_B1, idesc, accum);
            }
            accum = 1;
        }
        
        tmem_commit_mma_fn(&s->s_bar);
        tmem_wait_mma_fn(&s->s_bar, phase);
        phase ^= 1;
        __syncthreads();
        
        float S_local[128] = {0};
        for (uint32_t c = 0; c < 128; c += 4) {
            uint32_t regs[4];
            read_from_tmem_packed(regs, tmem_S + (tid << 16) + c);
            S_local[c] = __uint_as_float(regs[0]);
            S_local[c+1] = __uint_as_float(regs[1]);
            S_local[c+2] = __uint_as_float(regs[2]);
            S_local[c+3] = __uint_as_float(regs[3]);
        }

        for (uint32_t half = 0; half < 2; half++) {
            uint32_t c_start = half * 64;
            for (uint32_t cc = 0; cc < 64; cc++) {
                uint32_t c = c_start + cc;
                float s_val = S_local[c] * scale;
                float l_val = s->s_L[tid];
                float p = fast_exp2f_fn((s_val - l_val) * 1.44269504f);
                
                uint32_t swizzled_c = ((tid % 8) ^ (cc / 8)) * 8 + (cc % 8);
                if (half == 0) {
                    s->s_PT[tid * 128 + swizzled_c] = __float2bfloat16(p);
                } else {
                    s->s_PT[tid * 128 + 64 + swizzled_c] = __float2bfloat16(p);
                }
            }
        }
        __syncthreads();

        accum = 0;
        // dP = dO @ V^T
        for (uint32_t step = 0; step < 64; step += 16) {
            uint64_t desc_A = make_smem_desc_k_major(&s->s_dO0[tid * 64 + step], s->s_dO0);
            uint64_t desc_B = make_smem_desc_k_major(&s->s_V0[tid * 64 + step], s->s_V0);
            uint32_t idesc = make_instr_desc_fn(128, 64, 0, 0);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_dP, desc_A, desc_B, idesc, accum);
            }
            
            uint64_t desc_A1 = make_smem_desc_k_major(&s->s_dO1[tid * 64 + step], s->s_dO1);
            uint64_t desc_B1 = make_smem_desc_k_major(&s->s_V1[tid * 64 + step], s->s_V1);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_dP + (64 << 2), desc_A1, desc_B1, idesc, accum);
            }
            accum = 1;
        }
        
        tmem_commit_mma_fn(&s->s_bar);
        tmem_wait_mma_fn(&s->s_bar, phase);
        phase ^= 1;
        __syncthreads();

        float dP_local[128] = {0};
        for (uint32_t c = 0; c < 128; c += 4) {
            uint32_t regs[4];
            read_from_tmem_packed(regs, tmem_dP + (tid << 16) + c);
            dP_local[c] = __uint_as_float(regs[0]);
            dP_local[c+1] = __uint_as_float(regs[1]);
            dP_local[c+2] = __uint_as_float(regs[2]);
            dP_local[c+3] = __uint_as_float(regs[3]);
        }

        float D = 0;
        for (uint32_t half = 0; half < 2; half++) {
            uint32_t c_start = half * 64;
            for (uint32_t cc = 0; cc < 64; cc++) {
                uint32_t swizzled_c = ((tid % 8) ^ (cc / 8)) * 8 + (cc % 8);
                float p = __bfloat162float(half == 0 ? s->s_PT[tid * 128 + swizzled_c] : s->s_PT[tid * 128 + 64 + swizzled_c]);
                float dp = dP_local[c_start + cc];
                D += p * dp;
            }
        }

        for (uint32_t half = 0; half < 2; half++) {
            uint32_t c_start = half * 64;
            for (uint32_t cc = 0; cc < 64; cc++) {
                uint32_t swizzled_c = ((tid % 8) ^ (cc / 8)) * 8 + (cc % 8);
                float p = __bfloat162float(half == 0 ? s->s_PT[tid * 128 + swizzled_c] : s->s_PT[tid * 128 + 64 + swizzled_c]);
                float dp = dP_local[c_start + cc];
                float ds = (dp - D) * p * scale;
                
                if (half == 0) {
                    s->s_dPT[tid * 128 + swizzled_c] = __float2bfloat16(ds);
                } else {
                    s->s_dPT[tid * 128 + 64 + swizzled_c] = __float2bfloat16(ds);
                }
            }
        }
        __syncthreads();

        // dQ += dS @ V
        for (uint32_t step = 0; step < 64; step += 16) {
            uint64_t desc_A = make_smem_desc_k_major(&s->s_dPT[tid * 128 + step], s->s_dPT);
            uint64_t desc_B = make_smem_desc_mn_major(&s->s_V0[step * 4096], s->s_V0);
            uint32_t idesc = make_instr_desc_fn(128, 64, 0, 1);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_dQ0, desc_A, desc_B, idesc, 1);
            }
            
            uint64_t desc_A1 = make_smem_desc_k_major(&s->s_dPT[tid * 128 + 64 + step], s->s_dPT);
            uint64_t desc_B1 = make_smem_desc_mn_major(&s->s_V1[step * 4096], s->s_V1);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_dQ1, desc_A1, desc_B1, idesc, 1);
            }
        }
        
        tmem_commit_mma_fn(&s->s_bar);
        tmem_wait_mma_fn(&s->s_bar, phase);
        phase ^= 1;
        __syncthreads();

        for (uint32_t c = 0; c < 64; c += 4) {
            uint32_t regs0[4], regs1[4];
            read_from_tmem_packed(regs0, tmem_dQ0 + (tid << 16) + c);
            dQ_acc0[c] += __uint_as_float(regs0[0]);
            dQ_acc0[c+1] += __uint_as_float(regs0[1]);
            dQ_acc0[c+2] += __uint_as_float(regs0[2]);
            dQ_acc0[c+3] += __uint_as_float(regs0[3]);

            read_from_tmem_packed(regs1, tmem_dQ1 + (tid << 16) + c);
            dQ_acc1[c] += __uint_as_float(regs1[0]);
            dQ_acc1[c+1] += __uint_as_float(regs1[1]);
            dQ_acc1[c+2] += __uint_as_float(regs1[2]);
            dQ_acc1[c+3] += __uint_as_float(regs1[3]);
        }
        __syncthreads();
    }

    for (uint32_t c = 0; c < 64; c++) {
        if (m_block + tid < S) {
            atomicAdd(&dQ_workspace[(uint64_t)(m_block + tid) * 128 + c], dQ_acc0[c]);
            atomicAdd(&dQ_workspace[(uint64_t)(m_block + tid) * 128 + 64 + c], dQ_acc1[c]);
        }
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_S, 128);
        tmem_dealloc_fn(tmem_dP, 128);
        tmem_dealloc_fn(tmem_dQ0, 64);
        tmem_dealloc_fn(tmem_dQ1, 64);
    }
}

__global__ void bwd_dkv_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L, __nv_bfloat16* dK, __nv_bfloat16* dV,
    uint32_t S, float scale)
{
    uint32_t n_block = blockIdx.x * 128;
    uint32_t bh = blockIdx.z;
    uint32_t offset_bh = bh * S;

    if (n_block >= S) return;

    extern __shared__ __align__(1024) char smem_buf[];
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
        tma_load_3d_fn(&tma_K, &s->s_bar, s->s_K0, 0, offset_bh + n_block, bh);
        tma_load_3d_fn(&tma_K, &s->s_bar, s->s_K1, 64, offset_bh + n_block, bh);
        tma_load_3d_fn(&tma_V, &s->s_bar, s->s_V0, 0, offset_bh + n_block, bh);
        tma_load_3d_fn(&tma_V, &s->s_bar, s->s_V1, 64, offset_bh + n_block, bh);
    }
    mbarrier_wait_fn(&s->s_bar, phase);
    phase ^= 1;
    __syncthreads();
    fence_proxy_async_fn();

    uint32_t tmem_S, tmem_dP, tmem_PT, tmem_dPT, tmem_dK0, tmem_dK1, tmem_dV0, tmem_dV1;

    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S, 128);
        tmem_alloc_fn(&tmem_dP, 128);
        tmem_alloc_fn(&tmem_PT, 128);
        tmem_alloc_fn(&tmem_dPT, 128);
        tmem_alloc_fn(&tmem_dK0, 64);
        tmem_alloc_fn(&tmem_dK1, 64);
        tmem_alloc_fn(&tmem_dV0, 64);
        tmem_alloc_fn(&tmem_dV1, 64);
    }
    __syncthreads();

    float dK0_acc[64] = {0}, dK1_acc[64] = {0};
    float dV0_acc[64] = {0}, dV1_acc[64] = {0};

    for (uint32_t m_block = 0; m_block < S; m_block += 128) {
        if (threadIdx.x == 0) {
            uint32_t tx_bytes = 4 * (128 * 64 * 2);
            mbarrier_arrive_and_expect_tx_fn(&s->s_bar, tx_bytes);
            tma_load_3d_fn(&tma_Q, &s->s_bar, s->s_Q0, 0, offset_bh + m_block, bh);
            tma_load_3d_fn(&tma_Q, &s->s_bar, s->s_Q1, 64, offset_bh + m_block, bh);
            tma_load_3d_fn(&tma_dO, &s->s_bar, s->s_dO0, 0, offset_bh + m_block, bh);
            tma_load_3d_fn(&tma_dO, &s->s_bar, s->s_dO1, 64, offset_bh + m_block, bh);
        }
        
        if (tid < 128) {
            s->s_L[tid] = (m_block + tid < S) ? L[offset_bh + m_block + tid] : 0.0f;
        }
        mbarrier_wait_fn(&s->s_bar, phase);
        phase ^= 1;
        __syncthreads();
        fence_proxy_async_fn();

        uint32_t accum = 0;
        // S = Q @ K^T
        for (uint32_t step = 0; step < 64; step += 16) {
            uint64_t desc_A = make_smem_desc_k_major(&s->s_Q0[tid * 64 + step], s->s_Q0);
            uint64_t desc_B = make_smem_desc_k_major(&s->s_K0[tid * 64 + step], s->s_K0);
            uint32_t idesc = make_instr_desc_fn(128, 64, 0, 0);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_S, desc_A, desc_B, idesc, accum);
            }
            
            uint64_t desc_A1 = make_smem_desc_k_major(&s->s_Q1[tid * 64 + step], s->s_Q1);
            uint64_t desc_B1 = make_smem_desc_k_major(&s->s_K1[tid * 64 + step], s->s_K1);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_S + (64 << 2), desc_A1, desc_B1, idesc, accum);
            }
            accum = 1;
        }
        
        tmem_commit_mma_fn(&s->s_bar);
        tmem_wait_mma_fn(&s->s_bar, phase);
        phase ^= 1;
        __syncthreads();

        float S_local[128] = {0};
        for (uint32_t c = 0; c < 128; c += 4) {
            uint32_t regs[4];
            read_from_tmem_packed(regs, tmem_S + (tid << 16) + c);
            S_local[c] = __uint_as_float(regs[0]);
            S_local[c+1] = __uint_as_float(regs[1]);
            S_local[c+2] = __uint_as_float(regs[2]);
            S_local[c+3] = __uint_as_float(regs[3]);
        }

        for (uint32_t half = 0; half < 2; half++) {
            uint32_t c_start = half * 64;
            for (uint32_t cc = 0; cc < 64; cc++) {
                uint32_t c = c_start + cc;
                float s_val = S_local[c] * scale;
                float l_val = s->s_L[tid];
                float p = fast_exp2f_fn((s_val - l_val) * 1.44269504f);
                
                uint32_t swizzled_c = ((tid % 8) ^ (cc / 8)) * 8 + (cc % 8);
                if (half == 0) {
                    s->s_PT[tid * 128 + swizzled_c] = __float2bfloat16(p);
                } else {
                    s->s_PT[tid * 128 + 64 + swizzled_c] = __float2bfloat16(p);
                }
            }
        }
        __syncthreads();

        accum = 0;
        // dP = dO @ V^T
        for (uint32_t step = 0; step < 64; step += 16) {
            uint64_t desc_A = make_smem_desc_k_major(&s->s_dO0[tid * 64 + step], s->s_dO0);
            uint64_t desc_B = make_smem_desc_k_major(&s->s_V0[tid * 64 + step], s->s_V0);
            uint32_t idesc = make_instr_desc_fn(128, 64, 0, 0);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_dP, desc_A, desc_B, idesc, accum);
            }
            
            uint64_t desc_A1 = make_smem_desc_k_major(&s->s_dO1[tid * 64 + step], s->s_dO1);
            uint64_t desc_B1 = make_smem_desc_k_major(&s->s_V1[tid * 64 + step], s->s_V1);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_dP + (64 << 2), desc_A1, desc_B1, idesc, accum);
            }
            accum = 1;
        }
        
        tmem_commit_mma_fn(&s->s_bar);
        tmem_wait_mma_fn(&s->s_bar, phase);
        phase ^= 1;
        __syncthreads();

        float dP_local[128] = {0};
        for (uint32_t c = 0; c < 128; c += 4) {
            uint32_t regs[4];
            read_from_tmem_packed(regs, tmem_dP + (tid << 16) + c);
            dP_local[c] = __uint_as_float(regs[0]);
            dP_local[c+1] = __uint_as_float(regs[1]);
            dP_local[c+2] = __uint_as_float(regs[2]);
            dP_local[c+3] = __uint_as_float(regs[3]);
        }

        float D = 0;
        for (uint32_t half = 0; half < 2; half++) {
            uint32_t c_start = half * 64;
            for (uint32_t cc = 0; cc < 64; cc++) {
                uint32_t swizzled_c = ((tid % 8) ^ (cc / 8)) * 8 + (cc % 8);
                float p = __bfloat162float(half == 0 ? s->s_PT[tid * 128 + swizzled_c] : s->s_PT[tid * 128 + 64 + swizzled_c]);
                float dp = dP_local[c_start + cc];
                D += p * dp;
            }
        }

        for (uint32_t half = 0; half < 2; half++) {
            uint32_t c_start = half * 64;
            for (uint32_t cc = 0; cc < 64; cc++) {
                uint32_t swizzled_c = ((tid % 8) ^ (cc / 8)) * 8 + (cc % 8);
                float p = __bfloat162float(half == 0 ? s->s_PT[tid * 128 + swizzled_c] : s->s_PT[tid * 128 + 64 + swizzled_c]);
                float dp = dP_local[c_start + cc];
                float ds = (dp - D) * p * scale;
                
                if (half == 0) {
                    s->s_dPT[tid * 128 + swizzled_c] = __float2bfloat16(ds);
                } else {
                    s->s_dPT[tid * 128 + 64 + swizzled_c] = __float2bfloat16(ds);
                }
            }
        }
        __syncthreads();

        // dK += dS^T @ Q
        for (uint32_t step = 0; step < 64; step += 16) {
            uint64_t desc_A = make_smem_desc_mn_major(&s->s_dPT[step * 4096], s->s_dPT);
            uint64_t desc_B = make_smem_desc_mn_major(&s->s_Q0[step * 4096], s->s_Q0);
            uint32_t idesc = make_instr_desc_fn(128, 64, 1, 1);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_dK0, desc_A, desc_B, idesc, accum);
            }
            
            uint64_t desc_A1 = make_smem_desc_mn_major(&s->s_dPT[262144 + step * 4096], s->s_dPT);
            uint64_t desc_B1 = make_smem_desc_mn_major(&s->s_Q1[step * 4096], s->s_Q1);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_dK1, desc_A1, desc_B1, idesc, accum);
            }
        }

        // dV += P^T @ dO
        for (uint32_t step = 0; step < 64; step += 16) {
            uint64_t desc_A = make_smem_desc_mn_major(&s->s_PT[step * 4096], s->s_PT);
            uint64_t desc_B = make_smem_desc_mn_major(&s->s_dO0[step * 4096], s->s_dO0);
            uint32_t idesc = make_instr_desc_fn(128, 64, 1, 1);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_dV0, desc_A, desc_B, idesc, accum);
            }
            
            uint64_t desc_A1 = make_smem_desc_mn_major(&s->s_PT[262144 + step * 4096], s->s_PT);
            uint64_t desc_B1 = make_smem_desc_mn_major(&s->s_dO1[step * 4096], s->s_dO1);
            if (threadIdx.x == 0) {
                umma_f16_cta1_fn(tmem_dV1, desc_A1, desc_B1, idesc, accum);
            }
        }
        
        tmem_commit_mma_fn(&s->s_bar);
        tmem_wait_mma_fn(&s->s_bar, phase);
        phase ^= 1;
        __syncthreads();

        for (uint32_t c = 0; c < 64; c += 4) {
            uint32_t regs0[4], regs1[4], regs2[4], regs3[4];
            read_from_tmem_packed(regs0, tmem_dK0 + (tid << 16) + c);
            dK0_acc[c] += __uint_as_float(regs0[0]);
            dK0_acc[c+1] += __uint_as_float(regs0[1]);
            dK0_acc[c+2] += __uint_as_float(regs0[2]);
            dK0_acc[c+3] += __uint_as_float(regs0[3]);

            read_from_tmem_packed(regs1, tmem_dK1 + (tid << 16) + c);
            dK1_acc[c] += __uint_as_float(regs1[0]);
            dK1_acc[c+1] += __uint_as_float(regs1[1]);
            dK1_acc[c+2] += __uint_as_float(regs1[2]);
            dK1_acc[c+3] += __uint_as_float(regs1[3]);

            read_from_tmem_packed(regs2, tmem_dV0 + (tid << 16) + c);
            dV0_acc[c] += __uint_as_float(regs2[0]);
            dV0_acc[c+1] += __uint_as_float(regs2[1]);
            dV0_acc[c+2] += __uint_as_float(regs2[2]);
            dV0_acc[c+3] += __uint_as_float(regs2[3]);

            read_from_tmem_packed(regs3, tmem_dV1 + (tid << 16) + c);
            dV1_acc[c] += __uint_as_float(regs3[0]);
            dV1_acc[c+1] += __uint_as_float(regs3[1]);
            dV1_acc[c+2] += __uint_as_float(regs3[2]);
            dV1_acc[c+3] += __uint_as_float(regs3[3]);
        }
        __syncthreads();
    }

    for (uint32_t c = 0; c < 64; c++) {
        if (n_block + tid < S) {
            uint32_t global_offset_k0 = (uint64_t)(n_block + tid) * 128 + c;
            dK[global_offset_k0] = __float2bfloat16(dK0_acc[c]);
            
            uint32_t global_offset_k1 = (uint64_t)(n_block + tid) * 128 + 64 + c;
            dK[global_offset_k1] = __float2bfloat16(dK1_acc[c]);

            uint32_t global_offset_v0 = (uint64_t)(n_block + tid) * 128 + c;
            dV[global_offset_v0] = __float2bfloat16(dV0_acc[c]);

            uint32_t global_offset_v1 = (uint64_t)(n_block + tid) * 128 + 64 + c;
            dV[global_offset_v1] = __float2bfloat16(dV1_acc[c]);
        }
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_S, 128);
        tmem_dealloc_fn(tmem_dP, 128);
        tmem_dealloc_fn(tmem_PT, 128);
        tmem_dealloc_fn(tmem_dPT, 128);
        tmem_dealloc_fn(tmem_dK0, 64);
        tmem_dealloc_fn(tmem_dK1, 64);
        tmem_dealloc_fn(tmem_dV0, 64);
        tmem_dealloc_fn(tmem_dV1, 64);
    }
}

namespace tvm_ffi_mha_bwd_d128 {

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t dim0, uint64_t dim1, uint64_t dim2, 
                                     uint32_t box0, uint32_t box1, uint32_t box2, 
                                     CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
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

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    uint64_t num_elements = B * H * S * d;
    float* dQ_workspace;
    CUDA_CHECK(cudaMallocAsync(&dQ_workspace, num_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dQ_workspace, 0, num_elements * sizeof(float), stream));

    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    create_tma_3d_descriptor_2B(&tma_Q, (void*)q_ptr, d, S, B*H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_3d_descriptor_2B(&tma_K, (void*)k_ptr, d, S, B*H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_3d_descriptor_2B(&tma_V, (void*)v_ptr, d, S, B*H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_3d_descriptor_2B(&tma_O, (void*)o_ptr, d, S, B*H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_3d_descriptor_2B(&tma_dO, (void*)do_ptr, d, S, B*H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);

    float scale = 1.0f / sqrtf((float)d);

    int smem_size = sizeof(SharedStorage);
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    bwd_dq_kernel<<<dim3((S + 127) / 128, 1, B * H), 128, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO, l_ptr, dQ_workspace, S, scale
    );

    bwd_dkv_kernel<<<dim3((S + 127) / 128, 1, B * H), 128, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO, l_ptr, dk_ptr, dv_ptr, S, scale
    );

    convert_fp32_to_bf16<<<(num_elements + 255) / 256, 256, 0, stream>>>(dq_ptr, dQ_workspace, num_elements);

    CUDA_CHECK(cudaFreeAsync(dQ_workspace, stream));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd_d128::run);

} // namespace tvm_ffi_mha_bwd_d128