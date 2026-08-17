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

namespace tvm_ffi_attention_bwd {

__device__ __forceinline__ bool elect_one_sync_fn() {
    uint32_t pred;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xFFFFFFFF;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(pred));
    return pred != 0;
}

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

__device__ __forceinline__ void wait_for_umma(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_128b_swizzle(void* smem_ptr, uint32_t lbo, uint32_t sbo, void* base) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(base);
    uint32_t pattern_start = (addr >> 7) << 7;
    uint32_t base_offset = (addr >> 7) & 0x7;
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    d |= (uint64_t)base_offset << 49;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* smem_ptr, uint32_t k_dim, uint32_t stride) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t pattern_start = (addr >> 7) << 7;
    uint32_t base_offset = (addr >> 7) & 0x7;
    uint32_t sbo = 1024;
    uint32_t lbo = (k_dim / 8) * sbo; 
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    d |= (uint64_t)base_offset << 49;
    return d;
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

__global__ void attention_backward_dQ(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    uint32_t S, float scale_factor)
{
    uint32_t batch_head = blockIdx.z;
    uint32_t m_block = blockIdx.x * 128;

    extern __shared__ __nv_bfloat16 smem_buf[];
    __nv_bfloat16* smem_Q = smem_buf;
    __nv_bfloat16* smem_K = smem_Q + 128 * 128;
    __nv_bfloat16* smem_V = smem_K + 128 * 128;
    __nv_bfloat16* smem_dO = smem_V + 128 * 128;
    __nv_bfloat16* smem_O = smem_dO + 128 * 128;
    __nv_bfloat16* smem_P = smem_O + 128 * 128;
    __nv_bfloat16* smem_dS = smem_P + 128 * 128;
    float* smem_D = (float*)(smem_dS + 128 * 128);

    const __nv_bfloat16* q_ptr = Q + batch_head * S * 128;
    const __nv_bfloat16* k_ptr = K + batch_head * S * 128;
    const __nv_bfloat16* v_ptr = V + batch_head * S * 128;
    const __nv_bfloat16* o_ptr = O + batch_head * S * 128;
    const __nv_bfloat16* do_ptr = dO + batch_head * S * 128;
    __nv_bfloat16* dq_ptr = dQ + batch_head * S * 128;
    const float* l_ptr = L + batch_head * S;

    __shared__ alignas(16) uint64_t bar;
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&bar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (threadIdx.x < 128) {
        int m = threadIdx.x;
        float d_val = 0;
        for(int i=0; i<128; i+=2) {
            d_val += __low2float(*((uint32_t*)&o_ptr[(m_block + m) * 128 + i])) * __low2float(*((uint32_t*)&do_ptr[(m_block + m) * 128 + i])) +
                     __high2float(*((uint32_t*)&o_ptr[(m_block + m) * 128 + i])) * __high2float(*((uint32_t*)&do_ptr[(m_block + m) * 128 + i]));
        }
        smem_D[m] = d_val;
    }
    __syncthreads();

    if (elect_one_sync_fn()) {
        uint32_t tmem_dQ, tmem_S;
        tmem_alloc_fn(&tmem_dQ, 128);
        tmem_alloc_fn(&tmem_S, 128);
    }
    __syncthreads();

    uint32_t phase = 0;

    // Load Q and dO once using naive loads (Assume S % 128 == 0)
    for(int i=0; i<128; i+=2) {
        uint32_t q_packed = *(uint32_t*)&q_ptr[m_block * 128 + threadIdx.x * 128 + i];
        smem_Q[threadIdx.x * 128 + i] = __float2bfloat16(__low2float(q_packed));
        smem_Q[threadIdx.x * 128 + i + 1] = __float2bfloat16(__high2float(q_packed));
        
        uint32_t do_packed = *(uint32_t*)&do_ptr[m_block * 128 + threadIdx.x * 128 + i];
        smem_dO[threadIdx.x * 128 + i] = __float2bfloat16(__low2float(do_packed));
        smem_dO[threadIdx.x * 128 + i + 1] = __float2bfloat16(__high2float(do_packed));
    }
    __syncthreads();

    mbarrier_arrive_and_expect_tx_fn(&bar, 0);
    for(int i=0; i<8; i++) {
        uint32_t base = q_ptr + m_block * 128 + i * 16;
        uint64_t desc_a = make_smem_desc_128b_swizzle(smem_Q, 1, 1024, (void*)base);
        uint64_t desc_b = make_smem_desc_128b_swizzle(smem_K, 1, 1024, (void*)base);
        uint32_t idesc = make_instr_desc_fn(128, 128);
        if (threadIdx.x == 0) {
            asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                :: "r"(tmem_S), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(i==0 ? 0 : 1));
        }
    }
    if (threadIdx.x == 0) {
        umma_commit_1sm_fn(&bar);
    }
    wait_for_umma(&bar, phase);
    phase ^= 1;
    __syncthreads();

    for(uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_S + col, &r0, &r1, &r2, &r3);
        float s0 = __uint_as_float(r0);
        float s1 = __uint_as_float(r1);
        float s2 = __uint_as_float(r2);
        float s3 = __uint_as_float(r3);
        
        float l_val = l_ptr[m_block + threadIdx.x];
        float p0 = __expf(s0 * scale_factor - l_val);
        float p1 = __expf(s1 * scale_factor - l_val);
        float p2 = __expf(s2 * scale_factor - l_val);
        float p3 = __expf(s3 * scale_factor - l_val);
        
        int m = threadIdx.x * 128 + col;
        smem_P[m] = __float2bfloat16(p0);
        smem_P[m + 1] = __float2bfloat16(p1);
        smem_P[m + 2] = __float2bfloat16(p2);
        smem_P[m + 3] = __float2bfloat16(p3);
    }
    __syncthreads();

    mbarrier_arrive_and_expect_tx_fn(&bar, 0);
    for(int i=0; i<8; i++) {
        uint32_t base = do_ptr + m_block * 128 + i * 16;
        uint64_t desc_a = make_smem_desc_128b_swizzle(smem_dO, 1, 1024, (void*)base);
        uint64_t desc_b = make_smem_desc_128b_swizzle(smem_V, 1, 1024, (void*)base);
        uint32_t idesc = make_instr_desc_fn(128, 128);
        if (threadIdx.x == 0) {
            asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                :: "r"(tmem_S), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(1));
        }
    }
    if (threadIdx.x == 0) {
        umma_commit_1sm_fn(&bar);
    }
    wait_for_umma(&bar, phase);
    phase ^= 1;
    __syncthreads();

    for(uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_S + col, &r0, &r1, &r2, &r3);
        float dp0 = __uint_as_float(r0);
        float dp1 = __uint_as_float(r1);
        float dp2 = __uint_as_float(r2);
        float dp3 = __uint_as_float(r3);
        
        int m = threadIdx.x * 128 + col;
        float d_val = smem_D[threadIdx.x];
        
        float p0 = __bfloat162float(smem_P[m]);
        float p1 = __bfloat162float(smem_P[m + 1]);
        float p2 = __bfloat162float(smem_P[m + 2]);
        float p3 = __bfloat162float(smem_P[m + 3]);
        
        float ds0 = p0 * (dp0 - d_val) * scale_factor;
        float ds1 = p1 * (dp1 - d_val) * scale_factor;
        float ds2 = p2 * (dp2 - d_val) * scale_factor;
        float ds3 = p3 * (dp3 - d_val) * scale_factor;
        
        smem_dS[m] = __float2bfloat16(ds0);
        smem_dS[m + 1] = __float2bfloat16(ds1);
        smem_dS[m + 2] = __float2bfloat16(ds2);
        smem_dS[m + 3] = __float2bfloat16(ds3);
    }
    __syncthreads();

    mbarrier_arrive_and_expect_tx_fn(&bar, 0);
    for(int i=0; i<8; i++) {
        uint32_t base = smem_dS + (i * 16);
        uint64_t desc_a = make_smem_desc_128b_swizzle(smem_dS, 1, 1024, (void*)base);
        uint64_t desc_b = make_smem_desc_128b_swizzle(smem_K, 1, 1024, (void*)base);
        uint32_t idesc = make_instr_desc_fn(128, 128);
        if (threadIdx.x == 0) {
            asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                :: "r"(tmem_dQ), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(1));
        }
    }
    if (threadIdx.x == 0) {
        umma_commit_1sm_fn(&bar);
    }
    wait_for_umma(&bar, phase);
    phase ^= 1;
    __syncthreads();

    if (threadIdx.x < 128) {
        int m = threadIdx.x;
        for(uint32_t col = 0; col < 128; col += 2) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_dQ + col, &r0, &r1, &r2, &r3);
            float dq0 = __uint_as_float(r0);
            float dq1 = __uint_as_float(r1);
            
            uint32_t dq_packed = ((uint32_t)(__bfloat162float(__float2bfloat16(__float_as_uint(dq1))) << 16) | 
                                  (uint32_t)__bfloat162float(__float2bfloat16(__float_as_uint(dq0))));
            atomicAdd(*(uint32_t*)&dq_ptr[m * 128 + col], dq_packed);
        }
    }

    if (elect_one_sync_fn()) {
        tmem_dealloc_fn(tmem_S, 128);
        tmem_dealloc_fn(tmem_dQ, 128);
    }
}

__global__ void attention_backward_dK_dV(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    uint32_t S, float scale_factor)
{
    uint32_t batch_head = blockIdx.z;
    uint32_t n_block = blockIdx.x * 128;

    extern __shared__ __nv_bfloat16 smem_buf[];
    __nv_bfloat16* smem_Q = smem_buf;
    __nv_bfloat16* smem_K = smem_Q + 128 * 128;
    __nv_bfloat16* smem_V = smem_K + 128 * 128;
    __nv_bfloat16* smem_dO = smem_V + 128 * 128;
    __nv_bfloat16* smem_O = smem_dO + 128 * 128;
    __nv_bfloat16* smem_P = smem_O + 128 * 128;
    __nv_bfloat16* smem_dS = smem_P + 128 * 128;
    float* smem_D = (float*)(smem_dS + 128 * 128);
    float* smem_L = smem_D + 128;

    const __nv_bfloat16* q_ptr = Q + batch_head * S * 128;
    const __nv_bfloat16* k_ptr = K + batch_head * S * 128;
    const __nv_bfloat16* v_ptr = V + batch_head * S * 128;
    const __nv_bfloat16* o_ptr = O + batch_head * S * 128;
    const __nv_bfloat16* do_ptr = dO + batch_head * S * 128;
    __nv_bfloat16* dk_ptr = dK + batch_head * S * 128;
    __nv_bfloat16* dv_ptr = dV + batch_head * S * 128;
    const float* l_ptr = L + batch_head * S;

    __shared__ alignas(16) uint64_t bar;
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&bar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (elect_one_sync_fn()) {
        uint32_t tmem_dK, tmem_dV;
        tmem_alloc_fn(&tmem_dK, 128);
        tmem_alloc_fn(&tmem_dV, 128);
    }
    __syncthreads();

    uint32_t phase = 0;

    // Load K and V once
    for(int i=0; i<128; i+=2) {
        uint32_t k_packed = *(uint32_t*)&k_ptr[n_block * 128 + threadIdx.x * 128 + i];
        smem_K[threadIdx.x * 128 + i] = __float2bfloat16(__low2float(k_packed));
        smem_K[threadIdx.x * 128 + i + 1] = __float2bfloat16(__high2float(k_packed));
        
        uint32_t v_packed = *(uint32_t*)&v_ptr[n_block * 128 + threadIdx.x * 128 + i];
        smem_V[threadIdx.x * 128 + i] = __float2bfloat16(__low2float(v_packed));
        smem_V[threadIdx.x * 128 + i + 1] = __float2bfloat16(__high2float(v_packed));
    }
    __syncthreads();

    for (int m_block = 0; m_block < S; m_block += 128) {
        if (threadIdx.x < 128) {
            int m = threadIdx.x;
            float d_val = 0;
            for(int i=0; i<128; i+=2) {
                d_val += __low2float(*((uint32_t*)&o_ptr[(m_block + m) * 128 + i])) * __low2float(*((uint32_t*)&do_ptr[(m_block + m) * 128 + i])) +
                         __high2float(*((uint32_t*)&o_ptr[(m_block + m) * 128 + i])) * __high2float(*((uint32_t*)&do_ptr[(m_block + m) * 128 + i]));
            }
            smem_D[m] = d_val;
            smem_L[m] = l_ptr[m_block + m];
        }

        for(int i=0; i<128; i+=2) {
            uint32_t q_packed = *(uint32_t*)&q_ptr[m_block * 128 + threadIdx.x * 128 + i];
            smem_Q[threadIdx.x * 128 + i] = __float2bfloat16(__low2float(q_packed));
            smem_Q[threadIdx.x * 128 + i + 1] = __float2bfloat16(__high2float(q_packed));
            
            uint32_t do_packed = *(uint32_t*)&do_ptr[m_block * 128 + threadIdx.x * 128 + i];
            smem_dO[threadIdx.x * 128 + i] = __float2bfloat16(__low2float(do_packed));
            smem_dO[threadIdx.x * 128 + i + 1] = __float2bfloat16(__high2float(do_packed));
        }
        __syncthreads();

        if (elect_one_sync_fn()) {
            uint32_t tmem_S, tmem_dS;
            tmem_alloc_fn(&tmem_S, 64);
            tmem_alloc_fn(&tmem_dS, 128);
        }
        __syncthreads();

        mbarrier_arrive_and_expect_tx_fn(&bar, 0);
        for(int i=0; i<8; i++) {
            uint32_t base_q = q_ptr + m_block * 128 + i * 16;
            uint64_t desc_a = make_smem_desc_128b_swizzle(smem_Q, 1, 1024, (void*)base_q);
            uint32_t base_k = k_ptr + n_block * 128 + i * 16;
            uint64_t desc_b = make_smem_desc_128b_swizzle(smem_K, 1, 1024, (void*)base_k);
            uint32_t idesc = make_instr_desc_fn(128, 128);
            if (threadIdx.x == 0) {
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                    :: "r"(tmem_S), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(i==0 ? 0 : 1));
            }
        }
        if (threadIdx.x == 0) {
            umma_commit_1sm_fn(&bar);
        }
        wait_for_umma(&bar, phase);
        phase ^= 1;
        __syncthreads();

        for(uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S + col, &r0, &r1, &r2, &r3);
            float s0 = __uint_as_float(r0);
            float s1 = __uint_as_float(r1);
            float s2 = __uint_as_float(r2);
            float s3 = __uint_as_float(r3);
            
            float l_val = smem_L[threadIdx.x];
            float p0 = __expf(s0 * scale_factor - l_val);
            float p1 = __expf(s1 * scale_factor - l_val);
            float p2 = __expf(s2 * scale_factor - l_val);
            float p3 = __expf(s3 * scale_factor - l_val);
            
            int m = threadIdx.x * 128 + col;
            smem_P[m] = __float2bfloat16(p0);
            smem_P[m + 1] = __float2bfloat16(p1);
            smem_P[m + 2] = __float2bfloat16(p2);
            smem_P[m + 3] = __float2bfloat16(p3);
        }
        __syncthreads();

        mbarrier_arrive_and_expect_tx_fn(&bar, 0);
        for(int i=0; i<8; i++) {
            uint32_t base_do = do_ptr + m_block * 128 + i * 16;
            uint64_t desc_a = make_smem_desc_128b_swizzle(smem_dO, 1, 1024, (void*)base_do);
            uint32_t base_v = v_ptr + n_block * 128 + i * 16;
            uint64_t desc_b = make_smem_desc_128b_swizzle(smem_V, 1, 1024, (void*)base_v);
            uint32_t idesc = make_instr_desc_fn(128, 128);
            if (threadIdx.x == 0) {
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                    :: "r"(tmem_S), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(1));
            }
        }
        if (threadIdx.x == 0) {
            umma_commit_1sm_fn(&bar);
        }
        wait_for_umma(&bar, phase);
        phase ^= 1;
        __syncthreads();

        for(uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S + col, &r0, &r1, &r2, &r3);
            float dp0 = __uint_as_float(r0);
            float dp1 = __uint_as_float(r1);
            float dp2 = __uint_as_float(r2);
            float dp3 = __uint_as_float(r3);
            
            int m = threadIdx.x * 128 + col;
            float d_val = smem_D[threadIdx.x];
            
            float p0 = __bfloat162float(smem_P[m]);
            float p1 = __bfloat162float(smem_P[m + 1]);
            float p2 = __bfloat162float(smem_P[m + 2]);
            float p3 = __bfloat162float(smem_P[m + 3]);
            
            float ds0 = p0 * (dp0 - d_val) * scale_factor;
            float ds1 = p1 * (dp1 - d_val) * scale_factor;
            float ds2 = p2 * (dp2 - d_val) * scale_factor;
            float ds3 = p3 * (dp3 - d_val) * scale_factor;
            
            smem_dS[m] = __float2bfloat16(ds0);
            smem_dS[m + 1] = __float2bfloat16(ds1);
            smem_dS[m + 2] = __float2bfloat16(ds2);
            smem_dS[m + 3] = __float2bfloat16(ds3);
        }
        __syncthreads();

        mbarrier_arrive_and_expect_tx_fn(&bar, 0);
        for(int i=0; i<8; i++) {
            uint32_t base = smem_dS + (i * 16 * 128);
            uint64_t desc_a = make_smem_desc_mn_major(smem_dS, 128, 128);
            uint32_t base_q = q_ptr + m_block * 128 + (i * 16);
            uint64_t desc_b = make_smem_desc_128b_swizzle(smem_Q, 1, 1024, (void*)base_q);
            uint32_t idesc = make_instr_desc_fn(128, 128);
            if (threadIdx.x == 0) {
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                    :: "r"(tmem_dK), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(1));
            }
        }
        
        for(int i=0; i<8; i++) {
            uint32_t base = smem_P + (i * 16 * 128);
            uint64_t desc_a = make_smem_desc_mn_major(smem_P, 128, 128);
            uint32_t base_do = do_ptr + m_block * 128 + (i * 16);
            uint64_t desc_b = make_smem_desc_128b_swizzle(smem_dO, 1, 1024, (void*)base_do);
            uint32_t idesc = make_instr_desc_fn(128, 128);
            if (threadIdx.x == 0) {
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                    :: "r"(tmem_dV), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(1));
            }
        }
        
        if (threadIdx.x == 0) {
            umma_commit_1sm_fn(&bar);
        }
        wait_for_umma(&bar, phase);
        phase ^= 1;
        __syncthreads();

        if (elect_one_sync_fn()) {
            tmem_dealloc_fn(tmem_S, 64);
            tmem_dealloc_fn(tmem_dS, 128);
        }
        __syncthreads();
    }

    if (threadIdx.x < 128) {
        int m = threadIdx.x;
        for(uint32_t col = 0; col < 128; col += 2) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_dK + col, &r0, &r1, &r2, &r3);
            float dk0 = __uint_as_float(r0);
            float dk1 = __uint_as_float(r1);
            
            uint32_t dk_packed = ((uint32_t)(__bfloat162float(__float2bfloat16(__float_as_uint(dk1))) << 16) | 
                                  (uint32_t)__bfloat162float(__float2bfloat16(__float_as_uint(dk0))));
            atomicAdd(*(uint32_t*)&dk_ptr[m * 128 + col], dk_packed);
            
            tmem_load_4x_fn(tmem_dV + col, &r0, &r1, &r2, &r3);
            float dv0 = __uint_as_float(r0);
            float dv1 = __uint_as_float(r1);
            
            uint32_t dv_packed = ((uint32_t)(__bfloat162float(__float2bfloat16(__float_as_uint(dv1))) << 16) | 
                                  (uint32_t)__bfloat162float(__float2bfloat16(__float_as_uint(dv0))));
            atomicAdd(*(uint32_t*)&dv_ptr[m * 128 + col], dv_packed);
        }
    }

    if (elect_one_sync_fn()) {
        tmem_dealloc_fn(tmem_dK, 128);
        tmem_dealloc_fn(tmem_dV, 128);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3); 
    
    float scale_factor = 1.0f / sqrtf((float)d);

    dim3 grid((S + 127) / 128, (S + 127) / 128, B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    size_t dQ_shmem = 7 * 128 * 128 * sizeof(__nv_bfloat16) + 128 * sizeof(float);
    size_t dK_dV_shmem = 8 * 128 * 128 * sizeof(__nv_bfloat16) + 128 * sizeof(float) + 128 * sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute(attention_backward_dQ, cudaFuncAttributeMaxDynamicSharedMemorySize, dQ_shmem));
    CUDA_CHECK(cudaFuncSetAttribute(attention_backward_dK_dV, cudaFuncAttributeMaxDynamicSharedMemorySize, dK_dV_shmem));

    attention_backward_dQ<<<grid, block, dQ_shmem, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        S, scale_factor);
    
    CUDA_CHECK(cudaGetLastError());

    attention_backward_dK_dV<<<grid, block, dK_dV_shmem, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S, scale_factor);
    
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_attention_bwd::run);

} // namespace tvm_ffi_attention_bwd