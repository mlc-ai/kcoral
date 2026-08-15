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

namespace tvm_ffi_mha_bwd {

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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
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

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
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

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
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

__device__ __forceinline__ void write_swizzled_u32(__nv_bfloat16* ptr, uint32_t row, uint32_t col, uint32_t v0, uint32_t v1) {
    uint32_t x = col / 8;
    uint32_t sx = x ^ (row % 8);
    uint32_t sc = sx * 8 + (col % 8);
    uint32_t offset = row * 64 + sc;
    *reinterpret_cast<uint32_t*>(&ptr[offset]) = v0;
    *reinterpret_cast<uint32_t*>(&ptr[offset + 2]) = v1;
}

__device__ __forceinline__ uint32_t read_swizzled_u32(const __nv_bfloat16* ptr, uint32_t r, uint32_t c) {
    uint32_t g = c / 8;
    uint32_t g_swizzled = g ^ (r % 8);
    uint32_t true_c = g_swizzled * 8 + (c % 8);
    return *reinterpret_cast<const uint32_t*>(&ptr[r * 64 + true_c]);
}

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

__global__ void mha_bwd_kernel(
    __grid_constant__ const CUtensorMap tma_Q,
    __grid_constant__ const CUtensorMap tma_K,
    __grid_constant__ const CUtensorMap tma_V,
    __grid_constant__ const CUtensorMap tma_O,
    __grid_constant__ const CUtensorMap tma_dO,
    const float* L,
    __nv_bfloat16* dQ,
    float* fp32_dK,
    float* fp32_dV,
    uint32_t S, uint32_t H, uint32_t B)
{
    uint32_t b_idx = blockIdx.z;
    uint32_t h_idx = blockIdx.y;
    uint32_t q_tile = blockIdx.x;
    uint32_t q_start = q_tile * 64;
    uint32_t num_k_tiles = (S + 63) / 64;

    extern __shared__ __align__(128) char smem[];
    __nv_bfloat16* s_Q0 = (__nv_bfloat16*)smem;                
    __nv_bfloat16* s_Q1 = (__nv_bfloat16*)(smem + 8192);       
    __nv_bfloat16* s_O0 = (__nv_bfloat16*)(smem + 16384);      
    __nv_bfloat16* s_O1 = (__nv_bfloat16*)(smem + 24576);      
    __nv_bfloat16* s_dO0 = (__nv_bfloat16*)(smem + 32768);     
    __nv_bfloat16* s_dO1 = (__nv_bfloat16*)(smem + 40960);     
    __nv_bfloat16* s_K0 = (__nv_bfloat16*)(smem + 49152);      
    __nv_bfloat16* s_K1 = (__nv_bfloat16*)(smem + 57344);      
    __nv_bfloat16* s_V0 = (__nv_bfloat16*)(smem + 65536);      
    __nv_bfloat16* s_V1 = (__nv_bfloat16*)(smem + 73728);      
    __nv_bfloat16* s_P_T = (__nv_bfloat16*)(smem + 81920);    
    __nv_bfloat16* s_dS = (__nv_bfloat16*)(smem + 90112);    
    float* s_D = (float*)(smem + 98304);                     
    uint64_t* bar_Q0 = (uint64_t*)(smem + 98560);
    uint64_t* bar_Q1 = (uint64_t*)(smem + 98568);
    uint64_t* bar_O0 = (uint64_t*)(smem + 98576);
    uint64_t* bar_O1 = (uint64_t*)(smem + 98584);
    uint64_t* bar_dO0 = (uint64_t*)(smem + 98592);
    uint64_t* bar_dO1 = (uint64_t*)(smem + 98600);
    uint64_t* bar_K0 = (uint64_t*)(smem + 98608);
    uint64_t* bar_K1 = (uint64_t*)(smem + 98616);
    uint64_t* bar_V0 = (uint64_t*)(smem + 98624);
    uint64_t* bar_V1 = (uint64_t*)(smem + 98632);
    uint64_t* bar_S = (uint64_t*)(smem + 98640);
    uint64_t* bar_dP = (uint64_t*)(smem + 98648);
    uint64_t* bar_dQ = (uint64_t*)(smem + 98656);
    uint64_t* bar_dK = (uint64_t*)(smem + 98664);
    uint64_t* bar_dV = (uint64_t*)(smem + 98672);

    uint32_t tmem_S_T, tmem_dP_T, tmem_dQ0, tmem_dQ1, tmem_dK0, tmem_dK1, tmem_dV0;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_S_T, 64);
        tmem_alloc_fn(&tmem_dP_T, 64);
        tmem_alloc_fn(&tmem_dQ0, 64);
        tmem_alloc_fn(&tmem_dQ1, 64);
        tmem_alloc_fn(&tmem_dK0, 64);
        tmem_alloc_fn(&tmem_dK1, 64);
        tmem_alloc_fn(&tmem_dV0, 64);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_Q0, 1);
        init_smem_barrier_fn(bar_Q1, 1);
        init_smem_barrier_fn(bar_O0, 1);
        init_smem_barrier_fn(bar_O1, 1);
        init_smem_barrier_fn(bar_dO0, 1);
        init_smem_barrier_fn(bar_dO1, 1);
        init_smem_barrier_fn(bar_K0, 1);
        init_smem_barrier_fn(bar_K1, 1);
        init_smem_barrier_fn(bar_V0, 1);
        init_smem_barrier_fn(bar_V1, 1);
        init_smem_barrier_fn(bar_S, 1);
        init_smem_barrier_fn(bar_dP, 1);
        init_smem_barrier_fn(bar_dQ, 1);
        init_smem_barrier_fn(bar_dK, 1);
        init_smem_barrier_fn(bar_dV, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t phase_Q0 = 0, phase_Q1 = 0, phase_O0 = 0, phase_O1 = 0, phase_dO0 = 0, phase_dO1 = 0;
    uint32_t phase_K0 = 0, phase_K1 = 0, phase_V0 = 0, phase_V1 = 0;
    uint32_t phase_S = 0, phase_dP = 0, phase_dQ = 0, phase_dK = 0, phase_dV = 0;

    uint32_t q_outer = b_idx * H * S + h_idx * S + q_start;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_Q0, 8192);
        tma_load_2d_fn(&tma_Q, bar_Q0, s_Q0, 0, q_outer);
        mbarrier_arrive_and_expect_tx_fn(bar_Q1, 8192);
        tma_load_2d_fn(&tma_Q, bar_Q1, s_Q1, 64, q_outer);
        
        mbarrier_arrive_and_expect_tx_fn(bar_O0, 8192);
        tma_load_2d_fn(&tma_O, bar_O0, s_O0, 0, q_outer);
        mbarrier_arrive_and_expect_tx_fn(bar_O1, 8192);
        tma_load_2d_fn(&tma_O, bar_O1, s_O1, 64, q_outer);
        
        mbarrier_arrive_and_expect_tx_fn(bar_dO0, 8192);
        tma_load_2d_fn(&tma_dO, bar_dO0, s_dO0, 0, q_outer);
        mbarrier_arrive_and_expect_tx_fn(bar_dO1, 8192);
        tma_load_2d_fn(&tma_dO, bar_dO1, s_dO1, 64, q_outer);
    }
    mbarrier_wait_fn(bar_Q0, phase_Q0); phase_Q0 ^= 1;
    mbarrier_wait_fn(bar_Q1, phase_Q1); phase_Q1 ^= 1;
    mbarrier_wait_fn(bar_O0, phase_O0); phase_O0 ^= 1;
    mbarrier_wait_fn(bar_O1, phase_O1); phase_O1 ^= 1;
    mbarrier_wait_fn(bar_dO0, phase_dO0); phase_dO0 ^= 1;
    mbarrier_wait_fn(bar_dO1, phase_dO1); phase_dO1 ^= 1;
    __syncthreads();

    uint32_t row = threadIdx.x;
    if (row < 64) {
        float sum = 0;
        for(int i=0; i<64; i++) {
            sum += __bfloat162float(s_O0[row*64 + i]) * __bfloat162float(s_dO0[row*64 + i]);
            sum += __bfloat162float(s_O1[row*64 + i]) * __bfloat162float(s_dO1[row*64 + i]);
        }
        if (q_start + row < S) {
            s_D[row] = sum;
        } else {
            s_D[row] = 0;
        }
    }
    __syncthreads();

    auto desc_K_major = [](void* ptr) {
        return make_smem_desc_sm100_fn(ptr, 1, 1024);
    };
    auto desc_N_major = [](void* ptr) {
        return make_smem_desc_sm100_fn(ptr, 8192, 1024);
    };

    auto advance_desc = [&](uint64_t desc, int steps) {
        return desc + (steps << 4);
    };

    uint32_t idesc_no_trans = (1<<4) | (1<<7) | (1<<10) | (0<<15) | (0<<16) | (8<<17) | (4<<24);
    uint32_t idesc_transB = (1<<4) | (1<<7) | (1<<10) | (0<<15) | (1<<16) | (8<<17) | (4<<24);
    uint32_t idesc_transA_transB = (1<<4) | (1<<7) | (1<<10) | (1<<15) | (1<<16) | (8<<17) | (4<<24);

    uint32_t k_outer = b_idx * H * S + h_idx * S;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_K0, 8192);
        tma_load_2d_fn(&tma_K, bar_K0, s_K0, 0, k_outer);
        mbarrier_arrive_and_expect_tx_fn(bar_K1, 8192);
        tma_load_2d_fn(&tma_K, bar_K1, s_K1, 64, k_outer);
        
        mbarrier_arrive_and_expect_tx_fn(bar_V0, 8192);
        tma_load_2d_fn(&tma_V, bar_V0, s_V0, 0, k_outer);
        mbarrier_arrive_and_expect_tx_fn(bar_V1, 8192);
        tma_load_2d_fn(&tma_V, bar_V1, s_V1, 64, k_outer);
    }

    for (uint32_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        if (k_tile > 0) {
            mbarrier_wait_fn(bar_K0, phase_K0);
            phase_K0 ^= 1;
            mbarrier_wait_fn(bar_K1, phase_K1);
            phase_K1 ^= 1;
            mbarrier_wait_fn(bar_V0, phase_V0);
            phase_V0 ^= 1;
            mbarrier_wait_fn(bar_V1, phase_V1);
            phase_V1 ^= 1;
            __syncthreads();
        }

        uint32_t k_start = k_tile * 64;
        uint32_t next_k_start = (k_tile + 1) * 64;
        uint32_t next_k_outer = b_idx * H * S + h_idx * S + next_k_start;

        if (k_tile == 0 || k_tile + 1 < num_k_tiles) {
            if (next_k_start < S) {
                if (threadIdx.x == 0) {
                    mbarrier_arrive_and_expect_tx_fn(bar_K0, 8192);
                    tma_load_2d_fn(&tma_K, bar_K0, s_K0, 0, next_k_outer);
                    mbarrier_arrive_and_expect_tx_fn(bar_K1, 8192);
                    tma_load_2d_fn(&tma_K, bar_K1, s_K1, 64, next_k_outer);
                    
                    mbarrier_arrive_and_expect_tx_fn(bar_V0, 8192);
                    tma_load_2d_fn(&tma_V, bar_V0, s_V0, 0, next_k_outer);
                    mbarrier_arrive_and_expect_tx_fn(bar_V1, 8192);
                    tma_load_2d_fn(&tma_V, bar_V1, s_V1, 64, next_k_outer);
                }
            }
        }

        if (k_tile > 0 || (k_tile == 0 && num_k_tiles > 0)) {
            if (threadIdx.x == 0) {
                uint64_t desc_Q0 = desc_K_major(s_Q0);
                uint64_t desc_Q1 = desc_K_major(s_Q1);
                uint64_t desc_K0 = desc_K_major(s_K0);
                uint64_t desc_K1 = desc_K_major(s_K1);

                umma_f16_cg1_fn(tmem_S_T, desc_Q0, desc_K0, idesc_no_trans, false);
                umma_f16_cg1_fn(tmem_S_T, advance_desc(desc_Q0, 1), advance_desc(desc_K0, 1), idesc_no_trans, true);
                umma_f16_cg1_fn(tmem_S_T, advance_desc(desc_Q0, 2), advance_desc(desc_K0, 2), idesc_no_trans, true);
                umma_f16_cg1_fn(tmem_S_T, advance_desc(desc_Q0, 3), advance_desc(desc_K0, 3), idesc_no_trans, true);

                umma_f16_cg1_fn(tmem_S_T, desc_Q1, desc_K1, idesc_no_trans, true);
                umma_f16_cg1_fn(tmem_S_T, advance_desc(desc_Q1, 1), advance_desc(desc_K1, 1), idesc_no_trans, true);
                umma_f16_cg1_fn(tmem_S_T, advance_desc(desc_Q1, 2), advance_desc(desc_K1, 2), idesc_no_trans, true);
                umma_f16_cg1_fn(tmem_S_T, advance_desc(desc_Q1, 3), advance_desc(desc_K1, 3), idesc_no_trans, true);
                
                umma_commit_1sm_fn(bar_S);
            }
            mbarrier_wait_fn(bar_S, phase_S);
            phase_S ^= 1;

            if (row < 64) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_S_T + row*64, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                
                float lse = (q_start + row < S) ? L[b_idx * H * S + h_idx * S + q_start + row] : 0;
                float p0 = expf(__uint_as_float(r0) * (1.0f/sqrtf(128)) - lse);
                float p1 = expf(__uint_as_float(r1) * (1.0f/sqrtf(128)) - lse);
                float p2 = expf(__uint_as_float(r2) * (1.0f/sqrtf(128)) - lse);
                float p3 = expf(__uint_as_float(r3) * (1.0f/sqrtf(128)) - lse);
                
                if (k_start >= S) p0 = 0;
                if (k_start + 1 >= S) p1 = 0;
                if (k_start + 2 >= S) p2 = 0;
                if (k_start + 3 >= S) p3 = 0;
                
                uint32_t packed_p0 = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
                uint32_t packed_p2 = pack_bf16_fn(__float_as_uint(p2), __float_as_uint(p3));
                write_swizzled_u32(s_P_T, row, 0, packed_p0, packed_p2);
                
                tmem_load_4x_fn(tmem_S_T + row*64 + 32, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                
                p0 = expf(__uint_as_float(r0) * (1.0f/sqrtf(128)) - lse);
                p1 = expf(__uint_as_float(r1) * (1.0f/sqrtf(128)) - lse);
                p2 = expf(__uint_as_float(r2) * (1.0f/sqrtf(128)) - lse);
                p3 = expf(__uint_as_float(r3) * (1.0f/sqrtf(128)) - lse);
                
                if (k_start + 32 >= S) p0 = 0;
                if (k_start + 33 >= S) p1 = 0;
                if (k_start + 34 >= S) p2 = 0;
                if (k_start + 35 >= S) p3 = 0;
                
                packed_p0 = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
                packed_p2 = pack_bf16_fn(__float_as_uint(p2), __float_as_uint(p3));
                write_swizzled_u32(s_P_T, row, 32, packed_p0, packed_p2);
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                uint64_t desc_V0 = desc_K_major(s_V0);
                uint64_t desc_V1 = desc_K_major(s_V1);
                uint64_t desc_dO0 = desc_K_major(s_dO0);
                uint64_t desc_dO1 = desc_K_major(s_dO1);

                umma_f16_cg1_fn(tmem_dP_T, desc_V0, desc_dO0, idesc_no_trans, false);
                umma_f16_cg1_fn(tmem_dP_T, advance_desc(desc_V0, 1), advance_desc(desc_dO0, 1), idesc_no_trans, true);
                umma_f16_cg1_fn(tmem_dP_T, advance_desc(desc_V0, 2), advance_desc(desc_dO0, 2), idesc_no_trans, true);
                umma_f16_cg1_fn(tmem_dP_T, advance_desc(desc_V0, 3), advance_desc(desc_dO0, 3), idesc_no_trans, true);

                umma_f16_cg1_fn(tmem_dP_T, desc_V1, desc_dO1, idesc_no_trans, true);
                umma_f16_cg1_fn(tmem_dP_T, advance_desc(desc_V1, 1), advance_desc(desc_dO1, 1), idesc_no_trans, true);
                umma_f16_cg1_fn(tmem_dP_T, advance_desc(desc_V1, 2), advance_desc(desc_dO1, 2), idesc_no_trans, true);
                umma_f16_cg1_fn(tmem_dP_T, advance_desc(desc_V1, 3), advance_desc(desc_dO1, 3), idesc_no_trans, true);
                
                umma_commit_1sm_fn(bar_dP);
            }
            mbarrier_wait_fn(bar_dP, phase_dP);
            phase_dP ^= 1;

            if (row < 64) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_dP_T + row*64, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                
                float dp0 = __uint_as_float(r0);
                float dp1 = __uint_as_float(r1);
                float dp2 = __uint_as_float(r2);
                float dp3 = __uint_as_float(r3);
                
                float d_val = s_D[row];
                
                uint32_t packed_p0 = read_swizzled_u32(s_P_T, row, 0);
                float p0 = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&packed_p0));
                float p1 = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&packed_p0 + 1));
                
                uint32_t packed_p2 = read_swizzled_u32(s_P_T, row, 2);
                float p2 = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&packed_p2));
                float p3 = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&packed_p2 + 1));
                
                float ds0 = p0 * (dp0 - d_val);
                float ds1 = p1 * (dp1 - d_val);
                float ds2 = p2 * (dp2 - d_val);
                float ds3 = p3 * (dp3 - d_val);
                
                uint32_t packed_ds0 = pack_bf16_fn(__float_as_uint(ds0), __float_as_uint(ds1));
                uint32_t packed_ds2 = pack_bf16_fn(__float_as_uint(ds2), __float_as_uint(ds3));
                write_swizzled_u32(s_dS, row, 0, packed_ds0, packed_ds2);
                
                tmem_load_4x_fn(tmem_dP_T + row*64 + 32, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                
                dp0 = __uint_as_float(r0); dp1 = __uint_as_float(r1);
                dp2 = __uint_as_float(r2); dp3 = __uint_as_float(r3);
                
                packed_p0 = read_swizzled_u32(s_P_T, row, 32);
                p0 = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&packed_p0));
                p1 = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&packed_p0 + 1));
                
                packed_p2 = read_swizzled_u32(s_P_T, row, 34);
                p2 = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&packed_p2));
                p3 = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&packed_p2 + 1));
                
                ds0 = p0 * (dp0 - d_val); ds1 = p1 * (dp1 - d_val);
                ds2 = p2 * (dp2 - d_val); ds3 = p3 * (dp3 - d_val);
                
                packed_ds0 = pack_bf16_fn(__float_as_uint(ds0), __float_as_uint(ds1));
                packed_ds2 = pack_bf16_fn(__float_as_uint(ds2), __float_as_uint(ds3));
                write_swizzled_u32(s_dS, row, 32, packed_ds0, packed_ds2);
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                uint64_t desc_P_T = desc_K_major(s_P_T);
                uint64_t desc_dO0_N = desc_N_major(s_dO0);
                uint64_t desc_dO1_N = desc_N_major(s_dO1);

                umma_f16_cg1_fn(tmem_dV0, desc_P_T, desc_dO0_N, idesc_transA_transB, false);
                umma_f16_cg1_fn(tmem_dV0, advance_desc(desc_P_T, 1), advance_desc(desc_dO0_N, 1), idesc_transA_transB, true);
                umma_f16_cg1_fn(tmem_dV0, advance_desc(desc_P_T, 2), advance_desc(desc_dO0_N, 2), idesc_transA_transB, true);
                umma_f16_cg1_fn(tmem_dV0, advance_desc(desc_P_T, 3), advance_desc(desc_dO0_N, 3), idesc_transA_transB, true);
                
                umma_commit_1sm_fn(bar_dV);
            }
            mbarrier_wait_fn(bar_dV, phase_dV); phase_dV ^= 1;

            if (row < 64) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_dV0 + row*64, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                uint32_t dv_base = b_idx * H * S * 128 + h_idx * S * 128 + (k_start + row) * 128;
                if (k_start + row < S) {
                    atomicAdd(&fp32_dV[dv_base + 0], __uint_as_float(r0));
                    atomicAdd(&fp32_dV[dv_base + 1], __uint_as_float(r1));
                    atomicAdd(&fp32_dV[dv_base + 2], __uint_as_float(r2));
                    atomicAdd(&fp32_dV[dv_base + 3], __uint_as_float(r3));
                }
                tmem_load_4x_fn(tmem_dV0 + row*64 + 32, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                if (k_start + row < S) {
                    atomicAdd(&fp32_dV[dv_base + 32], __uint_as_float(r0));
                    atomicAdd(&fp32_dV[dv_base + 33], __uint_as_float(r1));
                    atomicAdd(&fp32_dV[dv_base + 34], __uint_as_float(r2));
                    atomicAdd(&fp32_dV[dv_base + 35], __uint_as_float(r3));
                }
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                uint64_t desc_P_T = desc_K_major(s_P_T);
                uint64_t desc_dO1_N = desc_N_major(s_dO1);

                umma_f16_cg1_fn(tmem_dK0, desc_P_T, desc_dO1_N, idesc_transA_transB, false);
                umma_f16_cg1_fn(tmem_dK0, advance_desc(desc_P_T, 1), advance_desc(desc_dO1_N, 1), idesc_transA_transB, true);
                umma_f16_cg1_fn(tmem_dK0, advance_desc(desc_P_T, 2), advance_desc(desc_dO1_N, 2), idesc_transA_transB, true);
                umma_f16_cg1_fn(tmem_dK0, advance_desc(desc_P_T, 3), advance_desc(desc_dO1_N, 3), idesc_transA_transB, true);
                
                umma_commit_1sm_fn(bar_dV);
            }
            mbarrier_wait_fn(bar_dV, phase_dV); phase_dV ^= 1;

            if (row < 64) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_dK0 + row*64, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                uint32_t dv_base = b_idx * H * S * 128 + h_idx * S * 128 + (k_start + row) * 128;
                if (k_start + row < S) {
                    atomicAdd(&fp32_dV[dv_base + 64], __uint_as_float(r0));
                    atomicAdd(&fp32_dV[dv_base + 65], __uint_as_float(r1));
                    atomicAdd(&fp32_dV[dv_base + 66], __uint_as_float(r2));
                    atomicAdd(&fp32_dV[dv_base + 67], __uint_as_float(r3));
                }
                tmem_load_4x_fn(tmem_dK0 + row*64 + 32, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                if (k_start + row < S) {
                    atomicAdd(&fp32_dV[dv_base + 96], __uint_as_float(r0));
                    atomicAdd(&fp32_dV[dv_base + 97], __uint_as_float(r1));
                    atomicAdd(&fp32_dV[dv_base + 98], __uint_as_float(r2));
                    atomicAdd(&fp32_dV[dv_base + 99], __uint_as_float(r3));
                }
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                uint64_t desc_dS = desc_K_major(s_dS);
                uint64_t desc_Q0_N = desc_N_major(s_Q0);
                uint64_t desc_Q1_N = desc_N_major(s_Q1);

                umma_f16_cg1_fn(tmem_dK0, desc_dS, desc_Q0_N, idesc_transA_transB, false);
                umma_f16_cg1_fn(tmem_dK0, advance_desc(desc_dS, 1), advance_desc(desc_Q0_N, 1), idesc_transA_transB, true);
                umma_f16_cg1_fn(tmem_dK0, advance_desc(desc_dS, 2), advance_desc(desc_Q0_N, 2), idesc_transA_transB, true);
                umma_f16_cg1_fn(tmem_dK0, advance_desc(desc_dS, 3), advance_desc(desc_Q0_N, 3), idesc_transA_transB, true);
                
                umma_commit_1sm_fn(bar_dK);
            }
            mbarrier_wait_fn(bar_dK, phase_dK); phase_dK ^= 1;

            if (row < 64) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_dK0 + row*64, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                uint32_t dk_base = b_idx * H * S * 128 + h_idx * S * 128 + (k_start + row) * 128;
                if (k_start + row < S) {
                    atomicAdd(&fp32_dK[dk_base + 0], __uint_as_float(r0));
                    atomicAdd(&fp32_dK[dk_base + 1], __uint_as_float(r1));
                    atomicAdd(&fp32_dK[dk_base + 2], __uint_as_float(r2));
                    atomicAdd(&fp32_dK[dk_base + 3], __uint_as_float(r3));
                }
                tmem_load_4x_fn(tmem_dK0 + row*64 + 32, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                if (k_start + row < S) {
                    atomicAdd(&fp32_dK[dk_base + 32], __uint_as_float(r0));
                    atomicAdd(&fp32_dK[dk_base + 33], __uint_as_float(r1));
                    atomicAdd(&fp32_dK[dk_base + 34], __uint_as_float(r2));
                    atomicAdd(&fp32_dK[dk_base + 35], __uint_as_float(r3));
                }
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                uint64_t desc_dS = desc_K_major(s_dS);
                uint64_t desc_Q1_N = desc_N_major(s_Q1);

                umma_f16_cg1_fn(tmem_dK1, desc_dS, desc_Q1_N, idesc_transA_transB, false);
                umma_f16_cg1_fn(tmem_dK1, advance_desc(desc_dS, 1), advance_desc(desc_Q1_N, 1), idesc_transA_transB, true);
                umma_f16_cg1_fn(tmem_dK1, advance_desc(desc_dS, 2), advance_desc(desc_Q1_N, 2), idesc_transA_transB, true);
                umma_f16_cg1_fn(tmem_dK1, advance_desc(desc_dS, 3), advance_desc(desc_Q1_N, 3), idesc_transA_transB, true);
                
                umma_commit_1sm_fn(bar_dK);
            }
            mbarrier_wait_fn(bar_dK, phase_dK); phase_dK ^= 1;

            if (row < 64) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_dK1 + row*64, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                uint32_t dk_base = b_idx * H * S * 128 + h_idx * S * 128 + (k_start + row) * 128;
                if (k_start + row < S) {
                    atomicAdd(&fp32_dK[dk_base + 64], __uint_as_float(r0));
                    atomicAdd(&fp32_dK[dk_base + 65], __uint_as_float(r1));
                    atomicAdd(&fp32_dK[dk_base + 66], __uint_as_float(r2));
                    atomicAdd(&fp32_dK[dk_base + 67], __uint_as_float(r3));
                }
                tmem_load_4x_fn(tmem_dK1 + row*64 + 32, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                if (k_start + row < S) {
                    atomicAdd(&fp32_dK[dk_base + 96], __uint_as_float(r0));
                    atomicAdd(&fp32_dK[dk_base + 97], __uint_as_float(r1));
                    atomicAdd(&fp32_dK[dk_base + 98], __uint_as_float(r2));
                    atomicAdd(&fp32_dK[dk_base + 99], __uint_as_float(r3));
                }
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                uint64_t desc_dS = desc_K_major(s_dS);
                uint64_t desc_K0_N = desc_N_major(s_K0);
                uint64_t desc_K1_N = desc_N_major(s_K1);

                umma_f16_cg1_fn(tmem_dQ0, desc_dS, desc_K0_N, idesc_transB, k_tile == 0 ? false : true, k_tile > 0);
                umma_f16_cg1_fn(tmem_dQ0, advance_desc(desc_dS, 1), advance_desc(desc_K0_N, 1), idesc_transB, true, k_tile > 0);
                umma_f16_cg1_fn(tmem_dQ0, advance_desc(desc_dS, 2), advance_desc(desc_K0_N, 2), idesc_transB, true, k_tile > 0);
                umma_f16_cg1_fn(tmem_dQ0, advance_desc(desc_dS, 3), advance_desc(desc_K0_N, 3), idesc_transB, true, k_tile > 0);

                umma_f16_cg1_fn(tmem_dQ1, desc_dS, desc_K1_N, idesc_transB, k_tile == 0 ? false : true, k_tile > 0);
                umma_f16_cg1_fn(tmem_dQ1, advance_desc(desc_dS, 1), advance_desc(desc_K1_N, 1), idesc_transB, true, k_tile > 0);
                umma_f16_cg1_fn(tmem_dQ1, advance_desc(desc_dS, 2), advance_desc(desc_K1_N, 2), idesc_transB, true, k_tile > 0);
                umma_f16_cg1_fn(tmem_dQ1, advance_desc(desc_dS, 3), advance_desc(desc_K1_N, 3), idesc_transB, true, k_tile > 0);
                
                umma_commit_1sm_fn(bar_dQ);
            }
            mbarrier_wait_fn(bar_dQ, phase_dQ); phase_dQ ^= 1;
        }
    }

    if (row < 64) {
        for(uint32_t col=0; col<64; col+=4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_dQ0 + row*64 + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            uint32_t dq_base = b_idx * H * S * 128 + h_idx * S * 128 + (q_start + row) * 128;
            if (q_start + row < S) {
                dQ[dq_base + col] = __float2bfloat16(__uint_as_float(r0));
                dQ[dq_base + col + 1] = __float2bfloat16(__uint_as_float(r1));
                dQ[dq_base + col + 2] = __float2bfloat16(__uint_as_float(r2));
                dQ[dq_base + col + 3] = __float2bfloat16(__uint_as_float(r3));
            }
        }
        for(uint32_t col=0; col<64; col+=4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_dQ1 + row*64 + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            uint32_t dq_base = b_idx * H * S * 128 + h_idx * S * 128 + (q_start + row) * 128;
            if (q_start + row < S) {
                dQ[dq_base + col + 64] = __float2bfloat16(__uint_as_float(r0));
                dQ[dq_base + col + 65] = __float2bfloat16(__uint_as_float(r1));
                dQ[dq_base + col + 66] = __float2bfloat16(__uint_as_float(r2));
                dQ[dq_base + col + 67] = __float2bfloat16(__uint_as_float(r3));
            }
        }
    }

    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_S_T, 64);
        tmem_dealloc_fn(tmem_dP_T, 64);
        tmem_dealloc_fn(tmem_dQ0, 64);
        tmem_dealloc_fn(tmem_dQ1, 64);
        tmem_dealloc_fn(tmem_dK0, 64);
        tmem_dealloc_fn(tmem_dK1, 64);
        tmem_dealloc_fn(tmem_dV0, 64);
    }
}

__global__ void convert_fp32_to_bf16(const float* src, __nv_bfloat16* dst, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, 
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t d = Q.size(3);
    
    if (S == 0) return;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), d, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), d, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), d, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_O, O.data_ptr(), d, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dO, dO.data_ptr(), d, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    uint32_t size_elements = B * H * S * d;
    float* fp32_dK;
    float* fp32_dV;
    CUDA_CHECK(cudaMallocAsync(&fp32_dK, size_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&fp32_dV, size_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(fp32_dK, 0, size_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(fp32_dV, 0, size_elements * sizeof(float), stream));
    
    uint32_t num_q_tiles = (S + 63) / 64;
    dim3 grid(num_q_tiles, H, B);
    dim3 block(128);
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
    mha_bwd_kernel<<<grid, block, 98304, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO, 
        static_cast<const float*>(L.data_ptr()), 
        static_cast<__nv_bfloat16*>(dQ.data_ptr()), 
        fp32_dK, fp32_dV, 
        S, H, B
    );
    CUDA_CHECK(cudaGetLastError());
    
    uint32_t conv_threads = 256;
    uint32_t conv_blocks = (size_elements + conv_threads - 1) / conv_threads;
    convert_fp32_to_bf16<<<conv_blocks, conv_threads, 0, stream>>>(fp32_dK, static_cast<__nv_bfloat16*>(dK.data_ptr()), size_elements);
    convert_fp32_to_bf16<<<conv_blocks, conv_threads, 0, stream>>>(fp32_dV, static_cast<__nv_bfloat16*>(dV.data_ptr()), size_elements);
    
    CUDA_CHECK(cudaFreeAsync(fp32_dK, stream));
    CUDA_CHECK(cudaFreeAsync(fp32_dV, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd