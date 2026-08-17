#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <stdio.h>
#include <math.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

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

__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_commit(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a) : "memory");
}

__device__ __forceinline__ void umma_f16(uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, bool accumulate) {
    uint32_t p = accumulate ? 1 : 0;
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(p));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100(void* smem_ptr, bool trans) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    
    uint32_t lbo = trans ? 8192 : 0; // For physical row-major, MN-Major uses LBO=8192
    uint32_t sbo = 1024;
    
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N, bool trans_a, bool trans_b) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= ((trans_a ? 1u : 0u) << 15);
    d |= ((trans_b ? 1u : 0u) << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
     "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
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

CUresult create_tma_2d_descriptor(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim) {
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
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

__global__ void compute_D_kernel(
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    float* __restrict__ D,
    int B, int H, int S, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * H * S) {
        float sum = 0;
        const __nv_bfloat16* o_ptr = O + idx * 128;
        const __nv_bfloat16* do_ptr = dO + idx * 128;
        for (int i = 0; i < 128; i++) {
            sum += __bfloat162float(o_ptr[i]) * __bfloat162float(do_ptr[i]);
        }
        D[idx] = sum;
    }
}

__global__ void mha_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ D,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int S, float scale)
{
    int k_start = blockIdx.x * 64;
    int head_idx = blockIdx.y;
    if (k_start >= S) return;
    
    __shared__ uint32_t tmem_S, tmem_dP, tmem_dQ0, tmem_dQ1, tmem_dK0, tmem_dK1, tmem_dV0, tmem_dV1;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_S, 64);
        tmem_alloc_fn(&tmem_dP, 64);
        tmem_alloc_fn(&tmem_dQ0, 64);
        tmem_alloc_fn(&tmem_dQ1, 64);
        tmem_alloc_fn(&tmem_dK0, 64);
        tmem_alloc_fn(&tmem_dK1, 64);
        tmem_alloc_fn(&tmem_dV0, 64);
        tmem_alloc_fn(&tmem_dV1, 64);
    }
    __syncthreads();
    
    alignas(1024) __shared__ __nv_bfloat16 smem_Q0[64][64];
    alignas(1024) __shared__ __nv_bfloat16 smem_Q1[64][64];
    alignas(1024) __shared__ __nv_bfloat16 smem_K0[64][64];
    alignas(1024) __shared__ __nv_bfloat16 smem_K1[64][64];
    alignas(1024) __shared__ __nv_bfloat16 smem_V0[64][64];
    alignas(1024) __shared__ __nv_bfloat16 smem_V1[64][64];
    alignas(1024) __shared__ __nv_bfloat16 smem_dO0[64][64];
    alignas(1024) __shared__ __nv_bfloat16 smem_dO1[64][64];
    
    alignas(1024) __shared__ uint4 smem_P[64][8]; 
    alignas(1024) __shared__ uint4 smem_dS[64][8]; 
    
    __shared__ uint64_t mbar_K, mbar_V, mbar_Q, mbar_dO, mbar_umma;
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_K, 1);
        init_smem_barrier_fn(&mbar_V, 1);
        init_smem_barrier_fn(&mbar_Q, 1);
        init_smem_barrier_fn(&mbar_dO, 1);
        init_smem_barrier_fn(&mbar_umma, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_K, 64 * 128 * 2);
        tma_load_2d_fn(&tma_K, &mbar_K, smem_K0, 0, head_idx * S + k_start);
        tma_load_2d_fn(&tma_K, &mbar_K, smem_K1, 64, head_idx * S + k_start);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_V, 64 * 128 * 2);
        tma_load_2d_fn(&tma_V, &mbar_V, smem_V0, 0, head_idx * S + k_start);
        tma_load_2d_fn(&tma_V, &mbar_V, smem_V1, 64, head_idx * S + k_start);
    }
    mbarrier_wait_fn(&mbar_K, 0);
    mbarrier_wait_fn(&mbar_V, 0);
    
    int row = threadIdx.x; 
    uint32_t umma_phase = 0;
    uint32_t q_phase = 0;
    
    for (int q_start = 0; q_start < S; q_start += 64) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_Q, 64 * 128 * 2);
            tma_load_2d_fn(&tma_Q, &mbar_Q, smem_Q0, 0, head_idx * S + q_start);
            tma_load_2d_fn(&tma_Q, &mbar_Q, smem_Q1, 64, head_idx * S + q_start);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar_dO, 64 * 128 * 2);
            tma_load_2d_fn(&tma_dO, &mbar_dO, smem_dO0, 0, head_idx * S + q_start);
            tma_load_2d_fn(&tma_dO, &mbar_dO, smem_dO1, 64, head_idx * S + q_start);
        }
        
        float l_val = 0, d_val = 0;
        if (row < 64 && q_start + row < S) {
            l_val = L[head_idx * S + q_start + row];
            d_val = D[head_idx * S + q_start + row];
        }
        
        mbarrier_wait_fn(&mbar_Q, q_phase);
        mbarrier_wait_fn(&mbar_dO, q_phase);
        q_phase ^= 1;
        
        if (threadIdx.x == 0) {
            for (int k_step = 0; k_step < 4; k_step++) {
                uint64_t desc_Q0 = make_smem_desc_sm100(&smem_Q0[0][k_step * 16], false);
                uint64_t desc_K0 = make_smem_desc_sm100(&smem_K0[0][k_step * 16], false);
                umma_f16(tmem_S, desc_Q0, desc_K0, make_instr_desc(64, 64, false, false), k_step > 0);
                
                uint64_t desc_dO0 = make_smem_desc_sm100(&smem_dO0[0][k_step * 16], false);
                uint64_t desc_V0 = make_smem_desc_sm100(&smem_V0[0][k_step * 16], false);
                umma_f16(tmem_dP, desc_dO0, desc_V0, make_instr_desc(64, 64, false, false), k_step > 0);
            }
            for (int k_step = 0; k_step < 4; k_step++) {
                uint64_t desc_Q1 = make_smem_desc_sm100(&smem_Q1[0][k_step * 16], false);
                uint64_t desc_K1 = make_smem_desc_sm100(&smem_K1[0][k_step * 16], false);
                umma_f16(tmem_S, desc_Q1, desc_K1, make_instr_desc(64, 64, false, false), true); 
                
                uint64_t desc_dO1 = make_smem_desc_sm100(&smem_dO1[0][k_step * 16], false);
                uint64_t desc_V1 = make_smem_desc_sm100(&smem_V1[0][k_step * 16], false);
                umma_f16(tmem_dP, desc_dO1, desc_V1, make_instr_desc(64, 64, false, false), true); 
            }
            tcgen05_commit(&mbar_umma);
        }
        mbarrier_wait_fn(&mbar_umma, umma_phase);
        umma_phase ^= 1;
        
        for (int col = 0; col < 64; col += 8) {
            uint32_t rs[8], rdp[8];
            tmem_load_8x_fn(tmem_S + col, &rs[0], &rs[1], &rs[2], &rs[3], &rs[4], &rs[5], &rs[6], &rs[7]);
            tmem_load_8x_fn(tmem_dP + col, &rdp[0], &rdp[1], &rdp[2], &rdp[3], &rdp[4], &rdp[5], &rdp[6], &rdp[7]);
            tcgen05_wait_ld();
            
            if (row < 64) {
                uint32_t vp[4], vds[4];
                for(int c = 0; c < 4; c++) {
                    float s0 = __uint_as_float(rs[c*2]), s1 = __uint_as_float(rs[c*2+1]);
                    float dp0 = __uint_as_float(rdp[c*2]), dp1 = __uint_as_float(rdp[c*2+1]);
                    
                    float p0 = 0, p1 = 0, ds0 = 0, ds1 = 0;
                    if (q_start + row < S) {
                        if (k_start + col + c * 2 < S) {
                            p0 = expf(s0 * scale - l_val);
                            ds0 = p0 * (dp0 - d_val) * scale;
                        }
                        if (k_start + col + c * 2 + 1 < S) {
                            p1 = expf(s1 * scale - l_val);
                            ds1 = p1 * (dp1 - d_val) * scale;
                        }
                    }
                    vp[c] = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
                    vds[c] = pack_bf16_fn(__float_as_uint(ds0), __float_as_uint(ds1));
                }
                
                int span_idx = col / 8;
                int swizzled_x = (row % 8) ^ span_idx;
                smem_P[row][swizzled_x] = make_uint4(vp[0], vp[1], vp[2], vp[3]);
                smem_dS[row][swizzled_x] = make_uint4(vds[0], vds[1], vds[2], vds[3]);
            }
        }
        __syncthreads();
        fence_async_shared_fn();
        
        if (threadIdx.x == 0) {
            for (int k_step = 0; k_step < 4; k_step++) {
                uint64_t desc_dS = make_smem_desc_sm100(&smem_dS[0][k_step * 2], false); 
                uint64_t desc_K0 = make_smem_desc_sm100(&smem_K0[k_step * 16][0], true);
                umma_f16(tmem_dQ0, desc_dS, desc_K0, make_instr_desc(64, 64, false, true), k_step > 0);
                
                uint64_t desc_K1 = make_smem_desc_sm100(&smem_K1[k_step * 16][0], true);
                umma_f16(tmem_dQ1, desc_dS, desc_K1, make_instr_desc(64, 64, false, true), k_step > 0);
                
                uint64_t desc_dS_T = make_smem_desc_sm100(&smem_dS[k_step * 16][0], true); 
                uint64_t desc_Q0 = make_smem_desc_sm100(&smem_Q0[k_step * 16][0], true);
                umma_f16(tmem_dK0, desc_dS_T, desc_Q0, make_instr_desc(64, 64, true, true), k_step > 0 || q_start > 0);
                
                uint64_t desc_Q1 = make_smem_desc_sm100(&smem_Q1[k_step * 16][0], true);
                umma_f16(tmem_dK1, desc_dS_T, desc_Q1, make_instr_desc(64, 64, true, true), k_step > 0 || q_start > 0);
                
                uint64_t desc_P_T = make_smem_desc_sm100(&smem_P[k_step * 16][0], true);
                uint64_t desc_dO0 = make_smem_desc_sm100(&smem_dO0[k_step * 16][0], true);
                umma_f16(tmem_dV0, desc_P_T, desc_dO0, make_instr_desc(64, 64, true, true), k_step > 0 || q_start > 0);
                
                uint64_t desc_dO1 = make_smem_desc_sm100(&smem_dO1[k_step * 16][0], true);
                umma_f16(tmem_dV1, desc_P_T, desc_dO1, make_instr_desc(64, 64, true, true), k_step > 0 || q_start > 0);
            }
            tcgen05_commit(&mbar_umma);
        }
        mbarrier_wait_fn(&mbar_umma, umma_phase);
        umma_phase ^= 1;
        
        for (int col = 0; col < 64; col += 8) {
            uint32_t rq0[8], rq1[8];
            tmem_load_8x_fn(tmem_dQ0 + col, &rq0[0], &rq0[1], &rq0[2], &rq0[3], &rq0[4], &rq0[5], &rq0[6], &rq0[7]);
            tmem_load_8x_fn(tmem_dQ1 + col, &rq1[0], &rq1[1], &rq1[2], &rq1[3], &rq1[4], &rq1[5], &rq1[6], &rq1[7]);
            tcgen05_wait_ld();
            
            if (row < 64 && q_start + row < S) {
                uint32_t v0 = pack_bf16_fn(rq0[0], rq0[1]);
                uint32_t v1 = pack_bf16_fn(rq0[2], rq0[3]);
                uint32_t v2 = pack_bf16_fn(rq0[4], rq0[5]);
                uint32_t v3 = pack_bf16_fn(rq0[6], rq0[7]);
                
                __nv_bfloat162 val0 = *reinterpret_cast<__nv_bfloat162*>(&v0);
                __nv_bfloat162 val1 = *reinterpret_cast<__nv_bfloat162*>(&v1);
                __nv_bfloat162 val2 = *reinterpret_cast<__nv_bfloat162*>(&v2);
                __nv_bfloat162 val3 = *reinterpret_cast<__nv_bfloat162*>(&v3);
                
                __nv_bfloat162* dq_ptr0 = reinterpret_cast<__nv_bfloat162*>(dQ + head_idx * S * 128 + (q_start + row) * 128 + col);
                atomicAdd(&dq_ptr0[0], val0);
                atomicAdd(&dq_ptr0[1], val1);
                atomicAdd(&dq_ptr0[2], val2);
                atomicAdd(&dq_ptr0[3], val3);
                    
                v0 = pack_bf16_fn(rq1[0], rq1[1]);
                v1 = pack_bf16_fn(rq1[2], rq1[3]);
                v2 = pack_bf16_fn(rq1[4], rq1[5]);
                v3 = pack_bf16_fn(rq1[6], rq1[7]);
                
                val0 = *reinterpret_cast<__nv_bfloat162*>(&v0);
                val1 = *reinterpret_cast<__nv_bfloat162*>(&v1);
                val2 = *reinterpret_cast<__nv_bfloat162*>(&v2);
                val3 = *reinterpret_cast<__nv_bfloat162*>(&v3);
                
                __nv_bfloat162* dq_ptr1 = reinterpret_cast<__nv_bfloat162*>(dQ + head_idx * S * 128 + (q_start + row) * 128 + 64 + col);
                atomicAdd(&dq_ptr1[0], val0);
                atomicAdd(&dq_ptr1[1], val1);
                atomicAdd(&dq_ptr1[2], val2);
                atomicAdd(&dq_ptr1[3], val3);
            }
        }
        __syncthreads();
    }
    
    for (int col = 0; col < 64; col += 8) {
        uint32_t rk0[8], rk1[8], rv0[8], rv1[8];
        tmem_load_8x_fn(tmem_dK0 + col, &rk0[0], &rk0[1], &rk0[2], &rk0[3], &rk0[4], &rk0[5], &rk0[6], &rk0[7]);
        tmem_load_8x_fn(tmem_dK1 + col, &rk1[0], &rk1[1], &rk1[2], &rk1[3], &rk1[4], &rk1[5], &rk1[6], &rk1[7]);
        tmem_load_8x_fn(tmem_dV0 + col, &rv0[0], &rv0[1], &rv0[2], &rv0[3], &rv0[4], &rv0[5], &rv0[6], &rv0[7]);
        tmem_load_8x_fn(tmem_dV1 + col, &rv1[0], &rv1[1], &rv1[2], &rv1[3], &rv1[4], &rv1[5], &rv1[6], &rv1[7]);
        tcgen05_wait_ld();
        
        if (row < 64 && k_start + row < S) {
            __nv_bfloat16* dk_ptr = dK + head_idx * S * 128 + (k_start + row) * 128;
            __nv_bfloat16* dv_ptr = dV + head_idx * S * 128 + (k_start + row) * 128;
            
            uint4 vk0 = make_uint4(pack_bf16_fn(rk0[0], rk0[1]), pack_bf16_fn(rk0[2], rk0[3]), pack_bf16_fn(rk0[4], rk0[5]), pack_bf16_fn(rk0[6], rk0[7]));
            uint4 vk1 = make_uint4(pack_bf16_fn(rk1[0], rk1[1]), pack_bf16_fn(rk1[2], rk1[3]), pack_bf16_fn(rk1[4], rk1[5]), pack_bf16_fn(rk1[6], rk1[7]));
            uint4 vv0 = make_uint4(pack_bf16_fn(rv0[0], rv0[1]), pack_bf16_fn(rv0[2], rv0[3]), pack_bf16_fn(rv0[4], rv0[5]), pack_bf16_fn(rv0[6], rv0[7]));
            uint4 vv1 = make_uint4(pack_bf16_fn(rv1[0], rv1[1]), pack_bf16_fn(rv1[2], rv1[3]), pack_bf16_fn(rv1[4], rv1[5]), pack_bf16_fn(rv1[6], rv1[7]));
            
            *(uint4*)(dk_ptr + col) = vk0;
            *(uint4*)(dk_ptr + 64 + col) = vk1;
            *(uint4*)(dv_ptr + col) = vv0;
            *(uint4*)(dv_ptr + 64 + col) = vv1;
        }
    }
    
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_S, 64);
        tmem_dealloc_fn(tmem_dP, 64);
        tmem_dealloc_fn(tmem_dQ0, 64);
        tmem_dealloc_fn(tmem_dQ1, 64);
        tmem_dealloc_fn(tmem_dK0, 64);
        tmem_dealloc_fn(tmem_dK1, 64);
        tmem_dealloc_fn(tmem_dV0, 64);
        tmem_dealloc_fn(tmem_dV1, 64);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int d = Q.size(3); 
    
    float scale = 1.0f / sqrtf((float)d);
    
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
    
    CUDA_CHECK(cudaMemsetAsync(dq_ptr, 0, B * H * S * d * sizeof(__nv_bfloat16), stream));
    
    float* D_ptr;
    CUDA_CHECK(cudaMallocAsync(&D_ptr, B * H * S * sizeof(float), stream));
    
    int num_threads = 256;
    int num_blocks = (B * H * S + num_threads - 1) / num_threads;
    compute_D_kernel<<<num_blocks, num_threads, 0, stream>>>(o_ptr, do_ptr, D_ptr, B, H, S, d);
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_dO;
    if (create_tma_2d_descriptor(&tma_Q, (void*)q_ptr, d, B * H * S, 64, 64) != 0 ||
        create_tma_2d_descriptor(&tma_K, (void*)k_ptr, d, B * H * S, 64, 64) != 0 ||
        create_tma_2d_descriptor(&tma_V, (void*)v_ptr, d, B * H * S, 64, 64) != 0 ||
        create_tma_2d_descriptor(&tma_dO, (void*)do_ptr, d, B * H * S, 64, 64) != 0) {
        fprintf(stderr, "TMA descriptor creation failed\n");
        exit(1);
    }
    
    dim3 grid((S + 63) / 64, B * H, 1);
    dim3 block(128, 1, 1);
    mha_bwd_kernel<<<grid, block, 0, stream>>>(tma_Q, tma_K, tma_V, tma_dO, D_ptr, l_ptr, dq_ptr, dk_ptr, dv_ptr, S, scale);
    
    CUDA_CHECK(cudaFreeAsync(D_ptr, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda