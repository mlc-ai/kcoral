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
        const char* err_name;                                      \
        cuGetErrorName(_e, &err_name);                             \
        fprintf(stderr, "CU error %s at %s:%d\n",                 \
                err_name, __FILE__, __LINE__);                     \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd_d128 {

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_none(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 61;
    return d;
}

__device__ __forceinline__ uint64_t desc_k_major(void* ptr, uint32_t C) {
    return make_smem_desc_none(ptr, 16, 16 * C);
}

__device__ __forceinline__ uint64_t desc_mn_major(void* ptr, uint32_t C) {
    return make_smem_desc_none(ptr, 16, 2 * C);
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N, bool a_mn_major, bool b_mn_major) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= ((a_mn_major ? 1u : 0u) << 15);
    d |= ((b_mn_major ? 1u : 0u) << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3, uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),"=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

__device__ __forceinline__ void tmem_epilogue_4w_fn(__nv_bfloat16* D, __nv_bfloat16* smem_out, uint32_t S, uint32_t m_block, uint32_t tmem_col) {
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_col + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * 128 + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    for (uint32_t step = 0; step < 16; ++step) {
        uint32_t row = step * 4 + warp_id;
        uint32_t global_row = m_block * 64 + row;
        uint32_t col_start = lane_id * 4;
        if (global_row < S) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * 128 + col_start]);
            *reinterpret_cast<uint2*>(D + (uint64_t)global_row * 128 + col_start) = data;
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void tmem_epilogue_atomic_4w_fn(__nv_bfloat16* D, __nv_bfloat16* smem_out, uint32_t S, uint32_t m_block, uint32_t tmem_col) {
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_col + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * 128 + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    for (uint32_t step = 0; step < 16; ++step) {
        uint32_t row = step * 4 + warp_id;
        uint32_t global_row = m_block * 64 + row;
        uint32_t col_start = lane_id * 4;
        if (global_row < S) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * 128 + col_start]);
            __nv_bfloat162* p = (__nv_bfloat162*)(D + (uint64_t)global_row * 128 + col_start);
            __nv_bfloat162* v = (__nv_bfloat162*)&data;
            atomicAdd(p, v[0]);
            atomicAdd(p + 1, v[1]);
        }
    }
    __syncthreads();
}

__global__ void mha_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L_ptr,
    __nv_bfloat16* __restrict__ dQ_ptr,
    __nv_bfloat16* __restrict__ dK_ptr,
    __nv_bfloat16* __restrict__ dV_ptr,
    uint32_t S
) {
    uint32_t bh = blockIdx.y;
    uint32_t i = blockIdx.x; 
    
    extern __shared__ uint8_t smem[];
    __nv_bfloat16* smem_K = (__nv_bfloat16*)smem;                 
    __nv_bfloat16* smem_V = smem_K + 64 * 128;                    
    __nv_bfloat16* smem_Q = smem_V + 64 * 128;                    
    __nv_bfloat16* smem_O = smem_Q + 64 * 128;                    
    __nv_bfloat16* smem_dO = smem_O + 64 * 128;                   
    __nv_bfloat16* smem_P = smem_dO + 64 * 128;                   
    __nv_bfloat16* smem_dS_T = smem_P + 64 * 64;                  
    __nv_bfloat16* smem_dS = smem_dS_T + 64 * 64;                 
    __nv_bfloat16* smem_out = smem_dS + 64 * 64;                  
    
    uint64_t* mbar_kv = (uint64_t*)(smem_out + 64 * 128);
    uint64_t* mbar_q = mbar_kv + 1;
    uint64_t* mbar_mma = mbar_q + 1;
    
    L_ptr += bh * S;
    dQ_ptr += bh * S * 128;
    dK_ptr += bh * S * 128;
    dV_ptr += bh * S * 128;
    
    uint32_t tmem_alloc;
    if (threadIdx.x == 0) tmem_alloc_fn(&tmem_alloc, 256);
    __syncthreads();
    
    uint32_t TMEM_S = 0, TMEM_DP = 32, TMEM_DV = 64, TMEM_DK = 128, TMEM_DQ = 192;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_kv, 1);
        init_smem_barrier_fn(mbar_q, 1);
        init_smem_barrier_fn(mbar_mma, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_kv, 16384 * 2);
        tma_load_2d_fn(&tma_K, mbar_kv, smem_K, 0, i * 64 + bh * S);
        tma_load_2d_fn(&tma_V, mbar_kv, smem_V, 0, i * 64 + bh * S);
    }
    mbarrier_wait_fn(mbar_kv, 0);
    
    float D_j[64];
    float L_j_local[64];
    
    uint32_t instr_1 = make_instr_desc(64, 64, false, false);
    uint32_t instr_2 = make_instr_desc(64, 64, false, false);
    uint32_t instr_3 = make_instr_desc(64, 128, false, true);
    uint32_t instr_4 = make_instr_desc(64, 128, false, true);
    uint32_t instr_5 = make_instr_desc(64, 128, true, true);
    
    uint32_t num_q_blocks = (S + 63) / 64;
    for (uint32_t j = 0; j < num_q_blocks; ++j) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_q, 16384 * 3);
            tma_load_2d_fn(&tma_Q, mbar_q, smem_Q, 0, j * 64 + bh * S);
            tma_load_2d_fn(&tma_O, mbar_q, smem_O, 0, j * 64 + bh * S);
            tma_load_2d_fn(&tma_dO, mbar_q, smem_dO, 0, j * 64 + bh * S);
        }
        
        if (threadIdx.x < 64) {
            uint32_t row = j * 64 + threadIdx.x;
            L_j_local[threadIdx.x] = (row < S) ? L_ptr[row] : -1e20f;
        }
        
        mbarrier_wait_fn(mbar_q, j & 1);
        
        if (threadIdx.x < 64) {
            float sum = 0.0f;
            uint32_t row = threadIdx.x;
            for (int c = 0; c < 128; c += 2) {
                __nv_bfloat162 o  = *(__nv_bfloat162*)&smem_O[row * 128 + c];
                __nv_bfloat162 do_ = *(__nv_bfloat162*)&smem_dO[row * 128 + c];
                sum += __bfloat162float(o.x) * __bfloat162float(do_.x) + __bfloat162float(o.y) * __bfloat162float(do_.y);
            }
            D_j[row] = sum;
        }
        __syncthreads();
        fence_async_shared_fn();
        
        for (int k = 0; k < 128; k += 16) {
            uint32_t accum = (k > 0) ? 1 : 0;
            uint32_t ak = (uint32_t)__cvta_generic_to_shared(smem_K + k);
            uint32_t bq = (uint32_t)__cvta_generic_to_shared(smem_Q + k);
            umma_f16_cg1_fn(TMEM_S, desc_k_major((void*)ak, 128), desc_k_major((void*)bq, 128), instr_1, accum);
            
            uint32_t av = (uint32_t)__cvta_generic_to_shared(smem_V + k);
            uint32_t bdo = (uint32_t)__cvta_generic_to_shared(smem_dO + k);
            umma_f16_cg1_fn(TMEM_DP, desc_k_major((void*)av, 128), desc_k_major((void*)bdo, 128), instr_2, accum);
        }
        
        if (threadIdx.x == 0) umma_commit_1sm_fn(mbar_mma);
        mbarrier_wait_fn(mbar_mma, (j * 2) & 1);
        
        for (int c = 0; c < 64; c += 8) {
            uint32_t rS[8], rDP[8];
            tmem_load_8x_fn(TMEM_S + c, &rS[0], &rS[1], &rS[2], &rS[3], &rS[4], &rS[5], &rS[6], &rS[7]);
            tmem_load_8x_fn(TMEM_DP + c, &rDP[0], &rDP[1], &rDP[2], &rDP[3], &rDP[4], &rDP[5], &rDP[6], &rDP[7]);
            tmem_load_fence_fn();
            
            if (threadIdx.x < 64) {
                uint32_t t = threadIdx.x;
                float p[8], ds[8];
                for (int m = 0; m < 8; m++) {
                    float s = __uint_as_float(rS[m]);
                    float dp = __uint_as_float(rDP[m]);
                    float lse = L_j_local[c + m];
                    float d_val = D_j[c + m];
                    p[m] = fast_exp2f_fn((s - lse) * 1.44269504f);
                    ds[m] = p[m] * (dp - d_val);
                }
                
                uint32_t p01 = pack_bf16_fn(__float_as_uint(p[0]), __float_as_uint(p[1]));
                uint32_t p23 = pack_bf16_fn(__float_as_uint(p[2]), __float_as_uint(p[3]));
                uint32_t p45 = pack_bf16_fn(__float_as_uint(p[4]), __float_as_uint(p[5]));
                uint32_t p67 = pack_bf16_fn(__float_as_uint(p[6]), __float_as_uint(p[7]));
                uint32_t addr_P = (uint32_t)__cvta_generic_to_shared(&smem_P[t * 64 + c]);
                st_shared_128_fn(addr_P, p01, p23, p45, p67);
                
                uint32_t ds01 = pack_bf16_fn(__float_as_uint(ds[0]), __float_as_uint(ds[1]));
                uint32_t ds23 = pack_bf16_fn(__float_as_uint(ds[2]), __float_as_uint(ds[3]));
                uint32_t ds45 = pack_bf16_fn(__float_as_uint(ds[4]), __float_as_uint(ds[5]));
                uint32_t ds67 = pack_bf16_fn(__float_as_uint(ds[6]), __float_as_uint(ds[7]));
                uint32_t addr_dS_T = (uint32_t)__cvta_generic_to_shared(&smem_dS_T[t * 64 + c]);
                st_shared_128_fn(addr_dS_T, ds01, ds23, ds45, ds67);
                
                for (int m = 0; m < 8; m++) {
                    smem_dS[(c + m) * 64 + t] = __float2bfloat16(ds[m]);
                }
            }
        }
        __syncthreads();
        fence_async_shared_fn();
        
        for (int k = 0; k < 64; k += 16) {
            uint32_t accum_v_k = (j == 0 && k == 0) ? 0 : 1;
            uint32_t accum_q = (k == 0) ? 0 : 1;
            
            uint32_t ap = (uint32_t)__cvta_generic_to_shared(smem_P + k);
            uint32_t bdo = (uint32_t)__cvta_generic_to_shared(smem_dO + k * 128);
            umma_f16_cg1_fn(TMEM_DV, desc_k_major((void*)ap, 64), desc_mn_major((void*)bdo, 128), instr_3, accum_v_k);
            
            uint32_t ads_t = (uint32_t)__cvta_generic_to_shared(smem_dS_T + k);
            uint32_t bq = (uint32_t)__cvta_generic_to_shared(smem_Q + k * 128);
            umma_f16_cg1_fn(TMEM_DK, desc_k_major((void*)ads_t, 64), desc_mn_major((void*)bq, 128), instr_4, accum_v_k);
            
            uint32_t ads = (uint32_t)__cvta_generic_to_shared(smem_dS + k * 64); 
            uint32_t bk = (uint32_t)__cvta_generic_to_shared(smem_K + k * 128);
            umma_f16_cg1_fn(TMEM_DQ, desc_mn_major((void*)ads, 64), desc_mn_major((void*)bk, 128), instr_5, accum_q);
        }
        
        if (threadIdx.x == 0) umma_commit_1sm_fn(mbar_mma);
        mbarrier_wait_fn(mbar_mma, (j * 2 + 1) & 1);
        
        tmem_epilogue_atomic_4w_fn(dQ_ptr, smem_out, S, j, TMEM_DQ);
    }
    
    tmem_epilogue_4w_fn(dK_ptr, smem_out, S, i, TMEM_DK);
    tmem_epilogue_4w_fn(dV_ptr, smem_out, S, i, TMEM_DV);
    
    if (threadIdx.x == 0) tmem_dealloc_fn(tmem_alloc, 256);
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

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    __nv_bfloat16* do_ptr = static_cast<__nv_bfloat16*>(dO.data_ptr());
    float* l_ptr = static_cast<float*>(L.data_ptr());
    
    __nv_bfloat16* dq_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dk_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dv_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaMemsetAsync(dq_ptr, 0, B * H * S * D * sizeof(__nv_bfloat16), stream));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, q_ptr, 128, B * H * S, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, k_ptr, 128, B * H * S, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, v_ptr, 128, B * H * S, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_O, o_ptr, 128, B * H * S, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_dO, do_ptr, 128, B * H * S, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    uint32_t num_kv_blocks = (S + 63) / 64;
    dim3 grid(num_kv_blocks, B * H, 1);
    dim3 block(128, 1, 1);
    uint32_t smem_bytes = 160 * 1024; 
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    mha_bwd_kernel<<<grid, block, smem_bytes, stream>>>(tma_Q, tma_K, tma_V, tma_O, tma_dO, l_ptr, dq_ptr, dk_ptr, dv_ptr, S);
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}