#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
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

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tma_load_5d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3, int32_t c4) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4) : "memory");
}

__device__ __forceinline__ void tma_store_5d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3, int32_t c4) {
    asm volatile(
        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4, %5, %6}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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

__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N, uint32_t a_maj, uint32_t b_maj) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (a_maj << 15);   
    d |= (b_maj << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint64_t advance_desc_swizzle(uint64_t desc, uint32_t bytes) {
    uint32_t addr = (desc & 0x3FFF) << 4;
    addr += bytes;
    desc = (desc & ~0x3FFFull) | ((addr >> 4) & 0x3FFF);
    uint64_t base_offset = (addr >> 7) & 0x7;
    desc = (desc & ~(0x7ull << 49)) | (base_offset << 49);
    return desc;
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ __nv_bfloat16* get_swizzled_ptr(__nv_bfloat16* base, uint32_t row, uint32_t col) {
    uint32_t c_bytes = col * 2;
    uint32_t c_swizzled = c_bytes ^ ((row % 8) * 16);
    return (__nv_bfloat16*)((char*)base + row * 128 + c_swizzled);
}

__device__ __forceinline__ void compute_S_ij(uint32_t tmem_s, void* Q0, void* Q1, void* K0, void* K1, bool accumulate) {
    uint32_t idesc = make_idesc(128, 128, 0, 0);
    
    uint64_t desc_a0 = make_smem_desc_sm100_fn(Q0, 1, 1024);
    uint64_t desc_b0 = make_smem_desc_sm100_fn(K0, 1, 1024);
    for (int k = 0; k < 64; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a0, k * 2);
        uint64_t db = advance_desc_swizzle(desc_b0, k * 2);
        umma_f16_cg1_fn(tmem_s, da, db, idesc, accumulate ? 1 : (k == 0 ? 0 : 1));
    }
    
    uint64_t desc_a1 = make_smem_desc_sm100_fn(Q1, 1, 1024);
    uint64_t desc_b1 = make_smem_desc_sm100_fn(K1, 1, 1024);
    for (int k = 0; k < 64; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a1, k * 2);
        uint64_t db = advance_desc_swizzle(desc_b1, k * 2);
        umma_f16_cg1_fn(tmem_s, da, db, idesc, 1);
    }
}

__device__ __forceinline__ void compute_dQ(uint32_t tmem_dq, void* dS0, void* dS1, void* K0, void* K1, bool accumulate) {
    uint32_t idesc = make_idesc(128, 64, 0, 1);
    
    uint64_t desc_a0 = make_smem_desc_sm100_fn(dS0, 1, 1024);
    uint64_t desc_b0 = make_smem_desc_sm100_fn(K0, 16384, 1024);
    for (int k = 0; k < 64; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a0, k * 2);
        uint64_t db = advance_desc_swizzle(desc_b0, k * 128);
        umma_f16_cg1_fn(tmem_dq, da, db, idesc, accumulate ? 1 : (k == 0 ? 0 : 1));
    }
    uint64_t desc_a1 = make_smem_desc_sm100_fn(dS1, 1, 1024);
    uint64_t base_b1 = (uint64_t)K0 + 64 * 128;
    uint64_t desc_b1 = make_smem_desc_sm100_fn((void*)base_b1, 16384, 1024);
    for (int k = 0; k < 64; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a1, k * 2);
        uint64_t db = advance_desc_swizzle(desc_b1, k * 128);
        umma_f16_cg1_fn(tmem_dq, da, db, idesc, 1);
    }
    
    uint32_t tmem_dq1 = tmem_dq + 64;
    desc_b0 = make_smem_desc_sm100_fn(K1, 16384, 1024);
    for (int k = 0; k < 64; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a0, k * 2);
        uint64_t db = advance_desc_swizzle(desc_b0, k * 128);
        umma_f16_cg1_fn(tmem_dq1, da, db, idesc, accumulate ? 1 : (k == 0 ? 0 : 1));
    }
    base_b1 = (uint64_t)K1 + 64 * 128;
    desc_b1 = make_smem_desc_sm100_fn((void*)base_b1, 16384, 1024);
    for (int k = 0; k < 64; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a1, k * 2);
        uint64_t db = advance_desc_swizzle(desc_b1, k * 128);
        umma_f16_cg1_fn(tmem_dq1, da, db, idesc, 1);
    }
}

__device__ __forceinline__ void compute_dK(uint32_t tmem_dk, void* dS0, void* dS1, void* Q0, void* Q1, bool accumulate) {
    uint32_t idesc = make_idesc(64, 64, 1, 1);
    
    uint32_t tmem_dk0_top = tmem_dk;
    uint32_t tmem_dk0_bot = tmem_dk + (64 << 16);
    uint64_t desc_a0 = make_smem_desc_sm100_fn(dS0, 16384, 1024);
    uint64_t desc_b0 = make_smem_desc_sm100_fn(Q0, 16384, 1024);
    for (int k = 0; k < 128; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a0, k * 128);
        uint64_t db = advance_desc_swizzle(desc_b0, k * 128);
        umma_f16_cg1_fn(tmem_dk0_top, da, db, idesc, accumulate ? 1 : (k == 0 ? 0 : 1));
    }
    uint64_t desc_a1 = make_smem_desc_sm100_fn(dS1, 16384, 1024);
    for (int k = 0; k < 128; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a1, k * 128);
        uint64_t db = advance_desc_swizzle(desc_b0, k * 128);
        umma_f16_cg1_fn(tmem_dk0_bot, da, db, idesc, accumulate ? 1 : (k == 0 ? 0 : 1));
    }
    
    uint32_t tmem_dk1_top = tmem_dk + 64;
    uint32_t tmem_dk1_bot = tmem_dk + 64 + (64 << 16);
    uint64_t desc_b1 = make_smem_desc_sm100_fn(Q1, 16384, 1024);
    for (int k = 0; k < 128; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a0, k * 128);
        uint64_t db = advance_desc_swizzle(desc_b1, k * 128);
        umma_f16_cg1_fn(tmem_dk1_top, da, db, idesc, accumulate ? 1 : (k == 0 ? 0 : 1));
    }
    for (int k = 0; k < 128; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a1, k * 128);
        uint64_t db = advance_desc_swizzle(desc_b1, k * 128);
        umma_f16_cg1_fn(tmem_dk1_bot, da, db, idesc, accumulate ? 1 : (k == 0 ? 0 : 1));
    }
}

__global__ __launch_bounds__(128, 1)
void bwd_dq_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dQ,
    const float* L_ptr, int S
) {
    int i = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    
    __shared__ uint32_t smem_tmem[1];
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&smem_tmem[0], 512);
    }
    __syncthreads();
    uint32_t tmem_base = smem_tmem[0];
    uint32_t tmem_dQ = tmem_base;
    uint32_t tmem_S  = tmem_base + 128;
    uint32_t tmem_dP = tmem_base + 256;
    
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem_pool;
    __nv_bfloat16* smem_Q1 = smem_Q0 + 128 * 64;
    __nv_bfloat16* smem_K0 = smem_Q1 + 128 * 64;
    __nv_bfloat16* smem_K1 = smem_K0 + 128 * 64;
    __nv_bfloat16* smem_V0 = smem_K1 + 128 * 64;
    __nv_bfloat16* smem_V1 = smem_V0 + 128 * 64;
    __nv_bfloat16* smem_O0 = smem_V1 + 128 * 64;
    __nv_bfloat16* smem_O1 = smem_O0 + 128 * 64;
    __nv_bfloat16* smem_dO0 = smem_O1 + 128 * 64;
    __nv_bfloat16* smem_dO1 = smem_dO0 + 128 * 64;
    __nv_bfloat16* smem_dS0 = smem_dO1 + 128 * 64;
    __nv_bfloat16* smem_dS1 = smem_dS0 + 128 * 64;

    uint64_t* mbar_Q = (uint64_t*)(smem_dS1 + 128 * 64);
    uint64_t* mbar_K = mbar_Q + 1;
    uint64_t* mbar_V = mbar_K + 1;
    uint64_t* mbar_O = mbar_V + 1;
    uint64_t* mbar_dO = mbar_O + 1;
    uint64_t* mbar_mma = mbar_dO + 1;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_O, 1);
        init_smem_barrier_fn(mbar_dO, 1);
        init_smem_barrier_fn(mbar_mma, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
        tma_load_5d_fn(&tma_Q, mbar_Q, smem_Q0, 0, 0, i * 128, h, b);
        tma_load_5d_fn(&tma_Q, mbar_Q, smem_Q1, 0, 1, i * 128, h, b);
        
        mbarrier_arrive_and_expect_tx_fn(mbar_O, 32768);
        tma_load_5d_fn(&tma_O, mbar_O, smem_O0, 0, 0, i * 128, h, b);
        tma_load_5d_fn(&tma_O, mbar_O, smem_O1, 0, 1, i * 128, h, b);
        
        mbarrier_arrive_and_expect_tx_fn(mbar_dO, 32768);
        tma_load_5d_fn(&tma_dO, mbar_dO, smem_dO0, 0, 0, i * 128, h, b);
        tma_load_5d_fn(&tma_dO, mbar_dO, smem_dO1, 0, 1, i * 128, h, b);
    }
    
    mbarrier_wait_fn(mbar_Q, 0);
    mbarrier_wait_fn(mbar_O, 0);
    mbarrier_wait_fn(mbar_dO, 0);
    
    float L_local = 0.0f;
    int seq_idx = i * 128 + threadIdx.x;
    if (seq_idx < S) {
        L_local = L_ptr[(b * gridDim.y + h) * S + seq_idx];
    }
    
    float scale = 1.0f / sqrtf(128.0f);
    float Delta_local = 0.0f;
    
    if (seq_idx < S) {
        for (int c = 0; c < 64; c += 8) {
            uint4 o_vec0 = *(uint4*)get_swizzled_ptr(smem_O0, threadIdx.x, c);
            uint4 do_vec0 = *(uint4*)get_swizzled_ptr(smem_dO0, threadIdx.x, c);
            __nv_bfloat16* o_arr0 = (__nv_bfloat16*)&o_vec0;
            __nv_bfloat16* do_arr0 = (__nv_bfloat16*)&do_vec0;
            for(int k=0; k<8; k++) Delta_local += __bfloat162float(o_arr0[k]) * __bfloat162float(do_arr0[k]);
            
            uint4 o_vec1 = *(uint4*)get_swizzled_ptr(smem_O1, threadIdx.x, c);
            uint4 do_vec1 = *(uint4*)get_swizzled_ptr(smem_dO1, threadIdx.x, c);
            __nv_bfloat16* o_arr1 = (__nv_bfloat16*)&o_vec1;
            __nv_bfloat16* do_arr1 = (__nv_bfloat16*)&do_vec1;
            for(int k=0; k<8; k++) Delta_local += __bfloat162float(o_arr1[k]) * __bfloat162float(do_arr1[k]);
        }
    }
    
    int phase_K = 0, phase_V = 0, phase_mma = 0;
    bool first_dq = true;
    for (int j = 0; j <= i; j++) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K, 32768);
            tma_load_5d_fn(&tma_K, mbar_K, smem_K0, 0, 0, j * 128, h, b);
            tma_load_5d_fn(&tma_K, mbar_K, smem_K1, 0, 1, j * 128, h, b);
            
            mbarrier_arrive_and_expect_tx_fn(mbar_V, 32768);
            tma_load_5d_fn(&tma_V, mbar_V, smem_V0, 0, 0, j * 128, h, b);
            tma_load_5d_fn(&tma_V, mbar_V, smem_V1, 0, 1, j * 128, h, b);
        }
        mbarrier_wait_fn(mbar_K, phase_K); phase_K ^= 1;
        mbarrier_wait_fn(mbar_V, phase_V); phase_V ^= 1;
        
        fence_async_shared_fn();
        if (threadIdx.x == 0) {
            compute_S_ij(tmem_S, smem_Q0, smem_Q1, smem_K0, smem_K1, false);
            compute_S_ij(tmem_dP, smem_dO0, smem_dO1, smem_V0, smem_V1, false);
            
            uint32_t mbar_mma_addr = (uint32_t)__cvta_generic_to_shared(mbar_mma);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_mma_addr));
        }
        mbarrier_wait_fn(mbar_mma, phase_mma); phase_mma ^= 1;
        
        for (int col = 0; col < 128; col += 4) {
            uint32_t s_r0, s_r1, s_r2, s_r3;
            uint32_t dp_r0, dp_r1, dp_r2, dp_r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(s_r0),"=r"(s_r1),"=r"(s_r2),"=r"(s_r3) : "r"(tmem_S + col));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(dp_r0),"=r"(dp_r1),"=r"(dp_r2),"=r"(dp_r3) : "r"(tmem_dP + col));
            tmem_load_fence_fn();
            
            float s_arr[4] = {__uint_as_float(s_r0), __uint_as_float(s_r1), __uint_as_float(s_r2), __uint_as_float(s_r3)};
            float dp_arr[4] = {__uint_as_float(dp_r0), __uint_as_float(dp_r1), __uint_as_float(dp_r2), __uint_as_float(dp_r3)};
            
            __nv_bfloat16 ds_bf16[4];
            for(int k=0; k<4; k++) {
                float p_val = 0.0f;
                float ds_val = 0.0f;
                bool valid = (seq_idx < S) && (j * 128 + col + k < S);
                if (i == j && threadIdx.x < col + k) valid = false;
                
                if (valid) {
                    float s_val = s_arr[k] * scale;
                    p_val = expf(s_val - L_local);
                    ds_val = p_val * (dp_arr[k] - Delta_local) * scale;
                }
                ds_bf16[k] = __float2bfloat16(ds_val);
            }
            
            if (col < 64) {
                *(uint2*)get_swizzled_ptr(smem_dS0, threadIdx.x, col) = *(uint2*)ds_bf16;
            } else {
                *(uint2*)get_swizzled_ptr(smem_dS1, threadIdx.x, col - 64) = *(uint2*)ds_bf16;
            }
        }
        
        __syncthreads();
        fence_async_shared_fn();
        
        if (threadIdx.x == 0) {
            compute_dQ(tmem_dQ, smem_dS0, smem_dS1, smem_K0, smem_K1, !first_dq);
            
            uint32_t mbar_mma_addr = (uint32_t)__cvta_generic_to_shared(mbar_mma);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_mma_addr));
        }
        first_dq = false;
        mbarrier_wait_fn(mbar_mma, phase_mma); phase_mma ^= 1;
    }
    
    for (int col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dQ + col));
        tmem_load_fence_fn();
        __nv_bfloat16 dq_bf16[4];
        dq_bf16[0] = __float2bfloat16(__uint_as_float(r0));
        dq_bf16[1] = __float2bfloat16(__uint_as_float(r1));
        dq_bf16[2] = __float2bfloat16(__uint_as_float(r2));
        dq_bf16[3] = __float2bfloat16(__uint_as_float(r3));
        
        if (col < 64) {
            *(uint2*)get_swizzled_ptr(smem_Q0, threadIdx.x, col) = *(uint2*)dq_bf16;
        } else {
            *(uint2*)get_swizzled_ptr(smem_Q1, threadIdx.x, col - 64) = *(uint2*)dq_bf16;
        }
    }
    __syncthreads();
    fence_async_shared_fn();
    
    if (threadIdx.x == 0) {
        tma_store_5d_fn(&tma_dQ, smem_Q0, 0, 0, i * 128, h, b);
        tma_store_5d_fn(&tma_dQ, smem_Q1, 0, 1, i * 128, h, b);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 512);
    }
}

__global__ __launch_bounds__(128, 1)
void bwd_dkv_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dK,
    const __grid_constant__ CUtensorMap tma_dV,
    const float* L_ptr, int S
) {
    int j = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    
    __shared__ uint32_t smem_tmem[1];
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&smem_tmem[0], 512);
    }
    __syncthreads();
    uint32_t tmem_base = smem_tmem[0];
    uint32_t tmem_dK = tmem_base;
    uint32_t tmem_dV = tmem_base + 128;
    uint32_t tmem_S  = tmem_base + 256;
    uint32_t tmem_dP = tmem_base + 384;
    
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem_pool;
    __nv_bfloat16* smem_Q1 = smem_Q0 + 128 * 64;
    __nv_bfloat16* smem_K0 = smem_Q1 + 128 * 64;
    __nv_bfloat16* smem_K1 = smem_K0 + 128 * 64;
    __nv_bfloat16* smem_V0 = smem_K1 + 128 * 64;
    __nv_bfloat16* smem_V1 = smem_V0 + 128 * 64;
    __nv_bfloat16* smem_O0 = smem_V1 + 128 * 64;
    __nv_bfloat16* smem_O1 = smem_O0 + 128 * 64;
    __nv_bfloat16* smem_dO0 = smem_O1 + 128 * 64;
    __nv_bfloat16* smem_dO1 = smem_dO0 + 128 * 64;
    __nv_bfloat16* smem_dS0 = smem_dO1 + 128 * 64;
    __nv_bfloat16* smem_dS1 = smem_dS0 + 128 * 64;
    __nv_bfloat16* smem_P0 = smem_dS1 + 128 * 64;
    __nv_bfloat16* smem_P1 = smem_P0 + 128 * 64;

    uint64_t* mbar_Q = (uint64_t*)(smem_P1 + 128 * 64);
    uint64_t* mbar_K = mbar_Q + 1;
    uint64_t* mbar_V = mbar_K + 1;
    uint64_t* mbar_O = mbar_V + 1;
    uint64_t* mbar_dO = mbar_O + 1;
    uint64_t* mbar_mma = mbar_dO + 1;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_O, 1);
        init_smem_barrier_fn(mbar_dO, 1);
        init_smem_barrier_fn(mbar_mma, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_K, 32768);
        tma_load_5d_fn(&tma_K, mbar_K, smem_K0, 0, 0, j * 128, h, b);
        tma_load_5d_fn(&tma_K, mbar_K, smem_K1, 0, 1, j * 128, h, b);
        
        mbarrier_arrive_and_expect_tx_fn(mbar_V, 32768);
        tma_load_5d_fn(&tma_V, mbar_V, smem_V0, 0, 0, j * 128, h, b);
        tma_load_5d_fn(&tma_V, mbar_V, smem_V1, 0, 1, j * 128, h, b);
    }
    mbarrier_wait_fn(mbar_K, 0);
    mbarrier_wait_fn(mbar_V, 0);
    
    float scale = 1.0f / sqrtf(128.0f);
    int phase_Q = 0, phase_O = 0, phase_dO = 0, phase_mma = 0;
    bool first_dkv = true;
    int num_i = (S + 127) / 128;
    
    for (int i = j; i < num_i; i++) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
            tma_load_5d_fn(&tma_Q, mbar_Q, smem_Q0, 0, 0, i * 128, h, b);
            tma_load_5d_fn(&tma_Q, mbar_Q, smem_Q1, 0, 1, i * 128, h, b);
            
            mbarrier_arrive_and_expect_tx_fn(mbar_O, 32768);
            tma_load_5d_fn(&tma_O, mbar_O, smem_O0, 0, 0, i * 128, h, b);
            tma_load_5d_fn(&tma_O, mbar_O, smem_O1, 0, 1, i * 128, h, b);
            
            mbarrier_arrive_and_expect_tx_fn(mbar_dO, 32768);
            tma_load_5d_fn(&tma_dO, mbar_dO, smem_dO0, 0, 0, i * 128, h, b);
            tma_load_5d_fn(&tma_dO, mbar_dO, smem_dO1, 0, 1, i * 128, h, b);
        }
        mbarrier_wait_fn(mbar_Q, phase_Q); phase_Q ^= 1;
        mbarrier_wait_fn(mbar_O, phase_O); phase_O ^= 1;
        mbarrier_wait_fn(mbar_dO, phase_dO); phase_dO ^= 1;
        
        float L_local = 0.0f;
        int seq_idx = i * 128 + threadIdx.x;
        if (seq_idx < S) {
            L_local = L_ptr[(b * gridDim.y + h) * S + seq_idx];
        }
        
        float Delta_local = 0.0f;
        if (seq_idx < S) {
            for (int c = 0; c < 64; c += 8) {
                uint4 o_vec0 = *(uint4*)get_swizzled_ptr(smem_O0, threadIdx.x, c);
                uint4 do_vec0 = *(uint4*)get_swizzled_ptr(smem_dO0, threadIdx.x, c);
                __nv_bfloat16* o_arr0 = (__nv_bfloat16*)&o_vec0;
                __nv_bfloat16* do_arr0 = (__nv_bfloat16*)&do_vec0;
                for(int k=0; k<8; k++) Delta_local += __bfloat162float(o_arr0[k]) * __bfloat162float(do_arr0[k]);
                
                uint4 o_vec1 = *(uint4*)get_swizzled_ptr(smem_O1, threadIdx.x, c);
                uint4 do_vec1 = *(uint4*)get_swizzled_ptr(smem_dO1, threadIdx.x, c);
                __nv_bfloat16* o_arr1 = (__nv_bfloat16*)&o_vec1;
                __nv_bfloat16* do_arr1 = (__nv_bfloat16*)&do_vec1;
                for(int k=0; k<8; k++) Delta_local += __bfloat162float(o_arr1[k]) * __bfloat162float(do_arr1[k]);
            }
        }
        
        fence_async_shared_fn();
        if (threadIdx.x == 0) {
            compute_S_ij(tmem_S, smem_Q0, smem_Q1, smem_K0, smem_K1, false);
            compute_S_ij(tmem_dP, smem_dO0, smem_dO1, smem_V0, smem_V1, false);
            
            uint32_t mbar_mma_addr = (uint32_t)__cvta_generic_to_shared(mbar_mma);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_mma_addr));
        }
        mbarrier_wait_fn(mbar_mma, phase_mma); phase_mma ^= 1;
        
        for (int col = 0; col < 128; col += 4) {
            uint32_t s_r0, s_r1, s_r2, s_r3;
            uint32_t dp_r0, dp_r1, dp_r2, dp_r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(s_r0),"=r"(s_r1),"=r"(s_r2),"=r"(s_r3) : "r"(tmem_S + col));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(dp_r0),"=r"(dp_r1),"=r"(dp_r2),"=r"(dp_r3) : "r"(tmem_dP + col));
            tmem_load_fence_fn();
            
            float s_arr[4] = {__uint_as_float(s_r0), __uint_as_float(s_r1), __uint_as_float(s_r2), __uint_as_float(s_r3)};
            float dp_arr[4] = {__uint_as_float(dp_r0), __uint_as_float(dp_r1), __uint_as_float(dp_r2), __uint_as_float(dp_r3)};
            
            __nv_bfloat16 p_bf16[4], ds_bf16[4];
            for(int k=0; k<4; k++) {
                float p_val = 0.0f;
                float ds_val = 0.0f;
                bool valid = (seq_idx < S) && (j * 128 + col + k < S);
                if (i == j && threadIdx.x < col + k) valid = false;
                
                if (valid) {
                    float s_val = s_arr[k] * scale;
                    p_val = expf(s_val - L_local);
                    ds_val = p_val * (dp_arr[k] - Delta_local) * scale;
                }
                
                p_bf16[k] = __float2bfloat16(p_val);
                ds_bf16[k] = __float2bfloat16(ds_val);
            }
            
            if (col < 64) {
                *(uint2*)get_swizzled_ptr(smem_P0, threadIdx.x, col) = *(uint2*)p_bf16;
                *(uint2*)get_swizzled_ptr(smem_dS0, threadIdx.x, col) = *(uint2*)ds_bf16;
            } else {
                *(uint2*)get_swizzled_ptr(smem_P1, threadIdx.x, col - 64) = *(uint2*)p_bf16;
                *(uint2*)get_swizzled_ptr(smem_dS1, threadIdx.x, col - 64) = *(uint2*)ds_bf16;
            }
        }
        
        __syncthreads();
        fence_async_shared_fn();
        
        if (threadIdx.x == 0) {
            compute_dK(tmem_dK, smem_dS0, smem_dS1, smem_Q0, smem_Q1, !first_dkv);
            compute_dK(tmem_dV, smem_P0, smem_P1, smem_dO0, smem_dO1, !first_dkv);
            
            uint32_t mbar_mma_addr = (uint32_t)__cvta_generic_to_shared(mbar_mma);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_mma_addr));
        }
        first_dkv = false;
        mbarrier_wait_fn(mbar_mma, phase_mma); phase_mma ^= 1;
    }
    
    for (int col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dK + col));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_dV + col));
        tmem_load_fence_fn();
        __nv_bfloat16 dk_bf16[4], dv_bf16[4];
        dk_bf16[0] = __float2bfloat16(__uint_as_float(r0));
        dk_bf16[1] = __float2bfloat16(__uint_as_float(r1));
        dk_bf16[2] = __float2bfloat16(__uint_as_float(r2));
        dk_bf16[3] = __float2bfloat16(__uint_as_float(r3));
        dv_bf16[0] = __float2bfloat16(__uint_as_float(r4));
        dv_bf16[1] = __float2bfloat16(__uint_as_float(r5));
        dv_bf16[2] = __float2bfloat16(__uint_as_float(r6));
        dv_bf16[3] = __float2bfloat16(__uint_as_float(r7));
        
        if (col < 64) {
            *(uint2*)get_swizzled_ptr(smem_K0, threadIdx.x, col) = *(uint2*)dk_bf16;
            *(uint2*)get_swizzled_ptr(smem_V0, threadIdx.x, col) = *(uint2*)dv_bf16;
        } else {
            *(uint2*)get_swizzled_ptr(smem_K1, threadIdx.x, col - 64) = *(uint2*)dk_bf16;
            *(uint2*)get_swizzled_ptr(smem_V1, threadIdx.x, col - 64) = *(uint2*)dv_bf16;
        }
    }
    __syncthreads();
    fence_async_shared_fn();
    
    if (threadIdx.x == 0) {
        tma_store_5d_fn(&tma_dK, smem_K0, 0, 0, j * 128, h, b);
        tma_store_5d_fn(&tma_dK, smem_K1, 0, 1, j * 128, h, b);
        tma_store_5d_fn(&tma_dV, smem_V0, 0, 0, j * 128, h, b);
        tma_store_5d_fn(&tma_dV, smem_V1, 0, 1, j * 128, h, b);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 512);
    }
}

CUresult create_tma_5d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3, uint64_t dim4,
                                     uint64_t stride1, uint64_t stride2, uint64_t stride3, uint64_t stride4,
                                     uint32_t box0, uint32_t box1, uint32_t box2, uint32_t box3, uint32_t box4,
                                     CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[5] = {dim0, dim1, dim2, dim3, dim4};
    cuuint64_t globalStrides[4] = {stride1, stride2, stride3, stride4};
    cuuint32_t boxDim[5] = {box0, box1, box2, box3, box4};
    cuuint32_t elementStrides[5] = {1, 1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

int64_t get_stride_bytes(tvm::ffi::TensorView t, int dim) {
    const DLTensor* dlt = t.operator->();
    if (dlt->strides) {
        return dlt->strides[dim] * 2;
    }
    int64_t stride = 2;
    for (int i = dlt->ndim - 1; i > dim; --i) {
        stride *= dlt->shape[i];
    }
    return stride;
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    uint64_t stride_S_Q = get_stride_bytes(Q, 2);
    uint64_t stride_H_Q = get_stride_bytes(Q, 1);
    uint64_t stride_B_Q = get_stride_bytes(Q, 0);
    
    uint64_t stride_S_K = get_stride_bytes(K, 2);
    uint64_t stride_H_K = get_stride_bytes(K, 1);
    uint64_t stride_B_K = get_stride_bytes(K, 0);
    
    uint64_t stride_S_V = get_stride_bytes(V, 2);
    uint64_t stride_H_V = get_stride_bytes(V, 1);
    uint64_t stride_B_V = get_stride_bytes(V, 0);
    
    uint64_t stride_S_O = get_stride_bytes(O, 2);
    uint64_t stride_H_O = get_stride_bytes(O, 1);
    uint64_t stride_B_O = get_stride_bytes(O, 0);
    
    uint64_t stride_S_dO = get_stride_bytes(dO, 2);
    uint64_t stride_H_dO = get_stride_bytes(dO, 1);
    uint64_t stride_B_dO = get_stride_bytes(dO, 0);
    
    uint64_t stride_S_dQ = get_stride_bytes(dQ, 2);
    uint64_t stride_H_dQ = get_stride_bytes(dQ, 1);
    uint64_t stride_B_dQ = get_stride_bytes(dQ, 0);
    
    uint64_t stride_S_dK = get_stride_bytes(dK, 2);
    uint64_t stride_H_dK = get_stride_bytes(dK, 1);
    uint64_t stride_B_dK = get_stride_bytes(dK, 0);
    
    uint64_t stride_S_dV = get_stride_bytes(dV, 2);
    uint64_t stride_H_dV = get_stride_bytes(dV, 1);
    uint64_t stride_B_dV = get_stride_bytes(dV, 0);
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO, tma_dQ, tma_dK, tma_dV;
    
    CU_CHECK(create_tma_5d_descriptor_2B(&tma_Q, Q.data_ptr(), 64, 2, S, H, B, 128, stride_S_Q, stride_H_Q, stride_B_Q, 64, 1, 128, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_5d_descriptor_2B(&tma_K, K.data_ptr(), 64, 2, S, H, B, 128, stride_S_K, stride_H_K, stride_B_K, 64, 1, 128, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_5d_descriptor_2B(&tma_V, V.data_ptr(), 64, 2, S, H, B, 128, stride_S_V, stride_H_V, stride_B_V, 64, 1, 128, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_5d_descriptor_2B(&tma_O, O.data_ptr(), 64, 2, S, H, B, 128, stride_S_O, stride_H_O, stride_B_O, 64, 1, 128, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_5d_descriptor_2B(&tma_dO, dO.data_ptr(), 64, 2, S, H, B, 128, stride_S_dO, stride_H_dO, stride_B_dO, 64, 1, 128, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_5d_descriptor_2B(&tma_dQ, dQ.data_ptr(), 64, 2, S, H, B, 128, stride_S_dQ, stride_H_dQ, stride_B_dQ, 64, 1, 128, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_5d_descriptor_2B(&tma_dK, dK.data_ptr(), 64, 2, S, H, B, 128, stride_S_dK, stride_H_dK, stride_B_dK, 64, 1, 128, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_5d_descriptor_2B(&tma_dV, dV.data_ptr(), 64, 2, S, H, B, 128, stride_S_dV, stride_H_dV, stride_B_dV, 64, 1, 128, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 230000));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 230000));
    
    dim3 grid((S + 127) / 128, H, B);
    dim3 block(128);
    
    bwd_dq_kernel<<<grid, block, 230000, stream>>>(tma_Q, tma_K, tma_V, tma_O, tma_dO, tma_dQ, (float*)L.data_ptr(), S);
    CUDA_CHECK(cudaGetLastError());
    
    bwd_dkv_kernel<<<grid, block, 230000, stream>>>(tma_Q, tma_K, tma_V, tma_O, tma_dO, tma_dK, tma_dV, (float*)L.data_ptr(), S);
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}