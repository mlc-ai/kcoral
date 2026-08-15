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
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n",                 \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

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

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
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

__device__ __forceinline__ void tma_expect_and_arrive(uint64_t* mbar, uint32_t bytes) {
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, bytes);
    } else {
        mbarrier_arrive_fn(mbar);
    }
}

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_copy_1d_g2s_fn(void const* gmem, uint64_t* mbar, void* smem, int32_t bytes) {
    uint32_t smem_mbar = (uint32_t)__cvta_generic_to_shared(mbar);
    uint32_t smem_ptr  = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :: "r"(smem_ptr), "l"(gmem), "r"(bytes), "r"(smem_mbar) : "memory");
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

__device__ __forceinline__ uint64_t advance_desc_k(uint64_t desc, uint32_t bytes) {
    uint32_t addr = (desc & 0x3FFF) << 4;
    addr += bytes;
    uint64_t new_desc = desc & ~0x3FFFull;
    new_desc |= (addr >> 4) & 0x3FFF;
    new_desc &= ~(0x7ull << 49);
    new_desc |= (uint64_t)((addr >> 7) & 0x7) << 49;
    return new_desc;
}

__device__ __forceinline__ void tcgen05_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a) : "memory");
}

__device__ __forceinline__ void issue_umma_loop(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    uint32_t K_dim, uint32_t accum) {
    for (uint32_t k = 0; k < K_dim; k += 16) {
        uint32_t current_accum = (k == 0) ? accum : 1;
        asm volatile(
            "{\n.reg .pred p;\n"
            "setp.ne.b32 p, %4, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(current_accum));
        
        desc_a = advance_desc_k(desc_a, 32);
        desc_b = advance_desc_k(desc_b, 32);
    }
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major_1span(void* smem_ptr) {
    uint32_t SBO = 1024;
    uint32_t LBO = 1;
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((LBO & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((SBO & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46; 
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major_2span(void* smem_ptr) {
    uint32_t SBO = 2048;
    uint32_t LBO = 1;
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((LBO & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((SBO & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46; 
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major_2span(void* smem_ptr) {
    uint32_t SBO = 2048;
    uint32_t LBO = 256;
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((LBO & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((SBO & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46; 
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t swizzle_idx_1span(uint32_t row, uint32_t col) {
    uint32_t chunk_in_span = col / 8;
    uint32_t swizzled_chunk = (row % 8) ^ chunk_in_span;
    uint32_t swizzled_col = swizzled_chunk * 8 + (col % 8);
    return row * 64 + swizzled_col;
}

__device__ __forceinline__ uint32_t swizzle_idx_2span(uint32_t row, uint32_t col) {
    uint32_t span_idx = col / 64;
    uint32_t col_in_span = col % 64;
    uint32_t chunk_in_span = col_in_span / 8;
    uint32_t swizzled_chunk = (row % 8) ^ chunk_in_span;
    uint32_t swizzled_col = span_idx * 64 + swizzled_chunk * 8 + (col % 8);
    return row * 128 + swizzled_col;
}

__device__ __forceinline__ void tmem_softmax_to_smem_fn(
    uint32_t tmem_base_col, __nv_bfloat16* smem_PT, float* smem_LSE) {
    int tid = threadIdx.x;
    for (int c = 0; c < 64; c += 4) {
        uint32_t col = tmem_base_col + c;
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float s0 = __uint_as_float(r0) * 0.088388347f;
        float s1 = __uint_as_float(r1) * 0.088388347f;
        float s2 = __uint_as_float(r2) * 0.088388347f;
        float s3 = __uint_as_float(r3) * 0.088388347f;
        
        float lse0 = smem_LSE[c + 0];
        float lse1 = smem_LSE[c + 1];
        float lse2 = smem_LSE[c + 2];
        float lse3 = smem_LSE[c + 3];
        
        float p0 = fast_exp2f_fn((s0 - lse0) * 1.44269504f);
        float p1 = fast_exp2f_fn((s1 - lse1) * 1.44269504f);
        float p2 = fast_exp2f_fn((s2 - lse2) * 1.44269504f);
        float p3 = fast_exp2f_fn((s3 - lse3) * 1.44269504f);
        
        uint32_t p01 = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
        uint32_t p23 = pack_bf16_fn(__float_as_uint(p2), __float_as_uint(p3));
        uint2 val = make_uint2(p01, p23);
        *(uint2*)&smem_PT[swizzle_idx_1span(tid, c)] = val;
    }
}

__device__ __forceinline__ void tmem_ds_to_smem_fn(
    uint32_t tmem_dP_col, __nv_bfloat16* smem_PT, float* smem_D,
    __nv_bfloat16* smem_dST, __nv_bfloat16* smem_dS) {
    int tid = threadIdx.x;
    for (int c = 0; c < 64; c += 4) {
        uint32_t col = tmem_dP_col + c;
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float dp0 = __uint_as_float(r0);
        float dp1 = __uint_as_float(r1);
        float dp2 = __uint_as_float(r2);
        float dp3 = __uint_as_float(r3);
        
        float d_val0 = smem_D[c + 0];
        float d_val1 = smem_D[c + 1];
        float d_val2 = smem_D[c + 2];
        float d_val3 = smem_D[c + 3];
        
        uint2 p_pack = *(uint2*)&smem_PT[swizzle_idx_1span(tid, c)];
        __nv_bfloat162 p01 = *(__nv_bfloat162*)&p_pack.x;
        __nv_bfloat162 p23 = *(__nv_bfloat162*)&p_pack.y;
        
        float p0 = __bfloat162float(p01.x);
        float p1 = __bfloat162float(p01.y);
        float p2 = __bfloat162float(p23.x);
        float p3 = __bfloat162float(p23.y);
        
        float scale = 0.088388347f;
        float ds0 = p0 * (dp0 - d_val0) * scale;
        float ds1 = p1 * (dp1 - d_val1) * scale;
        float ds2 = p2 * (dp2 - d_val2) * scale;
        float ds3 = p3 * (dp3 - d_val3) * scale;
        
        uint32_t ds01 = pack_bf16_fn(__float_as_uint(ds0), __float_as_uint(ds1));
        uint32_t ds23 = pack_bf16_fn(__float_as_uint(ds2), __float_as_uint(ds3));
        uint2 val = make_uint2(ds01, ds23);
        
        *(uint2*)&smem_dST[swizzle_idx_1span(tid, c)] = val;
        
        smem_dS[swizzle_idx_2span(c + 0, tid)] = __float2bfloat16(ds0);
        smem_dS[swizzle_idx_2span(c + 1, tid)] = __float2bfloat16(ds1);
        smem_dS[swizzle_idx_2span(c + 2, tid)] = __float2bfloat16(ds2);
        smem_dS[swizzle_idx_2span(c + 3, tid)] = __float2bfloat16(ds3);
    }
}

__device__ __forceinline__ void tmem_dq_to_smem_and_atomic_fn(
    uint32_t tmem_dq_col, __nv_bfloat16* smem_dQ, __nv_bfloat16* global_dQ) {
    int tid = threadIdx.x;
    for (int c = 0; c < 128; c += 4) {
        uint32_t col = tmem_dq_col + c;
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        if (tid < 64) {
            uint32_t q01 = pack_bf16_fn(r0, r1);
            uint32_t q23 = pack_bf16_fn(r2, r3);
            uint2 val = make_uint2(q01, q23);
            *(uint2*)&smem_dQ[swizzle_idx_2span(tid, c)] = val;
        }
    }
    __syncthreads();
    
    int total_pairs = (64 * 128) / 2;
    for (int i = tid; i < total_pairs; i += blockDim.x) {
        int r = (i * 2) / 128;
        int c = (i * 2) % 128;
        __nv_bfloat162 val = *(__nv_bfloat162*)&smem_dQ[swizzle_idx_2span(r, c)];
        atomicAdd((__nv_bfloat162*)global_dQ + i, val);
    }
    __syncthreads();
}

__device__ __forceinline__ void tmem_store_coalesced_fn(
    uint32_t tmem_col, __nv_bfloat16* smem_out, __nv_bfloat16* D) {
    int tid = threadIdx.x;
    for (int c = 0; c < 128; c += 4) {
        uint32_t col = tmem_col + c;
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t v01 = pack_bf16_fn(r0, r1);
        uint32_t v23 = pack_bf16_fn(r2, r3);
        uint2 val = make_uint2(v01, v23);
        *(uint2*)&smem_out[swizzle_idx_2span(tid, c)] = val;
    }
    __syncthreads();
    
    int total_pairs = (128 * 128) / 2;
    for (int i = tid; i < total_pairs; i += blockDim.x) {
        int r = (i * 2) / 128;
        int c = (i * 2) % 128;
        __nv_bfloat162 val = *(__nv_bfloat162*)&smem_out[swizzle_idx_2span(r, c)];
        ((__nv_bfloat162*)D)[i] = val;
    }
    __syncthreads();
}

union SharedStorage {
    struct {
        __nv_bfloat16 Q[2][64 * 128];
        __nv_bfloat16 dO[2][64 * 128];
        float D[2][64];
        float LSE[2][64];
        
        __nv_bfloat16 K[128 * 128];
        __nv_bfloat16 V[128 * 128];
        
        __nv_bfloat16 PT[128 * 64];
        __nv_bfloat16 dST[128 * 64];
        __nv_bfloat16 dS_dQ[64 * 128];
        
        uint64_t mbar_Q[2];
        uint64_t mbar_dO[2];
        uint64_t mbar_D[2];
        uint64_t mbar_LSE[2];
        uint64_t mbar_K;
        uint64_t mbar_V;
    };
    struct {
        __nv_bfloat16 epilogue_buf[128 * 128];
    };
};

__global__ void compute_D_kernel(const __nv_bfloat16* O, const __nv_bfloat16* dO, float* D, int d, int total_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < total_rows) {
        float sum = 0;
        const __nv_bfloat162* o_row = (const __nv_bfloat162*)(O + row * d);
        const __nv_bfloat162* do_row = (const __nv_bfloat162*)(dO + row * d);
        for (int i = 0; i < d / 2; i++) {
            __nv_bfloat162 o_val = o_row[i];
            __nv_bfloat162 do_val = do_row[i];
            sum += __bfloat162float(o_val.x) * __bfloat162float(do_val.x);
            sum += __bfloat162float(o_val.y) * __bfloat162float(do_val.y);
        }
        D[row] = sum;
    }
}

__global__ void bwd_kernel(
    const CUtensorMap tma_K, const CUtensorMap tma_V, 
    const CUtensorMap tma_Q, const CUtensorMap tma_dO,
    float* D_ptr, float* LSE_ptr, 
    __nv_bfloat16* dQ_ptr, __nv_bfloat16* dK_ptr, __nv_bfloat16* dV_ptr,
    int S) {
    
    extern __shared__ char dynamic_smem[];
    SharedStorage* smem = (SharedStorage*)dynamic_smem;
    __shared__ uint32_t tmem_base;
    __shared__ __align__(8) uint64_t mbar_umma;

    int n_block = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int M_blocks = S / 64;
    int tid = threadIdx.x;
    
    if (tid < 32) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(&tmem_base)), "r"(512));
    }
    if (tid == 0) {
        init_smem_barrier_fn(&smem->mbar_K, 128);
        init_smem_barrier_fn(&smem->mbar_V, 128);
        for (int i=0; i<2; i++) {
            init_smem_barrier_fn(&smem->mbar_Q[i], 128);
            init_smem_barrier_fn(&smem->mbar_dO[i], 128);
            init_smem_barrier_fn(&smem->mbar_D[i], 128);
            init_smem_barrier_fn(&smem->mbar_LSE[i], 128);
        }
        init_smem_barrier_fn(&mbar_umma, 1);
    }
    __syncthreads();
    if (tid == 0) fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t phase_K = 0, phase_V = 0, phase_umma = 0;
    uint32_t phase_Q[2] = {0, 0};
    uint32_t phase_dO[2] = {0, 0};
    uint32_t phase_D[2] = {0, 0};
    uint32_t phase_LSE[2] = {0, 0};

    tma_expect_and_arrive(&smem->mbar_K, 128 * 128 * 2);
    tma_expect_and_arrive(&smem->mbar_V, 128 * 128 * 2);
    tma_expect_and_arrive(&smem->mbar_Q[0], 64 * 128 * 2);
    tma_expect_and_arrive(&smem->mbar_dO[0], 64 * 128 * 2);
    tma_expect_and_arrive(&smem->mbar_D[0], 64 * 4);
    tma_expect_and_arrive(&smem->mbar_LSE[0], 64 * 4);
    
    if (tid == 0) {
        int kv_row = b * gridDim.y * S + h * S + n_block * 128;
        tma_load_3d_fn(&tma_K, &smem->mbar_K, smem->K, 0, 0, kv_row);
        tma_load_3d_fn(&tma_V, &smem->mbar_V, smem->V, 0, 0, kv_row);
        
        int q_row = b * gridDim.y * S + h * S;
        tma_load_3d_fn(&tma_Q, &smem->mbar_Q[0], smem->Q[0], 0, 0, q_row);
        tma_load_3d_fn(&tma_dO, &smem->mbar_dO[0], smem->dO[0], 0, 0, q_row);
        
        tma_copy_1d_g2s_fn((const char*)D_ptr + q_row * 4, &smem->mbar_D[0], smem->D[0], 256);
        tma_copy_1d_g2s_fn((const char*)LSE_ptr + q_row * 4, &smem->mbar_LSE[0], smem->LSE[0], 256);
        
        if (M_blocks > 1) {
            mbarrier_arrive_and_expect_tx_fn(&smem->mbar_Q[1], 64 * 128 * 2);
            mbarrier_arrive_and_expect_tx_fn(&smem->mbar_dO[1], 64 * 128 * 2);
            mbarrier_arrive_and_expect_tx_fn(&smem->mbar_D[1], 64 * 4);
            mbarrier_arrive_and_expect_tx_fn(&smem->mbar_LSE[1], 64 * 4);
            
            q_row += 64;
            tma_load_3d_fn(&tma_Q, &smem->mbar_Q[1], smem->Q[1], 0, 0, q_row);
            tma_load_3d_fn(&tma_dO, &smem->mbar_dO[1], smem->dO[1], 0, 0, q_row);
            tma_copy_1d_g2s_fn((const char*)D_ptr + q_row * 4, &smem->mbar_D[1], smem->D[1], 256);
            tma_copy_1d_g2s_fn((const char*)LSE_ptr + q_row * 4, &smem->mbar_LSE[1], smem->LSE[1], 256);
        }
    } else if (M_blocks > 1) {
        mbarrier_arrive_fn(&smem->mbar_Q[1]);
        mbarrier_arrive_fn(&smem->mbar_dO[1]);
        mbarrier_arrive_fn(&smem->mbar_D[1]);
        mbarrier_arrive_fn(&smem->mbar_LSE[1]);
    }

    mbarrier_wait_fn(&smem->mbar_K, phase_K);
    mbarrier_wait_fn(&smem->mbar_V, phase_V);
    mbarrier_wait_fn(&smem->mbar_Q[0], phase_Q[0]);
    mbarrier_wait_fn(&smem->mbar_dO[0], phase_dO[0]);
    mbarrier_wait_fn(&smem->mbar_D[0], phase_D[0]);
    mbarrier_wait_fn(&smem->mbar_LSE[0], phase_LSE[0]);
    phase_Q[0] ^= 1; phase_dO[0] ^= 1; phase_D[0] ^= 1; phase_LSE[0] ^= 1;

    uint32_t idesc_S  = make_instr_desc_fn(128, 64, 0, 1);
    uint32_t idesc_dP = make_instr_desc_fn(128, 64, 0, 1);
    uint32_t idesc_dV = make_instr_desc_fn(128, 128, 0, 0);
    uint32_t idesc_dK = make_instr_desc_fn(128, 128, 0, 0);
    uint32_t idesc_dQ = make_instr_desc_fn(64, 128, 0, 0);

    uint64_t desc_K = make_smem_desc_k_major_2span(smem->K);
    uint64_t desc_V = make_smem_desc_k_major_2span(smem->V);
    uint64_t desc_Q_0 = make_smem_desc_mn_major_2span(smem->Q[0]);
    
    issue_umma_loop(0, desc_K, desc_Q_0, idesc_S, 128, 0);
    if (tid == 0) tcgen05_commit_cg1_fn(&mbar_umma);
    mbarrier_wait_fn(&mbar_umma, phase_umma); phase_umma ^= 1;

    tmem_softmax_to_smem_fn(0, smem->PT, smem->LSE[0]);

    uint64_t desc_dO_0_mn = make_smem_desc_mn_major_2span(smem->dO[0]);
    issue_umma_loop(64, desc_V, desc_dO_0_mn, idesc_dP, 128, 0);
    uint64_t desc_PT = make_smem_desc_k_major_1span(smem->PT);
    uint64_t desc_dO_0_k = make_smem_desc_k_major_2span(smem->dO[0]);
    issue_umma_loop(128, desc_PT, desc_dO_0_k, idesc_dV, 128, 0);

    if (tid == 0) tcgen05_commit_cg1_fn(&mbar_umma);
    mbarrier_wait_fn(&mbar_umma, phase_umma); phase_umma ^= 1;

    tmem_ds_to_smem_fn(64, smem->PT, smem->D[0], smem->dST, smem->dS_dQ);

    for (int j = 1; j < M_blocks; j++) {
        int cur = j % 2;
        int next = (j + 1) % 2;
        
        mbarrier_wait_fn(&smem->mbar_Q[cur], phase_Q[cur]);
        mbarrier_wait_fn(&smem->mbar_dO[cur], phase_dO[cur]);
        mbarrier_wait_fn(&smem->mbar_D[cur], phase_D[cur]);
        mbarrier_wait_fn(&smem->mbar_LSE[cur], phase_LSE[cur]);
        phase_Q[cur] ^= 1; phase_dO[cur] ^= 1; phase_D[cur] ^= 1; phase_LSE[cur] ^= 1;

        uint64_t desc_dST = make_smem_desc_k_major_1span(smem->dST);
        uint64_t desc_Q_prev_k = make_smem_desc_k_major_2span(smem->Q[1 - cur]);
        issue_umma_loop(256, desc_dST, desc_Q_prev_k, idesc_dK, 64, (j == 1) ? 0 : 1);

        uint64_t desc_dS = make_smem_desc_k_major_2span(smem->dS_dQ);
        issue_umma_loop(384, desc_dS, desc_K, idesc_dQ, 128, 0);

        if (tid == 0) tcgen05_commit_cg1_fn(&mbar_umma);
        mbarrier_wait_fn(&mbar_umma, phase_umma); phase_umma ^= 1;

        __nv_bfloat16* global_dQ = dQ_ptr + (b * gridDim.y * S + h * S + (j - 1) * 64) * 128;
        tmem_dq_to_smem_and_atomic_fn(384, smem->dS_dQ, global_dQ);

        uint64_t desc_Q_cur_mn = make_smem_desc_mn_major_2span(smem->Q[cur]);
        issue_umma_loop(0, desc_K, desc_Q_cur_mn, idesc_S, 128, 0);

        if (tid == 0) tcgen05_commit_cg1_fn(&mbar_umma);
        mbarrier_wait_fn(&mbar_umma, phase_umma); phase_umma ^= 1;

        tmem_softmax_to_smem_fn(0, smem->PT, smem->LSE[cur]);

        if (j + 1 < M_blocks) {
            tma_expect_and_arrive(&smem->mbar_Q[next], 64 * 128 * 2);
            tma_expect_and_arrive(&smem->mbar_dO[next], 64 * 128 * 2);
            tma_expect_and_arrive(&smem->mbar_D[next], 64 * 4);
            tma_expect_and_arrive(&smem->mbar_LSE[next], 64 * 4);
            
            if (tid == 0) {
                int q_row = b * gridDim.y * S + h * S + (j + 1) * 64;
                tma_load_3d_fn(&tma_Q, &smem->mbar_Q[next], smem->Q[next], 0, 0, q_row);
                tma_load_3d_fn(&tma_dO, &smem->mbar_dO[next], smem->dO[next], 0, 0, q_row);
                tma_copy_1d_g2s_fn((const char*)D_ptr + q_row * 4, &smem->mbar_D[next], smem->D[next], 256);
                tma_copy_1d_g2s_fn((const char*)LSE_ptr + q_row * 4, &smem->mbar_LSE[next], smem->LSE[next], 256);
            }
        }

        uint64_t desc_dO_cur_mn = make_smem_desc_mn_major_2span(smem->dO[cur]);
        issue_umma_loop(64, desc_V, desc_dO_cur_mn, idesc_dP, 128, 0);
        uint64_t desc_dO_cur_k = make_smem_desc_k_major_2span(smem->dO[cur]);
        issue_umma_loop(128, desc_PT, desc_dO_cur_k, idesc_dV, 128, 1);

        if (tid == 0) tcgen05_commit_cg1_fn(&mbar_umma);
        mbarrier_wait_fn(&mbar_umma, phase_umma); phase_umma ^= 1;

        tmem_ds_to_smem_fn(64, smem->PT, smem->D[cur], smem->dST, smem->dS_dQ);
    }

    int last = (M_blocks - 1) % 2;
    uint64_t desc_dST = make_smem_desc_k_major_1span(smem->dST);
    uint64_t desc_Q_last_k = make_smem_desc_k_major_2span(smem->Q[last]);
    issue_umma_loop(256, desc_dST, desc_Q_last_k, idesc_dK, 64, (M_blocks == 1) ? 0 : 1);

    uint64_t desc_dS = make_smem_desc_k_major_2span(smem->dS_dQ);
    issue_umma_loop(384, desc_dS, desc_K, idesc_dQ, 128, 0);

    if (tid == 0) tcgen05_commit_cg1_fn(&mbar_umma);
    mbarrier_wait_fn(&mbar_umma, phase_umma); phase_umma ^= 1;

    __nv_bfloat16* global_dQ_last = dQ_ptr + (b * gridDim.y * S + h * S + (M_blocks - 1) * 64) * 128;
    tmem_dq_to_smem_and_atomic_fn(384, smem->dS_dQ, global_dQ_last);

    uint64_t v_offset = (b * gridDim.y * S + h * S + n_block * 128) * 128;
    tmem_store_coalesced_fn(128, smem->epilogue_buf, dV_ptr + v_offset);
    tmem_store_coalesced_fn(256, smem->epilogue_buf, dK_ptr + v_offset);

    if (tid < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :: "r"(tmem_base), "r"(512));
    }
}

CUresult create_tma_3d_descriptor(CUtensorMap* d, void* globalAddress, uint64_t total_rows, uint32_t box_rows) {
    cuuint64_t globalDim[3] = {64, 2, total_rows};
    cuuint64_t globalStrides[2] = {128, 256}; 
    cuuint32_t boxDim[3] = {64, 2, box_rows};
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
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int d = 128;
    int total_rows = B * H * S;

    float* D_ptr;
    CUDA_CHECK(cudaMallocAsync(&D_ptr, total_rows * sizeof(float), stream));
    
    int threads = 256;
    int blocks = (total_rows + threads - 1) / threads;
    compute_D_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        D_ptr, d, total_rows);

    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, total_rows * d * sizeof(__nv_bfloat16), stream));

    CUtensorMap tma_K, tma_V, tma_Q, tma_dO;
    CU_CHECK(create_tma_3d_descriptor(&tma_K, K.data_ptr(), total_rows, 128));
    CU_CHECK(create_tma_3d_descriptor(&tma_V, V.data_ptr(), total_rows, 128));
    CU_CHECK(create_tma_3d_descriptor(&tma_Q, Q.data_ptr(), total_rows, 64));
    CU_CHECK(create_tma_3d_descriptor(&tma_dO, dO.data_ptr(), total_rows, 64));

    dim3 grid(S / 128, H, B);
    dim3 block(128);
    int smem_size = sizeof(SharedStorage);
    cudaFuncSetAttribute(bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    bwd_kernel<<<grid, block, smem_size, stream>>>(
        tma_K, tma_V, tma_Q, tma_dO,
        D_ptr, static_cast<float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S);
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(D_ptr, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}