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
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                (int)_e, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

__device__ __forceinline__ int swizzle_128B(int row, int col) {
    return ((row % 8) ^ (col / 8)) * 8 + (col % 8);
}

__device__ __forceinline__ void atomicAdd_bf16(__nv_bfloat16* address, __nv_bfloat16 val) {
    float val_f = __bfloat162float(val);
    int32_t* addr_i = (int32_t*)((uintptr_t)address & ~1);
    bool is_odd = ((uintptr_t)address) & 1;
    
    while (true) {
        int32_t old = *addr_i;
        uint16_t old_bf = is_odd ? (old >> 16) : (old & 0xFFFF);
        float old_f = __bfloat162float(*(reinterpret_cast<__nv_bfloat16*>(&old_bf)));
        float new_f = old_f + val_f;
        __nv_bfloat16 new_bf16 = __float2bfloat16(new_f);
        uint16_t new_bf = *(reinterpret_cast<uint16_t*>(&new_bf16));
        
        int32_t new_val = is_odd ? (old & 0xFFFF) | (new_bf << 16) : (old & 0xFFFF0000) | new_bf;
        int32_t replaced = atomicCAS(addr_i, old, new_val);
        if (replaced == old) break;
    }
}

__device__ __forceinline__ uint32_t packed_bf16_from_floats(float a, float b) {
    __nv_bfloat162 res;
    res.x = __float2bfloat16(a);
    res.y = __float2bfloat16(b);
    uint32_t out;
    asm volatile("mov.b32 %0, {%1, %2};" : "=r"(out) : "h"(*(uint16_t*)&res.x), "h"(*(uint16_t*)&res.y));
    return out;
}

__device__ __forceinline__ void transpose_128x128(__nv_bfloat16* smem) {
    for (int col = threadIdx.x; col < 128; col += 128) {
        for (int row = 0; row < col; row++) {
            int sc_c = swizzle_128B(row, col);
            int sc_r = swizzle_128B(col, row);
            __nv_bfloat16 tmp = smem[row * 128 + sc_c];
            smem[row * 128 + sc_c] = smem[col * 128 + sc_r];
            smem[col * 128 + sc_r] = tmp;
        }
    }
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
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

__device__ __forceinline__ void commit_mbarrier(uint64_t* bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
}

__device__ __forceinline__ void tmem_alloc_cta(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cta(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void umma_cg1(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* smem_ptr) {
    return make_smem_desc(smem_ptr, 1, 1024);
}

__device__ __forceinline__ uint64_t make_smem_desc_n_major(void* smem_ptr) {
    return make_smem_desc(smem_ptr, 16384, 1024);
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N, int a_major, int b_major) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= ((uint32_t)a_major << 15);
    d |= ((uint32_t)b_major << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void gemm_kk(uint32_t tmem_base, uint64_t desc_a_base, uint64_t desc_b_base, bool accumulate) {
    for (int k = 0; k < 8; k++) {
        uint32_t idesc = make_instr_desc(128, 128, 0, 0);
        uint32_t acc = accumulate ? 1 : (k == 0 ? 0 : 1);
        umma_cg1(tmem_base + (k * 2), desc_a_base + (k * 2), desc_b_base + (k * 2), idesc, acc);
    }
}

__device__ __forceinline__ void gemm_kn(uint32_t tmem_base, uint64_t desc_a_base, uint64_t desc_b_base, bool accumulate) {
    for (int k = 0; k < 8; k++) {
        uint32_t idesc = make_instr_desc(128, 128, 0, 1);
        uint32_t acc = accumulate ? 1 : (k == 0 ? 0 : 1);
        umma_cg1(tmem_base + (k * 2), desc_a_base + (k * 2), desc_b_base + (k * 8), idesc, acc);
    }
}

__global__ __launch_bounds__(128)
void bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int S,
    float attn_scale,
    uint64_t batch_head_stride)
{
    int global_j = blockIdx.x * 128;
    int batch_head_idx = blockIdx.y;

    if (global_j >= S) return;

    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* s_k = (__nv_bfloat16*)(((uintptr_t)(smem_pool) + 1023) & ~1023);
    __nv_bfloat16* s_v = (__nv_bfloat16*)(smem_pool + 32768);                
    __nv_bfloat16* s_q = (__nv_bfloat16*)(smem_pool + 65536);                
    __nv_bfloat16* s_o = (__nv_bfloat16*)(smem_pool + 98304);                
    __nv_bfloat16* s_do = (__nv_bfloat16*)(smem_pool + 131072);               
    __nv_bfloat16* s_p = (__nv_bfloat16*)(smem_pool + 163840);                
    __nv_bfloat16* s_ds = (__nv_bfloat16*)(smem_pool + 196608);                
    float* s_d = (float*)(smem_pool + 229376);                                
    float* s_l = (float*)(smem_pool + 229888);                                
    uint64_t* mbar = (uint64_t*)(smem_pool + 230400);                         

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t tmem_s, tmem_dp, tmem_dv, tmem_dk;
    if (threadIdx.x == 0) {
        tmem_alloc_cta(&tmem_s, 128);
        tmem_alloc_cta(&tmem_dp, 128);
        tmem_alloc_cta(&tmem_dv, 128);
        tmem_alloc_cta(&tmem_dk, 128);
    }
    __syncthreads();
    
    const float* l_ptr = L + batch_head_idx * batch_head_stride;
    
    uint32_t phase = 0;
    bool first_iter = true;

    for (int global_i = global_j; global_i < S; global_i += 128) {
        if (threadIdx.x == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])), "r"(8 * 16384) : "memory");
            
            asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                :: "r"((uint32_t)__cvta_generic_to_shared(s_q)), "l"((uint64_t)&tma_Q), "r"(0), "r"(batch_head_idx * S + global_i), "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])) : "memory");
            asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                :: "r"((uint32_t)__cvta_generic_to_shared((char*)s_q + 16384)), "l"((uint64_t)&tma_Q), "r"(64), "r"(batch_head_idx * S + global_i), "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])) : "memory");
            
            asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                :: "r"((uint32_t)__cvta_generic_to_shared(s_o)), "l"((uint64_t)&tma_O), "r"(0), "r"(batch_head_idx * S + global_i), "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])) : "memory");
            asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                :: "r"((uint32_t)__cvta_generic_to_shared((char*)s_o + 16384)), "l"((uint64_t)&tma_O), "r"(64), "r"(batch_head_idx * S + global_i), "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])) : "memory");
            
            asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                :: "r"((uint32_t)__cvta_generic_to_shared(s_do)), "l"((uint64_t)&tma_dO), "r"(0), "r"(batch_head_idx * S + global_i), "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])) : "memory");
            asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                :: "r"((uint32_t)__cvta_generic_to_shared((char*)s_do + 16384)), "l"((uint64_t)&tma_dO), "r"(64), "r"(batch_head_idx * S + global_i), "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])) : "memory");
            
            asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                :: "r"((uint32_t)__cvta_generic_to_shared(s_k)), "l"((uint64_t)&tma_K), "r"(0), "r"(batch_head_idx * S + global_j), "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])) : "memory");
            asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                :: "r"((uint32_t)__cvta_generic_to_shared((char*)s_k + 16384)), "l"((uint64_t)&tma_K), "r"(64), "r"(batch_head_idx * S + global_j), "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])) : "memory");
            
            asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                :: "r"((uint32_t)__cvta_generic_to_shared(s_v)), "l"((uint64_t)&tma_V), "r"(0), "r"(batch_head_idx * S + global_j), "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])) : "memory");
            asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                :: "r"((uint32_t)__cvta_generic_to_shared((char*)s_v + 16384)), "l"((uint64_t)&tma_V), "r"(64), "r"(batch_head_idx * S + global_j), "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])) : "memory");
        }
        
        if (global_i + threadIdx.x < S) {
            s_l[threadIdx.x] = l_ptr[global_i + threadIdx.x];
        } else {
            s_l[threadIdx.x] = 0.0f;
        }
        
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        __syncthreads(); 
        
        float d_val = 0;
        for (int c = 0; c < 128; ++c) {
            int sc = swizzle_128B(threadIdx.x, c);
            d_val += __bfloat162float(s_o[(threadIdx.x) * 128 + sc]) * 
                     __bfloat162float(s_do[(threadIdx.x) * 128 + sc]);
        }
        if (threadIdx.x < 128) {
            s_d[threadIdx.x] = d_val;
        }
        __syncthreads(); 
        
        uint32_t desc_k = make_smem_desc_k_major(s_k);
        uint32_t desc_q = make_smem_desc_k_major(s_q);
        
        gemm_kk(tmem_s, desc_k, desc_q, false);
        commit_mbarrier(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        int g_i = global_i + threadIdx.x;
        for (int i = 0; i < 4; i++) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_s + (threadIdx.x / 32) * 32 + i * 32));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            int col_base = (threadIdx.x / 32) * 32 + i * 32;
            int g_j0 = global_j + col_base;
            int g_j1 = global_j + col_base + 8;
            int g_j2 = global_j + col_base + 16;
            int g_j3 = global_j + col_base + 24;
            
            float p0 = (g_j0 <= g_i && g_j0 < S) ? expf(__uint_as_float(r0) * attn_scale - s_l[threadIdx.x]) : 0.0f;
            float p1 = (g_j1 <= g_i && g_j1 < S) ? expf(__uint_as_float(r1) * attn_scale - s_l[threadIdx.x]) : 0.0f;
            float p2 = (g_j2 <= g_i && g_j2 < S) ? expf(__uint_as_float(r2) * attn_scale - s_l[threadIdx.x]) : 0.0f;
            float p3 = (g_j3 <= g_i && g_j3 < S) ? expf(__uint_as_float(r3) * attn_scale - s_l[threadIdx.x]) : 0.0f;
            
            int sc0 = swizzle_128B(threadIdx.x, col_base);
            int sc1 = swizzle_128B(threadIdx.x, col_base + 8);
            int sc2 = swizzle_128B(threadIdx.x, col_base + 16);
            int sc3 = swizzle_128B(threadIdx.x, col_base + 24);
            
            uint32_t row_data = (threadIdx.x) * 128;
            *(uint32_t*)&s_p[row_data + sc0] = packed_bf16_from_floats(p0, p1);
            *(uint32_t*)&s_p[row_data + sc2] = packed_bf16_from_floats(p2, p3);
        }
        __syncthreads();
        
        uint32_t desc_v = make_smem_desc_k_major(s_v);
        uint32_t desc_do = make_smem_desc_k_major(s_do);
        
        gemm_kk(tmem_dp, desc_v, desc_do, false);
        commit_mbarrier(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        for (int i = 0; i < 4; i++) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dp + (threadIdx.x / 32) * 32 + i * 32));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            int col_base = (threadIdx.x / 32) * 32 + i * 32;
            int g_j0 = global_j + col_base;
            int g_j1 = global_j + col_base + 8;
            int g_j2 = global_j + col_base + 16;
            int g_j3 = global_j + col_base + 24;
            
            float dp0 = (g_j0 < S) ? __uint_as_float(r0) : 0.0f;
            float dp1 = (g_j1 < S) ? __uint_as_float(r1) : 0.0f;
            float dp2 = (g_j2 < S) ? __uint_as_float(r2) : 0.0f;
            float dp3 = (g_j3 < S) ? __uint_as_float(r3) : 0.0f;
            
            float ds0 = __bfloat162float(s_p[(threadIdx.x) * 128 + swizzle_128B(threadIdx.x, g_j0)]) * (dp0 - s_d[threadIdx.x]);
            float ds1 = __bfloat162float(s_p[(threadIdx.x) * 128 + swizzle_128B(threadIdx.x, g_j1)]) * (dp1 - s_d[threadIdx.x]);
            float ds2 = __bfloat162float(s_p[(threadIdx.x) * 128 + swizzle_128B(threadIdx.x, g_j2)]) * (dp2 - s_d[threadIdx.x]);
            float ds3 = __bfloat162float(s_p[(threadIdx.x) * 128 + swizzle_128B(threadIdx.x, g_j3)]) * (dp3 - s_d[threadIdx.x]);
            
            int sc0 = swizzle_128B(threadIdx.x, col_base);
            int sc1 = swizzle_128B(threadIdx.x, col_base + 8);
            int sc2 = swizzle_128B(threadIdx.x, col_base + 16);
            int sc3 = swizzle_128B(threadIdx.x, col_base + 24);
            
            uint32_t row_data = (threadIdx.x) * 128;
            *(uint32_t*)&s_ds[row_data + sc0] = packed_bf16_from_floats(ds0, ds1);
            *(uint32_t*)&s_ds[row_data + sc2] = packed_bf16_from_floats(ds2, ds3);
        }
        __syncthreads();
        
        transpose_128x128(s_p);
        __syncthreads();
        
        uint32_t desc_p = make_smem_desc_k_major(s_p);
        uint32_t desc_do_nm = make_smem_desc_n_major(s_do);
        
        gemm_kn(tmem_dv, desc_p, desc_do_nm, !first_iter);
        commit_mbarrier(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        transpose_128x128(s_ds);
        __syncthreads();
        
        uint32_t desc_ds = make_smem_desc_k_major(s_ds);
        uint32_t desc_q_nm = make_smem_desc_n_major(s_q);
        
        gemm_kn(tmem_dk, desc_ds, desc_q_nm, !first_iter);
        commit_mbarrier(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        transpose_128x128(s_ds);
        __syncthreads();
        
        uint32_t desc_ds2 = make_smem_desc_k_major(s_ds);
        uint32_t desc_k_nm = make_smem_desc_n_major(s_k);
        
        gemm_kn(tmem_s, desc_ds2, desc_k_nm, false);
        commit_mbarrier(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_s + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int g_i = global_i + threadIdx.x;
            if (g_i < S && col < 128) {
                atomicAdd_bf16(&dQ[batch_head_idx * S * 128 + g_i * 128 + col], __float2bfloat16(__uint_as_float(r0)));
                atomicAdd_bf16(&dQ[batch_head_idx * S * 128 + g_i * 128 + col + 1], __float2bfloat16(__uint_as_float(r1)));
                atomicAdd_bf16(&dQ[batch_head_idx * S * 128 + g_i * 128 + col + 2], __float2bfloat16(__uint_as_float(r2)));
                atomicAdd_bf16(&dQ[batch_head_idx * S * 128 + g_i * 128 + col + 3], __float2bfloat16(__uint_as_float(r3)));
            }
        }
        
        first_iter = false;
        __syncthreads();
    } 
    
    for (int col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dk + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        int g_j = global_j + threadIdx.x;
        if (g_j < S && col < 128) {
            dK[batch_head_idx * S * 128 + g_j * 128 + col] = __float2bfloat16(__uint_as_float(r0));
            dK[batch_head_idx * S * 128 + g_j * 128 + col + 1] = __float2bfloat16(__uint_as_float(r1));
            dK[batch_head_idx * S * 128 + g_j * 128 + col + 2] = __float2bfloat16(__uint_as_float(r2));
            dK[batch_head_idx * S * 128 + g_j * 128 + col + 3] = __float2bfloat16(__uint_as_float(r3));
        }
    }
    
    for (int col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dv + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        int g_j = global_j + threadIdx.x;
        if (g_j < S && col < 128) {
            dV[batch_head_idx * S * 128 + g_j * 128 + col] = __float2bfloat16(__uint_as_float(r0));
            dV[batch_head_idx * S * 128 + g_j * 128 + col + 1] = __float2bfloat16(__uint_as_float(r1));
            dV[batch_head_idx * S * 128 + g_j * 128 + col + 2] = __float2bfloat16(__uint_as_float(r2));
            dV[batch_head_idx * S * 128 + g_j * 128 + col + 3] = __float2bfloat16(__uint_as_float(r3));
        }
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_cta(tmem_s, 128);
        tmem_dealloc_cta(tmem_dp, 128);
        tmem_dealloc_cta(tmem_dv, 128);
        tmem_dealloc_cta(tmem_dk, 128);
    }
}

namespace tvm_ffi_mha_bwd {

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {64, 128}; 
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, // tensorRank
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
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, B * H * S * d * sizeof(__nv_bfloat16), stream));
    
    int num_j = (S + 127) / 128;
    dim3 grid(num_j, B * H);
    dim3 block(128);
    
    float attn_scale = 1.0f / sqrtf((float)d);
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), d, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), d, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), d, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_O, O.data_ptr(), d, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_dO, dO.data_ptr(), d, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int smem_size = 266240;
    CUDA_CHECK(cudaFuncSetAttribute(bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, bwd_kernel,
        tma_Q, tma_K, tma_V, tma_O, tma_dO,
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S, attn_scale, L.stride(1)
    ));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd