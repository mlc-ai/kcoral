#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    }                                                            \
} while(0)

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t tmem_addr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(tmem_addr));
}

__device__ __forceinline__ uint32_t tmem_addr(uint32_t tmem_base, uint32_t r_base, uint32_t col) {
    uint32_t base_col = tmem_base & 0xFFFF;
    return (r_base << 16) | (base_col + col);
}

__device__ __forceinline__ void umma_commit_1sm_cta(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"(a));
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

__device__ __forceinline__ uint32_t make_instr_desc_fn_transposed_B(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (1u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_transposed_A(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (1u << 15);   
    d |= (0u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_both_transposed(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (1u << 15);   
    d |= (1u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
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

__device__ __forceinline__ void umma_64x64_kt(
    uint32_t tmem_c, 
    void* a_ptr, bool a_transposed, 
    void* b_ptr, bool b_transposed, 
    uint32_t accum) 
{
    for(int k = 0; k < 64; k += 16) {
        uint64_t desc_a, desc_b;
        if (a_transposed) {
            desc_a = make_smem_desc_sm100_fn((__nv_bfloat16*)a_ptr + k * 64, 8192, 1024);
        } else {
            desc_a = make_smem_desc_sm100_fn((__nv_bfloat16*)a_ptr + k, 1, 1024);
        }
        
        if (b_transposed) {
            desc_b = make_smem_desc_sm100_fn((__nv_bfloat16*)b_ptr + k * 64, 8192, 1024);
        } else {
            desc_b = make_smem_desc_sm100_fn((__nv_bfloat16*)b_ptr + k, 1, 1024);
        }

        uint32_t idesc;
        if (a_transposed && b_transposed) {
            idesc = make_instr_desc_fn_both_transposed(64, 64);
        } else if (a_transposed) {
            idesc = make_instr_desc_fn_transposed_A(64, 64);
        } else if (b_transposed) {
            idesc = make_instr_desc_fn_transposed_B(64, 64);
        } else {
            idesc = make_instr_desc_fn(64, 64);
        }

        asm volatile(
            "{\n.reg .pred p;\n"
            "setp.ne.b32 p, %4, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
    }
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t pack_bf16_from_float_fn(float fa, float fb) {
    __nv_bfloat16 a = __float2bfloat16(fa);
    __nv_bfloat16 b = __float2bfloat16(fb);
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ uint32_t swizzle_128B(uint32_t row, uint32_t col_bytes) {
    return (((row & 7) ^ (col_bytes >> 4)) << 4) | (col_bytes & 15);
}

__device__ __forceinline__ __nv_bfloat16 read_smem_64x64_swizzled_128B(const __nv_bfloat16* smem, uint32_t row, uint32_t col) {
    uint32_t col_bytes = col * 2;
    uint32_t offset = (row * 128 + swizzle_128B(row, col_bytes)) * 2;
    return *(const __nv_bfloat16*)((const char*)smem + offset);
}

__device__ __forceinline__ void write_smem_64x64_swizzled_128B(
    __nv_bfloat16* smem, uint32_t row, uint32_t col, __nv_bfloat16 val) {
    uint32_t col_bytes = col * 2;
    uint32_t offset = (row * 128 + swizzle_128B(row, col_bytes)) * 2;
    *(uint16_t*)((char*)smem + offset) = *(uint16_t*)&val;
}

__device__ __forceinline__ void compute_D_and_L(const float* L_ptr, uint32_t bh, uint32_t S, uint32_t j, int tid, 
                                float* m_LT, __nv_bfloat16* m_DT,
                                const __nv_bfloat16* m_O0, const __nv_bfloat16* m_dO0,
                                const __nv_bfloat16* m_O1, const __nv_bfloat16* m_dO1) {
    int q_idx = j * 64 + tid;
    float lse = (q_idx < S) ? L_ptr[bh * S + q_idx] : 0.0f;
    m_LT[tid] = lse;
    
    float sum = 0;
    if (q_idx < S) {
        for(int i=0; i<64; ++i) {
            sum += __bfloat162float(read_smem_64x64_swizzled_128B(m_O0, tid, i)) * __bfloat162float(read_smem_64x64_swizzled_128B(m_dO0, tid, i));
            sum += __bfloat162float(read_smem_64x64_swizzled_128B(m_O1, tid, i)) * __bfloat162float(read_smem_64x64_swizzled_128B(m_dO1, tid, i));
        }
    }
    m_DT[tid] = __float2bfloat16(sum);
}

__device__ __forceinline__ void compute_P_and_dS_causal(
    uint32_t tmem_S_T, uint32_t tmem_dP_T, 
    float* m_LT, uint32_t bh, uint32_t S, 
    uint32_t global_q_start, uint32_t global_k_start, 
    __nv_bfloat16* m_PT, __nv_bfloat16* m_dST, __nv_bfloat16* m_DT) 
{
    int warp_id = threadIdx.x / 32;
    int r_base = warp_id * 32;
    int r_idx = r_base + threadIdx.x % 32;
    
    for (int c = 0; c < 64; c += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_addr(tmem_S_T, r_base, c), &r0, &r1, &r2, &r3);
        
        uint32_t r0d, r1d, r2d, r3d;
        tmem_load_4x_fn(tmem_addr(tmem_dP_T, r_base, c), &r0d, &r1d, &r2d, &r3d);
        
        tmem_load_fence_fn();
        
        int q_idx = global_q_start + r_idx;
        int k0 = global_k_start + c;
        int k1 = global_k_start + c + 1;
        int k2 = global_k_start + c + 2;
        int k3 = global_k_start + c + 3;
        
        float lse = m_LT[r_idx];
        
        float p0 = (k0 <= q_idx && q_idx < S && k0 < S) ? fast_exp2f_fn((__uint_as_float(r0) - lse) * 1.44269504f) : 0.0f;
        float p1 = (k1 <= q_idx && q_idx < S && k1 < S) ? fast_exp2f_fn((__uint_as_float(r1) - lse) * 1.44269504f) : 0.0f;
        float p2 = (k2 <= q_idx && q_idx < S && k2 < S) ? fast_exp2f_fn((__uint_as_float(r2) - lse) * 1.44269504f) : 0.0f;
        float p3 = (k3 <= q_idx && q_idx < S && k3 < S) ? fast_exp2f_fn((__uint_as_float(r3) - lse) * 1.44269504f) : 0.0f;
        
        write_smem_64x64_swizzled_128B(m_PT, r_idx, c, __float2bfloat16(p0));
        write_smem_64x64_swizzled_128B(m_PT, r_idx, c+1, __float2bfloat16(p1));
        write_smem_64x64_swizzled_128B(m_PT, r_idx, c+2, __float2bfloat16(p2));
        write_smem_64x64_swizzled_128B(m_PT, r_idx, c+3, __float2bfloat16(p3));
        
        float dp0 = __uint_as_float(r0d);
        float dp1 = __uint_as_float(r1d);
        float dp2 = __uint_as_float(r2d);
        float dp3 = __uint_as_float(r3d);
        
        float ds0 = (k0 <= q_idx && q_idx < S && k0 < S) ? p0 * (dp0 - __bfloat162float(m_DT[r_idx])) : 0.0f;
        float ds1 = (k1 <= q_idx && q_idx < S && k1 < S) ? p1 * (dp1 - __bfloat162float(m_DT[r_idx])) : 0.0f;
        float ds2 = (k2 <= q_idx && q_idx < S && k2 < S) ? p2 * (dp2 - __bfloat162float(m_DT[r_idx])) : 0.0f;
        float ds3 = (k3 <= q_idx && q_idx < S && k3 < S) ? p3 * (dp3 - __bfloat162float(m_DT[r_idx])) : 0.0f;
        
        write_smem_64x64_swizzled_128B(m_dST, r_idx, c, __float2bfloat16(ds0));
        write_smem_64x64_swizzled_128B(m_dST, r_idx, c+1, __float2bfloat16(ds1));
        write_smem_64x64_swizzled_128B(m_dST, r_idx, c+2, __float2bfloat16(ds2));
        write_smem_64x64_swizzled_128B(m_dST, r_idx, c+3, __float2bfloat16(ds3));
    }
}

__device__ __forceinline__ void atomic_add_dK(uint32_t tmem_dK_T, __nv_bfloat16* dK_ptr, uint32_t bh, uint32_t S, uint32_t k, uint32_t feat_offset) {
    int warp_id = threadIdx.x / 32;
    int r_base = warp_id * 32;
    int r_idx = r_base + threadIdx.x % 32;
    
    for (int c = 0; c < 64; c += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_addr(tmem_dK_T, r_base, c), &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        
        int k_idx = k * 64 + r_idx;
        if (k_idx < S) {
            uint32_t ptr_idx = bh * S * 128 + k_idx * 128 + feat_offset + c;
            uint32_t v0 = pack_bf16_from_float_fn(__uint_as_float(r0), __uint_as_float(r1));
            uint32_t v1 = pack_bf16_from_float_fn(__uint_as_float(r2), __uint_as_float(r3));
            
            atomicAdd((unsigned int*)(&dK_ptr[ptr_idx]), v0);
            atomicAdd((unsigned int*)(&dK_ptr[ptr_idx + 2]), v1);
        }
    }
}

__device__ __forceinline__ void atomic_add_dV(uint32_t tmem_dV_T, __nv_bfloat16* dV_ptr, uint32_t bh, uint32_t S, uint32_t k, uint32_t feat_offset) {
    int warp_id = threadIdx.x / 32;
    int r_base = warp_id * 32;
    int r_idx = r_base + threadIdx.x % 32;
    
    for (int c = 0; c < 64; c += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_addr(tmem_dV_T, r_base, c), &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        
        int k_idx = k * 64 + r_idx;
        if (k_idx < S) {
            uint32_t ptr_idx = bh * S * 128 + k_idx * 128 + feat_offset + c;
            uint32_t v0 = pack_bf16_from_float_fn(__uint_as_float(r0), __uint_as_float(r1));
            uint32_t v1 = pack_bf16_from_float_fn(__uint_as_float(r2), __uint_as_float(r3));
            
            atomicAdd((unsigned int*)(&dV_ptr[ptr_idx]), v0);
            atomicAdd((unsigned int*)(&dV_ptr[ptr_idx + 2]), v1);
        }
    }
}

__device__ __forceinline__ void store_tmem_64(__nv_bfloat16* gmem, uint32_t bh, uint32_t S, uint32_t tile_idx, uint32_t tmem_base, uint32_t feat_offset) {
    int warp_id = threadIdx.x / 32;
    int r_base = warp_id * 32;
    int r_idx = r_base + threadIdx.x % 32;
    
    for (int c = 0; c < 64; c += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_addr(tmem_base, r_base, c), &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        
        int row_idx = tile_idx * 64 + r_idx;
        if (row_idx < S) {
            uint32_t ptr_idx = bh * S * 128 + row_idx * 128 + feat_offset + c;
            uint32_t v0 = pack_bf16_from_float_fn(__uint_as_float(r0), __uint_as_float(r1));
            uint32_t v1 = pack_bf16_from_float_fn(__uint_as_float(r2), __uint_as_float(r3));
            
            *(uint32_t*)(&gmem[ptr_idx]) = v0;
            *(uint32_t*)(&gmem[ptr_idx + 2]) = v1;
        }
    }
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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
    const __grid_constant__ CUtensorMap tma_Q, 
    const __grid_constant__ CUtensorMap tma_K, 
    const __grid_constant__ CUtensorMap tma_V, 
    const __grid_constant__ CUtensorMap tma_dO, 
    const __grid_constant__ CUtensorMap tma_O,
    const float* L_ptr,
    __nv_bfloat16* dQ_ptr, __nv_bfloat16* dK_ptr, __nv_bfloat16* dV_ptr,
    uint32_t S, uint32_t d) 
{
    uint32_t bh = blockIdx.y;
    uint32_t j = blockIdx.x;
    
    extern __shared__ char smem_raw[];
    char* smem = (char*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
    
    __nv_bfloat16* m_Q0 = (__nv_bfloat16*)smem;
    __nv_bfloat16* m_Q1 = (__nv_bfloat16*)(smem + 8192);
    __nv_bfloat16* m_K0 = (__nv_bfloat16*)(smem + 16384);
    __nv_bfloat16* m_K1 = (__nv_bfloat16*)(smem + 24576);
    __nv_bfloat16* m_V0 = (__nv_bfloat16*)(smem + 32768);
    __nv_bfloat16* m_V1 = (__nv_bfloat16*)(smem + 40960);
    __nv_bfloat16* m_dO0 = (__nv_bfloat16*)(smem + 49152);
    __nv_bfloat16* m_dO1 = (__nv_bfloat16*)(smem + 57344);
    __nv_bfloat16* m_O0 = (__nv_bfloat16*)(smem + 65536);
    __nv_bfloat16* m_O1 = (__nv_bfloat16*)(smem + 73728);
    __nv_bfloat16* m_dST = (__nv_bfloat16*)(smem + 81920);
    __nv_bfloat16* m_PT = (__nv_bfloat16*)(smem + 90112);
    __nv_bfloat16* m_DT = (__nv_bfloat16*)(smem + 98304);
    float* m_LT = (float*)(smem + 98432);
    uint64_t* mbar = (uint64_t*)(smem + 98784);
    
    uint32_t tmem_S_T, tmem_dP_T, tmem_dQ_T0, tmem_dQ_T1, tmem_dK_T0, tmem_dK_T1, tmem_dV_T0, tmem_dV_T1;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S_T, 128);
        tmem_alloc_fn(&tmem_dP_T, 128);
        tmem_alloc_fn(&tmem_dQ_T0, 128);
        tmem_alloc_fn(&tmem_dQ_T1, 128);
        tmem_alloc_fn(&tmem_dK_T0, 128);
        tmem_alloc_fn(&tmem_dK_T1, 128);
        tmem_alloc_fn(&tmem_dV_T0, 128);
        tmem_alloc_fn(&tmem_dV_T1, 128);
        init_smem_barrier_fn(mbar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        uint32_t seq_off = bh * S + j * 64;
        mbarrier_arrive_and_expect_tx_fn(mbar, 6 * 8192);
        tma_load_2d_fn(&tma_Q, mbar, m_Q0, 0, seq_off);
        tma_load_2d_fn(&tma_Q, mbar, m_Q1, 64, seq_off);
        tma_load_2d_fn(&tma_dO, mbar, m_dO0, 0, seq_off);
        tma_load_2d_fn(&tma_dO, mbar, m_dO1, 64, seq_off);
        tma_load_2d_fn(&tma_O, mbar, m_O0, 0, seq_off);
        tma_load_2d_fn(&tma_O, mbar, m_O1, 64, seq_off);
    }
    mbarrier_wait_fn(mbar, 0);
    fence_proxy_async_fn();
    
    compute_D_and_L(L_ptr, bh, S, j, threadIdx.x, m_LT, m_DT, m_O0, m_dO0, m_O1, m_dO1);
    
    uint32_t phase = 0;
    uint32_t accum_dQ0 = 0, accum_dQ1 = 0;
    uint32_t accum_dK0 = 0, accum_dK1 = 0;
    uint32_t accum_dV0 = 0, accum_dV1 = 0;
    
    for (int k = 0; k <= j; ++k) {
        if (threadIdx.x == 0) {
            uint32_t seq_off = bh * S + k * 64;
            mbarrier_arrive_and_expect_tx_fn(mbar, 4 * 8192);
            tma_load_2d_fn(&tma_K, mbar, m_K0, 0, seq_off);
            tma_load_2d_fn(&tma_K, mbar, m_K1, 64, seq_off);
            tma_load_2d_fn(&tma_V, mbar, m_V0, 0, seq_off);
            tma_load_2d_fn(&tma_V, mbar, m_V1, 64, seq_off);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        fence_proxy_async_fn();
        
        if (threadIdx.x == 0) {
            umma_64x64_kt(tmem_S_T, m_Q0, false, m_K0, true, 0);
            umma_64x64_kt(tmem_S_T, m_Q1, false, m_K1, true, 1);
            umma_commit_1sm_cta(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        if (threadIdx.x == 0) {
            umma_64x64_kt(tmem_dP_T, m_V0, false, m_dO0, true, 0);
            umma_64x64_kt(tmem_dP_T, m_V1, false, m_dO1, true, 1);
            umma_commit_1sm_cta(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        compute_P_and_dS_causal(tmem_S_T, tmem_dP_T, m_LT, bh, S, j * 64, k * 64, m_PT, m_dST, m_DT);
        __syncthreads();
        
        if (threadIdx.x == 0) {
            umma_64x64_kt(tmem_dQ_T0, m_dST, false, m_K0, false, accum_dQ0);
            umma_64x64_kt(tmem_dQ_T1, m_dST, false, m_K1, false, accum_dQ1);
            umma_commit_1sm_cta(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        accum_dQ0 = 1; accum_dQ1 = 1;
        
        if (threadIdx.x == 0) {
            umma_64x64_kt(tmem_dK_T0, m_dST, true, m_Q0, false, accum_dK0);
            umma_64x64_kt(tmem_dK_T1, m_dST, true, m_Q1, false, accum_dK1);
            umma_commit_1sm_cta(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        accum_dK0 = 1; accum_dK1 = 1;
        
        if (threadIdx.x == 0) {
            umma_64x64_kt(tmem_dV_T0, m_PT, false, m_dO0, false, accum_dV0);
            umma_64x64_kt(tmem_dV_T1, m_PT, false, m_dO1, false, accum_dV1);
            umma_commit_1sm_cta(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        accum_dV0 = 1; accum_dV1 = 1;
    }
    
    store_tmem_64(dQ_ptr, bh, S, j, tmem_dQ_T0, 0);
    store_tmem_64(dQ_ptr, bh, S, j, tmem_dQ_T1, 64);
    atomic_add_dK(tmem_dK_T0, dK_ptr, bh, S, k, 0);
    atomic_add_dK(tmem_dK_T1, dK_ptr, bh, S, k, 64);
    atomic_add_dV(tmem_dV_T0, dV_ptr, bh, S, k, 0);
    atomic_add_dV(tmem_dV_T1, dV_ptr, bh, S, k, 64);
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_S_T, 128);
        tmem_dealloc_fn(tmem_dP_T, 128);
        tmem_dealloc_fn(tmem_dQ_T0, 128);
        tmem_dealloc_fn(tmem_dQ_T1, 128);
        tmem_dealloc_fn(tmem_dK_T0, 128);
        tmem_dealloc_fn(tmem_dK_T1, 128);
        tmem_dealloc_fn(tmem_dV_T0, 128);
        tmem_dealloc_fn(tmem_dV_T1, 128);
    }
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int64_t B = Q.size(0);
  int64_t H = Q.size(1);
  int64_t S = Q.size(2);
  int64_t d = Q.size(3);

  CUtensorMap tma_Q, tma_K, tma_V, tma_dO, tma_O;
  CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), d, B * H * S, 64, 64, 
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), d, B * H * S, 64, 64, 
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), d, B * H * S, 64, 64, 
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  CU_CHECK(create_tma_2d_descriptor_2B(&tma_dO, dO.data_ptr(), d, B * H * S, 64, 64, 
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  CU_CHECK(create_tma_2d_descriptor_2B(&tma_O, O.data_ptr(), d, B * H * S, 64, 64, 
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  dim3 grid((S + 63) / 64, B * H);
  dim3 block(128);
  
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  
  int smem_size = 130 * 1024;
  CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  
  mha_bwd_kernel<<<grid, block, smem_size, stream>>>(
      tma_Q, tma_K, tma_V, tma_dO, tma_O,
      static_cast<const float*>(L.data_ptr()),
      static_cast<__nv_bfloat16*>(dQ.data_ptr()), 
      static_cast<__nv_bfloat16*>(dK.data_ptr()), 
      static_cast<__nv_bfloat16*>(dV.data_ptr()),
      S, d);
      
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_example_cuda