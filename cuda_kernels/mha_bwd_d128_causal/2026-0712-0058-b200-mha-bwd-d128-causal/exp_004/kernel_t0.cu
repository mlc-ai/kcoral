#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
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

// Helper functions for TMEM and SM100 features
__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t pack_bf16_fn(__nv_bfloat16 a, __nv_bfloat16 b) {
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ uint32_t pack_bf16_from_float_fn(float fa, float fb) {
    __nv_bfloat16 a = __float2bfloat16(fa);
    __nv_bfloat16 b = __float2bfloat16(fb);
    return pack_bf16_fn(a, b);
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ uint32_t swizzle_128B(uint32_t row, uint32_t col_bytes) {
    return (((row & 7) ^ (col_bytes >> 4)) << 4) | (col_bytes & 15);
}

__device__ __forceinline__ void umma_128x128_kt(
    uint32_t tmem_c, 
    void* a_ptr, bool a_transposed, 
    void* b_ptr, bool b_transposed, 
    uint32_t accum) 
{
    for(int k=0; k<128; k+=16) {
        uint64_t desc_a, desc_b;
        if (a_transposed) {
            desc_a = make_smem_desc_sm100_fn((__nv_bfloat16*)a_ptr + k, 1, 1024);
        } else {
            desc_a = make_smem_desc_sm100_fn((__nv_bfloat16*)a_ptr + k * 128, 1, 1024);
        }
        
        if (b_transposed) {
            desc_b = make_smem_desc_sm100_fn((__nv_bfloat16*)b_ptr + k, 1, 1024);
        } else {
            desc_b = make_smem_desc_sm100_fn((__nv_bfloat16*)b_ptr + k * 128, 1, 1024);
        }

        uint32_t idesc;
        if (a_transposed && b_transposed) {
            idesc = make_instr_desc_fn(128, 128);
        } else if (a_transposed && !b_transposed) {
            idesc = make_instr_desc_fn_transposed_B(128, 128);
        } else if (!a_transposed && b_transposed) {
            idesc = make_instr_desc_fn_transposed_B(128, 128);
        } else {
            idesc = make_instr_desc_fn(128, 128);
        }

        asm volatile(
            "{\n.reg .pred p;\n"
            "setp.ne.b32 p, %4, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
        
        accum = 1;
    }
}

__device__ __forceinline__ void load_tile_128(const __nv_bfloat16* gmem, uint32_t bh, uint32_t S, uint32_t tile_idx, __nv_bfloat16* smem) {
    int tid = threadIdx.x;
    for(int i=0; i<16; ++i) {
        int4 val = {0,0,0,0};
        int g_idx = tile_idx * 128 + tid * 128 + i * 8;
        if (tile_idx * 128 + tid < S) {
            val = *reinterpret_cast<const int4*>(&gmem[bh * S * 128 + g_idx]);
        }
        *reinterpret_cast<int4*>(&smem[tid * 128 + i * 8]) = val;
    }
}

__device__ __forceinline__ void compute_D(const __nv_bfloat16* O_ptr, const __nv_bfloat16* dO_ptr, 
                                uint32_t bh, uint32_t S, uint32_t j, __nv_bfloat16* m_DT) {
    int tid = threadIdx.x;
    int q_idx = j * 128 + tid;
    float sum = 0;
    if (q_idx < S) {
        const __nv_bfloat16* o_row = O_ptr + bh * S * 128 + q_idx * 128;
        const __nv_bfloat16* do_row = dO_ptr + bh * S * 128 + q_idx * 128;
        for(int i=0; i<128; ++i) {
            sum += __bfloat162float(o_row[i]) * __bfloat162float(do_row[i]);
        }
    }
    m_DT[tid] = __float2bfloat16(sum);
}

__device__ __forceinline__ void write_smem_128_128_swizzled_128B(
    __nv_bfloat16* smem, uint32_t row, uint32_t col, __nv_bfloat16 val) {
    uint32_t col_bytes = col * 2;
    uint32_t offset = (row * 128 + swizzle_128B(row, col_bytes)) * 2;
    *(uint16_t*)((char*)smem + offset) = *(uint16_t*)&val;
}

__device__ __forceinline__ void compute_P_and_dS(
    uint32_t tmem_S_T, uint32_t tmem_dP_T, 
    const float* L_ptr, uint32_t bh, uint32_t S, 
    uint32_t j, uint32_t bid_x, 
    __nv_bfloat16* m_PT, __nv_bfloat16* m_dST, __nv_bfloat16* m_DT) 
{
    int warp_id = threadIdx.x / 32;
    int r_base = warp_id * 32;
    
    for (int c = 0; c < 128; c += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn((r_base << 16) | c, &r0, &r1, &r2, &r3);
        
        uint32_t r0d, r1d, r2d, r3d;
        tmem_load_4x_fn((r_base << 16) | c, &r0d, &r1d, &r2d, &r3d);
        
        tmem_load_fence_fn();
        
        float s0 = __uint_as_float(r0);
        float s1 = __uint_as_float(r1);
        float s2 = __uint_as_float(r2);
        float s3 = __uint_as_float(r3);
        
        float dp0 = __uint_as_float(r0d);
        float dp1 = __uint_as_float(r1d);
        float dp2 = __uint_as_float(r2d);
        float dp3 = __uint_as_float(r3d);
        
        int q_idx = j * 128 + (r_base + threadIdx.x % 32);
        int k0 = bid_x * 128 + c;
        int k1 = bid_x * 128 + c + 1;
        int k2 = bid_x * 128 + c + 2;
        int k3 = bid_x * 128 + c + 3;
        
        float lse = (q_idx < S) ? L_ptr[bh * S + q_idx] : 0.0f;
        
        float p0 = (k0 <= q_idx && q_idx < S) ? fast_exp2f_fn((s0 - lse) * 1.44269504f) : 0.0f;
        float p1 = (k1 <= q_idx && q_idx < S) ? fast_exp2f_fn((s1 - lse) * 1.44269504f) : 0.0f;
        float p2 = (k2 <= q_idx && q_idx < S) ? fast_exp2f_fn((s2 - lse) * 1.44269504f) : 0.0f;
        float p3 = (k3 <= q_idx && q_idx < S) ? fast_exp2f_fn((s3 - lse) * 1.44269504f) : 0.0f;
        
        write_smem_128_128_swizzled_128B(m_PT, r_base + threadIdx.x % 32, c, __float2bfloat16(p0));
        write_smem_128_128_swizzled_128B(m_PT, r_base + threadIdx.x % 32, c+1, __float2bfloat16(p1));
        write_smem_128_128_swizzled_128B(m_PT, r_base + threadIdx.x % 32, c+2, __float2bfloat16(p2));
        write_smem_128_128_swizzled_128B(m_PT, r_base + threadIdx.x % 32, c+3, __float2bfloat16(p3));
        
        float ds0 = p0 * (dp0 - __bfloat162float(m_DT[threadIdx.x % 32]));
        float ds1 = p1 * (dp1 - __bfloat162float(m_DT[threadIdx.x % 32]));
        float ds2 = p2 * (dp2 - __bfloat162float(m_DT[threadIdx.x % 32]));
        float ds3 = p3 * (dp3 - __bfloat162float(m_DT[threadIdx.x % 32]));
        
        if (!(k0 <= q_idx && q_idx < S)) ds0 = 0.0f;
        if (!(k1 <= q_idx && q_idx < S)) ds1 = 0.0f;
        if (!(k2 <= q_idx && q_idx < S)) ds2 = 0.0f;
        if (!(k3 <= q_idx && q_idx < S)) ds3 = 0.0f;
        
        write_smem_128_128_swizzled_128B(m_dST, r_base + threadIdx.x % 32, c, __float2bfloat16(ds0));
        write_smem_128_128_swizzled_128B(m_dST, r_base + threadIdx.x % 32, c+1, __float2bfloat16(ds1));
        write_smem_128_128_swizzled_128B(m_dST, r_base + threadIdx.x % 32, c+2, __float2bfloat16(ds2));
        write_smem_128_128_swizzled_128B(m_dST, r_base + threadIdx.x % 32, c+3, __float2bfloat16(ds3));
    }
}

// Causal Masking Fix: Pass global_q_start and global_k_start explicitly to resolve out-of-bounds logic errors.
__device__ __forceinline__ void compute_P_and_dS_causal(
    uint32_t tmem_S_T, uint32_t tmem_dP_T, 
    const float* L_ptr, uint32_t bh, uint32_t S, 
    uint32_t global_q_start, uint32_t global_k_start, 
    __nv_bfloat16* m_PT, __nv_bfloat16* m_dST, __nv_bfloat16* m_DT) 
{
    int warp_id = threadIdx.x / 32;
    int r_base = warp_id * 32;
    
    for (int c = 0; c < 128; c += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn((r_base << 16) | c, &r0, &r1, &r2, &r3);
        
        uint32_t r0d, r1d, r2d, r3d;
        tmem_load_4x_fn((r_base << 16) | c, &r0d, &r1d, &r2d, &r3d);
        
        tmem_load_fence_fn();
        
        float s0 = __uint_as_float(r0);
        float s1 = __uint_as_float(r1);
        float s2 = __uint_as_float(r2);
        float s3 = __uint_as_float(r3);
        
        float dp0 = __uint_as_float(r0d);
        float dp1 = __uint_as_float(r1d);
        float dp2 = __uint_as_float(r2d);
        float dp3 = __uint_as_float(r3d);
        
        int q_idx = global_q_start + (r_base + threadIdx.x % 32);
        int k0 = global_k_start + c;
        int k1 = global_k_start + c + 1;
        int k2 = global_k_start + c + 2;
        int k3 = global_k_start + c + 3;
        
        float lse = (q_idx < S) ? L_ptr[bh * S + q_idx] : 0.0f;
        
        float p0 = (k0 <= q_idx && q_idx < S) ? fast_exp2f_fn((s0 - lse) * 1.44269504f) : 0.0f;
        float p1 = (k1 <= q_idx && q_idx < S) ? fast_exp2f_fn((s1 - lse) * 1.44269504f) : 0.0f;
        float p2 = (k2 <= q_idx && q_idx < S) ? fast_exp2f_fn((s2 - lse) * 1.44269504f) : 0.0f;
        float p3 = (k3 <= q_idx && q_idx < S) ? fast_exp2f_fn((s3 - lse) * 1.44269504f) : 0.0f;
        
        write_smem_128_128_swizzled_128B(m_PT, r_base + threadIdx.x % 32, c, __float2bfloat16(p0));
        write_smem_128_128_swizzled_128B(m_PT, r_base + threadIdx.x % 32, c+1, __float2bfloat16(p1));
        write_smem_128_128_swizzled_128B(m_PT, r_base + threadIdx.x % 32, c+2, __float2bfloat16(p2));
        write_smem_128_128_swizzled_128B(m_PT, r_base + threadIdx.x % 32, c+3, __float2bfloat16(p3));
        
        float ds0 = p0 * (dp0 - __bfloat162float(m_DT[r_base + threadIdx.x % 32]));
        float ds1 = p1 * (dp1 - __bfloat162float(m_DT[r_base + threadIdx.x % 32]));
        float ds2 = p2 * (dp2 - __bfloat162float(m_DT[r_base + threadIdx.x % 32]));
        float ds3 = p3 * (dp3 - __bfloat162float(m_DT[r_base + threadIdx.x % 32]));
        
        if (!(k0 <= q_idx && q_idx < S)) ds0 = 0.0f;
        if (!(k1 <= q_idx && q_idx < S)) ds1 = 0.0f;
        if (!(k2 <= q_idx && q_idx < S)) ds2 = 0.0f;
        if (!(k3 <= q_idx && q_idx < S)) ds3 = 0.0f;
        
        write_smem_128_128_swizzled_128B(m_dST, r_base + threadIdx.x % 32, c, __float2bfloat16(ds0));
        write_smem_128_128_swizzled_128B(m_dST, r_base + threadIdx.x % 32, c+1, __float2bfloat16(ds1));
        write_smem_128_128_swizzled_128B(m_dST, r_base + threadIdx.x % 32, c+2, __float2bfloat16(ds2));
        write_smem_128_128_swizzled_128B(m_dST, r_base + threadIdx.x % 32, c+3, __float2bfloat16(ds3));
    }
}

__device__ __forceinline__ void atomic_add_dQ(uint32_t tmem_dQ_T, __nv_bfloat16* dQ_ptr, uint32_t bh, uint32_t S, uint32_t j) {
    int warp_id = threadIdx.x / 32;
    int r_base = warp_id * 32;
    
    for (int c = 0; c < 128; c += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn((r_base << 16) | c, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        
        int q_idx = j * 128 + (r_base + threadIdx.x % 32);
        if (q_idx < S) {
            uint32_t ptr_idx = bh * S * 128 + q_idx * 128 + c;
            uint32_t v0 = pack_bf16_from_float_fn(__uint_as_float(r0), __uint_as_float(r1));
            uint32_t v1 = pack_bf16_from_float_fn(__uint_as_float(r2), __uint_as_float(r3));
            
            atomicAdd((unsigned int*)(&dQ_ptr[ptr_idx]), v0);
            atomicAdd((unsigned int*)(&dQ_ptr[ptr_idx + 2]), v1);
        }
    }
}

__device__ __forceinline__ void store_tmem_128(uint32_t tmem, __nv_bfloat16* gmem, uint32_t bh, uint32_t S, uint32_t tile_idx) {
    int warp_id = threadIdx.x / 32;
    int r_base = warp_id * 32;
    
    for (int c = 0; c < 128; c += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn((r_base << 16) | c, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        
        int k_idx = tile_idx * 128 + (r_base + threadIdx.x % 32);
        if (k_idx < S) {
            uint32_t ptr_idx = bh * S * 128 + k_idx * 128 + c;
            uint32_t v0 = pack_bf16_from_float_fn(__uint_as_float(r0), __uint_as_float(r1));
            uint32_t v1 = pack_bf16_from_float_fn(__uint_as_float(r2), __uint_as_float(r3));
            
            *(uint32_t*)(&gmem[ptr_idx]) = v0;
            *(uint32_t*)(&gmem[ptr_idx + 2]) = v1;
        }
    }
}

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* Q_ptr, const __nv_bfloat16* K_ptr, const __nv_bfloat16* V_ptr, 
    const __nv_bfloat16* O_ptr, const __nv_bfloat16* dO_ptr, const float* L_ptr,
    __nv_bfloat16* dQ_ptr, __nv_bfloat16* dK_ptr, __nv_bfloat16* dV_ptr,
    uint32_t S, uint32_t d) 
{
    uint32_t bh = blockIdx.y;
    uint32_t bid_x = blockIdx.x;
    
    extern __shared__ char smem[];
    __nv_bfloat16* m_Q = (__nv_bfloat16*)smem;
    __nv_bfloat16* m_K = (__nv_bfloat16*)(smem + 32768);
    __nv_bfloat16* m_V = (__nv_bfloat16*)(smem + 65536);
    __nv_bfloat16* m_dO = (__nv_bfloat16*)(smem + 98304);
    __nv_bfloat16* m_PT = (__nv_bfloat16*)(smem + 131072);
    __nv_bfloat16* m_dST = (__nv_bfloat16*)(smem + 163840);
    __nv_bfloat16* m_DT = (__nv_bfloat16*)(smem + 196608);
    __nv_bfloat16* m_dQT = (__nv_bfloat16*)(smem + 229376);
    
    uint64_t* mbar = (uint64_t*)(smem + 262144);
    
    uint32_t tmem_S_T, tmem_dP_T, tmem_dV_T, tmem_dK_T, tmem_dQ_T;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S_T, 128);
        tmem_alloc_fn(&tmem_dP_T, 128);
        tmem_alloc_fn(&tmem_dV_T, 128);
        tmem_alloc_fn(&tmem_dK_T, 128);
        tmem_alloc_fn(&tmem_dQ_T, 128);
        init_smem_barrier_fn(mbar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    load_tile_128(K_ptr, bh, S, bid_x, m_K);
    load_tile_128(V_ptr, bh, S, bid_x, m_V);
    fence_proxy_async_fn();
    
    uint32_t phase = 0;
    
    int max_j = (S + 127) / 128 - 1;
    if (max_j < bid_x) max_j = bid_x;
    
    for (int j = bid_x; j <= max_j; ++j) {
        load_tile_128(Q_ptr, bh, S, j, m_Q);
        load_tile_128(dO_ptr, bh, S, j, m_dO);
        compute_D(O_ptr, dO_ptr, bh, S, j, m_DT);
        fence_proxy_async_fn();
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 0);
            umma_128x128_kt(tmem_S_T, m_K, true, m_Q, true, (j == bid_x) ? 0 : 1);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];" 
                :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        compute_P_and_dS_causal(tmem_S_T, tmem_dP_T, L_ptr, bh, S, j * 128, bid_x * 128, m_PT, m_dST, m_DT);
        fence_proxy_async_fn();
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 0);
            umma_128x128_kt(tmem_dP_T, m_V, true, m_dO, true, 0);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];" 
                :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        compute_P_and_dS_causal(tmem_S_T, tmem_dP_T, L_ptr, bh, S, j * 128, bid_x * 128, m_PT, m_dST, m_DT);
        fence_proxy_async_fn();
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 0);
            umma_128x128_kt(tmem_dV_T, m_PT, true, m_dO, false, (j == bid_x) ? 0 : 1);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];" 
                :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 0);
            umma_128x128_kt(tmem_dK_T, m_dST, true, m_Q, false, (j == bid_x) ? 0 : 1);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];" 
                :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 0);
            umma_128x128_kt(tmem_dQ_T, m_dST, false, m_K, true, 0);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];" 
                :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        atomic_add_dQ(tmem_dQ_T, dQ_ptr, bh, S, j);
    }
    
    store_tmem_128(tmem_dV_T, dV_ptr, bh, S, bid_x);
    store_tmem_128(tmem_dK_T, dK_ptr, bh, S, bid_x);
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_S_T, 128);
        tmem_dealloc_fn(tmem_dP_T, 128);
        tmem_dealloc_fn(tmem_dV_T, 128);
        tmem_dealloc_fn(tmem_dK_T, 128);
        tmem_dealloc_fn(tmem_dQ_T, 128);
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

  dim3 grid((S + 127) / 128, B * H);
  dim3 block(128);
  
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  
  int smem_size = 32768 * 9;
  CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  
  mha_bwd_kernel<<<grid, block, smem_size, stream>>>(
      static_cast<const __nv_bfloat16*>(Q.data_ptr()), 
      static_cast<const __nv_bfloat16*>(K.data_ptr()), 
      static_cast<const __nv_bfloat16*>(V.data_ptr()), 
      static_cast<const __nv_bfloat16*>(O.data_ptr()), 
      static_cast<const __nv_bfloat16*>(dO.data_ptr()), 
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