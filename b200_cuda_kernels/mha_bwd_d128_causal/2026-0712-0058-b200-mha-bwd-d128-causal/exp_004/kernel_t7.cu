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

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
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

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t tmem_addr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(tmem_addr));
}

__device__ __forceinline__ void umma_commit_1sm_cta(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N, bool a_transposed, bool b_transposed) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((a_transposed ? 1 : 0) << 15);   
    d |= ((b_transposed ? 1 : 0) << 16);   
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

__device__ __forceinline__ uint64_t make_smem_desc_k_major_chunk(const void* ptr, int chunk) {
    uint32_t addr_offset = chunk * 32; 
    uint32_t lbo = (addr_offset < 128) ? 1 : 16384;
    return make_smem_desc_sm100_fn((char*)ptr + addr_offset, lbo, 1024);
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major_chunk(const void* ptr, int chunk) {
    uint32_t addr_offset = chunk * 4096; 
    uint32_t lbo = (addr_offset < 8192) ? 1024 : 16384; 
    return make_smem_desc_sm100_fn((char*)ptr + addr_offset, lbo, 1024);
}

__device__ __forceinline__ void gemm_128x128(uint32_t tmem_c, const __nv_bfloat16* A, const __nv_bfloat16* B) {
    for (int chunk = 0; chunk < 8; ++chunk) {
        uint32_t idesc = make_instr_desc(128, 128, false, true);
        uint64_t desc_A = make_smem_desc_k_major_chunk(A, chunk);
        uint64_t desc_B = make_smem_desc_mn_major_chunk(B, chunk);
        asm volatile(
            "{\n.reg .pred p;\n"
            "setp.ne.b32 p, 1, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
            :: "r"(tmem_c), "l"(desc_A), "l"(desc_B), "r"(idesc));
    }
}

__device__ __forceinline__ void gemm_128x128_dQ(uint32_t tmem_c, const __nv_bfloat16* dST, const __nv_bfloat16* K) {
    for (int chunk = 0; chunk < 8; ++chunk) {
        uint32_t idesc = make_instr_desc(128, 128, false, false);
        uint64_t desc_A = make_smem_desc_k_major_chunk(dST, chunk);
        uint64_t desc_B = make_smem_desc_k_major_chunk(K, chunk);
        asm volatile(
            "{\n.reg .pred p;\n"
            "setp.ne.b32 p, 1, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
            :: "r"(tmem_c), "l"(desc_A), "l"(desc_B), "r"(idesc));
    }
}

__device__ __forceinline__ void gemm_128x128_dK(uint32_t tmem_c, const __nv_bfloat16* dST, const __nv_bfloat16* Q) {
    for (int chunk = 0; chunk < 8; ++chunk) {
        uint32_t idesc = make_instr_desc(128, 128, true, true);
        uint64_t desc_A = make_smem_desc_mn_major_chunk(dST, chunk);
        uint64_t desc_B = make_smem_desc_mn_major_chunk(Q, chunk);
        asm volatile(
            "{\n.reg .pred p;\n"
            "setp.ne.b32 p, 1, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
            :: "r"(tmem_c), "l"(desc_A), "l"(desc_B), "r"(idesc));
    }
}

__device__ __forceinline__ void gemm_128x128_dV(uint32_t tmem_c, const __nv_bfloat16* PT, const __nv_bfloat16* dO) {
    for (int chunk = 0; chunk < 8; ++chunk) {
        uint32_t idesc = make_instr_desc(128, 128, true, true);
        uint64_t desc_A = make_smem_desc_mn_major_chunk(PT, chunk);
        uint64_t desc_B = make_smem_desc_mn_major_chunk(dO, chunk);
        asm volatile(
            "{\n.reg .pred p;\n"
            "setp.ne.b32 p, 1, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
            :: "r"(tmem_c), "l"(desc_A), "l"(desc_B), "r"(idesc));
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

__device__ __forceinline__ __nv_bfloat16 read_smem_128_128_swizzled_128B(const __nv_bfloat16* smem, uint32_t row, uint32_t col) {
    uint32_t col_bytes = col * 2;
    uint32_t offset = (row * 128 + swizzle_128B(row, col_bytes)) * 2;
    return *(const __nv_bfloat16*)((const char*)smem + offset);
}

__device__ __forceinline__ void write_smem_128_128_swizzled_128B(
    __nv_bfloat16* smem, uint32_t row, uint32_t col, __nv_bfloat16 val) {
    uint32_t col_bytes = col * 2;
    uint32_t offset = (row * 128 + swizzle_128B(row, col_bytes)) * 2;
    *(uint16_t*)((char*)smem + offset) = *(uint16_t*)&val;
}

__device__ __forceinline__ void compute_D_and_L(const float* L_ptr, uint32_t bh, uint32_t S, uint32_t j, int tid, 
                                float* m_LT, __nv_bfloat16* m_DT,
                                const __nv_bfloat16* m_O_ptr, const __nv_bfloat16* m_dO_ptr) {
    int q_idx = j * 128 + tid;
    float lse = (q_idx < S) ? L_ptr[bh * S + q_idx] : 0.0f;
    m_LT[tid] = lse;
    
    float sum = 0;
    if (q_idx < S) {
        for(int i = 0; i < 128; i += 2) {
            __nv_bfloat16 o0 = read_smem_128_128_swizzled_128B(m_O_ptr, tid, i);
            __nv_bfloat16 o1 = read_smem_128_128_swizzled_128B(m_O_ptr, tid, i + 1);
            __nv_bfloat16 do0 = read_smem_128_128_swizzled_128B(m_dO_ptr, tid, i);
            __nv_bfloat16 do1 = read_smem_128_128_swizzled_128B(m_dO_ptr, tid, i + 1);
            sum += __bfloat162float(o0) * __bfloat162float(do0);
            sum += __bfloat162float(o1) * __bfloat162float(do1);
        }
    }
    m_DT[tid] = __float2bfloat16(sum);
}

__device__ __forceinline__ void compute_P_and_dS_causal(
    uint32_t tmem_S_T, uint32_t tmem_dP_T, 
    float* m_LT, uint32_t bh, uint32_t S, 
    uint32_t global_q_start, uint32_t global_k_start, 
    __nv_bfloat16* m_PT, __nv_bfloat16* m_dST, __nv_bfloat16* m_DT, float scale_factor) 
{
    int warp_id = threadIdx.x / 32;
    int r_base = warp_id * 32;
    int r_idx = r_base + threadIdx.x % 32;
    
    float S_reg[4], dP_reg[4];
    uint32_t r0, r1, r2, r3;
    tmem_load_4x_fn((r_base << 16) | 0, &r0, &r1, &r2, &r3);
    tmem_load_fence_fn();
    S_reg[0] = __uint_as_float(r0);
    S_reg[1] = __uint_as_float(r1);
    S_reg[2] = __uint_as_float(r2);
    S_reg[3] = __uint_as_float(r3);

    tmem_load_4x_fn((r_base << 16) | 0, &r0, &r1, &r2, &r3);
    tmem_load_fence_fn();
    dP_reg[0] = __uint_as_float(r0);
    dP_reg[1] = __uint_as_float(r1);
    dP_reg[2] = __uint_as_float(r2);
    dP_reg[3] = __uint_as_float(r3);
    
    for (int c = 0; c < 128; c += 4) {
        tmem_load_4x_fn((r_base << 16) | c, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        S_reg[0] = __uint_as_float(r0);
        S_reg[1] = __uint_as_float(r1);
        S_reg[2] = __uint_as_float(r2);
        S_reg[3] = __uint_as_float(r3);

        tmem_load_4x_fn((r_base << 16) | c, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        dP_reg[0] = __uint_as_float(r0);
        dP_reg[1] = __uint_as_float(r1);
        dP_reg[2] = __uint_as_float(r2);
        dP_reg[3] = __uint_as_float(r3);
        
        int q_idx = global_q_start + r_idx;
        int k0 = global_k_start + c;
        int k1 = global_k_start + c + 1;
        int k2 = global_k_start + c + 2;
        int k3 = global_k_start + c + 3;
        
        float lse = m_LT[r_idx];
        float d_val = __bfloat162float(m_DT[r_idx]);
        
        bool is_causal0 = (k0 <= q_idx && q_idx < S && k0 < S);
        bool is_causal1 = (k1 <= q_idx && q_idx < S && k1 < S);
        bool is_causal2 = (k2 <= q_idx && q_idx < S && k2 < S);
        bool is_causal3 = (k3 <= q_idx && q_idx < S && k3 < S);

        float p0 = is_causal0 ? fast_exp2f_fn((S_reg[0] - lse) * scale_factor) : 0.0f;
        float p1 = is_causal1 ? fast_exp2f_fn((S_reg[1] - lse) * scale_factor) : 0.0f;
        float p2 = is_causal2 ? fast_exp2f_fn((S_reg[2] - lse) * scale_factor) : 0.0f;
        float p3 = is_causal3 ? fast_exp2f_fn((S_reg[3] - lse) * scale_factor) : 0.0f;
        
        write_smem_128_128_swizzled_128B(m_PT, r_idx, c, __float2bfloat16(p0));
        write_smem_128_128_swizzled_128B(m_PT, r_idx, c+1, __float2bfloat16(p1));
        write_smem_128_128_swizzled_128B(m_PT, r_idx, c+2, __float2bfloat16(p2));
        write_smem_128_128_swizzled_128B(m_PT, r_idx, c+3, __float2bfloat16(p3));
        
        float dp0 = dP_reg[0];
        float dp1 = dP_reg[1];
        float dp2 = dP_reg[2];
        float dp3 = dP_reg[3];
        
        float ds0 = is_causal0 ? p0 * (dp0 - d_val) : 0.0f;
        float ds1 = is_causal1 ? p1 * (dp1 - d_val) : 0.0f;
        float ds2 = is_causal2 ? p2 * (dp2 - d_val) : 0.0f;
        float ds3 = is_causal3 ? p3 * (dp3 - d_val) : 0.0f;
        
        write_smem_128_128_swizzled_128B(m_dST, r_idx, c, __float2bfloat16(ds0));
        write_smem_128_128_swizzled_128B(m_dST, r_idx, c+1, __float2bfloat16(ds1));
        write_smem_128_128_swizzled_128B(m_dST, r_idx, c+2, __float2bfloat16(ds2));
        write_smem_128_128_swizzled_128B(m_dST, r_idx, c+3, __float2bfloat16(ds3));
    }
}

__device__ __forceinline__ void store_tmem_128_packed(__nv_bfloat16* gmem, uint32_t bh, uint32_t S, uint32_t tile_idx, uint32_t tmem_base) {
    int warp_id = threadIdx.x / 32;
    int r_base = warp_id * 32;
    
    for (int c = 0; c < 128; c += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn((r_base << 16) | c, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        
        int row_idx = tile_idx * 128 + r_base + threadIdx.x % 32;
        if (row_idx < S) {
            uint32_t ptr_idx0 = bh * S * 128 + row_idx * 128 + c;
            uint32_t v0 = pack_bf16_from_float_fn(__uint_as_float(r0), __uint_as_float(r1));
            uint32_t v1 = pack_bf16_from_float_fn(__uint_as_float(r2), __uint_as_float(r3));
            *(uint32_t*)(&gmem[ptr_idx0]) = v0;
            *(uint32_t*)(&gmem[ptr_idx0 + 2]) = v1;
            
            uint32_t ptr_idx1 = bh * S * 128 + row_idx * 128 + c + 64;
            uint32_t v2 = pack_bf16_from_float_fn(__uint_as_float(r0), __uint_as_float(r1));
            uint32_t v3 = pack_bf16_from_float_fn(__uint_as_float(r2), __uint_as_float(r3));
            *(uint32_t*)(&gmem[ptr_idx1]) = v2;
            *(uint32_t*)(&gmem[ptr_idx1 + 2]) = v3;
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

__global__ void mha_bwd_kernel_dQ(
    const __grid_constant__ CUtensorMap tma_Q, 
    const __grid_constant__ CUtensorMap tma_K, 
    const __grid_constant__ CUtensorMap tma_V, 
    const __grid_constant__ CUtensorMap tma_dO, 
    const __grid_constant__ CUtensorMap tma_O,
    const float* L_ptr,
    __nv_bfloat16* dQ_ptr,
    uint32_t S, uint32_t d) 
{
    uint32_t bh = blockIdx.y;
    uint32_t j = blockIdx.x;
    
    extern __shared__ char smem_raw[];
    char* smem = (char*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1024);
    
    __nv_bfloat16* m_Q_ptr = (__nv_bfloat16*)smem;
    __nv_bfloat16* m_K_ptr = (__nv_bfloat16*)(smem + 32768);
    __nv_bfloat16* m_V_ptr = (__nv_bfloat16*)(smem + 65536);
    __nv_bfloat16* m_dO_ptr = (__nv_bfloat16*)(smem + 98304);
    __nv_bfloat16* m_O_ptr = (__nv_bfloat16*)(smem + 131072);
    __nv_bfloat16* m_PT = (__nv_bfloat16*)(smem + 163840);
    __nv_bfloat16* m_dST = (__nv_bfloat16*)(smem + 196608);
    __nv_bfloat16* m_DT = (__nv_bfloat16*)(smem + 229376);
    float* m_LT = (float*)(smem + 229632);
    uint64_t* mbar = (uint64_t*)(smem + 230144);
    
    uint32_t tmem_S_T, tmem_dP_T, tmem_dQ_T0, tmem_dQ_T1;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S_T, 128);
        tmem_alloc_fn(&tmem_dP_T, 128);
        tmem_alloc_fn(&tmem_dQ_T0, 128);
        tmem_alloc_fn(&tmem_dQ_T1, 128);
        init_smem_barrier_fn(mbar, 2);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        uint32_t seq_off = bh * S + j * 128;
        mbarrier_arrive_and_expect_tx_fn(mbar, 98304);
        tma_load_2d_fn(&tma_Q, mbar, m_Q_ptr, 0, seq_off);
        tma_load_2d_fn(&tma_Q, mbar, m_Q_ptr + 8192, 64, seq_off);
        tma_load_2d_fn(&tma_O, mbar, m_O_ptr, 0, seq_off);
        tma_load_2d_fn(&tma_O, mbar, m_O_ptr + 8192, 64, seq_off);
        tma_load_2d_fn(&tma_dO, mbar, m_dO_ptr, 0, seq_off);
        tma_load_2d_fn(&tma_dO, mbar, m_dO_ptr + 8192, 64, seq_off);
    }
    mbarrier_wait_fn(mbar, 0);
    fence_proxy_async_fn();
    
    compute_D_and_L(L_ptr, bh, S, j, threadIdx.x, m_LT, m_DT, m_O_ptr, m_dO_ptr);
    
    uint32_t phase = 0;
    uint32_t accumulate_dQ = 0;
    float scale = 1.0f / sqrtf((float)d);
    float scale_factor = scale * 1.44269504f;
    
    for (int k = 0; k <= j; ++k) {
        if (threadIdx.x == 0) {
            uint32_t seq_off = bh * S + k * 128;
            mbarrier_arrive_and_expect_tx_fn(mbar, 65536);
            tma_load_2d_fn(&tma_K, mbar, m_K_ptr, 0, seq_off);
            tma_load_2d_fn(&tma_K, mbar, m_K_ptr + 8192, 64, seq_off);
            tma_load_2d_fn(&tma_V, mbar, m_V_ptr, 0, seq_off);
            tma_load_2d_fn(&tma_V, mbar, m_V_ptr + 8192, 64, seq_off);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        fence_proxy_async_fn();
        
        if (threadIdx.x == 0) {
            uint32_t idesc_S = make_instr_desc(128, 128, false, true);
            gemm_128x128(tmem_S_T, m_Q_ptr, m_K_ptr);
            umma_commit_1sm_cta(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        if (threadIdx.x == 0) {
            uint32_t idesc_dP = make_instr_desc(128, 128, false, true);
            gemm_128x128(tmem_dP_T, m_V_ptr, m_dO_ptr);
            umma_commit_1sm_cta(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        compute_P_and_dS_causal(tmem_S_T, tmem_dP_T, m_LT, bh, S, j * 128, k * 128, m_PT, m_dST, m_DT, scale_factor);
        __syncthreads(); 
        fence_proxy_async_fn();
        
        if (threadIdx.x == 0) {
            uint32_t idesc_dQ = make_instr_desc(128, 128, false, false);
            gemm_128x128_dQ(tmem_dQ_T0, m_dST, m_K_ptr);
            gemm_128x128_dQ(tmem_dQ_T1, m_dST, m_K_ptr + 8192);
            umma_commit_1sm_cta(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        accumulate_dQ = 1;
    }
    
    store_tmem_128_packed(dQ_ptr, bh, S, j, tmem_dQ_T0);
    store_tmem_128_packed(dQ_ptr, bh, S, j, tmem_dQ_T1);
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_S_T, 128);
        tmem_dealloc_fn(tmem_dP_T, 128);
        tmem_dealloc_fn(tmem_dQ_T0, 128);
        tmem_dealloc_fn(tmem_dQ_T1, 128);
    }
}

__global__ void mha_bwd_kernel_dK_dV(
    const __grid_constant__ CUtensorMap tma_Q, 
    const __grid_constant__ CUtensorMap tma_K, 
    const __grid_constant__ CUtensorMap tma_V, 
    const __grid_constant__ CUtensorMap tma_dO, 
    const __grid_constant__ CUtensorMap tma_O,
    const float* L_ptr,
    __nv_bfloat16* dK_ptr, __nv_bfloat16* dV_ptr,
    uint32_t S, uint32_t d) 
{
    uint32_t bh = blockIdx.y;
    uint32_t k = blockIdx.x;
    
    extern __shared__ char smem_raw[];
    char* smem = (char*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1024);
    
    __nv_bfloat16* m_Q_ptr = (__nv_bfloat16*)smem;
    __nv_bfloat16* m_K_ptr = (__nv_bfloat16*)(smem + 32768);
    __nv_bfloat16* m_V_ptr = (__nv_bfloat16*)(smem + 65536);
    __nv_bfloat16* m_dO_ptr = (__nv_bfloat16*)(smem + 98304);
    __nv_bfloat16* m_O_ptr = (__nv_bfloat16*)(smem + 131072);
    __nv_bfloat16* m_PT = (__nv_bfloat16*)(smem + 163840);
    __nv_bfloat16* m_dST = (__nv_bfloat16*)(smem + 196608);
    __nv_bfloat16* m_DT = (__nv_bfloat16*)(smem + 229376);
    float* m_LT = (float*)(smem + 229632);
    uint64_t* mbar = (uint64_t*)(smem + 230144);
    
    uint32_t tmem_S_T, tmem_dP_T, tmem_dK_T0, tmem_dK_T1, tmem_dV_T0, tmem_dV_T1;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S_T, 128);
        tmem_alloc_fn(&tmem_dP_T, 128);
        tmem_alloc_fn(&tmem_dK_T0, 128);
        tmem_alloc_fn(&tmem_dK_T1, 128);
        tmem_alloc_fn(&tmem_dV_T0, 128);
        tmem_alloc_fn(&tmem_dV_T1, 128);
        init_smem_barrier_fn(mbar, 2);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        uint32_t seq_off = bh * S + k * 128;
        mbarrier_arrive_and_expect_tx_fn(mbar, 65536);
        tma_load_2d_fn(&tma_K, mbar, m_K_ptr, 0, seq_off);
        tma_load_2d_fn(&tma_K, mbar, m_K_ptr + 8192, 64, seq_off);
        tma_load_2d_fn(&tma_V, mbar, m_V_ptr, 0, seq_off);
        tma_load_2d_fn(&tma_V, mbar, m_V_ptr + 8192, 64, seq_off);
    }
    mbarrier_wait_fn(mbar, 0);
    fence_proxy_async_fn();
    
    uint32_t phase = 0;
    uint32_t accumulate_dK_dV = 0;
    float scale = 1.0f / sqrtf((float)d);
    float scale_factor = scale * 1.44269504f;
    
    int max_j = (S + 127) / 128 - 1;
    
    for (int j = k; j <= max_j; ++j) {
        if (threadIdx.x == 0) {
            uint32_t seq_off = bh * S + j * 128;
            mbarrier_arrive_and_expect_tx_fn(mbar, 98304);
            tma_load_2d_fn(&tma_Q, mbar, m_Q_ptr, 0, seq_off);
            tma_load_2d_fn(&tma_Q, mbar, m_Q_ptr + 8192, 64, seq_off);
            tma_load_2d_fn(&tma_O, mbar, m_O_ptr, 0, seq_off);
            tma_load_2d_fn(&tma_O, mbar, m_O_ptr + 8192, 64, seq_off);
            tma_load_2d_fn(&tma_dO, mbar, m_dO_ptr, 0, seq_off);
            tma_load_2d_fn(&tma_dO, mbar, m_dO_ptr + 8192, 64, seq_off);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        fence_proxy_async_fn();
        
        compute_D_and_L(L_ptr, bh, S, j, threadIdx.x, m_LT, m_DT, m_O_ptr, m_dO_ptr);
        
        if (threadIdx.x == 0) {
            uint32_t idesc_S = make_instr_desc(128, 128, false, true);
            gemm_128x128(tmem_S_T, m_Q_ptr, m_K_ptr);
            umma_commit_1sm_cta(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        if (threadIdx.x == 0) {
            uint32_t idesc_dP = make_instr_desc(128, 128, false, true);
            gemm_128x128(tmem_dP_T, m_V_ptr, m_dO_ptr);
            umma_commit_1sm_cta(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        compute_P_and_dS_causal(tmem_S_T, tmem_dP_T, m_LT, bh, S, j * 128, k * 128, m_PT, m_dST, m_DT, scale_factor);
        __syncthreads(); 
        fence_proxy_async_fn();
        
        if (threadIdx.x == 0) {
            uint32_t idesc_dK = make_instr_desc(128, 128, true, true);
            gemm_128x128_dK(tmem_dK_T0, m_dST, m_Q_ptr);
            gemm_128x128_dK(tmem_dK_T1, m_dST, m_Q_ptr + 8192);
            umma_commit_1sm_cta(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        if (threadIdx.x == 0) {
            uint32_t idesc_dV = make_instr_desc(128, 128, true, true);
            gemm_128x128_dV(tmem_dV_T0, m_PT, m_dO_ptr);
            gemm_128x128_dV(tmem_dV_T1, m_PT, m_dO_ptr + 8192);
            umma_commit_1sm_cta(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        accumulate_dK_dV = 1;
    }
    
    store_tmem_128_packed(dK_ptr, bh, S, k, tmem_dK_T0);
    store_tmem_128_packed(dK_ptr, bh, S, k, tmem_dK_T1);
    store_tmem_128_packed(dV_ptr, bh, S, k, tmem_dV_T0);
    store_tmem_128_packed(dV_ptr, bh, S, k, tmem_dV_T1);
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_S_T, 128);
        tmem_dealloc_fn(tmem_dP_T, 128);
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
  uint64_t inner = d;
  uint64_t outer = B * H * S;
  
  CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), inner, outer, 64, 128, 
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), inner, outer, 64, 128, 
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), inner, outer, 64, 128, 
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  CU_CHECK(create_tma_2d_descriptor_2B(&tma_dO, dO.data_ptr(), inner, outer, 64, 128, 
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  CU_CHECK(create_tma_2d_descriptor_2B(&tma_O, O.data_ptr(), inner, outer, 64, 128, 
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  dim3 grid((S + 127) / 128, B * H);
  dim3 block(128);
  
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  
  int smem_size_dQ = 226 * 1024;
  CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel_dQ, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dQ));

  int smem_size_dK_dV = 226 * 1024;
  CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel_dK_dV, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dK_dV));
  
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem_size_dQ;
  config.stream = stream;
  
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  
  CUDA_CHECK(cudaLaunchKernelEx(&config, mha_bwd_kernel_dQ, 
      tma_Q, tma_K, tma_V, tma_dO, tma_O,
      static_cast<const float*>(L.data_ptr()),
      static_cast<__nv_bfloat16*>(dQ.data_ptr()),
      S, d));
      
  config.dynamicSmemBytes = smem_size_dK_dV;
  CUDA_CHECK(cudaLaunchKernelEx(&config, mha_bwd_kernel_dK_dV, 
      tma_Q, tma_K, tma_V, tma_dO, tma_O,
      static_cast<const float*>(L.data_ptr()),
      static_cast<__nv_bfloat16*>(dK.data_ptr()), 
      static_cast<__nv_bfloat16*>(dV.data_ptr()),
      S, d));
      
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_example_cuda