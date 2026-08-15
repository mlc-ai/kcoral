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

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
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

__device__ __forceinline__ void fence_proxy_async_shared_fn() {
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

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, bool is_mn_major, int stride_bytes) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    uint32_t sbo = 1024;
    uint32_t lbo = is_mn_major ? 8192 : 1;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool transpose_A, bool transpose_B) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((transpose_A ? 1 : 0) << 15);   
    d |= ((transpose_B ? 1 : 0) << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void gemm_S_S(
    bool first, uint32_t tmem_base, 
    void* ptr_A, bool is_A_mn, 
    void* ptr_B, bool is_B_mn, 
    bool transpose_A, bool transpose_B) {
    
    uint32_t idesc = make_instr_desc_fn(64, 64, transpose_A, transpose_B);
    for (int k = 0; k < 4; ++k) {
        int step_A = is_A_mn ? 2048 : 32;
        int step_B = is_B_mn ? 2048 : 32;
        uint64_t desc_A = make_smem_desc_sm100_fn((char*)ptr_A + k * step_A, is_A_mn, step_A);
        uint64_t desc_B = make_smem_desc_sm100_fn((char*)ptr_B + k * step_B, is_B_mn, step_B);
        uint32_t accum = (k == 0 && first) ? 0 : 1;
        umma_f16_cg1_fn(tmem_base, desc_A, desc_B, idesc, accum);
    }
}

__device__ __forceinline__ void gemm_D_S(
    bool first, uint32_t tmem_D, 
    void* smem_A, bool is_A_mn, 
    void* smem_B, bool is_B_mn) {
    uint32_t idesc = make_instr_desc_fn(64, 64, is_A_mn, is_B_mn);
    for (int k = 0; k < 4; ++k) {
        int step_A = is_A_mn ? 2048 : 32;
        int step_B = is_B_mn ? 2048 : 32;
        uint64_t desc_A = make_smem_desc_sm100_fn((char*)smem_A + k * step_A, is_A_mn, step_A);
        uint64_t desc_B = make_smem_desc_sm100_fn((char*)smem_B + k * step_B, is_B_mn, step_B);
        uint32_t accum = (k == 0 && first) ? 0 : 1;
        umma_f16_cg1_fn(tmem_D, desc_A, desc_B, idesc, accum);
    }
}

__device__ __forceinline__ int swizzle_128B(int row, int col) {
    int chunk_x = col / 8;
    int chunk_y = row % 8;
    int swizzled_chunk_x = chunk_y ^ chunk_x;
    return row * 64 + swizzled_chunk_x * 8 + (col % 8);
}

__global__ void __launch_bounds__(128, 2) mha_bwd_kernel_opt(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dQ,
    const __grid_constant__ CUtensorMap tma_dK,
    const __grid_constant__ CUtensorMap tma_dV,
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* L,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    uint32_t S_dim, uint32_t H_dim, float scale) 
{
    extern __shared__ char smem_raw[];
    char* smem_ptr = (char*)(((uintptr_t)smem_raw + 1023) & ~1023ULL);

    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem_ptr;                
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)(smem_ptr + 8192);       
    __nv_bfloat16* smem_K0 = (__nv_bfloat16*)(smem_ptr + 16384);      
    __nv_bfloat16* smem_K1 = (__nv_bfloat16*)(smem_ptr + 24576);      
    __nv_bfloat16* smem_V0 = (__nv_bfloat16*)(smem_ptr + 32768);      
    __nv_bfloat16* smem_V1 = (__nv_bfloat16*)(smem_ptr + 40960);      
    __nv_bfloat16* smem_dO0 = (__nv_bfloat16*)(smem_ptr + 49152);     
    __nv_bfloat16* smem_dO1 = (__nv_bfloat16*)(smem_ptr + 57344);     
    
    __nv_bfloat16* smem_P_T  = (__nv_bfloat16*)(smem_ptr + 65536);     
    __nv_bfloat16* smem_dS_T = (__nv_bfloat16*)(smem_ptr + 73728);     
    __nv_bfloat16* smem_dS_ST= (__nv_bfloat16*)(smem_ptr + 81920);     
    
    uint32_t* tmem_S  = (uint32_t*)(smem_ptr + 90112);               
    uint32_t* tmem_dP = (uint32_t*)(smem_ptr + 90112 + 4);           
    uint32_t* tmem_dK_0 = (uint32_t*)(smem_ptr + 90112 + 8);         
    uint32_t* tmem_dK_1 = (uint32_t*)(smem_ptr + 90112 + 12);        
    uint32_t* tmem_dV_0 = (uint32_t*)(smem_ptr + 90112 + 16);        
    uint32_t* tmem_dV_1 = (uint32_t*)(smem_ptr + 90112 + 20);        
    uint32_t* tmem_dQ_0 = (uint32_t*)(smem_ptr + 90112 + 24);        
    uint32_t* tmem_dQ_1 = (uint32_t*)(smem_ptr + 90112 + 28);         
    
    uint64_t* mbar_S    = (uint64_t*)(smem_ptr + 90136);               
    uint64_t* mbar_Q    = (uint64_t*)(smem_ptr + 90144);               
    uint64_t* mbar_K    = (uint64_t*)(smem_ptr + 90152);               
    uint64_t* mbar_V    = (uint64_t*)(smem_ptr + 90160);               
    uint64_t* mbar_dO   = (uint64_t*)(smem_ptr + 90168);               

    uint32_t g_idx = blockIdx.x;
    uint32_t S_blocks = (S_dim + 63) / 64;
    
    uint32_t seq_blk_idx = g_idx % S_blocks;
    uint32_t bid_y = g_idx / S_blocks;
    uint32_t head_idx = bid_y % H_dim;
    uint32_t batch_idx = bid_y / H_dim;

    uint32_t phase_Q = 0, phase_K = 0, phase_V = 0, phase_dO = 0, phase_S = 0;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_S, 1);
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_dO, 1);

        tmem_alloc_fn(tmem_S, 64);
        tmem_alloc_fn(tmem_dP, 64);
        tmem_alloc_fn(tmem_dK_0, 64);
        tmem_alloc_fn(tmem_dK_1, 64);
        tmem_alloc_fn(tmem_dV_0, 64);
        tmem_alloc_fn(tmem_dV_1, 64);
        tmem_alloc_fn(tmem_dQ_0, 64);
        tmem_alloc_fn(tmem_dQ_1, 64);
    }
    
    uint32_t tmem_S_addr, tmem_dP_addr, tmem_dK_0_addr, tmem_dK_1_addr, tmem_dV_0_addr, tmem_dV_1_addr, tmem_dQ_0_addr, tmem_dQ_1_addr;
    
    if (threadIdx.x == 0) {
        tmem_S_addr = *tmem_S;
        tmem_dP_addr = *tmem_dP;
        tmem_dK_0_addr = *tmem_dK_0;
        tmem_dK_1_addr = *tmem_dK_1;
        tmem_dV_0_addr = *tmem_dV_0;
        tmem_dV_1_addr = *tmem_dV_1;
        tmem_dQ_0_addr = *tmem_dQ_0;
        tmem_dQ_1_addr = *tmem_dQ_1;
    }
    
    __syncthreads();
    
    int c1 = (bid_y * S_dim) + (seq_blk_idx * 64);

    // -------------------------------------------------------------------------
    // PASS 1: Compute dK and dV
    // -------------------------------------------------------------------------
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_K, 16384);
        tma_load_2d_fn(&tma_K, mbar_K, smem_K0, 0, c1);
        tma_load_2d_fn(&tma_K, mbar_K, smem_K1, 64, c1);
        
        mbarrier_arrive_and_expect_tx_fn(mbar_V, 16384);
        tma_load_2d_fn(&tma_V, mbar_V, smem_V0, 0, c1);
        tma_load_2d_fn(&tma_V, mbar_V, smem_V1, 64, c1);
    }
    mbarrier_wait_fn(mbar_K, phase_K); phase_K ^= 1;
    mbarrier_wait_fn(mbar_V, phase_V); phase_V ^= 1;
    fence_proxy_async_shared_fn();
    __syncthreads();

    bool first_dV0 = true, first_dV1 = true;
    bool first_dK0 = true, first_dK1 = true;

    for (int q_blk = 0; q_blk < S_blocks; ++q_blk) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_Q, 16384);
            int q_c1 = (bid_y * S_dim) + (q_blk * 64);
            tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q0, 0, q_c1);
            tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q1, 64, q_c1);
            
            mbarrier_arrive_and_expect_tx_fn(mbar_dO, 16384);
            tma_load_2d_fn(&tma_dO, mbar_dO, smem_dO0, 0, q_c1);
            tma_load_2d_fn(&tma_dO, mbar_dO, smem_dO1, 64, q_c1);
        }
        mbarrier_wait_fn(mbar_Q, phase_Q); phase_Q ^= 1;
        mbarrier_wait_fn(mbar_dO, phase_dO); phase_dO ^= 1;
        fence_proxy_async_shared_fn();
        __syncthreads();

        uint32_t first_gemm = 1;
        gemm_S_S(first_gemm, tmem_S_addr, smem_Q0, false, smem_K0, false, false, false);
        first_gemm = 0;
        gemm_S_S(first_gemm, tmem_S_addr, smem_Q1, false, smem_K1, false, false, false);
        umma_commit_1sm_fn(mbar_S);
        mbarrier_wait_fn(mbar_S, phase_S); phase_S ^= 1;
        
        float log2e = 1.4426950408889634f;
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_S_addr + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            if (threadIdx.x < 64) {
                int row = threadIdx.x;
                float s0 = __uint_as_float(r0) * scale;
                float s1 = __uint_as_float(r1) * scale;
                float s2 = __uint_as_float(r2) * scale;
                float s3 = __uint_as_float(r3) * scale;
                
                float l0 = (q_blk * 64 + row < S_dim) ? L[(bid_y * S_dim) + (q_blk * 64) + row] : 0.0f;
                float l1 = l0; float l2 = l0; float l3 = l0;
                
                float p0 = fast_exp2f_fn((s0 - l0) * log2e);
                float p1 = fast_exp2f_fn((s1 - l1) * log2e);
                float p2 = fast_exp2f_fn((s2 - l2) * log2e);
                float p3 = fast_exp2f_fn((s3 - l3) * log2e);
                
                smem_P_T[swizzle_128B(col + 0, row)] = __float2bfloat16(p0);
                smem_P_T[swizzle_128B(col + 1, row)] = __float2bfloat16(p1);
                smem_P_T[swizzle_128B(col + 2, row)] = __float2bfloat16(p2);
                smem_P_T[swizzle_128B(col + 3, row)] = __float2bfloat16(p3);
            }
        }
        __syncthreads();
        fence_proxy_async_shared_fn();
        __syncthreads();
        
        first_gemm = 1;
        gemm_D_S(first_gemm, tmem_dV_0_addr, smem_P_T, false, smem_dO0, true);
        first_gemm = 0;
        gemm_D_S(first_gemm, tmem_dV_1_addr, smem_P_T, false, smem_dO1, true);
        umma_commit_1sm_fn(mbar_S);
        mbarrier_wait_fn(mbar_S, phase_S); phase_S ^= 1;

        first_gemm = 1;
        gemm_S_S(first_gemm, tmem_dP_addr, smem_dO0, false, smem_V0, false, false, false);
        first_gemm = 0;
        gemm_S_S(first_gemm, tmem_dP_addr, smem_dO1, false, smem_V1, false, false, false);
        umma_commit_1sm_fn(mbar_S);
        mbarrier_wait_fn(mbar_S, phase_S); phase_S ^= 1;

        float r_val = 0;
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dP_addr + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            if (threadIdx.x < 64) {
                int row = threadIdx.x;
                float dp0 = __uint_as_float(r0);
                float dp1 = __uint_as_float(r1);
                float dp2 = __uint_as_float(r2);
                float dp3 = __uint_as_float(r3);
                float p0 = __bfloat162float(smem_P_T[swizzle_128B(col + 0, row)]);
                float p1 = __bfloat162float(smem_P_T[swizzle_128B(col + 1, row)]);
                float p2 = __bfloat162float(smem_P_T[swizzle_128B(col + 2, row)]);
                float p3 = __bfloat162float(smem_P_T[swizzle_128B(col + 3, row)]);
                r_val += p0 * dp0 + p1 * dp1 + p2 * dp2 + p3 * dp3;
                
                float ds0 = scale * p0 * (dp0 - r_val);
                float ds1 = scale * p1 * (dp1 - r_val);
                float ds2 = scale * p2 * (dp2 - r_val);
                float ds3 = scale * p3 * (dp3 - r_val);
                
                smem_dS_T[swizzle_128B(col + 0, row)] = __float2bfloat16(ds0);
                smem_dS_T[swizzle_128B(col + 1, row)] = __float2bfloat16(ds1);
                smem_dS_T[swizzle_128B(col + 2, row)] = __float2bfloat16(ds2);
                smem_dS_T[swizzle_128B(col + 3, row)] = __float2bfloat16(ds3);
            }
        }
        
        fence_proxy_async_shared_fn();
        __syncthreads();

        first_gemm = 1;
        gemm_D_S(first_gemm, tmem_dK_0_addr, smem_dS_T, false, smem_Q0, true);
        first_gemm = 0;
        gemm_D_S(first_gemm, tmem_dK_1_addr, smem_dS_T, false, smem_Q1, true);
        umma_commit_1sm_fn(mbar_S);
        mbarrier_wait_fn(mbar_S, phase_S); phase_S ^= 1;
        
        first_dV0 = false; first_dV1 = false;
        first_dK0 = false; first_dK1 = false;
        __syncthreads();
    }

    if (seq_blk_idx * 64 < S_dim) {
        if (threadIdx.x < 64) {
            int row = threadIdx.x;
            for (int col = 0; col < 64; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dK_0_addr + col));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                smem_K0[swizzle_128B(row, col + 0)] = __float2bfloat16(__uint_as_float(r0));
                smem_K0[swizzle_128B(row, col + 1)] = __float2bfloat16(__uint_as_float(r1));
                smem_K0[swizzle_128B(row, col + 2)] = __float2bfloat16(__uint_as_float(r2));
                smem_K0[swizzle_128B(row, col + 3)] = __float2bfloat16(__uint_as_float(r3));
            }
        }
        __syncthreads();
        fence_proxy_async_shared_fn();
        __syncthreads();
        
        if (threadIdx.x == 0) {
            tma_store_2d_fn(&tma_dK, smem_K0, 0, (bid_y * S_dim) + (seq_blk_idx * 64));
            tma_store_commit_fn();
        }
        tma_store_wait_fn<0>();
        __syncthreads();

        if (threadIdx.x < 64) {
            int row = threadIdx.x;
            for (int col = 0; col < 64; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dK_1_addr + col));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                smem_K1[swizzle_128B(row, col + 0)] = __float2bfloat16(__uint_as_float(r0));
                smem_K1[swizzle_128B(row, col + 1)] = __float2bfloat16(__uint_as_float(r1));
                smem_K1[swizzle_128B(row, col + 2)] = __float2bfloat16(__uint_as_float(r2));
                smem_K1[swizzle_128B(row, col + 3)] = __float2bfloat16(__uint_as_float(r3));
            }
        }
        __syncthreads();
        fence_proxy_async_shared_fn();
        __syncthreads();
        
        if (threadIdx.x == 0) {
            tma_store_2d_fn(&tma_dK, smem_K1, 64, (bid_y * S_dim) + (seq_blk_idx * 64));
            tma_store_commit_fn();
        }
        tma_store_wait_fn<0>();
        __syncthreads();
        
        if (threadIdx.x < 64) {
            int row = threadIdx.x;
            for (int col = 0; col < 64; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dV_0_addr + col));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                smem_V0[swizzle_128B(row, col + 0)] = __float2bfloat16(__uint_as_float(r0));
                smem_V0[swizzle_128B(row, col + 1)] = __float2bfloat16(__uint_as_float(r1));
                smem_V0[swizzle_128B(row, col + 2)] = __float2bfloat16(__uint_as_float(r2));
                smem_V0[swizzle_128B(row, col + 3)] = __float2bfloat16(__uint_as_float(r3));
            }
        }
        __syncthreads();
        fence_proxy_async_shared_fn();
        __syncthreads();
        
        if (threadIdx.x == 0) {
            tma_store_2d_fn(&tma_dV, smem_V0, 0, (bid_y * S_dim) + (seq_blk_idx * 64));
            tma_store_commit_fn();
        }
        tma_store_wait_fn<0>();
        __syncthreads();

        if (threadIdx.x < 64) {
            int row = threadIdx.x;
            for (int col = 0; col < 64; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dV_1_addr + col));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                smem_V1[swizzle_128B(row, col + 0)] = __float2bfloat16(__uint_as_float(r0));
                smem_V1[swizzle_128B(row, col + 1)] = __float2bfloat16(__uint_as_float(r1));
                smem_V1[swizzle_128B(row, col + 2)] = __float2bfloat16(__uint_as_float(r2));
                smem_V1[swizzle_128B(row, col + 3)] = __float2bfloat16(__uint_as_float(r3));
            }
        }
        __syncthreads();
        fence_proxy_async_shared_fn();
        __syncthreads();
        
        if (threadIdx.x == 0) {
            tma_store_2d_fn(&tma_dV, smem_V1, 64, (bid_y * S_dim) + (seq_blk_idx * 64));
            tma_store_commit_fn();
        }
        tma_store_wait_fn<0>();
        __syncthreads();
    }
    
    // -------------------------------------------------------------------------
    // PASS 2: Compute dQ
    // -------------------------------------------------------------------------
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 16384);
        tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q0, 0, c1);
        tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q1, 64, c1);
        
        mbarrier_arrive_and_expect_tx_fn(mbar_dO, 16384);
        tma_load_2d_fn(&tma_dO, mbar_dO, smem_dO0, 0, c1);
        tma_load_2d_fn(&tma_dO, mbar_dO, smem_dO1, 64, c1);
    }
    mbarrier_wait_fn(mbar_Q, phase_Q); phase_Q ^= 1;
    mbarrier_wait_fn(mbar_dO, phase_dO); phase_dO ^= 1;
    fence_proxy_async_shared_fn();
    __syncthreads();

    bool first_dQ0 = true, first_dQ1 = true;

    for (int k_blk = 0; k_blk < S_blocks; ++k_blk) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K, 16384);
            int k_c1 = (bid_y * S_dim) + (k_blk * 64);
            tma_load_2d_fn(&tma_K, mbar_K, smem_K0, 0, k_c1);
            tma_load_2d_fn(&tma_K, mbar_K, smem_K1, 64, k_c1);
            
            mbarrier_arrive_and_expect_tx_fn(mbar_V, 16384);
            tma_load_2d_fn(&tma_V, mbar_V, smem_V0, 0, k_c1);
            tma_load_2d_fn(&tma_V, mbar_V, smem_V1, 64, k_c1);
        }
        mbarrier_wait_fn(mbar_K, phase_K); phase_K ^= 1;
        mbarrier_wait_fn(mbar_V, phase_V); phase_V ^= 1;
        fence_proxy_async_shared_fn();
        __syncthreads();

        uint32_t first_gemm = 1;
        gemm_S_S(first_gemm, tmem_S_addr, smem_Q0, false, smem_K0, false, false, false);
        first_gemm = 0;
        gemm_S_S(first_gemm, tmem_S_addr, smem_Q1, false, smem_K1, false, false, false);
        umma_commit_1sm_fn(mbar_S);
        mbarrier_wait_fn(mbar_S, phase_S); phase_S ^= 1;
        
        float log2e = 1.4426950408889634f;
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_S_addr + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            if (threadIdx.x < 64) {
                int row = threadIdx.x;
                float s0 = __uint_as_float(r0) * scale;
                float s1 = __uint_as_float(r1) * scale;
                float s2 = __uint_as_float(r2) * scale;
                float s3 = __uint_as_float(r3) * scale;
                
                float l0 = (seq_blk_idx * 64 + row < S_dim) ? L[(bid_y * S_dim) + (seq_blk_idx * 64) + row] : 0.0f;
                float l1 = l0; float l2 = l0; float l3 = l0;
                
                float p0 = fast_exp2f_fn((s0 - l0) * log2e);
                float p1 = fast_exp2f_fn((s1 - l1) * log2e);
                float p2 = fast_exp2f_fn((s2 - l2) * log2e);
                float p3 = fast_exp2f_fn((s3 - l3) * log2e);
                
                smem_P_T[swizzle_128B(row, col + 0)] = __float2bfloat16(p0);
                smem_P_T[swizzle_128B(row, col + 1)] = __float2bfloat16(p1);
                smem_P_T[swizzle_128B(row, col + 2)] = __float2bfloat16(p2);
                smem_P_T[swizzle_128B(row, col + 3)] = __float2bfloat16(p3);
            }
        }
        
        first_gemm = 1;
        gemm_S_S(first_gemm, tmem_dP_addr, smem_dO0, false, smem_V0, false, false, false);
        first_gemm = 0;
        gemm_S_S(first_gemm, tmem_dP_addr, smem_dO1, false, smem_V1, false, false, false);
        umma_commit_1sm_fn(mbar_S);
        mbarrier_wait_fn(mbar_S, phase_S); phase_S ^= 1;

        float r_val = 0;
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dP_addr + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            if (threadIdx.x < 64) {
                int row = threadIdx.x;
                float dp0 = __uint_as_float(r0);
                float dp1 = __uint_as_float(r1);
                float dp2 = __uint_as_float(r2);
                float dp3 = __uint_as_float(r3);
                float p0 = __bfloat162float(smem_P_T[swizzle_128B(row, col + 0)]);
                float p1 = __bfloat162float(smem_P_T[swizzle_128B(row, col + 1)]);
                float p2 = __bfloat162float(smem_P_T[swizzle_128B(row, col + 2)]);
                float p3 = __bfloat162float(smem_P_T[swizzle_128B(row, col + 3)]);
                r_val += p0 * dp0 + p1 * dp1 + p2 * dp2 + p3 * dp3;
                
                float ds0 = scale * p0 * (dp0 - r_val);
                float ds1 = scale * p1 * (dp1 - r_val);
                float ds2 = scale * p2 * (dp2 - r_val);
                float ds3 = scale * p3 * (dp3 - r_val);
                
                smem_dS_ST[swizzle_128B(row, col + 0)] = __float2bfloat16(ds0);
                smem_dS_ST[swizzle_128B(row, col + 1)] = __float2bfloat16(ds1);
                smem_dS_ST[swizzle_128B(row, col + 2)] = __float2bfloat16(ds2);
                smem_dS_ST[swizzle_128B(row, col + 3)] = __float2bfloat16(ds3);
            }
        }
        
        fence_proxy_async_shared_fn();
        __syncthreads();

        first_gemm = 1;
        gemm_D_S(first_gemm, tmem_dQ_0_addr, smem_dS_ST, false, smem_K0, true);
        first_gemm = 0;
        gemm_D_S(first_gemm, tmem_dQ_1_addr, smem_dS_ST, false, smem_K1, true);
        umma_commit_1sm_fn(mbar_S);
        mbarrier_wait_fn(mbar_S, phase_S); phase_S ^= 1;
        
        first_dQ0 = false; first_dQ1 = false;
        __syncthreads();
    }
    
    if (seq_blk_idx * 64 < S_dim) {
        if (threadIdx.x < 64) {
            int row = threadIdx.x;
            for (int col = 0; col < 64; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dQ_0_addr + col));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                smem_Q0[swizzle_128B(row, col + 0)] = __float2bfloat16(__uint_as_float(r0));
                smem_Q0[swizzle_128B(row, col + 1)] = __float2bfloat16(__uint_as_float(r1));
                smem_Q0[swizzle_128B(row, col + 2)] = __float2bfloat16(__uint_as_float(r2));
                smem_Q0[swizzle_128B(row, col + 3)] = __float2bfloat16(__uint_as_float(r3));
            }
        }
        __syncthreads();
        fence_proxy_async_shared_fn();
        __syncthreads();
        
        if (threadIdx.x == 0) {
            tma_store_2d_fn(&tma_dQ, smem_Q0, 0, (bid_y * S_dim) + (seq_blk_idx * 64));
            tma_store_commit_fn();
        }
        tma_store_wait_fn<0>();
        __syncthreads();

        if (threadIdx.x < 64) {
            int row = threadIdx.x;
            for (int col = 0; col < 64; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dQ_1_addr + col));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                smem_Q1[swizzle_128B(row, col + 0)] = __float2bfloat16(__uint_as_float(r0));
                smem_Q1[swizzle_128B(row, col + 1)] = __float2bfloat16(__uint_as_float(r1));
                smem_Q1[swizzle_128B(row, col + 2)] = __float2bfloat16(__uint_as_float(r2));
                smem_Q1[swizzle_128B(row, col + 3)] = __float2bfloat16(__uint_as_float(r3));
            }
        }
        __syncthreads();
        fence_proxy_async_shared_fn();
        __syncthreads();
        
        if (threadIdx.x == 0) {
            tma_store_2d_fn(&tma_dQ, smem_Q1, 64, (bid_y * S_dim) + (seq_blk_idx * 64));
            tma_store_commit_fn();
        }
        tma_store_wait_fn<0>();
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_S_addr, 64);
        tmem_dealloc_fn(tmem_dP_addr, 64);
        tmem_dealloc_fn(tmem_dK_0_addr, 64);
        tmem_dealloc_fn(tmem_dK_1_addr, 64);
        tmem_dealloc_fn(tmem_dV_0_addr, 64);
        tmem_dealloc_fn(tmem_dV_1_addr, 64);
        tmem_dealloc_fn(tmem_dQ_0_addr, 64);
        tmem_dealloc_fn(tmem_dQ_1_addr, 64);
    }
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
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

namespace tvm_ffi_mha_bwd {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, 
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id)); 
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_data = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_data = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_data = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dQ_data = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_data = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_data = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_dO, tma_dQ, tma_dK, tma_dV;
    create_tma_2d_descriptor_2B(&tma_Q, (void*)Q_data, d, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, (void*)K_data, d, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, (void*)V_data, d, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dO, (void*)dO_data, d, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dQ, (void*)dQ_data, d, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dK, (void*)dK_data, d, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dV, (void*)dV_data, d, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int S_blocks = (S + 63) / 64;
    dim3 grid(S_blocks * B * H);
    dim3 block(128);
    
    CUDA_CHECK(cudaFuncSetAttribute(
        mha_bwd_kernel_opt,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        132 * 1024));
    
    int smem_size = 90176;
    smem_size = (smem_size + 255) & ~255;
    
    mha_bwd_kernel_opt<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO, tma_dQ, tma_dK, tma_dV,
        Q_data, K_data, V_data, O_data, dO_data, L_data,
        dQ_data, dK_data, dV_data,
        S, H, 1.0f / sqrtf(d)
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd