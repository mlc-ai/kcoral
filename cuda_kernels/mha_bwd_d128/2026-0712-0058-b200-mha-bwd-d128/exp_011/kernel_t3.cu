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
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

using BF16 = __nv_bfloat16;

__device__ __forceinline__ void init_barrier(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void tmem_alloc(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
       :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
       :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100(BF16* smem_ptr, bool is_k_major, bool swizzle) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    
    uint32_t lbo, sbo;
    if (is_k_major) {
        lbo = 1024; 
        sbo = 1024; 
    } else {
        lbo = 8192; 
        sbo = 1024; 
    }
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;   
    
    d |= (uint64_t)1 << 46;   
    if (swizzle) {
        d |= (uint64_t)2 << 61; 
    }
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_packed_64x64(bool transpose_A, bool transpose_B) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((transpose_A ? 1 : 0) << 15);   
    d |= ((transpose_B ? 1 : 0) << 16);   
    d |= ((64 / 8) << 17);     
    d |= ((128 / 16) << 24);    
    return d;
}

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar, uint16_t ctaMask) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"(ctaMask));
}

__device__ __forceinline__ void load_smem_to_tmem(BF16* smem, uint32_t tmem_base) {
    int row = threadIdx.x;
    if (row < 64) {
        for (int col = 0; col < 64; col++) {
            float val = __bfloat162float(smem[row * 64 + col]);
            uint32_t addr = (tmem_base & 0xFFFF) + col;
            asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 %0, [%1];"
                :: "r"(*(uint32_t*)&val), "r"(addr));
        }
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void clear_tmem(uint32_t tmem_base) {
    int row = threadIdx.x;
    if (row < 64) {
        for (int col = 0; col < 64; col++) {
            float val = 0.0f;
            uint32_t addr = (tmem_base & 0xFFFF) + col;
            asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 %0, [%1];"
                :: "r"(*(uint32_t*)&val), "r"(addr));
        }
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ BF16* get_swizzled_ptr(BF16* smem, int row, int col) {
    int chunk = col / 8;
    int offset = col % 8;
    int swizzled_chunk = (row % 8) ^ chunk;
    int phys_col = swizzled_chunk * 8 + offset;
    return &smem[row * 64 + phys_col];
}

__device__ __forceinline__ void write_swizzled(BF16* smem, int row, int col, float val) {
    int chunk = col / 8;
    int offset = col % 8;
    int swizzled_chunk = (row % 8) ^ chunk;
    int phys_col = swizzled_chunk * 8 + offset;
    smem[row * 64 + phys_col] = __float2bfloat16(val);
}

template<int M, int N, int K>
__device__ __forceinline__ uint32_t gemm_64x64x64(uint32_t tmem_C, BF16* A, BF16* B, bool trans_A, bool trans_B) {
    uint32_t idesc = make_instr_desc_packed_64x64(trans_A, trans_B);
    for(int k = 0; k < 64; k += 16) {
        uint64_t desc_A = make_smem_desc_sm100(A + k, trans_A ? false : true, true);
        uint64_t desc_B = make_smem_desc_sm100(B + k, trans_B ? false : true, true);
        
        if (k == 0) clear_tmem(tmem_C);
        
        uint32_t tmem_c = tmem_C + (k * 2);
        umma_f16_cg2_fn(tmem_c, desc_A, desc_B, idesc, k == 0 ? 0 : 1);
    }
    return tmem_C;
}

// ----------------------------------------------------------------------
__global__ void compute_D_kernel(
    const BF16* __restrict__ dO,
    const BF16* __restrict__ O,
    float* __restrict__ D,
    int64_t num_rows)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        float sum = 0;
        int64_t offset = idx * 128;
        for (int i = 0; i < 128; ++i) {
            sum += __bfloat162float(dO[offset + i]) * __bfloat162float(O[offset + i]);
        }
        D[idx] = sum;
    }
}

// ----------------------------------------------------------------------
__global__ void bwd_d128_kernel(
    __grid_constant__ const CUtensorMap tma_Q,
    __grid_constant__ const CUtensorMap tma_K,
    __grid_constant__ const CUtensorMap tma_V,
    __grid_constant__ const CUtensorMap tma_dO,
    const float* __restrict__ L_all,
    const float* __restrict__ D_all,
    BF16* __restrict__ dQ_all,
    BF16* __restrict__ dK_all,
    BF16* __restrict__ dV_all,
    int64_t S_len)
{
    extern __shared__ char smem_dynamic_buf[];
    char* smem_dynamic_base = (char*)(((uintptr_t)smem_dynamic_buf + 1023) & ~1023);

    BF16* smem_Q0 = (BF16*)(smem_dynamic_base + 0);
    BF16* smem_Q1 = (BF16*)(smem_dynamic_base + 8192);
    BF16* smem_K0 = (BF16*)(smem_dynamic_base + 16384);
    BF16* smem_K1 = (BF16*)(smem_dynamic_base + 24576);
    BF16* smem_V0 = (BF16*)(smem_dynamic_base + 32768);
    BF16* smem_V1 = (BF16*)(smem_dynamic_base + 40960);
    BF16* smem_dO0 = (BF16*)(smem_dynamic_base + 49152);
    BF16* smem_dO1 = (BF16*)(smem_dynamic_base + 57344);
    BF16* smem_P = (BF16*)(smem_dynamic_base + 65536);
    BF16* smem_dS = (BF16*)(smem_dynamic_base + 73728);
    float* smem_D = (float*)(smem_dynamic_base + 81920);
    float* smem_L = (float*)(smem_dynamic_base + 82432);
    uint64_t* mbar_K = (uint64_t*)(smem_dynamic_base + 82944);
    uint64_t* mbar_Q_dO = (uint64_t*)(smem_dynamic_base + 82952);
    uint64_t* mbar_V = (uint64_t*)(smem_dynamic_base + 82960);

    if (threadIdx.x == 0) {
        init_barrier(mbar_K, 1);
        init_barrier(mbar_Q_dO, 1);
        init_barrier(mbar_V, 1);
    }
    fence_barrier_init();
    __syncthreads();

    __shared__ uint32_t tmem_C;
    
    if (threadIdx.x == 0) {
        tmem_alloc(&tmem_C, 128);
    }
    __syncthreads();

    int bh = blockIdx.y;
    int Q_blk = blockIdx.x;
    int cta_id = blockIdx.z;
    int num_blocks = (S_len + 63) / 64;
    uint32_t phase_K = 0, phase_Q = 0, phase_V = 0;
    uint32_t scale = 1.0f / sqrtf(128);

    int my_q_blk = Q_blk * 64 + cta_id * 64;
    uint32_t bytes_Q = 64 * 64 * 2;
    
    if (threadIdx.x == 0) {
        mbarrier_expect_tx(mbar_Q_dO, bytes_Q * 4);
        uint32_t smem_Q0_int = (uint32_t)__cvta_generic_to_shared(smem_Q0);
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
            :: "r"(smem_Q0_int), "l"((uint64_t)&tma_Q), "r"((uint32_t)__cvta_generic_to_shared(&mbar_Q_dO[0])), "r"(0), "r"(bh * S_len + my_q_blk));
        
        uint32_t smem_Q1_int = (uint32_t)__cvta_generic_to_shared(smem_Q1);
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
            :: "r"(smem_Q1_int), "l"((uint64_t)&tma_Q), "r"((uint32_t)__cvta_generic_to_shared(&mbar_Q_dO[0])), "r"(64), "r"(bh * S_len + my_q_blk));
               
        uint32_t smem_dO0_int = (uint32_t)__cvta_generic_to_shared(smem_dO0);
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
            :: "r"(smem_dO0_int), "l"((uint64_t)&tma_dO), "r"((uint32_t)__cvta_generic_to_shared(&mbar_Q_dO[0])), "r"(0), "r"(bh * S_len + my_q_blk));
               
        uint32_t smem_dO1_int = (uint32_t)__cvta_generic_to_shared(smem_dO1);
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
            :: "r"(smem_dO1_int), "l"((uint64_t)&tma_dO), "r"((uint32_t)__cvta_generic_to_shared(&mbar_Q_dO[0])), "r"(64), "r"(bh * S_len + my_q_blk));
    }
    
    if (threadIdx.x < 64) {
        smem_L[threadIdx.x] = (my_q_blk * 64 + threadIdx.x < S_len) ? L_all[bh * S_len + my_q_blk * 64 + threadIdx.x] : 0.0f;
        smem_D[threadIdx.x] = (my_q_blk * 64 + threadIdx.x < S_len) ? D_all[bh * S_len + my_q_blk * 64 + threadIdx.x] : 0.0f;
    }
    
    mbarrier_wait(mbar_Q_dO, phase_Q);
    phase_Q ^= 1;

    uint32_t tmem_dQ0 = tmem_C;
    uint32_t tmem_dQ1 = tmem_C + 64;
    clear_tmem(tmem_dQ0);
    clear_tmem(tmem_dQ1);

    uint32_t tmem_S = tmem_C; 

    for (int i = 0; i < num_blocks; ++i) {
        if (threadIdx.x == 0) {
            mbarrier_expect_tx(mbar_K, bytes_Q * 2);
            uint32_t smem_K0_int = (uint32_t)__cvta_generic_to_shared(smem_K0);
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" 
                :: "r"(smem_K0_int), "l"((uint64_t)&tma_K), "r"((uint32_t)__cvta_generic_to_shared(&mbar_K[0])), "r"(0), "r"(bh * S_len + i * 64));
            
            uint32_t smem_K1_int = (uint32_t)__cvta_generic_to_shared(smem_K1);
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" 
                :: "r"(smem_K1_int), "l"((uint64_t)&tma_K), "r"((uint32_t)__cvta_generic_to_shared(&mbar_K[0])), "r"(64), "r"(bh * S_len + i * 64));
            
            mbarrier_expect_tx(mbar_V, bytes_Q * 2);
            uint32_t smem_V0_int = (uint32_t)__cvta_generic_to_shared(smem_V0);
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" 
                :: "r"(smem_V0_int), "l"((uint64_t)&tma_V), "r"((uint32_t)__cvta_generic_to_shared(&mbar_V[0])), "r"(0), "r"(bh * S_len + i * 64));
            
            uint32_t smem_V1_int = (uint32_t)__cvta_generic_to_shared(smem_V1);
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" 
                :: "r"(smem_V1_int), "l"((uint64_t)&tma_V), "r"((uint32_t)__cvta_generic_to_shared(&mbar_V[0])), "r"(64), "r"(bh * S_len + i * 64));
        }
        mbarrier_wait(mbar_K, phase_K);
        mbarrier_wait(mbar_V, phase_V);
        phase_K ^= 1;
        phase_V ^= 1;

        uint32_t r0, r1, r2, r3;
        for (int c = 0; c < 64; c += 4) {
            uint32_t col_addr = (tmem_S & 0xFFFF) + c;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col_addr));
            int row = threadIdx.x;
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            float l_val = (row < 64) ? smem_L[row] : 0.0f;
            float p0 = expf(f0 * scale - l_val);
            float p1 = expf(f1 * scale - l_val);
            float p2 = expf(f2 * scale - l_val);
            float p3 = expf(f3 * scale - l_val);
            
            if (my_q_blk * 64 + row >= S_len || i * 64 + c >= S_len || row >= 64) {
                p0 = 0; p1 = 0; p2 = 0; p3 = 0;
            }
            
            write_swizzled(smem_P, row, c, p0);
            write_swizzled(smem_P, row, c + 1, p1);
            write_swizzled(smem_P, row, c + 2, p2);
            write_swizzled(smem_P, row, c + 3, p3);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t tmem_dP = tmem_S; 
        gemm_64x64x64<64, 64, 64>(tmem_dP, smem_V0, smem_dO0, false, true);

        for (int c = 0; c < 64; c += 4) {
            uint32_t col_addr = (tmem_dP & 0xFFFF) + c;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col_addr));
            int row = threadIdx.x;
            float dp0 = __uint_as_float(r0);
            float dp1 = __uint_as_float(r1);
            float dp2 = __uint_as_float(r2);
            float dp3 = __uint_as_float(r3);
            
            float d_val = (row < 64) ? smem_D[row] : 0.0f;
            float ds0 = __bfloat162float(*get_swizzled_ptr(smem_P, row, c)) * (dp0 - d_val);
            float ds1 = __bfloat162float(*get_swizzled_ptr(smem_P, row, c + 1)) * (dp1 - d_val);
            float ds2 = __bfloat162float(*get_swizzled_ptr(smem_P, row, c + 2)) * (dp2 - d_val);
            float ds3 = __bfloat162float(*get_swizzled_ptr(smem_P, row, c + 3)) * (dp3 - d_val);
            
            if (my_q_blk * 64 + row >= S_len || i * 64 + c >= S_len || row >= 64) {
                ds0 = 0; ds1 = 0; ds2 = 0; ds3 = 0;
            }
            
            write_swizzled(smem_dS, row, c, ds0);
            write_swizzled(smem_dS, row, c + 1, ds1);
            write_swizzled(smem_dS, row, c + 2, ds2);
            write_swizzled(smem_dS, row, c + 3, ds3);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

        uint32_t tmem_dQ0_curr = tmem_dQ0;
        gemm_64x64x64<64, 64, 64>(tmem_dQ0_curr, smem_dS, smem_K0, false, true);

        uint32_t tmem_dQ1_curr = tmem_dQ1;
        gemm_64x64x64<64, 64, 64>(tmem_dQ1_curr, smem_dS, smem_K1, false, true);

        __syncthreads();
        for (int idx = threadIdx.x; idx < 64 * 64; idx += blockDim.x) {
            int row = idx / 64;
            int col = idx % 64;
            smem_K0[col + row * 64] = *get_swizzled_ptr(smem_dS, row, col);
        }
        __syncthreads();
        
        uint32_t tmem_dK = tmem_S; 
        gemm_64x64x64<64, 64, 64>(tmem_dK, smem_K0, smem_Q0, true, true);
        
        for (int c = 0; c < 64; c += 4) {
            uint32_t col_addr = (tmem_dK & 0xFFFF) + c;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col_addr));
            int row = threadIdx.x;
            if (row < 64 && i * 64 + row < S_len) {
                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                __nv_bfloat16 b0 = __float2bfloat16(f0);
                __nv_bfloat16 b1 = __float2bfloat16(f1);
                __nv_bfloat162 val = *(const __nv_bfloat162*)(&b0);
                atomicAdd((__nv_bfloat162*)(dK_all + bh * S_len * 128 + i * 64 * 128 + row * 128 + c), val);
                
                __nv_bfloat162 val1 = *(const __nv_bfloat162*)(&b1);
                atomicAdd((__nv_bfloat162*)(dK_all + bh * S_len * 128 + i * 64 * 128 + row * 128 + c + 1), val1);
            }
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        __syncthreads();
        for (int idx = threadIdx.x; idx < 64 * 64; idx += blockDim.x) {
            int row = idx / 64;
            int col = idx % 64;
            smem_K1[col + row * 64] = *get_swizzled_ptr(smem_P, row, col);
        }
        __syncthreads();
        
        uint32_t tmem_dV = tmem_S; 
        gemm_64x64x64<64, 64, 64>(tmem_dV, smem_K1, smem_dO0, true, true);

        for (int c = 0; c < 64; c += 4) {
            uint32_t col_addr = (tmem_dV & 0xFFFF) + c;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col_addr));
            int row = threadIdx.x;
            if (row < 64 && i * 64 + row < S_len) {
                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                __nv_bfloat16 b0 = __float2bfloat16(f0);
                __nv_bfloat16 b1 = __float2bfloat16(f1);
                __nv_bfloat162 val = *(const __nv_bfloat162*)(&b0);
                atomicAdd((__nv_bfloat162*)(dV_all + bh * S_len * 128 + i * 64 * 128 + row * 128 + c), val);
                
                __nv_bfloat162 val1 = *(const __nv_bfloat162*)(&b1);
                atomicAdd((__nv_bfloat162*)(dV_all + bh * S_len * 128 + i * 64 * 128 + row * 128 + c + 1), val1);
            }
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    }
    
    for (int c = 0; c < 64; c += 4) {
        uint32_t col_addr0 = (tmem_dQ0 & 0xFFFF) + c;
        uint32_t col_addr1 = (tmem_dQ1 & 0xFFFF) + c;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col_addr0));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col_addr1));
        int row = threadIdx.x;
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        if (my_q_blk * 64 + row < S_len) {
            BF16* dQ0_ptr = dQ_all + bh * S_len * 128 + (my_q_blk * 64 + row) * 128;
            dQ0_ptr[c] = __float2bfloat16(f0);
            dQ0_ptr[c + 1] = __float2bfloat16(f1);
            dQ0_ptr[c + 2] = __float2bfloat16(f2);
            dQ0_ptr[c + 3] = __float2bfloat16(f3);
            
            BF16* dQ1_ptr = dQ_all + bh * S_len * 128 + (my_q_blk * 64 + row) * 128 + 64;
            dQ1_ptr[c] = __float2bfloat16(f0);
            dQ1_ptr[c + 1] = __float2bfloat16(f1);
            dQ1_ptr[c + 2] = __float2bfloat16(f2);
            dQ1_ptr[c + 3] = __float2bfloat16(f3);
        }
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

    tmem_dealloc(tmem_C, 128);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3); 
    
    if (d != 128) {
        fprintf(stderr, "Expected d=128, got %ld\n", d);
        exit(1);
    }
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int64_t num_rows = B * H * S;
    float* D_all;
    CUDA_CHECK(cudaMallocAsync(&D_all, num_rows * sizeof(float), stream));
    
    int threads_D = 256;
    int blocks_D = (num_rows + threads_D - 1) / threads_D;
    compute_D_kernel<<<blocks_D, threads_D, 0, stream>>>(
        static_cast<const BF16*>(dO.data_ptr()), 
        static_cast<const BF16*>(O.data_ptr()), 
        D_all, num_rows);
    CUDA_CHECK(cudaGetLastError());

    CUtensorMap tma_Q, tma_K, tma_V, tma_dO, tma_O;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), d, B * H * S, 64, 64, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), d, B * H * S, 64, 64, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), d, B * H * S, 64, 64, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_dO, dO.data_ptr(), d, B * H * S, 64, 64, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_O, O.data_ptr(), d, B * H * S, 64, 64, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    dim3 grid((S + 63) / 64, B * H, 4);
    dim3 block(128);
    
    int smem_size = 82960 + 8; 
    CUDA_CHECK(cudaFuncSetAttribute(bwd_d128_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 4;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, bwd_d128_kernel, tma_Q, tma_K, tma_V, tma_dO, L.data_ptr(), D_all, 
        static_cast<BF16*>(dQ.data_ptr()), static_cast<BF16*>(dK.data_ptr()), static_cast<BF16*>(dV.data_ptr()), S));
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaFreeAsync(D_all, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 2, globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);