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

__device__ __forceinline__ uint64_t make_smem_desc_sm100(uint32_t tmem_ptr, bool is_k_major, bool swizzle) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(&tmem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    if (is_k_major) {
        d |= (uint64_t)((128 * 128 * 2 >> 4) & 0x3FFFF) << 16; 
        d |= (uint64_t)((1024 >> 4) & 0x3FFFF) << 32; 
    } else {
        d |= (uint64_t)((1024 >> 4) & 0x3FFFF) << 16; 
        d |= (uint64_t)((128 * 128 * 2 >> 4) & 0x3FFFF) << 32; 
    }
    d |= (uint64_t)1 << 46;   
    if (swizzle) {
        d |= (uint64_t)2 << 61; 
    }
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_packed_128x128(bool transpose_A, bool transpose_B) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((transpose_A ? 1 : 0) << 15);   
    d |= ((transpose_B ? 1 : 0) << 16);   
    d |= ((128 / 8) << 17);     
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

__device__ __forceinline__ void load_smem_to_tmem(BF16* smem, uint32_t tmem_base) {
    for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
        int row = i / 128;
        int col = i % 128;
        float val = __bfloat162float(smem[row * 128 + col]);
        asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 %0, [%1];"
            :: "r"(*(uint32_t*)&val), "r"(tmem_base + (row << 8) + col));
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void clear_tmem_128x128(uint32_t tmem_base) {
    for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
        int row = i / 128;
        int col = i % 128;
        float val = 0.0f;
        asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 %0, [%1];"
            :: "r"(*(uint32_t*)&val), "r"(tmem_base + (row << 8) + col));
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void load_packed_floats(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0), "=r"(*r1), "=r"(*r2), "=r"(*r3) : "r"(col));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void store_packed_floats(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        :: "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(col));
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ uint32_t pack(float f0, float f1) {
    __nv_bfloat16 b0 = __float2bfloat16(f0);
    __nv_bfloat16 b1 = __float2bfloat16(f1);
    uint32_t res;
    asm("mov.b32 %0, {%1, %2};" : "=r"(res) : "h"(*(uint16_t*)&b0), "h"(*(uint16_t*)&b1));
    return res;
}

__device__ __forceinline__ void store_bf16_row(
    BF16* D, uint32_t tid, uint32_t M, uint32_t N,
    uint32_t m_base, uint32_t n_base, uint32_t BN) {
    uint32_t m_idx = m_base + tid;
    if (m_idx >= M) return;
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        load_packed_floats(col, &r0, &r1, &r2, &r3);
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        uint32_t nc = n_base + col;
        BF16* out = D + (uint64_t)m_idx * N + nc;
        uint2 data = *reinterpret_cast<uint2*>(&pack(f0, f1));
        if (nc + 1 < N) *reinterpret_cast<uint2*>(out) = data;
        data = *reinterpret_cast<uint2*>(&pack(f2, f3));
        if (nc + 3 < N) *reinterpret_cast<uint2*>(out + 2) = data;
    }
}

template<int M, int N, int K>
__device__ __forceinline__ uint32_t gemm_128x128x128_cta_local(uint32_t tmem_A_base, uint32_t tmem_B_base, uint32_t tmem_C_base, float scale) {
    uint32_t idesc = make_instr_desc_packed_128x128(false, false);
    for(int k_chunk = 0; k_chunk < 128; k_chunk += 16) {
        uint32_t tmem_A = tmem_A_base + (k_chunk * 2);
        uint32_t tmem_B = tmem_B_base + (k_chunk * 2);
        uint32_t tmem_C = tmem_C_base;
        
        if (k_chunk == 0) clear_tmem_128x128(tmem_C);
        
        uint64_t desc_A = make_smem_desc_sm100(tmem_A, false, true);
        uint64_t desc_B = make_smem_desc_sm100(tmem_B, false, true);
        
        umma_f16_cg2_fn(tmem_C, desc_A, desc_B, idesc, k_chunk == 0 ? 0 : 1);
    }
    return tmem_C_base;
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
    __grid_constant__ const CUtensorMap tma_O,
    const float* __restrict__ L_all,
    const float* __restrict__ D_all,
    BF16* __restrict__ dQ_all,
    BF16* __restrict__ dK_all,
    BF16* __restrict__ dV_all,
    int64_t S_len)
{
    extern __shared__ char smem_dynamic_buf[];
    char* smem_dynamic_base = (char*)(((uintptr_t)smem_dynamic_buf + 1023) & ~1023);

    BF16* smem_Q = (BF16*)(smem_dynamic_base + 0);
    BF16* smem_K = (BF16*)(smem_dynamic_base + 32768);
    BF16* smem_V = (BF16*)(smem_dynamic_base + 65536);
    BF16* smem_dO = (BF16*)(smem_dynamic_base + 98304);
    BF16* smem_O = (BF16*)(smem_dynamic_base + 131072);
    BF16* smem_P = (BF16*)(smem_dynamic_base + 163840);
    BF16* smem_dS = (BF16*)(smem_dynamic_base + 196608);
    float* smem_D = (float*)(smem_dynamic_base + 229376);
    float* smem_L = (float*)(smem_dynamic_base + 229888);
    uint64_t* mbar_K = (uint64_t*)(smem_dynamic_base + 230400);
    uint64_t* mbar_Q_dO_O = (uint64_t*)(smem_dynamic_base + 230408);
    uint64_t* mbar_V = (uint64_t*)(smem_dynamic_base + 230416);

    if (threadIdx.x == 0) {
        init_barrier(mbar_K, 1);
        init_barrier(mbar_Q_dO_O, 1);
        init_barrier(mbar_V, 1);
    }
    fence_barrier_init();
    __syncthreads();

    __shared__ uint32_t tmem_Q;
    __shared__ uint32_t tmem_K;
    __shared__ uint32_t tmem_V;
    __shared__ uint32_t tmem_O;
    
    if (threadIdx.x == 0) {
        tmem_alloc(&tmem_Q, 128);
        tmem_alloc(&tmem_K, 128);
        tmem_alloc(&tmem_V, 128);
        tmem_alloc(&tmem_O, 128);
    }
    __syncthreads();

    int bh = blockIdx.y;
    int Q_blk = blockIdx.x;
    int num_blocks = (S_len + 127) / 128;
    uint32_t phase_K = 0, phase_Q = 0, phase_V = 0;
    uint32_t scale = 1.0f / sqrtf(128);

    uint16_t mask = 0x3;
    if (cta_id == 0) {
        uint32_t bytes_Q = 128 * 128 * 2;
        mbarrier_expect_tx(mbar_Q_dO_O, bytes_Q * 3);
        uint64_t Q_ptr = (uint64_t)(Q_all + bh * S_len * 128 + Q_blk * 128 * 128);
        uint32_t smem_Q_int = (uint32_t)__cvta_generic_to_shared(smem_Q);
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5, %6;"
            :: "r"(smem_Q_int), "l"((uint64_t)&tma_Q), "r"((uint32_t)__cvta_generic_to_shared(&mbar_Q_dO_O[0])), 
               "r"(0), "r"(bh * S_len + Q_blk * 128), "h"(mask), "l"(0ULL));
        
        uint64_t dO_ptr = (uint64_t)(dO_all + bh * S_len * 128 + Q_blk * 128 * 128);
        uint32_t smem_dO_int = (uint32_t)__cvta_generic_to_shared(smem_dO);
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5, %6;"
            :: "r"(smem_dO_int), "l"((uint64_t)&tma_dO), "r"((uint32_t)__cvta_generic_to_shared(&mbar_Q_dO_O[0])), 
               "r"(0), "r"(bh * S_len + Q_blk * 128), "h"(mask), "l"(0ULL));
               
        uint64_t O_ptr = (uint64_t)(O_all + bh * S_len * 128 + Q_blk * 128 * 128);
        uint32_t smem_O_int = (uint32_t)__cvta_generic_to_shared(smem_O);
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5, %6;"
            :: "r"(smem_O_int), "l"((uint64_t)&tma_O), "r"((uint32_t)__cvta_generic_to_shared(&mbar_Q_dO_O[0])), 
               "r"(0), "r"(bh * S_len + Q_blk * 128), "h"(mask), "l"(0ULL));
    }
    if (threadIdx.x < 128) {
        smem_L[threadIdx.x] = (Q_blk * 128 + threadIdx.x < S_len) ? L_all[bh * S_len + Q_blk * 128 + threadIdx.x] : 0.0f;
        smem_D[threadIdx.x] = (Q_blk * 128 + threadIdx.x < S_len) ? D_all[bh * S_len + Q_blk * 128 + threadIdx.x] : 0.0f;
    }
    mbarrier_wait(mbar_Q_dO_O, phase_Q);
    phase_Q ^= 1;

    load_smem_to_tmem(smem_Q, tmem_Q);
    load_smem_to_tmem(smem_dO, tmem_O); 
    
    // Flip Logic - Let's assume CTA 0 processes even indexed Blocks, CTA 1 processes odd indexed blocks
    int start_kv = cta_id;
    int end_kv = num_blocks;
    int stride_kv = 2;
    
    for (int i = start_kv; i < end_kv; i += stride_kv) {
        if (threadIdx.x == 0) {
            uint32_t bytes_K = 128 * 128 * 2;
            mbarrier_expect_tx(mbar_K, bytes_K);
            uint64_t K_ptr = (uint64_t)(K_all + bh * S_len * 128 + i * 128 * 128);
            uint32_t smem_K_int = (uint32_t)__cvta_generic_to_shared(smem_K);
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" 
                :: "r"(smem_K_int), "l"((uint64_t)&tma_K), "r"((uint32_t)__cvta_generic_to_shared(&mbar_K[0])), "r"(0), "r"(bh * S_len + i * 128));
            
            mbarrier_expect_tx(mbar_V, bytes_K);
            uint64_t V_ptr = (uint64_t)(V_all + bh * S_len * 128 + i * 128 * 128);
            uint32_t smem_V_int = (uint32_t)__cvta_generic_to_shared(smem_V);
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" 
                :: "r"(smem_V_int), "l"((uint64_t)&tma_V), "r"((uint32_t)__cvta_generic_to_shared(&mbar_V[0])), "r"(0), "r"(bh * S_len + i * 128));
        }
        mbarrier_wait(mbar_K, phase_K);
        mbarrier_wait(mbar_V, phase_V);
        phase_K ^= 1;
        phase_V ^= 1;

        load_smem_to_tmem(smem_K, tmem_K);
        load_smem_to_tmem(smem_V, tmem_V);

        uint32_t tmem_S = tmem_S_local; 
        gemm_128x128x128_cta_local(tmem_S, tmem_Q, tmem_K, (i == 0) ? 0.0f : 1.0f);

        int lane_id = threadIdx.x;
        uint32_t r0, r1, r2, r3;
        for (int c = 0; c < 128; c += 4) {
            load_packed_floats(tmem_S + c, &r0, &r1, &r2, &r3);
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            float l_val = smem_L[lane_id];
            float p0 = expf(f0 * scale - l_val);
            float p1 = expf(f1 * scale - l_val);
            float p2 = expf(f2 * scale - l_val);
            float p3 = expf(f3 * scale - l_val);
            
            if (Q_blk * 128 + lane_id >= S_len || i * 128 + c >= S_len) {
                p0 = 0; p1 = 0; p2 = 0; p3 = 0;
            }
            
            smem_P[lane_id * 128 + c] = __float2bfloat16(p0);
            smem_P[lane_id * 128 + c + 1] = __float2bfloat16(p1);
            smem_P[lane_id * 128 + c + 2] = __float2bfloat16(p2);
            smem_P[lane_id * 128 + c + 3] = __float2bfloat16(p3);
        }

        uint32_t tmem_dP = tmem_S; 
        gemm_128x128x128_cta_local(tmem_dP, tmem_V, tmem_O, 0.0f);

        for (int c = 0; c < 128; c += 4) {
            load_packed_floats(tmem_dP + c, &r0, &r1, &r2, &r3);
            float dp0 = __uint_as_float(r0);
            float dp1 = __uint_as_float(r1);
            float dp2 = __uint_as_float(r2);
            float dp3 = __uint_as_float(r3);
            
            float d_val = smem_D[lane_id];
            float ds0 = __bfloat162float(smem_P[lane_id * 128 + c]) * (dp0 - d_val);
            float ds1 = __bfloat162float(smem_P[lane_id * 128 + c + 1]) * (dp1 - d_val);
            float ds2 = __bfloat162float(smem_P[lane_id * 128 + c + 2]) * (dp2 - d_val);
            float ds3 = __bfloat162float(smem_P[lane_id * 128 + c + 3]) * (dp3 - d_val);
            
            if (Q_blk * 128 + lane_id >= S_len || i * 128 + c >= S_len) {
                ds0 = 0; ds1 = 0; ds2 = 0; ds3 = 0;
            }
            
            smem_dS[lane_id * 128 + c] = __float2bfloat16(ds0);
            smem_dS[lane_id * 128 + c + 1] = __float2bfloat16(ds1);
            smem_dS[lane_id * 128 + c + 2] = __float2bfloat16(ds2);
            smem_dS[lane_id * 128 + c + 3] = __float2bfloat16(ds3);
        }

        __syncthreads();
        for (int idx = threadIdx.x; idx < 128 * 128; idx += blockDim.x) {
            int row = idx / 128;
            int col = idx % 128;
            smem_O[col + row * 128] = smem_K[row * 128 + col];
        }
        __syncthreads();

        uint32_t tmem_dQ = tmem_D;
        gemm_128x128x128_cta_local(tmem_dQ, smem_dS, smem_O, 1.0f);
        
        __syncthreads();
        for (int idx = threadIdx.x; idx < 128 * 128; idx += blockDim.x) {
            int row = idx / 128;
            int col = idx % 128;
            smem_K[col + row * 128] = smem_P[row * 128 + col];
        }
        __syncthreads();
        
        uint32_t tmem_dK = tmem_K; 
        gemm_128x128x128_cta_local(tmem_dK, smem_K, smem_dO, 0.0f);
        
        store_bf16_row(dK_all + bh * S_len * 128, threadIdx.x, S_len, 128, i * 128, 0, 128);
        store_bf16_row(dV_all + bh * S_len * 128, threadIdx.x, S_len, 128, i * 128, 0, 128);
    }

    store_bf16_row(dQ_all + bh * S_len * 128, threadIdx.x, S_len, 128, Q_blk * 128, 0, 128);
}