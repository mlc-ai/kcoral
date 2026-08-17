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
        return;                                                    \
    }                                                              \
} while(0)

// Helper for SM100 3D TMA descriptor setup
static CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, uint32_t box2, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

// SM100 Kernel inline helpers
__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}
__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
}
__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}
__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile("{\n.reg .pred P;\nWAIT_%=:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra WAIT_%=;\n}\n" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}
__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}
__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};" : "=r"(result) : "h"(*reinterpret_cast<uint16_t*>(&a)), "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}
__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
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
__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3, uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),"=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}
__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ uint64_t make_smem_desc_sm100_none_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 61;   // SWIZZLE_NONE
    return d;
}
__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4) | (1u << 7) | (1u << 10) | (0u << 15) | (0u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}
__device__ __forceinline__ uint64_t advance_smem_desc(uint64_t desc, uint32_t byte_offset) {
    uint32_t addr_16b = desc & 0x3FFF;
    uint32_t addr = addr_16b << 4;
    addr += byte_offset;
    desc &= ~0x3FFFull;
    desc |= (addr >> 4) & 0x3FFF;
    uint32_t base_offset = (addr >> 7) & 0x7;
    desc &= ~(0x7ull << 49);
    desc |= ((uint64_t)base_offset << 49);
    return desc;
}

__device__ __forceinline__ uint64_t make_smem_desc_kmaj(void* smem_ptr) { return make_smem_desc_sm100_none_fn(smem_ptr, 2048, 128); }
__device__ __forceinline__ uint64_t make_smem_desc_mnmaj(void* smem_ptr) { return make_smem_desc_sm100_none_fn(smem_ptr, 128, 2048); }
__device__ __forceinline__ uint64_t advance_desc_kmaj(uint64_t desc) { return advance_smem_desc(desc, 32); }
__device__ __forceinline__ uint64_t advance_desc_mnmaj(uint64_t desc) { return advance_smem_desc(desc, 4096); }

__device__ __forceinline__ void compute_umma_128(uint32_t tmem_D, uint64_t desc_A, uint64_t desc_B, uint32_t idesc, bool a_mnmaj, bool b_mnmaj, bool accum_base) {
    for(int step=0; step<8; ++step) {
        int accum = accum_base || (step > 0);
        asm volatile(
            "{\n.reg .pred p;\n"
            "setp.ne.b32 p, %4, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
            :: "r"(tmem_D), "l"(desc_A), "l"(desc_B), "r"(idesc), "r"(accum));
        desc_A = a_mnmaj ? advance_desc_mnmaj(desc_A) : advance_desc_kmaj(desc_A);
        desc_B = b_mnmaj ? advance_desc_mnmaj(desc_B) : advance_desc_kmaj(desc_B);
    }
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ __nv_bfloat16 read_linear(const __nv_bfloat16* smem, int row, int col) {
    return smem[row * 128 + col];
}

__device__ __forceinline__ void store_linear_16B(uint32_t smem_base_addr, int row, int col_16B, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    uint32_t addr = smem_base_addr + row * 256 + col_16B * 16;
    st_shared_128_fn(addr, v0, v1, v2, v3);
}

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_base_fn(
    uint32_t tmem_base, __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t S, uint32_t block_i) {
    
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_base + col));
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
    for (uint32_t step = 0; step < 32; ++step) {
        uint32_t row = step * 4 + warp_id;
        uint32_t global_row = block_i * 128 + row;
        uint32_t col_start = lane_id * 4;
        if (global_row < S) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * 128 + col_start]);
            *reinterpret_cast<uint2*>(D + (uint64_t)global_row * 128 + col_start) = data;
        }
    }
    __syncthreads();
}

union SharedStorage {
    struct {
        __nv_bfloat16 Q[128*128];
        __nv_bfloat16 dO[128*128];
        __nv_bfloat16 O[128*128];
        __nv_bfloat16 K[128*128];
        __nv_bfloat16 V[128*128];
        __nv_bfloat16 dS[128*128];
        float D[128];
        float L[128];
    } pass1;
    struct {
        __nv_bfloat16 K[128*128];
        __nv_bfloat16 V[128*128];
        __nv_bfloat16 Q[128*128];
        __nv_bfloat16 dO[128*128];
        __nv_bfloat16 O[128*128];
        __nv_bfloat16 dS[128*128];
        __nv_bfloat16 P[128*128];
        float D[128];
        float L[128];
    } pass2;
};

__global__ void mha_bwd_d128_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L_ptr,
    __nv_bfloat16* dQ_ptr,
    __nv_bfloat16* dK_ptr,
    __nv_bfloat16* dV_ptr,
    int S
) {
    int bh_idx = blockIdx.x;
    extern __shared__ __align__(128) uint8_t smem_pool[];
    SharedStorage* smem = (SharedStorage*)smem_pool;
    
    __shared__ uint64_t mbar;
    __shared__ uint64_t mbar_umma;
    __shared__ uint32_t tmem_base_addr;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar, 128);
        init_smem_barrier_fn(&mbar_umma, 1);
        fence_smem_barrier_init_fn();
    }
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&tmem_base_addr, 512);
    }
    __syncthreads();
    
    uint32_t tmem_base = tmem_base_addr;
    int phase = 0;
    int phase_umma = 0;
    float scale_log2e = (1.0f / 11.3137085f) * 1.44269504f;
    
    uint32_t idesc_k_k = make_instr_desc_fn(128, 128);
    uint32_t idesc_k_n = idesc_k_k | (1u << 16);
    uint32_t idesc_m_n = idesc_k_k | (1u << 15) | (1u << 16);
    
    // ------------- PASS 1: Compute dQ -------------
    uint32_t tmem_dQ = tmem_base + 0;
    uint32_t tmem_S = tmem_base + 128;
    uint32_t tmem_dP = tmem_base + 256;
    
    for (int q_j = 0; q_j < S; q_j += 128) {
        if (threadIdx.x == 0) {
            tma_load_3d_fn(&tma_Q, &mbar, smem->pass1.Q, 0, q_j, bh_idx);
            tma_load_3d_fn(&tma_dO, &mbar, smem->pass1.dO, 0, q_j, bh_idx);
            tma_load_3d_fn(&tma_O, &mbar, smem->pass1.O, 0, q_j, bh_idx);
            mbarrier_arrive_and_expect_tx_fn(&mbar, 3 * 32768);
        } else { mbarrier_arrive_fn(&mbar); }
        mbarrier_wait_fn(&mbar, phase); phase ^= 1;
        
        int row = threadIdx.x;
        float d_val = 0;
        if (q_j + row < S) {
            smem->pass1.L[row] = L_ptr[bh_idx * S + q_j + row];
            for(int i = 0; i < 128; ++i) {
                d_val += __bfloat162float(read_linear(smem->pass1.dO, row, i)) * __bfloat162float(read_linear(smem->pass1.O, row, i));
            }
        }
        smem->pass1.D[row] = d_val;
        __syncthreads();
        
        for (int k_i = 0; k_i < S; k_i += 128) {
            if (threadIdx.x == 0) {
                tma_load_3d_fn(&tma_K, &mbar, smem->pass1.K, 0, k_i, bh_idx);
                tma_load_3d_fn(&tma_V, &mbar, smem->pass1.V, 0, k_i, bh_idx);
                mbarrier_arrive_and_expect_tx_fn(&mbar, 2 * 32768);
            } else { mbarrier_arrive_fn(&mbar); }
            mbarrier_wait_fn(&mbar, phase); phase ^= 1;
            
            if (threadIdx.x == 0) {
                uint64_t desc_Q_k = make_smem_desc_kmaj(smem->pass1.Q);
                uint64_t desc_K_k = make_smem_desc_kmaj(smem->pass1.K);
                compute_umma_128(tmem_S, desc_Q_k, desc_K_k, idesc_k_k, false, false, false);
                
                uint64_t desc_dO_k = make_smem_desc_kmaj(smem->pass1.dO);
                uint64_t desc_V_k = make_smem_desc_kmaj(smem->pass1.V);
                compute_umma_128(tmem_dP, desc_dO_k, desc_V_k, idesc_k_k, false, false, false);
                
                umma_commit_cg1_fn(&mbar_umma);
            }
            mbarrier_wait_fn(&mbar_umma, phase_umma); phase_umma ^= 1;
            
            float l_val = smem->pass1.L[row];
            for (int col_16B = 0; col_16B < 16; ++col_16B) {
                uint32_t s[8], dp[8];
                tmem_load_8x_fn(tmem_S + col_16B * 8, &s[0], &s[1], &s[2], &s[3], &s[4], &s[5], &s[6], &s[7]);
                tmem_load_8x_fn(tmem_dP + col_16B * 8, &dp[0], &dp[1], &dp[2], &dp[3], &dp[4], &dp[5], &dp[6], &dp[7]);
                tmem_load_fence_fn();
                
                uint32_t ds_packed[4];
                for(int i=0; i<4; ++i) {
                    int col0 = col_16B * 8 + i * 2;
                    int col1 = col0 + 1;
                    bool valid0 = (q_j + row < S) && (k_i + col0 < S);
                    bool valid1 = (q_j + row < S) && (k_i + col1 < S);
                    
                    float s0 = __uint_as_float(s[i*2]);
                    float s1 = __uint_as_float(s[i*2+1]);
                    float dp0 = __uint_as_float(dp[i*2]);
                    float dp1 = __uint_as_float(dp[i*2+1]);
                    
                    float p0 = valid0 ? fast_exp2f_fn(s0 * scale_log2e - l_val * 1.44269504f) : 0.0f;
                    float p1 = valid1 ? fast_exp2f_fn(s1 * scale_log2e - l_val * 1.44269504f) : 0.0f;
                    
                    float ds0 = p0 * (dp0 - d_val);
                    float ds1 = p1 * (dp1 - d_val);
                    ds_packed[i] = pack_bf16_fn(__float_as_uint(ds0), __float_as_uint(ds1));
                }
                store_linear_16B((uint32_t)__cvta_generic_to_shared(smem->pass1.dS), row, col_16B, ds_packed[0], ds_packed[1], ds_packed[2], ds_packed[3]);
            }
            fence_async_shared_fn();
            __syncthreads();
            
            if (threadIdx.x == 0) {
                uint64_t desc_dS_k = make_smem_desc_kmaj(smem->pass1.dS);
                uint64_t desc_K_mn = make_smem_desc_mnmaj(smem->pass1.K);
                compute_umma_128(tmem_dQ, desc_dS_k, desc_K_mn, idesc_k_n, false, true, (k_i > 0));
                umma_commit_cg1_fn(&mbar_umma);
            }
            mbarrier_wait_fn(&mbar_umma, phase_umma); phase_umma ^= 1;
            __syncthreads();
        }
        tmem_epilogue_coalesced_4w_base_fn(tmem_dQ, dQ_ptr + (uint64_t)bh_idx * S * 128, smem->pass1.Q, S, q_j / 128);
    }
    
    // ------------- PASS 2: Compute dK and dV -------------
    uint32_t tmem_dK = tmem_base + 0;
    uint32_t tmem_dV = tmem_base + 128;
    uint32_t tmem_S2 = tmem_base + 256;
    uint32_t tmem_dP2 = tmem_base + 384;
    
    for (int k_i = 0; k_i < S; k_i += 128) {
        if (threadIdx.x == 0) {
            tma_load_3d_fn(&tma_K, &mbar, smem->pass2.K, 0, k_i, bh_idx);
            tma_load_3d_fn(&tma_V, &mbar, smem->pass2.V, 0, k_i, bh_idx);
            mbarrier_arrive_and_expect_tx_fn(&mbar, 2 * 32768);
        } else { mbarrier_arrive_fn(&mbar); }
        mbarrier_wait_fn(&mbar, phase); phase ^= 1;
        
        for (int q_j = 0; q_j < S; q_j += 128) {
            if (threadIdx.x == 0) {
                tma_load_3d_fn(&tma_Q, &mbar, smem->pass2.Q, 0, q_j, bh_idx);
                tma_load_3d_fn(&tma_dO, &mbar, smem->pass2.dO, 0, q_j, bh_idx);
                tma_load_3d_fn(&tma_O, &mbar, smem->pass2.O, 0, q_j, bh_idx);
                mbarrier_arrive_and_expect_tx_fn(&mbar, 3 * 32768);
            } else { mbarrier_arrive_fn(&mbar); }
            mbarrier_wait_fn(&mbar, phase); phase ^= 1;
            
            int row = threadIdx.x;
            float d_val = 0;
            if (q_j + row < S) {
                smem->pass2.L[row] = L_ptr[bh_idx * S + q_j + row];
                for(int i=0; i<128; ++i) d_val += __bfloat162float(read_linear(smem->pass2.dO, row, i)) * __bfloat162float(read_linear(smem->pass2.O, row, i));
            }
            smem->pass2.D[row] = d_val;
            __syncthreads();
            
            if (threadIdx.x == 0) {
                uint64_t desc_Q_k = make_smem_desc_kmaj(smem->pass2.Q);
                uint64_t desc_K_k = make_smem_desc_kmaj(smem->pass2.K);
                compute_umma_128(tmem_S2, desc_Q_k, desc_K_k, idesc_k_k, false, false, false);
                
                uint64_t desc_dO_k = make_smem_desc_kmaj(smem->pass2.dO);
                uint64_t desc_V_k = make_smem_desc_kmaj(smem->pass2.V);
                compute_umma_128(tmem_dP2, desc_dO_k, desc_V_k, idesc_k_k, false, false, false);
                
                umma_commit_cg1_fn(&mbar_umma);
            }
            mbarrier_wait_fn(&mbar_umma, phase_umma); phase_umma ^= 1;
            
            float l_val = smem->pass2.L[row];
            for (int col_16B = 0; col_16B < 16; ++col_16B) {
                uint32_t s[8], dp[8];
                tmem_load_8x_fn(tmem_S2 + col_16B * 8, &s[0], &s[1], &s[2], &s[3], &s[4], &s[5], &s[6], &s[7]);
                tmem_load_8x_fn(tmem_dP2 + col_16B * 8, &dp[0], &dp[1], &dp[2], &dp[3], &dp[4], &dp[5], &dp[6], &dp[7]);
                tmem_load_fence_fn();
                
                uint32_t ds_packed[4], p_packed[4];
                for(int i=0; i<4; ++i) {
                    int col0 = col_16B * 8 + i * 2;
                    int col1 = col0 + 1;
                    bool valid0 = (q_j + row < S) && (k_i + col0 < S);
                    bool valid1 = (q_j + row < S) && (k_i + col1 < S);
                    
                    float s0 = __uint_as_float(s[i*2]);
                    float s1 = __uint_as_float(s[i*2+1]);
                    float dp0 = __uint_as_float(dp[i*2]);
                    float dp1 = __uint_as_float(dp[i*2+1]);
                    
                    float p0 = valid0 ? fast_exp2f_fn(s0 * scale_log2e - l_val * 1.44269504f) : 0.0f;
                    float p1 = valid1 ? fast_exp2f_fn(s1 * scale_log2e - l_val * 1.44269504f) : 0.0f;
                    
                    float ds0 = p0 * (dp0 - d_val);
                    float ds1 = p1 * (dp1 - d_val);
                    ds_packed[i] = pack_bf16_fn(__float_as_uint(ds0), __float_as_uint(ds1));
                    p_packed[i] = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
                }
                store_linear_16B((uint32_t)__cvta_generic_to_shared(smem->pass2.dS), row, col_16B, ds_packed[0], ds_packed[1], ds_packed[2], ds_packed[3]);
                store_linear_16B((uint32_t)__cvta_generic_to_shared(smem->pass2.P), row, col_16B, p_packed[0], p_packed[1], p_packed[2], p_packed[3]);
            }
            fence_async_shared_fn();
            __syncthreads();
            
            if (threadIdx.x == 0) {
                uint64_t desc_P_mn = make_smem_desc_mnmaj(smem->pass2.P);
                uint64_t desc_dO_mn = make_smem_desc_mnmaj(smem->pass2.dO);
                compute_umma_128(tmem_dV, desc_P_mn, desc_dO_mn, idesc_m_n, true, true, (q_j > 0));
                
                uint64_t desc_dS_mn = make_smem_desc_mnmaj(smem->pass2.dS);
                uint64_t desc_Q_mn = make_smem_desc_mnmaj(smem->pass2.Q);
                compute_umma_128(tmem_dK, desc_dS_mn, desc_Q_mn, idesc_m_n, true, true, (q_j > 0));
                
                umma_commit_cg1_fn(&mbar_umma);
            }
            mbarrier_wait_fn(&mbar_umma, phase_umma); phase_umma ^= 1;
            __syncthreads();
        }
        
        tmem_epilogue_coalesced_4w_base_fn(tmem_dK, dK_ptr + (uint64_t)bh_idx * S * 128, smem->pass2.Q, S, k_i / 128);
        tmem_epilogue_coalesced_4w_base_fn(tmem_dV, dV_ptr + (uint64_t)bh_idx * S * 128, smem->pass2.dO, S, k_i / 128);
    }
    
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 512);
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
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    
    CUresult res;
    res = create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, S, B * H, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
    if(res != CUDA_SUCCESS) { fprintf(stderr, "TMA Encode Error Q: %d\n", res); return; }
    res = create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), 128, S, B * H, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
    if(res != CUDA_SUCCESS) { fprintf(stderr, "TMA Encode Error K: %d\n", res); return; }
    res = create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), 128, S, B * H, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
    if(res != CUDA_SUCCESS) { fprintf(stderr, "TMA Encode Error V: %d\n", res); return; }
    res = create_tma_3d_descriptor_2B(&tma_O, O.data_ptr(), 128, S, B * H, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
    if(res != CUDA_SUCCESS) { fprintf(stderr, "TMA Encode Error O: %d\n", res); return; }
    res = create_tma_3d_descriptor_2B(&tma_dO, dO.data_ptr(), 128, S, B * H, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
    if(res != CUDA_SUCCESS) { fprintf(stderr, "TMA Encode Error dO: %d\n", res); return; }
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int blocks = B * H;
    int threads = 128;
    int smem_size = 227 * 1024;
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_d128_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    mha_bwd_d128_kernel<<<blocks, threads, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO,
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}