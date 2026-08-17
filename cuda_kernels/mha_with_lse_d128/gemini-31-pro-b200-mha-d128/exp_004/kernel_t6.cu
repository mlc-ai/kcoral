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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n",                  \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha {

CUresult create_tma_3d_descriptor(CUtensorMap* d, void* globalAddress,
    uint64_t dim2, uint64_t dim1, uint64_t dim0,
    uint32_t box2, uint32_t box1, uint32_t box0,
    CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_3d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, void* pattern_start, uint32_t lbo, uint32_t sbo, uint32_t swizzle_type) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t base_addr = (uint32_t)__cvta_generic_to_shared(pattern_start);
    uint32_t base_offset = (base_addr >> 7) & 0x7;
    
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)(base_offset & 0x7) << 49;
    d |= (uint64_t)swizzle_type << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   // K-Major
    d |= (0u << 16);   // K-Major
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_PV_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   // K-Major
    d |= (1u << 16);   // N-Major
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
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

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ void store_swizzled_bf16x4(void* base, int row, int col_bf16, __nv_bfloat16 p0, __nv_bfloat16 p1, __nv_bfloat16 p2, __nv_bfloat16 p3) {
    int chunk_idx = (col_bf16 / 8) % 8;
    int swizzled_chunk = chunk_idx ^ (row % 8);
    int final_col = swizzled_chunk * 8 + (col_bf16 % 8);
    __nv_bfloat16* ptr = (__nv_bfloat16*)base + row * 64 + final_col;
    ptr[0] = p0;
    ptr[1] = p1;
    ptr[2] = p2;
    ptr[3] = p3;
}

__device__ __forceinline__ void store_swizzled_bf16x2(void* base, int row, int col_bf16, __nv_bfloat16 o0, __nv_bfloat16 o1) {
    int chunk_idx = (col_bf16 / 8) % 8;
    int swizzled_chunk = chunk_idx ^ (row % 8);
    int final_col = swizzled_chunk * 8 + (col_bf16 % 8);
    __nv_bfloat16* ptr = (__nv_bfloat16*)base + row * 64 + final_col;
    ptr[0] = o0;
    ptr[1] = o1;
}

struct SharedStorage {
    __align__(1024) __nv_bfloat16 Q[2][2][128][64]; // [wg_id][tile][128][64]
    __align__(1024) __nv_bfloat16 K[2][2][64][64];  // [db][tile][64][64]
    __align__(1024) __nv_bfloat16 V[2][2][64][64];  // [db][tile][64][64]
    __align__(1024) __nv_bfloat16 P[2][128][64];    // [wg_id][128][64]
    
    __align__(8) uint64_t bar_Q;
    __align__(8) uint64_t bar_KV[2];
    __align__(8) uint64_t bar_S[2];
    __align__(8) uint64_t bar_O[2];
    
    __align__(8) uint32_t tmem_S[2];
    __align__(8) uint32_t tmem_O[2];
};

__global__ void mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* LSE,
    int S,
    int H
) {
    extern __shared__ char smem_buf[];
    uintptr_t smem_ptr = reinterpret_cast<uintptr_t>(smem_buf);
    smem_ptr = (smem_ptr + 1023) & ~1023ULL;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_ptr);

    int wg_id = threadIdx.x / 128;
    int tid = threadIdx.x % 128;

    int q_idx_0 = blockIdx.x * 256;
    int q_idx_1 = q_idx_0 + 128;
    int my_q_idx = (wg_id == 0) ? q_idx_0 : q_idx_1;
    
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int bh_idx = batch_idx * H + head_idx;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&smem.bar_Q, 1);
        init_smem_barrier_fn(&smem.bar_KV[0], 1);
        init_smem_barrier_fn(&smem.bar_KV[1], 1);
        init_smem_barrier_fn(&smem.bar_S[0], 1);
        init_smem_barrier_fn(&smem.bar_S[1], 1);
        init_smem_barrier_fn(&smem.bar_O[0], 1);
        init_smem_barrier_fn(&smem.bar_O[1], 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&smem.tmem_S[0], 128);
        tmem_alloc_cg1_fn(&smem.tmem_O[0], 128);
    } else if (threadIdx.x >= 128 && threadIdx.x < 160) {
        tmem_alloc_cg1_fn(&smem.tmem_S[1], 128);
        tmem_alloc_cg1_fn(&smem.tmem_O[1], 128);
    }
    __syncthreads();

    uint32_t phase_Q = 0;
    uint32_t phase_KV[2] = {0, 0};
    uint32_t phase_S = 0;
    uint32_t phase_O = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&smem.bar_Q, 65536);
        tma_load_3d_fn(&tma_Q, &smem.bar_Q, smem.Q[0][0], 0, q_idx_0, bh_idx);
        tma_load_3d_fn(&tma_Q, &smem.bar_Q, smem.Q[0][1], 64, q_idx_0, bh_idx);
        tma_load_3d_fn(&tma_Q, &smem.bar_Q, smem.Q[1][0], 0, q_idx_1, bh_idx);
        tma_load_3d_fn(&tma_Q, &smem.bar_Q, smem.Q[1][1], 64, q_idx_1, bh_idx);
        
        mbarrier_arrive_and_expect_tx_fn(&smem.bar_KV[0], 32768);
        tma_load_3d_fn(&tma_K, &smem.bar_KV[0], smem.K[0][0], 0, 0, bh_idx);
        tma_load_3d_fn(&tma_K, &smem.bar_KV[0], smem.K[0][1], 64, 0, bh_idx);
        tma_load_3d_fn(&tma_V, &smem.bar_KV[0], smem.V[0][0], 0, 0, bh_idx);
        tma_load_3d_fn(&tma_V, &smem.bar_KV[0], smem.V[0][1], 64, 0, bh_idx);
    }
    
    mbarrier_wait_fn(&smem.bar_Q, phase_Q);
    phase_Q ^= 1;
    __syncthreads();

    float O_accum_reg[128];
    #pragma unroll
    for (int i = 0; i < 128; ++i) O_accum_reg[i] = 0.0f;
    float m_old = -1e20f;
    float row_sum_reg = 0.0f;

    float scale = 0.0883883476483f;
    uint32_t idesc_QK = make_instr_desc_fn(128, 64);
    uint32_t idesc_PV = make_instr_desc_PV_fn(128, 64);

    for (int kv_idx = 0; kv_idx < S; kv_idx += 64) {
        int db = (kv_idx / 64) % 2;
        int next_db = 1 - db;

        if (kv_idx + 64 < S) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&smem.bar_KV[next_db], 32768);
                tma_load_3d_fn(&tma_K, &smem.bar_KV[next_db], smem.K[next_db][0], 0, kv_idx + 64, bh_idx);
                tma_load_3d_fn(&tma_K, &smem.bar_KV[next_db], smem.K[next_db][1], 64, kv_idx + 64, bh_idx);
                tma_load_3d_fn(&tma_V, &smem.bar_KV[next_db], smem.V[next_db][0], 0, kv_idx + 64, bh_idx);
                tma_load_3d_fn(&tma_V, &smem.bar_KV[next_db], smem.V[next_db][1], 64, kv_idx + 64, bh_idx);
            }
        }

        mbarrier_wait_fn(&smem.bar_KV[db], phase_KV[db]);
        phase_KV[db] ^= 1;

        if (tid == 0) {
            for (int i = 0; i < 8; ++i) {
                bool is_tile1 = (i >= 4);
                int local_i = i % 4;
                void* base_Q = is_tile1 ? (void*)smem.Q[wg_id][1] : (void*)smem.Q[wg_id][0];
                void* base_K = is_tile1 ? (void*)smem.K[db][1] : (void*)smem.K[db][0];
                uint8_t* ptr_Q = (uint8_t*)base_Q + local_i * 32;
                uint8_t* ptr_K = (uint8_t*)base_K + local_i * 32;
                
                uint64_t desc_Q = make_smem_desc_sm100_fn(ptr_Q, base_Q, 1, 1024, 2);
                uint64_t desc_K = make_smem_desc_sm100_fn(ptr_K, base_K, 1, 1024, 2);
                umma_f16_cg1_fn(smem.tmem_S[wg_id], desc_Q, desc_K, idesc_QK, (i == 0) ? 0 : 1);
            }
            umma_commit_cg1_fn(&smem.bar_S[wg_id]);
        }
        mbarrier_wait_fn(&smem.bar_S[wg_id], phase_S);
        phase_S ^= 1;

        float m_new = m_old;
        for (int c = 0; c < 64; c += 16) {
            uint32_t r[16];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(smem.tmem_S[wg_id] + c));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(smem.tmem_S[wg_id] + c + 4));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]) : "r"(smem.tmem_S[wg_id] + c + 8));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]) : "r"(smem.tmem_S[wg_id] + c + 12));
            tmem_load_fence_fn();

            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                float v = __uint_as_float(r[j]) * scale;
                if (kv_idx + c + j < S && v > m_new) m_new = v;
            }
        }

        float factor = (m_old <= -1e19f) ? 0.0f : fast_exp2f_fn((m_old - m_new) * 1.4426950408889634f);
        #pragma unroll 16
        for (int c = 0; c < 128; ++c) {
            O_accum_reg[c] *= factor;
        }
        row_sum_reg *= factor;

        for (int c = 0; c < 64; c += 16) {
            uint32_t r[16];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(smem.tmem_S[wg_id] + c));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(smem.tmem_S[wg_id] + c + 4));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]) : "r"(smem.tmem_S[wg_id] + c + 8));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]) : "r"(smem.tmem_S[wg_id] + c + 12));
            tmem_load_fence_fn();

            for (int j = 0; j < 16; j += 4) {
                float v0 = __uint_as_float(r[j+0]) * scale;
                float v1 = __uint_as_float(r[j+1]) * scale;
                float v2 = __uint_as_float(r[j+2]) * scale;
                float v3 = __uint_as_float(r[j+3]) * scale;

                float p0 = (kv_idx + c + j + 0 >= S) ? 0.0f : fast_exp2f_fn((v0 - m_new) * 1.4426950408889634f);
                float p1 = (kv_idx + c + j + 1 >= S) ? 0.0f : fast_exp2f_fn((v1 - m_new) * 1.4426950408889634f);
                float p2 = (kv_idx + c + j + 2 >= S) ? 0.0f : fast_exp2f_fn((v2 - m_new) * 1.4426950408889634f);
                float p3 = (kv_idx + c + j + 3 >= S) ? 0.0f : fast_exp2f_fn((v3 - m_new) * 1.4426950408889634f);

                row_sum_reg += p0 + p1 + p2 + p3;
                store_swizzled_bf16x4(smem.P[wg_id], tid, c + j, 
                    __float2bfloat16(p0), __float2bfloat16(p1), __float2bfloat16(p2), __float2bfloat16(p3));
            }
        }
        m_old = m_new;
        fence_proxy_async_fn();
        __syncthreads(); // Synchronize all so P is visible and K/V usage aligns

        if (tid == 0) {
            for (int i = 0; i < 4; ++i) { 
                void* base_P = (void*)smem.P[wg_id];
                uint8_t* ptr_P = (uint8_t*)base_P + i * 32;
                void* base_V0 = (void*)smem.V[db][0];
                uint8_t* ptr_V0 = (uint8_t*)base_V0 + i * 32;
                
                uint64_t desc_P = make_smem_desc_sm100_fn(ptr_P, base_P, 1, 1024, 2);
                uint64_t desc_V0 = make_smem_desc_sm100_fn(ptr_V0, base_V0, 1, 1024, 2);
                umma_f16_cg1_fn(smem.tmem_O[wg_id], desc_P, desc_V0, idesc_PV, 0); 
                
                void* base_V1 = (void*)smem.V[db][1];
                uint8_t* ptr_V1 = (uint8_t*)base_V1 + i * 32;
                uint64_t desc_V1 = make_smem_desc_sm100_fn(ptr_V1, base_V1, 1, 1024, 2);
                umma_f16_cg1_fn(smem.tmem_O[wg_id] + 64, desc_P, desc_V1, idesc_PV, 0);
            }
            umma_commit_cg1_fn(&smem.bar_O[wg_id]);
        }
        mbarrier_wait_fn(&smem.bar_O[wg_id], phase_O);
        phase_O ^= 1;

        for (int c = 0; c < 128; c += 16) {
            uint32_t r[16];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(smem.tmem_O[wg_id] + c));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(smem.tmem_O[wg_id] + c + 4));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]) : "r"(smem.tmem_O[wg_id] + c + 8));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]) : "r"(smem.tmem_O[wg_id] + c + 12));
            tmem_load_fence_fn();

            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                O_accum_reg[c + j] += __uint_as_float(r[j]);
            }
        }
        __syncthreads();
    }

    float inv_sum = (row_sum_reg > 0.0f) ? (1.0f / row_sum_reg) : 0.0f;
    for (int c = 0; c < 128; c += 2) {
        float o0 = O_accum_reg[c] * inv_sum;
        float o1 = O_accum_reg[c+1] * inv_sum;
        if (c < 64) {
            store_swizzled_bf16x2(smem.Q[wg_id][0], tid, c, __float2bfloat16(o0), __float2bfloat16(o1));
        } else {
            store_swizzled_bf16x2(smem.Q[wg_id][1], tid, c - 64, __float2bfloat16(o0), __float2bfloat16(o1));
        }
    }

    fence_proxy_async_fn();
    __syncthreads();

    if (threadIdx.x == 0) {
        tma_store_3d_fn(&tma_O, smem.Q[0][0], 0, q_idx_0, bh_idx);
        tma_store_3d_fn(&tma_O, smem.Q[0][1], 64, q_idx_0, bh_idx);
        tma_store_3d_fn(&tma_O, smem.Q[1][0], 0, q_idx_1, bh_idx);
        tma_store_3d_fn(&tma_O, smem.Q[1][1], 64, q_idx_1, bh_idx);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    
    if (my_q_idx + tid < S) {
        float lse_val = m_old + logf(row_sum_reg);
        LSE[batch_idx * H * S + head_idx * S + my_q_idx + tid] = lse_val;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(smem.tmem_S[0], 128);
        tmem_dealloc_cg1_fn(smem.tmem_O[0], 128);
    } else if (threadIdx.x >= 128 && threadIdx.x < 160) {
        tmem_dealloc_cg1_fn(smem.tmem_S[1], 128);
        tmem_dealloc_cg1_fn(smem.tmem_O[1], 128);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    CU_CHECK(create_tma_3d_descriptor(&tma_Q, Q.data_ptr(), B * H, S, 128, 1, 128, 64, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_3d_descriptor(&tma_K, K.data_ptr(), B * H, S, 128, 1, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_3d_descriptor(&tma_V, V.data_ptr(), B * H, S, 128, 1, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_3d_descriptor(&tma_O, O.data_ptr(), B * H, S, 128, 1, 128, 64, CU_TENSOR_MAP_SWIZZLE_128B));
    
    int num_blocks = (S + 255) / 256;
    dim3 grid(num_blocks, H, B);
    dim3 block(256);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    size_t smem_bytes = sizeof(SharedStorage) + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    
    mha_fwd_kernel<<<grid, block, smem_bytes, stream>>>(
        tma_Q, tma_K, tma_V, tma_O,
        static_cast<float*>(LSE.data_ptr()),
        static_cast<int>(S),
        static_cast<int>(H)
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_mha