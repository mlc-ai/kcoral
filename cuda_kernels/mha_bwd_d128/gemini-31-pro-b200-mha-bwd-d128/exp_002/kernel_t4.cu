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

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
                 :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"((uint64_t)d),
                    "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
                 :: "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(smem)), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
template<int N> __device__ __forceinline__ void tma_store_wait_fn() { asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory"); }

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}
__device__ __forceinline__ void fence_mbarrier_init_fn() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
}
__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}
__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile("{\n.reg .pred P;\nWAIT_%=:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra WAIT_%=;\n}\n"
                 :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tcgen05_fence_after_fn() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tcgen05_fence_before_fn() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_fn() { asm volatile("fence.proxy.async;\n" ::: "memory"); }
__device__ __forceinline__ void tmem_load_fence_fn() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3, uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),"=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = ((uint64_t)((uint32_t)__cvta_generic_to_shared(smem_ptr) & 0x3FFFF) >> 4);
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_idesc_fn(uint32_t M, uint32_t N, int tA, int tB) {
    uint32_t d = (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)tA << 15) | ((uint32_t)tB << 16) | ((N / 8) << 17) | ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__global__ void compute_D_kernel(const __nv_bfloat16* O, const __nv_bfloat16* dO, float* D, int S) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;
    int row_offset = (blockIdx.z * gridDim.y + blockIdx.y) * S + s;
    const __nv_bfloat162* O_row = (const __nv_bfloat162*)(O + row_offset * 128);
    const __nv_bfloat162* dO_row = (const __nv_bfloat162*)(dO + row_offset * 128);
    float sum = 0;
    for (int i = 0; i < 64; ++i) {
        float2 o = __bfloat1622float2(O_row[i]);
        float2 do_ = __bfloat1622float2(dO_row[i]);
        sum += o.x * do_.x + o.y * do_.y;
    }
    D[row_offset] = sum;
}

template <bool IS_Q_CENTRIC>
__global__ void bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q_0, const __grid_constant__ CUtensorMap tma_Q_1,
    const __grid_constant__ CUtensorMap tma_K_0, const __grid_constant__ CUtensorMap tma_K_1,
    const __grid_constant__ CUtensorMap tma_V_0, const __grid_constant__ CUtensorMap tma_V_1,
    const __grid_constant__ CUtensorMap tma_dO_0, const __grid_constant__ CUtensorMap tma_dO_1,
    const __grid_constant__ CUtensorMap tma_dQ_0, const __grid_constant__ CUtensorMap tma_dQ_1,
    const __grid_constant__ CUtensorMap tma_dK_0, const __grid_constant__ CUtensorMap tma_dK_1,
    const __grid_constant__ CUtensorMap tma_dV_0, const __grid_constant__ CUtensorMap tma_dV_1,
    const float* L_ptr, const float* D_ptr, int S, float scale) 
{
    int b = blockIdx.z, h = blockIdx.y, seq_chunk = blockIdx.x;
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane = threadIdx.x % 32;
    int base_idx = (b * gridDim.y + h) * S;

    __shared__ __align__(1024) __nv_bfloat16 smem_Q_0[128][64];
    __shared__ __align__(1024) __nv_bfloat16 smem_Q_1[128][64];
    __shared__ __align__(1024) __nv_bfloat16 smem_K_0[64][64];
    __shared__ __align__(1024) __nv_bfloat16 smem_K_1[64][64];
    __shared__ __align__(1024) __nv_bfloat16 smem_V_0[64][64];
    __shared__ __align__(1024) __nv_bfloat16 smem_V_1[64][64];
    __shared__ __align__(1024) __nv_bfloat16 smem_dO_0[128][64];
    __shared__ __align__(1024) __nv_bfloat16 smem_dO_1[128][64];
    __shared__ __align__(1024) __nv_bfloat16 smem_P[128][64];
    __shared__ __align__(1024) __nv_bfloat16 smem_dS[128][64];

    __shared__ uint64_t mbar;
    __shared__ uint64_t mbar_inner;
    __shared__ uint64_t mbar_mma_0;
    __shared__ uint64_t mbar_mma_1;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar, 128);
        init_smem_barrier_fn(&mbar_inner, 128);
        init_smem_barrier_fn(&mbar_mma_0, 1);
        init_smem_barrier_fn(&mbar_mma_1, 1);
        fence_mbarrier_init_fn();
    }
    __syncthreads();

    uint64_t desc_Q_0_K = make_smem_desc_fn(smem_Q_0, 1, 1024);
    uint64_t desc_Q_1_K = make_smem_desc_fn(smem_Q_1, 1, 1024);
    uint64_t desc_K_0_K = make_smem_desc_fn(smem_K_0, 1, 1024);
    uint64_t desc_K_1_K = make_smem_desc_fn(smem_K_1, 1, 1024);
    uint64_t desc_V_0_K = make_smem_desc_fn(smem_V_0, 1, 1024);
    uint64_t desc_V_1_K = make_smem_desc_fn(smem_V_1, 1, 1024);
    uint64_t desc_dO_0_K = make_smem_desc_fn(smem_dO_0, 1, 1024);
    uint64_t desc_dO_1_K = make_smem_desc_fn(smem_dO_1, 1, 1024);
    uint64_t desc_dS_K  = make_smem_desc_fn(smem_dS, 1, 1024);

    uint64_t desc_K_0_MN = make_smem_desc_fn(smem_K_0, 8192, 1024);
    uint64_t desc_K_1_MN = make_smem_desc_fn(smem_K_1, 8192, 1024);
    uint64_t desc_P_MN   = make_smem_desc_fn(smem_P, 16384, 1024);
    uint64_t desc_dS_MN  = make_smem_desc_fn(smem_dS, 16384, 1024);

    uint32_t idesc_S  = make_idesc_fn(128, 64, 0, 0); 
    uint32_t idesc_dP = make_idesc_fn(128, 64, 0, 0); 
    uint32_t idesc_dQ = make_idesc_fn(128, 64, 0, 1); 
    uint32_t idesc_dK = make_idesc_fn(64, 64, 1, 0);  
    uint32_t idesc_dV = make_idesc_fn(64, 64, 1, 0);  

    __shared__ uint32_t tmem_base;
    if (warp_id == 0) tmem_alloc_fn(&tmem_base, 512); 
    __syncthreads();

    uint32_t tmem_accum_0 = tmem_base + 0;
    uint32_t tmem_accum_1 = tmem_base + 64; 

    if (IS_Q_CENTRIC) {
        uint32_t tmem_S = tmem_base + 128;
        uint32_t tmem_dP = tmem_base + 192;

        int q_idx = seq_chunk;
        int q_s = base_idx + q_idx * 128;

        if (threadIdx.x == 0) {
            uint32_t bytes = 4 * 128 * 64 * 2;
            mbarrier_arrive_and_expect_tx_fn(&mbar, bytes);
            tma_load_2d_fn(&tma_Q_0, &mbar, smem_Q_0, 0, q_s);
            tma_load_2d_fn(&tma_Q_1, &mbar, smem_Q_1, 64, q_s);
            tma_load_2d_fn(&tma_dO_0, &mbar, smem_dO_0, 0, q_s);
            tma_load_2d_fn(&tma_dO_1, &mbar, smem_dO_1, 64, q_s);
        } else {
            mbarrier_arrive_fn(&mbar);
        }
        mbarrier_wait_fn(&mbar, 0);

        float l_val = L_ptr[base_idx + q_idx * 128 + warp_id * 32 + lane];
        float d_val = D_ptr[base_idx + q_idx * 128 + warp_id * 32 + lane];

        for (int k_idx = 0; k_idx < S / 64; ++k_idx) {
            int k_s = base_idx + k_idx * 64;
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_inner, 4 * 64 * 64 * 2);
                tma_load_2d_fn(&tma_K_0, &mbar_inner, smem_K_0, 0, k_s);
                tma_load_2d_fn(&tma_K_1, &mbar_inner, smem_K_1, 64, k_s);
                tma_load_2d_fn(&tma_V_0, &mbar_inner, smem_V_0, 0, k_s);
                tma_load_2d_fn(&tma_V_1, &mbar_inner, smem_V_1, 64, k_s);
            } else {
                mbarrier_arrive_fn(&mbar_inner);
            }
            mbarrier_wait_fn(&mbar_inner, k_idx % 2);
            fence_proxy_async_fn();
            tcgen05_fence_before_fn();

            if (threadIdx.x == 0) {
                for (int i = 0; i < 4; ++i) umma_f16_cg1_fn(tmem_S, desc_Q_0_K + i * 2, desc_K_0_K + i * 2, idesc_S, (i == 0) ? 0 : 1);
                for (int i = 0; i < 4; ++i) umma_f16_cg1_fn(tmem_S, desc_Q_1_K + i * 2, desc_K_1_K + i * 2, idesc_S, 1);
                for (int i = 0; i < 4; ++i) umma_f16_cg1_fn(tmem_dP, desc_dO_0_K + i * 2, desc_V_0_K + i * 2, idesc_dP, (i == 0) ? 0 : 1);
                for (int i = 0; i < 4; ++i) umma_f16_cg1_fn(tmem_dP, desc_dO_1_K + i * 2, desc_V_1_K + i * 2, idesc_dP, 1);
            }

            tcgen05_fence_after_fn();
            if (threadIdx.x == 0) umma_commit_cg1_fn(&mbar_mma_0);
            mbarrier_wait_fn(&mbar_mma_0, k_idx % 2);

            for (int c = 0; c < 64; c += 8) {
                uint32_t r[8], dp[8];
                tmem_load_8x_fn(tmem_S + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
                tmem_load_8x_fn(tmem_dP + c, &dp[0], &dp[1], &dp[2], &dp[3], &dp[4], &dp[5], &dp[6], &dp[7]);
                tmem_load_fence_fn();
                for (int i = 0; i < 4; ++i) {
                    float s0 = expf(__uint_as_float(r[2*i]) * scale - l_val);
                    float s1 = expf(__uint_as_float(r[2*i+1]) * scale - l_val);
                    float dp0 = __uint_as_float(dp[2*i]), dp1 = __uint_as_float(dp[2*i+1]);
                    float ds0 = s0 * (dp0 - d_val) * scale, ds1 = s1 * (dp1 - d_val) * scale;
                    
                    union { __nv_bfloat162 bf; uint32_t u; } p_cvt, ds_cvt;
                    p_cvt.bf = __floats2bfloat162_rn(s0, s1);
                    ds_cvt.bf = __floats2bfloat162_rn(ds0, ds1);

                    int smem_c = c + i * 2, smem_r = warp_id * 32 + lane;
                    *(__nv_bfloat162*)&smem_P[smem_r][smem_c] = p_cvt.bf;
                    *(__nv_bfloat162*)&smem_dS[smem_r][smem_c] = ds_cvt.bf;
                }
            }
            __syncthreads();
            fence_proxy_async_fn();
            tcgen05_fence_before_fn();

            if (threadIdx.x == 0) {
                for (int i = 0; i < 4; ++i) umma_f16_cg1_fn(tmem_accum_0, desc_dS_K + i * 2, desc_K_0_MN + i * 128, idesc_dQ, (k_idx == 0 && i == 0) ? 0 : 1);
                for (int i = 0; i < 4; ++i) umma_f16_cg1_fn(tmem_accum_1, desc_dS_K + i * 2, desc_K_1_MN + i * 128, idesc_dQ, (k_idx == 0 && i == 0) ? 0 : 1);
            }

            tcgen05_fence_after_fn();
            if (threadIdx.x == 0) umma_commit_cg1_fn(&mbar_mma_1);
            mbarrier_wait_fn(&mbar_mma_1, k_idx % 2);
            __syncthreads();
        }

        for (int part = 0; part < 2; ++part) {
            uint32_t t_base = (part == 0) ? tmem_accum_0 : tmem_accum_1;
            for (int c = 0; c < 64; c += 8) {
                uint32_t r[8];
                tmem_load_8x_fn(t_base + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
                tmem_load_fence_fn();
                for (int i = 0; i < 4; ++i) {
                    union { __nv_bfloat162 bf; uint32_t u; } cvt;
                    cvt.bf = __floats2bfloat162_rn(__uint_as_float(r[2*i]), __uint_as_float(r[2*i+1]));
                    int smem_c = c + i * 2, smem_r = warp_id * 32 + lane;
                    if (part == 0) *(__nv_bfloat162*)&smem_Q_0[smem_r][smem_c] = cvt.bf;
                    else *(__nv_bfloat162*)&smem_Q_1[smem_r][smem_c] = cvt.bf;
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            tma_store_2d_fn(&tma_dQ_0, smem_Q_0, 0, q_s);
            tma_store_2d_fn(&tma_dQ_1, smem_Q_1, 64, q_s);
            tma_store_commit_fn();
            tma_store_wait_fn<0>();
        }
        __syncthreads();
    } else {
        uint32_t tmem_accum_2 = tmem_base + 128;
        uint32_t tmem_accum_3 = tmem_base + 192;
        uint32_t tmem_S = tmem_base + 256;
        uint32_t tmem_dP = tmem_base + 320;

        int k_idx = seq_chunk;
        int k_s = base_idx + k_idx * 64;

        if (threadIdx.x == 0) {
            uint32_t bytes = 4 * 64 * 64 * 2;
            mbarrier_arrive_and_expect_tx_fn(&mbar, bytes);
            tma_load_2d_fn(&tma_K_0, &mbar, smem_K_0, 0, k_s);
            tma_load_2d_fn(&tma_K_1, &mbar, smem_K_1, 64, k_s);
            tma_load_2d_fn(&tma_V_0, &mbar, smem_V_0, 0, k_s);
            tma_load_2d_fn(&tma_V_1, &mbar, smem_V_1, 64, k_s);
        } else {
            mbarrier_arrive_fn(&mbar);
        }
        mbarrier_wait_fn(&mbar, 0);

        for (int q_idx = 0; q_idx < S / 128; ++q_idx) {
            int q_s = base_idx + q_idx * 128;
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_inner, 4 * 128 * 64 * 2);
                tma_load_2d_fn(&tma_Q_0, &mbar_inner, smem_Q_0, 0, q_s);
                tma_load_2d_fn(&tma_Q_1, &mbar_inner, smem_Q_1, 64, q_s);
                tma_load_2d_fn(&tma_dO_0, &mbar_inner, smem_dO_0, 0, q_s);
                tma_load_2d_fn(&tma_dO_1, &mbar_inner, smem_dO_1, 64, q_s);
            } else {
                mbarrier_arrive_fn(&mbar_inner);
            }
            mbarrier_wait_fn(&mbar_inner, q_idx % 2);
            fence_proxy_async_fn();
            tcgen05_fence_before_fn();

            if (threadIdx.x == 0) {
                for (int i = 0; i < 4; ++i) umma_f16_cg1_fn(tmem_S, desc_Q_0_K + i * 2, desc_K_0_K + i * 2, idesc_S, (i == 0) ? 0 : 1);
                for (int i = 0; i < 4; ++i) umma_f16_cg1_fn(tmem_S, desc_Q_1_K + i * 2, desc_K_1_K + i * 2, idesc_S, 1);
                for (int i = 0; i < 4; ++i) umma_f16_cg1_fn(tmem_dP, desc_dO_0_K + i * 2, desc_V_0_K + i * 2, idesc_dP, (i == 0) ? 0 : 1);
                for (int i = 0; i < 4; ++i) umma_f16_cg1_fn(tmem_dP, desc_dO_1_K + i * 2, desc_V_1_K + i * 2, idesc_dP, 1);
            }

            tcgen05_fence_after_fn();
            if (threadIdx.x == 0) umma_commit_cg1_fn(&mbar_mma_0);
            mbarrier_wait_fn(&mbar_mma_0, q_idx % 2);

            float l_val = L_ptr[base_idx + q_idx * 128 + warp_id * 32 + lane];
            float d_val = D_ptr[base_idx + q_idx * 128 + warp_id * 32 + lane];

            for (int c = 0; c < 64; c += 8) {
                uint32_t r[8], dp[8];
                tmem_load_8x_fn(tmem_S + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
                tmem_load_8x_fn(tmem_dP + c, &dp[0], &dp[1], &dp[2], &dp[3], &dp[4], &dp[5], &dp[6], &dp[7]);
                tmem_load_fence_fn();
                for (int i = 0; i < 4; ++i) {
                    float s0 = expf(__uint_as_float(r[2*i]) * scale - l_val);
                    float s1 = expf(__uint_as_float(r[2*i+1]) * scale - l_val);
                    float dp0 = __uint_as_float(dp[2*i]), dp1 = __uint_as_float(dp[2*i+1]);
                    float ds0 = s0 * (dp0 - d_val) * scale, ds1 = s1 * (dp1 - d_val) * scale;
                    
                    union { __nv_bfloat162 bf; uint32_t u; } p_cvt, ds_cvt;
                    p_cvt.bf = __floats2bfloat162_rn(s0, s1);
                    ds_cvt.bf = __floats2bfloat162_rn(ds0, ds1);

                    int smem_c = c + i * 2, smem_r = warp_id * 32 + lane;
                    *(__nv_bfloat162*)&smem_P[smem_r][smem_c] = p_cvt.bf;
                    *(__nv_bfloat162*)&smem_dS[smem_r][smem_c] = ds_cvt.bf;
                }
            }
            __syncthreads();
            fence_proxy_async_fn();
            tcgen05_fence_before_fn();

            if (threadIdx.x == 0) {
                for (int i = 0; i < 8; ++i) umma_f16_cg1_fn(tmem_accum_0, desc_dS_MN + i * 128, desc_Q_0_K + i * 128, idesc_dK, (q_idx == 0 && i == 0) ? 0 : 1);
                for (int i = 0; i < 8; ++i) umma_f16_cg1_fn(tmem_accum_1, desc_dS_MN + i * 128, desc_Q_1_K + i * 128, idesc_dK, (q_idx == 0 && i == 0) ? 0 : 1);
                for (int i = 0; i < 8; ++i) umma_f16_cg1_fn(tmem_accum_2, desc_P_MN + i * 128, desc_dO_0_K + i * 128, idesc_dV, (q_idx == 0 && i == 0) ? 0 : 1);
                for (int i = 0; i < 8; ++i) umma_f16_cg1_fn(tmem_accum_3, desc_P_MN + i * 128, desc_dO_1_K + i * 128, idesc_dV, (q_idx == 0 && i == 0) ? 0 : 1);
            }

            tcgen05_fence_after_fn();
            if (threadIdx.x == 0) umma_commit_cg1_fn(&mbar_mma_1);
            mbarrier_wait_fn(&mbar_mma_1, q_idx % 2);
            __syncthreads();
        }

        for (int part = 0; part < 4; ++part) {
            uint32_t t_base;
            if (part == 0) t_base = tmem_accum_0;
            else if (part == 1) t_base = tmem_accum_1;
            else if (part == 2) t_base = tmem_accum_2;
            else t_base = tmem_accum_3;

            for (int c = 0; c < 64; c += 8) {
                uint32_t r[8];
                tmem_load_8x_fn(t_base + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
                tmem_load_fence_fn();
                for (int i = 0; i < 4; ++i) {
                    union { __nv_bfloat162 bf; uint32_t u; } cvt;
                    cvt.bf = __floats2bfloat162_rn(__uint_as_float(r[2*i]), __uint_as_float(r[2*i+1]));
                    int smem_c = c + i * 2, smem_r = warp_id * 32 + lane;
                    if (smem_r < 64) {
                        if (part == 0) *(__nv_bfloat162*)&smem_K_0[smem_r][smem_c] = cvt.bf;
                        else if (part == 1) *(__nv_bfloat162*)&smem_K_1[smem_r][smem_c] = cvt.bf;
                        else if (part == 2) *(__nv_bfloat162*)&smem_V_0[smem_r][smem_c] = cvt.bf;
                        else *(__nv_bfloat162*)&smem_V_1[smem_r][smem_c] = cvt.bf;
                    }
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            tma_store_2d_fn(&tma_dK_0, smem_K_0, 0, k_s);
            tma_store_2d_fn(&tma_dK_1, smem_K_1, 64, k_s);
            tma_store_2d_fn(&tma_dV_0, smem_V_0, 0, k_s);
            tma_store_2d_fn(&tma_dV_1, smem_V_1, 64, k_s);
            tma_store_commit_fn();
            tma_store_wait_fn<0>();
        }
        __syncthreads();
    }

    if (warp_id == 0) tmem_dealloc_fn(tmem_base, 512);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int B = Q.size(0), H = Q.size(1), S = Q.size(2), d = 128;
    float scale = 1.0f / sqrtf(128.0f);

    float* d_D = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_D, B * H * S * sizeof(float), stream));
    compute_D_kernel<<<dim3(S / 128, H, B), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(O.data_ptr()), 
        static_cast<const __nv_bfloat16*>(dO.data_ptr()), 
        d_D, S);

    CUtensorMap tma_Q_0, tma_Q_1, tma_K_0, tma_K_1, tma_V_0, tma_V_1, tma_dO_0, tma_dO_1;
    CUtensorMap tma_dQ_0, tma_dQ_1, tma_dK_0, tma_dK_1, tma_dV_0, tma_dV_1;

    auto make_tma = [&](CUtensorMap* map, void* ptr, uint32_t inner) {
        create_tma_2d_descriptor_2B(map, ptr, d, B * H * S, 64, inner, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    };

    make_tma(&tma_Q_0, Q.data_ptr(), 128); make_tma(&tma_Q_1, Q.data_ptr(), 128);
    make_tma(&tma_K_0, K.data_ptr(), 64); make_tma(&tma_K_1, K.data_ptr(), 64);
    make_tma(&tma_V_0, V.data_ptr(), 64); make_tma(&tma_V_1, V.data_ptr(), 64);
    make_tma(&tma_dO_0, dO.data_ptr(), 128); make_tma(&tma_dO_1, dO.data_ptr(), 128);
    make_tma(&tma_dQ_0, dQ.data_ptr(), 128); make_tma(&tma_dQ_1, dQ.data_ptr(), 128);
    make_tma(&tma_dK_0, dK.data_ptr(), 64); make_tma(&tma_dK_1, dK.data_ptr(), 64);
    make_tma(&tma_dV_0, dV.data_ptr(), 64); make_tma(&tma_dV_1, dV.data_ptr(), 64);

    bwd_kernel<false><<<dim3(S / 64, H, B), 128, 0, stream>>>(
        tma_Q_0, tma_Q_1, tma_K_0, tma_K_1, tma_V_0, tma_V_1, tma_dO_0, tma_dO_1,
        tma_dQ_0, tma_dQ_1, tma_dK_0, tma_dK_1, tma_dV_0, tma_dV_1,
        static_cast<const float*>(L.data_ptr()), d_D, S, scale);
    
    bwd_kernel<true><<<dim3(S / 128, H, B), 128, 0, stream>>>(
        tma_Q_0, tma_Q_1, tma_K_0, tma_K_1, tma_V_0, tma_V_1, tma_dO_0, tma_dO_1,
        tma_dQ_0, tma_dQ_1, tma_dK_0, tma_dK_1, tma_dV_0, tma_dV_1,
        static_cast<const float*>(L.data_ptr()), d_D, S, scale);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(d_D, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);