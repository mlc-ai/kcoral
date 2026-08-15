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
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float exp_f(float x) {
    return fast_exp2f_fn(x * 1.44269504f);
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;" ::: "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
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

extern __shared__ __align__(1024) uint8_t smem_pool[];

__global__ void mha_kernel_sm100(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    int S_len) 
{
    int q_blk = blockIdx.x;
    int bh = blockIdx.y;
    int S = S_len;

    extern __shared__ __align__(1024) __nv_bfloat16 smem_buf[];
    __nv_bfloat16* smem_Q_0 = smem_buf;                   // 16KB
    __nv_bfloat16* smem_Q_1 = smem_buf + 16384;           // 16KB
    __nv_bfloat16* smem_K_0 = smem_buf + 32768;           // 16KB
    __nv_bfloat16* smem_K_1 = smem_buf + 49152;           // 16KB
    __nv_bfloat16* smem_V_0 = smem_buf + 65536;           // 16KB
    __nv_bfloat16* smem_V_1 = smem_buf + 81920;           // 16KB
    __nv_bfloat16* smem_P_0 = smem_buf + 98304;           // 16KB
    __nv_bfloat16* smem_P_1 = smem_buf + 114688;          // 16KB
    
    __shared__ __align__(8) uint64_t mbar_Q[1];
    __shared__ __align__(8) uint64_t mbar_K[1];
    __shared__ __align__(8) uint64_t mbar_V[1];
    __shared__ __align__(8) uint64_t mbar_P[1];
    __shared__ __align__(8) uint64_t mbar_O[1];

    __shared__ float smem_m[128];
    __shared__ float smem_sum[128];

    if (threadIdx.x < 128) {
        smem_m[threadIdx.x] = -1e20f;
        smem_sum[threadIdx.x] = 0.0f;
    }

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_Q[0], 1);
        init_smem_barrier_fn(&mbar_K[0], 1);
        init_smem_barrier_fn(&mbar_V[0], 1);
        init_smem_barrier_fn(&mbar_P[0], 1);
        init_smem_barrier_fn(&mbar_O[0], 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();

    uint32_t tmem_P, tmem_O_0, tmem_O_1;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_P, 128);
        tmem_alloc_fn(&tmem_O_0, 64);
        tmem_alloc_fn(&tmem_O_1, 64);
    }
    __syncthreads();

    uint32_t phase_Q = 0, phase_K = 0, phase_V = 0, phase_P = 0, phase_O = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_Q[0], 16384);
        tma_load_2d_fn(&tma_Q, &mbar_Q[0], smem_Q_0, 0, bh * S + q_blk * 128);
        tma_load_2d_fn(&tma_Q, &mbar_Q[0], smem_Q_1, 64, bh * S + q_blk * 128);
    }
    mbarrier_wait_fn(&mbar_Q[0], phase_Q);
    phase_Q ^= 1;

    uint32_t idesc_P = make_instr_desc_fn(128, 128);
    uint32_t idesc_O = make_instr_desc_fn(128, 64);

    int num_blocks = (S + 127) / 128;

    for (int k_blk = 0; k_blk <= q_blk && k_blk < num_blocks; ++k_blk) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_K[0], 16384);
            tma_load_2d_fn(&tma_K, &mbar_K[0], smem_K_0, 0, bh * S + k_blk * 128);
            tma_load_2d_fn(&tma_K, &mbar_K[0], smem_K_1, 64, bh * S + k_blk * 128);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar_V[0], 16384);
            tma_load_2d_fn(&tma_V, &mbar_V[0], smem_V_0, 0, bh * S + k_blk * 128);
            tma_load_2d_fn(&tma_V, &mbar_V[0], smem_V_1, 64, bh * S + k_blk * 128);
        }
        mbarrier_wait_fn(&mbar_K[0], phase_K);
        phase_K ^= 1;
        mbarrier_wait_fn(&mbar_V[0], phase_V);
        phase_V ^= 1;

        if ((threadIdx.x == 0)) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_P[0], 16384);
            asm volatile("cp.async.shared::cta.zero [%0], 4096;\n" :: "r"((uint32_t)__cvta_generic_to_shared(tmem_P)), "r"(threadIdx.x));
        }
        
        fence_proxy_async_fn();
        
        uint64_t desc_Q = make_smem_desc_sm100_fn(smem_Q_0, 0, 0);
        uint64_t desc_K = make_smem_desc_sm100_fn(smem_K_0, 0, 0);
        
        if ((threadIdx.x == 0)) {
            for (int i = 0; i < 4; ++i) {
                umma_f16_cg1_fn(tmem_P, desc_Q + (i * 128 << 4), desc_K + (i * 128 << 4), idesc_P, 1);
            }
            uint64_t desc_Q1 = make_smem_desc_sm100_fn(smem_Q_1, 0, 0);
            uint64_t desc_K1 = make_smem_desc_sm100_fn(smem_K_1, 0, 0);
            for (int i = 0; i < 4; ++i) {
                umma_f16_cg1_fn(tmem_P, desc_Q1 + (i * 128 << 4), desc_K1 + (i * 128 << 4), idesc_P, 1);
            }
            umma_commit_1sm_fn(&mbar_P[0]);
        }
        mbarrier_wait_fn(&mbar_P[0], phase_P);
        phase_P ^= 1;

        float p_val[128];
        for (int i = 0; i < 16; ++i) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(i * 2));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            p_val[i * 8 + threadIdx.x] = __uint_as_float(r0);
            p_val[i * 8 + threadIdx.x + 4] = __uint_as_float(r1);
            p_val[i * 8 + threadIdx.x + 8] = __uint_as_float(r2);
            p_val[i * 8 + threadIdx.x + 12] = __uint_as_float(r3);
        }

        float m_val[128];
        for(int i = 0; i < 128; ++i) m_val[i] = -1e20f;
        for(int i = 0; i < 128; ++i) {
            int real_q_row = q_blk * 128 + threadIdx.x;
            int real_k_col = k_blk * 128 + i;
            bool valid = (real_q_row < S) && (real_k_col < S) && (real_k_col <= real_q_row);
            if (valid) {
                p_val[i] /= 11.3137085f;
                m_val[i] = p_val[i];
            }
        }

        for(int offset = 64; offset > 0; offset /= 2) {
            for(int i = 0; i < 128; ++i) {
                float m2 = __shfl_xor_sync(0xffffffff, m_val[i], offset);
                m_val[i] = fmaxf(m_val[i], m2);
            }
        }

        float sum_P[128];
        for(int i = 0; i < 128; ++i) sum_P[i] = 0.0f;
        for(int i = 0; i < 128; ++i) {
            int real_q_row = q_blk * 128 + threadIdx.x;
            int real_k_col = k_blk * 128 + i;
            bool valid = (real_q_row < S) && (real_k_col < S) && (real_k_col <= real_q_row);
            if (valid) {
                p_val[i] = exp_f(p_val[i] - m_val[i]);
                sum_P[i] = p_val[i];
            } else {
                p_val[i] = 0.0f;
            }
        }

        for(int offset = 64; offset > 0; offset /= 2) {
            for(int i = 0; i < 128; ++i) {
                float s2 = __shfl_xor_sync(0xffffffff, sum_P[i], offset);
                sum_P[i] += s2;
            }
        }

        for(int i = 0; i < 128; ++i) {
            p_val[i] /= sum_P[i];
        }

        for(int i = 0; i < 128; ++i) {
            uint32_t chunk_x = i / 8;
            uint32_t chunk_y = tid % 8;
            uint32_t swizzled_chunk = chunk_x ^ chunk_y;
            uint32_t swizzled_i = (swizzled_chunk * 8) + (i % 8);
            if (i < 64) {
                smem_P_0[threadIdx.x * 64 + swizzled_i] = __float2bfloat16(p_val[i]);
            } else {
                smem_P_1[threadIdx.x * 64 + swizzled_i] = __float2bfloat16(p_val[i]);
            }
        }

        for(int i = 0; i < 128; ++i) {
            int real_q_row = q_blk * 128 + threadIdx.x;
            if (real_q_row < S) {
                float m2 = m_val[i];
                float s2 = sum_P[i];
                int idx = threadIdx.x;
                float old_m = smem_m[idx];
                float new_m = fmaxf(old_m, m2);
                atomicAdd(&smem_sum[idx], s2 * exp_f(m2 - new_m));
                if (old_m != -1e20f) {
                    atomicAdd(&smem_sum[idx], smem_sum[idx] * exp_f(old_m - new_m) - smem_sum[idx] * exp_f(old_m - new_m)); // Placeholder scaling correction
                }
                smem_m[idx] = new_m;
            }
        }

        fence_async_shared_fn();

        if ((threadIdx.x == 0)) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_O[0], 16384);
            asm volatile("cp.async.shared::cta.zero [%0], 4096;\n" :: "r"((uint32_t)__cvta_generic_to_shared(tmem_O_0)), "r"(threadIdx.x));
            asm volatile("cp.async.shared::cta.zero [%0], 4096;\n" :: "r"((uint32_t)__cvta_generic_to_shared(tmem_O_1)), "r"(threadIdx.x));
        }
        
        fence_proxy_async_fn();
        
        uint64_t desc_P_0 = make_smem_desc_sm100_fn(smem_P_0, 0, 0);
        uint64_t desc_V_0 = make_smem_desc_sm100_fn(smem_V_0, 0, 0);
        
        if ((threadIdx.x == 0)) {
            for (int j = 0; j < 4; ++j) {
                uint32_t accum_o = (k_blk == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_O_0, desc_P_0 + (j * 64 << 4), desc_V_0 + (j * 128 << 4), idesc_O, accum_o);
            }
            uint64_t desc_P_1 = make_smem_desc_sm100_fn(smem_P_1, 0, 0);
            uint64_t desc_V_1 = make_smem_desc_sm100_fn(smem_V_1, 0, 0);
            for (int j = 0; j < 4; ++j) {
                uint32_t accum_o = 1;
                umma_f16_cg1_fn(tmem_O_1, desc_P_1 + (j * 64 << 4), desc_V_1 + (j * 128 << 4), idesc_O, accum_o);
            }
            umma_commit_1sm_fn(&mbar_O[0]);
        }
        mbarrier_wait_fn(&mbar_O[0], phase_O);
        phase_O ^= 1;
    }

    float o_val_0[64];
    float o_val_1[64];
    for (int i = 0; i < 8; ++i) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(i * 2));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        o_val_0[i * 8 + threadIdx.x] = __uint_as_float(r0);
        o_val_0[i * 8 + threadIdx.x + 4] = __uint_as_float(r1);
        o_val_1[i * 8 + threadIdx.x] = __uint_as_float(r2);
        o_val_1[i * 8 + threadIdx.x + 4] = __uint_as_float(r3);
    }

    for(int col = 0; col < 64; col+=2) {
        uint32_t packed_0 = pack_bf16_fn(__float_as_uint(o_val_0[col]), __float_as_uint(o_val_0[col+1]));
        uint32_t packed_1 = pack_bf16_fn(__float_as_uint(o_val_1[col]), __float_as_uint(o_val_1[col+1]));
        
        int my_row = q_blk * 128 + threadIdx.x;
        if (my_row < S) {
            uint64_t idx_0 = (bh * S + my_row) * 128 + col;
            uint64_t idx_1 = (bh * S + my_row) * 128 + col + 64;
            *(uint32_t*)(O + idx_0) = packed_0;
            *(uint32_t*)(O + idx_1) = packed_1;
        }
    }

    if (threadIdx.x < 128) {
        int row_idx = q_blk * 128 + threadIdx.x;
        if (row_idx < S) {
            float m = smem_m[threadIdx.x];
            float s = smem_sum[threadIdx.x];
            LSE[bh * S + row_idx] = m + logf(s);
        }
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_P, 128);
        tmem_dealloc_fn(tmem_O_0, 64);
        tmem_dealloc_fn(tmem_O_1, 64);
    }
}

namespace tvm_ffi_mha {

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

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 
    
    CUtensorMap tma_Q, tma_K, tma_V;
    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, q_ptr, D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, k_ptr, D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, v_ptr, D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    int smem_size = 128 * 128 * 2;
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel_sm100, tma_Q, tma_K, tma_V, static_cast<__nv_bfloat16*>(O.data_ptr()), static_cast<float*>(LSE.data_ptr()), S));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(config.stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha