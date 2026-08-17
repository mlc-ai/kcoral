#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

namespace fa4_sm100 {

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

__device__ __forceinline__ void tma_load_2d_cg1_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_f16_cg1_tmem_A_fn(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cluster.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
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

extern __shared__ __align__(128) uint8_t smem_pool[];

__global__ __launch_bounds__(128) void fa4_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ o_ptr,
    float* __restrict__ lse_ptr,
    int B, int H, int S, int D)
{
    __nv_bfloat16* Q_0_smem = (__nv_bfloat16*)(smem_pool);
    __nv_bfloat16* Q_1_smem = Q_0_smem + 128 * 64;

    __nv_bfloat16* K_0_smem[2];
    K_0_smem[0] = Q_1_smem + 128 * 64;
    K_0_smem[1] = K_0_smem[0] + 128 * 64;

    __nv_bfloat16* K_1_smem[2];
    K_1_smem[0] = K_0_smem[1] + 128 * 64;
    K_1_smem[1] = K_1_smem[0] + 128 * 64;

    __nv_bfloat16* V_0_smem[2];
    V_0_smem[0] = K_1_smem[1] + 128 * 64;
    V_0_smem[1] = V_0_smem[0] + 128 * 64;

    __nv_bfloat16* V_1_smem[2];
    V_1_smem[0] = V_0_smem[1] + 128 * 64;
    V_1_smem[1] = V_1_smem[0] + 128 * 64;

    uint64_t* mbar = (uint64_t*)(V_1_smem[1] + 128 * 64);
    uint64_t* mbar_umma = mbar + 2;

    int b = blockIdx.z;
    int h = blockIdx.y;
    int s_q = blockIdx.x * 128;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar[0], 1);
        init_smem_barrier_fn(&mbar[1], 1);
        init_smem_barrier_fn(&mbar_umma[0], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    __shared__ uint32_t tmem_alloc_smem;
    if (threadIdx.x < 32) { // 1 warp granularity for tmem allocations
        tmem_alloc_cg1_fn(&tmem_alloc_smem, 512);
    }
    __syncthreads();
    uint32_t tmem_alloc_addr = tmem_alloc_smem;

    uint32_t O_tmem_base = tmem_alloc_addr + 0;
    uint32_t S_tmem_base = tmem_alloc_addr + 128;
    uint32_t P_tmem_base = tmem_alloc_addr + 256;

    for (int c = 0; c < 128; c += 4) {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                     :: "r"(0), "r"(0), "r"(0), "r"(0), "r"(O_tmem_base + c));
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float scale = 0.08838834764831843f; // 1.0f / sqrtf(128.0f)

    int buf_idx = 0;
    int phase[2] = {0, 0};
    int phase_umma = 0;

    if (threadIdx.x == 0) {
        int tx_bytes = (128 * 64 * 2) * 6; // Q0, Q1, K0, K1, V0, V1
        mbarrier_arrive_and_expect_tx_fn(&mbar[0], tx_bytes);
        
        int outer_coord = b * H * S + h * S + s_q;
        tma_load_2d_cg1_fn(&tma_Q, &mbar[0], Q_0_smem, 0, outer_coord);
        tma_load_2d_cg1_fn(&tma_Q, &mbar[0], Q_1_smem, 64, outer_coord);
        
        int outer_coord_kv_0 = b * H * S + h * S + 0;
        tma_load_2d_cg1_fn(&tma_K, &mbar[0], K_0_smem[0], 0, outer_coord_kv_0);
        tma_load_2d_cg1_fn(&tma_K, &mbar[0], K_1_smem[0], 64, outer_coord_kv_0);
        tma_load_2d_cg1_fn(&tma_V, &mbar[0], V_0_smem[0], 0, outer_coord_kv_0);
        tma_load_2d_cg1_fn(&tma_V, &mbar[0], V_1_smem[0], 64, outer_coord_kv_0);
    }

    uint32_t idesc_S = 0;
    idesc_S |= (1u << 4) | (1u << 7) | (1u << 10) | (0u << 15) | (0u << 16) | ((128 / 8) << 17) | ((128 / 16) << 24);
    
    uint32_t idesc_O = 0;
    idesc_O |= (1u << 4) | (1u << 7) | (1u << 10) | (0u << 15) | (1u << 16) | ((64 / 8) << 17) | ((128 / 16) << 24);

    for (int n = 0; n < S; n += 128) {
        mbarrier_wait_fn(&mbar[buf_idx], phase[buf_idx]);
        
        int next_n = n + 128;
        int next_buf = 1 - buf_idx;
        if (next_n < S) {
            if (threadIdx.x == 0) {
                int tx_bytes = (128 * 64 * 2) * 4;
                mbarrier_arrive_and_expect_tx_fn(&mbar[next_buf], tx_bytes);
                int outer_coord_kv = b * H * S + h * S + next_n;
                tma_load_2d_cg1_fn(&tma_K, &mbar[next_buf], K_0_smem[next_buf], 0, outer_coord_kv);
                tma_load_2d_cg1_fn(&tma_K, &mbar[next_buf], K_1_smem[next_buf], 64, outer_coord_kv);
                tma_load_2d_cg1_fn(&tma_V, &mbar[next_buf], V_0_smem[next_buf], 0, outer_coord_kv);
                tma_load_2d_cg1_fn(&tma_V, &mbar[next_buf], V_1_smem[next_buf], 64, outer_coord_kv);
            }
        }

        if (threadIdx.x == 0) {
            uint64_t desc_Q0 = make_smem_desc_sm100_fn(Q_0_smem, 1, 1024);
            uint64_t desc_Q1 = make_smem_desc_sm100_fn(Q_1_smem, 1, 1024);
            uint64_t desc_K0 = make_smem_desc_sm100_fn(K_0_smem[buf_idx], 1, 1024);
            uint64_t desc_K1 = make_smem_desc_sm100_fn(K_1_smem[buf_idx], 1, 1024);

            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(S_tmem_base, desc_Q0 + k * 2, desc_K0 + k * 2, idesc_S, (k == 0) ? 0 : 1);
            }
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(S_tmem_base, desc_Q1 + k * 2, desc_K1 + k * 2, idesc_S, 1);
            }
            umma_commit_cg1_fn(&mbar_umma[0]);
        }
        mbarrier_wait_fn(&mbar_umma[0], phase_umma);
        phase_umma ^= 1;

        float row_max = -INFINITY;
        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(S_tmem_base + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            if (n + c >= S) f0 = -INFINITY;
            if (n + c + 1 >= S) f1 = -INFINITY;
            if (n + c + 2 >= S) f2 = -INFINITY;
            if (n + c + 3 >= S) f3 = -INFINITY;
            
            f0 *= scale; f1 *= scale; f2 *= scale; f3 *= scale;
            row_max = fmaxf(row_max, f0);
            row_max = fmaxf(row_max, f1);
            row_max = fmaxf(row_max, f2);
            row_max = fmaxf(row_max, f3);
        }

        float old_max = m_i;
        m_i = fmaxf(m_i, row_max);
        float scale_O = (m_i == -INFINITY) ? 0.0f : expf(old_max - m_i);

        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(O_tmem_base + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float f0 = __uint_as_float(r0) * scale_O;
            float f1 = __uint_as_float(r1) * scale_O;
            float f2 = __uint_as_float(r2) * scale_O;
            float f3 = __uint_as_float(r3) * scale_O;
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                         :: "r"(__float_as_uint(f0)), "r"(__float_as_uint(f1)), 
                            "r"(__float_as_uint(f2)), "r"(__float_as_uint(f3)), "r"(O_tmem_base + c));
        }

        float row_sum = 0;
        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(S_tmem_base + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            if (n + c >= S) f0 = -INFINITY;
            if (n + c + 1 >= S) f1 = -INFINITY;
            if (n + c + 2 >= S) f2 = -INFINITY;
            if (n + c + 3 >= S) f3 = -INFINITY;
            
            f0 = expf(f0 * scale - m_i);
            f1 = expf(f1 * scale - m_i);
            f2 = expf(f2 * scale - m_i);
            f3 = expf(f3 * scale - m_i);
            
            row_sum += f0 + f1 + f2 + f3;
            
            uint32_t bf01 = pack_bf16_fn(__float_as_uint(f0), __float_as_uint(f1));
            uint32_t bf23 = pack_bf16_fn(__float_as_uint(f2), __float_as_uint(f3));
            
            asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32 [%2], {%0,%1};"
                         :: "r"(bf01), "r"(bf23), "r"(P_tmem_base + c / 2));
        }
        l_i = l_i * scale_O + row_sum;
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");

        if (threadIdx.x == 0) {
            uint64_t desc_V0 = make_smem_desc_sm100_fn(V_0_smem[buf_idx], 16384, 1024);
            uint64_t desc_V1 = make_smem_desc_sm100_fn(V_1_smem[buf_idx], 16384, 1024);
            
            for (int k = 0; k < 8; ++k) {
                umma_f16_cg1_tmem_A_fn(O_tmem_base, P_tmem_base + k * 8, desc_V0 + k * 128, idesc_O, 1);
            }
            for (int k = 0; k < 8; ++k) {
                umma_f16_cg1_tmem_A_fn(O_tmem_base + 64, P_tmem_base + k * 8, desc_V1 + k * 128, idesc_O, 1);
            }
            umma_commit_cg1_fn(&mbar_umma[0]);
        }
        mbarrier_wait_fn(&mbar_umma[0], phase_umma);
        phase_umma ^= 1;

        phase[buf_idx] ^= 1;
        buf_idx = next_buf;
    }

    float inv_l = 1.0f / l_i;
    
    __syncthreads();
    __nv_bfloat16* O_smem = (__nv_bfloat16*)smem_pool;
    
    for (int c = 0; c < 128; c += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(O_tmem_base + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        float f0 = __uint_as_float(r0) * inv_l;
        float f1 = __uint_as_float(r1) * inv_l;
        float f2 = __uint_as_float(r2) * inv_l;
        float f3 = __uint_as_float(r3) * inv_l;
        
        uint32_t base = threadIdx.x * 128 + c;
        O_smem[base + 0] = __float2bfloat16(f0);
        O_smem[base + 1] = __float2bfloat16(f1);
        O_smem[base + 2] = __float2bfloat16(f2);
        O_smem[base + 3] = __float2bfloat16(f3);
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = 128 / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= 128) continue;
        uint32_t global_row = s_q + row;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = col_start;
        if (global_row < S && global_col < D) {
            uint2 data = *reinterpret_cast<uint2*>(&O_smem[row * 128 + col_start]);
            uint64_t offset = ((uint64_t)b * H * S + h * S + global_row) * D + global_col;
            *reinterpret_cast<uint2*>(o_ptr + offset) = data;
        }
    }

    int q_idx = s_q + threadIdx.x;
    if (q_idx < S) {
        uint64_t offset = (uint64_t)b * H * S + h * S + q_idx;
        lse_ptr[offset] = m_i + logf(l_i);
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_alloc_addr, 512);
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

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3); 

    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V;
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, q_ptr, D, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, k_ptr, D, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, v_ptr, D, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    dim3 grid((S + 127) / 128, H, B);
    dim3 block(128); 
    
    int smem_bytes = 160 * 1024 + 1024; 
    CUDA_CHECK(cudaFuncSetAttribute(fa4_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    fa4_kernel<<<grid, block, smem_bytes, stream>>>(
        tma_Q, tma_K, tma_V, o_ptr, lse_ptr, B, H, S, D
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, fa4_sm100::run);

}  // namespace fa4_sm100