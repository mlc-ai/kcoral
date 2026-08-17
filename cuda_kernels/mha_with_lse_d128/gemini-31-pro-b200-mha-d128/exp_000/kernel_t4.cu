#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <math.h>

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

namespace tvm_ffi_flash_attn_sm100 {

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
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

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float fast_expf(float x) {
    return fast_exp2f_fn(x * 1.4426950408889634f);
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t get_desc_Q(void* smem_Q, int k) {
    uint32_t block_idx = k / 4;
    uint32_t k_offset = (k % 4) * 32;
    char* ptr = (char*)smem_Q + block_idx * 16384 + k_offset;
    return make_smem_desc_sm100_fn(ptr, 1, 1024);
}

__device__ __forceinline__ uint64_t get_desc_K(void* smem_K, int k) {
    uint32_t block_idx = k / 4;
    uint32_t k_offset = (k % 4) * 32;
    char* ptr = (char*)smem_K + block_idx * 16384 + k_offset;
    return make_smem_desc_sm100_fn(ptr, 1, 1024);
}

__device__ __forceinline__ uint64_t get_desc_P(void* smem_P, int k) {
    uint32_t block_idx = k / 4;
    uint32_t k_offset = (k % 4) * 32;
    char* ptr = (char*)smem_P + block_idx * 16384 + k_offset;
    return make_smem_desc_sm100_fn(ptr, 1, 1024);
}

__device__ __forceinline__ uint64_t get_desc_V(void* smem_V, int k) {
    char* ptr = (char*)smem_V + k * 2048; 
    return make_smem_desc_sm100_fn(ptr, 2048, 1024);
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
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
        :: "r"(a) : "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void store_P_swizzled(uint32_t row, uint32_t c, float f0, float f1, float f2, float f3, __nv_bfloat16* smem_P) {
    int block_idx = c / 64;
    int x = (c % 64) / 8;
    int new_x = (row % 8) ^ x;
    int swizzled_c = new_x * 8 + (c % 8);
    int smem_idx = block_idx * 128 * 64 + row * 64 + swizzled_c;
    
    uint32_t out0 = pack_bf16_fn(__float_as_uint(f0), __float_as_uint(f1));
    uint32_t out1 = pack_bf16_fn(__float_as_uint(f2), __float_as_uint(f3));
    *reinterpret_cast<uint2*>(&smem_P[smem_idx]) = make_uint2(out0, out1);
}

__global__ void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int H, int B
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int m_block = blockIdx.x;

    __shared__ __align__(1024) __nv_bfloat16 smem_Q[2][128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 smem_K[2][128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 smem_V[2][128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 smem_P[2][128 * 64];

    __shared__ __align__(16) uint64_t mbar_Q;
    __shared__ __align__(16) uint64_t mbar_K;
    __shared__ __align__(16) uint64_t mbar_V;
    __shared__ __align__(16) uint64_t mbar_UMMA_S;
    __shared__ __align__(16) uint64_t mbar_UMMA_O;
    __shared__ uint32_t smem_tmem_addr;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_Q, 1);
        init_smem_barrier_fn(&mbar_K, 1);
        init_smem_barrier_fn(&mbar_V, 1);
        init_smem_barrier_fn(&mbar_UMMA_S, 1);
        init_smem_barrier_fn(&mbar_UMMA_O, 1);
    }
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&smem_tmem_addr, 256);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();

    uint32_t tmem_base = smem_tmem_addr;
    uint32_t tmem_O_left = tmem_base;
    uint32_t tmem_O_right = tmem_base + 64;
    uint32_t tmem_S = tmem_base + 128;

    uint32_t idesc_QK = (1u << 4) | (1u << 7) | (1u << 10) | (0u << 15) | (0u << 16) | ((128 / 8) << 17) | ((128 / 16) << 24);
    uint32_t idesc_PV_half = (1u << 4) | (1u << 7) | (1u << 10) | (0u << 15) | (1u << 16) | (8u << 17) | (8u << 24);

    int row_offset_Q = (b * H + h) * S + m_block * 128;
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_Q, 32768);
        tma_load_2d_fn(&tma_Q, &mbar_Q, smem_Q[0], 0, row_offset_Q);
        tma_load_2d_fn(&tma_Q, &mbar_Q, smem_Q[1], 64, row_offset_Q);
    }
    mbarrier_wait_fn(&mbar_Q, 0);

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    int umma_phase_S = 0;
    int umma_phase_O = 0;
    
    int total_kv_blocks = (S + 127) / 128;

    for (int kv_block = 0; kv_block < total_kv_blocks; ++kv_block) {
        int row_offset_K = (b * H + h) * S + kv_block * 128;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_K, 32768);
            tma_load_2d_fn(&tma_K, &mbar_K, smem_K[0], 0, row_offset_K);
            tma_load_2d_fn(&tma_K, &mbar_K, smem_K[1], 64, row_offset_K);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar_V, 32768);
            tma_load_2d_fn(&tma_V, &mbar_V, smem_V[0], 0, row_offset_K);
            tma_load_2d_fn(&tma_V, &mbar_V, smem_V[1], 64, row_offset_K);
        }
        
        mbarrier_wait_fn(&mbar_K, kv_block & 1);
        mbarrier_wait_fn(&mbar_V, kv_block & 1);
        
        fence_proxy_async_fn();

        if (threadIdx.x == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for (int k = 0; k < 8; ++k) {
                uint64_t desc_A = get_desc_Q(smem_Q, k);
                uint64_t desc_B = get_desc_K(smem_K, k);
                umma_f16_cg1_fn(tmem_S, desc_A, desc_B, idesc_QK, (k > 0));
            }
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            umma_commit_cg1_fn(&mbar_UMMA_S);
        }
        
        mbarrier_wait_fn(&mbar_UMMA_S, umma_phase_S);
        umma_phase_S ^= 1;
        
        float row_max = -INFINITY;
        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_S + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float f0 = __uint_as_float(r0) * 0.08838834764f; 
            float f1 = __uint_as_float(r1) * 0.08838834764f;
            float f2 = __uint_as_float(r2) * 0.08838834764f;
            float f3 = __uint_as_float(r3) * 0.08838834764f;
            
            int global_k = kv_block * 128 + c;
            if (global_k + 0 >= S) f0 = -INFINITY;
            if (global_k + 1 >= S) f1 = -INFINITY;
            if (global_k + 2 >= S) f2 = -INFINITY;
            if (global_k + 3 >= S) f3 = -INFINITY;
            
            row_max = fmaxf(row_max, f0);
            row_max = fmaxf(row_max, f1);
            row_max = fmaxf(row_max, f2);
            row_max = fmaxf(row_max, f3);
        }

        float m_new = fmaxf(m_prev, row_max);
        float scale_O = (m_prev == -INFINITY) ? 0.0f : fast_expf(m_prev - m_new);

        float row_sum = 0.0f;
        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_S + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float f0 = __uint_as_float(r0) * 0.08838834764f;
            float f1 = __uint_as_float(r1) * 0.08838834764f;
            float f2 = __uint_as_float(r2) * 0.08838834764f;
            float f3 = __uint_as_float(r3) * 0.08838834764f;
            
            int global_k = kv_block * 128 + c;
            if (global_k + 0 >= S) f0 = -INFINITY;
            if (global_k + 1 >= S) f1 = -INFINITY;
            if (global_k + 2 >= S) f2 = -INFINITY;
            if (global_k + 3 >= S) f3 = -INFINITY;
            
            f0 = fast_expf(f0 - m_new);
            f1 = fast_expf(f1 - m_new);
            f2 = fast_expf(f2 - m_new);
            f3 = fast_expf(f3 - m_new);
            
            row_sum += f0 + f1 + f2 + f3;
            
            store_P_swizzled(threadIdx.x, c, f0, f1, f2, f3, (__nv_bfloat16*)smem_P);
        }

        int need_scale = __any_sync(0xFFFFFFFF, scale_O != 1.0f);
        if (need_scale && kv_block > 0) {
            for (int c = 0; c < 128; c += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_O_left + c));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float f0 = __uint_as_float(r0) * scale_O;
                float f1 = __uint_as_float(r1) * scale_O;
                float f2 = __uint_as_float(r2) * scale_O;
                float f3 = __uint_as_float(r3) * scale_O;
                
                r0 = __float_as_uint(f0);
                r1 = __float_as_uint(f1);
                r2 = __float_as_uint(f2);
                r3 = __float_as_uint(f3);
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                    :: "r"(tmem_O_left + c), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }

        l_prev = l_prev * scale_O + row_sum;
        m_prev = m_new;

        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        __syncthreads();
        fence_proxy_async_fn();

        if (threadIdx.x == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for (int k = 0; k < 8; ++k) {
                uint64_t desc_A = get_desc_P(smem_P, k);
                uint64_t desc_B_left = get_desc_V(smem_V[0], k);
                uint64_t desc_B_right = get_desc_V(smem_V[1], k);
                
                umma_f16_cg1_fn(tmem_O_left, desc_A, desc_B_left, idesc_PV_half, (kv_block > 0 || k > 0));
                umma_f16_cg1_fn(tmem_O_right, desc_A, desc_B_right, idesc_PV_half, (kv_block > 0 || k > 0));
            }
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            umma_commit_cg1_fn(&mbar_UMMA_O);
        }
        
        mbarrier_wait_fn(&mbar_UMMA_O, umma_phase_O);
        umma_phase_O ^= 1;
        __syncthreads();
    }

    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    __nv_bfloat16* smem_O = (__nv_bfloat16*)smem_Q;
    for (int c = 0; c < 128; c += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_O_left + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0) * inv_l;
        float f1 = __uint_as_float(r1) * inv_l;
        float f2 = __uint_as_float(r2) * inv_l;
        float f3 = __uint_as_float(r3) * inv_l;
        
        uint32_t base = threadIdx.x * 128 + c;
        uint32_t out0 = pack_bf16_fn(__float_as_uint(f0), __float_as_uint(f1));
        uint32_t out1 = pack_bf16_fn(__float_as_uint(f2), __float_as_uint(f3));
        *reinterpret_cast<uint2*>(&smem_O[base]) = make_uint2(out0, out1);
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    for (uint32_t step = 0; step < 32; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= 128) continue;
        uint32_t global_row = m_block * 128 + row;
        uint32_t col_start = lane_id * 4;
        
        if (global_row < S) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_O[row * 128 + col_start]);
            *reinterpret_cast<uint2*>(O + (b * H + h) * S * 128 + global_row * 128 + col_start) = data;
        }
    }

    if (threadIdx.x < 128) {
        int row = threadIdx.x;
        int global_row = m_block * 128 + row;
        if (global_row < S) {
            float final_lse = (m_prev == -INFINITY) ? -INFINITY : (m_prev + logf(l_prev));
            LSE[(b * H + h) * S + global_row] = final_lse;
        }
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 256);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    CUtensorMap tma_Q, tma_K, tma_V;
    uint64_t rows = B * H * S;
    uint64_t cols = D;
    
    CUresult res;
    res = create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), cols, rows, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA Q failed\n"); exit(1); }
    
    res = create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), cols, rows, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA K failed\n"); exit(1); }
    
    res = create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), cols, rows, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA V failed\n"); exit(1); }
    
    int64_t m_blocks = (S + 127) / 128;
    dim3 grid(m_blocks, H, B);
    dim3 block(128);
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_kernel<<<grid, block, 0, stream>>>(tma_Q, tma_K, tma_V, static_cast<__nv_bfloat16*>(O.data_ptr()), static_cast<float*>(LSE.data_ptr()), S, H, B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_flash_attn_sm100::run);

}