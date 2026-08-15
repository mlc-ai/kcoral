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

namespace tvm_ffi_attention_bwd {

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
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
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_128b_swizzle(void* smem_ptr, bool is_mn_major, uint32_t k_dim, uint32_t offset_elements) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t base_offset = (addr >> 7) & 0x7;
    uint32_t sbo = 1024;
    uint32_t lbo = is_mn_major ? (k_dim / 8) * sbo : 0;
    
    uint32_t byte_offset = is_mn_major ? (offset_elements * 2048) : (offset_elements * 32);
    uint32_t final_addr = addr + byte_offset;
    uint32_t final_base_offset = (final_addr >> 7) & 0x7;

    uint64_t d = 0;
    d |= (uint64_t)(final_addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    d |= (uint64_t)final_base_offset << 49;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool a_major, bool b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((a_major ? 1 : 0) << 15);   
    d |= ((b_major ? 1 : 0) << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint32_t swizzle_128B_2B(uint32_t row, uint32_t col) {
    uint32_t x_chunk = col / 8;
    uint32_t rem = col % 8;
    uint32_t y_chunk = row % 8;
    uint32_t swizzled_x = x_chunk ^ y_chunk;
    return (row * 128) + (swizzled_x * 8) + rem;
}

union FloatToBf162 {
    __nv_bfloat162 bf162;
    struct {
        __nv_bfloat16 lo;
        __nv_bfloat16 hi;
    };
};

__global__ void attention_backward_dQ(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    uint32_t S, float scale_factor)
{
    uint32_t batch_head = blockIdx.y;
    uint32_t m_block = blockIdx.x * 128;
    
    extern __shared__ __align__(1024) char smem_raw[];
    __nv_bfloat16* smem_buf = (__nv_bfloat16*)smem_raw;

    __nv_bfloat16* smem_Q = smem_buf;
    __nv_bfloat16* smem_K = smem_Q + 128 * 128;
    __nv_bfloat16* smem_V = smem_K + 128 * 128;
    __nv_bfloat16* smem_dO = smem_V + 128 * 128;
    __nv_bfloat16* smem_O = smem_dO + 128 * 128;
    __nv_bfloat16* smem_dS = smem_O + 128 * 128;
    float* smem_D = (float*)(smem_dS + 128 * 128);
    float* smem_L = smem_D + 128;

    __nv_bfloat16* smem_P = smem_O;

    uint32_t row_idx = batch_head * S + m_block;
    __nv_bfloat16* dq_ptr = dQ;

    __shared__ alignas(16) uint64_t bar_local[2];
    __shared__ uint32_t tmem_dQ, tmem_S, tmem_dP;

    if (threadIdx.x < 128) {
        if (threadIdx.x == 0) {
            init_smem_barrier_fn(&bar_local[0], 1);
            init_smem_barrier_fn(&bar_local[1], 1);
            tmem_alloc_fn(&tmem_dQ, 128);
            tmem_alloc_fn(&tmem_S, 128);
            tmem_alloc_fn(&tmem_dP, 128);
        }
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    uint32_t phase[2] = {0, 0};

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&bar_local[0], 98304);
        tma_load_2d_fn(&tma_Q, &bar_local[0], smem_Q, 0, row_idx);
        tma_load_2d_fn(&tma_Q, &bar_local[0], smem_Q + 8192, 64, row_idx);
        tma_load_2d_fn(&tma_O, &bar_local[0], smem_O, 0, row_idx);
        tma_load_2d_fn(&tma_O, &bar_local[0], smem_O + 8192, 64, row_idx);
        tma_load_2d_fn(&tma_dO, &bar_local[0], smem_dO, 0, row_idx);
        tma_load_2d_fn(&tma_dO, &bar_local[0], smem_dO + 8192, 64, row_idx);
    }
    mbarrier_wait_fn(&bar_local[0], phase[0]);
    phase[0] ^= 1;
    __syncthreads();
    fence_proxy_async_fn();

    if (threadIdx.x < 128) {
        int m = threadIdx.x;
        float d_val = 0;
        for(int c=0; c<128; c++) {
            d_val += __bfloat162float(smem_O[swizzle_128B_2B(m, c)]) * 
                     __bfloat162float(smem_dO[swizzle_128B_2B(m, c)]);
        }
        smem_D[m] = d_val;
        smem_L[m] = (m_block + m < S) ? L[row_idx + m] : 0.0f;
    }
    __syncthreads();

    for (int n_block = 0; n_block < S; n_block += 128) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&bar_local[1], 65536);
            tma_load_2d_fn(&tma_K, &bar_local[1], smem_K, 0, batch_head * S + n_block);
            tma_load_2d_fn(&tma_K, &bar_local[1], smem_K + 8192, 64, batch_head * S + n_block);
            tma_load_2d_fn(&tma_V, &bar_local[1], smem_V, 0, batch_head * S + n_block);
            tma_load_2d_fn(&tma_V, &bar_local[1], smem_V + 8192, 64, batch_head * S + n_block);
        }
        mbarrier_wait_fn(&bar_local[1], phase[1]);
        phase[1] ^= 1;
        __syncthreads();
        fence_proxy_async_fn();

        if (threadIdx.x == 0) {
            for(int i=0; i<8; i++) {
                uint64_t desc_a = make_smem_desc_128b_swizzle(smem_Q, false, 128, i);
                uint64_t desc_b = make_smem_desc_128b_swizzle(smem_K, true, 128, i); 
                uint32_t idesc = make_instr_desc_fn(128, 128, false, true);
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                    :: "r"(tmem_S), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(0));
            }
            umma_commit_1sm_fn(&bar_local[1]);
        }
        mbarrier_wait_fn(&bar_local[1], phase[1]);
        phase[1] ^= 1;
        __syncthreads();

        for(uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float s0 = __uint_as_float(r0);
            float s1 = __uint_as_float(r1);
            float s2 = __uint_as_float(r2);
            float s3 = __uint_as_float(r3);
            
            float l_val = smem_L[threadIdx.x];
            float p0 = __expf(s0 * scale_factor - l_val);
            float p1 = __expf(s1 * scale_factor - l_val);
            float p2 = __expf(s2 * scale_factor - l_val);
            float p3 = __expf(s3 * scale_factor - l_val);
            
            uint32_t idx0 = swizzle_128B_2B(threadIdx.x, col);
            uint32_t idx1 = swizzle_128B_2B(threadIdx.x, col + 1);
            uint32_t idx2 = swizzle_128B_2B(threadIdx.x, col + 2);
            uint32_t idx3 = swizzle_128B_2B(threadIdx.x, col + 3);

            smem_P[idx0] = __float2bfloat16(p0);
            smem_P[idx1] = __float2bfloat16(p1);
            smem_P[idx2] = __float2bfloat16(p2);
            smem_P[idx3] = __float2bfloat16(p3);
        }
        __syncthreads();
        fence_proxy_async_fn();

        if (threadIdx.x == 0) {
            for(int i=0; i<8; i++) {
                uint64_t desc_a = make_smem_desc_128b_swizzle(smem_V, false, 128, i);
                uint64_t desc_b = make_smem_desc_128b_swizzle(smem_dO, true, 128, i); 
                uint32_t idesc = make_instr_desc_fn(128, 128, false, true);
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                    :: "r"(tmem_dP), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(0));
            }
            umma_commit_1sm_fn(&bar_local[1]);
        }
        mbarrier_wait_fn(&bar_local[1], phase[1]);
        phase[1] ^= 1;
        __syncthreads();

        for(uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_dP + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float dp0 = __uint_as_float(r0);
            float dp1 = __uint_as_float(r1);
            float dp2 = __uint_as_float(r2);
            float dp3 = __uint_as_float(r3);
            
            float d_val = smem_D[threadIdx.x];
            
            uint32_t idx0 = swizzle_128B_2B(threadIdx.x, col);
            uint32_t idx1 = swizzle_128B_2B(threadIdx.x, col + 1);
            uint32_t idx2 = swizzle_128B_2B(threadIdx.x, col + 2);
            uint32_t idx3 = swizzle_128B_2B(threadIdx.x, col + 3);

            float p0 = __bfloat162float(smem_P[idx0]);
            float p1 = __bfloat162float(smem_P[idx1]);
            float p2 = __bfloat162float(smem_P[idx2]);
            float p3 = __bfloat162float(smem_P[idx3]);
            
            float ds0 = p0 * (dp0 - d_val) * scale_factor;
            float ds1 = p1 * (dp1 - d_val) * scale_factor;
            float ds2 = p2 * (dp2 - d_val) * scale_factor;
            float ds3 = p3 * (dp3 - d_val) * scale_factor;
            
            smem_dS[idx0] = __float2bfloat16(ds0);
            smem_dS[idx1] = __float2bfloat16(ds1);
            smem_dS[idx2] = __float2bfloat16(ds2);
            smem_dS[idx3] = __float2bfloat16(ds3);
        }
        __syncthreads();
        fence_proxy_async_fn();

        if (threadIdx.x == 0) {
            for(int i=0; i<8; i++) {
                uint64_t desc_a = make_smem_desc_128b_swizzle(smem_dS, false, 128, i);
                uint64_t desc_b = make_smem_desc_128b_swizzle(smem_K, true, 128, i); 
                uint32_t idesc = make_instr_desc_fn(128, 128, false, true);
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                    :: "r"(tmem_dQ), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(n_block == 0 ? 0 : 1));
            }
            umma_commit_1sm_fn(&bar_local[1]);
        }
        mbarrier_wait_fn(&bar_local[1], phase[1]);
        phase[1] ^= 1;
        __syncthreads();
    }

    if (threadIdx.x < 128) {
        int m = threadIdx.x;
        if (m_block + m < S) {
            for(uint32_t col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_dQ + col, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                float dq0 = __uint_as_float(r0);
                float dq1 = __uint_as_float(r1);
                float dq2 = __uint_as_float(r2);
                float dq3 = __uint_as_float(r3);
                
                FloatToBf162 ft;
                ft.lo = __float2bfloat16(dq0);
                ft.hi = __float2bfloat16(dq1);
                *( (__nv_bfloat162*) &dq_ptr[row_idx * 128 + col] ) = ft.bf162;
                
                ft.lo = __float2bfloat16(dq2);
                ft.hi = __float2bfloat16(dq3);
                *( (__nv_bfloat162*) &dq_ptr[row_idx * 128 + col + 2] ) = ft.bf162;
            }
        }
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_dQ, 128);
        tmem_dealloc_fn(tmem_S, 128);
        tmem_dealloc_fn(tmem_dP, 128);
    }
}

__global__ void attention_backward_dK_dV(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    uint32_t S, float scale_factor)
{
    uint32_t batch_head = blockIdx.y;
    uint32_t n_block = blockIdx.x * 128;
    
    extern __shared__ __align__(1024) char smem_raw[];
    __nv_bfloat16* smem_buf = (__nv_bfloat16*)smem_raw;

    __nv_bfloat16* smem_Q = smem_buf;
    __nv_bfloat16* smem_K = smem_Q + 128 * 128;
    __nv_bfloat16* smem_V = smem_K + 128 * 128;
    __nv_bfloat16* smem_dO = smem_V + 128 * 128;
    __nv_bfloat16* smem_O = smem_dO + 128 * 128;
    __nv_bfloat16* smem_P = smem_O + 128 * 128;
    float* smem_D = (float*)(smem_P + 128 * 128);
    float* smem_L = smem_D + 128;

    __nv_bfloat16* smem_dS = smem_O;

    uint32_t row_idx = batch_head * S + n_block;
    __nv_bfloat16* dk_ptr = dK;
    __nv_bfloat16* dv_ptr = dV;

    __shared__ alignas(16) uint64_t bar_local[2];
    __shared__ uint32_t tmem_dK, tmem_dV, tmem_S, tmem_dP;

    if (threadIdx.x < 128) {
        if (threadIdx.x == 0) {
            init_smem_barrier_fn(&bar_local[0], 1);
            init_smem_barrier_fn(&bar_local[1], 1);
            tmem_alloc_fn(&tmem_dK, 128);
            tmem_alloc_fn(&tmem_dV, 128);
            tmem_alloc_fn(&tmem_S, 128);
            tmem_alloc_fn(&tmem_dP, 128);
        }
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    uint32_t phase[2] = {0, 0};

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&bar_local[1], 65536);
        tma_load_2d_fn(&tma_K, &bar_local[1], smem_K, 0, row_idx);
        tma_load_2d_fn(&tma_K, &bar_local[1], smem_K + 8192, 64, row_idx);
        tma_load_2d_fn(&tma_V, &bar_local[1], smem_V, 0, row_idx);
        tma_load_2d_fn(&tma_V, &bar_local[1], smem_V + 8192, 64, row_idx);
    }
    mbarrier_wait_fn(&bar_local[1], phase[1]);
    phase[1] ^= 1;
    __syncthreads();
    fence_proxy_async_fn();

    for (int m_block = 0; m_block < S; m_block += 128) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&bar_local[0], 98304);
            tma_load_2d_fn(&tma_Q, &bar_local[0], smem_Q, 0, batch_head * S + m_block);
            tma_load_2d_fn(&tma_Q, &bar_local[0], smem_Q + 8192, 64, batch_head * S + m_block);
            tma_load_2d_fn(&tma_O, &bar_local[0], smem_O, 0, batch_head * S + m_block);
            tma_load_2d_fn(&tma_O, &bar_local[0], smem_O + 8192, 64, batch_head * S + m_block);
            tma_load_2d_fn(&tma_dO, &bar_local[0], smem_dO, 0, batch_head * S + m_block);
            tma_load_2d_fn(&tma_dO, &bar_local[0], smem_dO + 8192, 64, batch_head * S + m_block);
        }
        mbarrier_wait_fn(&bar_local[0], phase[0]);
        phase[0] ^= 1;
        __syncthreads();
        fence_proxy_async_fn();

        if (threadIdx.x < 128) {
            int m = threadIdx.x;
            float d_val = 0;
            for(int c=0; c<128; c++) {
                d_val += __bfloat162float(smem_O[swizzle_128B_2B(m, c)]) * 
                         __bfloat162float(smem_dO[swizzle_128B_2B(m, c)]);
            }
            smem_D[m] = d_val;
            smem_L[m] = (m_block + m < S) ? L[batch_head * S + m_block + m] : 0.0f;
        }
        __syncthreads();
        fence_proxy_async_fn();

        if (threadIdx.x == 0) {
            for(int i=0; i<8; i++) {
                uint64_t desc_a = make_smem_desc_128b_swizzle(smem_Q, false, 128, i);
                uint64_t desc_b = make_smem_desc_128b_swizzle(smem_K, true, 128, i); 
                uint32_t idesc = make_instr_desc_fn(128, 128, false, true);
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                    :: "r"(tmem_S), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(0));
            }
            umma_commit_1sm_fn(&bar_local[0]);
        }
        mbarrier_wait_fn(&bar_local[0], phase[0]);
        phase[0] ^= 1;
        __syncthreads();

        for(uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float s0 = __uint_as_float(r0);
            float s1 = __uint_as_float(r1);
            float s2 = __uint_as_float(r2);
            float s3 = __uint_as_float(r3);
            
            float l_val = smem_L[threadIdx.x];
            float p0 = __expf(s0 * scale_factor - l_val);
            float p1 = __expf(s1 * scale_factor - l_val);
            float p2 = __expf(s2 * scale_factor - l_val);
            float p3 = __expf(s3 * scale_factor - l_val);
            
            uint32_t idx0 = swizzle_128B_2B(threadIdx.x, col);
            uint32_t idx1 = swizzle_128B_2B(threadIdx.x, col + 1);
            uint32_t idx2 = swizzle_128B_2B(threadIdx.x, col + 2);
            uint32_t idx3 = swizzle_128B_2B(threadIdx.x, col + 3);

            smem_P[idx0] = __float2bfloat16(p0);
            smem_P[idx1] = __float2bfloat16(p1);
            smem_P[idx2] = __float2bfloat16(p2);
            smem_P[idx3] = __float2bfloat16(p3);
        }
        __syncthreads();
        fence_proxy_async_fn();

        if (threadIdx.x == 0) {
            for(int i=0; i<8; i++) {
                uint64_t desc_a = make_smem_desc_128b_swizzle(smem_V, false, 128, i);
                uint64_t desc_b = make_smem_desc_128b_swizzle(smem_dO, true, 128, i); 
                uint32_t idesc = make_instr_desc_fn(128, 128, false, true);
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                    :: "r"(tmem_dP), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(0));
            }
            umma_commit_1sm_fn(&bar_local[0]);
        }
        mbarrier_wait_fn(&bar_local[0], phase[0]);
        phase[0] ^= 1;
        __syncthreads();

        for(uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_dP + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float dp0 = __uint_as_float(r0);
            float dp1 = __uint_as_float(r1);
            float dp2 = __uint_as_float(r2);
            float dp3 = __uint_as_float(r3);
            
            float d_val = smem_D[threadIdx.x];
            
            uint32_t idx0 = swizzle_128B_2B(threadIdx.x, col);
            uint32_t idx1 = swizzle_128B_2B(threadIdx.x, col + 1);
            uint32_t idx2 = swizzle_128B_2B(threadIdx.x, col + 2);
            uint32_t idx3 = swizzle_128B_2B(threadIdx.x, col + 3);

            float p0 = __bfloat162float(smem_P[idx0]);
            float p1 = __bfloat162float(smem_P[idx1]);
            float p2 = __bfloat162float(smem_P[idx2]);
            float p3 = __bfloat162float(smem_P[idx3]);
            
            float ds0 = p0 * (dp0 - d_val) * scale_factor;
            float ds1 = p1 * (dp1 - d_val) * scale_factor;
            float ds2 = p2 * (dp2 - d_val) * scale_factor;
            float ds3 = p3 * (dp3 - d_val) * scale_factor;
            
            smem_dS[idx0] = __float2bfloat16(ds0);
            smem_dS[idx1] = __float2bfloat16(ds1);
            smem_dS[idx2] = __float2bfloat16(ds2);
            smem_dS[idx3] = __float2bfloat16(ds3);
        }
        __syncthreads();
        fence_proxy_async_fn();

        if (threadIdx.x == 0) {
            for(int i=0; i<8; i++) {
                uint64_t desc_a = make_smem_desc_128b_swizzle(smem_dS, true, 128, i); 
                uint64_t desc_b = make_smem_desc_128b_swizzle(smem_Q, true, 128, i); 
                uint32_t idesc = make_instr_desc_fn(128, 128, true, true); 
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                    :: "r"(tmem_dK), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(m_block == 0 ? 0 : 1));
            }
            
            for(int i=0; i<8; i++) {
                uint64_t desc_a = make_smem_desc_128b_swizzle(smem_P, true, 128, i); 
                uint64_t desc_b = make_smem_desc_128b_swizzle(smem_dO, true, 128, i); 
                uint32_t idesc = make_instr_desc_fn(128, 128, true, true); 
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                    :: "r"(tmem_dV), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(m_block == 0 ? 0 : 1));
            }
            umma_commit_1sm_fn(&bar_local[0]);
        }
        mbarrier_wait_fn(&bar_local[0], phase[0]);
        phase[0] ^= 1;
        __syncthreads();
    }

    if (threadIdx.x < 128) {
        int m = threadIdx.x;
        if (n_block + m < S) {
            for(uint32_t col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_dK + col, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                float dk0 = __uint_as_float(r0);
                float dk1 = __uint_as_float(r1);
                float dk2 = __uint_as_float(r2);
                float dk3 = __uint_as_float(r3);
                
                FloatToBf162 ft;
                ft.lo = __float2bfloat16(dk0);
                ft.hi = __float2bfloat16(dk1);
                *( (__nv_bfloat162*) &dk_ptr[row_idx * 128 + col] ) = ft.bf162;
                
                ft.lo = __float2bfloat16(dk2);
                ft.hi = __float2bfloat16(dk3);
                *( (__nv_bfloat162*) &dk_ptr[row_idx * 128 + col + 2] ) = ft.bf162;

                tmem_load_4x_fn(tmem_dV + col, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                float dv0 = __uint_as_float(r0);
                float dv1 = __uint_as_float(r1);
                float dv2 = __uint_as_float(r2);
                float dv3 = __uint_as_float(r3);
                
                ft.lo = __float2bfloat16(dv0);
                ft.hi = __float2bfloat16(dv1);
                *( (__nv_bfloat162*) &dv_ptr[row_idx * 128 + col] ) = ft.bf162;
                
                ft.lo = __float2bfloat16(dv2);
                ft.hi = __float2bfloat16(dv3);
                *( (__nv_bfloat162*) &dv_ptr[row_idx * 128 + col + 2] ) = ft.bf162;
            }
        }
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_dK, 128);
        tmem_dealloc_fn(tmem_dV, 128);
        tmem_dealloc_fn(tmem_S, 128);
        tmem_dealloc_fn(tmem_dP, 128);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3); 
    
    float scale_factor = 1.0f / sqrtf((float)d);
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), d, B*H*S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), d, B*H*S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), d, B*H*S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_O, O.data_ptr(), d, B*H*S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dO, dO.data_ptr(), d, B*H*S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    
    size_t shmem = 220000;
    CUDA_CHECK(cudaFuncSetAttribute(attention_backward_dQ, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
    CUDA_CHECK(cudaFuncSetAttribute(attention_backward_dK_dV, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));

    attention_backward_dQ<<<grid, block, shmem, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO,
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        S, scale_factor);
    
    attention_backward_dK_dV<<<grid, block, shmem, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO,
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S, scale_factor);
    
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_attention_bwd::run);

} // namespace tvm_ffi_attention_bwd