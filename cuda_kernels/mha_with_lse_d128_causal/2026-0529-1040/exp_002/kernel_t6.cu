#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t smem_dim0, uint32_t smem_dim1, uint32_t smem_dim2, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {smem_dim0, smem_dim1, smem_dim2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3, 
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

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ void tma_store_fence_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.expect_tx.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
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

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
     "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}

__device__ __forceinline__ void tmem_store_8x_fn(uint32_t col, 
    uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3,
    uint32_t r4, uint32_t r5, uint32_t r6, uint32_t r7) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        :: "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3),
           "r"(r4), "r"(r5), "r"(r6), "r"(r7) : "memory");
}

__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_wait_st() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t swizzle_mode) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;
    
    d |= (uint64_t)swizzle_mode << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzle_128B_K_fn(void* smem_ptr) {
    return make_smem_desc_sm100_fn(smem_ptr, 1, 1024, 2); 
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzle_128B_MN_128_fn(void* smem_ptr) {
    return make_smem_desc_sm100_fn(smem_ptr, 16384, 1024, 2); 
}

__device__ __forceinline__ uint32_t make_idesc_QK(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4) | (1u << 7) | (1u << 10);
    d |= (0u << 15); 
    d |= (0u << 16); 
    d |= ((N / 8) << 17);  
    d |= ((M / 16) << 24); 
    return d;
}

__device__ __forceinline__ uint32_t make_idesc_PV_64(uint32_t M) {
    uint32_t d = 0;
    d |= (1u << 4) | (1u << 7) | (1u << 10);
    d |= (0u << 15); 
    d |= (1u << 16); 
    d |= ((64 / 8) << 17);  
    d |= ((M / 16) << 24); 
    return d;
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_f16_cg1_tmemA_fn(uint32_t tmem_d, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(tmem_d), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__global__ void fa_sm100_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* LSE_ptr,
    uint32_t S_len) 
{
    uint32_t m_idx = blockIdx.x * 128;
    if (m_idx >= S_len) return;
    
    uint32_t batch_head = blockIdx.y;

    extern __shared__ __align__(1024) uint8_t smem[];
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)smem;               
    __nv_bfloat16* smem_Q2 = smem_Q1 + 128 * 64;                 
    
    __nv_bfloat16* smem_KV = smem_Q2 + 128 * 64; 
    
    __nv_bfloat16* p_K1[2] = {smem_KV, smem_KV + 128 * 128 * 2};
    __nv_bfloat16* p_K2[2] = {smem_KV + 128 * 64, smem_KV + 128 * 128 * 2 + 128 * 64};
    __nv_bfloat16* p_V1[2] = {smem_KV + 128 * 128, smem_KV + 128 * 128 * 3};
    __nv_bfloat16* p_V2[2] = {smem_KV + 128 * 128 + 128 * 64, smem_KV + 128 * 128 * 3 + 128 * 64};

    uint64_t* mbar_Q = (uint64_t*)(smem_KV + 128 * 128 * 4);
    uint64_t* mbar_KV[2] = {mbar_Q + 1, mbar_Q + 2};
    uint64_t* mbar_umma = mbar_Q + 3;
    uint32_t* tmem_addrs = (uint32_t*)(mbar_umma + 1);

    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_addrs[0], 128); // S (FP32)
        tmem_alloc_fn(&tmem_addrs[1], 128); // O (FP32)
        tmem_alloc_fn(&tmem_addrs[2], 64);  // P (BF16)
    }
    __syncthreads();

    uint32_t tmem_S = tmem_addrs[0];
    uint32_t tmem_O = tmem_addrs[1];
    uint32_t tmem_P = tmem_addrs[2];

    uint32_t phase_kv[2] = {0, 0};
    uint32_t phase_umma = 0;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_KV[0], 1);
        init_smem_barrier_fn(mbar_KV[1], 1);
        init_smem_barrier_fn(mbar_umma, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        mbarrier_expect_tx_fn(mbar_Q, 2 * 64 * 128 * 2);
        tma_load_3d_fn(&tma_Q, mbar_Q, smem_Q1, 0, m_idx, batch_head);
        tma_load_3d_fn(&tma_Q, mbar_Q, smem_Q2, 64, m_idx, batch_head);
        mbarrier_arrive_fn(mbar_Q);
    }
    
    #pragma unroll
    for (uint32_t c = 0; c < 128; c += 16) {
        tmem_store_8x_fn(tmem_O + c, 0, 0, 0, 0, 0, 0, 0, 0);
        tmem_store_8x_fn(tmem_O + c + 8, 0, 0, 0, 0, 0, 0, 0, 0);
    }
    tcgen05_wait_st();
    
    if (threadIdx.x == 0) {
        mbarrier_expect_tx_fn(mbar_KV[0], 4 * 64 * 128 * 2);
        tma_load_3d_fn(&tma_K, mbar_KV[0], p_K1[0], 0, 0, batch_head);
        tma_load_3d_fn(&tma_K, mbar_KV[0], p_K2[0], 64, 0, batch_head);
        tma_load_3d_fn(&tma_V, mbar_KV[0], p_V1[0], 0, 0, batch_head);
        tma_load_3d_fn(&tma_V, mbar_KV[0], p_V2[0], 64, 0, batch_head);
        mbarrier_arrive_fn(mbar_KV[0]);
    }
    
    mbarrier_wait_fn(mbar_Q, 0);

    float thread_m = -INFINITY;
    float thread_l = 0.0f;
    uint32_t idesc_QK = make_idesc_QK(128, 128); 
    uint32_t idesc_PV = make_idesc_PV_64(128); 

    for (uint32_t n_idx = 0; n_idx <= m_idx; n_idx += 128) {
        uint32_t pipe_idx = (n_idx / 128) % 2;
        uint32_t next_pipe_idx = 1 - pipe_idx;
        
        if (n_idx + 128 <= m_idx) {
            if (threadIdx.x == 0) {
                mbarrier_expect_tx_fn(mbar_KV[next_pipe_idx], 4 * 64 * 128 * 2);
                tma_load_3d_fn(&tma_K, mbar_KV[next_pipe_idx], p_K1[next_pipe_idx], 0, n_idx + 128, batch_head);
                tma_load_3d_fn(&tma_K, mbar_KV[next_pipe_idx], p_K2[next_pipe_idx], 64, n_idx + 128, batch_head);
                tma_load_3d_fn(&tma_V, mbar_KV[next_pipe_idx], p_V1[next_pipe_idx], 0, n_idx + 128, batch_head);
                tma_load_3d_fn(&tma_V, mbar_KV[next_pipe_idx], p_V2[next_pipe_idx], 64, n_idx + 128, batch_head);
                mbarrier_arrive_fn(mbar_KV[next_pipe_idx]);
            }
        }
        
        mbarrier_wait_fn(mbar_KV[pipe_idx], phase_kv[pipe_idx]);
        phase_kv[pipe_idx] ^= 1;
        
        if (threadIdx.x == 0) {
            tcgen05_fence_after_fn();
            for (uint32_t k_step = 0; k_step < 64; k_step += 16) {
                uint64_t desc_Q = make_smem_desc_swizzle_128B_K_fn((uint8_t*)smem_Q1 + k_step * 2);
                uint64_t desc_K = make_smem_desc_swizzle_128B_K_fn((uint8_t*)p_K1[pipe_idx] + k_step * 2);
                uint32_t accum = (k_step == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_S, desc_Q, desc_K, idesc_QK, accum);
            }
            for (uint32_t k_step = 0; k_step < 64; k_step += 16) {
                uint64_t desc_Q = make_smem_desc_swizzle_128B_K_fn((uint8_t*)smem_Q2 + k_step * 2);
                uint64_t desc_K = make_smem_desc_swizzle_128B_K_fn((uint8_t*)p_K2[pipe_idx] + k_step * 2);
                umma_f16_cg1_fn(tmem_S, desc_Q, desc_K, idesc_QK, 1);
            }
            umma_commit_cg1_fn(mbar_umma);
        }
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
        
        float row_m = -INFINITY;
        #pragma unroll
        for (uint32_t c = 0; c < 128; c += 16) {
            uint32_t r0[8], r1[8];
            tmem_load_8x_fn(tmem_S + c, &r0[0], &r0[1], &r0[2], &r0[3], &r0[4], &r0[5], &r0[6], &r0[7]);
            tmem_load_8x_fn(tmem_S + c + 8, &r1[0], &r1[1], &r1[2], &r1[3], &r1[4], &r1[5], &r1[6], &r1[7]);
            tcgen05_wait_ld();
            
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float f = __uint_as_float(r0[i]);
                uint32_t global_row = m_idx + threadIdx.x;
                uint32_t global_col = n_idx + c + i;
                if (global_row >= S_len || global_col > global_row) f = -INFINITY;
                else f *= 0.08838834764831843f; 
                row_m = max(row_m, f);
                r0[i] = __float_as_uint(f);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float f = __uint_as_float(r1[i]);
                uint32_t global_row = m_idx + threadIdx.x;
                uint32_t global_col = n_idx + c + 8 + i;
                if (global_row >= S_len || global_col > global_row) f = -INFINITY;
                else f *= 0.08838834764831843f; 
                row_m = max(row_m, f);
                r1[i] = __float_as_uint(f);
            }
            tmem_store_8x_fn(tmem_S + c, r0[0], r0[1], r0[2], r0[3], r0[4], r0[5], r0[6], r0[7]);
            tmem_store_8x_fn(tmem_S + c + 8, r1[0], r1[1], r1[2], r1[3], r1[4], r1[5], r1[6], r1[7]);
        }
        tcgen05_wait_st();
        
        float new_m = max(thread_m, row_m);
        float exp_diff = fast_exp2f_fn((thread_m - new_m) * 1.4426950408889634f);
        
        if (new_m != thread_m && thread_l > 0.0f) {
            #pragma unroll
            for (uint32_t c = 0; c < 128; c += 16) {
                uint32_t o0[4], o1[4], o2[4], o3[4];
                tmem_load_4x_fn(tmem_O + c, &o0[0], &o0[1], &o0[2], &o0[3]);
                tmem_load_4x_fn(tmem_O + c + 4, &o1[0], &o1[1], &o1[2], &o1[3]);
                tmem_load_4x_fn(tmem_O + c + 8, &o2[0], &o2[1], &o2[2], &o2[3]);
                tmem_load_4x_fn(tmem_O + c + 12, &o3[0], &o3[1], &o3[2], &o3[3]);
                tcgen05_wait_ld();
                
                #pragma unroll
                for(int i=0; i<4; ++i) o0[i] = __float_as_uint(__uint_as_float(o0[i]) * exp_diff);
                #pragma unroll
                for(int i=0; i<4; ++i) o1[i] = __float_as_uint(__uint_as_float(o1[i]) * exp_diff);
                #pragma unroll
                for(int i=0; i<4; ++i) o2[i] = __float_as_uint(__uint_as_float(o2[i]) * exp_diff);
                #pragma unroll
                for(int i=0; i<4; ++i) o3[i] = __float_as_uint(__uint_as_float(o3[i]) * exp_diff);
                
                tmem_store_4x_fn(tmem_O + c, o0[0], o0[1], o0[2], o0[3]);
                tmem_store_4x_fn(tmem_O + c + 4, o1[0], o1[1], o1[2], o1[3]);
                tmem_store_4x_fn(tmem_O + c + 8, o2[0], o2[1], o2[2], o2[3]);
                tmem_store_4x_fn(tmem_O + c + 12, o3[0], o3[1], o3[2], o3[3]);
            }
            tcgen05_wait_st();
        }
        
        float row_sum = 0.0f;
        #pragma unroll
        for (uint32_t c = 0; c < 128; c += 16) {
            uint32_t r0[8], r1[8];
            tmem_load_8x_fn(tmem_S + c, &r0[0], &r0[1], &r0[2], &r0[3], &r0[4], &r0[5], &r0[6], &r0[7]);
            tmem_load_8x_fn(tmem_S + c + 8, &r1[0], &r1[1], &r1[2], &r1[3], &r1[4], &r1[5], &r1[6], &r1[7]);
            tcgen05_wait_ld();
            
            uint32_t p0[4], p1[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float f0 = __uint_as_float(r0[i*2]);
                float f1 = __uint_as_float(r0[i*2+1]);
                f0 = fast_exp2f_fn((f0 - new_m) * 1.4426950408889634f);
                f1 = fast_exp2f_fn((f1 - new_m) * 1.4426950408889634f);
                row_sum += f0 + f1;
                p0[i] = pack_bf16_fn(__float_as_uint(f0), __float_as_uint(f1));
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float f0 = __uint_as_float(r1[i*2]);
                float f1 = __uint_as_float(r1[i*2+1]);
                f0 = fast_exp2f_fn((f0 - new_m) * 1.4426950408889634f);
                f1 = fast_exp2f_fn((f1 - new_m) * 1.4426950408889634f);
                row_sum += f0 + f1;
                p1[i] = pack_bf16_fn(__float_as_uint(f0), __float_as_uint(f1));
            }
            tmem_store_4x_fn(tmem_P + c/2, p0[0], p0[1], p0[2], p0[3]);
            tmem_store_4x_fn(tmem_P + c/2 + 4, p1[0], p1[1], p1[2], p1[3]);
        }
        tcgen05_wait_st();
        
        thread_l = thread_l * exp_diff + row_sum;
        thread_m = new_m;
        
        tcgen05_fence_before_fn();
        __syncthreads();
        if (threadIdx.x == 0) {
            tcgen05_fence_after_fn();
            for (uint32_t k_step = 0; k_step < 128; k_step += 16) {
                uint32_t tmem_P_k = tmem_P + (k_step / 2);
                uint64_t desc_V = make_smem_desc_swizzle_128B_MN_128_fn((uint8_t*)p_V1[pipe_idx] + k_step * 128);
                umma_f16_cg1_tmemA_fn(tmem_O, tmem_P_k, desc_V, idesc_PV, 1);
            }
            for (uint32_t k_step = 0; k_step < 128; k_step += 16) {
                uint32_t tmem_P_k = tmem_P + (k_step / 2);
                uint64_t desc_V = make_smem_desc_swizzle_128B_MN_128_fn((uint8_t*)p_V2[pipe_idx] + k_step * 128);
                umma_f16_cg1_tmemA_fn(tmem_O + 64, tmem_P_k, desc_V, idesc_PV, 1);
            }
            umma_commit_cg1_fn(mbar_umma);
        }
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
        __syncthreads();
    }
    
    float inv_l = (thread_l > 0.0f) ? (1.0f / thread_l) : 0.0f;
    #pragma unroll
    for (uint32_t c = 0; c < 64; c += 16) {
        uint32_t o0[8], o1[8];
        tmem_load_8x_fn(tmem_O + c, &o0[0], &o0[1], &o0[2], &o0[3], &o0[4], &o0[5], &o0[6], &o0[7]);
        tmem_load_8x_fn(tmem_O + c + 8, &o1[0], &o1[1], &o1[2], &o1[3], &o1[4], &o1[5], &o1[6], &o1[7]);
        tcgen05_wait_ld();
        #pragma unroll
        for(int i=0; i<8; ++i) smem_Q1[threadIdx.x * 64 + c + i] = __float2bfloat16(__uint_as_float(o0[i]) * inv_l);
        #pragma unroll
        for(int i=0; i<8; ++i) smem_Q1[threadIdx.x * 64 + c + 8 + i] = __float2bfloat16(__uint_as_float(o1[i]) * inv_l);
    }
    #pragma unroll
    for (uint32_t c = 64; c < 128; c += 16) {
        uint32_t o0[8], o1[8];
        tmem_load_8x_fn(tmem_O + c, &o0[0], &o0[1], &o0[2], &o0[3], &o0[4], &o0[5], &o0[6], &o0[7]);
        tmem_load_8x_fn(tmem_O + c + 8, &o1[0], &o1[1], &o1[2], &o1[3], &o1[4], &o1[5], &o1[6], &o1[7]);
        tcgen05_wait_ld();
        #pragma unroll
        for(int i=0; i<8; ++i) smem_Q2[threadIdx.x * 64 + (c - 64) + i] = __float2bfloat16(__uint_as_float(o0[i]) * inv_l);
        #pragma unroll
        for(int i=0; i<8; ++i) smem_Q2[threadIdx.x * 64 + (c - 64) + 8 + i] = __float2bfloat16(__uint_as_float(o1[i]) * inv_l);
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        tma_store_fence_fn();
        tma_store_3d_fn(&tma_O, smem_Q1, 0, m_idx, batch_head);
        tma_store_3d_fn(&tma_O, smem_Q2, 64, m_idx, batch_head);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    
    uint32_t global_row = m_idx + threadIdx.x;
    if (global_row < S_len) {
        float lse_val = thread_m + logf(thread_l);
        LSE_ptr[batch_head * S_len + global_row] = lse_val;
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_S, 128);
        tmem_dealloc_fn(tmem_O, 128);
        tmem_dealloc_fn(tmem_P, 64);
    }
}

namespace tvm_ffi_mha_causal {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, B*H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), D, S, B*H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), D, S, B*H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_3d_descriptor_2B(&tma_O, O.data_ptr(), D, S, B*H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    int grid_x = (S + 127) / 128;
    dim3 grid(grid_x, B * H, 1);
    dim3 block(128, 1, 1);
    
    int smem_size = 196 * 1024; 
    
    CUDA_CHECK(cudaFuncSetAttribute(fa_sm100_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    fa_sm100_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, tma_O, static_cast<float*>(LSE.data_ptr()), S);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_causal::run);

}