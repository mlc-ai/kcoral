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
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n",                  \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 248;" ::: "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
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

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
     "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
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
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ uint32_t swizzle_128B_addr(uint32_t base_addr, uint32_t row, uint32_t col_elements) {
    uint32_t chunk_x = col_elements / 8;
    uint32_t remainder = col_elements % 8;
    uint32_t swizzled_chunk_x = chunk_x ^ (row % 8);
    uint32_t byte_offset = swizzled_chunk_x * 16 + remainder * 2;
    return base_addr + row * 128 + byte_offset;
}

__device__ __forceinline__ uint64_t make_desc_Q0(uint32_t addr) {
    uint64_t d = 0;
    d |= ((uint64_t)addr & 0x3FFFF) >> 4;
    d |= ((uint64_t)1) << 16;
    d |= ((uint64_t)64) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_desc_V(uint32_t addr) {
    uint64_t d = 0;
    d |= ((uint64_t)addr & 0x3FFFF) >> 4;
    d |= ((uint64_t)128) << 16;
    d |= ((uint64_t)64) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_idesc_QK() {
    uint32_t d = 0;
    d |= (1u << 4);  
    d |= (1u << 7);  
    d |= (1u << 10); 
    d |= (0u << 15);
    d |= (0u << 16);
    d |= (16u << 17);
    d |= (8u << 24);
    return d;
}

__device__ __forceinline__ uint32_t make_idesc_PV() {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (0u << 15);
    d |= (1u << 16);
    d |= (8u << 17);
    d |= (8u << 24);
    return d;
}

extern __shared__ char smem_buf[];

__global__ __launch_bounds__(256, 1) void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE,
    int S, int H) 
{
    setmaxnreg_inc_sync_fn();
    int tid = threadIdx.x;
    int b_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int q_s_idx = blockIdx.x * 128;
    
    int wg = tid / 128; // 0 or 1
    int tid_in_wg = tid % 128; // 0..127
    
    uint32_t base_Q0 = __cvta_generic_to_shared(smem_buf);
    uint32_t base_Q1 = base_Q0 + 16384;
    uint32_t base_K0_0 = base_Q1 + 16384;
    uint32_t base_K1_0 = base_K0_0 + 16384;
    uint32_t base_V0_0 = base_K1_0 + 16384;
    uint32_t base_V1_0 = base_V0_0 + 16384;
    
    uint32_t base_K0_1 = base_V1_0 + 16384;
    uint32_t base_K1_1 = base_K0_1 + 16384;
    uint32_t base_V0_1 = base_K1_1 + 16384;
    uint32_t base_V1_1 = base_V0_1 + 16384;
    
    uint32_t base_P0 = base_V1_1 + 16384;
    uint32_t base_P1 = base_P0 + 16384;
    
    uint64_t* mbar_Q = (uint64_t*)(smem_buf + 196608);
    uint64_t* mbar_K0 = (uint64_t*)(smem_buf + 196616);
    uint64_t* mbar_V0 = (uint64_t*)(smem_buf + 196624);
    uint64_t* mbar_K1 = (uint64_t*)(smem_buf + 196632);
    uint64_t* mbar_V1 = (uint64_t*)(smem_buf + 196640);
    uint64_t* mbar_umma_QK = (uint64_t*)(smem_buf + 196648);
    uint64_t* mbar_umma_PV = (uint64_t*)(smem_buf + 196656);
    uint32_t* ptmem_base = (uint32_t*)(smem_buf + 196664);
    
    float* smem_max = (float*)(smem_buf + 196704);
    float* smem_sum = (float*)(smem_buf + 196704 + 1024);
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar_Q, 256);
        init_smem_barrier_fn(mbar_K0, 256);
        init_smem_barrier_fn(mbar_V0, 256);
        init_smem_barrier_fn(mbar_K1, 256);
        init_smem_barrier_fn(mbar_V1, 256);
        init_smem_barrier_fn(mbar_umma_QK, 1);
        init_smem_barrier_fn(mbar_umma_PV, 1);
    }
    
    if (tid < 32) {
        tmem_alloc_cg1_fn(ptmem_base, 256);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    
    uint32_t tmem_base = *ptmem_base;
    uint32_t tmem_P = tmem_base;
    uint32_t tmem_O_new_0 = tmem_base + 128;
    uint32_t tmem_O_new_1 = tmem_base + 192;
    
    int phase_Q = 0;
    int phase_K[2] = {0, 0};
    int phase_V[2] = {0, 0};
    int phase_umma_QK = 0, phase_umma_PV = 0;
    uint64_t* mbar_K[2] = {mbar_K0, mbar_K1};
    uint64_t* mbar_V[2] = {mbar_V0, mbar_V1};
    uint32_t base_K0_arr[2] = {base_K0_0, base_K0_1};
    uint32_t base_K1_arr[2] = {base_K1_0, base_K1_1};
    uint32_t base_V0_arr[2] = {base_V0_0, base_V0_1};
    uint32_t base_V1_arr[2] = {base_V1_0, base_V1_1};
    
    uint64_t d_Q0 = make_desc_Q0(base_Q0);
    uint64_t d_Q1 = make_desc_Q0(base_Q1);
    uint64_t d_P0 = make_desc_Q0(base_P0);
    uint64_t d_P1 = make_desc_Q0(base_P1);
    uint32_t idesc_QK = make_idesc_QK();
    uint32_t idesc_PV = make_idesc_PV();
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
        tma_load_4d_fn(&tma_Q, mbar_Q, (void*)base_Q0, 0, q_s_idx, h_idx, b_idx);
        tma_load_4d_fn(&tma_Q, mbar_Q, (void*)base_Q1, 64, q_s_idx, h_idx, b_idx);
        
        if (0 < S) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K[0], 32768);
            tma_load_4d_fn(&tma_K, mbar_K[0], (void*)base_K0_arr[0], 0, 0, h_idx, b_idx);
            tma_load_4d_fn(&tma_K, mbar_K[0], (void*)base_K1_arr[0], 64, 0, h_idx, b_idx);

            mbarrier_arrive_and_expect_tx_fn(mbar_V[0], 32768);
            tma_load_4d_fn(&tma_V, mbar_V[0], (void*)base_V0_arr[0], 0, 0, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, mbar_V[0], (void*)base_V1_arr[0], 64, 0, h_idx, b_idx);
        }
    } else {
        mbarrier_arrive_fn(mbar_Q);
        if (0 < S) {
            mbarrier_arrive_fn(mbar_K[0]);
            mbarrier_arrive_fn(mbar_V[0]);
        }
    }
    mbarrier_wait_fn(mbar_Q, phase_Q & 1);
    phase_Q ^= 1;
    
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float O_regs[64];
    #pragma unroll
    for (int i = 0; i < 64; ++i) O_regs[i] = 0.0f;
    
    for (int kv_s_idx = 0; kv_s_idx < S; kv_s_idx += 128) {
        int step_idx = (kv_s_idx / 128);
        int buf_idx = step_idx % 2;
        int next_buf_idx = (step_idx + 1) % 2;
        int next_kv_s_idx = kv_s_idx + 128;

        mbarrier_wait_fn(mbar_K[buf_idx], phase_K[buf_idx] & 1);
        phase_K[buf_idx] ^= 1;
        
        fence_proxy_async_fn();
        
        tcgen05_fence_before_fn();
        if (tid == 0) {
            uint64_t d_K0 = make_desc_Q0(base_K0_arr[buf_idx]);
            uint64_t d_K1 = make_desc_Q0(base_K1_arr[buf_idx]);
            #pragma unroll
            for (int step = 0; step < 4; ++step) {
                uint32_t accum = (step == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_P, d_Q0 + step * 2, d_K0 + step * 2, idesc_QK, accum);
            }
            #pragma unroll
            for (int step = 0; step < 4; ++step) {
                umma_f16_cg1_fn(tmem_P, d_Q1 + step * 2, d_K1 + step * 2, idesc_QK, 1);
            }
            umma_commit_cg1_fn(mbar_umma_QK);
        }
        mbarrier_wait_fn(mbar_umma_QK, phase_umma_QK & 1);
        phase_umma_QK ^= 1;
        tcgen05_fence_after_fn();
        
        bool is_full = (kv_s_idx + 128 <= S);
        float partial_max = -INFINITY;
        int chunk_start = wg * 64;
        
        uint32_t rP[64];
        #pragma unroll
        for (int c = 0; c < 64; c += 8) {
            uint32_t col = (tmem_P & 0xFFFF) + chunk_start + c;
            tmem_load_8x_fn(col, &rP[c], &rP[c+1], &rP[c+2], &rP[c+3], &rP[c+4], &rP[c+5], &rP[c+6], &rP[c+7]);
        }
        tmem_load_fence_fn();
        
        if (is_full) {
            #pragma unroll
            for (int c = 0; c < 64; c++) {
                float p = __uint_as_float(rP[c]) * 0.08838834764f;
                partial_max = fmaxf(partial_max, p);
            }
        } else {
            #pragma unroll
            for (int c = 0; c < 64; c++) {
                float p = __uint_as_float(rP[c]) * 0.08838834764f;
                if (kv_s_idx + chunk_start + c >= S) p = -INFINITY;
                partial_max = fmaxf(partial_max, p);
            }
        }
        
        partial_max = fmaxf(partial_max, -10000.0f);
        smem_max[wg * 128 + tid_in_wg] = partial_max;
        __syncthreads();
        float row_max = fmaxf(smem_max[tid_in_wg], smem_max[128 + tid_in_wg]);
        float m_new = fmaxf(m_prev, row_max);
        float scale = fast_exp2f_fn((m_prev - m_new) * 1.44269504089f);
        
        float partial_sum = 0.0f;
        
        if (is_full) {
            #pragma unroll
            for (int c = 0; c < 64; c += 4) {
                float p0 = __uint_as_float(rP[c]) * 0.08838834764f;
                float p1 = __uint_as_float(rP[c+1]) * 0.08838834764f;
                float p2 = __uint_as_float(rP[c+2]) * 0.08838834764f;
                float p3 = __uint_as_float(rP[c+3]) * 0.08838834764f;
                
                p0 = fast_exp2f_fn((p0 - m_new) * 1.44269504089f);
                p1 = fast_exp2f_fn((p1 - m_new) * 1.44269504089f);
                p2 = fast_exp2f_fn((p2 - m_new) * 1.44269504089f);
                p3 = fast_exp2f_fn((p3 - m_new) * 1.44269504089f);
                
                partial_sum += p0 + p1 + p2 + p3;
                
                uint32_t bf16_01;
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(bf16_01) : "f"(p1), "f"(p0));
                uint32_t bf16_23;
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(bf16_23) : "f"(p3), "f"(p2));
                
                int global_c = chunk_start + c;
                uint32_t smem_addr = (global_c < 64) ? swizzle_128B_addr(base_P0, tid_in_wg, global_c) : swizzle_128B_addr(base_P1, tid_in_wg, global_c - 64);
                asm volatile("st.shared.v2.b32 [%0], {%1, %2};" :: "r"(smem_addr), "r"(bf16_01), "r"(bf16_23));
            }
        } else {
            #pragma unroll
            for (int c = 0; c < 64; c += 4) {
                float p0 = __uint_as_float(rP[c]) * 0.08838834764f;
                float p1 = __uint_as_float(rP[c+1]) * 0.08838834764f;
                float p2 = __uint_as_float(rP[c+2]) * 0.08838834764f;
                float p3 = __uint_as_float(rP[c+3]) * 0.08838834764f;
                
                if (kv_s_idx + chunk_start + c + 0 >= S) p0 = -INFINITY;
                if (kv_s_idx + chunk_start + c + 1 >= S) p1 = -INFINITY;
                if (kv_s_idx + chunk_start + c + 2 >= S) p2 = -INFINITY;
                if (kv_s_idx + chunk_start + c + 3 >= S) p3 = -INFINITY;
                
                p0 = fast_exp2f_fn((p0 - m_new) * 1.44269504089f);
                p1 = fast_exp2f_fn((p1 - m_new) * 1.44269504089f);
                p2 = fast_exp2f_fn((p2 - m_new) * 1.44269504089f);
                p3 = fast_exp2f_fn((p3 - m_new) * 1.44269504089f);
                
                partial_sum += p0 + p1 + p2 + p3;
                
                uint32_t bf16_01;
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(bf16_01) : "f"(p1), "f"(p0));
                uint32_t bf16_23;
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(bf16_23) : "f"(p3), "f"(p2));
                
                int global_c = chunk_start + c;
                uint32_t smem_addr = (global_c < 64) ? swizzle_128B_addr(base_P0, tid_in_wg, global_c) : swizzle_128B_addr(base_P1, tid_in_wg, global_c - 64);
                asm volatile("st.shared.v2.b32 [%0], {%1, %2};" :: "r"(smem_addr), "r"(bf16_01), "r"(bf16_23));
            }
        }
        
        smem_sum[wg * 128 + tid_in_wg] = partial_sum;
        __syncthreads();
        float row_sum = smem_sum[tid_in_wg] + smem_sum[128 + tid_in_wg];
        float l_new = l_prev * scale + row_sum;

        if (next_kv_s_idx < S) {
            if (tid == 0) {
                mbarrier_arrive_and_expect_tx_fn(mbar_K[next_buf_idx], 32768);
                tma_load_4d_fn(&tma_K, mbar_K[next_buf_idx], (void*)base_K0_arr[next_buf_idx], 0, next_kv_s_idx, h_idx, b_idx);
                tma_load_4d_fn(&tma_K, mbar_K[next_buf_idx], (void*)base_K1_arr[next_buf_idx], 64, next_kv_s_idx, h_idx, b_idx);
                
                mbarrier_arrive_and_expect_tx_fn(mbar_V[next_buf_idx], 32768);
                tma_load_4d_fn(&tma_V, mbar_V[next_buf_idx], (void*)base_V0_arr[next_buf_idx], 0, next_kv_s_idx, h_idx, b_idx);
                tma_load_4d_fn(&tma_V, mbar_V[next_buf_idx], (void*)base_V1_arr[next_buf_idx], 64, next_kv_s_idx, h_idx, b_idx);
            } else {
                mbarrier_arrive_fn(mbar_K[next_buf_idx]);
                mbarrier_arrive_fn(mbar_V[next_buf_idx]);
            }
        }
        
        mbarrier_wait_fn(mbar_V[buf_idx], phase_V[buf_idx] & 1);
        phase_V[buf_idx] ^= 1;
        
        fence_async_shared_fn();
        __syncthreads();
        
        tcgen05_fence_before_fn();
        if (tid == 0) {
            uint64_t d_V0 = make_desc_V(base_V0_arr[buf_idx]);
            uint64_t d_V1 = make_desc_V(base_V1_arr[buf_idx]);
            
            #pragma unroll
            for (int step = 0; step < 4; ++step) {
                uint32_t accum = (step == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_O_new_0, d_P0 + step * 2, d_V0 + step * 128, idesc_PV, accum);
            }
            #pragma unroll
            for (int step = 0; step < 4; ++step) {
                umma_f16_cg1_fn(tmem_O_new_0, d_P1 + step * 2, d_V0 + 512 + step * 128, idesc_PV, 1);
            }
            
            #pragma unroll
            for (int step = 0; step < 4; ++step) {
                uint32_t accum = (step == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_O_new_1, d_P0 + step * 2, d_V1 + step * 128, idesc_PV, accum);
            }
            #pragma unroll
            for (int step = 0; step < 4; ++step) {
                umma_f16_cg1_fn(tmem_O_new_1, d_P1 + step * 2, d_V1 + 512 + step * 128, idesc_PV, 1);
            }
            umma_commit_cg1_fn(mbar_umma_PV);
        }
        mbarrier_wait_fn(mbar_umma_PV, phase_umma_PV & 1);
        phase_umma_PV ^= 1;
        tcgen05_fence_after_fn();
        
        uint32_t rO[64];
        #pragma unroll
        for (int c = 0; c < 64; c += 8) {
            uint32_t col = (wg == 0 ? (tmem_O_new_0 & 0xFFFF) : (tmem_O_new_1 & 0xFFFF)) + c;
            tmem_load_8x_fn(col, &rO[c], &rO[c+1], &rO[c+2], &rO[c+3], &rO[c+4], &rO[c+5], &rO[c+6], &rO[c+7]);
        }
        tmem_load_fence_fn();
        
        #pragma unroll
        for (int c = 0; c < 64; c++) {
            O_regs[c] = fmaf(O_regs[c], scale, __uint_as_float(rO[c]));
        }
        
        m_prev = m_new;
        l_prev = l_new;
    }
    
    __syncthreads();
    
    #pragma unroll
    for (int c = 0; c < 64; c += 4) {
        float f0 = O_regs[c+0] / l_prev;
        float f1 = O_regs[c+1] / l_prev;
        float f2 = O_regs[c+2] / l_prev;
        float f3 = O_regs[c+3] / l_prev;
        uint32_t b01;
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b01) : "f"(f1), "f"(f0));
        uint32_t b23;
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b23) : "f"(f3), "f"(f2));
        
        int global_c = (wg == 0) ? c : c + 64;
        uint32_t smem_addr = (global_c < 64) ? swizzle_128B_addr(base_P0, tid_in_wg, global_c) : swizzle_128B_addr(base_P1, tid_in_wg, global_c - 64);
        asm volatile("st.shared.v2.b32 [%0], {%1, %2};" :: "r"(smem_addr), "r"(b01), "r"(b23));
    }
    __syncthreads();
    
    #pragma unroll
    for (int step = 0; step < 128; step += 8) {
        int row = step + (tid / 32);
        int col = (tid % 32) * 4;
        int global_s = q_s_idx + row;
        if (global_s < S) {
            uint32_t smem_addr = (col < 64) ? swizzle_128B_addr(base_P0, row, col) : swizzle_128B_addr(base_P1, row, col - 64);
            uint32_t b01, b23;
            asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];" : "=r"(b01), "=r"(b23) : "r"(smem_addr));
            uint2 data = make_uint2(b01, b23);
            *(uint2*)(&O[b_idx * H * S * 128 + h_idx * S * 128 + global_s * 128 + col]) = data;
        }
    }
    
    if (tid < 128) {
        int global_s = q_s_idx + tid;
        if (global_s < S) {
            LSE[b_idx * H * S + h_idx * S + global_s] = m_prev + logf(l_prev);
        }
    }
    
    if (tid < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 256);
    }
}

namespace tvm_ffi_example_cuda {
void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUtensorMap tma_Q, tma_K, tma_V;
    
    cuuint64_t globalDim_QK[4] = {(cuuint64_t)D, (cuuint64_t)S, (cuuint64_t)H, (cuuint64_t)B};
    cuuint64_t globalStrides_QK[3] = {
        (cuuint64_t)(D * 2), 
        (cuuint64_t)(D * 2 * S), 
        (cuuint64_t)(D * 2 * S * H)
    };
    cuuint32_t boxDim_QK[4] = {64, 128, 1, 1};
    cuuint32_t elemStrides_QK[4] = {1, 1, 1, 1};
    
    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_Q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, Q.data_ptr(),
        globalDim_QK, globalStrides_QK, boxDim_QK, elemStrides_QK,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));
    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_K, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, K.data_ptr(),
        globalDim_QK, globalStrides_QK, boxDim_QK, elemStrides_QK,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));
    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_V, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, V.data_ptr(),
        globalDim_QK, globalStrides_QK, boxDim_QK, elemStrides_QK,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));
    
    dim3 grid((S + 127) / 128, H, B);
    dim3 block(256);
    
    int smem_bytes = 198800;
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    
    CUDA_CHECK(cudaFuncSetAttribute((const void*)mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    
    cudaLaunchKernelEx(&config, mha_kernel, tma_Q, tma_K, tma_V, (__nv_bfloat16*)O.data_ptr(), (float*)LSE.data_ptr(), (int)S, (int)H);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);
}