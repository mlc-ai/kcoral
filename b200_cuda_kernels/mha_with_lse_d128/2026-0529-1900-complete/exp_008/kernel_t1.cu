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


__device__ __forceinline__ void setmaxnreg_inc_sync_fn(int num) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 248;" ::: "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ void tma_load_5d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3, int32_t c4) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4) : "memory");
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

__device__ __forceinline__ uint64_t make_desc_Q0(uint32_t addr) {
    uint64_t d = 0;
    d |= ((uint64_t)addr & 0x3FFFF) >> 4;
    d |= ((uint64_t)1) << 16;
    d |= ((uint64_t)64) << 32; // 1024 >> 4 = 64
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t make_desc_V(uint32_t addr) {
    uint64_t d = 0;
    d |= ((uint64_t)addr & 0x3FFFF) >> 4;
    d |= ((uint64_t)1024) << 16; // 16384 >> 4 = 1024
    d |= ((uint64_t)64) << 32;   // 1024 >> 4 = 64
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t make_desc_P(uint32_t addr) {
    uint64_t d = 0;
    d |= ((uint64_t)addr & 0x3FFFF) >> 4;
    d |= ((uint64_t)1) << 16;     // LBO = 1
    d |= ((uint64_t)128) << 32;   // SBO = 128 (2048 >> 4)
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 61; // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint32_t make_idesc_QK() {
    uint32_t d = 0;
    d |= (1u << 4);  
    d |= (1u << 7);  
    d |= (1u << 10); 
    d |= (0u << 15); 
    d |= (0u << 16); 
    d |= (16u << 17); // N = 128 (128/8 = 16)
    d |= (8u << 24);  // M = 128 (128/16 = 8)
    return d;
}

__device__ __forceinline__ uint32_t make_idesc_PV() {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (0u << 15); 
    d |= (1u << 16); // V MN-Major
    d |= (8u << 17); // N = 64 (64/8 = 8)
    d |= (8u << 24); // M = 128 (128/16 = 8)
    return d;
}

extern __shared__ char smem_buf[];

__global__ void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE,
    int S, int H) 
{
    setmaxnreg_inc_sync_fn(248);
    int tid = threadIdx.x;
    int b_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int q_s_idx = blockIdx.x * 128;
    
    __nv_bfloat16 (*smem_Q0)[64] = (__nv_bfloat16 (*)[64])(smem_buf);
    __nv_bfloat16 (*smem_Q1)[64] = (__nv_bfloat16 (*)[64])(smem_buf + 16384);
    __nv_bfloat16 (*smem_K0)[64] = (__nv_bfloat16 (*)[64])(smem_buf + 32768);
    __nv_bfloat16 (*smem_K1)[64] = (__nv_bfloat16 (*)[64])(smem_buf + 49152);
    __nv_bfloat16 (*smem_V0)[64] = (__nv_bfloat16 (*)[64])(smem_buf + 65536);
    __nv_bfloat16 (*smem_V1)[64] = (__nv_bfloat16 (*)[64])(smem_buf + 81920);
    __nv_bfloat16 (*smem_P)[128] = (__nv_bfloat16 (*)[128])(smem_buf + 98304);
    
    uint64_t* mbar_Q = (uint64_t*)(smem_buf + 133120);
    uint64_t* mbar_K = (uint64_t*)(smem_buf + 133128);
    uint64_t* mbar_V = (uint64_t*)(smem_buf + 133136);
    uint64_t* mbar_umma_QK = (uint64_t*)(smem_buf + 133144);
    uint64_t* mbar_umma_PV = (uint64_t*)(smem_buf + 133152);
    uint32_t* ptmem_base = (uint32_t*)(smem_buf + 133160);
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar_Q, 128);
        init_smem_barrier_fn(mbar_K, 128);
        init_smem_barrier_fn(mbar_V, 128);
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
    
    int phase_Q = 0, phase_K = 0, phase_V = 0;
    int phase_umma_QK = 0, phase_umma_PV = 0;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
        tma_load_5d_fn(&tma_Q, mbar_Q, smem_Q0, 0, 0, q_s_idx, h_idx, b_idx);
        tma_load_5d_fn(&tma_Q, mbar_Q, smem_Q1, 0, 1, q_s_idx, h_idx, b_idx);
    } else {
        mbarrier_arrive_fn(mbar_Q);
    }
    mbarrier_wait_fn(mbar_Q, phase_Q & 1);
    phase_Q ^= 1;
    
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float O_regs[128];
    for (int i = 0; i < 128; ++i) O_regs[i] = 0.0f;
    
    for (int kv_s_idx = 0; kv_s_idx < S; kv_s_idx += 128) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K, 32768);
            tma_load_5d_fn(&tma_K, mbar_K, smem_K0, 0, 0, kv_s_idx, h_idx, b_idx);
            tma_load_5d_fn(&tma_K, mbar_K, smem_K1, 0, 1, kv_s_idx, h_idx, b_idx);
            
            mbarrier_arrive_and_expect_tx_fn(mbar_V, 32768);
            tma_load_5d_fn(&tma_V, mbar_V, smem_V0, 0, 0, kv_s_idx, h_idx, b_idx);
            tma_load_5d_fn(&tma_V, mbar_V, smem_V1, 0, 1, kv_s_idx, h_idx, b_idx);
        } else {
            mbarrier_arrive_fn(mbar_K);
            mbarrier_arrive_fn(mbar_V);
        }
        mbarrier_wait_fn(mbar_K, phase_K & 1);
        mbarrier_wait_fn(mbar_V, phase_V & 1);
        phase_K ^= 1;
        phase_V ^= 1;
        
        fence_proxy_async_fn();
        
        tcgen05_fence_before_fn();
        if (tid == 0) {
            for (int step = 0; step < 4; ++step) {
                uint32_t q0_addr = __cvta_generic_to_shared(smem_Q0) + step * 32;
                uint32_t k0_addr = __cvta_generic_to_shared(smem_K0) + step * 32;
                uint32_t accum = (step == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_P, make_desc_Q0(q0_addr), make_desc_Q0(k0_addr), make_idesc_QK(), accum);
            }
            for (int step = 0; step < 4; ++step) {
                uint32_t q1_addr = __cvta_generic_to_shared(smem_Q1) + step * 32;
                uint32_t k1_addr = __cvta_generic_to_shared(smem_K1) + step * 32;
                umma_f16_cg1_fn(tmem_P, make_desc_Q0(q1_addr), make_desc_Q0(k1_addr), make_idesc_QK(), 1);
            }
            umma_commit_cg1_fn(mbar_umma_QK);
        }
        mbarrier_wait_fn(mbar_umma_QK, phase_umma_QK & 1);
        phase_umma_QK ^= 1;
        tcgen05_fence_after_fn();
        
        float row_max = -INFINITY;
        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            uint32_t col = (tmem_P & 0xFFFF) + c;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float p0 = __uint_as_float(r0) * 0.08838834764f;
            float p1 = __uint_as_float(r1) * 0.08838834764f;
            float p2 = __uint_as_float(r2) * 0.08838834764f;
            float p3 = __uint_as_float(r3) * 0.08838834764f;
            
            if (kv_s_idx + c + 0 >= S) p0 = -INFINITY;
            if (kv_s_idx + c + 1 >= S) p1 = -INFINITY;
            if (kv_s_idx + c + 2 >= S) p2 = -INFINITY;
            if (kv_s_idx + c + 3 >= S) p3 = -INFINITY;
            
            row_max = fmaxf(row_max, fmaxf(fmaxf(p0, p1), fmaxf(p2, p3)));
        }
        row_max = fmaxf(row_max, -10000.0f);
        float m_new = fmaxf(m_prev, row_max);
        float scale = fast_exp2f_fn((m_prev - m_new) * 1.44269504089f);
        
        float row_sum = 0.0f;
        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            uint32_t col = (tmem_P & 0xFFFF) + c;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float p0 = __uint_as_float(r0) * 0.08838834764f;
            float p1 = __uint_as_float(r1) * 0.08838834764f;
            float p2 = __uint_as_float(r2) * 0.08838834764f;
            float p3 = __uint_as_float(r3) * 0.08838834764f;
            
            if (kv_s_idx + c + 0 >= S) p0 = -INFINITY;
            if (kv_s_idx + c + 1 >= S) p1 = -INFINITY;
            if (kv_s_idx + c + 2 >= S) p2 = -INFINITY;
            if (kv_s_idx + c + 3 >= S) p3 = -INFINITY;
            
            p0 = fast_exp2f_fn((p0 - m_new) * 1.44269504089f);
            p1 = fast_exp2f_fn((p1 - m_new) * 1.44269504089f);
            p2 = fast_exp2f_fn((p2 - m_new) * 1.44269504089f);
            p3 = fast_exp2f_fn((p3 - m_new) * 1.44269504089f);
            
            row_sum += p0 + p1 + p2 + p3;
            
            uint32_t smem_addr = __cvta_generic_to_shared(&smem_P[tid][c]);
            uint32_t bf16_01 = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
            uint32_t bf16_23 = pack_bf16_fn(__float_as_uint(p2), __float_as_uint(p3));
            asm volatile("st.shared.v2.b32 [%0], {%1, %2};" :: "r"(smem_addr), "r"(bf16_01), "r"(bf16_23));
        }
        
        float l_new = l_prev * scale + row_sum;
        
        fence_async_shared_fn();
        __syncthreads();
        
        tcgen05_fence_before_fn();
        if (tid == 0) {
            for (int step = 0; step < 8; ++step) {
                uint32_t p_addr = __cvta_generic_to_shared(&smem_P[0][0]) + step * 32;
                uint32_t v0_addr = __cvta_generic_to_shared(&smem_V0[0][0]) + step * 2048;
                uint32_t accum = (step == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_O_new_0, make_desc_P(p_addr), make_desc_V(v0_addr), make_idesc_PV(), accum);
            }
            for (int step = 0; step < 8; ++step) {
                uint32_t p_addr = __cvta_generic_to_shared(&smem_P[0][0]) + step * 32;
                uint32_t v1_addr = __cvta_generic_to_shared(&smem_V1[0][0]) + step * 2048;
                uint32_t accum = (step == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_O_new_1, make_desc_P(p_addr), make_desc_V(v1_addr), make_idesc_PV(), accum);
            }
            umma_commit_cg1_fn(mbar_umma_PV);
        }
        mbarrier_wait_fn(mbar_umma_PV, phase_umma_PV & 1);
        phase_umma_PV ^= 1;
        tcgen05_fence_after_fn();
        
        for (int c = 0; c < 64; c += 4) {
            uint32_t r0, r1, r2, r3;
            uint32_t col = (tmem_O_new_0 & 0xFFFF) + c;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            O_regs[c+0] = O_regs[c+0] * scale + __uint_as_float(r0);
            O_regs[c+1] = O_regs[c+1] * scale + __uint_as_float(r1);
            O_regs[c+2] = O_regs[c+2] * scale + __uint_as_float(r2);
            O_regs[c+3] = O_regs[c+3] * scale + __uint_as_float(r3);
        }
        for (int c = 0; c < 64; c += 4) {
            uint32_t r0, r1, r2, r3;
            uint32_t col = (tmem_O_new_1 & 0xFFFF) + c;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            O_regs[c+64+0] = O_regs[c+64+0] * scale + __uint_as_float(r0);
            O_regs[c+64+1] = O_regs[c+64+1] * scale + __uint_as_float(r1);
            O_regs[c+64+2] = O_regs[c+64+2] * scale + __uint_as_float(r2);
            O_regs[c+64+3] = O_regs[c+64+3] * scale + __uint_as_float(r3);
        }
        
        m_prev = m_new;
        l_prev = l_new;
    }
    
    __syncthreads();
    
    for (int c = 0; c < 128; c += 4) {
        float f0 = O_regs[c+0] / l_prev;
        float f1 = O_regs[c+1] / l_prev;
        float f2 = O_regs[c+2] / l_prev;
        float f3 = O_regs[c+3] / l_prev;
        uint32_t b01 = pack_bf16_fn(__float_as_uint(f0), __float_as_uint(f1));
        uint32_t b23 = pack_bf16_fn(__float_as_uint(f2), __float_as_uint(f3));
        uint32_t smem_addr = __cvta_generic_to_shared(&smem_P[tid][c]);
        asm volatile("st.shared.v2.b32 [%0], {%1, %2};" :: "r"(smem_addr), "r"(b01), "r"(b23));
    }
    __syncthreads();
    
    for (int step = 0; step < 128; step += 4) {
        int row = step + (tid / 32);
        int col = (tid % 32) * 4;
        int global_s = q_s_idx + row;
        if (global_s < S) {
            uint2 data = *(uint2*)(&smem_P[row][col]);
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
    
    cuuint64_t globalDim[5] = {64, 2, (cuuint64_t)S, (cuuint64_t)H, (cuuint64_t)B};
    cuuint64_t globalStrides[4] = {128, 256, 256 * (cuuint64_t)S, 256 * (cuuint64_t)S * (cuuint64_t)H};
    cuuint32_t boxDim[5] = {64, 1, 128, 1, 1};
    cuuint32_t elemStrides[5] = {1, 1, 1, 1, 1};
    
    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_Q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, Q.data_ptr(),
        globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));
    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_K, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, K.data_ptr(),
        globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));
    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_V, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, V.data_ptr(),
        globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));
    
    dim3 grid((S + 127) / 128, H, B);
    dim3 block(128);
    
    int smem_bytes = 133500;
    
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