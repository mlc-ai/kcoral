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

namespace tvm_ffi_kernel {

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_dim0, uint64_t gmem_dim1, uint64_t gmem_dim2, uint32_t smem_dim0, uint32_t smem_dim1, uint32_t smem_dim2, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[3] = {gmem_dim0, gmem_dim1, gmem_dim2};
    cuuint64_t globalStrides[2] = {gmem_dim0 * 2, gmem_dim0 * gmem_dim1 * 2};
    cuuint32_t boxDim[3] = {smem_dim0, smem_dim1, smem_dim2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim,
        globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, l2Promotion, oobFill
    );
}

__device__ __forceinline__ uint64_t desc_k_major_128b(void* ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (addr & 0x3FFFF) >> 4;
    d |= (lbo & 0x3FFFF) >> 4 << 16;
    d |= (sbo & 0x3FFFF) >> 4 << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    d |= (uint64_t)base_offset << 49;
    return d;
}

__device__ __forceinline__ uint64_t desc_mn_major_128b(void* ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (addr & 0x3FFFF) >> 4;
    d |= (lbo & 0x3FFFF) >> 4 << 16;
    d |= (sbo & 0x3FFFF) >> 4 << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    d |= (uint64_t)base_offset << 49;
    return d;
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_cp_128_row(void* s_mem_ptr, uint32_t t_mem_addr) {
    uint64_t desc_s = desc_k_major_128b(s_mem_ptr, 0, 1024);
    asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], %1;" :: "r"(t_mem_addr), "l"(desc_s));
}

__device__ __forceinline__ void tmem_cp_128_row_mn(void* s_mem_ptr, uint32_t t_mem_addr) {
    uint64_t desc_s = desc_mn_major_128b(s_mem_ptr, 8192, 1024);
    asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], %1;" :: "r"(t_mem_addr), "l"(desc_s));
}

__device__ __forceinline__ void write_zeros_to_tmem(uint32_t t_mem_addr, int num_cols) {
    for (uint32_t c = 0; c < num_cols; c += 4) {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {0,0,0,0}, [%0];"
                     :: "r"(t_mem_addr + c) : "memory");
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void load_Q_128x64_to_tmem(char* smem_Q, uint32_t tmem_base, uint32_t* tmem_Q_ptr) {
    *tmem_Q_ptr = tmem_base;
    for (int col = 0; col < 64; col += 8) {
        tmem_cp_128_row(smem_Q + col, tmem_base + col);
    }
}

__device__ __forceinline__ void load_K_128x64_to_tmem(char* smem_K, uint32_t tmem_base, uint32_t* tmem_K_ptr) {
    *tmem_K_ptr = tmem_base;
    for (int col = 0; col < 64; col += 8) {
        tmem_cp_128_row(smem_K + col, tmem_base + col);
    }
}

__device__ __forceinline__ void load_p_128x64_to_tmem(__nv_bfloat16* smem_P, uint32_t tmem_base, uint32_t* tmem_P_ptr, int i) {
    *tmem_P_ptr = tmem_base + i * 64;
    char* s_mem_ptr = (char*)smem_P + i * 8192;
    for (int col = 0; col < 64; col += 8) {
        tmem_cp_128_row(s_mem_ptr, *tmem_P_ptr + col);
    }
}

__device__ __forceinline__ void load_V_128x64_to_tmem(char* smem_V, uint32_t tmem_base, uint32_t* tmem_V_ptr) {
    *tmem_V_ptr = tmem_base;
    for (int row = 0; row < 128; row += 64) {
        tmem_cp_128_row_mn(smem_V + row * 64, tmem_base + row * 2);
    }
}

__device__ __forceinline__ void umma_f16_cg1(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(1));
}

__device__ __forceinline__ void commit_umma_1sm(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void store_p_to_smem_unswizzled(__nv_bfloat16* smem_P, int tid, float* p_acc) {
    for(int j = 0; j < 128; j++) {
        __nv_bfloat16 p_bf = __float2bfloat16(p_acc[j]);
        smem_P[tid * 128 + j] = p_bf;
    }
}

__global__ void mha_with_lse_d128_causal(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE, int32_t S_i) 
{
    setmaxnreg_inc_sync_fn<256>();

    int32_t tid = threadIdx.x;
    int32_t block_idx = blockIdx.x;
    int32_t bh = blockIdx.y;
    
    extern __shared__ char smem_base[];
    char* smem = (char*)smem_base;
    char* smem_Q_0 = (char*)(((uintptr_t)smem + 1023) & ~1023);
    char* smem_Q_1 = smem_Q_0 + 16384;
    char* smem_K_0 = smem_Q_1 + 16384;
    char* smem_K_1 = smem_K_0 + 16384;
    char* smem_V_0 = smem_K_1 + 16384;
    char* smem_V_1 = smem_V_0 + 16384;
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem_V_1 + 16384);
    
    __shared__ __align__(8) uint64_t bar_Q[1];
    __shared__ __align__(8) uint64_t bar_K[1];
    __shared__ __align__(8) uint64_t bar_V[1];
    __shared__ __align__(8) uint64_t bar_S[1];
    __shared__ __align__(8) uint64_t bar_O[1];
    
    __shared__ __align__(4) uint32_t tmem_addr[5];
    
    if (tid == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_K, 1);
        init_smem_barrier_fn(bar_V, 1);
        init_smem_barrier_fn(bar_S, 1);
        init_smem_barrier_fn(bar_O, 1);
        
        tmem_alloc_fn(&tmem_addr[0], 64);
        tmem_alloc_fn(&tmem_addr[1], 64);
        tmem_alloc_fn(&tmem_addr[2], 64);
        tmem_alloc_fn(&tmem_addr[3], 64);
        tmem_alloc_fn(&tmem_addr[4], 128);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t tmem_base = tmem_addr[0];
    uint32_t tmem_Q_0 = tmem_base;
    uint32_t tmem_Q_1 = tmem_base + 64;
    uint32_t tmem_K_0 = tmem_base + 128;
    uint32_t tmem_K_1 = tmem_base + 192;
    uint32_t tmem_S = tmem_base + 256;
    
    int32_t phase_Q = 0;
    int32_t phase_K = 0;
    int32_t phase_V = 0;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_Q, 32768);
        tma_load_3d_fn(&tma_Q, bar_Q, smem_Q_0, 0, bh * S_i + block_idx * 128, bh);
        tma_load_3d_fn(&tma_Q, bar_Q, smem_Q_1, 64, bh * S_i + block_idx * 128, bh);
        
        mbarrier_arrive_and_expect_tx_fn(bar_K, 32768);
        tma_load_3d_fn(&tma_K, bar_K, smem_K_0, 0, bh * S_i + 0 * 128, bh);
        tma_load_3d_fn(&tma_K, bar_K, smem_K_1, 64, bh * S_i + 0 * 128, bh);
        
        mbarrier_arrive_and_expect_tx_fn(bar_V, 32768);
        tma_load_3d_fn(&tma_V, bar_V, smem_V_0, 0, bh * S_i + 0 * 128, bh);
        tma_load_3d_fn(&tma_V, bar_V, smem_V_1, 64, bh * S_i + 0 * 128, bh);
    }
    mbarrier_wait_fn(bar_Q, phase_Q);
    mbarrier_wait_fn(bar_K, phase_K);
    mbarrier_wait_fn(bar_V, phase_V);
    __syncthreads();
    
    uint32_t tmem_O_0 = tmem_base;
    uint32_t tmem_O_1 = tmem_base + 64;
    write_zeros_to_tmem(tmem_O_0, 64);
    write_zeros_to_tmem(tmem_O_1, 64);
    
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    
    int32_t num_steps = block_idx + 1;
    if (num_steps > (S_i + 127) / 128) {
        num_steps = (S_i + 127) / 128;
    }
    
    uint32_t idesc_qk_major_accum = (1 << 4) | (1 << 7) | (1 << 10) | (0 << 15) | (0 << 16) | ((128 / 8) << 17) | ((128 / 16) << 24);
    uint32_t idesc_pj = (1 << 4) | (1 << 7) | (1 << 10) | (0 << 15) | (1 << 16) | ((128 / 8) << 17) | ((128 / 16) << 24);
    
    for (int step = 0; step < num_steps; step++) {
        if (step < num_steps - 1) {
            if (tid == 0) {
                int next_step = step + 1;
                mbarrier_arrive_and_expect_tx_fn(bar_K, 32768);
                tma_load_3d_fn(&tma_K, bar_K, smem_K_0, 0, bh * S_i + next_step * 128, bh);
                tma_load_3d_fn(&tma_K, bar_K, smem_K_1, 64, bh * S_i + next_step * 128, bh);
                
                mbarrier_arrive_and_expect_tx_fn(bar_V, 32768);
                tma_load_3d_fn(&tma_V, bar_V, smem_V_0, 0, bh * S_i + next_step * 128, bh);
                tma_load_3d_fn(&tma_V, bar_V, smem_V_1, 64, bh * S_i + next_step * 128, bh);
            }
        }
        
        mbarrier_wait_fn(bar_K, phase_K);
        mbarrier_wait_fn(bar_V, phase_V);
        
        if (tid == 0) {
            write_zeros_to_tmem(tmem_S, 128);
            
            mbarrier_arrive_and_expect_tx_fn(bar_S, 65536);
            load_Q_128x64_to_tmem(smem_Q_0, tmem_base, &tmem_Q_0);
            load_Q_128x64_to_tmem(smem_Q_1, tmem_base + 64, &tmem_Q_1);
            load_K_128x64_to_tmem(smem_K_0, tmem_base + 128, &tmem_K_0);
            load_K_128x64_to_tmem(smem_K_1, tmem_base + 192, &tmem_K_1);
        }
        mbarrier_wait_fn(bar_S, phase_Q); 
        
        for (int k_step = 0; k_step < 4; k_step++) {
            umma_f16_cg1(tmem_S, tmem_Q_0 + k_step * 16, tmem_K_0 + k_step * 16, idesc_qk_major_accum);
            umma_f16_cg1(tmem_S, tmem_Q_1 + k_step * 16, tmem_K_1 + k_step * 16, idesc_qk_major_accum);
        }
        commit_umma_1sm(bar_S);
        mbarrier_wait_fn(bar_S, phase_Q);
        phase_Q ^= 1;
        
        float s_acc[128];
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S + col, &r0, &r1, &r2, &r3);
            s_acc[col] = __uint_as_float(r0);
            s_acc[col+1] = __uint_as_float(r1);
            s_acc[col+2] = __uint_as_float(r2);
            s_acc[col+3] = __uint_as_float(r3);
        }
        
        int global_i = block_idx * 128 + tid;
        float m_curr = -INFINITY;
        for (int j = 0; j < 128; j++) {
            int global_j = step * 128 + j;
            if (global_j > global_i || global_j >= S_i) {
                s_acc[j] = -INFINITY;
            } else {
                s_acc[j] *= 0.08838834764f;
            }
            m_curr = fmaxf(m_curr, s_acc[j]);
        }
        
        bool row_valid = m_curr > -INFINITY;
        
        float m_new = fmaxf(m_prev, m_curr);
        bool rescale = (m_prev > -INFINITY && row_valid && (m_new - m_prev) > 0.1f);
        
        if (rescale) {
            float factor = fast_exp2f_fn((m_prev - m_new) * 1.44269504089f);
            for (int col = 0; col < 64; col += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_O_0 + col, &r0, &r1, &r2, &r3);
                r0 = __float_as_uint(__uint_as_float(r0) * factor);
                r1 = __float_as_uint(__uint_as_float(r1) * factor);
                r2 = __float_as_uint(__uint_as_float(r2) * factor);
                r3 = __float_as_uint(__uint_as_float(r3) * factor);
                tmem_load_4x_fn(tmem_O_1 + col, &r0, &r1, &r2, &r3); // Reuse regs
                
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                             :: "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(tmem_O_0 + col) : "memory");
                
                tmem_load_4x_fn(tmem_O_1 + col, &r0, &r1, &r2, &r3);
                r0 = __float_as_uint(__uint_as_float(r0) * factor);
                r1 = __float_as_uint(__uint_as_float(r1) * factor);
                r2 = __float_as_uint(__uint_as_float(r2) * factor);
                r3 = __float_as_uint(__uint_as_float(r3) * factor);
                
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                             :: "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(tmem_O_1 + col) : "memory");
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            l_prev *= factor;
            m_prev = m_new;
        } else if (m_prev == -INFINITY) {
            m_prev = m_curr;
        } else {
            m_prev = m_new;
        }
        
        float l_curr = 0.0f;
        for (int j = 0; j < 128; j++) {
            float p = 0.0f;
            if (s_acc[j] > -INFINITY) {
                p = fast_exp2f_fn((s_acc[j] - m_prev) * 1.44269504089f);
            }
            l_curr += p;
            s_acc[j] = p; 
        }
        if (row_valid) {
            l_prev += l_curr;
        }
        
        store_p_to_smem_unswizzled(smem_P, tid, s_acc);
        
        __syncthreads();
        fence_proxy_async_fn(); 
        
        if (tid == 0) {
            write_zeros_to_tmem(tmem_O_0, 64);
            write_zeros_to_tmem(tmem_O_1, 64);
            
            mbarrier_arrive_and_expect_tx_fn(bar_S, 65536);
            load_p_128x64_to_tmem(smem_P, tmem_base, &tmem_Q_0, 0);
            load_p_128x64_to_tmem(smem_P, tmem_base, &tmem_Q_1, 1);
            load_V_128x64_to_tmem(smem_V_0, tmem_base + 128, &tmem_K_0);
            load_V_128x64_to_tmem(smem_V_1, tmem_base + 192, &tmem_K_1);
        }
        mbarrier_wait_fn(bar_S, phase_Q);
        
        uint32_t tmem_P_0 = tmem_Q_0;
        uint32_t tmem_P_1 = tmem_Q_1;
        uint32_t tmem_V_0 = tmem_K_0;
        uint32_t tmem_V_1 = tmem_K_1;
        
        for (int k_step = 0; k_step < 4; k_step++) {
            umma_f16_cg1(tmem_O_0, tmem_P_0 + k_step * 16, tmem_V_0 + k_step * 2, idesc_pj);
            umma_f16_cg1(tmem_O_1, tmem_P_1 + k_step * 16, tmem_V_1 + k_step * 2, idesc_pj);
        }
        commit_umma_1sm(bar_O);
        mbarrier_wait_fn(bar_O, phase_K);
        phase_K ^= 1;
        
        __syncthreads(); 
        phase_V ^= 1;
    }
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_S, 65536);
        load_Q_128x64_to_tmem(smem_Q_0, tmem_base, &tmem_Q_0);
        load_Q_128x64_to_tmem(smem_Q_1, tmem_base + 64, &tmem_Q_1);
    }
    mbarrier_wait_fn(bar_S, phase_Q);
    
    if (l_prev > 0.0f) {
        float f_final = 1.0f / l_prev;
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_O_0 + col, &r0, &r1, &r2, &r3);
            r0 = __float_as_uint(__uint_as_float(r0) * f_final);
            r1 = __float_as_uint(__uint_as_float(r1) * f_final);
            r2 = __float_as_uint(__uint_as_float(r2) * f_final);
            r3 = __float_as_uint(__uint_as_float(r3) * f_final);
            
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         :: "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(tmem_O_0 + col) : "memory");
            
            tmem_load_4x_fn(tmem_O_1 + col, &r0, &r1, &r2, &r3);
            r0 = __float_as_uint(__uint_as_float(r0) * f_final);
            r1 = __float_as_uint(__uint_as_float(r1) * f_final);
            r2 = __float_as_uint(__uint_as_float(r2) * f_final);
            r3 = __float_as_uint(__uint_as_float(r3) * f_final);
            
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         :: "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(tmem_O_1 + col) : "memory");
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
    
    __syncthreads();
    
    int32_t global_i = block_idx * 128 + tid;
    if (global_i < S_i) {
        float final_l = l_prev;
        float final_m = m_prev;
        
        for(int d=0; d<64; d+=8) {
            float4 out0;
            uint32_t* out0_u32 = (uint32_t*)&out0;
            tmem_load_4x_fn(tmem_base + d/2, &out0_u32[0], &out0_u32[1], &out0_u32[2], &out0_u32[3]);
            uint16_t* out0_h = (uint16_t*)&out0;
            for(int i=0; i<8; i++) out0_h[i] = __float2bfloat16(__uint_as_float(out0_u32[i]));
            *(float4*)&O[(uint64_t)bh * S_i * 128 + (uint64_t)global_i * 128 + d] = out0;
            
            float4 out1;
            uint32_t* out1_u32 = (uint32_t*)&out1;
            tmem_load_4x_fn(tmem_base + 64 + d/2, &out1_u32[0], &out1_u32[1], &out1_u32[2], &out1_u32[3]);
            uint16_t* out1_h = (uint16_t*)&out1;
            for(int i=0; i<8; i++) out1_h[i] = __float2bfloat16(__uint_as_float(out1_u32[i]));
            *(float4*)&O[(uint64_t)bh * S_i * 128 + (uint64_t)global_i * 128 + d + 64] = out1;
        }
        
        LSE[(uint64_t)bh * S_i + (block_idx * 128 + tid)] = final_m + logf(final_l);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B_i = Q.size(0);
    int64_t H_i = Q.size(1);
    int64_t S_i = Q.size(2);
    int64_t D_i = Q.size(3); 
    
    if (D_i != 128) {
        fprintf(stderr, "Expected head dimension 128, got %ld\n", D_i);
        exit(1);
    }
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V;
    CUresult res_q = create_tma_3d_descriptor_2B(&tma_Q, (void*)Q_ptr, 128, S_i, B_i * H_i, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUresult res_k = create_tma_3d_descriptor_2B(&tma_K, (void*)K_ptr, 128, S_i, B_i * H_i, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUresult res_v = create_tma_3d_descriptor_2B(&tma_V, (void*)V_ptr, 128, S_i, B_i * H_i, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    if (res_q != CUDA_SUCCESS || res_k != CUDA_SUCCESS || res_v != CUDA_SUCCESS) {
        fprintf(stderr, "TMA descriptor creation failed!\n");
        exit(1);
    }
    
    int32_t num_blocks = (S_i + 127) / 128;
    dim3 grid(num_blocks, B_i * H_i);
    dim3 block(128);
    
    int smem_size = 115000;
    CUDA_CHECK(cudaFuncSetAttribute((const void*)mha_with_lse_d128_causal, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    mha_with_lse_d128_causal<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S_i);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel