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
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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
    d |= ((uint64_t)((lbo & 0x3FFFF) >> 4)) << 16;
    d |= ((uint64_t)((sbo & 0x3FFFF) >> 4)) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    d |= ((uint64_t)base_offset) << 49;
    return d;
}

__device__ __forceinline__ uint64_t desc_mn_major_128b(void* ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (addr & 0x3FFFF) >> 4;
    d |= ((uint64_t)((lbo & 0x3FFFF) >> 4)) << 16;
    d |= ((uint64_t)((sbo & 0x3FFFF) >> 4)) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    d |= ((uint64_t)base_offset) << 49;
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
    uint64_t desc_s = desc_k_major_128b(s_mem_ptr, 1, 1024);
    asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], %1;" :: "r"(t_mem_addr), "l"(desc_s));
}

__device__ __forceinline__ void tmem_cp_128_row_mn(void* s_mem_ptr, uint32_t t_mem_addr) {
    uint64_t desc_s = desc_mn_major_128b(s_mem_ptr, 16384, 1024);
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

__device__ __forceinline__ void umma_f16_cg1(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void commit_umma_1sm(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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
    
    extern __shared__ __align__(1024) char smem_pool[];
    char* smem = (char*)(((uintptr_t)smem_pool + 1023) & ~1023);
    
    char* smem_Q = smem;
    char* smem_K = smem_Q + 32768;
    char* smem_V = smem_K + 32768;
    char* smem_P = smem_K; // Reuse K's space post-QKT
    
    __shared__ __align__(8) uint64_t bar_Q[1];
    __shared__ __align__(8) uint64_t bar_K[1];
    __shared__ __align__(8) uint64_t bar_V[1];
    __shared__ __align__(8) uint64_t bar_S[1];
    __shared__ __align__(8) uint64_t bar_O[1];
    
    __shared__ __align__(4) uint32_t tmem_addr[8];
    
    if (tid == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_K, 1);
        init_smem_barrier_fn(bar_V, 1);
        init_smem_barrier_fn(bar_S, 1);
        init_smem_barrier_fn(bar_O, 1);
        
        tmem_alloc_fn(&tmem_addr[0], 4096);
        tmem_addr[1] = tmem_addr[0] + 64;
        tmem_addr[2] = tmem_addr[0] + 128;
        tmem_addr[3] = tmem_addr[0] + 192;
        tmem_addr[4] = tmem_addr[0] + 256;
        tmem_addr[5] = tmem_addr[0] + 320;
        tmem_addr[6] = tmem_addr[0] + 384;
        tmem_addr[7] = tmem_addr[0] + 448;
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t tmem_base = tmem_addr[0];
    int32_t phase_Q = 0;
    int32_t phase_K = 0;
    int32_t phase_V = 0;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_Q, 32768);
        tma_load_3d_fn(&tma_Q, bar_Q, smem_Q, 0, bh * S_i + block_idx * 128, bh);
        tma_load_3d_fn(&tma_Q, bar_Q, smem_Q + 16384, 64, bh * S_i + block_idx * 128, bh);
        
        mbarrier_arrive_and_expect_tx_fn(bar_K, 32768);
        tma_load_3d_fn(&tma_K, bar_K, smem_K, 0, bh * S_i + 0 * 128, bh);
        tma_load_3d_fn(&tma_K, bar_K, smem_K + 16384, 64, bh * S_i + 0 * 128, bh);
        
        mbarrier_arrive_and_expect_tx_fn(bar_V, 32768);
        tma_load_3d_fn(&tma_V, bar_V, smem_V, 0, bh * S_i + 0 * 128, bh);
        tma_load_3d_fn(&tma_V, bar_V, smem_V + 16384, 64, bh * S_i + 0 * 128, bh);
    }
    mbarrier_wait_fn(bar_Q, phase_Q);
    mbarrier_wait_fn(bar_K, phase_K);
    mbarrier_wait_fn(bar_V, phase_V);
    __syncthreads();
    
    float o_acc_flat[128];
    #pragma unroll
    for(int i=0; i<128; i++) o_acc_flat[i] = 0.0f;
    
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    
    int32_t num_steps = block_idx + 1;
    int32_t total_steps = (S_i + 127) / 128;
    if (num_steps > total_steps) {
        num_steps = total_steps;
    }
    
    uint32_t idesc_qk_major_accum = (1 << 4) | (1 << 7) | (1 << 10) | (0 << 15) | (0 << 16) | ((128 / 8) << 17) | ((128 / 16) << 24);
    uint32_t idesc_pv = (1 << 4) | (1 << 7) | (1 << 10) | (0 << 15) | (1 << 16) | ((64 / 8) << 17) | ((128 / 16) << 24);
    
    for (int step = 0; step < num_steps; step++) {
        if (step < num_steps - 1) {
            if (tid == 0) {
                int next_step = step + 1;
                mbarrier_arrive_and_expect_tx_fn(bar_K, 32768);
                tma_load_3d_fn(&tma_K, bar_K, smem_K, 0, bh * S_i + next_step * 128, bh);
                tma_load_3d_fn(&tma_K, bar_K, smem_K + 16384, 64, bh * S_i + next_step * 128, bh);
                
                mbarrier_arrive_and_expect_tx_fn(bar_V, 32768);
                tma_load_3d_fn(&tma_V, bar_V, smem_V, 0, bh * S_i + next_step * 128, bh);
                tma_load_3d_fn(&tma_V, bar_V, smem_V + 16384, 64, bh * S_i + next_step * 128, bh);
            }
        }
        
        mbarrier_wait_fn(bar_K, phase_K);
        mbarrier_wait_fn(bar_V, phase_V);
        
        uint32_t tmem_S = tmem_addr[4];
        uint32_t tmem_Q0 = tmem_addr[0];
        uint32_t tmem_Q1 = tmem_addr[1];
        uint32_t tmem_K0 = tmem_addr[2];
        uint32_t tmem_K1 = tmem_addr[3];
        
        if (tid == 0) {
            write_zeros_to_tmem(tmem_S, 128);
            
            mbarrier_arrive_and_expect_tx_fn(bar_S, 65536);
            load_Q_128x64_to_tmem(smem_Q, tmem_base, &tmem_Q0);
            load_Q_128x64_to_tmem(smem_Q + 16384, tmem_base + 64, &tmem_Q1);
            load_K_128x64_to_tmem(smem_K, tmem_base + 128, &tmem_K0);
            load_K_128x64_to_tmem(smem_K + 16384, tmem_base + 192, &tmem_K1);
        }
        mbarrier_wait_fn(bar_S, phase_Q); 
        
        for (int k_step = 0; k_step < 4; k_step++) {
            umma_f16_cg1(tmem_S, desc_k_major_128b(smem_Q, 1, 1024) + (uint64_t)(k_step * 16), desc_k_major_128b(smem_K, 1, 1024) + (uint64_t)(k_step * 16), idesc_qk_major_accum, 1);
            umma_f16_cg1(tmem_S, desc_k_major_128b(smem_Q + 8192, 1, 1024) + (uint64_t)(k_step * 16), desc_k_major_128b(smem_K + 8192, 1, 1024) + (uint64_t)(k_step * 16), idesc_qk_major_accum, 1);
        }
        commit_umma_1sm(bar_S);
        mbarrier_wait_fn(bar_S, phase_Q);
        phase_Q ^= 1;
        
        float s_acc_flat[128];
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S + col, &r0, &r1, &r2, &r3);
            s_acc_flat[col] = __uint_as_float(r0);
            s_acc_flat[col+1] = __uint_as_float(r1);
            s_acc_flat[col+2] = __uint_as_float(r2);
            s_acc_flat[col+3] = __uint_as_float(r3);
        }
        
        int32_t global_i = block_idx * 128 + tid;
        float m_curr = -INFINITY;
        for (int j = 0; j < 128; j++) {
            int global_j = step * 128 + j;
            if (global_j > global_i || global_j >= S_i) {
                s_acc_flat[j] = -INFINITY;
            } else {
                s_acc_flat[j] *= 0.08838834764f;
            }
            m_curr = fmaxf(m_curr, s_acc_flat[j]);
        }
        
        bool row_valid = m_curr > -INFINITY;
        float m_new = fmaxf(m_prev, m_curr);
        bool rescale = (m_prev > -INFINITY && row_valid && (m_new > m_prev));
        
        if (rescale) {
            float factor = fast_exp2f_fn((m_prev - m_new) * 1.44269504089f);
            for (int d = 0; d < 128; d++) {
                o_acc_flat[d] *= factor;
            }
            l_prev *= factor;
            m_prev = m_new;
        } else if (m_prev == -INFINITY) {
            m_prev = m_curr;
        } else {
            m_prev = m_new;
        }
        
        float l_curr = 0.0f;
        for (int j = 0; j < 128; j++) {
            if (s_acc_flat[j] > -INFINITY) {
                s_acc_flat[j] = fast_exp2f_fn((s_acc_flat[j] - m_prev) * 1.44269504089f);
                l_curr += s_acc_flat[j];
            } else {
                s_acc_flat[j] = 0.0f;
            }
        }
        if (row_valid) {
            l_prev += l_curr;
        }
        
        __syncthreads();
        __nv_bfloat16* smem_P_buf = (__nv_bfloat16*)smem_P;
        for (int j = 0; j < 128; j++) {
            float p = s_acc_flat[j];
            __nv_bfloat16 p_bf = __float2bfloat16(p);
            int chunk = j / 8;
            int rem = j % 8;
            int swizzled_chunk = chunk ^ (tid % 8);
            int swizzled_j = swizzled_chunk * 8 + rem;
            smem_P_buf[tid * 128 + swizzled_j] = p_bf;
        }
        
        __syncthreads();
        fence_proxy_async_fn(); 
        
        uint32_t tmem_P0 = tmem_addr[0];
        uint32_t tmem_P1 = tmem_addr[1];
        uint32_t tmem_V0 = tmem_addr[2];
        uint32_t tmem_V1 = tmem_addr[3];
        uint32_t tmem_O0 = tmem_addr[4];
        uint32_t tmem_O1 = tmem_addr[5];
        
        if (tid == 0) {
            write_zeros_to_tmem(tmem_O0, 64);
            write_zeros_to_tmem(tmem_O1, 64);
            
            mbarrier_arrive_and_expect_tx_fn(bar_S, 65536);
            load_p_128x64_to_tmem(smem_P_buf, tmem_base, &tmem_P0, 0);
            load_p_128x64_to_tmem(smem_P_buf, tmem_base, &tmem_P1, 1);
            load_V_128x64_to_tmem(smem_V, tmem_base + 128, &tmem_V0);
            load_V_128x64_to_tmem(smem_V + 16384, tmem_base + 192, &tmem_V1);
        }
        mbarrier_wait_fn(bar_S, phase_Q);
        
        for (int k_step = 0; k_step < 4; k_step++) {
            umma_f16_cg1(tmem_O0, desc_k_major_128b(smem_P_buf, 1, 1024) + (uint64_t)(k_step * 16), desc_mn_major_128b(smem_V + k_step * 2048, 16384, 1024), idesc_pv, 1);
            umma_f16_cg1(tmem_O1, desc_k_major_128b((__nv_bfloat16*)smem_P_buf + 4096, 1, 1024) + (uint64_t)(k_step * 16), desc_mn_major_128b(smem_V + 16384 + k_step * 2048, 16384, 1024), idesc_pv, 1);
        }
        commit_umma_1sm(bar_O);
        mbarrier_wait_fn(bar_O, phase_K);
        phase_K ^= 1;
        
        __syncthreads();
        phase_V ^= 1;
    }
    
    tmem_load_fence_fn();
    
    uint32_t tmem_O0 = tmem_addr[4];
    uint32_t tmem_O1 = tmem_addr[5];
    for (int col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_O0 + col, &r0, &r1, &r2, &r3);
        o_acc_flat[col] += __uint_as_float(r0);
        o_acc_flat[col+1] += __uint_as_float(r1);
        o_acc_flat[col+2] += __uint_as_float(r2);
        o_acc_flat[col+3] += __uint_as_float(r3);
        
        tmem_load_4x_fn(tmem_O1 + col, &r0, &r1, &r2, &r3);
        o_acc_flat[64 + col] += __uint_as_float(r0);
        o_acc_flat[64 + col+1] += __uint_as_float(r1);
        o_acc_flat[64 + col+2] += __uint_as_float(r2);
        o_acc_flat[64 + col+3] += __uint_as_float(r3);
    }
    
    float final_l = l_prev;
    float final_m = m_prev;
    
    if (final_l > 0.0f) {
        float f_final = 1.0f / final_l;
        for (int d = 0; d < 128; d++) {
            o_acc_flat[d] *= f_final;
        }
    } else {
        for (int d = 0; d < 128; d++) {
            o_acc_flat[d] = 0.0f;
        }
    }
    
    int32_t global_i = block_idx * 128 + tid;
    if (global_i < S_i) {
        for(int d=0; d<128; d++) {
            O[(uint64_t)bh * S_i * 128 + (uint64_t)global_i * 128 + d] = __float2bfloat16(o_acc_flat[d]);
        }
        LSE[(uint64_t)bh * S_i + (uint64_t)global_i] = final_m + logf(final_l);
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