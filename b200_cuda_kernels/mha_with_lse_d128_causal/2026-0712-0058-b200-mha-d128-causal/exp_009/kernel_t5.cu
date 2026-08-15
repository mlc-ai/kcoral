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

__device__ __forceinline__ uint32_t swizzle_128B_2B(uint32_t row, uint32_t col) {
    uint32_t chunk_x = col / 64;
    uint32_t chunk_y = row % 8;
    uint32_t swizzled_chunk = chunk_x ^ chunk_y;
    return row * 128 + swizzled_chunk * 64 + (col % 64);
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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
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

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t tmem_addr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(tmem_addr));
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
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool a_major = 0, bool b_major = 0) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (a_major << 15);
    d |= (b_major << 16);
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
    __nv_bfloat16* smem_Q_1 = smem_buf + 8192;            // 16KB
    __nv_bfloat16* smem_K_0 = smem_buf + 16384;           // 16KB
    __nv_bfloat16* smem_K_1 = smem_buf + 24576;           // 16KB
    __nv_bfloat16* smem_V_0 = smem_buf + 32768;           // 16KB
    __nv_bfloat16* smem_V_1 = smem_buf + 40960;           // 16KB
    __nv_bfloat16* smem_P_0 = smem_buf + 49152;           // 16KB
    __nv_bfloat16* smem_P_1 = smem_buf + 57344;           // 16KB
    
    __shared__ __align__(8) uint64_t mbar_Q[1];
    __shared__ __align__(8) uint64_t mbar_K[2];
    __shared__ __align__(8) uint64_t mbar_V[2];
    __shared__ __align__(8) uint64_t mbar_P[1];
    __shared__ __align__(8) uint64_t mbar_O[1];

    __shared__ float smem_m[128];
    __shared__ float smem_sum[128];
    __shared__ float smem_m_prev[128];
    __shared__ float smem_sum_prev[128];

    if (threadIdx.x < 128) {
        smem_m[threadIdx.x] = -1e20f;
        smem_sum[threadIdx.x] = 0.0f;
        smem_m_prev[threadIdx.x] = -1e20f;
        smem_sum_prev[threadIdx.x] = 0.0f;
    }

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_Q[0], 1);
        init_smem_barrier_fn(&mbar_K[0], 1);
        init_smem_barrier_fn(&mbar_V[0], 1);
        init_smem_barrier_fn(&mbar_K[1], 1);
        init_smem_barrier_fn(&mbar_V[1], 1);
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

    uint32_t phase_Q = 0, phase_K[2] = {0, 0}, phase_V[2] = {0, 0}, phase_P = 0, phase_O = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_Q[0], 32768);
        tma_load_3d_fn(&tma_Q, &mbar_Q[0], smem_Q_0, 0, q_blk * 128, bh);
        tma_load_3d_fn(&tma_Q, &mbar_Q[0], smem_Q_1, 64, q_blk * 128, bh);
    }
    mbarrier_wait_fn(&mbar_Q[0], phase_Q);
    phase_Q ^= 1;

    uint32_t idesc_Q = make_instr_desc_fn(128, 64, 0, 0);
    uint32_t idesc_O = make_instr_desc_fn(128, 64, 0, 1);
    float sqrt_D = 11.3137085f; // sqrt(128)

    int num_blocks = (S + 127) / 128;
    int next_k_blk = 0;
    int next_valid = (next_k_blk <= q_blk) && (next_k_blk < num_blocks);
    
    if (next_valid) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_K[0], 32768);
            tma_load_3d_fn(&tma_K, &mbar_K[0], smem_K_0, 0, next_k_blk * 128, bh);
            tma_load_3d_fn(&tma_K, &mbar_K[0], smem_K_0 + 8192, 64, next_k_blk * 128, bh);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar_V[0], 32768);
            tma_load_3d_fn(&tma_V, &mbar_V[0], smem_V_0, 0, next_k_blk * 128, bh);
            tma_load_3d_fn(&tma_V, &mbar_V[0], smem_V_0 + 8192, 64, next_k_blk * 128, bh);
        }
    }

    for (int k_blk = 0; k_blk <= q_blk && k_blk < num_blocks; k_blk++) {
        int buf_idx = k_blk % 2;
        int next_buf = (k_blk + 1) % 2;
        
        mbarrier_wait_fn(&mbar_K[buf_idx], phase_K[buf_idx]);
        phase_K[buf_idx] ^= 1;
        mbarrier_wait_fn(&mbar_V[buf_idx], phase_V[buf_idx]);
        phase_V[buf_idx] ^= 1;
        
        __nv_bfloat16* cur_K_0 = (buf_idx == 0) ? smem_K_0 : smem_K_1;
        __nv_bfloat16* cur_K_1 = (buf_idx == 0) ? smem_K_0 + 8192 : smem_K_1 + 8192;
        __nv_bfloat16* cur_V_0 = (buf_idx == 0) ? smem_V_0 : smem_V_1;
        __nv_bfloat16* cur_V_1 = (buf_idx == 0) ? smem_V_0 + 8192 : smem_V_1 + 8192;
        
        uint64_t desc_Q_0 = make_smem_desc_sm100_fn(smem_Q_0, 1, 1024);
        uint64_t desc_K_0 = make_smem_desc_sm100_fn(cur_K_0, 1, 1024);
        
        fence_proxy_async_fn();
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_P[0], 16384);
            for (int K_iter = 0; K_iter < 4; ++K_iter) {
                uint32_t accum = (K_iter == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_P, desc_Q_0 + K_iter * 2, desc_K_0 + K_iter * 2, idesc_Q, accum);
            }
            uint64_t desc_Q_1 = make_smem_desc_sm100_fn(smem_Q_1, 1, 1024);
            uint64_t desc_K_1 = make_smem_desc_sm100_fn(cur_K_1, 1, 1024);
            for (int K_iter = 0; K_iter < 4; ++K_iter) {
                umma_f16_cg1_fn(tmem_P + (4096 << 8), desc_Q_1 + K_iter * 2, desc_K_1 + K_iter * 2, idesc_Q, 1);
            }
            umma_commit_1sm_fn(&mbar_P[0]);
        }
        mbarrier_wait_fn(&mbar_P[0], phase_P);
        phase_P ^= 1;

        float m_local = -1e20f;
        float p_val_local[128];
        
        for (int K_iter = 0; K_iter < 4; ++K_iter) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_P + ((threadIdx.x & 0xFF) << 8) + (K_iter * 16), &r0, &r1, &r2, &r3);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float p0 = __uint_as_float(r0);
            float p1 = __uint_as_float(r1);
            float p2 = __uint_as_float(r2);
            float p3 = __uint_as_float(r3);
            
            int q_idx = q_blk * 128 + threadIdx.x;
            int k_idx0 = k_blk * 128 + K_iter * 16;
            bool valid0 = (q_idx < S) && (k_idx0 < S) && (k_idx0 <= q_idx);
            if (valid0) { p0 /= sqrt_D; m_local = fmaxf(m_local, p0); } else { p0 = -1e20f; }
            
            int k_idx1 = k_blk * 128 + K_iter * 16 + 1;
            bool valid1 = (q_idx < S) && (k_idx1 < S) && (k_idx1 <= q_idx);
            if (valid1) { p1 /= sqrt_D; m_local = fmaxf(m_local, p1); } else { p1 = -1e20f; }
            
            int k_idx2 = k_blk * 128 + K_iter * 16 + 2;
            bool valid2 = (q_idx < S) && (k_idx2 < S) && (k_idx2 <= q_idx);
            if (valid2) { p2 /= sqrt_D; m_local = fmaxf(m_local, p2); } else { p2 = -1e20f; }
            
            int k_idx3 = k_blk * 128 + K_iter * 16 + 3;
            bool valid3 = (q_idx < S) && (k_idx3 < S) && (k_idx3 <= q_idx);
            if (valid3) { p3 /= sqrt_D; m_local = fmaxf(m_local, p3); } else { p3 = -1e20f; }
            
            p_val_local[K_iter * 4 + 0] = p0;
            p_val_local[K_iter * 4 + 1] = p1;
            p_val_local[K_iter * 4 + 2] = p2;
            p_val_local[K_iter * 4 + 3] = p3;
        }
        
        for (int K_iter = 0; K_iter < 4; ++K_iter) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_P + (2048 << 8) + ((threadIdx.x & 0xFF) << 8) + (K_iter * 16), &r0, &r1, &r2, &r3);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float p0 = __uint_as_float(r0);
            float p1 = __uint_as_float(r1);
            float p2 = __uint_as_float(r2);
            float p3 = __uint_as_float(r3);
            
            int q_idx = q_blk * 128 + threadIdx.x;
            int k_idx0 = k_blk * 128 + 64 + K_iter * 16;
            bool valid0 = (q_idx < S) && (k_idx0 < S) && (k_idx0 <= q_idx);
            if (valid0) { p0 /= sqrt_D; m_local = fmaxf(m_local, p0); } else { p0 = -1e20f; }
            
            int k_idx1 = k_blk * 128 + 64 + K_iter * 16 + 1;
            bool valid1 = (q_idx < S) && (k_idx1 < S) && (k_idx1 <= q_idx);
            if (valid1) { p1 /= sqrt_D; m_local = fmaxf(m_local, p1); } else { p1 = -1e20f; }
            
            int k_idx2 = k_blk * 128 + 64 + K_iter * 16 + 2;
            bool valid2 = (q_idx < S) && (k_idx2 < S) && (k_idx2 <= q_idx);
            if (valid2) { p2 /= sqrt_D; m_local = fmaxf(m_local, p2); } else { p2 = -1e20f; }
            
            int k_idx3 = k_blk * 128 + 64 + K_iter * 16 + 3;
            bool valid3 = (q_idx < S) && (k_idx3 < S) && (k_idx3 <= q_idx);
            if (valid3) { p3 /= sqrt_D; m_local = fmaxf(m_local, p3); } else { p3 = -1e20f; }
            
            p_val_local[16 + K_iter * 4 + 0] = p0;
            p_val_local[16 + K_iter * 4 + 1] = p1;
            p_val_local[16 + K_iter * 4 + 2] = p2;
            p_val_local[16 + K_iter * 4 + 3] = p3;
        }
        
        for (int offset = 2; offset < 8; offset *= 2) {
            m_local = fmaxf(m_local, __shfl_xor_sync(0xffffffff, m_local, offset));
        }
        
        float m_prev = smem_m_prev[threadIdx.x];
        float m_new = fmaxf(m_prev, m_local);
        
        float sum_scaled = smem_sum_prev[threadIdx.x] * exp_f(m_prev - m_new);
        
        float sum_local = 0.0f;
        for(int i = 0; i < 128; ++i) {
            if (p_val_local[i] > -1e19f) {
                p_val_local[i] = exp_f(p_val_local[i] - m_new);
                sum_local += p_val_local[i];
            } else {
                p_val_local[i] = 0.0f;
            }
        }
        
        for (int offset = 2; offset < 8; offset *= 2) {
            sum_local += __shfl_xor_sync(0xffffffff, sum_local, offset);
        }
        
        smem_m_prev[threadIdx.x] = m_new;
        smem_sum_prev[threadIdx.x] = sum_scaled + sum_local;
        
        for(int i = 0; i < 128; ++i) {
            uint32_t c = (i < 64) ? i : (i - 64);
            uint32_t swizzled_c = swizzle_128B_2B(threadIdx.x, c);
            if (i < 64) {
                smem_P_0[(threadIdx.x & 0xFF) * 64 + swizzled_c] = __float2bfloat16(p_val_local[i]);
            } else {
                smem_P_1[(threadIdx.x & 0xFF) * 64 + swizzled_c] = __float2bfloat16(p_val_local[i]);
            }
        }

        __syncthreads(); 
        fence_async_shared_fn();

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_O[0], 16384);
            
            uint64_t desc_P_0 = make_smem_desc_sm100_fn(smem_P_0, 1, 1024);
            uint64_t desc_P_1 = make_smem_desc_sm100_fn(smem_P_1, 1, 1024);
            uint64_t desc_V_0 = make_smem_desc_sm100_fn(cur_V_0, 1, 1024);
            uint64_t desc_V_1 = make_smem_desc_sm100_fn(cur_V_1, 1, 1024);
            
            for (int j = 0; j < 4; ++j) {
                uint32_t accum_o = (k_blk == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_O_0, desc_P_0 + j * 2, desc_V_0 + j * 64, idesc_O, accum_o);
            }
            for (int j = 0; j < 4; ++j) {
                uint32_t accum_o = (k_blk == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_O_1, desc_P_1 + j * 2, desc_V_1 + j * 64, idesc_O, accum_o);
            }
            umma_commit_1sm_fn(&mbar_O[0]);
        }
        mbarrier_wait_fn(&mbar_O[0], phase_O);
        phase_O ^= 1;
        
        if (next_valid) {
            if (threadIdx.x == 0) {
                __nv_bfloat16* n_K_0 = (next_buf == 0) ? smem_K_0 : smem_K_1;
                __nv_bfloat16* n_K_1 = (next_buf == 0) ? smem_K_0 + 8192 : smem_K_1 + 8192;
                mbarrier_arrive_and_expect_tx_fn(&mbar_K[next_buf], 32768);
                tma_load_3d_fn(&tma_K, &mbar_K[next_buf], n_K_0, 0, next_k_blk * 128, bh);
                tma_load_3d_fn(&tma_K, &mbar_K[next_buf], n_K_1, 64, next_k_blk * 128, bh);
                
                __nv_bfloat16* n_V_0 = (next_buf == 0) ? smem_V_0 : smem_V_1;
                __nv_bfloat16* n_V_1 = (next_buf == 0) ? smem_V_0 + 8192 : smem_V_1 + 8192;
                mbarrier_arrive_and_expect_tx_fn(&mbar_V[next_buf], 32768);
                tma_load_3d_fn(&tma_V, &mbar_V[next_buf], n_V_0, 0, next_k_blk * 128, bh);
                tma_load_3d_fn(&tma_V, &mbar_V[next_buf], n_V_1, 64, next_k_blk * 128, bh);
            }
        }
        
        next_k_blk++;
        next_valid = (next_k_blk <= q_blk) && (next_k_blk < num_blocks);
    }

    float o_val_0[64];
    float o_val_1[64];
    for (int i = 0; i < 8; ++i) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn((tmem_O_0 + ((threadIdx.x & 0xFF) << 8) + (i * 16)), &r0, &r1, &r2, &r3);
        tmem_load_4x_fn((tmem_O_1 + ((threadIdx.x & 0xFF) << 8) + (i * 16)), &r0, &r1, &r2, &r3);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        o_val_0[i * 8 + threadIdx.x % 8] = __uint_as_float(r0);
        o_val_0[i * 8 + threadIdx.x % 8 + 4] = __uint_as_float(r1);
        o_val_1[i * 8 + threadIdx.x % 8] = __uint_as_float(r2);
        o_val_1[i * 8 + threadIdx.x % 8 + 4] = __uint_as_float(r3);
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
            LSE[bh * S + row_idx] = smem_m_prev[threadIdx.x] + logf(smem_sum_prev[threadIdx.x]);
        }
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_P, 128);
        tmem_dealloc_fn(tmem_O_0, 64);
        tmem_dealloc_fn(tmem_O_1, 64);
    }
}

namespace tvm_ffi_mha {

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, uint32_t box2, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
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

    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, q_ptr, D, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, k_ptr, D, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, v_ptr, D, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B));

    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    int smem_size = 224 * 1024;
    
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