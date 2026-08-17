#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
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

namespace tvm_ffi_example_cuda {

// ============================================================================
// Hardware Intrinsic Definitions
// ============================================================================

__device__ __forceinline__ bool elect_one_sync_fn() {
    uint32_t pred;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xFFFFFFFF;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(pred));
    return pred != 0;
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
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
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
   :: "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_fence_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    // UMMA commit with cta_group::2 multicast barrier arrive.
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3)); // It hardcodes ctaMask=0x3 (CTAs 0 and 1 only). This works only for cluster_size=2.
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    // Build SM100 UMMA SMEM descriptor (version=1, SWIZZLE_128B).
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; // LBO is stored in bits [29:16]
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; // Another pattern is to set SBO=0 with K-walked manually via the base address
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)2 << 61;   // layout_type = SWIZZLE_128B
    uint32_t swizzled_byte_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)swizzled_byte_offset << 49;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t b_major = 0) {
    // Build SM100 UMMA instruction descriptor (BF16×BF16→FP32, K-major for A and B).
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (A is K-Major), 1 (A is M-Major)
    d |= (b_major << 16);   // b_major = 0 (B is K-Major), 1 (B is N-Major)
    d |= ((N / 8) << 17);     // n_dim. For cta_group::2, N should be the combined dimension (BN_per_cta × 2).
    d |= ((M / 16) << 24);    // m_dim. For cta_group::2, M should be the combined dimension (BM_per_cta × 2).
    return d;
}

// Fixes the fundamental breakage of the original exp_fn on negative inputs
__device__ __forceinline__ float exp_fn(float x) {
    if (x > 50.0f) return 1e20f;
    if (x < -50.0f) return 0.0f;
    
    // Utilize classical Cody-Waite range reduction to convert e^x into 2^(x / ln(2))
    x *= 1.44269504f; 
    
    int exp = (__float_as_int(x) >> 23) & 0xFF;
    int n = exp - 127;
    int frac_bits = __float_as_int(x) & 0x7FFFFF;
    
    // Properly handle negative inputs by adjusting magnitude and mantissa fraction.
    if (__float_as_int(x) & (1 << 31)) {
        n = -n;
        frac_bits = (1 << 23) - frac_bits;
        if (frac_bits != 0) n--;
    }
    
    float frac = __int_as_float(frac_bits << 9) * (1.0f / 8388608.0f);
    
    // Advanced Horner evaluation mapping minimising relative approximation error bounds over [0, 1)
    float poly = 1.0f + frac * (0.6951f + frac * (0.2276f + frac * 0.0771f));
    int exp_bits = (n + 127) << 23;
    return __int_as_float((__float_as_int(poly) & 0x7FFFFF) + exp_bits);
}

// ============================================================================
// Attention FWD Pass Kernel
// ============================================================================

__global__ void fa4_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    uint32_t S, uint32_t heads) 
{
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)(smem_pool + 0);           // 16KB
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem_pool + 16384);        // 32KB
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem_pool + 49152);        // 32KB
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem_pool + 81920);        // 16KB
    
    uint64_t* bar_kv = (uint64_t*)(smem_pool + 98304);                
    uint64_t* bar_q = (uint64_t*)(smem_pool + 98312);                 

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_kv, 1);
        init_smem_barrier_fn(bar_q, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t tmem_S0, tmem_S1, tmem_O0, tmem_O1;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S0, 256);
        tmem_alloc_fn(&tmem_S1, 256);
        tmem_alloc_fn(&tmem_O0, 256);
        tmem_alloc_fn(&tmem_O1, 256);
    }

    uint32_t tmem_S, tmem_O;
    uint32_t cta_offset = cluster_rank_fn();
    if (cta_offset == 0) {
        tmem_S = tmem_S0;
        tmem_O = tmem_O0;
    } else {
        tmem_S = tmem_S1;
        tmem_O = tmem_O1;
    }

    uint32_t batch = blockIdx.y / heads;
    uint32_t head = blockIdx.y % heads;
    uint32_t c1_base = batch * heads * S + head * S + blockIdx.x * 128;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_q, 16384);
        tma_load_2d_fn(&tma_Q, bar_q, smem_Q, 0, c1_base + cta_offset * 64);
        tma_load_2d_fn(&tma_Q, bar_q, smem_Q + 4096, 64, c1_base + cta_offset * 64);
    }
    mbarrier_wait_fn(bar_q, 0);

    float m_prev = -1e20f;
    float l_prev = 0.0f;
    float scale_sqrt_d = 1.0f / sqrtf(128);

    int kv_phase = 0;
    for (int s_idx = 0; s_idx < S; s_idx += 128) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_kv, 32768);
            tma_load_2d_fn(&tma_K, bar_kv, smem_K, 0, batch * heads * S + head * S + s_idx);
            tma_load_2d_fn(&tma_K, bar_kv, smem_K + 4096, 64, batch * heads * S + head * S + s_idx);
            tma_load_2d_fn(&tma_V, bar_kv, smem_V, 0, batch * heads * S + head * S + s_idx);
            tma_load_2d_fn(&tma_V, bar_kv, smem_V + 4096, 64, batch * heads * S + head * S + s_idx);
        }
        mbarrier_wait_fn(bar_kv, kv_phase);
        kv_phase ^= 1;

        uint32_t accum = (s_idx == 0) ? 0 : 1;
        uint32_t lane_offset = cta_offset << 9;
        for (int k = 0; k < 64; k += 16) {
            uint64_t desc_q = make_smem_desc_sm100_fn(smem_Q + k, 16, 1024);
            uint64_t desc_k = make_smem_desc_sm100_fn(smem_K + k, 16, 1024);
            uint32_t idesc_qkt = make_instr_desc_fn(128, 128);
            if (threadIdx.x == 0) {
                umma_f16_cg2_fn(tmem_S, desc_q, desc_k, idesc_qkt, accum);
            }
            accum = 1;
        }
        for (int k = 0; k < 64; k += 16) {
            uint64_t desc_q = make_smem_desc_sm100_fn(smem_Q + 4096 + k, 16, 1024);
            uint64_t desc_k = make_smem_desc_sm100_fn(smem_K + 4096 + k, 16, 1024);
            uint32_t idesc_qkt = make_instr_desc_fn(128, 128);
            if (threadIdx.x == 0) {
                umma_f16_cg2_fn(tmem_S, desc_q, desc_k, idesc_qkt, 1);
            }
        }
        
        if (threadIdx.x == 0) {
            umma_commit_2sm_fn(bar_kv);
        }
        mbarrier_wait_fn(bar_kv, kv_phase);
        kv_phase ^= 1;
        tmem_load_fence_fn();

        float S_val[128];
        uint32_t col = 0;
        for (int r_iter = 0; r_iter < 8; ++r_iter) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S + lane_offset + col, &r0, &r1, &r2, &r3);
            S_val[col+0] = __uint_as_float(r0);
            S_val[col+1] = __uint_as_float(r1);
            S_val[col+2] = __uint_as_float(r2);
            S_val[col+3] = __uint_as_float(r3);
            col += 4;
        }

        float local_m = -1e20f;
        uint32_t m_row = cta_offset * 64 + threadIdx.x;
        for(int c = 0; c < 128; ++c) {
            uint32_t global_col = s_idx + c;
            if (global_col < S && m_row < S) {
                S_val[c] *= scale_sqrt_d;
                local_m = fmaxf(local_m, S_val[c]);
            }
        }
        
        __shared__ float m_local[128];
        if (threadIdx.x < 128) {
            m_local[threadIdx.x] = local_m;
        }
        __syncthreads();

        float m_new = m_prev;
        if (threadIdx.x < 128) {
            m_new = fmaxf(m_new, m_local[threadIdx.x]);
        }

        float local_sum = 0.0f;
        for(int c = 0; c < 128; ++c) {
            uint32_t global_col = s_idx + c;
            float p = 0.0f;
            if (global_col < S && m_row < S) {
                float val = S_val[c] - m_new;
                p = exp_fn(val);
            }
            local_sum += p;
            
            if (c < 64) {
                int x = c / 8;
                int x_swizzled = x ^ (m_row % 8);
                int c_swizzled = x_swizzled * 8 + (c % 8);
                uint32_t c_idx = m_row * 64 + c_swizzled;
                smem_P[c_idx] = __float2bfloat16(p);
            } else {
                int x = (c - 64) / 8;
                int x_swizzled = x ^ (m_row % 8);
                int c_swizzled = x_swizzled * 8 + ((c - 64) % 8);
                uint32_t c_idx = 4096 + m_row * 64 + c_swizzled;
                smem_P[c_idx] = __float2bfloat16(p);
            }
        }
        
        if (threadIdx.x < 128) {
            float m_prev_val = m_prev;
            float inv_m_diff = exp_fn(m_local[threadIdx.x] - m_new);
            float exp_m_diff = exp_fn(m_prev_val - m_new);
            
            uint32_t o_col = 0;
            for(int r_iter = 0; r_iter < 8; ++r_iter) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_O + (threadIdx.x << 4) + o_col, &r0, &r1, &r2, &r3);
                r0 = __float_as_uint(__uint_as_float(r0) * exp_m_diff);
                r1 = __float_as_uint(__uint_as_float(r1) * exp_m_diff);
                r2 = __float_as_uint(__uint_as_float(r2) * exp_m_diff);
                r3 = __float_as_uint(__uint_as_float(r3) * exp_m_diff);
                tmem_store_4x_fn(tmem_O + (threadIdx.x << 4) + o_col, r0, r1, r2, r3);
                o_col += 4;
            }
            for(int r_iter = 0; r_iter < 8; ++r_iter) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_O + (threadIdx.x << 4) + 32 + o_col, &r0, &r1, &r2, &r3);
                r0 = __float_as_uint(__uint_as_float(r0) * exp_m_diff);
                r1 = __float_as_uint(__uint_as_float(r1) * exp_m_diff);
                r2 = __float_as_uint(__uint_as_float(r2) * exp_m_diff);
                r3 = __float_as_uint(__uint_as_float(r3) * exp_m_diff);
                tmem_store_4x_fn(tmem_O + (threadIdx.x << 4) + 32 + o_col, r0, r1, r2, r3);
            }

            for(int c = 0; c < 128; ++c) {
                float p_val = __bfloat162float(smem_P[(c < 64 ? 0 : 4096) + threadIdx.x * 64 + (((c < 64 ? c : c - 64) / 8) ^ (threadIdx.x % 8)) * 8 + (c % 8)]);
                p_val *= inv_m_diff;
                if (c < 64) {
                    int x = c / 8;
                    int x_swizzled = x ^ (threadIdx.x % 8);
                    int c_swizzled = x_swizzled * 8 + (c % 8);
                    uint32_t c_idx = threadIdx.x * 64 + c_swizzled;
                    smem_P[c_idx] = __float2bfloat16(p_val);
                } else {
                    int x = (c - 64) / 8;
                    int x_swizzled = x ^ (threadIdx.x % 8);
                    int c_swizzled = x_swizzled * 8 + ((c - 64) % 8);
                    uint32_t c_idx = 4096 + threadIdx.x * 64 + c_swizzled;
                    smem_P[c_idx] = __float2bfloat16(p_val);
                }
            }
            l_prev = local_sum * inv_m_diff + l_prev * exp_m_diff;
            m_prev = m_new;
        }
        __syncthreads();
        fence_async_shared_fn();

        for (int k = 0; k < 64; k += 16) {
            uint64_t desc_p = make_smem_desc_sm100_fn(smem_P + k, 16, 1024);
            uint64_t desc_v = make_smem_desc_sm100_fn(smem_V + k, 8192, 1024);
            uint32_t idesc_pv = make_instr_desc_fn(128, 64, 1);
            if (threadIdx.x == 0) {
                umma_f16_cg2_fn(tmem_O, desc_p, desc_v, idesc_pv, 1);
            }
        }
        for (int k = 0; k < 64; k += 16) {
            uint64_t desc_p = make_smem_desc_sm100_fn(smem_P + 4096 + k, 16, 1024);
            uint64_t desc_v = make_smem_desc_sm100_fn(smem_V + 4096 + k, 8192, 1024);
            uint32_t idesc_pv = make_instr_desc_fn(128, 64, 1);
            if (threadIdx.x == 0) {
                umma_f16_cg2_fn(tmem_O + 1024, desc_p, desc_v, idesc_pv, 1);
            }
        }
        
        if (threadIdx.x == 0) {
            umma_commit_2sm_fn(bar_kv);
        }
        mbarrier_wait_fn(bar_kv, kv_phase);
        kv_phase ^= 1;
        tmem_load_fence_fn();
        
        if (threadIdx.x == 0) {
            cluster_sync_fn();
        }
    }

    float O_val[128];
    uint32_t col = 0;
    for(int r_iter = 0; r_iter < 8; ++r_iter) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_O + (threadIdx.x << 4) + col, &r0, &r1, &r2, &r3);
        O_val[col+0] = __uint_as_float(r0);
        O_val[col+1] = __uint_as_float(r1);
        O_val[col+2] = __uint_as_float(r2);
        O_val[col+3] = __uint_as_float(r3);
        col += 4;
    }
    for(int r_iter = 0; r_iter < 8; ++r_iter) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_O + (threadIdx.x << 4) + 32 + col, &r0, &r1, &r2, &r3);
        O_val[col+0] = __uint_as_float(r0);
        O_val[col+1] = __uint_as_float(r1);
        O_val[col+2] = __uint_as_float(r2);
        O_val[col+3] = __uint_as_float(r3);
        col += 4;
    }

    float out_O[128];
    for(int k = 0; k < 128; ++k) {
        out_O[k] = O_val[k] / l_prev;
    }

    uint32_t BN = 128;
    uint32_t N = 128;
    uint32_t m_idx = c1_base + cta_offset * 64 + threadIdx.x;
    uint32_t n_base = 0;
    for (uint32_t col = 0; col < BN; col += 4) {
        float f0 = out_O[col+0];
        float f1 = out_O[col+1];
        float f2 = out_O[col+2];
        float f3 = out_O[col+3];
        uint32_t nc = n_base + col;
        __nv_bfloat16* out = O + (uint64_t)batch * heads * S * N + (uint64_t)head * S * N + (uint64_t)threadIdx.x * N + nc;
        if (threadIdx.x < S && nc < N) {
            out[0] = __float2bfloat16(f0);
            out[1] = __float2bfloat16(f1);
            out[2] = __float2bfloat16(f2);
            out[3] = __float2bfloat16(f3);
        }
    }

    if (threadIdx.x < 128) {
        uint32_t seq_idx = c1_base + threadIdx.x;
        LSE[batch * heads * S + seq_idx] = m_prev + logf(l_prev);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    uint64_t B = Q.size(0);
    uint64_t H = Q.size(1);
    uint64_t S = Q.size(2);
    uint64_t D = Q.size(3);

    CUtensorMap tma_Q, tma_K, tma_V;
    CUresult res;

    res = cuTensorMapEncodeTiled(&tma_Q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, Q.data_ptr(), (const cuuint64_t[]){D, B * H * S}, (const cuuint64_t[]){D * 2}, (const cuuint32_t[]){64, 64}, (const cuuint32_t[]){1, 1}, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA Q failed\n"); exit(1); }
    
    res = cuTensorMapEncodeTiled(&tma_K, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, K.data_ptr(), (const cuuint64_t[]){D, B * H * S}, (const cuuint64_t[]){D * 2}, (const cuuint32_t[]){64, 64}, (const cuuint32_t[]){1, 1}, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA K failed\n"); exit(1); }
    
    res = cuTensorMapEncodeTiled(&tma_V, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, V.data_ptr(), (const cuuint64_t[]){D, B * H * S}, (const cuuint64_t[]){D * 2}, (const cuuint32_t[]){64, 64}, (const cuuint32_t[]){1, 1}, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA V failed\n"); exit(1); }

    int smem_bytes = 98336;
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3((S + 127) / 128, B * H, 1);
    config.blockDim = dim3(128);
    config.dynamicSmemBytes = smem_bytes;
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaFuncSetAttribute(fa4_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    CUDA_CHECK(cudaLaunchKernelEx(&config, fa4_fwd_kernel, tma_Q, tma_K, tma_V, static_cast<__nv_bfloat16*>(O.data_ptr()), static_cast<float*>(LSE.data_ptr()), static_cast<uint32_t>(S), static_cast<uint32_t>(H)));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(config.stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_example_cuda