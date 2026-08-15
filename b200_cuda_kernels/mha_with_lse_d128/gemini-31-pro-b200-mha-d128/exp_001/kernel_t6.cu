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

inline CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
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

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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

__device__ __forceinline__ void wg_barrier_sync(int wg_idx) {
    if (wg_idx == 0) {
        asm volatile("barrier.sync 0, 128;");
    } else {
        asm volatile("barrier.sync 1, 128;");
    }
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

__device__ __forceinline__ void umma_f16_tmem_A_cg1_fn(
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
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo, int swizzle_type) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)(base_offset) << 49;
    d |= (uint64_t)swizzle_type << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_desc_Q_K(void* ptr, uint32_t offset_bytes) {
    return make_smem_desc((char*)ptr + offset_bytes, 1, 1024, 2);
}

__device__ __forceinline__ uint64_t make_desc_V(void* ptr, uint32_t offset_bytes) {
    return make_smem_desc((char*)ptr + offset_bytes, 16384, 1024, 2);
}

__device__ __forceinline__ uint32_t make_instr_desc_qk(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (0u << 15);
    d |= (0u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_pv(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (0u << 15);
    d |= (1u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
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

__device__ __forceinline__ void tmem_store_8x_fn(uint32_t col,
    uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3,
    uint32_t r4, uint32_t r5, uint32_t r6, uint32_t r7) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
   :: "r"(col), "r"(r0),"r"(r1),"r"(r2),"r"(r3),
      "r"(r4),"r"(r5),"r"(r6),"r"(r7) : "memory");
}

__device__ __forceinline__ void tmem_store_fence_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ float fma_exp2f(float x) {
    if (x < -126.0f) return 0.0f;
    int n_int;
    asm("cvt.rmi.s32.f32 %0, %1;" : "=r"(n_int) : "f"(x));
    float n;
    asm("cvt.rn.f32.s32 %0, %1;" : "=f"(n) : "r"(n_int));
    float f = x - n;
    float p = 0.0771f;
    p = fmaf(p, f, 0.2276f);
    p = fmaf(p, f, 0.6951f);
    p = fmaf(p, f, 1.0f);
    float exp2n = __uint_as_float((n_int + 127) << 23);
    return p * exp2n;
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

__global__ __launch_bounds__(256, 1) void mha_fwd_sm100_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S,
    float scale
) {
    setmaxnreg_inc_sync_fn<248>();
    
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_start = blockIdx.x * 2;
    
    int tid = threadIdx.x;
    int wg_idx = tid / 128;
    int lane = tid % 128;
    
    int q_step = q_start + wg_idx;
    int q_valid = S - q_step * 128;
    if (q_valid > 128) q_valid = 128;
    
    extern __shared__ __align__(1024) char smem[];
    char* smem_Q0 = smem;
    char* smem_Q1 = smem_Q0 + 32768;
    char* smem_K0 = smem_Q1 + 32768;
    char* smem_V0 = smem_K0 + 32768;
    char* smem_K1 = smem_V0 + 32768;
    char* smem_V1 = smem_K1 + 32768;
    
    __shared__ uint32_t smem_tmem_base;
    if (tid < 32) {
        tmem_alloc_cg1_fn(&smem_tmem_base, 512);
    }
    
    __shared__ uint64_t mbar_tma_Q[1];
    __shared__ uint64_t mbar_tma_KV[2];
    __shared__ uint64_t mbar_mma[2];
    
    if (tid == 0) {
        init_smem_barrier_fn(&mbar_tma_Q[0], 1);
        init_smem_barrier_fn(&mbar_tma_KV[0], 1);
        init_smem_barrier_fn(&mbar_tma_KV[1], 1);
        init_smem_barrier_fn(&mbar_mma[0], 1);
        init_smem_barrier_fn(&mbar_mma[1], 1);
    }
    
    float r_acc[128];
    #pragma unroll
    for (int c = 0; c < 128; c++) r_acc[c] = 0.0f;
    
    __syncthreads();
    if (tid == 0) fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t tmem_base = smem_tmem_base;
    uint32_t tmem_O_tmp = tmem_base + wg_idx * 256;
    uint32_t tmem_P     = tmem_base + wg_idx * 256 + 128;
    
    uint32_t phase_tma_KV[2] = {0, 0};
    uint32_t phase_mma = 0;
    
    int num_KV_blocks = (S + 127) / 128;
    
    if (tid == 0) {
        int q0_v = (S - q_start * 128 > 0);
        int q1_v = (S - (q_start + 1) * 128 > 0);
        uint32_t tx = (q0_v + q1_v) * 32768;
        if (tx > 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_tma_Q[0], tx);
            if (q0_v) {
                tma_load_2d_fn(&tma_Q, &mbar_tma_Q[0], smem_Q0, 0, (b * H + h) * S + q_start * 128);
                tma_load_2d_fn(&tma_Q, &mbar_tma_Q[0], smem_Q0 + 16384, 64, (b * H + h) * S + q_start * 128);
            }
            if (q1_v) {
                tma_load_2d_fn(&tma_Q, &mbar_tma_Q[0], smem_Q1, 0, (b * H + h) * S + (q_start + 1) * 128);
                tma_load_2d_fn(&tma_Q, &mbar_tma_Q[0], smem_Q1 + 16384, 64, (b * H + h) * S + (q_start + 1) * 128);
            }
        }
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_tma_KV[0], 65536);
        tma_load_2d_fn(&tma_K, &mbar_tma_KV[0], smem_K0, 0, (b * H + h) * S + 0);
        tma_load_2d_fn(&tma_K, &mbar_tma_KV[0], smem_K0 + 16384, 64, (b * H + h) * S + 0);
        tma_load_2d_fn(&tma_V, &mbar_tma_KV[0], smem_V0, 0, (b * H + h) * S + 0);
        tma_load_2d_fn(&tma_V, &mbar_tma_KV[0], smem_V0 + 16384, 64, (b * H + h) * S + 0);
        
        if (num_KV_blocks > 1) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_tma_KV[1], 65536);
            tma_load_2d_fn(&tma_K, &mbar_tma_KV[1], smem_K1, 0, (b * H + h) * S + 128);
            tma_load_2d_fn(&tma_K, &mbar_tma_KV[1], smem_K1 + 16384, 64, (b * H + h) * S + 128);
            tma_load_2d_fn(&tma_V, &mbar_tma_KV[1], smem_V1, 0, (b * H + h) * S + 128);
            tma_load_2d_fn(&tma_V, &mbar_tma_KV[1], smem_V1 + 16384, 64, (b * H + h) * S + 128);
        }
    }
    
    if (lane == 0) {
        mbarrier_wait_fn(&mbar_tma_Q[0], 0);
    }
    wg_barrier_sync(wg_idx);
    if (lane == 0) fence_proxy_async_fn();
    
    char* my_smem_Q = (wg_idx == 0) ? smem_Q0 : smem_Q1;
    uint32_t idesc_QK = make_instr_desc_qk(128, 128);
    uint32_t idesc_PV = make_instr_desc_pv(128, 128);
    
    if (q_valid > 0) {
        if (lane == 0) {
            for (int k = 0; k < 128; k += 16) {
                uint32_t tile_offset = (k >= 64) ? 16384 : 0;
                uint32_t k_in_tile = (k >= 64) ? k - 64 : k;
                uint64_t desc_Q_k = make_desc_Q_K(my_smem_Q + tile_offset, k_in_tile * 2);
                uint64_t desc_K_k = make_desc_Q_K(smem_K0 + tile_offset, k_in_tile * 2);
                umma_f16_cg1_fn((0<<16) | tmem_P, desc_Q_k, desc_K_k, idesc_QK, (k == 0) ? 0 : 1);
            }
            umma_commit_cg1_fn(&mbar_mma[wg_idx]);
        }
    }
    
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float m_prev = m_i;
    
    for (int step = 0; step < num_KV_blocks; step++) {
        int curr = step % 2;
        int next = (step + 1) % 2;
        char* smem_K_curr = (curr == 0) ? smem_K0 : smem_K1;
        char* smem_V_curr = (curr == 0) ? smem_V0 : smem_V1;
        char* smem_K_next = (next == 0) ? smem_K0 : smem_K1;
        int k_valid = S - step * 128;
        
        if (lane == 0) mbarrier_wait_fn(&mbar_tma_KV[curr], phase_tma_KV[curr]);
        wg_barrier_sync(wg_idx);
        if (lane == 0) fence_proxy_async_fn();
        
        if (q_valid > 0) {
            if (lane == 0) mbarrier_wait_fn(&mbar_mma[wg_idx], phase_mma);
            
            float m_curr = m_prev;
            #pragma unroll
            for (int c = 0; c < 128; c += 8) {
                uint32_t r_val[8];
                tmem_load_8x_fn(tmem_P + c, &r_val[0], &r_val[1], &r_val[2], &r_val[3], &r_val[4], &r_val[5], &r_val[6], &r_val[7]);
                tmem_load_fence_fn();
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    float val = __uint_as_float(r_val[i]);
                    val = (c + i >= k_valid) ? -INFINITY : (val * scale);
                    m_curr = fmaxf(m_curr, val);
                }
            }
            
            float l_curr_new = 0.0f;
            #pragma unroll
            for (int c = 0; c < 128; c += 16) {
                uint32_t r_val[16];
                tmem_load_8x_fn(tmem_P + c,     &r_val[0], &r_val[1], &r_val[2], &r_val[3], &r_val[4], &r_val[5], &r_val[6], &r_val[7]);
                tmem_load_8x_fn(tmem_P + c + 8, &r_val[8], &r_val[9], &r_val[10], &r_val[11], &r_val[12], &r_val[13], &r_val[14], &r_val[15]);
                tmem_load_fence_fn();
                
                uint32_t bf32[8];
                #pragma unroll
                for (int i = 0; i < 16; i += 2) {
                    float v0 = __uint_as_float(r_val[i]);
                    v0 = (c + i >= k_valid) ? -INFINITY : (v0 * scale);
                    v0 = fma_exp2f((v0 - m_curr) * 1.44269504f);
                    l_curr_new += v0;
                    
                    float v1 = __uint_as_float(r_val[i+1]);
                    v1 = (c + i + 1 >= k_valid) ? -INFINITY : (v1 * scale);
                    v1 = fma_exp2f((v1 - m_curr) * 1.44269504f);
                    l_curr_new += v1;
                    
                    bf32[i/2] = pack_bf16_fn(__float_as_uint(v0), __float_as_uint(v1));
                }
                tmem_store_8x_fn(tmem_P + (c/2), bf32[0], bf32[1], bf32[2], bf32[3], bf32[4], bf32[5], bf32[6], bf32[7]);
            }
            tmem_store_fence_fn();
            
            wg_barrier_sync(wg_idx);
            tcgen05_fence_after_fn();
            
            if (lane == 0) {
                for (int k = 0; k < 128; k += 16) {
                    uint32_t a_col = tmem_P + (k / 2);
                    uint64_t desc_V_k = make_desc_V(smem_V_curr, k * 128);
                    umma_f16_tmem_A_cg1_fn((0<<16) | tmem_O_tmp, (0<<16) | a_col, desc_V_k, idesc_PV, (k == 0) ? 0 : 1);
                }
                umma_commit_cg1_fn(&mbar_mma[wg_idx]);
            }
            if (lane == 0) mbarrier_wait_fn(&mbar_mma[wg_idx], phase_mma ^ 1);
            phase_mma ^= 1;
            
            if (step + 1 < num_KV_blocks) {
                if (lane == 0) mbarrier_wait_fn(&mbar_tma_KV[next], phase_tma_KV[next]);
                wg_barrier_sync(wg_idx);
                if (lane == 0) fence_proxy_async_fn();
                
                if (lane == 0) {
                    for (int k = 0; k < 128; k += 16) {
                        uint32_t tile_offset = (k >= 64) ? 16384 : 0;
                        uint32_t k_in_tile = (k >= 64) ? k - 64 : k;
                        uint64_t desc_Q_k = make_desc_Q_K(my_smem_Q + tile_offset, k_in_tile * 2);
                        uint64_t desc_K_k = make_desc_Q_K(smem_K_next + tile_offset, k_in_tile * 2);
                        umma_f16_cg1_fn((0<<16) | tmem_P, desc_Q_k, desc_K_k, idesc_QK, (k == 0) ? 0 : 1);
                    }
                    umma_commit_cg1_fn(&mbar_mma[wg_idx]);
                }
            }
            
            float rescale = fma_exp2f((m_prev - m_curr) * 1.44269504f);
            float l_curr = l_i * rescale + l_curr_new;
            
            #pragma unroll
            for (int c = 0; c < 128; c += 8) {
                uint32_t o_val[8];
                tmem_load_8x_fn(tmem_O_tmp + c, &o_val[0], &o_val[1], &o_val[2], &o_val[3], &o_val[4], &o_val[5], &o_val[6], &o_val[7]);
                tmem_load_fence_fn();
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    r_acc[c+i] = r_acc[c+i] * rescale + __uint_as_float(o_val[i]);
                }
            }
            
            m_prev = m_curr;
            m_i = m_curr;
            l_i = l_curr;
        }
        
        __syncthreads();
        if (tid == 0 && step + 2 < num_KV_blocks) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_tma_KV[curr], 65536);
            tma_load_2d_fn(&tma_K, &mbar_tma_KV[curr], smem_K_curr, 0, (b * H + h) * S + (step + 2) * 128);
            tma_load_2d_fn(&tma_K, &mbar_tma_KV[curr], smem_K_curr + 16384, 64, (b * H + h) * S + (step + 2) * 128);
            tma_load_2d_fn(&tma_V, &mbar_tma_KV[curr], smem_V_curr, 0, (b * H + h) * S + (step + 2) * 128);
            tma_load_2d_fn(&tma_V, &mbar_tma_KV[curr], smem_V_curr + 16384, 64, (b * H + h) * S + (step + 2) * 128);
        }
        if (step + 2 < num_KV_blocks) {
            phase_tma_KV[curr] ^= 1;
        }
    }
    
    char* my_smem_O = (wg_idx == 0) ? smem_K0 : smem_K1;
    if (q_valid > 0) {
        float inv_l = 1.0f / l_i;
        for (int c = 0; c < 128; c += 8) {
            float v0 = r_acc[c] * inv_l;
            float v1 = r_acc[c+1] * inv_l;
            float v2 = r_acc[c+2] * inv_l;
            float v3 = r_acc[c+3] * inv_l;
            float v4 = r_acc[c+4] * inv_l;
            float v5 = r_acc[c+5] * inv_l;
            float v6 = r_acc[c+6] * inv_l;
            float v7 = r_acc[c+7] * inv_l;
            uint32_t bf32_0 = pack_bf16_fn(__float_as_uint(v0), __float_as_uint(v1));
            uint32_t bf32_1 = pack_bf16_fn(__float_as_uint(v2), __float_as_uint(v3));
            uint32_t bf32_2 = pack_bf16_fn(__float_as_uint(v4), __float_as_uint(v5));
            uint32_t bf32_3 = pack_bf16_fn(__float_as_uint(v6), __float_as_uint(v7));
            
            uint4 val;
            val.x = bf32_0; val.y = bf32_1; val.z = bf32_2; val.w = bf32_3;
            ((uint4*)my_smem_O)[(lane * 128 + c) / 8] = val;
        }
    }
    __syncthreads();
    
    if (q_valid > 0) {
        uint4* g_O = (uint4*)(O + (b * H + h) * S * 128 + q_step * 128 * 128);
        uint4* s_O = (uint4*)my_smem_O;
        int total_uint4 = q_valid * 128 / 8;
        for(int i = lane; i < total_uint4; i += 128) {
            g_O[i] = s_O[i];
        }
        
        if (lane < q_valid) {
            float lse = m_i + logf(l_i);
            LSE[(b * H + h) * S + q_step * 128 + lane] = lse;
        }
    }
    
    __syncthreads();
    if (tid < 32) {
        tmem_dealloc_cg1_fn(smem_tmem_base, 512);
    }
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    
    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, B*H*S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, B*H*S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, B*H*S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    int num_q_blocks = (S + 127) / 128;
    int grid_x = (num_q_blocks + 1) / 2;
    dim3 grid(grid_x, H, B);
    dim3 block(256);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_sm100_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 196608));
    
    mha_fwd_sm100_kernel<<<grid, block, 196608, stream>>>(
        tma_Q, tma_K, tma_V,
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        B, H, S, 1.0f / sqrtf(128.0f)
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}