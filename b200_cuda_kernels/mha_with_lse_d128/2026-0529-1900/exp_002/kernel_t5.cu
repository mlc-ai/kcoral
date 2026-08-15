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
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace fa4_sm100 {

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

__device__ __forceinline__ void tma_load_2d_cg1_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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

__device__ __forceinline__ void tmem_load_32x_fn(uint32_t col, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                 : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                   "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
                   "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
                   "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
                 : "r"(col));
}

__device__ __forceinline__ void tmem_store_32x_fn(uint32_t col, uint32_t* r) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
                 :: "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),
                    "r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),"r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]),
                    "r"(r[16]),"r"(r[17]),"r"(r[18]),"r"(r[19]),"r"(r[20]),"r"(r[21]),"r"(r[22]),"r"(r[23]),
                    "r"(r[24]),"r"(r[25]),"r"(r[26]),"r"(r[27]),"r"(r[28]),"r"(r[29]),"r"(r[30]),"r"(r[31]),
                    "r"(col));
}

__device__ __forceinline__ void tmem_store_16x_fn(uint32_t col, uint32_t* r) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};"
                 :: "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),
                    "r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),"r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]),
                    "r"(col));
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

__device__ __forceinline__ void umma_f16_cg1_tmem_A_fn(
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

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count) : "memory");
}

extern __shared__ __align__(1024) uint8_t smem_pool[];

__global__ __launch_bounds__(256, 1) void fa4_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ o_ptr,
    float* __restrict__ lse_ptr,
    int B, int H, int S, int D)
{
    setmaxnreg_inc_sync_fn<256>();

    __nv_bfloat16* Q_smem[2][2]; 
    __nv_bfloat16* K_smem[2][2];
    __nv_bfloat16* V_smem[2][2];
    
    Q_smem[0][0] = (__nv_bfloat16*)smem_pool;
    Q_smem[0][1] = Q_smem[0][0] + 128 * 64;
    Q_smem[1][0] = Q_smem[0][1] + 128 * 64;
    Q_smem[1][1] = Q_smem[1][0] + 128 * 64;

    K_smem[0][0] = Q_smem[1][1] + 128 * 64;
    K_smem[0][1] = K_smem[0][0] + 128 * 64;
    K_smem[1][0] = K_smem[0][1] + 128 * 64;
    K_smem[1][1] = K_smem[1][0] + 128 * 64;

    V_smem[0][0] = K_smem[1][1] + 128 * 64;
    V_smem[0][1] = V_smem[0][0] + 128 * 64;
    V_smem[1][0] = V_smem[0][1] + 128 * 64;
    V_smem[1][1] = V_smem[1][0] + 128 * 64;

    uint64_t* mbar = (uint64_t*)(V_smem[1][1] + 128 * 64);
    uint64_t* mbar_umma = mbar + 2;

    int b = blockIdx.z;
    int h = blockIdx.y;
    int s_q = blockIdx.x * 256;

    if (s_q >= S) return;

    uint32_t wg_id = threadIdx.x / 128;
    uint32_t lane_id = threadIdx.x % 128;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar[0], 1);
        init_smem_barrier_fn(&mbar[1], 1);
        init_smem_barrier_fn(&mbar_umma[0], 1);
        init_smem_barrier_fn(&mbar_umma[1], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    __shared__ uint32_t tmem_alloc_smem;
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&tmem_alloc_smem, 512);
    }
    __syncthreads();
    uint32_t tmem_alloc_addr = tmem_alloc_smem;

    uint32_t O_tmem = tmem_alloc_addr + wg_id * 128;
    uint32_t S_tmem = tmem_alloc_addr + 256 + wg_id * 128;

    uint32_t zero[32];
    #pragma unroll
    for (int i=0; i<32; i++) zero[i] = 0;
    #pragma unroll 4
    for (int c = 0; c < 128; c += 32) {
        tmem_store_32x_fn(O_tmem + c, zero);
    }
    
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float scale = 0.08838834764831843f;
    float scale_log2 = scale * 1.4426950408889634f;

    int buf_idx = 0;
    int phase[2] = {0, 0};
    int phase_umma[2] = {0, 0};

    if (threadIdx.x == 0) {
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        int tx_bytes = (128 * 64 * 2) * 4; 
        tx_bytes += (128 * 64 * 2) * 2;
        if (s_q + 128 < S) tx_bytes += (128 * 64 * 2) * 2;
        
        mbarrier_arrive_and_expect_tx_fn(&mbar[0], tx_bytes);
        
        tma_load_2d_cg1_fn(&tma_Q, &mbar[0], Q_smem[0][0], 0, b * H * S + h * S + s_q);
        tma_load_2d_cg1_fn(&tma_Q, &mbar[0], Q_smem[0][1], 64, b * H * S + h * S + s_q);
        
        if (s_q + 128 < S) {
            tma_load_2d_cg1_fn(&tma_Q, &mbar[0], Q_smem[1][0], 0, b * H * S + h * S + s_q + 128);
            tma_load_2d_cg1_fn(&tma_Q, &mbar[0], Q_smem[1][1], 64, b * H * S + h * S + s_q + 128);
        }
        
        int outer_coord_kv_0 = b * H * S + h * S + 0;
        tma_load_2d_cg1_fn(&tma_K, &mbar[0], K_smem[0][0], 0, outer_coord_kv_0);
        tma_load_2d_cg1_fn(&tma_K, &mbar[0], K_smem[0][1], 64, outer_coord_kv_0);
        tma_load_2d_cg1_fn(&tma_V, &mbar[0], V_smem[0][0], 0, outer_coord_kv_0);
        tma_load_2d_cg1_fn(&tma_V, &mbar[0], V_smem[0][1], 64, outer_coord_kv_0);
    }

    uint32_t idesc_S = 0;
    idesc_S |= (1u << 4) | (1u << 7) | (1u << 10) | (0u << 15) | (0u << 16) | ((128 / 8) << 17) | ((128 / 16) << 24);
    
    uint32_t idesc_O = 0;
    idesc_O |= (1u << 4) | (1u << 7) | (1u << 10) | (0u << 15) | (1u << 16) | ((64 / 8) << 17) | ((128 / 16) << 24);

    for (int n = 0; n < S; n += 128) {
        mbarrier_wait_fn(&mbar[buf_idx], phase[buf_idx]);
        
        int next_n = n + 128;
        int next_buf = 1 - buf_idx;
        if (next_n < S) {
            if (threadIdx.x == 0) {
                int tx_bytes = (128 * 64 * 2) * 4;
                mbarrier_arrive_and_expect_tx_fn(&mbar[next_buf], tx_bytes);
                int outer_coord_kv = b * H * S + h * S + next_n;
                tma_load_2d_cg1_fn(&tma_K, &mbar[next_buf], K_smem[next_buf][0], 0, outer_coord_kv);
                tma_load_2d_cg1_fn(&tma_K, &mbar[next_buf], K_smem[next_buf][1], 64, outer_coord_kv);
                tma_load_2d_cg1_fn(&tma_V, &mbar[next_buf], V_smem[next_buf][0], 0, outer_coord_kv);
                tma_load_2d_cg1_fn(&tma_V, &mbar[next_buf], V_smem[next_buf][1], 64, outer_coord_kv);
            }
        }

        if (s_q + wg_id * 128 < S) {
            uint64_t desc_Q0 = make_smem_desc_sm100_fn(Q_smem[wg_id][0], 1, 1024);
            uint64_t desc_Q1 = make_smem_desc_sm100_fn(Q_smem[wg_id][1], 1, 1024);
            uint64_t desc_K0 = make_smem_desc_sm100_fn(K_smem[buf_idx][0], 1, 1024);
            uint64_t desc_K1 = make_smem_desc_sm100_fn(K_smem[buf_idx][1], 1, 1024);

            if (lane_id == 0) {
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                for (int k = 0; k < 4; ++k) {
                    umma_f16_cg1_fn(S_tmem, desc_Q0 + k * 2, desc_K0 + k * 2, idesc_S, (k == 0) ? 0 : 1);
                }
                for (int k = 0; k < 4; ++k) {
                    umma_f16_cg1_fn(S_tmem, desc_Q1 + k * 2, desc_K1 + k * 2, idesc_S, 1);
                }
                umma_commit_cg1_fn(&mbar_umma[wg_id]);
            }
            mbarrier_wait_fn(&mbar_umma[wg_id], phase_umma[wg_id]);
            phase_umma[wg_id] ^= 1;

            float row_max = -INFINITY;
            #pragma unroll 4
            for (int c = 0; c < 128; c += 32) {
                uint32_t rS[32];
                tmem_load_32x_fn(S_tmem + c, rS);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int i = 0; i < 32; ++i) {
                    float f = __uint_as_float(rS[i]);
                    if (n + c + i >= S) f = -INFINITY;
                    f *= scale_log2;
                    row_max = fmaxf(row_max, f);
                }
            }

            float old_max = m_i;
            m_i = fmaxf(m_i, row_max);
            float scale_O = (m_i == -INFINITY) ? 0.0f : fast_exp2f_fn(old_max - m_i);

            #pragma unroll 4
            for (int c = 0; c < 128; c += 32) {
                uint32_t rO[32];
                tmem_load_32x_fn(O_tmem + c, rO);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int i = 0; i < 32; ++i) {
                    float f = __uint_as_float(rO[i]) * scale_O;
                    rO[i] = __float_as_uint(f);
                }
                tmem_store_32x_fn(O_tmem + c, rO);
            }

            float row_sum = 0;
            #pragma unroll 4
            for (int c = 0; c < 128; c += 32) {
                uint32_t rS[32];
                tmem_load_32x_fn(S_tmem + c, rS);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                uint32_t rP[16];
                #pragma unroll
                for (int i = 0; i < 32; i += 2) {
                    float f0 = __uint_as_float(rS[i]);
                    if (n + c + i >= S) f0 = -INFINITY;
                    f0 = fast_exp2f_fn(f0 * scale_log2 - m_i);
                    
                    float f1 = __uint_as_float(rS[i+1]);
                    if (n + c + i + 1 >= S) f1 = -INFINITY;
                    f1 = fast_exp2f_fn(f1 * scale_log2 - m_i);
                    
                    row_sum += f0 + f1;
                    rP[i/2] = pack_bf16_fn(__float_as_uint(f0), __float_as_uint(f1));
                }
                tmem_store_16x_fn(S_tmem + c/2, rP);
            }
            l_i = l_i * scale_O + row_sum;

            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
            named_barrier_sync_fn(wg_id + 1, 128);

            uint64_t desc_V0 = make_smem_desc_sm100_fn(V_smem[buf_idx][0], 1024, 1024);
            uint64_t desc_V1 = make_smem_desc_sm100_fn(V_smem[buf_idx][1], 1024, 1024);
            
            if (lane_id == 0) {
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                for (int k = 0; k < 8; ++k) {
                    umma_f16_cg1_tmem_A_fn(O_tmem, S_tmem + k * 8, desc_V0 + k * 128, idesc_O, 1);
                }
                for (int k = 0; k < 8; ++k) {
                    umma_f16_cg1_tmem_A_fn(O_tmem + 64, S_tmem + k * 8, desc_V1 + k * 128, idesc_O, 1);
                }
                umma_commit_cg1_fn(&mbar_umma[wg_id]);
            }
            mbarrier_wait_fn(&mbar_umma[wg_id], phase_umma[wg_id]);
            phase_umma[wg_id] ^= 1;
        }

        __syncthreads();

        phase[buf_idx] ^= 1;
        buf_idx = next_buf;
    }

    if (s_q + wg_id * 128 < S) {
        float inv_l = 1.0f / l_i;
        __nv_bfloat16* O_smem_out = Q_smem[wg_id][0];
        
        #pragma unroll 4
        for (int c = 0; c < 128; c += 32) {
            uint32_t rO[32];
            tmem_load_32x_fn(O_tmem + c, rO);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                float f = __uint_as_float(rO[i]) * inv_l;
                uint32_t row = lane_id;
                uint32_t col = c + i;
                uint32_t swizzled_col = ((col / 8) ^ (row % 8)) * 8 + (col % 8);
                O_smem_out[row * 128 + swizzled_col] = __float2bfloat16(f);
            }
        }
    }
    
    __syncthreads();

    if (s_q + wg_id * 128 < S) {
        __nv_bfloat16* O_smem_out = Q_smem[wg_id][0];
        uint32_t warp_id_in_wg = lane_id / 32;
        uint32_t lane_in_warp = lane_id % 32;
        uint32_t num_steps = 128 / 4;
        #pragma unroll 1
        for (uint32_t step = 0; step < num_steps; ++step) {
            uint32_t row = step * 4 + warp_id_in_wg;
            if (row >= 128) continue;
            uint32_t global_row = s_q + wg_id * 128 + row;
            uint32_t col = lane_in_warp * 4;
            uint32_t swizzled_col = ((col / 8) ^ (row % 8)) * 8 + (col % 8);
            
            if (global_row < S && col < D) {
                uint2 data = *reinterpret_cast<uint2*>(&O_smem_out[row * 128 + swizzled_col]);
                uint64_t offset = ((uint64_t)b * H * S + h * S + global_row) * D + col;
                *reinterpret_cast<uint2*>(o_ptr + offset) = data;
            }
        }

        int q_idx = s_q + wg_id * 128 + lane_id;
        if (q_idx < S) {
            uint64_t offset = (uint64_t)b * H * S + h * S + q_idx;
            float ln2 = 0.6931471805599453f;
            lse_ptr[offset] = m_i * ln2 + logf(l_i);
        }
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_alloc_addr, 512);
    }
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
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

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3); 

    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V;
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, q_ptr, D, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, k_ptr, D, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, v_ptr, D, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    dim3 grid((S + 255) / 256, H, B);
    dim3 block(256); 
    
    int smem_bytes = 196608 + 1024; 
    CUDA_CHECK(cudaFuncSetAttribute(fa4_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    fa4_kernel<<<grid, block, smem_bytes, stream>>>(
        tma_Q, tma_K, tma_V, o_ptr, lse_ptr, B, H, S, D
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, fa4_sm100::run);

}  // namespace fa4_sm100