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

#define CUDA_DRIVER_CHECK(call) do {                               \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CUDA Driver error %d at %s:%d\n",         \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_bwd {

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
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
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}

__device__ __forceinline__ uint64_t make_desc(__nv_bfloat16* ptr, bool is_major) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t lbo = is_major ? 8192 : 16;
    uint32_t sbo = 1024;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool transpose_A = false, bool transpose_B = false) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (transpose_A << 15);
    d |= (transpose_B << 16);
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void read_swizzled_bf16_pair(const __nv_bfloat16* smem, uint32_t row, uint32_t col, float& f0, float& f1) {
    uint32_t offset = (((row % 8) ^ (col / 8)) * 8) + (col % 8);
    uint32_t val = *(const uint32_t*)&smem[offset];
    uint16_t b0 = val & 0xFFFF;
    uint16_t b1 = (val >> 16) & 0xFFFF;
    f0 = __bfloat162float(*(__nv_bfloat16*)&b0);
    f1 = __bfloat162float(*(__nv_bfloat16*)&b1);
}

__device__ __forceinline__ void write_swizzled_u32(__nv_bfloat16* smem, uint32_t row, uint32_t col, float f0, float f1) {
    __nv_bfloat16 b0 = __float2bfloat16(f0);
    __nv_bfloat16 b1 = __float2bfloat16(f1);
    uint32_t val;
    asm("mov.b32 %0, {%1, %2};" : "=r"(val) : "h"(*(uint16_t*)&b0), "h"(*(uint16_t*)&b1));
    uint32_t offset = (((row % 8) ^ (col / 8)) * 8) + (col % 8);
    *(uint32_t*)&smem[offset] = val;
}

__device__ __forceinline__ void tmem_to_gmem_64x64(
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t m_block, uint32_t n_block,
    uint32_t tmem_base, uint32_t stride) {
    
    if (threadIdx.x < 64) {
        for (uint32_t col = 0; col < 32; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_base + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            uint32_t base = threadIdx.x * 64 + col;
            smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
            smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
            smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
            smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
        }
    }
    __syncthreads();
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (64 + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= 64) continue;
        uint32_t global_row = m_block + row;
        uint32_t col_start = lane_id * 4;
        if (col_start >= 64) continue; 
        uint32_t global_col = n_block + col_start;
        if (global_row < M) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * 64 + col_start]);
            *reinterpret_cast<uint2*>(D + (uint64_t)global_row * stride + global_col) = data;
        }
    }
}

__global__ void mha_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_q,
    const __grid_constant__ CUtensorMap tma_k,
    const __grid_constant__ CUtensorMap tma_v,
    const __grid_constant__ CUtensorMap tma_o,
    const __grid_constant__ CUtensorMap tma_do,
    __nv_bfloat16* dQ,
    __nv_bfloat16* dK,
    __nv_bfloat16* dV,
    const float* L,
    int32_t S,
    int32_t B,
    int32_t H,
    int32_t part)
{
    extern __shared__ __align__(128) uint8_t smem_raw[];
    uint32_t raw_ptr = (uint32_t)__cvta_generic_to_shared(smem_raw);
    uint32_t ptr = (raw_ptr + 8191) & ~8191;
    
    __nv_bfloat16* smem_q = (__nv_bfloat16*)ptr;
    __nv_bfloat16* smem_k = smem_q + 16384;
    __nv_bfloat16* smem_v = smem_k + 16384;
    __nv_bfloat16* smem_do = smem_v + 16384;
    __nv_bfloat16* smem_o = smem_do + 16384;
    __nv_bfloat16* smem_p = smem_o + 16384;
    __nv_bfloat16* smem_dp = smem_p + 8192;
    __nv_bfloat16* smem_out = smem_dp + 8192;
    float* smem_l = (float*)(smem_out + 8192);
    float* smem_rowsum_P_D = smem_l + 128;
    
    uint64_t* mbar = (uint64_t*)(smem_rowsum_P_D + 128);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar[0], 1);
        init_smem_barrier_fn(&mbar[1], 1);
        init_smem_barrier_fn(&mbar[2], 1);
        init_smem_barrier_fn(&mbar[3], 1);
        init_smem_barrier_fn(&mbar[4], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    uint32_t tmem_d, tmem_s, tmem_dq0, tmem_dq1, tmem_dk0, tmem_dk1, tmem_dv0, tmem_dv1, tmem_dp;
    if (threadIdx.x == 0) {
        if (part == 1) {
            tmem_alloc_fn(&tmem_s, 64);
            tmem_alloc_fn(&tmem_d, 64);
            tmem_alloc_fn(&tmem_dp, 64);
            tmem_alloc_fn(&tmem_dq0, 64);
            tmem_alloc_fn(&tmem_dq1, 64);
        } else {
            tmem_alloc_fn(&tmem_s, 64);
            tmem_alloc_fn(&tmem_d, 64);
            tmem_alloc_fn(&tmem_dp, 64);
            tmem_alloc_fn(&tmem_dk0, 64);
            tmem_alloc_fn(&tmem_dk1, 64);
            tmem_alloc_fn(&tmem_dv0, 64);
            tmem_alloc_fn(&tmem_dv1, 64);
        }
    }
    __syncthreads();

    uint32_t MyRank = cluster_rank_fn();
    uint32_t MyRow = threadIdx.x;
    
    if (part == 1) {
        // ========== Compute dQ ==========
        uint32_t bhs_base = blockIdx.x * 64;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar[0], 16384 * 6);
            tma_load_2d_fn(&tma_q, &mbar[0], smem_q, 0, bhs_base);
            tma_load_2d_fn(&tma_q, &mbar[0], smem_q + 4096, 64, bhs_base);
            
            tma_load_2d_fn(&tma_do, &mbar[0], smem_do, 0, bhs_base);
            tma_load_2d_fn(&tma_do, &mbar[0], smem_do + 4096, 64, bhs_base);
            
            tma_load_2d_fn(&tma_o, &mbar[0], smem_o, 0, bhs_base);
            tma_load_2d_fn(&tma_o, &mbar[0], smem_o + 4096, 64, bhs_base);
        }
        if (threadIdx.x < 64) {
            smem_l[threadIdx.x] = (bhs_base + threadIdx.x < B * H * S) ? L[bhs_base + threadIdx.x] : 0.0f;
        }
        mbarrier_wait_fn(&mbar[0], 0);
        
        if (MyRow < 64) {
            float sum = 0.0f;
            for (uint32_t col = 0; col < 64; col += 2) {
                float f_do0, f_do1, f_o0, f_o1;
                read_swizzled_bf16_pair(smem_do, MyRow, col, f_do0, f_do1);
                read_swizzled_bf16_pair(smem_o, MyRow, col, f_o0, f_o1);
                sum += f_do0 * f_o0 + f_do1 * f_o1;

                read_swizzled_bf16_pair(smem_do + 4096, MyRow, col, f_do0, f_do1);
                read_swizzled_bf16_pair(smem_o + 4096, MyRow, col, f_o0, f_o1);
                sum += f_do0 * f_o0 + f_do1 * f_o1;
            }
            smem_rowsum_P_D[MyRow] = sum;
        }
        __syncthreads();
        
        uint32_t phase_d[2] = {0};
        uint32_t idesc = make_instr_desc_fn(64, 64, false, false);
        
        uint32_t phase_v[2] = {0};
        if (S > 0) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar[2], 16384 * 2);
                tma_load_2d_fn(&tma_v, &mbar[2], smem_v, 0, 0);
                tma_load_2d_fn(&tma_v, &mbar[2], smem_v + 4096, 64, 0);
            }
        }

        uint32_t phase_k[2] = {0};
        if (S > 0) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar[3], 16384 * 2);
                tma_load_2d_fn(&tma_k, &mbar[3], smem_k, 0, 0);
                tma_load_2d_fn(&tma_k, &mbar[3], smem_k + 4096, 64, 0);
            }
        }

        uint32_t phase_dq[2] = {0};
        uint32_t idesc_dq = make_instr_desc_fn(64, 64, false, true);
        
        // Single-buffered pipeline passing for Q, Double-buffered for K and V
        for (uint32_t s_base = 0; s_base < S; s_base += 64) {
            uint32_t v_idx = (s_base / 64) % 2;
            uint32_t next_v_idx = (v_idx + 1) % 2;
            uint32_t next_s_base = s_base + 64;
            
            if (next_s_base < S) {
                if (threadIdx.x == 0) {
                    mbarrier_arrive_and_expect_tx_fn(&mbar[2], 16384 * 2);
                    tma_load_2d_fn(&tma_v, &mbar[2], smem_v + (next_v_idx * 4096), 0, next_s_base);
                    tma_load_2d_fn(&tma_v, &mbar[2], smem_v + (next_v_idx * 4096) + 4096, 64, next_s_base);
                }
            }
            mbarrier_wait_fn(&mbar[2], phase_v[v_idx]);
            phase_v[v_idx] ^= 1;
            
            uint32_t k_idx = (s_base / 64) % 2;
            uint32_t next_k_idx = (k_idx + 1) % 2;
            uint32_t next_k_base = s_base + 64;
            
            if (next_k_base < S) {
                if (threadIdx.x == 0) {
                    mbarrier_arrive_and_expect_tx_fn(&mbar[3], 16384 * 2);
                    tma_load_2d_fn(&tma_k, &mbar[3], smem_k + (next_k_idx * 4096), 0, next_k_base);
                    tma_load_2d_fn(&tma_k, &mbar[3], smem_k + (next_k_idx * 4096) + 4096, 64, next_k_base);
                }
            }
            mbarrier_wait_fn(&mbar[3], phase_k[k_idx]);
            phase_k[k_idx] ^= 1;
            
            for (uint32_t k = 0; k < 64; k += 16) {
                uint64_t desc_a = make_desc(smem_q + k, false);
                uint64_t desc_b = make_desc(smem_k + k_idx*4096 + k*64, true);
                umma_f16_cg2_fn(tmem_s, desc_a, desc_b, idesc, (k == 0) ? 0 : 1);
                
                uint64_t desc_a1 = make_desc(smem_q + 4096 + k, false);
                uint64_t desc_b1 = make_desc(smem_k + k_idx*4096 + 4096 + k*64, true);
                umma_f16_cg2_fn(tmem_s, desc_a1, desc_b1, idesc, 1);
            }
            umma_commit_2sm_fn(&mbar[1]);
            mbarrier_wait_fn(&mbar[1], 0);
            
            for (uint32_t i = 0; i < 64; i += 4) {
                uint32_t r_s0, r_s1, r_s2, r_s3;
                tmem_load_4x_fn(tmem_s + threadIdx.x * 64 + i, &r_s0, &r_s1, &r_s2, &r_s3);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                float f_s0 = __uint_as_float(r_s0);
                float f_s1 = __uint_as_float(r_s1);
                float f_s2 = __uint_as_float(r_s2);
                float f_s3 = __uint_as_float(r_s3);
                
                float p0 = fast_exp2f_fn((f_s0 - smem_l[MyRow]) * 1.44269504f);
                float p1 = fast_exp2f_fn((f_s1 - smem_l[MyRow]) * 1.44269504f);
                float p2 = fast_exp2f_fn((f_s2 - smem_l[MyRow]) * 1.44269504f);
                float p3 = fast_exp2f_fn((f_s3 - smem_l[MyRow]) * 1.44269504f);
                
                write_swizzled_u32(smem_p, MyRow, i, p0, p1);
                write_swizzled_u32(smem_p, MyRow, i + 2, p2, p3);
            }
            __syncthreads();
            fence_async_shared_fn();
            
            for (uint32_t k = 0; k < 64; k += 16) {
                uint64_t desc_a = make_desc(smem_do + k, false);
                uint64_t desc_b = make_desc(smem_v + v_idx*4096 + k*64, true);
                umma_f16_cg2_fn(tmem_d, desc_a, desc_b, idesc, (phase_d[MyRank] == 0) ? 0 : 1);
                
                uint64_t desc_a1 = make_desc(smem_do + 4096 + k, false);
                uint64_t desc_b1 = make_desc(smem_v + v_idx*4096 + 4096 + k*64, true);
                umma_f16_cg2_fn(tmem_d, desc_a1, desc_b1, idesc, 1);
                
                phase_d[MyRank]++;
            }
            umma_commit_2sm_fn(&mbar[1]);
            mbarrier_wait_fn(&mbar[1], 0);
            
            for (uint32_t i = 0; i < 64; i += 4) {
                uint32_t r_d0, r_d1, r_d2, r_d3;
                tmem_load_4x_fn(tmem_d + threadIdx.x * 64 + i, &r_d0, &r_d1, &r_d2, &r_d3);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                float f_d0 = __uint_as_float(r_d0);
                float f_d1 = __uint_as_float(r_d1);
                float f_d2 = __uint_as_float(r_d2);
                float f_d3 = __uint_as_float(r_d3);
                
                float p0, p1, p2, p3;
                read_swizzled_bf16_pair(smem_p, MyRow, i, p0, p1);
                read_swizzled_bf16_pair(smem_p, MyRow, i + 2, p2, p3);
                
                float dp0 = p0 * (f_d0 - smem_rowsum_P_D[MyRow]);
                float dp1 = p1 * (f_d1 - smem_rowsum_P_D[MyRow]);
                float dp2 = p2 * (f_d2 - smem_rowsum_P_D[MyRow]);
                float dp3 = p3 * (f_d3 - smem_rowsum_P_D[MyRow]);
                
                write_swizzled_u32(smem_dp, MyRow, i, dp0, dp1);
                write_swizzled_u32(smem_dp, MyRow, i + 2, dp2, dp3);
            }
            __syncthreads();
            fence_async_shared_fn();
            
            for (uint32_t k = 0; k < 64; k += 16) {
                uint64_t desc_a = make_desc(smem_dp + k, false);
                uint64_t desc_b0 = make_desc(smem_k + k_idx*4096 + k*64, true);
                umma_f16_cg2_fn(tmem_dq0, desc_a, desc_b0, idesc_dq, (phase_dq[MyRank] == 0) ? 0 : 1);
                
                uint64_t desc_b1 = make_desc(smem_k + k_idx*4096 + 4096 + k*64, true);
                umma_f16_cg2_fn(tmem_dq1, desc_a, desc_b1, idesc_dq, (phase_dq[MyRank] == 0) ? 0 : 1);
                
                phase_dq[MyRank]++;
            }
            umma_commit_2sm_fn(&mbar[1]);
            mbarrier_wait_fn(&mbar[1], 0);
        }
        
        tmem_to_gmem_64x64(dQ, smem_out, B * H * S, bhs_base, 0, tmem_dq0, 128);
        tmem_to_gmem_64x64(dQ, smem_out, B * H * S, bhs_base, 64, tmem_dq1, 128);

    } else {
        // ========== Compute dK and dV ==========
        uint32_t s_base = blockIdx.x * 64;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar[3], 16384 * 2);
            tma_load_2d_fn(&tma_k, &mbar[3], smem_k, 0, s_base);
            tma_load_2d_fn(&tma_k, &mbar[3], smem_k + 4096, 64, s_base);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar[2], 16384 * 2);
            tma_load_2d_fn(&tma_v, &mbar[2], smem_v, 0, s_base);
            tma_load_2d_fn(&tma_v, &mbar[2], smem_v + 4096, 64, s_base);
        }
        mbarrier_wait_fn(&mbar[3], 0);
        mbarrier_wait_fn(&mbar[2], 0);

        uint32_t phase_dk[2] = {0}, phase_dv[2] = {0};
        uint32_t idesc_dk = make_instr_desc_fn(64, 64, true, true); 
        uint32_t idesc_dv = make_instr_desc_fn(64, 64, true, true); 
        
        uint32_t phase_q[2] = {0};
        uint32_t total_bh_s = B * H * S;

        if (total_bh_s > 0) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar[1], 16384 * 6);
                tma_load_2d_fn(&tma_q, &mbar[1], smem_q, 0, 0);
                tma_load_2d_fn(&tma_q, &mbar[1], smem_q + 4096, 64, 0);
                tma_load_2d_fn(&tma_do, &mbar[1], smem_do, 0, 0);
                tma_load_2d_fn(&tma_do, &mbar[1], smem_do + 4096, 64, 0);
                tma_load_2d_fn(&tma_o, &mbar[1], smem_o, 0, 0);
                tma_load_2d_fn(&tma_o, &mbar[1], smem_o + 4096, 64, 0);
            }
        }

        for (uint32_t bhs_base = 0; bhs_base < total_bh_s; bhs_base += 64) {
            uint32_t q_idx = (bhs_base / 64) % 2;
            uint32_t next_q_idx = (q_idx + 1) % 2;
            uint32_t next_bhs_base = bhs_base + 64;
            
            if (next_bhs_base < total_bh_s) {
                if (threadIdx.x == 0) {
                    mbarrier_arrive_and_expect_tx_fn(&mbar[1], 16384 * 6);
                    tma_load_2d_fn(&tma_q, &mbar[1], smem_q + (next_q_idx * 4096), 0, next_bhs_base);
                    tma_load_2d_fn(&tma_q, &mbar[1], smem_q + (next_q_idx * 4096) + 4096, 64, next_bhs_base);
                    tma_load_2d_fn(&tma_do, &mbar[1], smem_do + (next_q_idx * 4096), 0, next_bhs_base);
                    tma_load_2d_fn(&tma_do, &mbar[1], smem_do + (next_q_idx * 4096) + 4096, 64, next_bhs_base);
                    tma_load_2d_fn(&tma_o, &mbar[1], smem_o + (next_q_idx * 4096), 0, next_bhs_base);
                    tma_load_2d_fn(&tma_o, &mbar[1], smem_o + (next_q_idx * 4096) + 4096, 64, next_bhs_base);
                }
            }
            mbarrier_wait_fn(&mbar[1], phase_q[q_idx]);
            phase_q[q_idx] ^= 1;
            
            if (threadIdx.x < 64) {
                smem_l[threadIdx.x] = (bhs_base + threadIdx.x < total_bh_s) ? L[bhs_base + threadIdx.x] : 0.0f;
            }
            
            if (MyRow < 64) {
                float sum = 0.0f;
                for (uint32_t col = 0; col < 64; col += 2) {
                    float f_do0, f_do1, f_o0, f_o1;
                    read_swizzled_bf16_pair(smem_do + (q_idx * 4096), MyRow, col, f_do0, f_do1);
                    read_swizzled_bf16_pair(smem_o + (q_idx * 4096), MyRow, col, f_o0, f_o1);
                    sum += f_do0 * f_o0 + f_do1 * f_o1;

                    read_swizzled_bf16_pair(smem_do + (q_idx * 4096) + 4096, MyRow, col, f_do0, f_do1);
                    read_swizzled_bf16_pair(smem_o + (q_idx * 4096) + 4096, MyRow, col, f_o0, f_o1);
                    sum += f_do0 * f_o0 + f_do1 * f_o1;
                }
                smem_rowsum_P_D[MyRow] = sum;
            }
            __syncthreads();
            fence_async_shared_fn();
            
            uint32_t phase_d_local[2] = {0};
            uint32_t idesc = make_instr_desc_fn(64, 64, false, false);
            
            for (uint32_t k = 0; k < 64; k += 16) {
                uint64_t desc_a = make_desc(smem_q + q_idx*4096 + k, false);
                uint64_t desc_b = make_desc(smem_k + k*64, true);
                umma_f16_cg2_fn(tmem_s, desc_a, desc_b, idesc, (k == 0) ? 0 : 1);
                
                uint64_t desc_a1 = make_desc(smem_q + q_idx*4096 + 4096 + k, false);
                uint64_t desc_b1 = make_desc(smem_k + 4096 + k*64, true);
                umma_f16_cg2_fn(tmem_s, desc_a1, desc_b1, idesc, 1);
            }
            umma_commit_2sm_fn(&mbar[1]);
            mbarrier_wait_fn(&mbar[1], 0);
            
            for (uint32_t i = 0; i < 64; i += 4) {
                uint32_t r_s0, r_s1, r_s2, r_s3;
                tmem_load_4x_fn(tmem_s + threadIdx.x * 64 + i, &r_s0, &r_s1, &r_s2, &r_s3);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                float f_s0 = __uint_as_float(r_s0);
                float f_s1 = __uint_as_float(r_s1);
                float f_s2 = __uint_as_float(r_s2);
                float f_s3 = __uint_as_float(r_s3);
                
                float p0 = fast_exp2f_fn((f_s0 - smem_l[MyRow]) * 1.44269504f);
                float p1 = fast_exp2f_fn((f_s1 - smem_l[MyRow]) * 1.44269504f);
                float p2 = fast_exp2f_fn((f_s2 - smem_l[MyRow]) * 1.44269504f);
                float p3 = fast_exp2f_fn((f_s3 - smem_l[MyRow]) * 1.44269504f);
                
                write_swizzled_u32(smem_p, MyRow, i, p0, p1);
                write_swizzled_u32(smem_p, MyRow, i + 2, p2, p3);
            }
            __syncthreads();
            fence_async_shared_fn();
            
            for (uint32_t k = 0; k < 64; k += 16) {
                uint64_t desc_a = make_desc(smem_do + q_idx*4096 + k, false);
                uint64_t desc_b = make_desc(smem_v + k*64, true);
                umma_f16_cg2_fn(tmem_d, desc_a, desc_b, idesc, (phase_d_local[MyRank] == 0) ? 0 : 1);
                
                uint64_t desc_a1 = make_desc(smem_do + q_idx*4096 + 4096 + k, false);
                uint64_t desc_b1 = make_desc(smem_v + 4096 + k*64, true);
                umma_f16_cg2_fn(tmem_d, desc_a1, desc_b1, idesc, 1);
                
                phase_d_local[MyRank]++;
            }
            umma_commit_2sm_fn(&mbar[1]);
            mbarrier_wait_fn(&mbar[1], 0);
            
            for (uint32_t i = 0; i < 64; i += 4) {
                uint32_t r_d0, r_d1, r_d2, r_d3;
                tmem_load_4x_fn(tmem_d + threadIdx.x * 64 + i, &r_d0, &r_d1, &r_d2, &r_d3);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                float f_d0 = __uint_as_float(r_d0);
                float f_d1 = __uint_as_float(r_d1);
                float f_d2 = __uint_as_float(r_d2);
                float f_d3 = __uint_as_float(r_d3);
                
                float p0, p1, p2, p3;
                read_swizzled_bf16_pair(smem_p, MyRow, i, p0, p1);
                read_swizzled_bf16_pair(smem_p, MyRow, i + 2, p2, p3);
                
                float dp0 = p0 * (f_d0 - smem_rowsum_P_D[MyRow]);
                float dp1 = p1 * (f_d1 - smem_rowsum_P_D[MyRow]);
                float dp2 = p2 * (f_d2 - smem_rowsum_P_D[MyRow]);
                float dp3 = p3 * (f_d3 - smem_rowsum_P_D[MyRow]);
                
                write_swizzled_u32(smem_dp, MyRow, i, dp0, dp1);
                write_swizzled_u32(smem_dp, MyRow, i + 2, dp2, dp3);
            }
            __syncthreads();
            fence_async_shared_fn();
            
            for (uint32_t k = 0; k < 64; k += 16) {
                uint64_t desc_dp = make_desc(smem_dp + k*64, true);
                uint64_t desc_q = make_desc(smem_q + q_idx*4096 + k, true);
                umma_f16_cg2_fn(tmem_dk0, desc_dp, desc_q, idesc_dk, (phase_dk[MyRank] == 0) ? 0 : 1);
                
                uint64_t desc_q1 = make_desc(smem_q + q_idx*4096 + 4096 + k, true);
                umma_f16_cg2_fn(tmem_dk1, desc_dp, desc_q1, idesc_dk, (phase_dk[MyRank] == 0) ? 0 : 1);
                
                uint64_t desc_p = make_desc(smem_p + k*64, true);
                uint64_t desc_do = make_desc(smem_do + q_idx*4096 + k, true);
                umma_f16_cg2_fn(tmem_dv0, desc_p, desc_do, idesc_dv, (phase_dv[MyRank] == 0) ? 0 : 1);
                
                uint64_t desc_do1 = make_desc(smem_do + q_idx*4096 + 4096 + k, true);
                umma_f16_cg2_fn(tmem_dv1, desc_p, desc_do1, idesc_dv, (phase_dv[MyRank] == 0) ? 0 : 1);
                
                phase_dk[MyRank]++;
                phase_dv[MyRank]++;
            }
            umma_commit_2sm_fn(&mbar[1]);
            mbarrier_wait_fn(&mbar[1], 0);
        }
        
        tmem_to_gmem_64x64(dK, smem_out, S, s_base, 0, tmem_dk0, 128);
        tmem_to_gmem_64x64(dK, smem_out, S, s_base, 64, tmem_dk1, 128);
        tmem_to_gmem_64x64(dV, smem_out, S, s_base, 0, tmem_dv0, 128);
        tmem_to_gmem_64x64(dV, smem_out, S, s_base, 64, tmem_dv1, 128);
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        if (part == 1) {
            tmem_dealloc_fn(tmem_s, 64);
            tmem_dealloc_fn(tmem_d, 64);
            tmem_dealloc_fn(tmem_dp, 64);
            tmem_dealloc_fn(tmem_dq0, 64);
            tmem_dealloc_fn(tmem_dq1, 64);
        } else {
            tmem_dealloc_fn(tmem_s, 64);
            tmem_dealloc_fn(tmem_d, 64);
            tmem_dealloc_fn(tmem_dp, 64);
            tmem_dealloc_fn(tmem_dk0, 64);
            tmem_dealloc_fn(tmem_dk1, 64);
            tmem_dealloc_fn(tmem_dv0, 64);
            tmem_dealloc_fn(tmem_dv1, 64);
        }
    }
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, CUtensorMapDataType dataType, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 2, globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3); 
    
    CUtensorMap tma_q, tma_k, tma_v, tma_o, tma_do;
    CUtensorMapDataType dataType = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    
    create_tma_2d_descriptor_2B(&tma_q, dataType, Q.data_ptr(), d, B*H*S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_k, dataType, K.data_ptr(), d, B*H*S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_v, dataType, V.data_ptr(), d, B*H*S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_o, dataType, O.data_ptr(), d, B*H*S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_do, dataType, dO.data_ptr(), d, B*H*S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    cudaLaunchConfig_t config = {};
    config.blockDim.x = 128;
    config.dynamicSmemBytes = 220 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    uint32_t gx1 = (B * H * S + 63) / 64;
    config.gridDim = dim3(gx1, 1);
    cudaLaunchKernelEx(&config, mha_bwd_kernel, tma_q, tma_k, tma_v, tma_o, tma_do, 
                        static_cast<__nv_bfloat16*>(dQ.data_ptr()), nullptr, nullptr, 
                        static_cast<const float*>(L.data_ptr()), S, B, H, 1);
    CUDA_CHECK(cudaGetLastError());
    
    uint32_t gy2 = (S + 63) / 64;
    config.gridDim = dim3(gy2, 1);
    cudaLaunchKernelEx(&config, mha_bwd_kernel, tma_q, tma_k, tma_v, tma_o, tma_do, 
                        nullptr, static_cast<__nv_bfloat16*>(dK.data_ptr()), static_cast<__nv_bfloat16*>(dV.data_ptr()), 
                        static_cast<const float*>(L.data_ptr()), S, B, H, 2);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(config.stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha_bwd