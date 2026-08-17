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

namespace tvm_ffi_mha_bwd {

// -------------------------------------------------------------------------
// Helper functions (Barriers, TMEM, MMA)
// -------------------------------------------------------------------------

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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    
    uint64_t base_offset = (addr >> 7) & 0x7;
    d |= (base_offset << 49);
    
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void mma_64x64_f16(
    uint32_t tmem_dest, void* smem_a, void* smem_b,
    bool a_mn_major, bool b_mn_major,
    uint32_t accum) 
{
    if (threadIdx.x != 0) return;

    uint32_t idesc = make_instr_desc_fn(64, 64);
    if (a_mn_major) idesc |= (1u << 15);
    if (b_mn_major) idesc |= (1u << 16);

    uint32_t a_sbo = 1024, a_lbo = a_mn_major ? 8192 : 1;
    uint32_t b_sbo = 1024, b_lbo = b_mn_major ? 8192 : 1;

    for (int k = 0; k < 4; ++k) {
        uint32_t offset_a = a_mn_major ? (k * 16 * 128) : (k * 16 * 2);
        uint32_t offset_b = b_mn_major ? (k * 16 * 128) : (k * 16 * 2);

        uint64_t desc_a = make_smem_desc_sm100_fn((char*)smem_a + offset_a, a_lbo, a_sbo);
        uint64_t desc_b = make_smem_desc_sm100_fn((char*)smem_b + offset_b, b_lbo, b_sbo);
        
        uint32_t acc = (k == 0) ? accum : 1;
        
        asm volatile(
            "{\n.reg .pred p;\n"
            "setp.ne.b32 p, %4, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
            :: "r"(tmem_dest), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(acc));
    }
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

// -------------------------------------------------------------------------
// Pipeline math functions
// -------------------------------------------------------------------------

__device__ __forceinline__ void compute_P_write_smem(
    uint32_t tmem_S, float* LSE, __nv_bfloat16* smem_PT0, __nv_bfloat16* smem_PT1) {
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t row = warp_id * 32 + lane_id; 
    
    float scale = 1.0f / sqrtf(128.0f);

    for (int col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t tmem_col = tmem_S + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        float lse0 = LSE[col + 0];
        float lse1 = LSE[col + 1];
        float lse2 = LSE[col + 2];
        float lse3 = LSE[col + 3];
        
        f0 = fast_exp2f_fn((f0 * scale - lse0) * 1.44269504f);
        f1 = fast_exp2f_fn((f1 * scale - lse1) * 1.44269504f);
        f2 = fast_exp2f_fn((f2 * scale - lse2) * 1.44269504f);
        f3 = fast_exp2f_fn((f3 * scale - lse3) * 1.44269504f);
        
        __nv_bfloat16* smem_PT = (col < 64) ? smem_PT0 : smem_PT1;
        uint32_t smem_col = col % 64;
        
        uint32_t swizzled_col = smem_col ^ ((row % 8) * 8);
        uint32_t smem_idx = row * 64 + swizzled_col; 
        
        uint32_t p01 = pack_bf16_fn(__float_as_uint(f0), __float_as_uint(f1));
        uint32_t p23 = pack_bf16_fn(__float_as_uint(f2), __float_as_uint(f3));
        *(uint2*)(&smem_PT[smem_idx]) = make_uint2(p01, p23);
    }
}

__device__ __forceinline__ void compute_dS_write_smem(
    uint32_t tmem_dP, __nv_bfloat16* smem_PT0, __nv_bfloat16* smem_PT1, float* D, __nv_bfloat16* smem_dST0, __nv_bfloat16* smem_dST1) {
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t row = warp_id * 32 + lane_id;

    for (int col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t tmem_col = tmem_dP + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float dp0 = __uint_as_float(r0);
        float dp1 = __uint_as_float(r1);
        float dp2 = __uint_as_float(r2);
        float dp3 = __uint_as_float(r3);
        
        __nv_bfloat16* smem_PT = (col < 64) ? smem_PT0 : smem_PT1;
        uint32_t smem_col = col % 64;
        uint32_t swizzled_col = smem_col ^ ((row % 8) * 8);
        uint32_t smem_idx = row * 64 + swizzled_col;
        
        uint2 pt_data = *(uint2*)(&smem_PT[smem_idx]);
        __nv_bfloat162 pt01 = *reinterpret_cast<__nv_bfloat162*>(&pt_data.x);
        __nv_bfloat162 pt23 = *reinterpret_cast<__nv_bfloat162*>(&pt_data.y);
#if __CUDA_ARCH__ >= 800
        float2 p01_f2 = __bfloat1622float2(pt01);
        float2 p23_f2 = __bfloat1622float2(pt23);
#else
        float2 p01_f2 = make_float2(__bfloat162float(pt01.x), __bfloat162float(pt01.y));
        float2 p23_f2 = make_float2(__bfloat162float(pt23.x), __bfloat162float(pt23.y));
#endif
        
        float d0 = D[col + 0];
        float d1 = D[col + 1];
        float d2 = D[col + 2];
        float d3 = D[col + 3];
        
        float ds0 = p01_f2.x * (dp0 - d0);
        float ds1 = p01_f2.y * (dp1 - d1);
        float ds2 = p23_f2.x * (dp2 - d2);
        float ds3 = p23_f2.y * (dp3 - d3);
        
        __nv_bfloat16* smem_dST = (col < 64) ? smem_dST0 : smem_dST1;
        uint32_t ds01 = pack_bf16_fn(__float_as_uint(ds0), __float_as_uint(ds1));
        uint32_t ds23 = pack_bf16_fn(__float_as_uint(ds2), __float_as_uint(ds3));
        *(uint2*)(&smem_dST[smem_idx]) = make_uint2(ds01, ds23);
    }
}

__device__ __forceinline__ void atomicAdd_dQ_fp32(
    uint32_t tmem_dQ, float* dQ_global, uint32_t M, uint32_t d, uint32_t m_base,
    uint32_t b, uint32_t h, float scale) {
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t row = warp_id * 32 + lane_id; 
    
    if (m_base + row >= M) return;
    
    for (int col = 0; col < 128; col += 4) { 
        uint32_t r0, r1, r2, r3;
        uint32_t tmem_col = tmem_dQ + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0) * scale;
        float f1 = __uint_as_float(r1) * scale;
        float f2 = __uint_as_float(r2) * scale;
        float f3 = __uint_as_float(r3) * scale;
        
        uint64_t idx = b * (gridDim.y * M * d) + h * (M * d) + (m_base + row) * d + col;
        
        atomicAdd(&dQ_global[idx + 0], f0);
        atomicAdd(&dQ_global[idx + 1], f1);
        atomicAdd(&dQ_global[idx + 2], f2);
        atomicAdd(&dQ_global[idx + 3], f3);
    }
}

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_base_fn(
    uint32_t tmem_base,
    __nv_bfloat16* D, __nv_bfloat16* smem_out0, __nv_bfloat16* smem_out1,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN, uint32_t b, uint32_t h, float scale) {
    
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t t_col = tmem_base + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(t_col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0) * scale;
        float f1 = __uint_as_float(r1) * scale;
        float f2 = __uint_as_float(r2) * scale;
        float f3 = __uint_as_float(r3) * scale;
        
        __nv_bfloat16* smem_out = (col < 64) ? smem_out0 : smem_out1;
        uint32_t smem_col = col % 64;
        uint32_t swizzled_col = smem_col ^ ((threadIdx.x % 8) * 8);
        uint32_t base = threadIdx.x * 64 + swizzled_col;
        
        uint32_t p01 = pack_bf16_fn(__float_as_uint(f0), __float_as_uint(f1));
        uint32_t p23 = pack_bf16_fn(__float_as_uint(f2), __float_as_uint(f3));
        *(uint2*)(&smem_out[base]) = make_uint2(p01, p23);
    }
    __syncthreads();
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = 128 / 4; 
    
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        uint32_t global_row = m_block * 128 + row;
        
        uint32_t col = lane_id * 4;
        __nv_bfloat16* smem_out = (col < 64) ? smem_out0 : smem_out1;
        uint32_t smem_col = col % 64;
        uint32_t swizzled_col = smem_col ^ ((row % 8) * 8);
        
        if (global_row < M) {
            uint64_t global_idx = b * (gridDim.y * M * 128) + h * (M * 128) + global_row * 128 + col;
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * 64 + swizzled_col]);
            *reinterpret_cast<uint2*>(D + global_idx) = data;
        }
    }
    __syncthreads(); 
}

// -------------------------------------------------------------------------
// Main Kernels
// -------------------------------------------------------------------------

__global__ void compute_D_kernel(const __nv_bfloat16* O, const __nv_bfloat16* dO, float* D, int S, int d) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < S) {
        int h = blockIdx.y;
        int b = blockIdx.z;
        int H = gridDim.y;
        uint64_t idx = b * (H * S) + h * S + s;
        uint64_t offset = idx * d;
        float sum = 0;
        for (int i = 0; i < d; ++i) {
            float o = __bfloat162float(O[offset + i]);
            float do_val = __bfloat162float(dO[offset + i]);
            sum += o * do_val;
        }
        D[idx] = sum;
    }
}

__global__ void convert_fp32_to_bf16(const float* src, __nv_bfloat16* dst, uint64_t n) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

extern __shared__ __align__(1024) char smem_buf[];

struct SharedStorage {
    __align__(1024) __nv_bfloat16 K0[128 * 64];
    __align__(1024) __nv_bfloat16 K1[128 * 64];
    __align__(1024) __nv_bfloat16 V0[128 * 64];
    __align__(1024) __nv_bfloat16 V1[128 * 64];
    __align__(1024) __nv_bfloat16 Q0[128 * 64];
    __align__(1024) __nv_bfloat16 Q1[128 * 64];
    __align__(1024) __nv_bfloat16 dO0[128 * 64];
    __align__(1024) __nv_bfloat16 dO1[128 * 64];
    __align__(1024) __nv_bfloat16 PT0[128 * 64];
    __align__(1024) __nv_bfloat16 PT1[128 * 64];
    __align__(1024) __nv_bfloat16 dST0[128 * 64];
    __align__(1024) __nv_bfloat16 dST1[128 * 64];
    
    __align__(8) uint64_t bar_Q_dO;
    __align__(8) uint64_t bar_K_V;
    __align__(8) uint64_t mbar_commit;
    
    __align__(4) float LSE[128];
    __align__(4) float D[128];
};

__global__ void mha_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q, 
    const __grid_constant__ CUtensorMap tma_K, 
    const __grid_constant__ CUtensorMap tma_V, 
    const __grid_constant__ CUtensorMap tma_dO,
    const float* LSE_global, const float* D_global,
    float* dQ_global_fp32, __nv_bfloat16* dK_global, __nv_bfloat16* dV_global,
    int S, int d) 
{
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_buf);

    int kv_idx = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int H = gridDim.y;
    int N_blocks = S / 128;

    init_smem_barrier_fn(&smem.bar_Q_dO, 1);
    init_smem_barrier_fn(&smem.bar_K_V, 1);
    init_smem_barrier_fn(&smem.mbar_commit, 1);
    fence_smem_barrier_init_fn();
    uint32_t phase_K = 0, phase_Q = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&smem.bar_K_V, 65536);
        tma_load_5d_fn(&tma_K, &smem.bar_K_V, smem.K0, 0, 0, kv_idx * 128, h, b);
        tma_load_5d_fn(&tma_K, &smem.bar_K_V, smem.K1, 0, 1, kv_idx * 128, h, b);
        tma_load_5d_fn(&tma_V, &smem.bar_K_V, smem.V0, 0, 0, kv_idx * 128, h, b);
        tma_load_5d_fn(&tma_V, &smem.bar_K_V, smem.V1, 0, 1, kv_idx * 128, h, b);
    }
    mbarrier_wait_fn(&smem.bar_K_V, phase_K);
    phase_K ^= 1;

    __shared__ uint32_t smem_tmem_ptr;
    if (threadIdx.x / 32 == 0) {
        tmem_alloc_cg1_fn(&smem_tmem_ptr, 512);
    }
    __syncthreads();
    uint32_t tmem_addr = smem_tmem_ptr; 
    
    uint32_t tmem_S = tmem_addr;
    uint32_t tmem_dV = tmem_addr + 128;
    uint32_t tmem_dP_dQ = tmem_addr + 256;
    uint32_t tmem_dK = tmem_addr + 384;

    uint32_t phase_commit = 0;

    for (int q_idx = 0; q_idx < N_blocks; ++q_idx) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem.bar_Q_dO, 65536);
            tma_load_5d_fn(&tma_Q, &smem.bar_Q_dO, smem.Q0, 0, 0, q_idx * 128, h, b);
            tma_load_5d_fn(&tma_Q, &smem.bar_Q_dO, smem.Q1, 0, 1, q_idx * 128, h, b);
            tma_load_5d_fn(&tma_dO, &smem.bar_Q_dO, smem.dO0, 0, 0, q_idx * 128, h, b);
            tma_load_5d_fn(&tma_dO, &smem.bar_Q_dO, smem.dO1, 0, 1, q_idx * 128, h, b);
        }
        
        uint64_t lse_base = b * (H * S) + h * S + q_idx * 128;
        if (threadIdx.x < 128) {
            smem.LSE[threadIdx.x] = LSE_global[lse_base + threadIdx.x];
            smem.D[threadIdx.x] = D_global[lse_base + threadIdx.x];
        }
        
        mbarrier_wait_fn(&smem.bar_Q_dO, phase_Q);
        __syncthreads(); 
        
        // S_j^T = K @ Q_j^T
        for (int m_idx = 0; m_idx < 2; ++m_idx) {
            for (int n_idx = 0; n_idx < 2; ++n_idx) {
                void* a0 = (char*)smem.K0 + m_idx * 64 * 128;
                void* b0 = (char*)smem.Q0 + n_idx * 64 * 128;
                mma_64x64_f16(tmem_S + (m_idx * 64 << 16) + n_idx * 64, a0, b0, false, false, 0);

                void* a1 = (char*)smem.K1 + m_idx * 64 * 128;
                void* b1 = (char*)smem.Q1 + n_idx * 64 * 128;
                mma_64x64_f16(tmem_S + (m_idx * 64 << 16) + n_idx * 64, a1, b1, false, false, 1);
            }
        }
                        
        // dP_j^T = V @ dO_j^T
        for (int m_idx = 0; m_idx < 2; ++m_idx) {
            for (int n_idx = 0; n_idx < 2; ++n_idx) {
                void* a0 = (char*)smem.V0 + m_idx * 64 * 128;
                void* b0 = (char*)smem.dO0 + n_idx * 64 * 128;
                mma_64x64_f16(tmem_dP_dQ + (m_idx * 64 << 16) + n_idx * 64, a0, b0, false, false, 0);

                void* a1 = (char*)smem.V1 + m_idx * 64 * 128;
                void* b1 = (char*)smem.dO1 + n_idx * 64 * 128;
                mma_64x64_f16(tmem_dP_dQ + (m_idx * 64 << 16) + n_idx * 64, a1, b1, false, false, 1);
            }
        }
                        
        if (threadIdx.x == 0) {
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                         :: "r"((uint32_t)__cvta_generic_to_shared(&smem.mbar_commit)));
        }
        mbarrier_wait_fn(&smem.mbar_commit, phase_commit);
        phase_commit ^= 1;
        
        compute_P_write_smem(tmem_S, smem.LSE, smem.PT0, smem.PT1);
        __syncthreads(); 
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        
        // dV_j += P_j^T @ dO_j
        uint32_t acc_dV = (q_idx == 0) ? 0 : 1;
        for (int m_idx = 0; m_idx < 2; ++m_idx) {
            for (int n_idx = 0; n_idx < 2; ++n_idx) {
                void* dO_0 = (n_idx == 0) ? smem.dO0 : smem.dO1;
                void* a0 = (char*)smem.PT0 + m_idx * 64 * 128;
                void* b0_top = dO_0;
                mma_64x64_f16(tmem_dV + (m_idx * 64 << 16) + n_idx * 64, a0, b0_top, false, true, acc_dV);

                void* a1 = (char*)smem.PT1 + m_idx * 64 * 128;
                void* b0_bot = (char*)dO_0 + 64 * 128;
                mma_64x64_f16(tmem_dV + (m_idx * 64 << 16) + n_idx * 64, a1, b0_bot, false, true, 1);
            }
        }
                        
        compute_dS_write_smem(tmem_dP_dQ, smem.PT0, smem.PT1, smem.D, smem.dST0, smem.dST1);
        __syncthreads(); 
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        
        // dK_{j-1} += dS_{j-1}^T @ Q_{j-1}
        uint32_t acc_dK = (q_idx == 0) ? 0 : 1;
        for (int m_idx = 0; m_idx < 2; ++m_idx) {
            for (int n_idx = 0; n_idx < 2; ++n_idx) {
                void* Q_0 = (n_idx == 0) ? smem.Q0 : smem.Q1;
                void* a0 = (char*)smem.dST0 + m_idx * 64 * 128;
                void* b0_top = Q_0;
                mma_64x64_f16(tmem_dK + (m_idx * 64 << 16) + n_idx * 64, a0, b0_top, false, true, acc_dK);

                void* a1 = (char*)smem.dST1 + m_idx * 64 * 128;
                void* b0_bot = (char*)Q_0 + 64 * 128;
                mma_64x64_f16(tmem_dK + (m_idx * 64 << 16) + n_idx * 64, a1, b0_bot, false, true, 1);
            }
        }
                        
        // dQ_{j-1} = dS_{j-1} @ K
        for (int m_idx = 0; m_idx < 2; ++m_idx) {
            for (int n_idx = 0; n_idx < 2; ++n_idx) {
                void* K_0 = (n_idx == 0) ? smem.K0 : smem.K1;
                void* a0_top = (m_idx == 0) ? smem.dST0 : smem.dST1; 
                void* b0_top = K_0;
                mma_64x64_f16(tmem_dP_dQ + (m_idx * 64 << 16) + n_idx * 64, a0_top, b0_top, true, true, 0);

                void* a0_bot = (char*)a0_top + 64 * 128;
                void* b0_bot = (char*)K_0 + 64 * 128;
                mma_64x64_f16(tmem_dP_dQ + (m_idx * 64 << 16) + n_idx * 64, a0_bot, b0_bot, true, true, 1);
            }
        }
                        
        if (threadIdx.x == 0) {
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                         :: "r"((uint32_t)__cvta_generic_to_shared(&smem.mbar_commit)));
        }
        mbarrier_wait_fn(&smem.mbar_commit, phase_commit);
        phase_commit ^= 1;
        
        uint32_t m_base = q_idx * 128;
        atomicAdd_dQ_fp32(tmem_dP_dQ, dQ_global_fp32, S, 128, m_base, b, h, 1.0f / sqrtf(128.0f));
        
        phase_Q ^= 1;
        __syncthreads();
    }

    float scale_dK = 1.0f / sqrtf(128.0f);
    tmem_epilogue_coalesced_4w_base_fn(tmem_dK, dK_global, smem.PT0, smem.PT1, S, d, kv_idx, 0, 128, 128, b, h, scale_dK);
    tmem_epilogue_coalesced_4w_base_fn(tmem_dV, dV_global, smem.PT0, smem.PT1, S, d, kv_idx, 0, 128, 128, b, h, 1.0f);

    __syncthreads();
    if (threadIdx.x / 32 == 0) {
        tmem_dealloc_cg1_fn(tmem_addr, 512);
    }
}

// -------------------------------------------------------------------------
// Host code
// -------------------------------------------------------------------------

CUresult create_tma_5d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t S, uint64_t H, uint64_t B) {
    cuuint64_t globalDim[5] = {64, 2, S, H, B};
    cuuint64_t globalStrides[4] = {128, 256, 256 * S, 256 * S * H};
    cuuint32_t boxDim[5] = {64, 1, 128, 1, 1};
    cuuint32_t elementStrides[5] = {1, 1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int B = 4;
    int H = 48;
    int S = Q.size(2);
    int d = 128;

    float* D_ptr;
    CUDA_CHECK(cudaMallocAsync(&D_ptr, B * H * S * sizeof(float), stream));

    dim3 grid_D((S + 255) / 256, H, B);
    compute_D_kernel<<<grid_D, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        D_ptr, S, d);

    CUtensorMap tma_Q, tma_K, tma_V, tma_dO;
    if (create_tma_5d_descriptor_2B(&tma_Q, Q.data_ptr(), S, H, B) != CUDA_SUCCESS) exit(1);
    if (create_tma_5d_descriptor_2B(&tma_K, K.data_ptr(), S, H, B) != CUDA_SUCCESS) exit(1);
    if (create_tma_5d_descriptor_2B(&tma_V, V.data_ptr(), S, H, B) != CUDA_SUCCESS) exit(1);
    if (create_tma_5d_descriptor_2B(&tma_dO, dO.data_ptr(), S, H, B) != CUDA_SUCCESS) exit(1);

    uint64_t dQ_size = (uint64_t)B * H * S * d;
    float* dQ_fp32;
    CUDA_CHECK(cudaMallocAsync(&dQ_fp32, dQ_size * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dQ_fp32, 0, dQ_size * sizeof(float), stream));

    dim3 grid(S / 128, H, B);
    dim3 block(128);

    int smem_size = sizeof(SharedStorage);
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    mha_bwd_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO,
        static_cast<const float*>(L.data_ptr()),
        D_ptr,
        dQ_fp32,
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S, d
    );
    CUDA_CHECK(cudaGetLastError());
    
    dim3 grid_conv((dQ_size + 255) / 256);
    convert_fp32_to_bf16<<<grid_conv, 256, 0, stream>>>(dQ_fp32, static_cast<__nv_bfloat16*>(dQ.data_ptr()), dQ_size);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaFreeAsync(D_ptr, stream));
    CUDA_CHECK(cudaFreeAsync(dQ_fp32, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd