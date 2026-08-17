#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <stdio.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N, int a_maj, int b_maj) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (a_maj << 15);
    d |= (b_maj << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint64_t advance_desc_Kmaj(uint64_t desc, int k_steps) {
    uint32_t addr = (desc & 0x3FFF) << 4;
    addr += k_steps * 32; // 16 elements = 32 bytes
    desc &= ~0x3FFFULL;
    desc |= (addr >> 4);
    return desc;
}

__device__ __forceinline__ uint64_t advance_desc_Nmaj(uint64_t desc, int k_steps) {
    uint32_t addr = (desc & 0x3FFF) << 4;
    addr += k_steps * 4096; // 16 rows = 4096 bytes (for 128x128 matrix)
    desc &= ~0x3FFFULL;
    desc |= (addr >> 4);
    return desc;
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    __nv_bfloat16* D, __nv_bfloat16* smem_out, uint32_t tmem_col,
    uint32_t S, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_col + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * BN + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_block * BM + row;
        if (global_row >= S) continue;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = n_block * BN + col_start;
        if (global_col + 3 < N) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
            *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void tmem_epilogue_atomic_4w_fn(
    __nv_bfloat16* D, __nv_bfloat16* smem_out, uint32_t tmem_col,
    uint32_t S, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_col + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * BN + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_block * BM + row;
        if (global_row >= S) continue;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = n_block * BN + col_start;
        if (global_col + 3 < N) {
            __nv_bfloat162 v01 = *reinterpret_cast<__nv_bfloat162*>(&smem_out[row * BN + col_start]);
            __nv_bfloat162 v23 = *reinterpret_cast<__nv_bfloat162*>(&smem_out[row * BN + col_start + 2]);
            atomicAdd(reinterpret_cast<__nv_bfloat162*>(D + global_row * N + global_col), v01);
            atomicAdd(reinterpret_cast<__nv_bfloat162*>(D + global_row * N + global_col + 2), v23);
        }
    }
    __syncthreads();
}

struct SharedStorage {
    __nv_bfloat16 K[128 * 128]; 
    __nv_bfloat16 V[128 * 128]; 
    __nv_bfloat16 Q[128 * 128]; 
    __nv_bfloat16 O[128 * 128]; 
    __nv_bfloat16 dO[128 * 128];
    __nv_bfloat16 dS[128 * 128];
    uint64_t mbar[2];
    float D[128];
    float L[128];
    uint32_t tmem_base;
};

__global__ void mha_bwd_kernel(
    CUtensorMap tma_Q, CUtensorMap tma_K, CUtensorMap tma_V, 
    CUtensorMap tma_O, CUtensorMap tma_dO,
    const float* L_ptr, __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int S, float scale) 
{
    setmaxnreg_inc_sync_fn<256>();
    __syncthreads();
    
    extern __shared__ SharedStorage smem[];
    
    int i = blockIdx.x; // K block index
    int h = blockIdx.y;
    int b = blockIdx.z;
    int S_blocks = (S + 127) / 128;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&smem->mbar[0], 1);
        init_smem_barrier_fn(&smem->mbar[1], 1);
        fence_smem_barrier_init_fn();
    }
    
    if (threadIdx.x < 32) {
        uint32_t a = (uint32_t)__cvta_generic_to_shared(&smem->tmem_base);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(512));
    }
    __syncthreads();
    
    uint32_t tmem_base = smem->tmem_base;
    uint32_t dV_tmem = tmem_base + 0;
    uint32_t dK_tmem = tmem_base + 128;
    uint32_t S_tmem  = tmem_base + 256;
    uint32_t P_tmem  = tmem_base + 256;
    uint32_t dQ_tmem = tmem_base + 256;
    uint32_t dP_tmem = tmem_base + 384;
    uint32_t dS_tmem = tmem_base + 384;
    
    for (int c = 0; c < 128; c += 4) {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dK_tmem + c), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dV_tmem + c), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    
    uint32_t phase_tma = 0;
    uint32_t phase_mma = 0;
    uint64_t b_h_offset = b * gridDim.y * S + h * S;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&smem->mbar[0], 32768 * 2);
        tma_load_2d_fn(&tma_K, &smem->mbar[0], smem->K, 0, b_h_offset + i * 128);
        tma_load_2d_fn(&tma_V, &smem->mbar[0], smem->V, 0, b_h_offset + i * 128);
    }
    mbarrier_wait_fn(&smem->mbar[0], phase_tma); phase_tma ^= 1;
    
    uint64_t desc_K_Kmaj = make_smem_desc_sm100_fn(smem->K, 1, 1024);
    uint64_t desc_Q_Kmaj = make_smem_desc_sm100_fn(smem->Q, 1, 1024);
    uint64_t desc_V_Kmaj = make_smem_desc_sm100_fn(smem->V, 1, 1024);
    uint64_t desc_dO_Kmaj = make_smem_desc_sm100_fn(smem->dO, 1, 1024);
    
    uint64_t desc_dO_Nmaj = make_smem_desc_sm100_fn(smem->dO, 16384, 1024);
    uint64_t desc_Q_Nmaj = make_smem_desc_sm100_fn(smem->Q, 16384, 1024);
    uint64_t desc_K_Nmaj = make_smem_desc_sm100_fn(smem->K, 16384, 1024);
    uint64_t desc_dS_Mmaj = make_smem_desc_sm100_fn(smem->dS, 16384, 1024);
    
    uint32_t idesc_S  = make_idesc(128, 128, 0, 0);
    uint32_t idesc_dP = make_idesc(128, 128, 0, 0);
    uint32_t idesc_dV = make_idesc(128, 128, 0, 1);
    uint32_t idesc_dK = make_idesc(128, 128, 0, 1);
    uint32_t idesc_dQ = make_idesc(128, 128, 1, 1);
    
    float p_reg[32];
    
    for (int j = i; j < S_blocks; ++j) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem->mbar[0], 32768 * 3);
            tma_load_2d_fn(&tma_Q, &smem->mbar[0], smem->Q, 0, b_h_offset + j * 128);
            tma_load_2d_fn(&tma_O, &smem->mbar[0], smem->O, 0, b_h_offset + j * 128);
            tma_load_2d_fn(&tma_dO, &smem->mbar[0], smem->dO, 0, b_h_offset + j * 128);
        }
        
        if (threadIdx.x < 128) {
            int idx = b * gridDim.y * S + h * S + j * 128 + threadIdx.x;
            smem->L[threadIdx.x] = (j * 128 + threadIdx.x < S) ? L_ptr[idx] : 0.0f;
        }
        
        mbarrier_wait_fn(&smem->mbar[0], phase_tma); phase_tma ^= 1;
        __syncthreads();
        
        int row = threadIdx.x;
        float d_val = 0;
        uint32_t smem_O_addr = (uint32_t)__cvta_generic_to_shared(smem->O);
        uint32_t smem_dO_addr = (uint32_t)__cvta_generic_to_shared(smem->dO);
        for (int x = 0; x < 128; x += 8) {
            int swizzled_x = (row % 8) ^ (x / 8);
            uint32_t offset = row * 256 + swizzled_x * 16;
            uint4 vO, vdO;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(vO.x),"=r"(vO.y),"=r"(vO.z),"=r"(vO.w) : "r"(smem_O_addr + offset));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(vdO.x),"=r"(vdO.y),"=r"(vdO.z),"=r"(vdO.w) : "r"(smem_dO_addr + offset));
            
            float2 o01 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&vO.x));
            float2 o23 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&vO.y));
            float2 o45 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&vO.z));
            float2 o67 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&vO.w));
            
            float2 do01 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&vdO.x));
            float2 do23 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&vdO.y));
            float2 do45 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&vdO.z));
            float2 do67 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&vdO.w));
            
            d_val += o01.x * do01.x + o01.y * do01.y;
            d_val += o23.x * do23.x + o23.y * do23.y;
            d_val += o45.x * do45.x + o45.y * do45.y;
            d_val += o67.x * do67.x + o67.y * do67.y;
        }
        smem->D[row] = d_val;
        __syncthreads();
        
        if (threadIdx.x == 0) {
            tcgen05_fence_after_fn();
            for (int k = 0; k < 8; ++k) {
                uint64_t dK = advance_desc_Kmaj(desc_K_Kmaj, k);
                uint64_t dQ = advance_desc_Kmaj(desc_Q_Kmaj, k);
                uint32_t accum = (k == 0) ? 0 : 1;
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(S_tmem), "l"(dK), "l"(dQ), "r"(idesc_S), "r"(accum));
            }
            umma_commit_cg1_fn(&smem->mbar[1]);
            mbarrier_wait_fn(&smem->mbar[1], phase_mma); phase_mma ^= 1;
        }
        __syncthreads();
        
        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(S_tmem + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            int K_pos = i * 128 + threadIdx.x;
            int Q_pos = j * 128 + c;
            
            float f0 = __uint_as_float(r0) * scale;
            float f1 = __uint_as_float(r1) * scale;
            float f2 = __uint_as_float(r2) * scale;
            float f3 = __uint_as_float(r3) * scale;
            
            if (Q_pos + 0 < K_pos || Q_pos + 0 >= S || K_pos >= S) f0 = -INFINITY;
            if (Q_pos + 1 < K_pos || Q_pos + 1 >= S || K_pos >= S) f1 = -INFINITY;
            if (Q_pos + 2 < K_pos || Q_pos + 2 >= S || K_pos >= S) f2 = -INFINITY;
            if (Q_pos + 3 < K_pos || Q_pos + 3 >= S || K_pos >= S) f3 = -INFINITY;
            
            f0 = fast_exp2f_fn((f0 - smem->L[c + 0]) * 1.44269504f);
            f1 = fast_exp2f_fn((f1 - smem->L[c + 1]) * 1.44269504f);
            f2 = fast_exp2f_fn((f2 - smem->L[c + 2]) * 1.44269504f);
            f3 = fast_exp2f_fn((f3 - smem->L[c + 3]) * 1.44269504f);
            
            if (Q_pos + 0 < K_pos || Q_pos + 0 >= S || K_pos >= S) f0 = 0.0f;
            if (Q_pos + 1 < K_pos || Q_pos + 1 >= S || K_pos >= S) f1 = 0.0f;
            if (Q_pos + 2 < K_pos || Q_pos + 2 >= S || K_pos >= S) f2 = 0.0f;
            if (Q_pos + 3 < K_pos || Q_pos + 3 >= S || K_pos >= S) f3 = 0.0f;
            
            uint32_t p01 = pack_bf16_fn(__float_as_uint(f0), __float_as_uint(f1));
            uint32_t p23 = pack_bf16_fn(__float_as_uint(f2), __float_as_uint(f3));
            
            asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32 [%0], {%1,%2};"
                :: "r"(P_tmem + c / 2), "r"(p01), "r"(p23) : "memory");
                
            p_reg[c/4*4 + 0] = f0;
            p_reg[c/4*4 + 1] = f1;
            p_reg[c/4*4 + 2] = f2;
            p_reg[c/4*4 + 3] = f3;
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        __syncthreads();
        
        if (threadIdx.x == 0) {
            tcgen05_fence_after_fn();
            for (int k = 0; k < 8; ++k) {
                uint64_t dV = advance_desc_Kmaj(desc_V_Kmaj, k);
                uint64_t ddO = advance_desc_Kmaj(desc_dO_Kmaj, k);
                uint32_t accum = (k == 0) ? 0 : 1;
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(dP_tmem), "l"(dV), "l"(ddO), "r"(idesc_dP), "r"(accum));
                    
                uint32_t P_tmem_k = P_tmem + k * 8;
                uint64_t ddO_N = advance_desc_Nmaj(desc_dO_Nmaj, k);
                asm volatile(
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, 1;\n"
                    :: "r"(dV_tmem), "r"(P_tmem_k), "l"(ddO_N), "r"(idesc_dV));
            }
            umma_commit_cg1_fn(&smem->mbar[1]);
            mbarrier_wait_fn(&smem->mbar[1], phase_mma); phase_mma ^= 1;
        }
        __syncthreads();
        
        uint32_t smem_dS_addr = (uint32_t)__cvta_generic_to_shared(smem->dS);
        for (int c = 0; c < 128; c += 8) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(dP_tmem + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float ds0 = p_reg[c+0] * (__uint_as_float(r0) - smem->D[c+0]);
            float ds1 = p_reg[c+1] * (__uint_as_float(r1) - smem->D[c+1]);
            float ds2 = p_reg[c+2] * (__uint_as_float(r2) - smem->D[c+2]);
            float ds3 = p_reg[c+3] * (__uint_as_float(r3) - smem->D[c+3]);
            float ds4 = p_reg[c+4] * (__uint_as_float(r4) - smem->D[c+4]);
            float ds5 = p_reg[c+5] * (__uint_as_float(r5) - smem->D[c+5]);
            float ds6 = p_reg[c+6] * (__uint_as_float(r6) - smem->D[c+6]);
            float ds7 = p_reg[c+7] * (__uint_as_float(r7) - smem->D[c+7]);
            
            uint32_t s01 = pack_bf16_fn(__float_as_uint(ds0), __float_as_uint(ds1));
            uint32_t s23 = pack_bf16_fn(__float_as_uint(ds2), __float_as_uint(ds3));
            uint32_t s45 = pack_bf16_fn(__float_as_uint(ds4), __float_as_uint(ds5));
            uint32_t s67 = pack_bf16_fn(__float_as_uint(ds6), __float_as_uint(ds7));
            
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                :: "r"(dS_tmem + c / 2), "r"(s01), "r"(s23), "r"(s45), "r"(s67) : "memory");
                
            int swizzled_x = (threadIdx.x % 8) ^ (c / 8);
            uint32_t addr = smem_dS_addr + threadIdx.x * 256 + swizzled_x * 16;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                :: "r"(addr), "r"(s01), "r"(s23), "r"(s45), "r"(s67));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        fence_proxy_async_fn();
        __syncthreads();
        
        if (threadIdx.x == 0) {
            tcgen05_fence_after_fn();
            for (int k = 0; k < 8; ++k) {
                uint32_t dS_tmem_k = dS_tmem + k * 8;
                uint64_t dQ_N = advance_desc_Nmaj(desc_Q_Nmaj, k);
                asm volatile(
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, 1;\n"
                    :: "r"(dK_tmem), "r"(dS_tmem_k), "l"(dQ_N), "r"(idesc_dK));
                    
                uint64_t ddS_M = advance_desc_Nmaj(desc_dS_Mmaj, k);
                uint64_t dK_N = advance_desc_Nmaj(desc_K_Nmaj, k);
                uint32_t accum = (k == 0) ? 0 : 1;
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(dQ_tmem), "l"(ddS_M), "l"(dK_N), "r"(idesc_dQ), "r"(accum));
            }
            umma_commit_cg1_fn(&smem->mbar[1]);
            mbarrier_wait_fn(&smem->mbar[1], phase_mma); phase_mma ^= 1;
        }
        __syncthreads();
        
        tmem_epilogue_atomic_4w_fn(dQ, smem->Q, dQ_tmem, S, 128, j, 0, 128, 128);
    }
    
    tmem_epilogue_coalesced_4w_fn(dK, smem->K, dK_tmem, S, 128, i, 0, 128, 128);
    tmem_epilogue_coalesced_4w_fn(dV, smem->V, dV_tmem, S, 128, i, 0, 128, 128);
    
    __syncthreads();
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_base), "r"(512));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

namespace tvm_ffi_example_cuda {

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides, boxDim,
        elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, 
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
         
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    float scale = 1.0f / sqrtf(128.0f);
    
    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, B * H * S * 128 * sizeof(uint16_t), stream));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, B * H * S, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, B * H * S, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, B * H * S, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_O, O.data_ptr(), 128, B * H * S, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dO, dO.data_ptr(), 128, B * H * S, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    int S_blocks = (S + 127) / 128;
    dim3 grid(S_blocks, H, B);
    dim3 block(128);
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = sizeof(SharedStorage);
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    cudaLaunchKernelEx(&config, mha_bwd_kernel, tma_Q, tma_K, tma_V, tma_O, tma_dO, 
        static_cast<const float*>(L.data_ptr()), 
        static_cast<__nv_bfloat16*>(dQ.data_ptr()), 
        static_cast<__nv_bfloat16*>(dK.data_ptr()), 
        static_cast<__nv_bfloat16*>(dV.data_ptr()), 
        S, scale);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda