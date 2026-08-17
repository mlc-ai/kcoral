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

// ---------------- PTX Wrappers ----------------

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
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void umma_f16_cta(
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
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ void cross_proxy_fence() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t major) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    d |= (uint64_t)major << 15;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
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

__device__ __forceinline__ uint32_t swizzle_128B_idx(uint32_t row, uint32_t col) {
    uint32_t x_chunk = col / 8; 
    uint32_t y_swizzled = (row % 8) ^ x_chunk;
    return row * 64 + y_swizzled * 8 + (col % 8);
}

__global__ void cast_fp32_to_bf16(__nv_bfloat16* out, const float* in, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(in[idx]);
    }
}

// ---------------- Kernel ----------------

__global__ __launch_bounds__(128) void bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    float* fp32_dQ, float* fp32_dK, float* fp32_dV,
    const float* L, uint32_t S, uint32_t d, uint32_t B, uint32_t H) 
{
    extern __shared__ __align__(128) uint8_t smem_raw[];
    char* smem_pool = (char*)smem_raw;
    
    uint32_t smem_offset = 0;
    uint32_t base_addr = (uint32_t)__cvta_generic_to_shared(smem_pool);
    uint32_t rem = base_addr % 1024;
    if (rem != 0) {
        smem_offset = 1024 - rem;
    }
    char* smem_aligned = smem_pool + smem_offset;
    
    uint32_t offset = 0;
    #define ALLOC_SMEM_128B(name, size) \
        name = (__nv_bfloat16*)(smem_aligned + offset); \
        offset += size; \
        offset = (offset + 1023) & ~1023;
    
    __nv_bfloat16 *smem_Q_0, *smem_Q_1, *smem_K_0, *smem_K_1, *smem_V_0, *smem_V_1;
    __nv_bfloat16 *smem_dO_0, *smem_dO_1, *smem_O_0, *smem_O_1, *smem_P_0, *smem_P_1, *smem_dS_0, *smem_dS_1;
    float* smem_D;
    uint64_t* mbar;
    
    ALLOC_SMEM_128B(smem_Q_0, 8192);
    ALLOC_SMEM_128B(smem_Q_1, 8192);
    ALLOC_SMEM_128B(smem_dO_0, 8192);
    ALLOC_SMEM_128B(smem_dO_1, 8192);
    ALLOC_SMEM_128B(smem_O_0, 8192);
    ALLOC_SMEM_128B(smem_O_1, 8192);
    ALLOC_SMEM_128B(smem_K_0, 8192);
    ALLOC_SMEM_128B(smem_K_1, 8192);
    ALLOC_SMEM_128B(smem_V_0, 8192);
    ALLOC_SMEM_128B(smem_V_1, 8192);
    ALLOC_SMEM_128B(smem_P_0, 8192);
    ALLOC_SMEM_128B(smem_P_1, 8192);
    ALLOC_SMEM_128B(smem_dS_0, 8192);
    ALLOC_SMEM_128B(smem_dS_1, 8192);
    
    offset = (offset + 3) & ~3;
    smem_D = (float*)(smem_aligned + offset);
    offset += 512;
    
    offset = (offset + 7) & ~7;
    mbar = (uint64_t*)(smem_aligned + offset);
    offset += 8;
    
    #undef ALLOC_SMEM_128B

    uint32_t* tmem_base = (uint32_t*)smem_D; 
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        tmem_alloc_fn(tmem_base, 512);
    }
    __syncthreads();

    uint32_t q_outer = gridDim.x * blockIdx.y + blockIdx.x;
    uint32_t q_tile = q_outer / (B * H);
    uint32_t b_idx = (q_outer % (B * H)) / H;
    uint32_t h_idx = q_outer % H;
    uint32_t num_kv_outer = (S + 127) / 128;
    uint32_t phase_tma = 0;
    uint32_t phase_umma = 0;

    uint32_t seq_offset_q = b_idx * H * S + h_idx * S + q_tile * 128;
    uint32_t d_offset = b_idx * H * S * 128 + h_idx * S * 128;
    
    float* my_dQ = fp32_dQ + d_offset + q_tile * 128 * 128;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 98304);
        tma_load_2d_fn(&tma_Q, mbar, smem_Q_0, 0, seq_offset_q);
        tma_load_2d_fn(&tma_Q, mbar, smem_Q_1, 64, seq_offset_q);
        tma_load_2d_fn(&tma_dO, mbar, smem_dO_0, 0, seq_offset_q);
        tma_load_2d_fn(&tma_dO, mbar, smem_dO_1, 64, seq_offset_q);
        tma_load_2d_fn(&tma_O, mbar, smem_O_0, 0, seq_offset_q);
        tma_load_2d_fn(&tma_O, mbar, smem_O_1, 64, seq_offset_q);
    }
    mbarrier_wait_fn(mbar, phase_tma);
    phase_tma ^= 1;
    
    float sum = 0;
    for (uint32_t i = 0; i < 8; i++) {
        uint4 do_vec0 = *(uint4*)&smem_dO_0[swizzle_128B_idx(threadIdx.x, i * 8)];
        uint4 o_vec0 = *(uint4*)&smem_O_0[swizzle_128B_idx(threadIdx.x, i * 8)];
        
        __nv_bfloat16* do_bf0 = (__nv_bfloat16*)&do_vec0;
        __nv_bfloat16* o_bf0 = (__nv_bfloat16*)&o_vec0;
        for(int j=0; j<8; j++) {
            sum += __bfloat162float(do_bf0[j]) * __bfloat162float(o_bf0[j]);
        }
    }
    for (uint32_t i = 0; i < 8; i++) {
        uint4 do_vec1 = *(uint4*)&smem_dO_1[swizzle_128B_idx(threadIdx.x, i * 8)];
        uint4 o_vec1 = *(uint4*)&smem_O_1[swizzle_128B_idx(threadIdx.x, i * 8)];
        
        __nv_bfloat16* do_bf1 = (__nv_bfloat16*)&do_vec1;
        __nv_bfloat16* o_bf1 = (__nv_bfloat16*)&o_vec1;
        for(int j=0; j<8; j++) {
            sum += __bfloat162float(do_bf1[j]) * __bfloat162float(o_bf1[j]);
        }
    }
    __shared__ float partial_D[128];
    partial_D[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x < 128) {
        smem_D[threadIdx.x] = partial_D[threadIdx.x];
    }
    __syncthreads();

    uint32_t tmem_S = 0;
    uint32_t tmem_dP = 128;
    uint32_t tmem_dV = 256;
    uint32_t tmem_dK = tmem_S;
    uint32_t tmem_dQ = tmem_dP;
    
    bool first_k_block = true;
    const float scale = 0.08838834764831845f;

    for (uint32_t kv_outer = 0; kv_outer < num_kv_outer; kv_outer++) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 65536);
            uint32_t seq_offset_kv = b_idx * H * S + h_idx * S + kv_outer * 128;
            tma_load_2d_fn(&tma_K, mbar, smem_K_0, 0, seq_offset_kv);
            tma_load_2d_fn(&tma_K, mbar, smem_K_1, 64, seq_offset_kv);
            tma_load_2d_fn(&tma_V, mbar, smem_V_0, 0, seq_offset_kv);
            tma_load_2d_fn(&tma_V, mbar, smem_V_1, 64, seq_offset_kv);
        }
        mbarrier_wait_fn(mbar, phase_tma);
        phase_tma ^= 1;
        
        uint64_t desc_Q_0_k = make_smem_desc(smem_Q_0, 1, 1024, 0);
        uint64_t desc_Q_1_k = make_smem_desc(smem_Q_1, 1, 1024, 0);
        uint64_t desc_K_0_k = make_smem_desc(smem_K_0, 1, 1024, 0);
        uint64_t desc_K_1_k = make_smem_desc(smem_K_1, 1, 1024, 0);
        
        uint32_t idesc_S = make_instr_desc_fn(128, 128, 0, 0);
        cross_proxy_fence();
        if (threadIdx.x == 0) {
            for(int k=0; k<64; ++k) {
                uint64_t da0 = desc_Q_0_k + (k * 2);
                uint64_t db0 = desc_K_0_k + (k * 2);
                umma_f16_cta(tmem_S, da0, db0, idesc_S, (k==0 && first_k_block)?0:1);
            }
            for(int k=0; k<64; ++k) {
                uint64_t da1 = desc_Q_1_k + (k * 2);
                uint64_t db1 = desc_K_1_k + (k * 2);
                umma_f16_cta(tmem_S, da1, db1, idesc_S, 1);
            }
            umma_commit_1sm_fn(mbar);
        }
        mbarrier_wait_fn(mbar, phase_umma);
        phase_umma ^= 1;
        
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t taddr_S = (threadIdx.x << 16) | col;
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(taddr_S));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float s0 = __uint_as_float(r0);
            float s1 = __uint_as_float(r1);
            float s2 = __uint_as_float(r2);
            float s3 = __uint_as_float(r3);
            
            float LSE_val = (q_tile * 128 + threadIdx.x < S) ? L[b_idx * H * S + h_idx * S + q_tile * 128 + threadIdx.x] : 0.0f;
            float p0 = expf(s0 * scale - LSE_val);
            float p1 = expf(s1 * scale - LSE_val);
            float p2 = expf(s2 * scale - LSE_val);
            float p3 = expf(s3 * scale - LSE_val);
            
            __nv_bfloat16 bp0 = __float2bfloat16(p0);
            __nv_bfloat16 bp1 = __float2bfloat16(p1);
            __nv_bfloat16 bp2 = __float2bfloat16(p2);
            __nv_bfloat16 bp3 = __float2bfloat16(p3);
            
            uint32_t packed0 = ((uint32_t)*(uint16_t*)&bp1 << 16) | *(uint16_t*)&bp0;
            uint32_t packed1 = ((uint32_t)*(uint16_t*)&bp3 << 16) | *(uint16_t*)&bp2;
            
            if (col < 64) {
                *(uint32_t*)&smem_P_0[swizzle_128B_idx(threadIdx.x, col)] = packed0;
                *(uint32_t*)&smem_P_0[swizzle_128B_idx(threadIdx.x, col + 2)] = packed1;
            } else {
                *(uint32_t*)&smem_P_1[swizzle_128B_idx(threadIdx.x, col - 64)] = packed0;
                *(uint32_t*)&smem_P_1[swizzle_128B_idx(threadIdx.x, col - 64 + 2)] = packed1;
            }
        }
        __syncthreads();
        
        uint64_t desc_V_0_k = make_smem_desc(smem_V_0, 1, 1024, 0);
        uint64_t desc_V_1_k = make_smem_desc(smem_V_1, 1, 1024, 0);
        uint64_t desc_dO_0_k = make_smem_desc(smem_dO_0, 1, 1024, 0);
        uint64_t desc_dO_1_k = make_smem_desc(smem_dO_1, 1, 1024, 0);
        
        uint32_t idesc_dP = make_instr_desc_fn(128, 128, 0, 0);
        cross_proxy_fence();
        if (threadIdx.x == 0) {
            for(int k=0; k<64; ++k) {
                umma_f16_cta(tmem_dP, desc_V_0_k + k*2, desc_dO_0_k + k*2, idesc_dP, (k==0)?0:1);
            }
            for(int k=0; k<64; ++k) {
                umma_f16_cta(tmem_dP, desc_V_1_k + k*2, desc_dO_1_k + k*2, idesc_dP, 1);
            }
            umma_commit_1sm_fn(mbar);
        }
        mbarrier_wait_fn(mbar, phase_umma);
        phase_umma ^= 1;
        
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t taddr_dP = (threadIdx.x << 16) | col;
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(taddr_dP));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float dp0 = __uint_as_float(r0);
            float dp1 = __uint_as_float(r1);
            float dp2 = __uint_as_float(r2);
            float dp3 = __uint_as_float(r3);
            
            float p0, p1, p2, p3;
            if (col < 64) {
                uint32_t val0 = *(uint32_t*)&smem_P_0[swizzle_128B_idx(threadIdx.x, col)];
                p0 = __bfloat162float(*(__nv_bfloat16*)&val0);
                p1 = __bfloat162float(*(__nv_bfloat16*)&(val0 >> 16));
                
                uint32_t val1 = *(uint32_t*)&smem_P_0[swizzle_128B_idx(threadIdx.x, col + 2)];
                p2 = __bfloat162float(*(__nv_bfloat16*)&val1);
                p3 = __bfloat162float(*(__nv_bfloat16*)&(val1 >> 16));
            } else {
                uint32_t val0 = *(uint32_t*)&smem_P_1[swizzle_128B_idx(threadIdx.x, col - 64)];
                p0 = __bfloat162float(*(__nv_bfloat16*)&val0);
                p1 = __bfloat162float(*(__nv_bfloat16*)&(val0 >> 16));
                
                uint32_t val1 = *(uint32_t*)&smem_P_1[swizzle_128B_idx(threadIdx.x, col - 64 + 2)];
                p2 = __bfloat162float(*(__nv_bfloat16*)&val1);
                p3 = __bfloat162float(*(__nv_bfloat16*)&(val1 >> 16));
            }
            
            float ds0 = p0 * (dp0 - smem_D[threadIdx.x]);
            float ds1 = p1 * (dp1 - smem_D[threadIdx.x]);
            float ds2 = p2 * (dp2 - smem_D[threadIdx.x]);
            float ds3 = p3 * (dp3 - smem_D[threadIdx.x]);
            
            __nv_bfloat16 bds0 = __float2bfloat16(ds0);
            __nv_bfloat16 bds1 = __float2bfloat16(ds1);
            __nv_bfloat16 bds2 = __float2bfloat16(ds2);
            __nv_bfloat16 bds3 = __float2bfloat16(ds3);
            
            uint32_t pdsp0 = ((uint32_t)*(uint16_t*)&bds1 << 16) | *(uint16_t*)&bds0;
            uint32_t pdsp1 = ((uint32_t)*(uint16_t*)&bds3 << 16) | *(uint16_t*)&bds2;
            
            if (col < 64) {
                *(uint32_t*)&smem_dS_0[swizzle_128B_idx(threadIdx.x, col)] = pdsp0;
                *(uint32_t*)&smem_dS_0[swizzle_128B_idx(threadIdx.x, col + 2)] = pdsp1;
            } else {
                *(uint32_t*)&smem_dS_1[swizzle_128B_idx(threadIdx.x, col - 64)] = pdsp0;
                *(uint32_t*)&smem_dS_1[swizzle_128B_idx(threadIdx.x, col - 64 + 2)] = pdsp1;
            }
        }
        __syncthreads();
        
        uint32_t idesc_dV = make_instr_desc_fn(128, 128, 1, 1);
        uint64_t desc_P_0_mn = make_smem_desc(smem_P_0, 16384, 1024, 1);
        uint64_t desc_P_1_mn = make_smem_desc(smem_P_1, 16384, 1024, 1);
        uint64_t desc_dO_0_mn = make_smem_desc(smem_dO_0, 16384, 1024, 1);
        uint64_t desc_dO_1_mn = make_smem_desc(smem_dO_1, 16384, 1024, 1);
        
        cross_proxy_fence();
        if (threadIdx.x == 0) {
            for(int k=0; k<64; ++k) {
                uint64_t da0 = desc_P_0_mn + (k * 1024);
                uint64_t db0 = desc_dO_0_mn + (k * 1024);
                umma_f16_cta(tmem_dV, da0, db0, idesc_dV, (k==0)?0:1);
            }
            for(int k=0; k<64; ++k) {
                uint64_t da1 = desc_P_1_mn + (k * 1024);
                uint64_t db1 = desc_dO_1_mn + (k * 1024);
                umma_f16_cta(tmem_dV, da1, db1, idesc_dV, 1);
            }
            umma_commit_1sm_fn(mbar);
        }
        
        uint32_t idesc_dK = make_instr_desc_fn(128, 128, 1, 1);
        uint64_t desc_dS_0_mn_dK = make_smem_desc(smem_dS_0, 16384, 1024, 1);
        uint64_t desc_dS_1_mn_dK = make_smem_desc(smem_dS_1, 16384, 1024, 1);
        uint64_t desc_Q_0_mn_dK = make_smem_desc(smem_Q_0, 16384, 1024, 1);
        uint64_t desc_Q_1_mn_dK = make_smem_desc(smem_Q_1, 16384, 1024, 1);
        
        if (threadIdx.x == 0) {
            for(int k=0; k<64; ++k) {
                uint64_t da0 = desc_dS_0_mn_dK + (k * 1024);
                uint64_t db0 = desc_Q_0_mn_dK + (k * 1024);
                umma_f16_cta(tmem_dK, da0, db0, idesc_dK, (k==0)?0:1);
            }
            for(int k=0; k<64; ++k) {
                uint64_t da1 = desc_dS_1_mn_dK + (k * 1024);
                uint64_t db1 = desc_Q_1_mn_dK + (k * 1024);
                umma_f16_cta(tmem_dK, da1, db1, idesc_dK, 1);
            }
            umma_commit_1sm_fn(mbar);
        }
        mbarrier_wait_fn(mbar, phase_umma);
        phase_umma ^= 1;
        
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t taddr_dV = (threadIdx.x << 16) | col;
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(taddr_dV));
            
            uint32_t taddr_dK = (threadIdx.x << 16) | col;
            uint32_t r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(taddr_dK));
            
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            float f4 = __uint_as_float(r4);
            float f5 = __uint_as_float(r5);
            float f6 = __uint_as_float(r6);
            float f7 = __uint_as_float(r7);
            
            uint32_t row = threadIdx.x;
            uint32_t g_col0 = col;
            uint32_t g_col1 = col + 1;
            uint32_t g_col2 = col + 2;
            uint32_t g_col3 = col + 3;
            
            if (row < 128) {
                if (g_col0 < 128) atomicAdd(&fp32_dV[d_offset + kv_outer * 128 * 128 + row * 128 + g_col0], f0);
                if (g_col1 < 128) atomicAdd(&fp32_dV[d_offset + kv_outer * 128 * 128 + row * 128 + g_col1], f1);
                if (g_col2 < 128) atomicAdd(&fp32_dV[d_offset + kv_outer * 128 * 128 + row * 128 + g_col2], f2);
                if (g_col3 < 128) atomicAdd(&fp32_dV[d_offset + kv_outer * 128 * 128 + row * 128 + g_col3], f3);
                
                if (g_col0 < 128) atomicAdd(&fp32_dK[d_offset + kv_outer * 128 * 128 + row * 128 + g_col0], f4);
                if (g_col1 < 128) atomicAdd(&fp32_dK[d_offset + kv_outer * 128 * 128 + row * 128 + g_col1], f5);
                if (g_col2 < 128) atomicAdd(&fp32_dK[d_offset + kv_outer * 128 * 128 + row * 128 + g_col2], f6);
                if (g_col3 < 128) atomicAdd(&fp32_dK[d_offset + kv_outer * 128 * 128 + row * 128 + g_col3], f7);
            }
        }
        
        uint32_t idesc_dQ = make_instr_desc_fn(128, 128, 0, 1);
        uint64_t desc_dS_0_maj0 = make_smem_desc(smem_dS_0, 1, 1024, 0);
        uint64_t desc_dS_1_maj0 = make_smem_desc(smem_dS_1, 1, 1024, 0);
        uint64_t desc_K_0_maj1 = make_smem_desc(smem_K_0, 8192, 1024, 1);
        uint64_t desc_K_1_maj1 = make_smem_desc(smem_K_1, 8192, 1024, 1);
        
        cross_proxy_fence();
        if (threadIdx.x == 0) {
            for(int k=0; k<64; ++k) {
                umma_f16_cta(tmem_dQ, desc_dS_0_maj0 + k*2, desc_K_0_maj1 + k*128, idesc_dQ, (k==0 && first_k_block)?0:1);
            }
            for(int k=0; k<64; ++k) {
                umma_f16_cta(tmem_dQ, desc_dS_1_maj0 + k*2, desc_K_1_maj1 + k*128, idesc_dQ, 1);
            }
            umma_commit_1sm_fn(mbar);
        }
        mbarrier_wait_fn(mbar, phase_umma);
        phase_umma ^= 1;
        
        first_k_block = false;
    }
    
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t taddr_dQ = (threadIdx.x << 16) | col;
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(taddr_dQ));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        uint32_t row = threadIdx.x;
        uint32_t g_col0 = col;
        uint32_t g_col1 = col + 1;
        uint32_t g_col2 = col + 2;
        uint32_t g_col3 = col + 3;
        
        if (g_row < 128) {
            if (g_col0 < 128) my_dQ[g_row * 128 + g_col0] = f0;
            if (g_col1 < 128) my_dQ[g_row * 128 + g_col1] = f1;
            if (g_col2 < 128) my_dQ[g_row * 128 + g_col2] = f2;
            if (g_col3 < 128) my_dQ[g_row * 128 + g_col3] = f3;
        }
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(*tmem_base, 512);
    }
    __syncthreads();
}

namespace tvm_ffi_attention_bwd {

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
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
         
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t d = Q.size(3);
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    CUresult res;
    res = create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA Q failed\n"); exit(1); }
    res = create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA K failed\n"); exit(1); }
    res = create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA V failed\n"); exit(1); }
    res = create_tma_2d_descriptor_2B(&tma_O, O.data_ptr(), 128, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA O failed\n"); exit(1); }
    res = create_tma_2d_descriptor_2B(&tma_dO, dO.data_ptr(), 128, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA dO failed\n"); exit(1); }
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    float *fp32_dQ, *fp32_dK, *fp32_dV;
    CUDA_CHECK(cudaMallocAsync(&fp32_dQ, B * H * S * d * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&fp32_dK, B * H * S * d * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&fp32_dV, B * H * S * d * sizeof(float), stream));
    
    CUDA_CHECK(cudaMemsetAsync(fp32_dQ, 0, B * H * S * d * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(fp32_dK, 0, B * H * S * d * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(fp32_dV, 0, B * H * S * d * sizeof(float), stream));
    
    uint32_t num_q_tiles = (S + 127) / 128;
    dim3 grid(num_q_tiles, B * H);
    dim3 block(128);
    
    uint32_t smem_size = 240000;
    CUDA_CHECK(cudaFuncSetAttribute(
        bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ));
    
    bwd_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO,
        fp32_dQ, fp32_dK, fp32_dV,
        static_cast<const float*>(L.data_ptr()),
        S, d, B, H
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    size_t n = B * H * S * d;
    cast_fp32_to_bf16<<<(n + 255) / 256, 256, 0, stream>>>(
        static_cast<__nv_bfloat16*>(dQ.data_ptr()), fp32_dQ, n);
    cast_fp32_to_bf16<<<(n + 255) / 256, 256, 0, stream>>>(
        static_cast<__nv_bfloat16*>(dK.data_ptr()), fp32_dK, n);
    cast_fp32_to_bf16<<<(n + 255) / 256, 256, 0, stream>>>(
        static_cast<__nv_bfloat16*>(dV.data_ptr()), fp32_dV, n);
    
    CUDA_CHECK(cudaFreeAsync(fp32_dQ, stream));
    CUDA_CHECK(cudaFreeAsync(fp32_dK, stream));
    CUDA_CHECK(cudaFreeAsync(fp32_dV, stream));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_attention_bwd::run);

}  // namespace tvm_ffi_attention_bwd