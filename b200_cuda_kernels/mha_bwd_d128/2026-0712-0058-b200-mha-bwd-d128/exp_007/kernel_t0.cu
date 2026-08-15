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
        ".mbarrier::arrive::one.shared.b64"
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

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
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

__device__ __forceinline__ uint32_t make_instr_desc_fn_maj(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
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

__device__ __forceinline__ void tmem_epilogue_bf16_transposed(
    __nv_bfloat16* smem_out, uint32_t stride_row,
    uint32_t BN) {
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t taddr = col;
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(taddr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        uint32_t row = threadIdx.x;
        smem_out[row * stride_row + col + 0] = __float2bfloat16(f0);
        smem_out[row * stride_row + col + 1] = __float2bfloat16(f1);
        smem_out[row * stride_row + col + 2] = __float2bfloat16(f2);
        smem_out[row * stride_row + col + 3] = __float2bfloat16(f3);
    }
}

// ---------------- Kernel ----------------

__global__ __launch_bounds__(128) void bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    const float* L, uint32_t S, uint32_t d, uint32_t B, uint32_t H) 
{
    extern __shared__ __align__(128) uint8_t smem_raw[];
    char* smem_pool = (char*)smem_raw;
    
    uint32_t smem_offset = 0;
    uint32_t align_up[1024] = {0}; // Padding for base alignment
    uint32_t base_addr = (uint32_t)__cvta_generic_to_shared(smem_pool);
    uint32_t rem = base_addr % 1024;
    if (rem != 0) {
        smem_offset = 1024 - rem;
    }
    char* smem_aligned = smem_pool + smem_offset;
    
    uint32_t offset = 0;
    #define ALLOC_SMEM(name, size) \
        name = (__nv_bfloat16*)(smem_aligned + offset); \
        offset += size; \
        offset = (offset + 1023) & ~1023;
    
    __nv_bfloat16 *smem_Q_0, *smem_Q_1, *smem_K_0, *smem_K_1, *smem_V_0, *smem_V_1;
    __nv_bfloat16 *smem_dO_0, *smem_dO_1, *smem_O_0, *smem_O_1, *smem_P, *smem_dS;
    float* smem_D;
    uint64_t* mbar;
    
    ALLOC_SMEM(smem_Q_0, 16384);
    ALLOC_SMEM(smem_Q_1, 16384);
    ALLOC_SMEM(smem_K_0, 16384);
    ALLOC_SMEM(smem_K_1, 16384);
    ALLOC_SMEM(smem_V_0, 16384);
    ALLOC_SMEM(smem_V_1, 16384);
    ALLOC_SMEM(smem_dO_0, 16384);
    ALLOC_SMEM(smem_dO_1, 16384);
    ALLOC_SMEM(smem_O_0, 16384);
    ALLOC_SMEM(smem_O_1, 16384);
    ALLOC_SMEM(smem_P, 16384);
    ALLOC_SMEM(smem_dS, 16384);
    
    offset = (offset + 3) & ~3;
    smem_D = (float*)(smem_aligned + offset);
    offset += 512;
    
    offset = (offset + 7) & ~7;
    mbar = (uint64_t*)(smem_aligned + offset);
    offset += 8;
    
    #undef ALLOC_SMEM

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

    uint32_t kv_outer = gridDim.x * blockIdx.y + blockIdx.x;
    uint32_t kv_tile = kv_outer / (B * H);
    uint32_t b_idx = (kv_outer % (B * H)) / H;
    uint32_t h_idx = kv_outer % H;
    uint32_t num_q_tiles = (S + 127) / 128;
    uint32_t phase = 0;

    uint32_t seq_offset_kv = b_idx * H * S + h_idx * S + kv_tile * 128;
    uint32_t d_offset = b_idx * H * S * 128 + h_idx * S * 128;
    
    __nv_bfloat16* global_dK = dK + d_offset + kv_tile * 128 * 128;
    __nv_bfloat16* global_dV = dV + d_offset + kv_tile * 128 * 128;
    __nv_bfloat16* global_dQ = dQ + d_offset;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 65536);
        tma_load_2d_fn(&tma_K, mbar, smem_K_0, 0, seq_offset_kv);
        tma_load_2d_fn(&tma_K, mbar, smem_K_1, 64, seq_offset_kv);
        tma_load_2d_fn(&tma_V, mbar, smem_V_0, 0, seq_offset_kv);
        tma_load_2d_fn(&tma_V, mbar, smem_V_1, 64, seq_offset_kv);
    }
    mbarrier_wait_fn(mbar, phase);
    phase ^= 1;
    
    uint64_t desc_K_0 = make_smem_desc_sm100_fn(smem_K_0, 1024, 1024);
    uint64_t desc_K_1 = make_smem_desc_sm100_fn(smem_K_1, 1024, 1024);
    uint64_t desc_V_0 = make_smem_desc_sm100_fn(smem_V_0, 1024, 1024);
    uint64_t desc_V_1 = make_smem_desc_sm100_fn(smem_V_1, 1024, 1024);
    
    uint32_t tmem_dV = (0 << 16) | 128;
    uint32_t tmem_dK = (0 << 16) | 384;
    uint32_t tmem_S = (0 << 16) | 0;
    uint32_t tmem_P = (0 << 16) | 0;
    uint32_t tmem_dP = (0 << 16) | 256;
    uint32_t tmem_dS = (0 << 16) | 256;
    uint32_t tmem_dQ = (0 << 16) | 0;

    for (uint32_t q_outer = 0; q_outer < num_q_tiles; q_outer++) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 65536);
            uint32_t seq_offset_q = b_idx * H * S + h_idx * S + q_outer * 128;
            tma_load_2d_fn(&tma_Q, mbar, smem_Q_0, 0, seq_offset_q);
            tma_load_2d_fn(&tma_Q, mbar, smem_Q_1, 64, seq_offset_q);
            tma_load_2d_fn(&tma_dO, mbar, smem_dO_0, 0, seq_offset_q);
            tma_load_2d_fn(&tma_dO, mbar, smem_dO_1, 64, seq_offset_q);
            tma_load_2d_fn(&tma_O, mbar, smem_O_0, 0, seq_offset_q);
            tma_load_2d_fn(&tma_O, mbar, smem_O_1, 64, seq_offset_q);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        float sum = 0;
        uint32_t row = threadIdx.x;
        for (uint32_t col = 0; col < 64; col++) {
            uint32_t idx = row * 64 + col;
            float do0 = __bfloat162float(smem_dO_0[idx]);
            float o0 = __bfloat162float(smem_O_0[idx]);
            sum += do0 * o0;
            
            float do1 = __bfloat162float(smem_dO_1[idx]);
            float o1 = __bfloat162float(smem_O_1[idx]);
            sum += do1 * o1;
        }
        __shared__ float partial_D[128];
        partial_D[row] = sum;
        __syncthreads();
        if (row < 128) {
            smem_D[row] = partial_D[row];
        }
        __syncthreads();
        
        uint32_t idesc_S = make_instr_desc_fn(128, 128);
        uint64_t desc_Q_0 = make_smem_desc_sm100_fn(smem_Q_0, 1024, 1024);
        uint64_t desc_Q_1 = make_smem_desc_sm100_fn(smem_Q_1, 1024, 1024);
        
        if (threadIdx.x == 0) {
            for(int k=0; k<4; ++k) {
                uint64_t da = desc_K_0 + k;
                uint64_t db = desc_Q_0 + k;
                umma_f16_cg1_fn(tmem_S, da, db, idesc_S, (k==0)?0:1);
            }
            for(int k=0; k<4; ++k) {
                uint64_t da = desc_K_1 + k;
                uint64_t db = desc_Q_1 + k;
                umma_f16_cg1_fn(tmem_S, da, db, idesc_S, 1);
            }
            umma_commit_1sm_fn(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        float LSE_val = (q_outer * 128 + threadIdx.x < S) ? L[b_idx * H * S + h_idx * S + q_outer * 128 + threadIdx.x] : 0.0f;
        tmem_epilogue_bf16_transposed(smem_P, 128, 128);
        __syncthreads(); 
        
        for(uint32_t i = threadIdx.x; i < 128*128; i += 128) {
            uint32_t r = i / 128;
            uint32_t c = i % 128;
            float s = __bfloat162float(smem_P[i]);
            float p = expf(s * 0.08838834764831845f - LSE_val);
            smem_P[r * 128 + c] = __float2bfloat16(p);
        }
        __syncthreads();
        
        uint32_t idesc_dP = make_instr_desc_fn(128, 128);
        uint64_t desc_dO_0 = make_smem_desc_sm100_fn(smem_dO_0, 1024, 1024);
        uint64_t desc_dO_1 = make_smem_desc_sm100_fn(smem_dO_1, 1024, 1024);
        
        if (threadIdx.x == 0) {
            for(int k=0; k<4; ++k) {
                uint64_t da = desc_V_0 + k;
                uint64_t db = desc_dO_0 + k;
                umma_f16_cg1_fn(tmem_dP, da, db, idesc_dP, (k==0)?0:1);
            }
            for(int k=0; k<4; ++k) {
                uint64_t da = desc_V_1 + k;
                uint64_t db = desc_dO_1 + k;
                umma_f16_cg1_fn(tmem_dP, da, db, idesc_dP, 1);
            }
            umma_commit_1sm_fn(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        tmem_epilogue_bf16_transposed(smem_dS, 128, 128);
        __syncthreads();
        
        for(uint32_t i = threadIdx.x; i < 128*128; i += 128) {
            uint32_t r = i / 128; 
            uint32_t c = i % 128; 
            float dp = __bfloat162float(smem_dS[i]);
            float p = __bfloat162float(smem_P[r * 128 + c]);
            float ds = p * (dp - smem_D[c]);
            smem_dS[r * 128 + c] = __float2bfloat16(ds);
        }
        __syncthreads();
        
        uint32_t idesc_dV = make_instr_desc_fn_maj(128, 128, 0, 1);
        uint64_t desc_P = make_smem_desc_sm100_fn(smem_P, 1024, 1024);
        
        if (threadIdx.x == 0) {
            for(int k=0; k<4; ++k) {
                uint64_t da = desc_P + k * 2;
                uint64_t db = desc_dO_0 + k * 128;
                umma_f16_cg1_fn(tmem_dV, da, db, idesc_dV, 1);
            }
            for(int k=0; k<4; ++k) {
                uint64_t da = desc_P + k * 2;
                uint64_t db = desc_dO_1 + k * 128;
                umma_f16_cg1_fn(tmem_dV, da, db, idesc_dV, 1);
            }
            umma_commit_1sm_fn(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        uint32_t idesc_dK = make_instr_desc_fn_maj(128, 128, 0, 1);
        uint64_t desc_dS = make_smem_desc_sm100_fn(smem_dS, 1024, 1024);
        
        if (threadIdx.x == 0) {
            for(int k=0; k<4; ++k) {
                uint64_t da = desc_dS + k * 2;
                uint64_t db = desc_Q_0 + k * 128;
                umma_f16_cg1_fn(tmem_dK, da, db, idesc_dK, 1);
            }
            for(int k=0; k<4; ++k) {
                uint64_t da = desc_dS + k * 2;
                uint64_t db = desc_Q_1 + k * 128;
                umma_f16_cg1_fn(tmem_dK, da, db, idesc_dK, 1);
            }
            umma_commit_1sm_fn(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        uint32_t idesc_dQ = make_instr_desc_fn_maj(128, 128, 1, 1);
        
        if (threadIdx.x == 0) {
            for(int k=0; k<4; ++k) {
                uint64_t da = desc_dS + k * 128;
                uint64_t db = desc_K_0 + k * 128;
                umma_f16_cg1_fn(tmem_dQ, da, db, idesc_dQ, (k==0)?0:1);
            }
            for(int k=0; k<4; ++k) {
                uint64_t da = desc_dS + k * 128;
                uint64_t db = desc_K_1 + k * 128;
                umma_f16_cg1_fn(tmem_dQ, da, db, idesc_dQ, 1);
            }
            umma_commit_1sm_fn(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t taddr = col;
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(taddr));
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
            uint32_t g_row = q_outer * 128 + row;
            
            if (g_row < S) {
                if (g_col0 < 128) atomicAdd(&global_dQ[g_row * 128 + g_col0], __float2bfloat16(f0));
                if (g_col1 < 128) atomicAdd(&global_dQ[g_row * 128 + g_col1], __float2bfloat16(f1));
                if (g_col2 < 128) atomicAdd(&global_dQ[g_row * 128 + g_col2], __float2bfloat16(f2));
                if (g_col3 < 128) atomicAdd(&global_dQ[g_row * 128 + g_col3], __float2bfloat16(f3));
            }
        }
    }
    
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t taddr_dV = 128 + col;
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(taddr_dV));
        
        uint32_t taddr_dK = 384 + col;
        uint32_t r4, r5, r6, r7;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(taddr_dK));
        
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t row = threadIdx.x;
        uint32_t g_col0 = col;
        uint32_t g_col1 = col + 1;
        uint32_t g_col2 = col + 2;
        uint32_t g_col3 = col + 3;
        uint32_t g_row = row;
        
        if (g_row < 128) {
            if (g_col0 < 128) global_dV[g_row * 128 + g_col0] = __float2bfloat16(__uint_as_float(r0));
            if (g_col1 < 128) global_dV[g_row * 128 + g_col1] = __float2bfloat16(__uint_as_float(r1));
            if (g_col2 < 128) global_dV[g_row * 128 + g_col2] = __float2bfloat16(__uint_as_float(r2));
            if (g_col3 < 128) global_dV[g_row * 128 + g_col3] = __float2bfloat16(__uint_as_float(r3));
            
            if (g_col0 < 128) global_dK[g_row * 128 + g_col0] = __float2bfloat16(__uint_as_float(r4));
            if (g_col1 < 128) global_dK[g_row * 128 + g_col1] = __float2bfloat16(__uint_as_float(r5));
            if (g_col2 < 128) global_dK[g_row * 128 + g_col2] = __float2bfloat16(__uint_as_float(r6));
            if (g_col3 < 128) global_dK[g_row * 128 + g_col3] = __float2bfloat16(__uint_as_float(r7));
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
    
    uint32_t num_kv_tiles = (S + 127) / 128;
    dim3 grid(num_kv_tiles, B * H);
    dim3 block(128);
    
    uint32_t smem_size = 196608 + 512 + 8 + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(
        bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ));
    
    bwd_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO,
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        S, d, B, H
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_attention_bwd::run);

}  // namespace tvm_ffi_attention_bwd