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

// Helper functions for SM100
__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
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

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint64_t build_desc_swizzle128(void* ptr, bool is_A, bool is_row_major, uint32_t inner_dim) {
    uint32_t lbo, sbo;
    if (is_A) {
        if (is_row_major) { // K-Major
            lbo = 1;
            sbo = 1024;
        } else { // MN-Major
            lbo = (inner_dim / 8) * 1024;
            sbo = 1024;
        }
    } else { // is B
        if (is_row_major) { // MN-Major
            lbo = (inner_dim / 8) * 1024;
            sbo = 1024;
        } else { // K-Major
            lbo = 1;
            sbo = 1024;
        }
    }
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1
    
    // Base offset for SWIZZLE_128B
    uint64_t base_offset = (addr >> 7) & 0x7;
    d |= (base_offset << 49);
    
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t advance_desc_swizzle128(uint64_t desc, bool is_A, bool is_row_major, uint32_t inner_dim, uint32_t step_K) {
    uint32_t addr = (desc & 0x3FFFF) << 4;
    if (is_A) {
        if (is_row_major) {
            addr += step_K * 2;
        } else {
            addr += step_K * inner_dim * 2;
        }
    } else { // is B
        if (is_row_major) {
            addr += step_K * inner_dim * 2;
        } else {
            addr += step_K * 2;
        }
    }
    desc &= ~0x3FFFFull;
    desc |= (addr >> 4);
    
    uint64_t base_offset = (addr >> 7) & 0x7;
    desc &= ~(0x7ull << 49);
    desc |= (base_offset << 49);
    return desc;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim,
        globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, l2Promotion, oobFill
    );
}

__global__ void precompute_D_kernel(const __nv_bfloat16* dO, const __nv_bfloat16* O, float* D, size_t total_S, int d) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_S) {
        float sum = 0;
        for (int i = 0; i < d; ++i) {
            float do_val = __bfloat162float(dO[idx * d + i]);
            float o_val  = __bfloat162float(O[idx * d + i]);
            sum += do_val * o_val;
        }
        D[idx] = sum;
    }
}

__global__ void fa_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L_ptr, const float* D_ptr,
    __nv_bfloat16* global_dQ, __nv_bfloat16* global_dK, __nv_bfloat16* global_dV,
    int S, int d, float scale)
{
    extern __shared__ char raw_smem[];
    // Align smem base on 1024-byte boundary for SWIZZLE_128B
    char* smem = (char*)(((size_t)raw_smem + 1023) & ~1023);

    __nv_bfloat16* smem_K   = (__nv_bfloat16*)smem;                 // 16KB
    __nv_bfloat16* smem_V   = smem_K + 64 * 128;                    // 16KB
    __nv_bfloat16* smem_Q   = smem_V + 64 * 128;                    // 16KB
    __nv_bfloat16* smem_dO  = smem_Q + 64 * 128;                    // 16KB
    
    // Non-swizzled out buffer just for threads reading and atomicAdds 
    __nv_bfloat16* smem_out = smem_dO + 64 * 128;                   // 16KB
    
    // In-smem intermediates explicitly swizzled across threads
    __nv_bfloat16* smem_PT  = smem_out + 64 * 128;                  // 8KB 
    __nv_bfloat16* smem_dS  = smem_PT + 64 * 64;                    // 8KB 
    __nv_bfloat16* smem_dST = smem_dS + 64 * 64;                    // 8KB 
    
    float* smem_L           = (float*)(smem_dST + 64 * 64);         // 256B
    float* smem_D           = smem_L + 64;                          // 256B
    uint64_t* mbar_tma      = (uint64_t*)(smem_D + 64);             // 8B
    uint64_t* mbar_mma      = mbar_tma + 1;                         // 8B
    uint32_t* shared_tmem   = (uint32_t*)(mbar_mma + 1);            // 4B

    int kv_idx = blockIdx.x;
    int bh = blockIdx.y;
    int row = threadIdx.x;

    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(shared_tmem, 512);
    }
    __syncthreads();
    
    uint32_t tmem_base = *shared_tmem;
    uint32_t TMEM_dQ = tmem_base;         
    uint32_t TMEM_S  = tmem_base + 128;   
    uint32_t TMEM_dP = tmem_base + 192;   
    uint32_t TMEM_dK = tmem_base + 256;   
    uint32_t TMEM_dV = tmem_base + 384;   

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_tma, 1);
        init_smem_barrier_fn(mbar_mma, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    uint32_t phase_tma = 0;
    uint32_t phase_mma = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_tma, 2 * 16384);
        tma_load_2d_fn(&tma_K, mbar_tma, smem_K, 0, bh * S + kv_idx * 64);
        tma_load_2d_fn(&tma_V, mbar_tma, smem_V, 0, bh * S + kv_idx * 64);
    }
    mbarrier_wait_fn(mbar_tma, phase_tma);
    phase_tma ^= 1;

    uint32_t idesc_S  = make_instr_desc_fn(64, 64)   | (0u << 15) | (0u << 16);
    uint32_t idesc_dP = make_instr_desc_fn(64, 64)   | (0u << 15) | (0u << 16);
    uint32_t idesc_dV = make_instr_desc_fn(64, 128)  | (0u << 15) | (1u << 16);
    uint32_t idesc_dK = make_instr_desc_fn(64, 128)  | (0u << 15) | (1u << 16);
    uint32_t idesc_dQ = make_instr_desc_fn(64, 128)  | (0u << 15) | (1u << 16);

    uint64_t desc_K = 0, desc_V = 0, desc_K_dQ = 0;
    if (threadIdx.x == 0) {
        desc_K = build_desc_swizzle128(smem_K, false, false, 128);
        desc_V = build_desc_swizzle128(smem_V, false, false, 128);
        desc_K_dQ = build_desc_swizzle128(smem_K, false, true, 128);
    }

    int max_q_idx = (S + 63) / 64 - 1;
    for (int q_idx = kv_idx; q_idx <= max_q_idx; ++q_idx) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_tma, 2 * 16384);
            tma_load_2d_fn(&tma_Q, mbar_tma, smem_Q, 0, bh * S + q_idx * 64);
            tma_load_2d_fn(&tma_dO, mbar_tma, smem_dO, 0, bh * S + q_idx * 64);
        }
        if (threadIdx.x < 64) {
            int q_global = q_idx * 64 + threadIdx.x;
            if (q_global < S) {
                smem_L[threadIdx.x] = L_ptr[(size_t)bh * S + q_global];
                smem_D[threadIdx.x] = D_ptr[(size_t)bh * S + q_global];
            } else {
                smem_L[threadIdx.x] = 0.0f;
                smem_D[threadIdx.x] = 0.0f;
            }
        }
        __syncthreads();
        mbarrier_wait_fn(mbar_tma, phase_tma);
        phase_tma ^= 1;

        if (threadIdx.x == 0) {
            uint64_t cur_desc_Q = build_desc_swizzle128(smem_Q, true, true, 128);
            uint64_t cur_desc_dO = build_desc_swizzle128(smem_dO, true, true, 128);
            uint64_t cur_desc_K = desc_K;
            uint64_t cur_desc_V = desc_V;

            for (int k = 0; k < 128; k += 16) {
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(TMEM_S, cur_desc_Q, cur_desc_K, idesc_S, accum);
                cur_desc_Q = advance_desc_swizzle128(cur_desc_Q, true, true, 128, 16);
                cur_desc_K = advance_desc_swizzle128(cur_desc_K, false, false, 128, 16);
            }
            for (int k = 0; k < 128; k += 16) {
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(TMEM_dP, cur_desc_dO, cur_desc_V, idesc_dP, accum);
                cur_desc_dO = advance_desc_swizzle128(cur_desc_dO, true, true, 128, 16);
                cur_desc_V = advance_desc_swizzle128(cur_desc_V, false, false, 128, 16);
            }
            umma_commit_cg1_fn(mbar_mma);
        }
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;

        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t sr[4], dpr[4];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(sr[0]),"=r"(sr[1]),"=r"(sr[2]),"=r"(sr[3]) : "r"(TMEM_S + col));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(dpr[0]),"=r"(dpr[1]),"=r"(dpr[2]),"=r"(dpr[3]) : "r"(TMEM_dP + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            if (row < 64) {
                float lse = smem_L[row];
                float d_val = smem_D[row];
                for (int i = 0; i < 4; ++i) {
                    float s = __uint_as_float(sr[i]) * scale;
                    float dp = __uint_as_float(dpr[i]);
                    float p, ds;
                    int q_global = q_idx * 64 + row;
                    int k_global = kv_idx * 64 + col + i;
                    
                    if (q_global < k_global || q_global >= S || k_global >= S) {
                        p = 0.0f;
                        ds = 0.0f;
                    } else {
                        p = fast_exp2f_fn((s - lse) * 1.4426950408889634f);
                        ds = p * (dp - d_val) * scale;
                    }
                    
                    __nv_bfloat16 p_bf16 = __float2bfloat16(p);
                    __nv_bfloat16 ds_bf16 = __float2bfloat16(ds);
                    
                    int c_idx = col + i;

                    int pt_chunk_idx = row / 8;
                    int pt_swizzled_chunk = (pt_chunk_idx & ~7) | ((pt_chunk_idx & 7) ^ (c_idx % 8));
                    int pt_swizzled_c = pt_swizzled_chunk * 8 + (row % 8);
                    smem_PT[c_idx * 64 + pt_swizzled_c] = p_bf16;

                    int ds_chunk_idx = c_idx / 8;
                    int ds_swizzled_chunk = (ds_chunk_idx & ~7) | ((ds_chunk_idx & 7) ^ (row % 8));
                    int ds_swizzled_c = ds_swizzled_chunk * 8 + (c_idx % 8);
                    smem_dS[row * 64 + ds_swizzled_c] = ds_bf16;

                    int dst_chunk_idx = row / 8;
                    int dst_swizzled_chunk = (dst_chunk_idx & ~7) | ((dst_chunk_idx & 7) ^ (c_idx % 8));
                    int dst_swizzled_c = dst_swizzled_chunk * 8 + (row % 8);
                    smem_dST[c_idx * 64 + dst_swizzled_c] = ds_bf16;
                }
            }
        }
        __syncthreads();
        fence_async_shared_fn();

        if (threadIdx.x == 0) {
            uint64_t cur_desc_PT = build_desc_swizzle128(smem_PT, true, true, 64);
            uint64_t cur_desc_dO_dV = build_desc_swizzle128(smem_dO, false, true, 128);
            for (int k = 0; k < 64; k += 16) {
                uint32_t accum = (q_idx == kv_idx && k == 0) ? 0 : 1;
                umma_f16_cg1_fn(TMEM_dV, cur_desc_PT, cur_desc_dO_dV, idesc_dV, accum);
                cur_desc_PT = advance_desc_swizzle128(cur_desc_PT, true, true, 64, 16);
                cur_desc_dO_dV = advance_desc_swizzle128(cur_desc_dO_dV, false, true, 128, 16);
            }

            uint64_t cur_desc_dST = build_desc_swizzle128(smem_dST, true, true, 64);
            uint64_t cur_desc_Q_dK = build_desc_swizzle128(smem_Q, false, true, 128);
            for (int k = 0; k < 64; k += 16) {
                uint32_t accum = (q_idx == kv_idx && k == 0) ? 0 : 1;
                umma_f16_cg1_fn(TMEM_dK, cur_desc_dST, cur_desc_Q_dK, idesc_dK, accum);
                cur_desc_dST = advance_desc_swizzle128(cur_desc_dST, true, true, 64, 16);
                cur_desc_Q_dK = advance_desc_swizzle128(cur_desc_Q_dK, false, true, 128, 16);
            }

            uint64_t cur_desc_dS = build_desc_swizzle128(smem_dS, true, true, 64);
            uint64_t cur_desc_K_dQ_local = desc_K_dQ;
            for (int k = 0; k < 64; k += 16) {
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(TMEM_dQ, cur_desc_dS, cur_desc_K_dQ_local, idesc_dQ, accum);
                cur_desc_dS = advance_desc_swizzle128(cur_desc_dS, true, true, 64, 16);
                cur_desc_K_dQ_local = advance_desc_swizzle128(cur_desc_K_dQ_local, false, true, 128, 16);
            }

            umma_commit_cg1_fn(mbar_mma);
        }
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;

        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(TMEM_dQ + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            if (row < 64) {
                smem_out[row * 128 + c + 0] = __float2bfloat16(__uint_as_float(r0));
                smem_out[row * 128 + c + 1] = __float2bfloat16(__uint_as_float(r1));
                smem_out[row * 128 + c + 2] = __float2bfloat16(__uint_as_float(r2));
                smem_out[row * 128 + c + 3] = __float2bfloat16(__uint_as_float(r3));
            }
        }
        __syncthreads();

        for (int i = threadIdx.x; i < 64 * 128 / 2; i += 128) {
            int r = (i * 2) / 128;
            int c = (i * 2) % 128;
            int q_global = q_idx * 64 + r;
            if (q_global < S) {
                __nv_bfloat162 data = *(__nv_bfloat162*)&smem_out[r * 128 + c];
                atomicAdd((__nv_bfloat162*)&global_dQ[( (size_t)bh * S + q_global ) * 128 + c], data);
            }
        }
        __syncthreads();
    }

    for (int c = 0; c < 128; c += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(TMEM_dK + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        if (row < 64) {
            smem_out[row * 128 + c + 0] = __float2bfloat16(__uint_as_float(r0));
            smem_out[row * 128 + c + 1] = __float2bfloat16(__uint_as_float(r1));
            smem_out[row * 128 + c + 2] = __float2bfloat16(__uint_as_float(r2));
            smem_out[row * 128 + c + 3] = __float2bfloat16(__uint_as_float(r3));
        }
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < 64 * 128 / 8; i += 128) {
        int r = (i * 8) / 128;
        int c = (i * 8) % 128;
        int k_global = kv_idx * 64 + r;
        if (k_global < S) {
            *(float4*)&global_dK[( (size_t)bh * S + k_global ) * 128 + c] = *(float4*)&smem_out[r * 128 + c];
        }
    }
    __syncthreads();

    for (int c = 0; c < 128; c += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(TMEM_dV + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        if (row < 64) {
            smem_out[row * 128 + c + 0] = __float2bfloat16(__uint_as_float(r0));
            smem_out[row * 128 + c + 1] = __float2bfloat16(__uint_as_float(r1));
            smem_out[row * 128 + c + 2] = __float2bfloat16(__uint_as_float(r2));
            smem_out[row * 128 + c + 3] = __float2bfloat16(__uint_as_float(r3));
        }
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < 64 * 128 / 8; i += 128) {
        int r = (i * 8) / 128;
        int c = (i * 8) % 128;
        int k_global = kv_idx * 64 + r;
        if (k_global < S) {
            *(float4*)&global_dV[( (size_t)bh * S + k_global ) * 128 + c] = *(float4*)&smem_out[r * 128 + c];
        }
    }

    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(*shared_tmem, 512);
    }
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);

    float* D_ptr;
    CUDA_CHECK(cudaMallocAsync(&D_ptr, B * H * S * sizeof(float), stream));
    
    int threads_pre = 256;
    int blocks_pre = (B * H * S + threads_pre - 1) / threads_pre;
    precompute_D_kernel<<<blocks_pre, threads_pre, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        D_ptr, B * H * S, d);
    
    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, B * H * S * d * sizeof(__nv_bfloat16), stream));

    CUtensorMap tma_Q, tma_K, tma_V, tma_dO;
    CUresult res;
    res = create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, B*H*S, 128, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA Q fail\n"); exit(1); }
    res = create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, B*H*S, 128, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA K fail\n"); exit(1); }
    res = create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, B*H*S, 128, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA V fail\n"); exit(1); }
    res = create_tma_2d_descriptor_2B(&tma_dO, dO.data_ptr(), 128, B*H*S, 128, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA dO fail\n"); exit(1); }

    int grid_x = (S + 63) / 64;
    int grid_y = B * H;
    dim3 grid(grid_x, grid_y);
    dim3 block(128);
    int smem_size = 104 * 1024 + 1024;

    CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<const void*>(fa_bwd_kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));

    float scale = 1.0f / sqrtf((float)d);

    fa_bwd_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO,
        static_cast<const float*>(L.data_ptr()),
        D_ptr,
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S, d, scale);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(D_ptr, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda