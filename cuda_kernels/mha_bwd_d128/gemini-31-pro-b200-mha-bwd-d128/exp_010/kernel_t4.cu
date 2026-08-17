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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n",                  \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

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

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
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

__device__ __forceinline__ void tma_load_2d_cta_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_none_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1
    d |= (uint64_t)0 << 61;   // layout_type = 0 (SWIZZLE_NONE)
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, int trans_A, int trans_B) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (trans_A << 15);
    d |= (trans_B << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint64_t advance_desc(uint64_t desc, uint32_t offset_bytes) {
    uint32_t current_addr = (desc & 0x3FFF) << 4;
    uint32_t new_addr = current_addr + offset_bytes;
    desc = (desc & ~0x3FFFull) | ((new_addr >> 4) & 0x3FFF);
    return desc;
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

__device__ __forceinline__ void matmul_128x128x128(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, 
    uint32_t idesc, uint32_t accum_init, 
    uint32_t step_A, uint32_t step_B) {
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        uint32_t accum = (k == 0) ? accum_init : 1;
        umma_f16_cg1_fn(tmem_c, desc_a, desc_b, idesc, accum);
        desc_a = advance_desc(desc_a, step_A);
        desc_b = advance_desc(desc_b, step_B);
    }
}

__device__ __forceinline__ void tcgen05_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(a));
}

struct SmemLayout {
    __nv_bfloat16 K[128 * 128];      // 0
    __nv_bfloat16 V[128 * 128];      // 32768
    __nv_bfloat16 Q[128 * 128];      // 65536
    __nv_bfloat16 dO[128 * 128];     // 98304
    union {
        __nv_bfloat16 O[128 * 128];      // 131072
        __nv_bfloat16 dS_T[128 * 128];   // 131072
    };
    __nv_bfloat16 P_T[128 * 128];    // 163840
    float L[128];                    // 196608
    float D[128];                    // 197120
    uint64_t mbar[4];                // 197632
    uint32_t tmem_base;              // 197664
};

__device__ __forceinline__ void load_L_j(float* L_global, float* L_smem) {
    L_smem[threadIdx.x] = L_global[threadIdx.x];
}

__device__ __forceinline__ void compute_D_j(SmemLayout* smem) {
    uint32_t row = threadIdx.x; 
    float sum = 0.0f;
    for (uint32_t col = 0; col < 128; col += 2) {
        __nv_bfloat162 o2 = *(__nv_bfloat162*)&smem->O[row * 128 + col];
        __nv_bfloat162 do2 = *(__nv_bfloat162*)&smem->dO[row * 128 + col];
        float2 fo2 = __bfloat1622float2(o2);
        float2 fdo2 = __bfloat1622float2(do2);
        sum += fo2.x * fdo2.x;
        sum += fo2.y * fdo2.y;
    }
    smem->D[row] = sum;
    __syncthreads();
}

__device__ __forceinline__ void compute_P_dS_and_store(
    SmemLayout* smem, 
    uint32_t S_tmem_col_base, uint32_t dP_tmem_col_base) {

    uint32_t row = threadIdx.x;
    constexpr float scale = 0.08838834764831843f; // 1/sqrt(128)
    constexpr float log2e = 1.4426950408889634f;

    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t s0, s1, s2, s3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(s0),"=r"(s1),"=r"(s2),"=r"(s3) : "r"(S_tmem_col_base + col));
        
        uint32_t dp0, dp1, dp2, dp3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(dp0),"=r"(dp1),"=r"(dp2),"=r"(dp3) : "r"(dP_tmem_col_base + col));
        
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

        float fs0 = __uint_as_float(s0) * scale;
        float fs1 = __uint_as_float(s1) * scale;
        float fs2 = __uint_as_float(s2) * scale;
        float fs3 = __uint_as_float(s3) * scale;

        float l0 = smem->L[col];
        float l1 = smem->L[col + 1];
        float l2 = smem->L[col + 2];
        float l3 = smem->L[col + 3];

        float p0 = fast_exp2f_fn((fs0 - l0) * log2e);
        float p1 = fast_exp2f_fn((fs1 - l1) * log2e);
        float p2 = fast_exp2f_fn((fs2 - l2) * log2e);
        float p3 = fast_exp2f_fn((fs3 - l3) * log2e);

        float d0 = smem->D[col];
        float d1 = smem->D[col + 1];
        float d2 = smem->D[col + 2];
        float d3 = smem->D[col + 3];

        float fdp0 = __uint_as_float(dp0);
        float fdp1 = __uint_as_float(dp1);
        float fdp2 = __uint_as_float(dp2);
        float fdp3 = __uint_as_float(dp3);

        float ds0 = p0 * (fdp0 - d0) * scale;
        float ds1 = p1 * (fdp1 - d1) * scale;
        float ds2 = p2 * (fdp2 - d2) * scale;
        float ds3 = p3 * (fdp3 - d3) * scale;

        uint32_t base = row * 128 + col;
        smem->P_T[base + 0] = __float2bfloat16(p0);
        smem->P_T[base + 1] = __float2bfloat16(p1);
        smem->P_T[base + 2] = __float2bfloat16(p2);
        smem->P_T[base + 3] = __float2bfloat16(p3);

        smem->dS_T[base + 0] = __float2bfloat16(ds0);
        smem->dS_T[base + 1] = __float2bfloat16(ds1);
        smem->dS_T[base + 2] = __float2bfloat16(ds2);
        smem->dS_T[base + 3] = __float2bfloat16(ds3);
    }
    __syncthreads();
}

__device__ __forceinline__ void tmem_atomic_add_dQ_fn(
    __nv_bfloat16* dQ_global, __nv_bfloat16* smem_stage,
    uint32_t dQ_tmem_col_base) {
    
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(dQ_tmem_col_base + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * 128 + col;
        smem_stage[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_stage[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_stage[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_stage[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = 128 / 4; 
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        uint32_t col_start = lane_id * 4;
        __nv_bfloat162 src0 = *(__nv_bfloat162*)&smem_stage[row * 128 + col_start];
        __nv_bfloat162 src1 = *(__nv_bfloat162*)&smem_stage[row * 128 + col_start + 2];
        __nv_bfloat162* dst = (__nv_bfloat162*)&dQ_global[row * 128 + col_start];
        atomicAdd(&dst[0], src0);
        atomicAdd(&dst[1], src1);
    }
    __syncthreads();
}

__device__ __forceinline__ void tmem_store_to_global_fn(
    __nv_bfloat16* global_dst, __nv_bfloat16* smem_stage,
    uint32_t tmem_col_base) {
    
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_col_base + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * 128 + col;
        smem_stage[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_stage[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_stage[base + 2] = __float2bfloat16(__float_as_uint(__uint_as_float(r2)));
        smem_stage[base + 3] = __float2bfloat16(__float_as_uint(__uint_as_float(r3)));
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = 128 / 4; 
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        uint32_t col_start = lane_id * 4;
        uint2 data = *reinterpret_cast<uint2*>(&smem_stage[row * 128 + col_start]);
        *reinterpret_cast<uint2*>(global_dst + row * 128 + col_start) = data;
    }
    __syncthreads();
}

__global__ void mha_bwd_d128_kernel(
    const __grid_constant__ CUtensorMap tma_Q, const __grid_constant__ CUtensorMap tma_K, const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O, const __grid_constant__ CUtensorMap tma_dO,
    float* L, __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int S) {
    
    extern __shared__ __align__(128) uint8_t smem_buf[];
    SmemLayout* smem = (SmemLayout*)smem_buf;
    
    int b = blockIdx.z;
    int h = blockIdx.y;
    int kv_block_idx = blockIdx.x; 
    
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&smem->tmem_base, 512);
    }
    __syncthreads();
    uint32_t tmem_base = smem->tmem_base;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&smem->mbar[0], 128); 
        init_smem_barrier_fn(&smem->mbar[2], 1);   
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    uint32_t phase_tma = 0;
    uint32_t phase_mma = 0;
    
    uint64_t desc_K_K   = make_smem_desc_sm100_none_fn(smem->K, 2048, 128);
    uint64_t desc_K_MN  = make_smem_desc_sm100_none_fn(smem->K, 128, 2048);
    
    uint64_t desc_Q_K   = make_smem_desc_sm100_none_fn(smem->Q, 2048, 128);
    uint64_t desc_Q_MN  = make_smem_desc_sm100_none_fn(smem->Q, 128, 2048);
    
    uint64_t desc_V_K   = make_smem_desc_sm100_none_fn(smem->V, 2048, 128);
    
    uint64_t desc_dO_K  = make_smem_desc_sm100_none_fn(smem->dO, 2048, 128);
    uint64_t desc_dO_MN = make_smem_desc_sm100_none_fn(smem->dO, 128, 2048);
    
    uint64_t desc_PT_K  = make_smem_desc_sm100_none_fn(smem->P_T, 2048, 128);
    
    uint64_t desc_dST_K = make_smem_desc_sm100_none_fn(smem->dS_T, 2048, 128);
    uint64_t desc_dST_MN= make_smem_desc_sm100_none_fn(smem->dS_T, 128, 2048);
    
    uint32_t idesc_NN = make_instr_desc_fn(128, 128, 0, 0); 
    uint32_t idesc_NT = make_instr_desc_fn(128, 128, 0, 1); 
    uint32_t idesc_TT = make_instr_desc_fn(128, 128, 1, 1);

    int seq_offset = (b * gridDim.y + h) * S;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&smem->mbar[0], 128 * 128 * 2 * 2);
        tma_load_2d_cta_fn(&tma_K, &smem->mbar[0], smem->K, 0, seq_offset + kv_block_idx * 128);
        tma_load_2d_cta_fn(&tma_V, &smem->mbar[0], smem->V, 0, seq_offset + kv_block_idx * 128);
    } else {
        mbarrier_arrive_fn(&smem->mbar[0]);
    }
    mbarrier_wait_fn(&smem->mbar[0], phase_tma);
    phase_tma ^= 1;
    __syncthreads();
    
    int num_q_blocks = S / 128;
    
    for (int j = 0; j < num_q_blocks; ++j) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem->mbar[0], 128 * 128 * 2 * 3);
            tma_load_2d_cta_fn(&tma_Q, &smem->mbar[0], smem->Q, 0, seq_offset + j * 128);
            tma_load_2d_cta_fn(&tma_dO, &smem->mbar[0], smem->dO, 0, seq_offset + j * 128);
            tma_load_2d_cta_fn(&tma_O, &smem->mbar[0], smem->O, 0, seq_offset + j * 128);
        } else {
            mbarrier_arrive_fn(&smem->mbar[0]);
        }
        
        float* L_global = L + seq_offset + (j * 128);
        load_L_j(L_global, smem->L);
        
        mbarrier_wait_fn(&smem->mbar[0], phase_tma);
        phase_tma ^= 1;
        fence_proxy_async_fn();
        __syncthreads();
        
        compute_D_j(smem);
        
        if (threadIdx.x == 0) {
            // S^T = K @ Q^T
            matmul_128x128x128(tmem_base + 0, desc_K_K, desc_Q_K, idesc_NN, 0, 32, 32);
            // dP^T = V @ dO^T
            matmul_128x128x128(tmem_base + 256, desc_V_K, desc_dO_K, idesc_NN, 0, 32, 32);
            tcgen05_commit_cg1_fn(&smem->mbar[2]);
        }
        mbarrier_wait_fn(&smem->mbar[2], phase_mma);
        phase_mma ^= 1;
        
        compute_P_dS_and_store(smem, tmem_base + 0, tmem_base + 256);
        
        if (threadIdx.x == 0) {
            uint32_t accum_init = (j == 0) ? 0 : 1;
            // dV = P^T @ dO
            matmul_128x128x128(tmem_base + 128, desc_PT_K, desc_dO_MN, idesc_NT, accum_init, 32, 4096);
            // dK = dS^T @ Q
            matmul_128x128x128(tmem_base + 384, desc_dST_K, desc_Q_MN, idesc_NT, accum_init, 32, 4096);
            // dQ = dS @ K
            matmul_128x128x128(tmem_base + 256, desc_dST_MN, desc_K_MN, idesc_TT, 0, 4096, 4096);
            tcgen05_commit_cg1_fn(&smem->mbar[2]);
        }
        mbarrier_wait_fn(&smem->mbar[2], phase_mma);
        phase_mma ^= 1;
        
        __nv_bfloat16* dQ_global = dQ + (b * gridDim.y * S * 128) + (h * S * 128) + (j * 128 * 128);
        tmem_atomic_add_dQ_fn(dQ_global, smem->P_T, tmem_base + 256);
    }
    
    __nv_bfloat16* dK_global = dK + (b * gridDim.y * S * 128) + (h * S * 128) + (kv_block_idx * 128 * 128);
    __nv_bfloat16* dV_global = dV + (b * gridDim.y * S * 128) + (h * S * 128) + (kv_block_idx * 128 * 128);
    
    tmem_store_to_global_fn(dK_global, smem->P_T, tmem_base + 384);
    tmem_store_to_global_fn(dV_global, smem->P_T, tmem_base + 128);
    
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 512);
    }
}

CUresult create_tma_2d_descriptor_2B(
    CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim,
    uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    
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
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);
    
    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    __nv_bfloat16* do_ptr = static_cast<__nv_bfloat16*>(dO.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    uint64_t gmem_outer = B * H * S;
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, q_ptr, d, gmem_outer, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, k_ptr, d, gmem_outer, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, v_ptr, d, gmem_outer, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_O, o_ptr, d, gmem_outer, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_dO, do_ptr, d, gmem_outer, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, B * H * S * d * sizeof(__nv_bfloat16), stream));
    
    dim3 grid(S / 128, H, B);
    dim3 block(128);
    int smem_size = sizeof(SmemLayout);
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_d128_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    mha_bwd_d128_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO,
        static_cast<float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S);
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda