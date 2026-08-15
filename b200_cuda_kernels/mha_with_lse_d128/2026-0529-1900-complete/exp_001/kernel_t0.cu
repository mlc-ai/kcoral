#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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

// Helper device functions
__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF;
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
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

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3, uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),"=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                 :: "r"(r0),"r"(r1),"r"(r2),"r"(r3), "r"(col));
}

__device__ __forceinline__ void tmem_store_8x_fn(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3, uint32_t r4, uint32_t r5, uint32_t r6, uint32_t r7) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%8], {%0,%1,%2,%3,%4,%5,%6,%7};"
                 :: "r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(r4),"r"(r5),"r"(r6),"r"(r7), "r"(col));
}

__device__ __forceinline__ void tcgen05_wait_st_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void umma_f16_cg2_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_f16_cg2_tmem_A_fn(uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t swizzle_mode) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)swizzle_mode << 61;
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

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)), "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    uint32_t tmem_base_col, float scale,
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_start, uint32_t n_start,
    uint32_t BM, uint32_t BN) {
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_base_col + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * BN + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0) * scale);
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1) * scale);
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2) * scale);
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3) * scale);
    }
    __syncthreads();
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_start + row;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = n_start + col_start;
        if (global_row < M && global_col + 3 < N) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
            *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
        }
    }
}

// Memory block limits and configurations
__shared__ alignas(1024) uint8_t smem_Q[32768];
__shared__ alignas(1024) uint8_t smem_K[2][16384];
__shared__ alignas(1024) uint8_t smem_V[2][16384];

__shared__ alignas(8) uint64_t mbar_Q[1];
__shared__ alignas(8) uint64_t mbar_K[2][1];
__shared__ alignas(8) uint64_t mbar_V[2][1];
__shared__ alignas(8) uint64_t mbar_umma[1];
__shared__ uint32_t smem_tmem_O, smem_tmem_S, smem_tmem_P;

__global__ void mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O_ptr,
    float* __restrict__ LSE_ptr,
    int S_TOTAL, int D
) {
    int b = blockIdx.y;
    int h = blockIdx.z;
    int cta_rank = cluster_rank_fn();
    int m_start = blockIdx.x * 256 + cta_rank * 128;
    int row_offset = (b * h + h) * S_TOTAL;
    
    // Setup and allocate Tensor Memory + Barriers
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K[0], 1);
        init_smem_barrier_fn(mbar_K[1], 1);
        init_smem_barrier_fn(mbar_V[0], 1);
        init_smem_barrier_fn(mbar_V[1], 1);
        init_smem_barrier_fn(mbar_umma, 1);
        fence_smem_barrier_init_fn();
    }
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&smem_tmem_O, 128);
        tmem_alloc_fn(&smem_tmem_S, 128);
        tmem_alloc_fn(&smem_tmem_P, 64);
    }
    __syncthreads();
    
    uint32_t tmem_O = smem_tmem_O;
    uint32_t tmem_S = smem_tmem_S;
    uint32_t tmem_P = smem_tmem_P;
    
    for (int col = 0; col < 128; col += 4) {
        tmem_store_4x_fn(tmem_O + col, 0, 0, 0, 0);
    }
    tcgen05_wait_st_fn();
    
    bool is_valid = (m_start < S_TOTAL);
    if (is_valid) {
        if (threadIdx.x == 0) mbarrier_arrive_and_expect_tx_fn(mbar_Q, 128 * 128 * 2);
        __syncthreads();
        tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q, 0, row_offset + m_start);
        if (threadIdx.x == 0) mbarrier_wait_fn(mbar_Q, 0);
    }
    __syncthreads();
    
    uint64_t desc_Q0 = make_smem_desc_sm100_fn(smem_Q, 1, 2048, 2);
    uint64_t desc_Q1 = make_smem_desc_sm100_fn(smem_Q + 128, 1, 2048, 2) | (1ULL << 49);
    
    int S_CHUNKS = (S_TOTAL + 127) / 128;
    float scale_S = 1.0f / sqrtf((float)D);
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    uint32_t phase_umma = 0;
    
    for (int s_idx = 0; s_idx < S_CHUNKS; ++s_idx) {
        int s_chunk = s_idx * 128;
        int buf = s_idx % 2;
        int phase_kv = s_idx / 2;
        
        if (cta_rank == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K[buf], 16384 * 2);
            mbarrier_arrive_and_expect_tx_fn(mbar_V[buf], 16384 * 2);
        }
        cluster_sync_fn();
        
        tma_load_2d_cg2_fn(&tma_K, mbar_K[buf], smem_K[buf], 0, row_offset + s_chunk + cta_rank * 64);
        tma_load_2d_cg2_fn(&tma_V, mbar_V[buf], smem_V[buf], cta_rank * 64, row_offset + s_chunk);
        
        if (cta_rank == 0) {
            mbarrier_wait_fn(mbar_K[buf], phase_kv);
            mbarrier_wait_fn(mbar_V[buf], phase_kv);
            tcgen05_fence_after_fn();
            
            uint64_t desc_K0 = make_smem_desc_sm100_fn(smem_K[buf], 1, 2048, 2);
            uint64_t desc_K1 = make_smem_desc_sm100_fn(smem_K[buf] + 128, 1, 2048, 2) | (1ULL << 49);
            
            uint32_t idesc_qk = make_instr_desc_fn(256, 128);
            umma_f16_cg2_fn(tmem_S, desc_Q0, desc_K0, idesc_qk, 0);
            umma_f16_cg2_fn(tmem_S, desc_Q1, desc_K1, idesc_qk, 1);
            umma_commit_2sm_fn(mbar_umma);
        }
        
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
        
        float row_max = -INFINITY;
        float row_S[128];
        for (int col = 0; col < 128; col += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_S + col, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            float* fr = (float*)r;
            for (int i = 0; i < 8; ++i) {
                int k_idx = s_chunk + col + i;
                float val = (k_idx < S_TOTAL) ? fr[i] * scale_S : -INFINITY;
                row_S[col + i] = val;
                if (val > row_max) row_max = val;
            }
        }
        
        float m_new = fmaxf(m_prev, row_max);
        float exp_diff = 1.0f;
        if (m_prev > -INFINITY && m_new > m_prev) {
            exp_diff = expf(m_prev - m_new);
            for (int col = 0; col < 128; col += 8) {
                uint32_t r[8];
                tmem_load_8x_fn(tmem_O + col, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
                tmem_load_fence_fn();
                float* fr = (float*)r;
                for (int i = 0; i < 8; ++i) fr[i] *= exp_diff;
                tmem_store_8x_fn(tmem_O + col, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
            }
        }
        tcgen05_wait_st_fn();
        
        float row_sum = 0.0f;
        for (int col = 0; col < 64; col += 4) {
            uint32_t p[4];
            for (int j = 0; j < 4; ++j) {
                int i = col * 2 + j * 2;
                float val0 = expf(row_S[i] - m_new);
                float val1 = expf(row_S[i+1] - m_new);
                row_sum += val0 + val1;
                p[j] = pack_bf16_fn(*(uint32_t*)&val0, *(uint32_t*)&val1);
            }
            tmem_store_4x_fn(tmem_P + col, p[0], p[1], p[2], p[3]);
        }
        tcgen05_wait_st_fn();
        
        l_prev = l_prev * exp_diff + row_sum;
        m_prev = m_new;
        
        tcgen05_fence_before_fn();
        cluster_sync_fn();
        
        if (cta_rank == 0) {
            tcgen05_fence_after_fn();
            uint64_t desc_V0 = make_smem_desc_sm100_fn(smem_V[buf], 128, 1024, 0);
            uint64_t desc_V1 = make_smem_desc_sm100_fn(smem_V[buf] + 8192, 128, 1024, 0);
            
            uint32_t idesc_pv = make_instr_desc_fn(256, 128);
            umma_f16_cg2_tmem_A_fn(tmem_O, tmem_P, desc_V0, idesc_pv, 1);
            umma_f16_cg2_tmem_A_fn(tmem_O, tmem_P + 32, desc_V1, idesc_pv, 1);
            umma_commit_2sm_fn(mbar_umma);
        }
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
    }
    
    if (is_valid) {
        if (m_start + threadIdx.x < S_TOTAL) {
            float lse = m_prev + logf(l_prev);
            LSE_ptr[row_offset + m_start + threadIdx.x] = lse;
        }
        
        float inv_l = 1.0f / l_prev;
        __nv_bfloat16* O_base = O_ptr + row_offset * 128;
        tmem_epilogue_coalesced_4w_fn(tmem_O, inv_l, O_base, (__nv_bfloat16*)smem_Q, S_TOTAL, 128, m_start, 0, 128, 128);
    }
    
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(smem_tmem_O, 128);
        tmem_dealloc_fn(smem_tmem_S, 128);
        tmem_dealloc_fn(smem_tmem_P, 64);
    }
    __syncthreads();
}

namespace tvm_ffi_example_cuda {

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

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), D, B * H * S, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
    create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), D, B * H * S, 128, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
    create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);

    int num_m_blocks = (S + 255) / 256;
    dim3 grid(num_m_blocks, B, H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_fwd_kernel, tma_Q, tma_K, tma_V, 
                       static_cast<__nv_bfloat16*>(O.data_ptr()), 
                       static_cast<float*>(LSE.data_ptr()), 
                       S, D));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}