#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <algorithm>
#include <cmath>

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        const char* err_str;                                     \
        cuGetErrorString(_e, &err_str);                          \
        fprintf(stderr, "CUDA Driver error %s at %s:%d\n",       \
                err_str, __FILE__, __LINE__);                    \
        exit(1);                                                 \
    }                                                            \
} while(0)

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_kernel {

// ----------------------------------------------------------------------
// PTX Helper Functions
// ----------------------------------------------------------------------

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
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
        ".mbarrier::arrive::one.shared::cta.b64"
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

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
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

// ----------------------------------------------------------------------
// Attention Kernel
// ----------------------------------------------------------------------

__global__ __launch_bounds__(128) void mha_attention_wgmma_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    uint32_t output_S, uint32_t output_D,
    uint32_t H) 
{
    // ---------------------- Setup ----------------------
    extern __shared__ __align__(1024) char smem_pool[];
    uint64_t* mbar = (uint64_t*)smem_pool;
    
    // Ensure 1024-byte alignment for all matrices so SWIZZLE_128B operates correctly
    char* matrix_pool = (char*)(((uintptr_t)(smem_pool + 8) + 1023) & ~1023);
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)matrix_pool;
    __nv_bfloat16* smem_Q1 = smem_Q0 + 4096;
    __nv_bfloat16* smem_K0 = smem_Q1 + 4096;
    __nv_bfloat16* smem_K1 = smem_K0 + 4096;
    __nv_bfloat16* smem_V0 = smem_K1 + 4096;
    __nv_bfloat16* smem_V1 = smem_V0 + 4096;
    __nv_bfloat16* smem_QK = smem_V1 + 4096; // Staging buffer for QK values
    __nv_bfloat16* smem_P  = smem_QK + 4096; // P values explicitly written with 128B swizzle
    float* smem_O0 = (float*)(smem_P + 4096);
    float* smem_O1 = smem_O0 + 4096;

    uint32_t* tmem_QK = (uint32_t*)(smem_pool + 8);
    uint32_t* tmem_PV0 = tmem_QK + 1;
    uint32_t* tmem_PV1 = tmem_PV0 + 1;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        tmem_alloc_fn(tmem_QK, 64);
        tmem_alloc_fn(tmem_PV0, 64);
        tmem_alloc_fn(tmem_PV1, 64);
    }
    __syncthreads();

    uint32_t seq_idx = blockIdx.x;
    uint32_t bh_idx = blockIdx.y;
    uint32_t b = bh_idx / H;
    uint32_t h = bh_idx % H;
    uint32_t batch_head_offset = (b * H + h) * output_S;
    uint32_t my_m_base = seq_idx * 128 + cluster_rank_fn() * 64; 
    
    constexpr int BM = 64;
    constexpr int BN = 64;
    float qk_scale = 1.0f / sqrtf((float)output_D);
    
    uint32_t idesc_QK = make_instr_desc_fn(BM, BN);
    uint32_t idesc_PV = make_instr_desc_fn(BM, BN);
    idesc_PV |= (1u << 16); // Transpose V
    
    uint32_t phase = 0;
    int K_step_size = 4; // 4 iterations of K=16 equals K=64

    int num_n_tiles = (output_S + 63) / 64;

    // ---------------------- Load Q ----------------------
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 8192 * 2);
        tma_load_2d_fn(&tma_Q, mbar, smem_Q0, 0, batch_head_offset + my_m_base);
        tma_load_2d_fn(&tma_Q, mbar, smem_Q1, 64, batch_head_offset + my_m_base);
    }
    mbarrier_wait_fn(mbar, phase);
    phase ^= 1;

    // ---------------------- Online Softmax State ----------------------
    float global_max = -1e20f;
    float global_sum = 0.0f;

    if (tid < 64) {
        smem_O0[tid * 64] = 0.0f;
        smem_O1[tid * 64] = 0.0f;
    }

    // ---------------------- QK Pass ----------------------
    for (int n_tile = 0; n_tile < num_n_tiles; n_tile++) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 8192 * 4);
            tma_load_2d_fn(&tma_K, mbar, smem_K0, 0, batch_head_offset + n_tile * 64);
            tma_load_2d_fn(&tma_K, mbar, smem_K1, 64, batch_head_offset + n_tile * 64);
            tma_load_2d_fn(&tma_V, mbar, smem_V0, 0, batch_head_offset + n_tile * 64);
            tma_load_2d_fn(&tma_V, mbar, smem_V1, 64, batch_head_offset + n_tile * 64);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        for (int k_part = 0; k_part < 2; k_part++) {
            uint64_t desc_Q = make_smem_desc_sm100_fn(k_part == 0 ? smem_Q0 : smem_Q1, 1, 1024);
            uint64_t desc_K = make_smem_desc_sm100_fn(k_part == 0 ? smem_K0 : smem_K1, 1, 1024);
            
            for (int k_idx = 0; k_idx < K_step_size; k_idx++) {
                uint64_t desc_A = desc_Q + k_idx * 2; 
                uint64_t desc_B = desc_K + k_idx * 2; 
                uint32_t accum = (k_part == 0 && k_idx == 0) ? 0 : 1;
                uint32_t tmem_col = *tmem_QK; 
                umma_f16_cg1_fn(tmem_col, desc_A, desc_B, idesc_QK, accum);
            }
        }
        umma_commit_1sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        int tid = threadIdx.x;
        int row = (tid / 32) * 32 + (tid % 32);
        
        float local_max = -1e20f;
        for (int load_idx = 0; load_idx < 16; load_idx++) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(*tmem_QK + load_idx * 4, &r0, &r1, &r2, &r3);
            float v0 = __uint_as_float(r0) * qk_scale;
            float v1 = __uint_as_float(r1) * qk_scale;
            float v2 = __uint_as_float(r2) * qk_scale;
            float v3 = __uint_as_float(r3) * qk_scale;
            
            int col = load_idx * 4;
            if (row < 64) {
                smem_QK[row * 64 + col + 0] = __float2bfloat16(v0);
                smem_QK[row * 64 + col + 1] = __float2bfloat16(v1);
                smem_QK[row * 64 + col + 2] = __float2bfloat16(v2);
                smem_QK[row * 64 + col + 3] = __float2bfloat16(v3);
            }
            local_max = fmaxf(local_max, fmaxf(fmaxf(v0, v1), fmaxf(v2, v3)));
        }
        tmem_load_fence_fn();

        float m_new = fmaxf(global_max, local_max);
        float P_corr = expf(global_max - m_new);
        global_sum *= P_corr;

        if (row < 64) {
            for(int i=0; i<64; ++i) {
                smem_O0[row * 64 + i] *= P_corr;
                smem_O1[row * 64 + i] *= P_corr;
            }
        }

        float local_sum = 0.0f;
        for (int load_idx = 0; load_idx < 16; load_idx++) {
            if (row < 64) {
                int col = load_idx * 4;
                float v0 = __bfloat162float(smem_QK[row * 64 + col + 0]);
                float v1 = __bfloat162float(smem_QK[row * 64 + col + 1]);
                float v2 = __bfloat162float(smem_QK[row * 64 + col + 2]);
                float v3 = __bfloat162float(smem_QK[row * 64 + col + 3]);

                float p0 = expf(v0 - m_new);
                float p1 = expf(v1 - m_new);
                float p2 = expf(v2 - m_new);
                float p3 = expf(v3 - m_new);

                if (n_tile * 64 + col + 0 >= output_S) p0 = 0.0f;
                if (n_tile * 64 + col + 1 >= output_S) p1 = 0.0f;
                if (n_tile * 64 + col + 2 >= output_S) p2 = 0.0f;
                if (n_tile * 64 + col + 3 >= output_S) p3 = 0.0f;

                local_sum += p0 + p1 + p2 + p3;

                // Translate logical (row, col) to physical 128B swizzled offsets exactly matching TMEM formatting rules
                int chunk = col / 8;
                int swizzled_chunk = chunk ^ (row % 8);
                int phys_col = swizzled_chunk * 8 + (col % 8);
                
                smem_P[row * 64 + phys_col + 0] = __float2bfloat16(p0);
                smem_P[row * 64 + phys_col + 1] = __float2bfloat16(p1);
                smem_P[row * 64 + phys_col + 2] = __float2bfloat16(p2);
                smem_P[row * 64 + phys_col + 3] = __float2bfloat16(p3);
            }
        }

        global_sum += local_sum;
        global_max = m_new;
        
        fence_async_shared_fn();

        for (int k_part = 0; k_part < 2; k_part++) {
            uint64_t desc_P = make_smem_desc_sm100_fn(smem_P, 1, 1024);
            uint64_t desc_V = make_smem_desc_sm100_fn(k_part == 0 ? smem_V0 : smem_V1, 8192, 1024); // V is N-major!
            
            for (int k_idx = 0; k_idx < K_step_size; k_idx++) {
                uint64_t desc_A = desc_P + k_idx * 2;
                uint64_t desc_B = desc_V + k_idx * 128; // Jump 128 due to major orientation
                
                uint32_t accum = (k_part == 0 && k_idx == 0) ? 0 : 1;
                uint32_t tmem_col_PV = (k_part == 0) ? *tmem_PV0 : *tmem_PV1;
                
                umma_f16_cg1_fn(tmem_col_PV, desc_A, desc_B, idesc_PV, accum);
            }
        }
        umma_commit_1sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        for (int load_idx = 0; load_idx < 16; load_idx++) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            tmem_load_8x_fn(*tmem_PV0 + load_idx * 4, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
            tmem_load_8x_fn(*tmem_PV1 + load_idx * 4, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);

            if (row < 64) {
                int col = load_idx * 4;
                smem_O0[row * 64 + col + 0] += __uint_as_float(r0);
                smem_O0[row * 64 + col + 1] += __uint_as_float(r1);
                smem_O0[row * 64 + col + 2] += __uint_as_float(r2);
                smem_O0[row * 64 + col + 3] += __uint_as_float(r3);
                
                smem_O1[row * 64 + col + 0] += __uint_as_float(r4);
                smem_O1[row * 64 + col + 1] += __uint_as_float(r5);
                smem_O1[row * 64 + col + 2] += __uint_as_float(r6);
                smem_O1[row * 64 + col + 3] += __uint_as_float(r7);
            }
        }
        tmem_load_fence_fn();
    }

    // ---------------------- Epilogue ----------------------
    if (tid < 64) {
        int m_idx = my_m_base + tid;
        if (m_idx < output_S) {
            for(int d = 0; d < 64; d++) {
                O[((b * H + h) * output_S + m_idx) * output_D + d] = __float2bfloat16(smem_O0[tid * 64 + d] / global_sum);
                O[((b * H + h) * output_S + m_idx) * output_D + d + 64] = __float2bfloat16(smem_O1[tid * 64 + d] / global_sum);
            }
            
            LSE[(b * H + h) * output_S + m_idx] = global_max + logf(global_sum);
        }
    }

    // ---------------------- Deallocate TMEM ----------------------
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(*tmem_QK, 64);
        tmem_dealloc_fn(*tmem_PV0, 64);
        tmem_dealloc_fn(*tmem_PV1, 64);
    }
}

// ----------------------------------------------------------------------
// TVM-FFI Binding
// ----------------------------------------------------------------------

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
         
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    if (S == 0) return;

    CUtensorMap tma_Q, tma_K, tma_V;
    void* q_ptr = Q.data_ptr();
    void* k_ptr = K.data_ptr();
    void* v_ptr = V.data_ptr();

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, q_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, k_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, v_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int blocks_x = (S + 127) / 128;
    dim3 grid(blocks_x, B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 114688; // 112 KB
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_attention_wgmma_kernel, 
        tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), static_cast<float*>(LSE.data_ptr()), 
        S, D, H));
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace mha_kernel