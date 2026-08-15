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
                err_str, __FILE__, __LINE__);                     \
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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
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
        "{\n"
        ".proxy_async;\n"
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cta.b64 [%0];\n"
        "}\n"
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

__device__ __forceinline__ uint32_t make_instr_desc_fn_trans_b(uint32_t M, uint32_t N) {
    uint32_t d = make_instr_desc_fn(M, N);
    d |= (1u << 16); 
    return d;
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, uint32_t box2, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3,
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

__global__ __launch_bounds__(128, 1) void mha_attention_wgmma_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    uint32_t output_S, uint32_t output_D,
    uint32_t H) 
{
    extern __shared__ __align__(1024) char smem_pool[];
    uint64_t* mbar = (uint64_t*)smem_pool;
    
    char* matrix_pool = (char*)(((uintptr_t)(smem_pool + 16) + 1023) & ~1023);
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)matrix_pool;
    __nv_bfloat16* smem_Q1 = smem_Q0 + 4096; // 4096 bf16 elements (8192 Bytes)
    __nv_bfloat16* smem_K0 = smem_Q1 + 4096;
    __nv_bfloat16* smem_K1 = smem_K0 + 4096;
    __nv_bfloat16* smem_V0 = smem_K1 + 4096;
    __nv_bfloat16* smem_V1 = smem_V0 + 4096;
    __nv_bfloat16* smem_P  = smem_V1 + 4096;
    float* smem_O_fp32 = (float*)(smem_P + 4096); // 4096 floats (16384 Bytes)
    
    // Ensure TMEM pointers are securely aligned sequentially
    uint32_t* tmem_QK = (uint32_t*)(smem_O_fp32 + 4096); 
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
    cluster_sync_fn();

    uint32_t seq_idx = blockIdx.x;
    uint32_t bh_idx = blockIdx.y;
    uint32_t b = bh_idx / H;
    uint32_t h = bh_idx % H;
    uint32_t batch_head_offset = (b * H + h) * output_S;
    uint32_t my_m_base = seq_idx * 64; 
    
    float qk_scale = 1.0f / sqrtf((float)output_D);
    
    uint32_t idesc_QK = make_instr_desc_fn(64, 64);
    uint32_t idesc_PV = make_instr_desc_fn_trans_b(64, 64);
    
    uint32_t phase = 0;
    int tid = threadIdx.x;

    int num_n_tiles = (output_S + 63) / 64;

    int my_base = (blockIdx.x * 64 + cluster_rank_fn() * 32);
    int my_row_base = (my_base - seq_idx * 64);
    int my_col_base = cluster_rank_fn() * 64;
    
    // ---------------------- Track State ----------------------
    float global_max[2] = {-1e20f, -1e20f};
    float global_sum[2] = {0.0f, 0.0f};
    float m_curr[2] = {-1e20f, -1e20f};
    float m_new[2] = {-1e20f, -1e20f};

    // ---------------------- Load Q ----------------------
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 16384);
        tma_load_3d_fn(&tma_Q, mbar, smem_Q0, 0, my_m_base, bh_idx);
        tma_load_3d_fn(&tma_Q, mbar, smem_Q1, 64, my_m_base, bh_idx);
    }
    mbarrier_wait_fn(mbar, phase);
    phase ^= 1;

    // ---------------------- QK Pass ----------------------
    for (int n_tile = 0; n_tile < num_n_tiles; n_tile++) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 16384);
            tma_load_3d_fn(&tma_K, mbar, smem_K0, 0, n_tile * 64, bh_idx);
            tma_load_3d_fn(&tma_K, mbar, smem_K1, 64, n_tile * 64, bh_idx);
            tma_load_3d_fn(&tma_V, mbar, smem_V0, 0, n_tile * 64, bh_idx);
            tma_load_3d_fn(&tma_V, mbar, smem_V1, 64, n_tile * 64, bh_idx);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        m_curr[0] = global_max[0];
        m_curr[1] = global_max[1];
        
        float local_max_val[2] = {-1e20f, -1e20f};

        for (int k_part = 0; k_part < 2; k_part++) {
            uint64_t desc_Q = make_smem_desc_sm100_fn(k_part == 0 ? smem_Q0 : smem_Q1, 1024, 1024);
            uint64_t desc_K = make_smem_desc_sm100_fn(k_part == 0 ? smem_K0 : smem_K1, 1024, 1024);
            
            for (int k_idx = 0; k_idx < 4; k_idx++) {
                uint64_t desc_A = desc_Q + (k_idx * 32 >> 4); 
                uint64_t desc_B = desc_K + (k_idx * 32 >> 4); 
                
                uint32_t accum = (k_part == 0 && k_idx == 0) ? 0 : 1;
                uint32_t tmem_col = *tmem_QK + k_idx * 16; 
                
                umma_f16_cg2_fn(tmem_col, desc_A, desc_B, idesc_QK, accum);
            }
        }
        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        if (tid < 64) {
            int my_tid = tid;
            for (int load_idx = 0; load_idx < 16; load_idx++) {
                int col = load_idx * 4;
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(*tmem_QK + col, &r0, &r1, &r2, &r3);
                
                float v0 = __uint_as_float(r0) * qk_scale;
                float v1 = __uint_as_float(r1) * qk_scale;
                float v2 = __uint_as_float(r2) * qk_scale;
                float v3 = __uint_as_float(r3) * qk_scale;
                
                int ithread = (my_tid + col) / 8;
                local_max_val[ithread] = fmaxf(local_max_val[ithread], fmaxf(fmaxf(v0, v1), fmaxf(v2, v3)));
            }
            tmem_load_fence_fn();
        }

        m_new[0] = fmaxf(m_curr[0], local_max_val[0]);
        m_new[1] = fmaxf(m_curr[1], local_max_val[1]);

        float P_corr_0 = expf(m_curr[0] - m_new[0]);
        float P_corr_1 = expf(m_curr[1] - m_new[1]);

        global_sum[0] *= P_corr_0;
        global_sum[1] *= P_corr_1;
        
        if (tid < 64) {
            int my_tid = tid;
            smem_O_fp32[my_base * 128 + my_tid * 128] *= P_corr_0;
            smem_O_fp32[my_base * 128 + my_tid * 128 + 64] *= P_corr_1;
        }

        float local_sum_val[2] = {0.0f, 0.0f};
        
        if (tid < 64) {
            int my_tid = tid;
            for (int load_idx = 0; load_idx < 16; load_idx++) {
                int col = load_idx * 4;
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(*tmem_QK + col, &r0, &r1, &r2, &r3);
                
                float v0 = __uint_as_float(r0) * qk_scale;
                float v1 = __uint_as_float(r1) * qk_scale;
                float v2 = __uint_as_float(r2) * qk_scale;
                float v3 = __uint_as_float(r3) * qk_scale;
                
                int ithread = (my_tid + col) / 8;
                float p0 = expf(v0 - m_new[ithread]);
                float p1 = expf(v1 - m_new[ithread]);
                float p2 = expf(v2 - m_new[ithread]);
                float p3 = expf(v3 - m_new[ithread]);
                
                local_sum_val[ithread] += p0 + p1 + p2 + p3;

                uint32_t p_base = (uint32_t)__cvta_generic_to_shared(smem_P);
                uint32_t packed[4];
                packed[0] = (((uint32_t)__bfloat162asseenint(__float2bfloat16(p1))) << 16) | ((uint32_t)__bfloat162asseenint(__float2bfloat16(p0)));
                packed[1] = (((uint32_t)__bfloat162asseenint(__float2bfloat16(p3))) << 16) | ((uint32_t)__bfloat162asseenint(__float2bfloat16(p2)));
                packed[2] = 0; 
                packed[3] = 0;
                
                asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                    :: "r"(p_base + my_tid * 64 + col), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
            }
            tmem_load_fence_fn();
        }

        global_sum[0] += local_sum_val[0] * P_corr_0;
        global_sum[1] += local_sum_val[1] * P_corr_1;
        global_max[0] = m_new[0];
        global_max[1] = m_new[1];
        
        __syncthreads();
        fence_async_shared_fn();

        for (int k_part = 0; k_part < 2; k_part++) {
            for (int k_idx = 0; k_idx < 4; k_idx++) {
                uint64_t desc_A = make_smem_desc_sm100_fn(smem_P + k_idx * 32, 1024, 1024);
                uint64_t desc_B = make_smem_desc_sm100_fn(k_part == 0 ? smem_V0 : smem_V1, 1024, 1024);
                uint64_t desc_B_final = desc_B + (k_idx * 16 * 64 >> 4);
                
                uint32_t accum = (k_idx == 0) ? 0 : 1;
                uint32_t tmem_col_PV = (k_part == 0) ? *tmem_PV0 : *tmem_PV1;
                
                umma_f16_cg2_fn(tmem_col_PV + k_idx * 16, desc_A, desc_B_final, idesc_PV, accum);
            }
        }
        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        if (tid < 64) {
            int my_tid = tid;
            for (int load_idx = 0; load_idx < 16; load_idx++) {
                int col = load_idx * 4;
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(*tmem_PV0 + col, &r0, &r1, &r2, &r3);
                smem_O_fp32[my_base * 128 + my_tid * 128 + col] += __uint_as_float(r0);
                smem_O_fp32[my_base * 128 + my_tid * 128 + col + 1] += __uint_as_float(r1);
                smem_O_fp32[my_base * 128 + my_tid * 128 + col + 2] += __uint_as_float(r2);
                smem_O_fp32[my_base * 128 + my_tid * 128 + col + 3] += __uint_as_float(r3);
                
                tmem_load_4x_fn(*tmem_PV1 + col, &r0, &r1, &r2, &r3);
                smem_O_fp32[my_base * 128 + my_tid * 128 + col + 64] += __uint_as_float(r0);
                smem_O_fp32[my_base * 128 + my_tid * 128 + col + 65] += __uint_as_float(r1);
                smem_O_fp32[my_base * 128 + my_tid * 128 + col + 66] += __uint_as_float(r2);
                smem_O_fp32[my_base * 128 + my_tid * 128 + col + 67] += __uint_as_float(r3);
            }
            tmem_load_fence_fn();
        }
        __syncthreads();
    }

    // ---------------------- Epilogue ----------------------
    if (tid < 64) {
        int my_tid = tid;
        int m_idx = my_m_base + my_row_base + my_tid;
        if (m_idx < output_S) {
            for(int d = 0; d < 128; d++) {
                float o_val = smem_O_fp32[my_base * 128 + my_tid * 128 + d] / global_sum[(my_tid + d) / 64];
                O[((b * H + h) * output_S + m_idx) * output_D + d] = __float2bfloat16(o_val);
            }
            
            float lse = (m_curr[(my_tid + 0) / 64] + logf(global_sum[(my_tid + 0) / 64]));
            LSE[(b * H + h) * output_S + m_idx] = lse;
        }
    }

    // ---------------------- Deallocate TMEM ----------------------
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(*tmem_QK, 64);
        tmem_dealloc_fn(*tmem_PV0, 64);
        tmem_dealloc_fn(*tmem_PV1, 64);
    }
    cluster_sync_fn();
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

    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, q_ptr, D, S, B * H, 64, 64, 1, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, k_ptr, D, S, B * H, 64, 64, 1, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, v_ptr, D, S, B * H, 64, 64, 1, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int blocks_x = (S + 63) / 64;
    dim3 grid(blocks_x, B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 90112 + 1024; 
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_attention_wgmma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 90112 + 1024));
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_attention_wgmma_kernel, 
        tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), static_cast<float*>(LSE.data_ptr()), 
        S, D, H));
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace mha_kernel