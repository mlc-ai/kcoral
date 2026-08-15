#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>
#include <mma.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                (int)_e, __FILE__, __LINE__);                    \
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

namespace tvm_ffi_kernel {

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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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

struct SharedStorage {
    alignas(16) __nv_bfloat16 s_Q[16384]; // 32 KB
    alignas(16) __nv_bfloat16 s_K[16384]; // 32 KB
    alignas(16) __nv_bfloat16 s_V[16384]; // 32 KB
    alignas(16) __nv_bfloat16 s_P[16384]; // 32 KB
    alignas(16) __nv_bfloat16 s_O[16384]; // 32 KB
    alignas(16) float s_M[128];          // 512 B
    alignas(16) float s_L[128];          // 512 B
    char pad[140 * 1024 - 16384 * sizeof(__nv_bfloat16) * 5 - 128 * sizeof(float) * 2];
};

__global__ __launch_bounds__(128) void run_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    uint32_t S)
{
    extern __shared__ char smem_pool[];
    SharedStorage* smem = (SharedStorage*)smem_pool;
    
    __shared__ alignas(8) uint64_t bar_Q, bar_KV;
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&bar_Q, 1);
        init_smem_barrier_fn(&bar_KV, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    
    uint32_t row_base = blockIdx.x * 128;
    uint32_t bh = blockIdx.y;
    uint32_t tid = threadIdx.x;
    
    if (tid < 128) {
        smem->s_M[tid] = -INFINITY;
        smem->s_L[tid] = 0.0f;
    }
    for (int i = tid; i < 128 * 128; i += 128) {
        smem->s_O[i] = __float2bfloat16(0.0f);
    }
    __syncthreads();
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(&bar_Q, 32768);
        tma_load_2d_fn(&tma_Q, &bar_Q, smem->s_Q, 0, bh * S + row_base);
        tma_load_2d_fn(&tma_Q, &bar_Q, smem->s_Q + 8192, 64, bh * S + row_base);
    }
    mbarrier_wait_fn(&bar_Q, 0);
    
    uint32_t num_iters = (S + 127) / 128;
    uint32_t phase_KV = 0;
    
    // Pre-declare WGMMA fragments to reduce register pressure
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_Q0, a_Q1, a_P0, a_P1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_K;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_V0, b_V1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> r_S0, r_S1, r_O0, r_O1, r_PV0, r_PV1;
    
    float global_M = -INFINITY;
    float global_L = 0.0f;
    float scale = 1.0f / sqrtf(128);
    
    for (uint32_t j = 0; j < num_iters; j++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(&bar_KV, 65536);
            tma_load_2d_fn(&tma_K, &bar_KV, smem->s_K, 0, bh * S + j * 128);
            tma_load_2d_fn(&tma_K, &bar_KV, smem->s_K + 8192, 64, bh * S + j * 128);
            tma_load_2d_fn(&tma_V, &bar_KV, smem->s_V, 0, bh * S + j * 128);
            tma_load_2d_fn(&tma_V, &bar_KV, smem->s_V + 8192, 64, bh * S + j * 128);
        }
        mbarrier_wait_fn(&bar_KV, phase_KV);
        phase_KV ^= 1;
        
        int m_base = (tid / 32) * 32;
        int m0 = m_base;
        int m1 = m_base + 16;
        
        wmma::fill_fragment(r_S0, 0);
        wmma::fill_fragment(r_S1, 0);
        for (int k = 0; k < 8; k++) {
            int k_base = k * 16;
            bool in_second_half_Q = (k_base >= 64);
            int col_offset_Q = in_second_half_Q ? (k_base - 64) : k_base;
            uint32_t half_offset_Q = in_second_half_Q ? 8192 : 0;
            
            wmma::load_matrix_sync(a_Q0, (const __nv_bfloat16*)smem->s_Q + half_offset_Q + m0 * 128 + col_offset_Q, 128);
            wmma::load_matrix_sync(a_Q1, (const __nv_bfloat16*)smem->s_Q + half_offset_Q + m1 * 128 + col_offset_Q, 128);
            
            bool in_second_half_K = (k_base >= 64);
            int col_offset_K = in_second_half_K ? (k_base - 64) : k_base;
            uint32_t half_offset_K = in_second_half_K ? 8192 : 0;
            
            wmma::load_matrix_sync(b_K, (const __nv_bfloat16*)smem->s_K + half_offset_K + 0 * 128 + col_offset_K, 128, wmma::mem_col_major);
            wmma::mma_sync(r_S0, a_Q0, b_K);
            wmma::mma_sync(r_S1, a_Q1, b_K);
        }
        
        wmma::store_matrix_sync(smem->s_P + m0 * 128 + 0, r_S0, 128, wmma::mem_row_major);
        wmma::store_matrix_sync(smem->s_P + m1 * 128 + 0, r_S1, 128, wmma::mem_row_major);
        __syncthreads();
        
        int row_in_block = tid;
        float max_val = -INFINITY;
        for (int col = 0; col < 128; col++) {
            int idx = row_in_block * 128 + col;
            float val = __bfloat162float(smem->s_P[idx]) * scale;
            if (j * 128 + col >= S) val = -INFINITY;
            max_val = fmaxf(max_val, val);
        }
        
        float sum_val = 0;
        for (int col = 0; col < 128; col++) {
            int idx = row_in_block * 128 + col;
            float val = __bfloat162float(smem->s_P[idx]) * scale;
            if (j * 128 + col >= S) val = -INFINITY;
            else val = expf(val - max_val);
            smem->s_P[idx] = __float2bfloat16(val);
            sum_val += val;
        }
        
        float gm = smem->s_M[row_in_block];
        float nm = fmaxf(gm, max_val);
        float alpha = expf(gm - nm);
        smem->s_M[row_in_block] = nm;
        smem->s_L[row_in_block] = smem->s_L[row_in_block] * alpha + sum_val;
        
        for (int i = tid; i < 128 * 128; i += 128) {
            int r = i / 128;
            float a = expf(smem->s_M[r] - nm); // Wait, smem->s_M[r] is ALREADY updated to nm!
            // BUG: I need the OLD global M before the update!
            float o = __bfloat162float(smem->s_O[i]);
            smem->s_O[i] = __float2bfloat16(o * a);
        }
        
        // Fix: Let's scale using the previously calculated 'alpha' for the specific row
        for (int col = 0; col < 128; col++) {
            int idx = row_in_block * 128 + col;
            float o = __bfloat162float(smem->s_O[idx]);
            smem->s_O[idx] = __float2bfloat16(o * alpha);
        }
        __syncthreads(); 
        
        wmma::fill_fragment(r_PV0, 0);
        wmma::fill_fragment(r_PV1, 0);
        for (int k = 0; k < 8; k++) {
            int k_base = k * 16;
            bool in_second_half_P = (k_base >= 64);
            int col_offset_P = in_second_half_P ? (k_base - 64) : k_base;
            uint32_t half_offset_P = in_second_half_P ? 8192 : 0;
            
            wmma::load_matrix_sync(a_P0, (const __nv_bfloat16*)smem->s_P + half_offset_P + m0 * 128 + col_offset_P, 128);
            wmma::load_matrix_sync(a_P1, (const __nv_bfloat16*)smem->s_P + half_offset_P + m1 * 128 + col_offset_P, 128);
            
            bool in_second_half_V = (k_base >= 64);
            int col_offset_V = in_second_half_V ? (k_base - 64) : k_base;
            uint32_t half_offset_V = in_second_half_V ? 8192 : 0;
            
            wmma::load_matrix_sync(b_V0, (const __nv_bfloat16*)smem->s_V + half_offset_V + k_base * 128 + 0, 128, wmma::mem_row_major);
            wmma::load_matrix_sync(b_V1, (const __nv_bfloat16*)smem->s_V + half_offset_V + k_base * 128 + 16, 128, wmma::mem_row_major);
            
            wmma::mma_sync(r_PV0, a_P0, b_V0);
            wmma::mma_sync(r_PV1, a_P1, b_V1);
        }
        
        wmma::store_matrix_sync(smem->s_K + m0 * 128, r_PV0, 128, wmma::mem_row_major);
        wmma::store_matrix_sync(smem->s_K + m1 * 128, r_PV1, 128, wmma::mem_row_major);
        __syncthreads();
        
        for (int i = tid; i < 128 * 128; i += 128) {
            float o = __bfloat162float(smem->s_O[i]);
            float pv = __bfloat162float(smem->s_K[i]);
            smem->s_O[i] = __float2bfloat16(o + pv);
        }
        __syncthreads();
    }
    
    for (int col = 0; col < 128; col++) {
        int idx = tid * 128 + col;
        float o = __bfloat162float(smem->s_O[idx]);
        smem->s_O[idx] = __float2bfloat16(o / smem->s_L[tid]);
    }
    __syncthreads();
    
    for (int i = tid; i < 128 * 128; i += 128) {
        int row = i / 128;
        int global_row = row_base + row;
        int col = i % 128;
        if (global_row < S && col < 128) {
            O[(uint64_t)(bh * S + global_row) * 128 + col] = smem->s_O[i];
        }
    }
    
    if (tid < 128) {
        int global_row = row_base + tid;
        if (global_row < S) {
            LSE[(uint64_t)bh * S + global_row] = smem->s_M[tid] + logf(smem->s_L[tid]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t D = Q.size(3);
    
    CUtensorMap tma_Q, tma_K, tma_V;
    __nv_bfloat16* Q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* K_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* V_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q_ptr, D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K_ptr, D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V_ptr, D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(run_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 140 * 1024));
    run_kernel<<<grid, block, 140 * 1024, stream>>>(tma_Q, tma_K, tma_V, static_cast<__nv_bfloat16*>(O.data_ptr()), static_cast<float*>(LSE.data_ptr()), S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel