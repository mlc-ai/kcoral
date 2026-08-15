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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

struct SharedStorage {
    alignas(128) __nv_bfloat16 s_Q0[8192];
    alignas(128) __nv_bfloat16 s_Q1[8192];
    alignas(128) __nv_bfloat16 s_K[16384];
    alignas(128) __nv_bfloat16 s_V[16384];
    alignas(128) __nv_bfloat16 s_P[16384];
    alignas(128) __nv_bfloat16 s_O[16384];
    alignas(128) float s_M[128];
    alignas(128) float s_L[128];
};

__shared__ alignas(8) uint64_t bar_Q;
__shared__ alignas(8) uint64_t bar_KV;

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
    __syncthreads();
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(&bar_Q, 32768);
        tma_load_2d_fn(&tma_Q, &bar_Q, smem->s_Q0, 0, bh * S + row_base);
        tma_load_2d_fn(&tma_Q, &bar_Q, smem->s_Q1, 64, bh * S + row_base);
    }
    mbarrier_wait_fn(&bar_Q, 0);
    fence_proxy_async_fn();
    
    uint32_t num_iters = (S + 127) / 128;
    uint32_t phase_KV = 0;
    
    float scale_factor = 1.0f / sqrtf(128);
    
    int warp_id = tid / 32;
    int row_base_warp = warp_id * 32;
    int row0 = row_base_warp + 0;
    int row1 = row_base_warp + 16;

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
        fence_proxy_async_fn();
        
        for (int n_tile = 0; n_tile < 8; n_tile++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> r_S0, r_S1;
            wmma::fill_fragment(r_S0, 0);
            wmma::fill_fragment(r_S1, 0);
            
            int n_tile_base = n_tile * 16;

            for (int k = 0; k < 8; k++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_Q0, a_Q1;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_K;
                
                bool in_Q0 = (k < 4);
                int k_off = (k < 4) ? k * 16 : (k - 4) * 16;
                uint32_t offset_Q = in_Q0 ? 0 : 0; 
                uint32_t offset_K = (k < 4) ? 0 : 8192;
                
                if (in_Q0) {
                    wmma::load_matrix_sync(a_Q0, (const __nv_bfloat16*)smem->s_Q0 + offset_Q + row0 * 64 + k_off, 64);
                    wmma::load_matrix_sync(a_Q1, (const __nv_bfloat16*)smem->s_Q0 + offset_Q + row1 * 64 + k_off, 64);
                } else {
                    wmma::load_matrix_sync(a_Q0, (const __nv_bfloat16*)smem->s_Q1 + offset_Q + row0 * 64 + k_off, 64);
                    wmma::load_matrix_sync(a_Q1, (const __nv_bfloat16*)smem->s_Q1 + offset_Q + row1 * 64 + k_off, 64);
                }
                
                wmma::load_matrix_sync(b_K, (const __nv_bfloat16*)smem->s_K + offset_K + n_tile_base * 64 + k_off, 64, wmma::mem_col_major);
                
                wmma::mma_sync(r_S0, a_Q0, b_K);
                wmma::mma_sync(r_S1, a_Q1, b_K);
            }
            
            wmma::store_matrix_sync(smem->s_P + row0 * 128 + n_tile_base, r_S0, 128, wmma::mem_row_major);
            wmma::store_matrix_sync(smem->s_P + row1 * 128 + n_tile_base, r_S1, 128, wmma::mem_row_major);
        }
        
        __syncthreads();
        
        float max_val = -INFINITY;
        for (int col = 0; col < 128; col++) {
            int c_x = col / 8;
            int c_rem = col % 8;
            int swizzled_c_x = (tid % 8) ^ c_x;
            int sc = c_x * 8 + c_rem; 
            int idx = tid * 128 + sc;
            float val = __bfloat162float(smem->s_P[idx]) * scale_factor;
            if (j * 128 + col >= S) val = -INFINITY;
            max_val = fmaxf(max_val, val);
        }
        
        float sum_val = 0;
        for (int col = 0; col < 128; col++) {
            int c_x = col / 8;
            int c_rem = col % 8;
            int swizzled_c_x = (tid % 8) ^ c_x;
            int sc = c_x * 8 + c_rem;
            int idx = tid * 128 + sc;
            float val = __bfloat162float(smem->s_P[idx]) * scale_factor;
            if (j * 128 + col >= S) val = -INFINITY;
            else val = expf(val - max_val);
            smem->s_P[idx] = __float2bfloat16(val);
            sum_val += val;
        }
        
        float global_M = smem->s_M[tid];
        float nm = fmaxf(global_M, max_val);
        float alpha = expf(global_M - nm);
        smem->s_M[tid] = nm;
        smem->s_L[tid] = smem->s_L[tid] * alpha + sum_val;
        
        for (int col = 0; col < 128; col++) {
            int c_x = col / 8;
            int c_rem = col % 8;
            int swizzled_c_x = (tid % 8) ^ c_x;
            int sc = c_x * 8 + c_rem;
            int idx = tid * 128 + sc;
            float o = __bfloat162float(smem->s_O[idx]);
            smem->s_O[idx] = __float2bfloat16(o * alpha);
        }
        __syncthreads(); 
        
        for (int n_tile = 0; n_tile < 8; n_tile++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> r_PV0, r_PV1;
            wmma::fill_fragment(r_PV0, 0);
            wmma::fill_fragment(r_PV1, 0);
            
            int n_tile_base = n_tile * 16;

            for (int k = 0; k < 8; k++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_P0, a_P1;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_V0, b_V1;
                
                wmma::load_matrix_sync(a_P0, (const __nv_bfloat16*)smem->s_P + row0 * 128 + k * 16, 128);
                wmma::load_matrix_sync(a_P1, (const __nv_bfloat16*)smem->s_P + row1 * 128 + k * 16, 128);
                
                bool in_V0 = (n_tile < 4);
                int n_off = (n_tile < 4) ? n_tile * 16 : (n_tile - 4) * 16;
                uint32_t offset_V = in_V0 ? 0 : 8192;
                
                wmma::load_matrix_sync(b_V0, (const __nv_bfloat16*)smem->s_V + offset_V + k * 16 * 64 + n_off, 64, wmma::mem_row_major);
                
                wmma::mma_sync(r_PV0, a_P0, b_V0);
                wmma::mma_sync(r_PV1, a_P1, b_V0); 
            }
            
            wmma::store_matrix_sync(smem->s_P + row0 * 128 + n_tile_base, r_PV0, 128, wmma::mem_row_major);
            wmma::store_matrix_sync(smem->s_P + row1 * 128 + n_tile_base, r_PV1, 128, wmma::mem_row_major);
            
            __syncthreads(); 
            
            for (int i = tid; i < 128 * 128; i += 128) {
                int r = i / 128;
                int c = i % 128;
                if (c >= n_tile_base && c < n_tile_base + 16 && r >= row_base_warp && r < row_base_warp + 32) {
                    int c_x = c / 8;
                    int c_rem = c % 8;
                    int swizzled_c_x = (r % 8) ^ c_x;
                    int sc = c_x * 8 + c_rem;
                    int idx = r * 128 + sc;
                    float pv = __bfloat162float(smem->s_P[idx]);
                    float o = __bfloat162float(smem->s_O[idx]);
                    smem->s_O[idx] = __float2bfloat16(o + pv);
                }
            }
            __syncthreads();
        }
    }
    
    for (int col = 0; col < 128; col++) {
        int c_x = col / 8;
        int c_rem = col % 8;
        int swizzled_c_x = (tid % 8) ^ c_x;
        int sc = c_x * 8 + c_rem;
        int idx = tid * 128 + sc;
        float o = __bfloat162float(smem->s_O[idx]);
        smem->s_O[idx] = __float2bfloat16(o / smem->s_L[tid]);
    }
    __syncthreads();
    
    for (int i = tid; i < 128 * 128; i += 128) {
        int row = i / 128;
        int col = i % 128;
        int global_row = row_base + row;
        if (global_row < S && col < 128) {
            int c_x = col / 8;
            int c_rem = col % 8;
            int swizzled_c_x = (row % 8) ^ c_x;
            int sc = c_x * 8 + c_rem;
            int idx = row * 128 + sc;
            
            int out_idx = (bh * S + global_row) * 128 + col;
            O[out_idx] = smem->s_O[idx];
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
    
    CUDA_CHECK(cudaFuncSetAttribute(run_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 164 * 1024));
    run_kernel<<<grid, block, 164 * 1024, stream>>>(tma_Q, tma_K, tma_V, static_cast<__nv_bfloat16*>(O.data_ptr()), static_cast<float*>(LSE.data_ptr()), S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel