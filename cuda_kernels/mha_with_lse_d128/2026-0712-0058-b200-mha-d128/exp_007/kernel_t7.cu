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

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_dim0, uint64_t gmem_dim1, uint64_t gmem_dim2, uint32_t smem_dim0, uint32_t smem_dim1, uint32_t smem_dim2, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[3] = {gmem_dim0, gmem_dim1, gmem_dim2};
    cuuint64_t globalStrides[2] = {gmem_dim0 * 2, gmem_dim0 * gmem_dim1 * 2};
    cuuint32_t boxDim[3] = {smem_dim0, smem_dim1, smem_dim2};
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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
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
    alignas(128) __nv_bfloat16 s_Q[8192];
    alignas(128) __nv_bfloat16 s_K[8192];
    alignas(128) __nv_bfloat16 s_V[8192];
    alignas(128) __nv_bfloat16 s_P[8192];
    alignas(128) __nv_bfloat16 s_O[8192];
    alignas(128) float s_M[64];
    alignas(128) float s_L[64];
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
    
    uint32_t row_base = blockIdx.x * 64;
    uint32_t bh = blockIdx.y;
    uint32_t tid = threadIdx.x;
    
    if (tid < 64) {
        smem->s_M[tid] = -INFINITY;
        smem->s_L[tid] = 0.0f;
    }
    __syncthreads();
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(&bar_Q, 16384);
        tma_load_3d_fn(&tma_Q, &bar_Q, smem->s_Q, 0, row_base, bh);
        tma_load_3d_fn(&tma_Q, &bar_Q, smem->s_Q + 4096, 64, row_base, bh);
    }
    mbarrier_wait_fn(&bar_Q, 0);
    fence_proxy_async_fn();
    
    uint32_t num_iters = (S + 63) / 64;
    uint32_t phase_KV = 0;
    
    float scale_factor = 1.0f / sqrtf(128);
    
    int warp_id = tid / 32;
    int row_base_warp = warp_id * 32;
    int row0 = row_base_warp + 0;
    int row1 = row_base_warp + 16;

    for (uint32_t j = 0; j < num_iters; j++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(&bar_KV, 32768);
            tma_load_3d_fn(&tma_K, &bar_KV, smem->s_K, 0, j * 64, bh);
            tma_load_3d_fn(&tma_K, &bar_KV, smem->s_K + 4096, 64, j * 64, bh);
            
            tma_load_3d_fn(&tma_V, &bar_KV, smem->s_V, 0, j * 64, bh);
            tma_load_3d_fn(&tma_V, &bar_KV, smem->s_V + 4096, 64, j * 64, bh);
        }
        mbarrier_wait_fn(&bar_KV, phase_KV);
        phase_KV ^= 1;
        fence_proxy_async_fn();
        
        for (int n_tile = 0; n_tile < 4; n_tile++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> r_S0, r_S1;
            wmma::fill_fragment(r_S0, 0);
            wmma::fill_fragment(r_S1, 0);
            
            int n_tile_base = n_tile * 16;

            for (int k = 0; k < 4; k++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_Q0, a_Q1;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_K;
                
                wmma::load_matrix_sync(a_Q0, (const __nv_bfloat16*)smem->s_Q + row0 * 64 + k * 16, 64);
                wmma::load_matrix_sync(a_Q1, (const __nv_bfloat16*)smem->s_Q + row1 * 64 + k * 16, 64);
                
                wmma::load_matrix_sync(b_K, (const __nv_bfloat16*)smem->s_K + n_tile_base * 64 + k * 16, 64, wmma::mem_col_major);
                
                wmma::mma_sync(r_S0, a_Q0, b_K);
                wmma::mma_sync(r_S1, a_Q1, b_K);
            }
            
            for (int k = 0; k < 4; k++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_Q0, a_Q1;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_K;
                
                wmma::load_matrix_sync(a_Q0, (const __nv_bfloat16*)smem->s_Q + 4096 + row0 * 64 + k * 16, 64);
                wmma::load_matrix_sync(a_Q1, (const __nv_bfloat16*)smem->s_Q + 4096 + row1 * 64 + k * 16, 64);
                
                wmma::load_matrix_sync(b_K, (const __nv_bfloat16*)smem->s_K + 4096 + n_tile_base * 64 + k * 16, 64, wmma::mem_col_major);
                
                wmma::mma_sync(r_S0, a_Q0, b_K);
                wmma::mma_sync(r_S1, a_Q1, b_K);
            }
            
            wmma::store_matrix_sync(smem->s_P + row0 * 64 + n_tile_base, r_S0, 64, wmma::mem_row_major);
            wmma::store_matrix_sync(smem->s_P + row1 * 64 + n_tile_base, r_S1, 64, wmma::mem_row_major);
        }
        
        __syncthreads();
        
        float max_val = -INFINITY;
        for (int col = 0; col < 64; col++) {
            int idx = tid * 64 + col;
            float val = __bfloat162float(smem->s_P[idx]) * scale_factor;
            if (j * 64 + col >= S) val = -INFINITY;
            max_val = fmaxf(max_val, val);
        }
        
        float sum_val = 0;
        for (int col = 0; col < 64; col++) {
            int idx = tid * 64 + col;
            float val = __bfloat162float(smem->s_P[idx]) * scale_factor;
            if (j * 64 + col >= S) val = -INFINITY;
            else val = expf(val - max_val);
            smem->s_P[idx] = __float2bfloat16(val);
            sum_val += val;
        }
        
        float global_M = smem->s_M[tid];
        float nm = fmaxf(global_M, max_val);
        float alpha = expf(global_M - nm);
        smem->s_M[tid] = nm;
        smem->s_L[tid] = smem->s_L[tid] * alpha + sum_val;
        
        for (int col = 0; col < 64; col++) {
            int idx0 = tid * 64 + col;
            float o = __bfloat162float(smem->s_O[idx0]);
            smem->s_O[idx0] = __float2bfloat16(o * alpha);
            
            int idx1 = 4096 + tid * 64 + col;
            o = __bfloat162float(smem->s_O[idx1]);
            smem->s_O[idx1] = __float2bfloat16(o * alpha);
        }
        __syncthreads(); 
        
        for (int n_tile = 0; n_tile < 4; n_tile++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> r_PV0, r_PV1;
            wmma::fill_fragment(r_PV0, 0);
            wmma::fill_fragment(r_PV1, 0);
            
            int n_tile_base = n_tile * 16;

            for (int k = 0; k < 4; k++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_P0, a_P1;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_V0, b_V1;
                
                wmma::load_matrix_sync(a_P0, (const __nv_bfloat16*)smem->s_P + row0 * 64 + k * 16, 64);
                wmma::load_matrix_sync(a_P1, (const __nv_bfloat16*)smem->s_P + row1 * 64 + k * 16, 64);
                
                wmma::load_matrix_sync(b_V0, (const __nv_bfloat16*)smem->s_V + k * 16 * 64 + n_tile_base, 64, wmma::mem_row_major);
                wmma::load_matrix_sync(b_V1, (const __nv_bfloat16*)smem->s_V + 4096 + k * 16 * 64 + n_tile_base, 64, wmma::mem_row_major);
                
                wmma::mma_sync(r_PV0, a_P0, b_V0);
                wmma::mma_sync(r_PV1, a_P1, b_V1);
            }
            
            wmma::store_matrix_sync(smem->s_P + row0 * 64 + n_tile_base, r_PV0, 64, wmma::mem_row_major);
            wmma::store_matrix_sync(smem->s_P + row1 * 64 + n_tile_base, r_PV1, 64, wmma::mem_row_major);
            
            wmma::store_matrix_sync(smem->s_P + 4096 + row0 * 64 + n_tile_base, r_PV0, 64, wmma::mem_row_major);
            wmma::store_matrix_sync(smem->s_P + 4096 + row1 * 64 + n_tile_base, r_PV1, 64, wmma::mem_row_major);
            
            __syncthreads(); 
            
            for (int i = tid; i < 64 * 128; i += 128) {
                int r = i / 128;
                int c = i % 128;
                if (r < 64) {
                    int half = c / 64;
                    int c_in_half = c % 64;
                    int idx_p = half * 4096 + r * 64 + c_in_half;
                    int idx_o = half * 4096 + r * 64 + c_in_half;
                    float pv = __bfloat162float(smem->s_P[idx_p]);
                    float o = __bfloat162float(smem->s_O[idx_o]);
                    smem->s_O[idx_o] = __float2bfloat16(o + pv);
                }
            }
            __syncthreads();
        }
    }
    
    for (int col = 0; col < 64; col++) {
        int idx0 = tid * 64 + col;
        float o0 = __bfloat162float(smem->s_O[idx0]) / smem->s_L[tid];
        smem->s_O[idx0] = __float2bfloat16(o0);
        
        int idx1 = 4096 + tid * 64 + col;
        float o1 = __bfloat162float(smem->s_O[idx1]) / smem->s_L[tid];
        smem->s_O[idx1] = __float2bfloat16(o1);
    }
    __syncthreads();
    
    for (int i = tid; i < 64 * 128; i += 128) {
        int row = i / 128;
        int col = i % 128;
        int global_row = row_base + row;
        if (row < 64 && col < 128) {
            int half = col / 64;
            int c_in_half = col % 64;
            int idx = half * 4096 + row * 64 + c_in_half;
            if (global_row < S) {
                int out_idx = (bh * S + global_row) * 128 + col;
                O[out_idx] = smem->s_O[idx];
            }
        }
    }
    
    if (tid < 64) {
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
    
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, Q_ptr, D, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, K_ptr, D, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, V_ptr, D, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    dim3 grid((S + 63) / 64, B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(run_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 140 * 1024));
    run_kernel<<<grid, block, 140 * 1024, stream>>>(tma_Q, tma_K, tma_V, static_cast<__nv_bfloat16*>(O.data_ptr()), static_cast<float*>(LSE.data_ptr()), S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel