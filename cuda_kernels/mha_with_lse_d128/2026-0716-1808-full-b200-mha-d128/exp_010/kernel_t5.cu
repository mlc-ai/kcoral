#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)


namespace tvm_ffi_optimized_cuda {

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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_cta_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void load_matrix_swizzled(const void* smem_ptr, float dst[16][16], int row_start, int col_start) {
    const __nv_bfloat16* ptr = (const __nv_bfloat16*)smem_ptr;
    for (int row = 0; row < 16; ++row) {
        int actual_row = row_start + row;
        int row_offset = actual_row * 64;
        int swizzled_row = row_offset & ~7;
        for (int i = 0; i < 8; ++i) {
            int swizzled_x = (actual_row % 8) ^ (i / 2);
            int chunk_col = (swizzled_x << 3) | ((i % 2) << 2);
            *reinterpret_cast<float*>(&dst[row][i]) = __bfloat162float(ptr[row_offset + chunk_col]);
        }
    }
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

__device__ __forceinline__ void gemm_16x16x16_init(__noinline__ nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>& D, 
                                             __noinline__ const nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major>& A, 
                                             __noinline__ const nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major>& B) {
    nvcuda::wmma::mma_sync(&D, A, B, &D);
}

__device__ __forceinline__ void gemm_16x16x16_acc(__noinline__ nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>& D_acc, 
                                                 __noinline__ const nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major>& A, 
                                                 __noinline__ const nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major>& B) {
    nvcuda::wmma::mma_sync(&D_acc, A, B, &D_acc);
}

__device__ __forceinline__ void rescale_o_acc(__noinline__ nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>& O_acc, float scale) {
    for(float& f : O_acc.storage) f *= scale;
}

__global__ void __launch_bounds__(128) mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* gmem_O,
    float* gmem_LSE,
    uint32_t S)
{
    extern __shared__ __align__(1024) char smem_pool[];
    
    // Aligned Allocations (Assume D=128 and standard head dim sizing)
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem_pool;                
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)(smem_pool + 4096);      
    __nv_bfloat16* smem_Q2 = (__nv_bfloat16*)(smem_pool + 8192);      
    __nv_bfloat16* smem_Q3 = (__nv_bfloat16*)(smem_pool + 12288);     
    __nv_bfloat16* smem_K0_pool = (__nv_bfloat16*)(smem_pool + 16384);     
    __nv_bfloat16* smem_K1_pool = (__nv_bfloat16*)(smem_pool + 32768);     
    __nv_bfloat16* smem_V0_pool = (__nv_bfloat16*)(smem_pool + 49152);     
    __nv_bfloat16* smem_V1_pool = (__nv_bfloat16*)(smem_pool + 65536);     
    
    float* smem_P = (float*)(smem_pool + 81920);
    __nv_bfloat16* smem_S = (__nv_bfloat16*)(smem_pool + 98304);
    
    // Ensure 16 byte alignment dynamically for temporary variables
    uintptr_t tmp_ptr = (uintptr_t)(smem_pool + 106496);
    tmp_ptr = (tmp_ptr + 15) & ~15;
    float* tmp_smem = (float*)tmp_ptr;
    
    uint64_t* mbar_Q = (uint64_t*)(smem_pool + 107008);
    uint64_t* mbar_KV = (uint64_t*)(smem_pool + 107016);

    uint32_t tid = threadIdx.x;
    uint32_t s_blk = blockIdx.y;
    uint32_t bh = blockIdx.x;
    uint32_t row_base = s_blk * 64;
    uint32_t warp_id = tid / 32;
    
    // Divide attention calculation into quadrants mapped cleanly to warp layouts
    uint32_t row_warp = warp_id / 2; 
    uint32_t col_warp = warp_id % 2;
    
    uint32_t warp_row_start = row_warp * 32 + (tid % 32) / 2;
    uint32_t warp_col_start = col_warp * 64 + (tid % 2) * 32;
    
    float (*smem_P_partial)[64] = (float (*)[64])smem_P;
    float global_max[2] = {-INFINITY, -INFINITY};
    float global_sum[2] = {0.0f, 0.0f};
    
    if (tid < 256) tmp_smem[tid] = 0;
    __syncthreads();

    if (tid == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(&mbar_KV[0], 1);
        init_smem_barrier_fn(&mbar_KV[1], 1);
    }
    __syncthreads();
    
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> out_acc[2][4]; 
    for (int half = 0; half < 2; ++half) {
        for (int j = 0; j < 4; ++j) {
            for (float& f : out_acc[half][j].storage) f = 0.0f;
        }
    }
    
    uint32_t phase[2] = {0, 0};
    uint32_t idx = 0;
    int kv_idx = 1;

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
        tma_load_2d_cta_fn(&tma_Q, mbar_Q, smem_Q0, 0, bh * S + row_base);
        tma_load_2d_cta_fn(&tma_Q, mbar_Q, smem_Q1, 64, bh * S + row_base);
        tma_load_2d_cta_fn(&tma_Q, mbar_Q, smem_Q2, 0, bh * S + row_base + 32);
        tma_load_2d_cta_fn(&tma_Q, mbar_Q, smem_Q3, 64, bh * S + row_base + 32);
        
        if (S > 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_KV[0], 65536);
            tma_load_2d_cta_fn(&tma_K, &mbar_KV[0], smem_K0_pool, 0, bh * S);
            tma_load_2d_cta_fn(&tma_K, &mbar_KV[0], smem_K1_pool, 64, bh * S);
            tma_load_2d_cta_fn(&tma_V, &mbar_KV[0], smem_V0_pool, 0, bh * S);
            tma_load_2d_cta_fn(&tma_V, &mbar_KV[0], smem_V1_pool, 64, bh * S);
        }
    }
    
    mbarrier_wait_fn(mbar_Q, 0);
    fence_proxy_async_shared_fn();

    for (; idx < S/64; idx++) {
        int cur_kv = kv_idx ^ 1;
        __nv_bfloat16* c_K0 = (cur_kv == 0) ? smem_K0_pool : (smem_K0_pool + 4096);
        __nv_bfloat16* c_K1 = (cur_kv == 0) ? smem_K1_pool : (smem_K1_pool + 4096);
        __nv_bfloat16* c_V0 = (cur_kv == 0) ? smem_V0_pool : (smem_V0_pool + 4096);
        __nv_bfloat16* c_V1 = (cur_kv == 0) ? smem_V1_pool : (smem_V1_pool + 4096);
        
        mbarrier_wait_fn(&mbar_KV[cur_kv], phase[cur_kv]);
        fence_proxy_async_shared_fn();
        phase[cur_kv] ^= 1;
        
        // Software Pipeline: prefetch KV asynchronously while utilizing current iteration's data
        if (idx + 1 < S/64) {
            if (tid == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_KV[kv_idx], 65536);
                __nv_bfloat16* n_K0 = (kv_idx == 0) ? smem_K0_pool : (smem_K0_pool + 4096);
                __nv_bfloat16* n_K1 = (kv_idx == 0) ? smem_K1_pool : (smem_K1_pool + 4096);
                __nv_bfloat16* n_V0 = (kv_idx == 0) ? smem_V0_pool : (smem_V0_pool + 4096);
                __nv_bfloat16* n_V1 = (kv_idx == 0) ? smem_V1_pool : (smem_V1_pool + 4096);
                
                tma_load_2d_cta_fn(&tma_K, &mbar_KV[kv_idx], n_K0, 0, bh * S + (idx + 1) * 64);
                tma_load_2d_cta_fn(&tma_K, &mbar_KV[kv_idx], n_K1, 64, bh * S + (idx + 1) * 64);
                tma_load_2d_cta_fn(&tma_V, &mbar_KV[kv_idx], n_V0, 0, bh * S + (idx + 1) * 64);
                tma_load_2d_cta_fn(&tma_V, &mbar_KV[kv_idx], n_V1, 64, bh * S + (idx + 1) * 64);
            }
        }
        kv_idx ^= 1;
        
        float local_max[2] = {-INFINITY, -INFINITY};
        
        // P Matrix accumulation across entire warp (Ensure every thread participates in computation)
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc[4];
        for (int j = 0; j < 4; ++j) {
            for (float& f : acc[j].storage) f = 0.0f;
        }
        
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> Q_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> K_frag;

        for (int k_step = 0; k_step < 4; ++k_step) {
            load_matrix_swizzled(smem_Q0 + warp_row_start * 64 + k_step * 16, Q_frag.storage, 0, 0);
            load_matrix_swizzled(c_K0 + k_step * 16 * 64, K_frag.storage, 0, 0);
            gemm_16x16x16_acc(acc[0], Q_frag, K_frag);
        }
        for (int k_step = 0; k_step < 4; ++k_step) {
            load_matrix_swizzled(smem_Q1 + warp_row_start * 64 + k_step * 16, Q_frag.storage, 0, 0);
            load_matrix_swizzled(c_K1 + k_step * 16 * 64, K_frag.storage, 0, 0);
            gemm_16x16x16_acc(acc[0], Q_frag, K_frag);
        }
        
        for (int j = 1; j < 4; ++j) {
            for (int k_step = 0; k_step < 4; ++k_step) {
                load_matrix_swizzled(smem_Q0 + warp_row_start * 64 + k_step * 16, Q_frag.storage, 0, 0);
                load_matrix_swizzled(c_K0 + k_step * 16 * 64 + j * 16, K_frag.storage, 0, 0);
                gemm_16x16x16_acc(acc[j], Q_frag, K_frag);
            }
            for (int k_step = 0; k_step < 4; ++k_step) {
                load_matrix_swizzled(smem_Q1 + warp_row_start * 64 + k_step * 16, Q_frag.storage, 0, 0);
                load_matrix_swizzled(c_K1 + k_step * 16 * 64 + j * 16, K_frag.storage, 0, 0);
                gemm_16x16x16_acc(acc[j], Q_frag, K_frag);
            }
        }
        
        __syncthreads(); 
        
        for (int j = 0; j < 4; ++j) {
            nvcuda::wmma::store_matrix_sync(&smem_P[warp_row_start * 64 + j * 16], 64, acc[j], false);
        }
        __syncthreads();
        
        // Extracted row logic mapped symmetrically across thread pairs leveraging shuffles
        uint32_t my_row = tid / 2;
        uint32_t my_col_start = (tid % 2) * 32;
        float rmax = -INFINITY;
        
        for (int c = my_col_start; c < my_col_start + 32; ++c) {
            int global_c = idx * 64 + c;
            if (global_c >= S) {
                smem_P[my_row * 64 + c] = -INFINITY;
            }
            float val = smem_P[my_row * 64 + c] * (1.0f / __sqrtf(128.0f));
            rmax = fmaxf(rmax, val);
        }
        rmax = fmaxf(rmax, __shfl_xor_sync(0xFFFFFFFF, rmax, 1));
        
        float prev_max = global_max[row_warp];
        float new_max = fmaxf(prev_max, rmax);
        float scale = __expf(prev_max - new_max);
        
        global_sum[row_warp] *= scale;
        global_max[row_warp] = new_max;
        
        float rsum = 0;
        for (int c = my_col_start; c < my_col_start + 32; ++c) {
            float val = smem_P[my_row * 64 + c] * (1.0f / __sqrtf(128.0f));
            float exp_val = __expf(val - new_max);
            rsum += exp_val;
            smem_S[my_row * 64 + c] = __float2bfloat16(exp_val);
        }
        rsum += __shfl_xor_sync(0xFFFFFFFF, rsum, 1);
        global_sum[row_warp] += rsum;
        
        __syncthreads();
        
        for (int j = 0; j < 4; ++j) {
            rescale_o_acc(out_acc[0][j], scale);
            rescale_o_acc(out_acc[1][j], scale);
        }
        
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> S_frag[4];
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> V_frag;

        for (int j = 0; j < 4; ++j) {
            load_matrix_swizzled((__nv_bfloat16*)(smem_S + warp_row_start * 64 + j * 16), S_frag[j].storage, 0, 0);
        }
        
        for (int half = 0; half < 2; ++half) {
            __nv_bfloat16* h_V = (half == 0) ? c_V0 : c_V1;
            for (int j = 0; j < 4; ++j) {
                for (int k_step = 0; k_step < 4; ++k_step) {
                    load_matrix_swizzled(h_V + k_step * 16 * 64 + j * 16, V_frag.storage, 0, 0);
                    if (k_step == 0 && idx == 0) {
                        gemm_16x16x16_init(out_acc[half][j], S_frag[k_step], V_frag);
                    } else {
                        gemm_16x16x16_acc(out_acc[half][j], S_frag[k_step], V_frag);
                    }
                }
            }
        }
        
        __syncthreads(); 
    }
    
    for (int half = 0; half < 2; ++half) {
        for (int j = 0; j < 4; ++j) {
            nvcuda::wmma::store_matrix_sync(tmp_smem, 16, out_acc[half][j], false);
            __syncthreads();
            
            for (int k = 0; k < 8; ++k) {
                int r = (tid * 8 + k) / 16;
                int c = (tid * 8 + k) % 16;
                tmp_smem[tid * 8 + k] /= global_sum[row_warp];
                __nv_bfloat16 out_val = __float2bfloat16(tmp_smem[tid * 8 + k]);
                int global_row = row_base + warp_row_start + r;
                int global_col = half * 64 + j * 16 + c;
                if (global_row < S && global_col < 128) {
                    gmem_O[bh * S * 128 + global_row * 128 + global_col] = out_val;
                }
            }
            __syncthreads();
        }
    }
    
    if (tid % 2 == 0) {
        int my_row = tid / 2;
        if (row_base + my_row < S) {
            *(gmem_LSE + bh * S + row_base + my_row) = global_max[row_warp] + __logf(global_sum[row_warp]);
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
    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, q_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, k_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, v_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    __nv_bfloat16* o_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid(B * H, S / 64);
    dim3 block(128);

    uint32_t smem_bytes = 128 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel, tma_Q, tma_K, tma_V, o_data, lse_data, S));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_optimized_cuda::run);

} // namespace tvm_ffi_optimized_cuda