#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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

namespace tvm_ffi_flash_attention_bwd {

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
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ __nv_bfloat16 read_smem(const __nv_bfloat16* smem, uint32_t row, uint32_t col) {
    uint32_t chunk = col / 8;
    uint32_t offset = col % 8;
    uint32_t swizzled_chunk = chunk ^ (row % 8);
    uint32_t swizzled_col = swizzled_chunk * 8 + offset;
    return smem[row * 64 + swizzled_col];
}

__device__ __forceinline__ void write_smem(__nv_bfloat16* smem, uint32_t row, uint32_t col, __nv_bfloat16 val) {
    uint32_t chunk = col / 8;
    uint32_t offset = col % 8;
    uint32_t swizzled_chunk = chunk ^ (row % 8);
    uint32_t swizzled_col = swizzled_chunk * 8 + offset;
    smem[row * 64 + swizzled_col] = val;
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(d, dataType, 2, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

__global__ void __launch_bounds__(128) flash_attention_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    uint32_t S)
{
    uint32_t my_tile = blockIdx.x;
    uint32_t batch_head = blockIdx.y;
    float scale = 1.0f / sqrtf(128.0f);

    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* smem_Q1 = smem_Q0 + 64*64;
    __nv_bfloat16* smem_K0 = smem_Q1 + 64*64;
    __nv_bfloat16* smem_K1 = smem_K0 + 64*64;
    __nv_bfloat16* smem_V0 = smem_K1 + 64*64;
    __nv_bfloat16* smem_V1 = smem_V0 + 64*64;
    __nv_bfloat16* smem_dO0 = smem_V1 + 64*64;
    __nv_bfloat16* smem_dO1 = smem_dO0 + 64*64;
    __nv_bfloat16* smem_O0 = smem_dO1 + 64*64;
    __nv_bfloat16* smem_O1 = smem_O0 + 64*64;
    __nv_bfloat16* smem_PT = smem_O1 + 64*64;
    __nv_bfloat16* smem_dST = smem_PT + 64*64;
    float* smem_D = (float*)(smem_dST + 64*64);
    float* smem_LSE = smem_D + 64;
    
    uint64_t* bar_kv_p1 = (uint64_t*)(smem_LSE + 64);
    uint64_t* bar_q_p1 = bar_kv_p1 + 1;
    uint64_t* bar_kv_p2 = bar_q_p1 + 1;
    uint64_t* bar_q_p2 = bar_kv_p2 + 1;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_kv_p1, 1);
        init_smem_barrier_fn(bar_q_p1, 1);
        init_smem_barrier_fn(bar_kv_p2, 1);
        init_smem_barrier_fn(bar_q_p2, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t last_q_tile = (S + 63) / 64 - 1;
    uint64_t b_off = (uint64_t)batch_head * S * 128;
    uint32_t query_start = my_tile * 64;
    uint32_t key_start = my_tile * 64;

    // ==========================================
    // PASS 1: Compute dQ
    // ==========================================
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_q_p1, 16384 * 6);
        tma_load_2d_fn(&tma_Q, bar_q_p1, smem_Q0, 0, query_start);
        tma_load_2d_fn(&tma_Q, bar_q_p1, smem_Q1, 64, query_start);
        tma_load_2d_fn(&tma_dO, bar_q_p1, smem_dO0, 0, query_start);
        tma_load_2d_fn(&tma_dO, bar_q_p1, smem_dO1, 64, query_start);
        tma_load_2d_fn(&tma_O, bar_q_p1, smem_O0, 0, query_start);
        tma_load_2d_fn(&tma_O, bar_q_p1, smem_O1, 64, query_start);
    }
    
    uint32_t phase = 0;
    
    for (uint32_t k_tile = 0; k_tile <= my_tile; ++k_tile) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_kv_p1, 16384 * 4);
            tma_load_2d_fn(&tma_K, bar_kv_p1, smem_K0, 0, k_tile * 64);
            tma_load_2d_fn(&tma_K, bar_kv_p1, smem_K1, 64, k_tile * 64);
            tma_load_2d_fn(&tma_V, bar_kv_p1, smem_V0, 0, k_tile * 64);
            tma_load_2d_fn(&tma_V, bar_kv_p1, smem_V1, 64, k_tile * 64);
        }
        
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(bar_kv_p1, phase);
        }
        __syncthreads();
        fence_proxy_async_fn();
        
        if (threadIdx.x < 64) {
            uint32_t tid_q = threadIdx.x;
            for (uint32_t j = 0; j < 64; ++j) {
                float s = 0;
                for (uint32_t d = 0; d < 64; ++d) {
                    s += __bfloat162float(read_smem(smem_Q0, tid_q, d)) * __bfloat162float(read_smem(smem_K0, j, d));
                    s += __bfloat162float(read_smem(smem_Q1, tid_q, d)) * __bfloat162float(read_smem(smem_K1, j, d));
                }
                s *= scale;
                
                if (query_start + tid_q >= key_start + j && query_start + tid_q < S && key_start + j < S) {
                     float lse = (my_tile * 64 + tid_q < S) ? L[batch_head * S + my_tile * 64 + tid_q] : 0.0f;
                     s = expf(s - lse);
                } else {
                    s = 0;
                }
                write_smem(smem_PT, j, tid_q, s);
                
                float dp = 0;
                for (uint32_t d = 0; d < 64; ++d) {
                    dp += __bfloat162float(read_smem(smem_V0, j, d)) * __bfloat162float(read_smem(smem_dO0, tid_q, d));
                    dp += __bfloat162float(read_smem(smem_V1, j, d)) * __bfloat162float(read_smem(smem_dO1, tid_q, d));
                }
                
                float ds = s * (dp - smem_D[tid_q]);
                write_smem(smem_dST, j, tid_q, ds);
            }
        }
        __syncthreads();
        
        uint32_t row = threadIdx.x % 64;
        uint32_t head_half = threadIdx.x / 64;
        float acc0 = 0, acc1 = 0;
        
        if (head_half == 0) {
            for (uint32_t col = 0; col < 64; ++col) {
                for (uint32_t j = 0; j < 64; ++j) {
                    float ds = __bfloat162float(read_smem(smem_dST, j, row));
                    acc0 += ds * __bfloat162float(read_smem(smem_K0, j, col));
                }
                if (query_start + row < S) {
                     atomicAdd(&dQ[b_off + (query_start + row) * 128 + col], __float2bfloat16(acc0 * scale));
                }
            }
        } else {
            for (uint32_t col = 0; col < 64; ++col) {
                for (uint32_t j = 0; j < 64; ++j) {
                    float ds = __bfloat162float(read_smem(smem_dST, j, row));
                    acc1 += ds * __bfloat162float(read_smem(smem_K1, j, col));
                }
                 if (query_start + row < S) {
                     atomicAdd(&dQ[b_off + (query_start + row) * 128 + 64 + col], __float2bfloat16(acc1 * scale));
                 }
            }
        }
        
        phase ^= 1;
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        mbarrier_wait_fn(bar_q_p1, 0);
    }
    __syncthreads();
    
    if (threadIdx.x < 64) {
        float d_val = 0;
        for(uint32_t d = 0; d < 64; ++d) {
            d_val += __bfloat162float(read_smem(smem_O0, threadIdx.x, d)) * __bfloat162float(read_smem(smem_dO0, threadIdx.x, d));
            d_val += __bfloat162float(read_smem(smem_O1, threadIdx.x, d)) * __bfloat162float(read_smem(smem_dO1, threadIdx.x, d));
        }
        smem_D[threadIdx.x] = d_val;
    }
    __syncthreads();
    
    // ==========================================
    // PASS 2: Compute dK and dV
    // ==========================================
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_kv_p2, 16384 * 4);
        tma_load_2d_fn(&tma_K, bar_kv_p2, smem_K0, 0, key_start);
        tma_load_2d_fn(&tma_K, bar_kv_p2, smem_K1, 64, key_start);
        tma_load_2d_fn(&tma_V, bar_kv_p2, smem_V0, 0, key_start);
        tma_load_2d_fn(&tma_V, bar_kv_p2, smem_V1, 64, key_start);
    }
    
    phase = 0;
    
    float dK_acc0[64] = {0}, dK_acc1[64] = {0};
    float dV_acc0[64] = {0}, dV_acc1[64] = {0};

    for (uint32_t q_tile = my_tile; q_tile <= last_q_tile; ++q_tile) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_q_p2, 16384 * 6);
            tma_load_2d_fn(&tma_Q, bar_q_p2, smem_Q0, 0, q_tile * 64);
            tma_load_2d_fn(&tma_Q, bar_q_p2, smem_Q1, 64, q_tile * 64);
            tma_load_2d_fn(&tma_dO, bar_q_p2, smem_dO0, 0, q_tile * 64);
            tma_load_2d_fn(&tma_dO, bar_q_p2, smem_dO1, 64, q_tile * 64);
            tma_load_2d_fn(&tma_O, bar_q_p2, smem_O0, 0, q_tile * 64);
            tma_load_2d_fn(&tma_O, bar_q_p2, smem_O1, 64, q_tile * 64);
        }
        
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(bar_q_p2, phase);
        }
        __syncthreads();
        fence_proxy_async_fn();
        
        if (threadIdx.x < 64) {
            float d_val = 0;
            for(uint32_t d = 0; d < 64; ++d) {
                d_val += __bfloat162float(read_smem(smem_O0, threadIdx.x, d)) * __bfloat162float(read_smem(smem_dO0, threadIdx.x, d));
                d_val += __bfloat162float(read_smem(smem_O1, threadIdx.x, d)) * __bfloat162float(read_smem(smem_dO1, threadIdx.x, d));
            }
            smem_D[threadIdx.x] = d_val;
        }
        __syncthreads();
        
        if (threadIdx.x < 64) {
            uint32_t tid_q = threadIdx.x;
            for (uint32_t j = 0; j < 64; ++j) {
                float s = 0;
                for (uint32_t d = 0; d < 64; ++d) {
                    s += __bfloat162float(read_smem(smem_Q0, tid_q, d)) * __bfloat162float(read_smem(smem_K0, j, d));
                    s += __bfloat162float(read_smem(smem_Q1, tid_q, d)) * __bfloat162float(read_smem(smem_K1, j, d));
                }
                s *= scale;
                
                if (q_tile * 64 + tid_q >= key_start + j && q_tile * 64 + tid_q < S && key_start + j < S) {
                     float lse = (q_tile * 64 + tid_q < S) ? L[batch_head * S + q_tile * 64 + tid_q] : 0.0f;
                     s = expf(s - lse);
                } else {
                    s = 0;
                }
                write_smem(smem_PT, j, tid_q, s);
                
                float dp = 0;
                for (uint32_t d = 0; d < 64; ++d) {
                    dp += __bfloat162float(read_smem(smem_V0, j, d)) * __bfloat162float(read_smem(smem_dO0, tid_q, d));
                    dp += __bfloat162float(read_smem(smem_V1, j, d)) * __bfloat162float(read_smem(smem_dO1, tid_q, d));
                }
                
                float ds = s * (dp - smem_D[tid_q]);
                write_smem(smem_dST, j, tid_q, ds);
            }
        }
        __syncthreads();
        
        uint32_t my_j = threadIdx.x % 64;
        uint32_t head_half = threadIdx.x / 64;
        
        for (uint32_t col = 0; col < 64; ++col) {
            float dk = 0, dv = 0;
            for (uint32_t i = 0; i < 64; ++i) {
                float ds_val = __bfloat162float(read_smem(smem_dST, my_j, i));
                float p_val = __bfloat162float(read_smem(smem_PT, my_j, i));
                
                dk += ds_val * __bfloat162float(head_half == 0 ? read_smem(smem_Q0, i, col) : read_smem(smem_Q1, i, col));
                dv += p_val * __bfloat162float(head_half == 0 ? read_smem(smem_dO0, i, col) : read_smem(smem_dO1, i, col));
            }
            
            if (head_half == 0) {
                dK_acc0[col] += dk * scale;
                dV_acc0[col] += dv;
            } else {
                dK_acc1[col] += dk * scale;
                dV_acc1[col] += dv;
            }
        }
        
        phase ^= 1;
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        mbarrier_wait_fn(bar_kv_p2, 0);
    }
    __syncthreads();
    
    uint32_t my_j_store = threadIdx.x % 64;
    uint32_t head_half_store = threadIdx.x / 64;
    
    if (key_start + my_j_store < S) {
        for (uint32_t col = 0; col < 64; ++col) {
            if (head_half_store == 0) {
                dK[b_off + (key_start + my_j_store) * 128 + col] = __float2bfloat16(dK_acc0[col]);
                dV[b_off + (key_start + my_j_store) * 128 + col] = __float2bfloat16(dV_acc0[col]);
            } else {
                dK[b_off + (key_start + my_j_store) * 128 + 64 + col] = __float2bfloat16(dK_acc1[col]);
                dV[b_off + (key_start + my_j_store) * 128 + 64 + col] = __float2bfloat16(dV_acc1[col]);
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    create_tma_2d_descriptor_2B(&tma_Q, (void*)Q_ptr, 128, S, 64, 64, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, (void*)K_ptr, 128, S, 64, 64, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, (void*)V_ptr, 128, S, 64, 64, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_O, (void*)O_ptr, 128, S, 64, 64, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dO, (void*)dO_ptr, 128, S, 64, 64, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    uint32_t num_tiles = (S + 63) / 64;
    dim3 grid(num_tiles, B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 102400;
    config.stream = stream;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, flash_attention_bwd_kernel, tma_Q, tma_K, tma_V, tma_O, tma_dO, L_ptr, dQ_ptr, dK_ptr, dV_ptr, S));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_flash_attention_bwd