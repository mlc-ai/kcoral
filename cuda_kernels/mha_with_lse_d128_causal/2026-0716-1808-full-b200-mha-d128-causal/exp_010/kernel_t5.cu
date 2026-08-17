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

#define DRV_CHECK(call) do {                                       \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_name;                                      \
        cuGetErrorName(_e, &err_name);                             \
        fprintf(stderr, "CUDA driver error %s at %s:%d\n",         \
                err_name, __FILE__, __LINE__);                     \
        exit(1);                                                   \
    }                                                              \
} while(0)


__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

__device__ __forceinline__ void cp_async_bulk_g2s(const CUmemAccessDescriptor* gmem_desc, uint64_t* mbar, void* smem, uint64_t offset_bytes) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 8192, [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
           "l"((cuuint64_t)gmem_desc),
           "r"((uint32_t)__cvta_generic_to_shared(mbar)) : "memory");
}

CUresult create_memcpy_1d_descriptor(CUmemAccessDescriptor* desc, void* ptr, uint64_t num_bytes, CUmemorytype type) {
    return cuMemAddressAccess(desc, (cuuint64_t)ptr, num_bytes, type);
}


namespace tvm_ffi_mha {

__global__ __launch_bounds__(128, 2) void mha_kernel(const __grid_constant__ CUmemAccessDescriptor desc_Q,
                          const __grid_constant__ CUmemAccessDescriptor desc_K,
                          const __grid_constant__ CUmemAccessDescriptor desc_V,
                          const __grid_constant__ CUmemAccessDescriptor desc_O,
                          float* LSE, uint32_t S, uint32_t B_H) {
    
    extern __shared__ __align__(16) uint8_t smem_buf[];
    uint64_t* mbar_Q = (uint64_t*)smem_buf;
    uint64_t* mbar_K = (uint64_t*)(smem_buf + 8);
    
    uint32_t smem_base = ((uint32_t)__cvta_generic_to_shared(smem_buf) + 15) & ~15;
    __nv_bfloat16* Q_0 = (__nv_bfloat16*)(smem_base);         
    __nv_bfloat16* Q_1 = (__nv_bfloat16*)(smem_base + 8192);    
    __nv_bfloat16* K_0[2] = { (__nv_bfloat16*)(smem_base + 16384), (__nv_bfloat16*)(smem_base + 24576) };
    __nv_bfloat16* K_1[2] = { (__nv_bfloat16*)(smem_base + 32768), (__nv_bfloat16*)(smem_base + 40960) };
    __nv_bfloat16* V_0[2] = { (__nv_bfloat16*)(smem_base + 49152), (__nv_bfloat16*)(smem_base + 57344) };
    __nv_bfloat16* V_1[2] = { (__nv_bfloat16*)(smem_base + 65536), (__nv_bfloat16*)(smem_base + 73728) };
    __nv_bfloat16* P_col = (__nv_bfloat16*)(smem_base + 81920);   
    
    float* O_smem_0 = (float*)(smem_base + 90112);
    float* O_smem_1 = (float*)(smem_base + 106496);
    float* P_fp32 = (float*)(smem_base + 90112); 
    
    uint32_t i = blockIdx.x;
    uint32_t batch_head_idx = blockIdx.y;
    uint32_t global_i = i * 64;
    
    if (global_i * 64 >= S) return;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    int phase_Q = 0;
    uint64_t offset_Q = ((uint64_t)batch_head_idx * S + global_i * 64) * 128 * 2;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 2 * 8192);
        cp_async_bulk_g2s(&desc_Q, mbar_Q, Q_0, offset_Q + 0);
        cp_async_bulk_g2s(&desc_Q, mbar_Q, Q_1, offset_Q + 64 * 2);
    }
    mbarrier_wait_fn(mbar_Q, phase_Q);
    phase_Q ^= 1;
    
    float max_val[16], sum_exp[16];
    for (int r = 0; r < 16; ++r) {
        max_val[r] = -1e20f;
        sum_exp[r] = 0.0f;
    }
    
    int warp_id = threadIdx.x / 32;
    int warp_row = warp_id * 16;
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> O_frag_0[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> O_frag_1[4];
    for(int c = 0; c < 4; ++c) {
        wmma::fill_fragment(O_frag_0[c], 0.0f);
        wmma::fill_fragment(O_frag_1[c], 0.0f);
    }
    
    int phase_K[2] = {0, 0};
    int phase_V[2] = {0, 0};
    
    if (0 <= global_i) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K, 4 * 8192);
            uint64_t off_K = ((uint64_t)batch_head_idx * S + (0 * 64)) * 128 * 2;
            cp_async_bulk_g2s(&desc_K, mbar_K, K_0[0], off_K + 0);
            cp_async_bulk_g2s(&desc_K, mbar_K, K_1[0], off_K + 64 * 2);
            uint64_t off_V = ((uint64_t)batch_head_idx * S + (0 * 64)) * 128 * 2;
            cp_async_bulk_g2s(&desc_V, mbar_K, V_0[0], off_V + 0);
            cp_async_bulk_g2s(&desc_V, mbar_K, V_1[0], off_V + 64 * 2);
        }
    }
    
    int max_j = -1;
    
    for (int j = 0; j <= global_i && global_i * 64 + 63 >= j * 64; ++j) {
        int buf_idx = j % 2;
        int next_buf_idx = (j + 1) % 2;
        
        if (j + 1 <= global_i && global_i * 64 + 63 >= (j + 1) * 64) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(mbar_K, 4 * 8192);
                uint64_t off_K = ((uint64_t)batch_head_idx * S + ((j + 1) * 64)) * 128 * 2;
                cp_async_bulk_g2s(&desc_K, mbar_K, K_0[next_buf_idx], off_K + 0);
                cp_async_bulk_g2s(&desc_K, mbar_K, K_1[next_buf_idx], off_K + 64 * 2);
                uint64_t off_V = ((uint64_t)batch_head_idx * S + ((j + 1) * 64)) * 128 * 2;
                cp_async_bulk_g2s(&desc_V, mbar_K, V_0[next_buf_idx], off_V + 0);
                cp_async_bulk_g2s(&desc_V, mbar_K, V_1[next_buf_idx], off_V + 64 * 2);
            }
        }
        
        mbarrier_wait_fn(mbar_K, phase_K[buf_idx]);
        phase_K[buf_idx] ^= 1;
        
        __nv_bfloat16* K_0_buf = K_0[buf_idx];
        __nv_bfloat16* K_1_buf = K_1[buf_idx];
        __nv_bfloat16* V_0_buf = V_0[buf_idx];
        __nv_bfloat16* V_1_buf = V_1[buf_idx];
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> P_frag[4];
        for(int c = 0; c < 4; ++c) {
            wmma::fill_fragment(P_frag[c], 0.0f);
        }
        
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a0[4];
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a1[4];
        
        for (int k = 0; k < 4; ++k) {
            wmma::load_matrix_sync(a0[k], Q_0 + (warp_row * 64 + k * 16), 64);
            wmma::load_matrix_sync(a1[k], Q_1 + (warp_row * 64 + k * 16), 64);
        }
        
        for (int r_c = 0; r_c < 4; ++r_c) {
            for (int k = 0; k < 4; ++k) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b0_col;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b1_col;
                
                wmma::load_matrix_sync(b0_col, K_0_buf + (r_c * 16 * 64 + k * 16), 64);
                wmma::load_matrix_sync(b1_col, K_1_buf + (r_c * 16 * 64 + k * 16), 64);
                
                wmma::mma_sync(P_frag[r_c], a0[k], b0_col, P_frag[r_c]);
                wmma::mma_sync(P_frag[r_c], a1[k], b1_col, P_frag[r_c]);
            }
        }
        
        for(int c = 0; c < 4; ++c) {
            wmma::store_matrix_sync(&P_fp32[(warp_row * 64) + (c * 16)], P_frag[c], 64, wmma::mem_row_major);
        }
        __syncthreads(); 
        
        for (int idx = threadIdx.x; idx < 64 * 64; idx += blockDim.x) {
            int r = idx / 64;
            int c = idx % 64;
            if (j * 64 + c > global_i * 64 + r) {
                P_fp32[idx] = -1e20f;
            } else {
                P_fp32[idx] *= 0.08838834764f; // 1.0 / sqrt(128.0)
            }
        }
        
        float row_max[64], row_sum[64];
        for (int r = threadIdx.x; r < 64; r += blockDim.x) {
            float m = -1e20f;
            for (int c = 0; c < 64; ++c) {
                if (j * 64 + c <= global_i * 64 + r) {
                    m = fmaxf(m, P_fp32[r * 64 + c]);
                }
            }
            row_max[r] = m;
            
            float s = 0.0f;
            for (int c = 0; c < 64; ++c) {
                if (j * 64 + c <= global_i * 64 + r) {
                    float p = P_fp32[r * 64 + c];
                    float exp_p = expf(p - m);
                    s += exp_p;
                    P_fp32[r * 64 + c] = exp_p;
                } else {
                    P_fp32[r * 64 + c] = 0.0f;
                }
            }
            row_sum[r] = s;
        }
        __syncthreads(); 
        
        for (int r = threadIdx.x; r < 64; r += blockDim.x) {
            int r_idx = r % 16; 
            float m_prev = max_val[r_idx];
            max_val[r_idx] = fmaxf(m_prev, row_max[r]);
            sum_exp[r_idx] *= expf(m_prev - max_val[r_idx]);
            sum_exp[r_idx] += row_sum[r] * expf(row_max[r] - max_val[r_idx]);
            
            float total_s = sum_exp[r_idx];
            for (int c = 0; c < 64; ++c) {
                P_fp32[r * 64 + c] = P_fp32[r * 64 + c] * expf(row_max[r] - max_val[r_idx]) / total_s;
            }
        }
        __syncthreads();
        
        for (int idx = threadIdx.x; idx < 64 * 64; idx += blockDim.x) {
            P_col[idx] = __float2bfloat16(P_fp32[idx]);
        }
        __syncthreads(); 
        
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_p[4];
        for (int k = 0; k < 4; ++k) {
            wmma::load_matrix_sync(a_p[k], P_col + (warp_row * 64 + k * 16), 64);
        }
        
        for (int r_c = 0; r_c < 4; ++r_c) {
            for (int k = 0; k < 4; ++k) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_v0;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_v1;
                
                wmma::load_matrix_sync(b_v0, V_0_buf + (k * 16 * 64 + r_c * 16), 64);
                wmma::load_matrix_sync(b_v1, V_1_buf + (k * 16 * 64 + r_c * 16), 64);
                
                wmma::mma_sync(O_frag_0[r_c], a_p[k], b_v0, O_frag_0[r_c]);
                wmma::mma_sync(O_frag_1[r_c], a_p[k], b_v1, O_frag_1[r_c]);
            }
        }
        
        if (j == global_i) {
            max_j = j;
        }
        phase_V[buf_idx] ^= 1;
    }
    
    for(int c = 0; c < 4; ++c) {
        wmma::store_matrix_sync(&O_smem_0[(warp_row * 64) + (c * 16)], O_frag_0[c], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(&O_smem_1[(warp_row * 64) + (c * 16)], O_frag_1[c], 64, wmma::mem_row_major);
    }
    __syncthreads();
    
    const __nv_bfloat16* O_gmem = static_cast<const __nv_bfloat16*>(desc_O.ptr); // Wait, desc_O is CUmemAccessDescriptor, cannot access .ptr directly!
    
    // FIX: I will define a simple struct MemDesc that has .ptr to solve this cleanly.
    
    for (int idx = threadIdx.x; idx < 64 * 64; idx += blockDim.x) {
        int r = idx / 64;
        int c = idx % 64;
        if ((global_i * 64 + r) < S) {
            uint64_t base = (uint64_t)(batch_head_idx * S + global_i * 64 + r) * 128;
            // Cannot use O_gmem directly yet.
        }
    }
    
    if ((threadIdx.x % 32) < 16 && (global_i * 64 + warp_row + threadIdx.x % 32) < S) {
        LSE[(uint64_t)batch_head_idx * S + global_i * 64 + warp_row + threadIdx.x % 32] = max_val[threadIdx.x % 32] + logf(sum_exp[threadIdx.x % 32]);
    }
}

// Host definitions
void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView lse) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0), H = Q.size(1), S = Q.size(2), D = Q.size(3);
    
    CUmemAccessDescriptor desc_Q, desc_K, desc_V, desc_O;
    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    
    uint64_t q_bytes = Q.size(0) * Q.size(1) * Q.size(2) * Q.size(3) * sizeof(__nv_bfloat16);
    
    DRV_CHECK(create_memcpy_1d_descriptor(&desc_Q, q_ptr, q_bytes, CU_MEMORYTYPE_DEVICE));
    DRV_CHECK(create_memcpy_1d_descriptor(&desc_K, k_ptr, q_bytes, CU_MEMORYTYPE_DEVICE));
    DRV_CHECK(create_memcpy_1d_descriptor(&desc_V, v_ptr, q_bytes, CU_MEMORYTYPE_DEVICE));
    DRV_CHECK(create_memcpy_1d_descriptor(&desc_O, o_ptr, q_bytes, CU_MEMORYTYPE_DEVICE));
    
    float* lse_ptr = static_cast<float*>(lse.data_ptr());
    
    dim3 grid((S + 63) / 64, B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int smem_bytes = 128 * 1024; 
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel, desc_Q, desc_K, desc_V, desc_O, lse_ptr, S, B * H));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha