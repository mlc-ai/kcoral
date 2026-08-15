#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <mma.h>
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

namespace tvm_ffi_example_cuda {

using namespace nvcuda;

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, uint32_t box2, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3, // tensorRank
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
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

__global__
__launch_bounds__(128)
void attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O, 
    float* __restrict__ LSE,
    int S_len)
{
    int batch_head = blockIdx.x; 
    int query_start = blockIdx.y * 64;
    int tid = threadIdx.x;
    
    if (query_start >= S_len) return;
    
    extern __shared__ __align__(128) char smem[];
    
    // Memory Layout Allocation
    // Q (64x128), K (128x128), V (128x128), P (64x128) -> All BF16
    // S (64x128) -> FP32 explicitly requested layout to avoid precision loss during calculations
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem;                      // 16KB
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem + 16384);             // 32KB
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem + 49152);             // 32KB
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem + 81920);             // 16KB
    float* smem_S = (float*)(smem + 98304);                             // 32KB
    
    uint64_t* bar_Q = (uint64_t*)(smem + 131072);
    uint64_t* bar_K = (uint64_t*)(smem + 131080);
    uint64_t* bar_V = (uint64_t*)(smem + 131088);
    
    if (tid == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_K, 1);
        init_smem_barrier_fn(bar_V, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    int warp_id = tid / 32;
    int tid_in_warp = tid % 32;
    
    // Zero out buffers cleanly avoiding undefined behavior leakage across blocks 
    for(int i = tid; i < 64 * 128; i += 128) {
        smem_Q[i] = 0;
        smem_P[i] = 0;
    }
    for(int i = tid; i < 128 * 128; i += 128) {
        smem_K[i] = 0;
        smem_V[i] = 0;
    }
    __syncthreads();
    
    int phase_Q = 0;
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_Q, 16384);
        tma_load_3d_fn(&tma_Q, bar_Q, smem_Q, 0, query_start, batch_head);
    }
    mbarrier_wait_fn(bar_Q, phase_Q);
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_O[4][2]; 
    
    // Utilize y-dimension parallelism scaling effectively mapping O accumulator states inline avoiding spills initially
    for(int i = 0; i < 4; ++i) {
        for(int j = 0; j < 2; ++j) {
            wmma::fill_fragment(acc_O[i][j], 0.0f);
        }
    }
    
    float m_old_A[2] = {-1e20f, -1e20f};
    float m_old_B[2] = {-1e20f, -1e20f};
    float d_old_A[2] = {0, 0};
    float d_old_B[2] = {0, 0};
    
    int phase_K = 0, phase_V = 0;
    float scale = 1.0f / sqrtf(128.0f);
    
    for (int j = 0; j < S_len; j += 128) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_K, 16384);
            tma_load_3d_fn(&tma_K, bar_K, smem_K, 0, j, batch_head);
            
            mbarrier_arrive_and_expect_tx_fn(bar_V, 16384);
            tma_load_3d_fn(&tma_V, bar_V, smem_V, 0, j, batch_head);
        }
        
        mbarrier_wait_fn(bar_K, phase_K);
        mbarrier_wait_fn(bar_V, phase_V);
        phase_K ^= 1;
        phase_V ^= 1;
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_S[4][2]; 
        for(int i = 0; i < 4; ++i) {
            for(int j_step = 0; j_step < 2; ++j_step) {
                wmma::fill_fragment(acc_S[i][j_step], 0.0f);
            }
        }
        
        for(int k_step = 0; k_step < 4; ++k_step) {
            for(int sub_step = 0; sub_step < 2; ++sub_step) {
                int k_off = k_step * 32 + sub_step * 16;
                
                const __nv_bfloat16* Q_ptr = &smem_Q[(warp_id * 16) * 128 + k_off]; 
                const __nv_bfloat16* K_ptr = &smem_K[k_off * 128]; 
                
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag;
                
                wmma::load_matrix_sync(q_frag, Q_ptr, 128);
                wmma::load_matrix_sync(k_frag, K_ptr, 128);
                
                for (int n_step = 0; n_step < 4; ++n_step) { 
                    for (int sub_n = 0; sub_n < 2; ++sub_n) { 
                        wmma::mma_sync(acc_S[n_step][sub_n], q_frag, k_frag, acc_S[n_step][sub_n]); 
                    } 
                } 
            } 
        } 
        
        for (int n_step = 0; n_step < 4; ++n_step) {
            for (int sub_n = 0; sub_n < 2; ++sub_n) {
                int n_off = n_step * 32 + sub_n * 16;
                wmma::store_matrix_sync(&smem_S[(warp_id * 16) * 128 + n_off], acc_S[n_step][sub_n], 128, wmma::mem_row_major);
            }
        }
        wmma::commit_sync();
        __syncthreads(); 
        
        float my_max_A[2] = {-1e20f, -1e20f};
        float my_max_B[2] = {-1e20f, -1e20f};
        
        int row = tid_in_warp / 2;
        int col_grp = tid_in_warp % 2;
        int c_start = col_grp * 64;
        
        for (int c = 0; c < 64; ++c) {
            float s0 = smem_S[(warp_id * 16 + row) * 128 + c_start + c];
            my_max_A[col_grp] = max(my_max_A[col_grp], s0 * scale);
            
            float s1 = smem_S[(warp_id * 16 + row + 1) * 128 + c_start + c];
            my_max_B[col_grp] = max(my_max_B[col_grp], s1 * scale);
        }
        
        my_max_A[0] = max(my_max_A[0], __shfl_xor_sync(0xffffffff, my_max_A[0], 1));
        my_max_A[1] = max(my_max_A[1], __shfl_xor_sync(0xffffffff, my_max_A[1], 1));
        my_max_B[0] = max(my_max_B[0], __shfl_xor_sync(0xffffffff, my_max_B[0], 1));
        my_max_B[1] = max(my_max_B[1], __shfl_xor_sync(0xffffffff, my_max_B[1], 1));
        
        float new_max_A[2], new_max_B[2];
        float factor_A[2], factor_B[2];
        
        for (int i = 0; i < 2; ++i) {
            new_max_A[i] = max(m_old_A[i], my_max_A[i]);
            new_max_B[i] = max(m_old_B[i], my_max_B[i]);
            factor_A[i] = expf(m_old_A[i] - new_max_A[i]);
            factor_B[i] = expf(m_old_B[i] - new_max_B[i]);
        }
        
        d_old_A[0] *= factor_A[0];
        d_old_A[1] *= factor_A[1];
        d_old_B[0] *= factor_B[0];
        d_old_B[1] *= factor_B[1];
        
        // Critical step: rescale actively tracked partial outputs O uniformly inline matching current block maximum limits
        for (int n_step = 0; n_step < 4; ++n_step) {
            for (int sub_n = 0; sub_n < 2; ++sub_n) {
                wmma::scale_accumulator(factor_A[col_grp], acc_O[n_step][sub_n]);
            }
        }
        
        float my_sum_A[2] = {0, 0};
        float my_sum_B[2] = {0, 0};
        
        for (int c = 0; c < 64; ++c) {
            float v0 = expf(smem_S[(warp_id * 16 + row) * 128 + c_start + c] * scale - new_max_A[col_grp]);
            my_sum_A[col_grp] += v0;
            // Ensure perfectly aligned native vectorization bounds utilizing chunking of 16 Byte spans mapping directly to SMEM banks
            *(reinterpret_cast<__nv_bfloat16*>(&smem_P[(warp_id * 16 + row) * 128 + c_start + c]) ) = __float2bfloat16(v0);
            
            float v1 = expf(smem_S[(warp_id * 16 + row + 1) * 128 + c_start + c] * scale - new_max_B[col_grp]);
            my_sum_B[col_grp] += v1;
            *(reinterpret_cast<__nv_bfloat16*>(&smem_P[(warp_id * 16 + row + 1) * 128 + c_start + c]) ) = __float2bfloat16(v1);
        }
        
        my_sum_A[0] += __shfl_xor_sync(0xffffffff, my_sum_A[0], 1);
        my_sum_A[1] += __shfl_xor_sync(0xffffffff, my_sum_A[1], 1);
        my_sum_B[0] += __shfl_xor_sync(0xffffffff, my_sum_B[0], 1);
        my_sum_B[1] += __shfl_xor_sync(0xffffffff, my_sum_B[1], 1);
        
        d_old_A[0] += my_sum_A[0];
        d_old_A[1] += my_sum_A[1];
        d_old_B[0] += my_sum_B[0];
        d_old_B[1] += my_sum_B[1];
        
        m_old_A[0] = new_max_A[0];
        m_old_A[1] = new_max_A[1];
        m_old_B[0] = new_max_B[0];
        m_old_B[1] = new_max_B[1];
        
        __syncthreads(); 
        
        // Final phase: Accumulate seamlessly against unmodified native V chunks mapping identically over identical swizzling bounds 
        for(int k_step = 0; k_step < 4; ++k_step) {
            for(int sub_step = 0; sub_step < 2; ++sub_step) {
                int k_off = k_step * 32 + sub_step * 16;
                
                const __nv_bfloat16* P_ptr = &smem_P[(warp_id * 16) * 128 + k_off]; 
                
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> p_frag;
                wmma::load_matrix_sync(p_frag, P_ptr, 128);
                
                for (int n_step = 0; n_step < 4; ++n_step) {
                    for (int sub_n = 0; sub_n < 2; ++sub_n) {
                        int n_off = n_step * 32 + sub_n * 16;
                        const __nv_bfloat16* V_ptr = &smem_V[k_off * 128 + n_off]; 
                        
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> v_frag;
                        wmma::load_matrix_sync(v_frag, V_ptr, 128);
                        
                        wmma::mma_sync(acc_O[n_step][sub_n], p_frag, v_frag, acc_O[n_step][sub_n]);
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Coalesced Epilogue Output
    for (int n_step = 0; n_step < 4; ++n_step) {
        for (int sub_n = 0; sub_n < 2; ++sub_n) {
            int n_off = n_step * 32 + sub_n * 16;
            wmma::store_matrix_sync(&smem_S[(warp_id * 16) * 128 + n_off], acc_O[n_step][sub_n], 128, wmma::mem_row_major);
        }
    }
    wmma::commit_sync();
    __syncthreads();
    
    for(int i = tid; i < 64 * 128; i += 128) {
        int r = i / 128; 
        int c = i % 128;
        if (query_start + r < S_len) {
            float out;
            if (c < 64) {
                out = smem_S[i] * (1.0f / d_old_A[r % 2]);
            } else {
                out = smem_S[i] * (1.0f / d_old_B[r % 2]);
            }
            
            __nv_bfloat16* out_ptr = &((__nv_bfloat16*)O)[batch_head * S_len * 128 + (query_start + r) * 128 + c];
            *out_ptr = __float2bfloat16(out);
        }
    }
    
    if (tid_in_warp < 16) {
        if (query_start + tid_in_warp < S_len) {
            LSE[batch_head * S_len + query_start + tid_in_warp] = m_old_A[tid_in_warp] + logf(d_old_A[tid_in_warp]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    CUtensorMap tma_Q, tma_K, tma_V;
    uint64_t S_len = S; 
    
    CUresult res_q = create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, S_len, B * H, 128, 64, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
    if (res_q != CUDA_SUCCESS) { fprintf(stderr, "Failed to create tma_Q\n"); exit(1); }
    
    CUresult res_k = create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), 128, S_len, B * H, 128, 64, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
    if (res_k != CUDA_SUCCESS) { fprintf(stderr, "Failed to create tma_K\n"); exit(1); }
    
    CUresult res_v = create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), 128, S_len, B * H, 128, 64, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
    if (res_v != CUDA_SUCCESS) { fprintf(stderr, "Failed to create tma_V\n"); exit(1); }
    
    dim3 grid(B * H, (S + 63) / 64); 
    dim3 block(128); 
    
    int smem_size = 131608; 
    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    attention_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V,
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda