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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                (int)_e, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_causal_attention {

// -------------------------------------------------------------------------------------
// Helper Functions
// -------------------------------------------------------------------------------------

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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void load_warp_matrix_row_major(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major>& frag,
                                                           const __nv_bfloat16* src, int row_start, int col_start, int stride) {
    const volatile __nv_bfloat16* src_ptr = src;
    __nv_bfloat16* frag_arr = reinterpret_cast<__nv_bfloat16*>(&frag);
    int lane_id = threadIdx.x % 32;
    
    for (int step = 0; step < 4; ++step) {
        float4 tmp = *(const float4*)(src_ptr + ((step * 32) + row_start) * stride + col_start + lane_id);
        frag_arr[step * 4 + 0] = __float2bfloat16(tmp.x);
        frag_arr[step * 4 + 1] = __float2bfloat16(tmp.y);
        frag_arr[step * 4 + 2] = __float2bfloat16(tmp.z);
        frag_arr[step * 4 + 3] = __float2bfloat16(tmp.w);
    }
}

__device__ __forceinline__ void load_warp_matrix_col_major(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major>& frag,
                                                           const __nv_bfloat16* src, int row_start, int col_start, int stride) {
    const volatile __nv_bfloat16* src_ptr = src;
    __nv_bfloat16* frag_arr = reinterpret_cast<__nv_bfloat16*>(&frag);
    int lane_id = threadIdx.x % 32;
    
    for (int step = 0; step < 4; ++step) {
        float4 tmp = *(const float4*)(src_ptr + (row_start + step * 32) * stride + col_start + lane_id);
        frag_arr[step * 4 + 0] = __float2bfloat16(tmp.x);
        frag_arr[step * 4 + 1] = __float2bfloat16(tmp.y);
        frag_arr[step * 4 + 2] = __float2bfloat16(tmp.z);
        frag_arr[step * 4 + 3] = __float2bfloat16(tmp.w);
    }
}

__device__ __forceinline__ void store_warp_matrix(__nv_bfloat16* dst, const nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>& frag,
                                                  int row_start, int col_start, int stride) {
    const float* val_arr = reinterpret_cast<const float*>(&frag);
    for (int idx = 0; idx < 8; ++idx) {
        int i_warp = (idx / 2) % 4;
        int j_warp = (idx % 2);
        int r = row_start + i_warp * 4;
        int c = col_start + j_warp * 4;
        dst[r * stride + c] = __float2bfloat16(val_arr[idx]);
    }
}

// -------------------------------------------------------------------------------------
// Causal Attention Kernel
// -------------------------------------------------------------------------------------

__global__ void causal_attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S_len, int B, int H)
{
    extern __shared__ char smem_raw[];
    size_t smem_addr = (size_t)smem_raw;
    smem_addr = (smem_addr + 1023) & ~1023;
    char* smem = (char*)smem_addr;

    // Memory Layout Planning & Allocation 
    // 1. s_Q_flat: 32KB
    // 2. s_K_flat: 64KB (Double buffered 128x128)
    // 3. s_V_flat: 64KB (Double buffered 128x128)
    // 4. s_P_bf16: 32KB (128x128 bf16)
    // 5. Meta arrays: 192 Bytes
    // 6. MBarriers: 40 Bytes
    // Total Requested: 197104 Bytes (~192.5 KB)
    
    __nv_bfloat16* s_Q_flat = (__nv_bfloat16*)smem;                   // 32768 Bytes effectively utilized initially
    __nv_bfloat16* s_K_flat = (__nv_bfloat16*)(smem + 32768);         // 65536 Bytes
    __nv_bfloat16* s_V_flat = (__nv_bfloat16*)(smem + 98304);         // 65536 Bytes
    __nv_bfloat16* s_P_bf16 = (__nv_bfloat16*)(smem + 163840);        // 32768 Bytes
    
    float* s_row_max = (float*)(smem + 196608);                       
    float* s_row_sum = (float*)(smem + 196672);                       
    float* s_prev_max = (float*)(smem + 196736);                      
    
    char* barrier_smem = smem + 196800;
    uint64_t* mbar_Q = (uint64_t*)barrier_smem;
    uint64_t* mbar_K = (uint64_t*)(barrier_smem + 8);
    uint64_t* mbar_V = (uint64_t*)(barrier_smem + 24);

    int bh = blockIdx.x * gridDim.y + blockIdx.y;
    int batch_idx = bh / H;
    int head_idx = bh % H;
    int bid = blockIdx.z;
    int q_blk = bid * 128;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    float* LSE_ptr = LSE + batch_idx * H * S_len + head_idx * S_len;
    __nv_bfloat16* O_ptr = O + batch_idx * H * S_len * 128 + head_idx * S_len * 128;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(&mbar_K[0], 1);
        init_smem_barrier_fn(&mbar_K[1], 1);
        init_smem_barrier_fn(&mbar_V[0], 1);
        init_smem_barrier_fn(&mbar_V[1], 1);
        
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768); 
        
        tma_load_3d_fn(&tma_Q, mbar_Q, s_Q_flat, 0, bh * S_len + q_blk, 0);
        tma_load_3d_fn(&tma_Q, mbar_Q, s_Q_flat + 8192, 64, bh * S_len + q_blk, 0);
    }
    
    __syncthreads();
    mbarrier_wait_fn(mbar_Q, 0);
    fence_proxy_async_fn();
    
    float prev_max_val = -1e20f;
    float prev_sum = 0.0f;
    
    int num_blocks = (S_len + 127) / 128;
    int max_j = min(num_blocks, (q_blk / 128) + 1);
    
    // Allocated persistently across sequence blocks loop. 
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc_S[8][8];
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            nvcuda::wmma::fill_fragment(acc_S[i][j], 0.0f);
        }
    }
    
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc_O[8];
    for(int i = 0; i < 8; ++i) nvcuda::wmma::fill_fragment(acc_O[i], 0.0f);

    for (int j = 0; j < max_j; ++j) {
        int phase = j % 2;
        int next_phase = (j + 1) % 2;
        
        __nv_bfloat16* cur_K = &s_K_flat[phase * 16384];
        __nv_bfloat16* cur_V = &s_V_flat[phase * 16384];
        __nv_bfloat16* nxt_K = &s_K_flat[next_phase * 16384];
        __nv_bfloat16* nxt_V = &s_V_flat[next_phase * 16384];
        
        int k_blk_idx = j * 128;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_K[phase], 32768);
            tma_load_3d_fn(&tma_K, &mbar_K[phase], cur_K, 0, bh * S_len + k_blk_idx, 0);
            tma_load_3d_fn(&tma_K, &mbar_K[phase], cur_K + 8192, 64, bh * S_len + k_blk_idx, 0);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar_V[phase], 32768);
            tma_load_3d_fn(&tma_V, &mbar_V[phase], cur_V, 0, bh * S_len + k_blk_idx, 0);
            tma_load_3d_fn(&tma_V, &mbar_V[phase], cur_V + 8192, 64, bh * S_len + k_blk_idx, 0);
            
            if (j + 1 < max_j) {
                int next_k_blk_idx = (j + 1) * 128;
                mbarrier_arrive_and_expect_tx_fn(&mbar_K[next_phase], 32768);
                tma_load_3d_fn(&tma_K, &mbar_K[next_phase], nxt_K, 0, bh * S_len + next_k_blk_idx, 0);
                tma_load_3d_fn(&tma_K, &mbar_K[next_phase], nxt_K + 8192, 64, bh * S_len + next_k_blk_idx, 0);
                
                mbarrier_arrive_and_expect_tx_fn(&mbar_V[next_phase], 32768);
                tma_load_3d_fn(&tma_V, &mbar_V[next_phase], nxt_V, 0, bh * S_len + next_k_blk_idx, 0);
                tma_load_3d_fn(&tma_V, &mbar_V[next_phase], nxt_V + 8192, 64, bh * S_len + next_k_blk_idx, 0);
            }
        }
        
        __syncthreads(); 
        mbarrier_wait_fn(&mbar_K[phase], (j / 2) % 2);
        mbarrier_wait_fn(&mbar_V[phase], (j / 2) % 2);
        fence_proxy_async_fn();
        
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> Q_frag_arr[8];
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major> K_frag_arr[8];

        for (int k_blk = 0; k_blk < 8; ++k_blk) {
            load_warp_matrix_row_major(Q_frag_arr[k_blk], s_Q_flat, 0, k_blk * 16, 128);
            load_warp_matrix_col_major(K_frag_arr[k_blk], cur_K, k_blk * 16, 0, 128);
            
            for (int n_blk = 0; n_blk < 8; ++n_blk) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major> K_frag;
                load_warp_matrix_col_major(K_frag, cur_K, k_blk * 16, n_blk * 16, 128);
                nvcuda::wmma::mma_sync(acc_S[n_blk][k_blk], Q_frag_arr[k_blk], K_frag, false);
            }
        }
        
        float thread_max[8][8][8];
        float thread_sum[8][8][8];
        
        for(int i=0; i<8; ++i) {
            for(int j2=0; j2<8; ++j2) {
                for(int k=0; k<8; ++k) {
                    thread_max[i][j2][k] = -1e20f;
                    thread_sum[i][j2][k] = 0.0f;
                }
            }
        }
        
        for (int n_blk = 0; n_blk < 8; ++n_blk) {
            for (int k_blk = 0; k_blk < 8; ++k_blk) {
                float* val_arr_S = reinterpret_cast<float*>(&acc_S[n_blk][k_blk]);
                for (int idx = 0; idx < 8; ++idx) {
                    float val = val_arr_S[idx];
                    int i_warp = (idx / 2) % 4;
                    int j_warp = (idx % 2) + (n_blk * 2);
                    
                    int q_idx = q_blk + warp_id * 32 + i_warp * 4;
                    int k_idx = k_blk_idx + k_blk * 16 + j_warp * 4;
                    
                    if (k_idx >= S_len || k_idx > q_idx) val = -1e20f;
                    val /= sqrtf(128.0f);
                    val_arr_S[idx] = val;
                    
                    thread_max[n_blk][k_blk][idx] = fmaxf(thread_max[n_blk][k_blk][idx], val);
                }
            }
        }
        
        float max_val = -1e20f;
        for (int n_blk = 0; n_blk < 8; ++n_blk) {
            for (int k_blk = 0; k_blk < 8; ++k_blk) {
                for (int idx = 0; idx < 8; ++idx) {
                    max_val = fmaxf(max_val, thread_max[n_blk][k_blk][idx]);
                }
            }
        }
        
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, 16));
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, 8));
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, 4));
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, 2));
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, 1));
        
        curr_max = fmaxf(prev_max_val, max_val);
        float scale = expf(prev_max_val - curr_max);
        
        float sum_val = 0.0f;
        for (int n_blk = 0; n_blk < 8; ++n_blk) {
            for (int k_blk = 0; k_blk < 8; ++k_blk) {
                thread_sum[n_blk][k_blk][0] *= scale;
                for (int idx = 0; idx < 8; ++idx) {
                    float exp_scale = expf(thread_max[n_blk][k_blk][idx] - curr_max);
                    thread_sum[n_blk][k_blk][idx] *= exp_scale;
                    sum_val += thread_sum[n_blk][k_blk][idx] * exp_scale;
                }
            }
        }
        
        sum_val += __shfl_xor_sync(0xFFFFFFFF, sum_val, 16);
        sum_val += __shfl_xor_sync(0xFFFFFFFF, sum_val, 8);
        sum_val += __shfl_xor_sync(0xFFFFFFFF, sum_val, 4);
        sum_val += __shfl_xor_sync(0xFFFFFFFF, sum_val, 2);
        sum_val += __shfl_xor_sync(0xFFFFFFFF, sum_val, 1);
        
        curr_sum = prev_sum * scale + sum_val;
        
        for (int n_blk = 0; n_blk < 8; ++n_blk) {
            for (int k_blk = 0; k_blk < 8; ++k_blk) {
                float* val_arr_S = reinterpret_cast<float*>(&acc_S[n_blk][k_blk]);
                for (int idx = 0; idx < 8; ++idx) {
                    float val = val_arr_S[idx];
                    int i_warp = (idx / 2) % 4;
                    int j_warp = (idx % 2) + (n_blk * 2);
                    
                    int q_idx = q_blk + warp_id * 32 + i_warp * 4;
                    int k_idx = k_blk_idx + k_blk * 16 + j_warp * 4;
                    
                    float exp_val = 0.0f;
                    if (k_idx < S_len && k_idx <= q_idx) {
                        exp_val = expf(val - thread_max[n_blk][k_blk][idx]);
                    }
                    
                    int col_vec = (k_blk * 16 + j_warp * 4) / 8;
                    int row = warp_id * 32 + i_warp * 4;
                    int swizzled_col_vec = col_vec ^ (row % 8);
                    int swizzled_col = swizzled_col_vec * 8 + ((k_blk * 16 + j_warp * 4) % 8);
                    int swizzled_idx = row * 128 + swizzled_col;
                    
                    s_P_bf16[swizzled_idx] = __float2bfloat16(exp_val);
                }
            }
        }
        
        __syncthreads(); 
        
        for (int n_blk = 0; n_blk < 8; ++n_blk) {
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc_O_n;
            nvcuda::wmma::fill_fragment(acc_O_n, 0.0f);
            
            for (int k_blk = 0; k_blk < 8; ++k_blk) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> P_frag;
                load_warp_matrix_row_major(P_frag, s_P_bf16, 0, k_blk * 16, 128);
                
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> V_frag_n;
                load_warp_matrix_row_major(V_frag_n, cur_V, k_blk * 16, n_blk * 16, 128);
                
                nvcuda::wmma::mma_sync(acc_O_n, P_frag, V_frag_n, false);
            }
            acc_O[n_blk] = acc_O_n;
        }
        
        __syncthreads(); 
        
        curr_max = curr_max; // Retain updated max 
        prev_max_val = curr_max;
        prev_sum = curr_sum;
        float safe_final_sum = curr_sum < 1e-20f ? 1e-20f : curr_sum;
        float lse_val = curr_max + logf(safe_final_sum);
        
        if (q_blk + lane_id < S_len) {
            LSE_ptr[q_blk + lane_id] = lse_val;
        }
    }
    
    __syncthreads(); 
    
    // Utilize highly specialized optimized WGMMA `store` intrinsic modes for coalesced global writes.
    for(int i = 0; i < 8; ++i) {
        nvcuda::wmma::store_matrix_sync(&O_ptr[q_blk * 128 + i * 16384], acc_O[i], 128, true);
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t gmem_dim0, uint64_t gmem_dim1, uint64_t gmem_dim2, 
                                     uint32_t smem_dim0, uint32_t smem_dim1, uint32_t smem_dim2, 
                                     CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {gmem_dim0, gmem_dim1, gmem_dim2};
    cuuint64_t globalStrides[2] = {gmem_dim0 * 2, gmem_dim0 * gmem_dim1 * 2};
    cuuint32_t boxDim[3] = {smem_dim0, smem_dim1, smem_dim2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  
  int64_t B = Q.size(0);
  int64_t H = Q.size(1);
  int64_t S_len = Q.size(2);
  int64_t D = Q.size(3);
  
  if (D != 128) {
    fprintf(stderr, "Error: D must be 128\n");
    exit(1);
  }
  
  const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
  
  __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* LSE_data = static_cast<float*>(LSE.data_ptr());
  
  CUtensorMap tma_Q, tma_K, tma_V;
  
  CU_CHECK(create_tma_3d_descriptor_2B(
      &tma_Q, (void*)Q_data, 128, S_len, B * H, 64, 128, 1,
      CU_TENSOR_MAP_SWIZZLE_128B
  ));
  CU_CHECK(create_tma_3d_descriptor_2B(
      &tma_K, (void*)K_data, 128, S_len, B * H, 64, 128, 1,
      CU_TENSOR_MAP_SWIZZLE_128B
  ));
  CU_CHECK(create_tma_3d_descriptor_2B(
      &tma_V, (void*)V_data, 128, S_len, B * H, 64, 128, 1,
      CU_TENSOR_MAP_SWIZZLE_128B
  ));
  
  int threads = 128;
  dim3 grid(B, H, (S_len + 127) / 128);
  
  int smem_size = 197632; 
  cudaFuncSetAttribute(causal_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  
  causal_attention_kernel<<<grid, threads, smem_size, static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))>>>(
      tma_Q, tma_K, tma_V, O_data, LSE_data, S_len, B, H);
      
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_causal_attention::run);

} // namespace tvm_ffi_causal_attention