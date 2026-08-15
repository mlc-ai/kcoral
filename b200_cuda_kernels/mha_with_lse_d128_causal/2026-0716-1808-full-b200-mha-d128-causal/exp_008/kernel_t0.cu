#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

namespace tvm_ffi_causal_attention {

using namespace nvcuda;

// -------------------------------------------------------------------------------------
// WMPA Fragment Loading and Storing Helpers
// -------------------------------------------------------------------------------------

__device__ __forceinline__ void load_warp_matrix_row_major(wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>& frag,
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

__device__ __forceinline__ void load_warp_matrix_col_major(wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>& frag,
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

__device__ __forceinline__ void store_warp_matrix(float* dst, const wmma::fragment<wmma::accumulator, 16, 16, 16, float>& frag,
                                                  int row_start, int col_start, int stride) {
    float* val_arr = reinterpret_cast<float*>(&frag);
    for (int idx = 0; idx < 8; ++idx) {
        dst[row_start * stride + col_start + idx] = val_arr[idx];
    }
}

// -------------------------------------------------------------------------------------
// Causal Attention Kernel
// -------------------------------------------------------------------------------------

__global__ void causal_attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S_len, int B, int H)
{
    extern __shared__ char smem_raw[];
    size_t smem_addr = (size_t)smem_raw;
    smem_addr = (smem_addr + 1023) & ~1023; // Align to 1024 bytes
    char* smem = (char*)smem_addr;

    __nv_bfloat16* s_Q_flat = (__nv_bfloat16*)smem;                   // 4096 Bytes
    __nv_bfloat16* s_K_flat = (__nv_bfloat16*)(smem + 4096);          // 65536 Bytes
    __nv_bfloat16* s_V_flat = (__nv_bfloat16*)(smem + 69632);         // 65536 Bytes
    float* s_S = (float*)(smem + 135168);                             // 8192 Bytes
    __nv_bfloat16* s_P_bf16 = (__nv_bfloat16*)(smem + 143360);        // 4096 Bytes
    float* s_row_max = (float*)(smem + 147456);                       // 64 Bytes
    float* s_row_sum = (float*)(smem + 147520);                       // 64 Bytes
    float* s_prev_max = (float*)(smem + 147584);                      // 64 Bytes

    int bh = ((blockIdx.x * blockDim.x) + threadIdx.x) / (blockDim.x * 16);
    int bid = blockIdx.x % ((S_len + 15) / 16);
    int lane_id = threadIdx.x % 32;
    int batch_head_idx = bh % (B * H);
    
    const __nv_bfloat16* Q_ptr = Q + batch_head_idx * S_len * 128;
    const __nv_bfloat16* K_ptr = K + batch_head_idx * S_len * 128;
    const __nv_bfloat16* V_ptr = V + batch_head_idx * S_len * 128;
    
    __nv_bfloat16* O_ptr = O + batch_head_idx * S_len * 128;
    float* LSE_ptr = LSE + batch_head_idx * S_len;
    
    int BM = 16;
    int num_blocks = (S_len + 127) / 128;
    int q_blk = bid * BM;
    
    int elem_per_load = 8;
    int total_Q_elems = 16 * 128;
    int loads_per_thread_Q = (total_Q_elems + elem_per_load * 128 - 1) / (elem_per_load * 128);
    
    for (int load_idx = 0; load_idx < loads_per_thread_Q; ++load_idx) {
        int idx = load_idx * 128 + lane_id;
        int row = idx / 16;
        int col = (idx % 16) * 8;
        
        float4 tmp = {0,0,0,0};
        if (q_blk + row < S_len) {
            tmp = *(const float4*)(Q_ptr + (q_blk + row) * 128 + col);
        }
        *(float4*)(&s_Q_flat[row * 128 + col]) = tmp;
    }
    
    __nv_bfloat16* O_frag_ptr = (__nv_bfloat16*)malloc(16 * 128 * sizeof(__nv_bfloat16));
    for(int i=0; i<16*128; ++i) O_frag_ptr[i] = __float2bfloat16(0.0f);

    float lse_val = -1e20f;
    
    float prev_max = -1e20f;
    float curr_max = -1e20f;
    float curr_sum = 0.0f;
    
    for (int j = 0; j < num_blocks; ++j) {
        int k_blk_idx = j * 128;
        
        int phase = j % 2;
        int next_phase = (j + 1) % 2;
        __nv_bfloat16* cur_K = &s_K_flat[phase * 128 * 128];
        __nv_bfloat16* cur_V = &s_V_flat[phase * 128 * 128];
        __nv_bfloat16* nxt_K = &s_K_flat[next_phase * 128 * 128];
        __nv_bfloat16* nxt_V = &s_V_flat[next_phase * 128 * 128];
        
        int total_KV_elems = 128 * 128;
        int loads_per_thread_KV = (total_KV_elems + elem_per_load * 128 - 1) / (elem_per_load * 128);
        
        for (int load_idx = 0; load_idx < loads_per_thread_KV; ++load_idx) {
            int idx = load_idx * 128 + lane_id;
            int row = idx / 16;
            int col = (idx % 16) * 8;
            
            float4 tmp_K = {0,0,0,0};
            if (k_blk_idx + row < S_len) {
                tmp_K = *(const float4*)(K_ptr + (k_blk_idx + row) * 128 + col);
            }
            *(float4*)(&cur_K[row * 128 + col]) = tmp_K;
            
            float4 tmp_V = {0,0,0,0};
            if (k_blk_idx + row < S_len) {
                tmp_V = *(const float4*)(V_ptr + (k_blk_idx + row) * 128 + col);
            }
            *(float4*)(&cur_V[row * 128 + col]) = tmp_V;
            
            if (j + 1 < num_blocks) {
                int next_k_blk_idx = (j + 1) * 128;
                float4 tmp_nxt_K = {0,0,0,0};
                if (next_k_blk_idx + row < S_len) {
                    tmp_nxt_K = *(const float4*)(K_ptr + (next_k_blk_idx + row) * 128 + col);
                }
                *(float4*)(&nxt_K[row * 128 + col]) = tmp_nxt_K;
                
                float4 tmp_nxt_V = {0,0,0,0};
                if (next_k_blk_idx + row < S_len) {
                    tmp_nxt_V = *(const float4*)(V_ptr + (next_k_blk_idx + row) * 128 + col);
                }
                *(float4*)(&nxt_V[row * 128 + col]) = tmp_nxt_V;
            }
        }
        
        __syncthreads();
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_S[8];
        for (int i = 0; i < 8; ++i) wmma::fill_fragment(acc_S[i], 0.0f);
        
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> Q_frag_arr[8];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> K_frag_arr[8];

        for (int k_blk = 0; k_blk < 8; ++k_blk) {
            load_warp_matrix_row_major(Q_frag_arr[k_blk], s_Q_flat, 0, k_blk * 16, 128);
            load_warp_matrix_col_major(K_frag_arr[k_blk], cur_K, k_blk * 16, 0, 128);
            
            for (int n_blk = 0; n_blk < 8; ++n_blk) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> K_frag;
                load_warp_matrix_col_major(K_frag, cur_K, k_blk * 16, n_blk * 16, 128);
                wmma::mma_sync(acc_S[n_blk], Q_frag_arr[k_blk], K_frag, false);
            }
        }
        
        float local_max = -1e20f;
        for (int n_blk = 0; n_blk < 8; ++n_blk) {
            float* val_arr = reinterpret_cast<float*>(&acc_S[n_blk]);
            for (int idx = 0; idx < 8; ++idx) {
                float val = val_arr[idx];
                int i_warp = (idx / 2) % 4; // Row within block
                int j_warp = (idx % 2) + (n_blk * 2); // Col within block
                
                int q_idx = q_blk + i_warp * 4 + lane_id / 8; // Global query seq pos
                int k_idx = k_blk_idx + j_warp * 8 + (lane_id % 8); // Global key seq pos
                
                if (k_idx >= S_len || k_idx > q_idx) {
                    val = -1e20f;
                }
                val = val / sqrtf(128.0f); // scale by sqrt(d)
                local_max = fmaxf(local_max, val);
                val_arr[idx] = val;
            }
        }
        
        for (int offset = 16; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
        }
        
        curr_max = fmaxf(prev_max, local_max);
        curr_sum = 0.0f;
        
        float scale = expf(prev_max - curr_max);
        
        for (int n_blk = 0; n_blk < 8; ++n_blk) {
            float* val_arr_O = reinterpret_cast<float*>(&acc_O[n_blk]);
            for (int idx = 0; idx < 8; ++idx) {
                val_arr_O[idx] *= scale;
            }
        }
        
        if (lane_id == 0) {
            s_row_max[0] = curr_max;
            s_row_sum[0] = curr_sum * scale;
            s_prev_max[0] = curr_max;
        }
        
        for (int n_blk = 0; n_blk < 8; ++n_blk) {
            float* val_arr = reinterpret_cast<float*>(&acc_S[n_blk]);
            for (int idx = 0; idx < 8; ++idx) {
                float val = val_arr[idx];
                int i_warp = (idx / 2) % 4;
                int j_warp = (idx % 2) + (n_blk * 2);
                
                int q_idx = q_blk + i_warp * 4 + lane_id / 8;
                int k_idx = k_blk_idx + j_warp * 8 + (lane_id % 8);
                
                if (k_idx >= S_len || k_idx > q_idx) {
                    val = -1e20f;
                }
                
                float exp_val = 0.0f;
                if (val != -1e20f) {
                    exp_val = expf(val - curr_max);
                }
                curr_sum += exp_val;
                
                s_S[lane_id * 128 + n_blk * 16 + idx] = exp_val;
            }
        }
        
        curr_sum += __shfl_xor_sync(0xFFFFFFFF, curr_sum, 16);
        curr_sum += __shfl_xor_sync(0xFFFFFFFF, curr_sum, 8);
        curr_sum += __shfl_xor_sync(0xFFFFFFFF, curr_sum, 4);
        curr_sum += __shfl_xor_sync(0xFFFFFFFF, curr_sum, 2);
        curr_sum += __shfl_xor_sync(0xFFFFFFFF, curr_sum, 1);
        
        if (lane_id == 0) {
            s_row_sum[0] = curr_sum;
        }
        
        __syncthreads();
        
        float final_sum = s_row_sum[0];
        float safe_final_sum = final_sum < 1e-20f ? 1e-20f : final_sum;
        
        for(int i = 0; i < 16*128; ++i) {
            float val = s_S[i];
            __nv_bfloat16 out_val = __float2bfloat16(val / safe_final_sum);
            s_P_bf16[i] = out_val;
        }
        
        __syncthreads();
        
        for (int k_blk = 0; k_blk < 8; ++k_blk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> P_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> V_frag;

            load_warp_matrix_row_major(P_frag, (__nv_bfloat16*)s_P_bf16, 0, k_blk * 16, 128);
            load_warp_matrix_row_major(V_frag, cur_V, k_blk * 16, 0, 128);
            
            for (int n_blk = 0; n_blk < 8; ++n_blk) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> V_frag_n;
                load_warp_matrix_row_major(V_frag_n, cur_V, k_blk * 16, n_blk * 16, 128);
                
                wmma::mma_sync(acc_O[n_blk], P_frag, V_frag_n, false);
            }
        }
        
        prev_max = curr_max;
        
        lse_val = curr_max + logf(safe_final_sum);
        
        __syncthreads();
    } 
    
    free(O_frag_ptr);
    
    for (int i = 0; i < 4; ++i) {
        int row_idx = bid * 4 + i;
        int global_q_row = row_idx * 128 + i; 
        
        if (global_q_row < S_len) {
            LSE_ptr[global_q_row] = lse_val;
        }
        
        for (int col_blk = 0; col_blk < 16; col_blk++) {
            int global_col = col_blk * 8 + (threadIdx.x % 8);
            int global_q_row_safe = global_q_row;
            
            if (global_q_row_safe < S_len && global_col < 128) {
                O_ptr[global_q_row_safe * 128 + global_col] = O_frag_ptr[(i * 16 + col_blk) * 128 + global_col];
            }
        }
    }
    
    __syncthreads();
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
  
  int threads = 128;
  int num_blocks = (S_len + 127) / 128;
  dim3 grid(((16 * num_blocks) + 3) / 4);
  
  int smem_size = 147648; 
  
  cudaFuncSetAttribute(causal_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  
  CUDA_CHECK(cudaMallocAsync(&Q_data, B * H * S_len * D * sizeof(__nv_bfloat16), static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
  CUDA_CHECK(cudaMallocAsync(&K_data, B * H * S_len * D * sizeof(__nv_bfloat16), static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
  CUDA_CHECK(cudaMallocAsync(&V_data, B * H * S_len * D * sizeof(__nv_bfloat16), static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
  CUDA_CHECK(cudaMallocAsync(&O_data, B * H * S_len * D * sizeof(__nv_bfloat16), static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
  CUDA_CHECK(cudaMallocAsync(&LSE_data, B * H * S_len * sizeof(float), static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
  
  causal_attention_kernel<<<grid, threads, smem_size, static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))>>>(
      Q_data, K_data, V_data, O_data, LSE_data, S_len, B, H);
      
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
  
  CUDA_CHECK(cudaFreeAsync(Q_data, static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
  CUDA_CHECK(cudaFreeAsync(K_data, static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
  CUDA_CHECK(cudaFreeAsync(V_data, static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
  CUDA_CHECK(cudaFreeAsync(O_data, static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
  CUDA_CHECK(cudaFreeAsync(LSE_data, static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_causal_attention::run);

} // namespace tvm_ffi_causal_attention