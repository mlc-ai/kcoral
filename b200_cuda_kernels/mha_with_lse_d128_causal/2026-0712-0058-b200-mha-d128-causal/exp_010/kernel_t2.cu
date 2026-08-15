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

using namespace nvcuda;

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_dec_sync_fn() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__global__ __launch_bounds__(128, 1) void flash_attention_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    int B, int H, int S, int D) 
{
    setmaxnreg_dec_sync_fn<128>();
    
    int q_block = blockIdx.x;
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int global_q_base = (batch_idx * H + head_idx) * S + q_block * 128;

    extern __shared__ __align__(128) char smem_pool[];
    
    struct BFloat16Array { __nv_bfloat16 data[128*128]; };
    struct FloatArray { float data[128*128]; };
    
    BFloat16Array* sq = (BFloat16Array*)smem_pool;                
    BFloat16Array* sk = (BFloat16Array*)(smem_pool + 32768);      
    BFloat16Array* sv = (BFloat16Array*)(smem_pool + 65536);      
    FloatArray* sp = (FloatArray*)(smem_pool + 98304);            
    FloatArray* so = (FloatArray*)(smem_pool + 163840);           
    float *sm = (float*)(smem_pool + 229376);                     
    float *sl = (float*)(smem_pool + 229888);                     

    int tid = threadIdx.x;
    int step = tid / 16;
    int step_idx = tid % 16;

    for (int i = tid; i < 128 * 128 / 16; i += 128) {
        int row = i / (128 / 16);
        int col_vec = (i % (128 / 16)) * 16;
        int global_idx = global_q_base * D + col_vec;
        if (q_block * 128 + row < S) {
            *reinterpret_cast<float4*>(&sq->data[row * 128 + col_vec]) = *reinterpret_cast<const float4*>(&Q[global_idx]);
        } else {
            *reinterpret_cast<float4*>(&sq->data[row * 128 + col_vec]) = make_float4(0, 0, 0, 0);
        }
    }

    if (tid < 128) {
        sm[tid] = -1e20f;
        sl[tid] = 0.0f;
    }

    float global_P_sum = 0.0f;
    float global_P_max = -1e20f;
    float scale = 1.0f / sqrtf(D);

    // Ensure so is tracked in registers for efficient accumulation
    float so_regs[128]; 
    #pragma unroll 1
    for(int i = 0; i < 128; ++i) so_regs[i] = 0.0f;

    for (int k_block = 0; k_block <= q_block; ++k_block) {
        int global_k_base = (batch_idx * H + head_idx) * S + k_block * 128;

        for (int i = tid; i < 128 * 128 / 16; i += 128) {
            int row = i / (128 / 16);
            int col_vec = (i % (128 / 16)) * 16;
            int global_idx = global_k_base * D + col_vec;
            if (k_block * 128 + row < S) {
                *reinterpret_cast<float4*>(&sk->data[row * 128 + col_vec]) = *reinterpret_cast<const float4*>(&K[global_idx]);
                *reinterpret_cast<float4*>(&sv->data[row * 128 + col_vec]) = *reinterpret_cast<const float4*>(&V[global_idx]);
            } else {
                *reinterpret_cast<float4*>(&sk->data[row * 128 + col_vec]) = make_float4(0, 0, 0, 0);
                *reinterpret_cast<float4*>(&sv->data[row * 128 + col_vec]) = make_float4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        float my_new_max[2] = {-1e20f, -1e20f};

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_P[2];
        wmma::fill_fragment(c_P[0]);
        wmma::fill_fragment(c_P[1]);
        
        wmma::fragment<wmma::matrix_a, 16, 16, 16, float, wmma::row_major> a_Q[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, float, wmma::col_major> b_K[2];
        for(int i = 0; i < 2; ++i) {
            wmma::fill_fragment(a_Q[i]);
            wmma::fill_fragment(b_K[i]);
        }

        for(int k_chunk_step = 0; k_chunk_step < 128; k_chunk_step += 16) {
            int row_start = step * 16;
            int offset = row_start * 128 + k_chunk_step;
            wmma::load_matrix_sync(a_Q[step], sq->data + offset);
            wmma::load_matrix_sync(b_K[step], sk->data + offset);
            
            offset = row_start * 128 + k_chunk_step + (step * 2); 
            wmma::load_matrix_sync(a_Q[1-step], sq->data + offset);
            wmma::load_matrix_sync(b_K[1-step], sk->data + offset);
            
            wmma::mma_sync(c_P[step], a_Q[step], b_K[step]);
            wmma::mma_sync(c_P[1-step], a_Q[1-step], b_K[1-step]);
        }

        float p_vals[16];
        for(int i = 0; i < 2; ++i) {
            int row_start = step * 16 + (i * 16);
            wmma::store_matrix_sync(sp->data + row_start * 128, c_P[i], 128, wmma::mem_row_major);
            
            for(int col = 0; col < 16; ++col) {
                wmma::store_matrix_sync(&p_vals[col], c_P[i], 16, wmma::mem_row_major);
                int tid_col = step_idx * 16 + col;
                int global_q_idx = q_block * 128 + row_start;
                int global_k_idx = k_block * 128 + tid_col;
                
                if (global_k_idx > global_q_idx || global_k_idx >= S) {
                    p_vals[col] = -1e20f;
                } else {
                    p_vals[col] *= scale;
                }
                my_new_max[i] = fmaxf(my_new_max[i], p_vals[col]);
                sp->data[row_start * 128 + tid_col] = p_vals[col];
            }
        }

        float my_alpha[2];
        for(int i = 0; i < 2; ++i) {
            int row_start = step * 16 + (i * 16);
            float new_global_max = fmaxf(my_new_max[i], sm[row_start]);
            my_alpha[i] = expf(sm[row_start] - new_global_max);
            
            if (new_global_max < -1e19f) {
                my_alpha[i] = 0.0f;
            }
            
            sm[row_start] = new_global_max; // Eagerly update SMEM to guarantee synchronization boundaries properly
            global_P_max = fmaxf(global_P_max, new_global_max);
            
            float sum_factor = 0.0f;
            for(int col = 0; col < 16; ++col) {
                int tid_col = step_idx * 16 + col;
                float p = sp->data[row_start * 128 + tid_col];
                float ep = expf(p - new_global_max);
                sum_factor += ep;
                
                float my_beta = expf(my_new_max[i] - new_global_max);
                float p_scaled = ep * my_beta;
                sp->data[row_start * 128 + tid_col] = p_scaled; 
            }
            global_P_sum = global_P_sum * my_alpha[i] + sum_factor;
        }

        for(int i = 0; i < 2; ++i) {
            int row_start = step * 16 + (i * 16);
            #pragma unroll 1
            for(int d = 0; d < 128; ++d) {
                so_regs[d] *= my_alpha[i];
            }
        }

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_O[2];
        wmma::fill_fragment(c_O[0]);
        wmma::fill_fragment(c_O[1]);
        
        wmma::fragment<wmma::matrix_a, 16, 16, 16, float, wmma::row_major> a_P[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, float, wmma::col_major> b_V[2];
        for(int i = 0; i < 2; ++i) {
            wmma::fill_fragment(a_P[i]);
            wmma::fill_fragment(b_V[i]);
        }

        for(int k_chunk_step = 0; k_chunk_step < 128; k_chunk_step += 16) {
            int row_start = step * 16;
            int offset = row_start * 128 + k_chunk_step;
            wmma::load_matrix_sync(a_P[step], sp->data + offset, 128, wmma::mem_row_major);
            wmma::load_matrix_sync(b_V[step], sv->data + offset, 128, wmma::mem_col_major);
            
            offset = row_start * 128 + k_chunk_step + (step * 2);
            wmma::load_matrix_sync(a_P[1-step], sp->data + offset, 128, wmma::mem_row_major);
            wmma::load_matrix_sync(b_V[1-step], sv->data + offset, 128, wmma::mem_col_major);
            
            wmma::mma_sync(c_O[step], a_P[step], b_V[step]);
            wmma::mma_sync(c_O[1-step], a_P[1-step], b_V[1-step]);
        }

        for(int i = 0; i < 2; ++i) {
            int row_start = step * 16 + (i * 16);
            wmma::store_matrix_sync(so->data + row_start * 128, c_O[i], 128, wmma::mem_row_major);
        }
    }

    for (int i = tid; i < 128 * 128 / 16; i += 128) {
        int row = i / (128 / 16);
        int col_vec = (i % (128 / 16)) * 16;
        int global_q_idx = q_block * 128 + row;
        if (global_q_idx < S) {
            float sum_i = sl[row];
            float4 o_vec = *reinterpret_cast<float4*>(&so->data[row * 128 + col_vec]);
            __nv_bfloat16 o0 = __float2bfloat16(o_vec.x / sum_i);
            __nv_bfloat16 o1 = __float2bfloat16(o_vec.y / sum_i);
            __nv_bfloat16 o2 = __float2bfloat16(o_vec.z / sum_i);
            __nv_bfloat16 o3 = __float2bfloat16(o_vec.w / sum_i);
            
            uint4 out_vec;
            uint32_t* p = (uint32_t*)&out_vec;
            p[0] = ((uint16_t)*(uint16_t*)&o1 << 16) | *(uint16_t*)&o0;
            p[1] = ((uint16_t)*(uint16_t*)&o3 << 16) | *(uint16_t*)&o2;
            
            int out_idx = (batch_idx * H + head_idx) * S * D + global_q_idx * D + col_vec;
            *(uint4*)&O[out_idx] = out_vec;
        }
    }

    if (tid < 128) {
        int global_q_idx = q_block * 128 + tid;
        if (global_q_idx < S) {
            float lse = sm[tid] + logf(sl[tid]);
            int lse_idx = (batch_idx * H + head_idx) * S + global_q_idx;
            LSE[lse_idx] = lse;
        }
    }
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int num_blocks_x = (S + 127) / 128;
    dim3 grid(num_blocks_x, H, B);
    dim3 block(128, 1, 1);

    int smem_size = 232448; 
    
    CUDA_CHECK(cudaFuncSetAttribute(flash_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    flash_attention_kernel<<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        B, H, S, D
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_example_cuda