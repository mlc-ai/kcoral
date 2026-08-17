#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <math.h>
#include <stdio.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

using namespace nvcuda;

__global__ void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int seq_len, int D
) {
    extern __shared__ char dynamic_smem[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)dynamic_smem; // 16KB (64x128)
    __nv_bfloat16* smem_K = smem_Q + 64 * 128;            // 16KB (64x128)
    __nv_bfloat16* smem_V = smem_K + 64 * 128;            // 16KB (64x128)
    float* smem_O = (float*)(smem_V + 64 * 128);          // 32KB (64x128)
    float* smem_S = smem_O + 64 * 128;                    // 16KB (64x64)
    float* smem_O_new = smem_S + 64 * 64;                 // 32KB (64x128)
    float* smem_l = smem_O_new + 64 * 128;                // 256B (64)
    float* smem_m = smem_l + 64;                          // 256B (64)

    int64_t b = blockIdx.z;
    int64_t h = blockIdx.y;
    int64_t row_start = blockIdx.x * 64;

    int64_t H_64 = H;
    int64_t seq_len_64 = seq_len;
    int64_t batch_head_offset = b * H_64 * seq_len_64 * 128 + h * seq_len_64 * 128;
    
    const __nv_bfloat16* Q_ptr = Q + batch_head_offset + row_start * 128;
    const __nv_bfloat16* K_base = K + batch_head_offset;
    const __nv_bfloat16* V_base = V + batch_head_offset;
    __nv_bfloat16* O_ptr = O + batch_head_offset + row_start * 128;
    
    int64_t lse_offset = b * H_64 * seq_len_64 + h * seq_len_64 + row_start;
    float* LSE_ptr = LSE + lse_offset;

    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;

    // initialize smem_O
    for (int i = 0; i < 16; ++i) {
        int idx = tid + i * 128;
        ((float4*)smem_O)[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float my_m = -INFINITY;
    float my_l = 0.0f;

    // load Q
    for (int i = 0; i < 8; ++i) { 
        int idx = tid + i * 128; 
        int r = idx / 16; 
        if (row_start + r < seq_len) {
            ((float4*)smem_Q)[idx] = ((const float4*)Q_ptr)[idx];
        } else {
            ((float4*)smem_Q)[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    for (int kv_start = 0; kv_start < seq_len; kv_start += 64) {
        // load K
        for (int i = 0; i < 8; ++i) { 
            int idx = tid + i * 128; 
            int r = idx / 16; 
            if (kv_start + r < seq_len) {
                ((float4*)smem_K)[idx] = ((const float4*)(K_base + kv_start * 128))[idx];
            } else {
                ((float4*)smem_K)[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        // load and transpose V to prepare for `.col_major` MMA load
        for (int i = 0; i < 8; ++i) { 
            int idx = tid + i * 128; 
            int r = idx / 16; 
            int c = (idx % 16) * 8; 
            if (kv_start + r < seq_len) {
                float4 v_val = ((const float4*)(V_base + kv_start * 128))[idx];
                __nv_bfloat16* v_bf = (__nv_bfloat16*)&v_val;
                for (int k = 0; k < 8; ++k) {
                    smem_V[(c + k) * 64 + r] = v_bf[k];
                }
            } else {
                for (int k = 0; k < 8; ++k) {
                    smem_V[(c + k) * 64 + r] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        // Q @ K^T
        for (int j = 0; j < 4; ++j) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int k = 0; k < 8; ++k) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, smem_Q + wid * 16 * 128 + k * 16, 128);
                wmma::load_matrix_sync(b_frag, smem_K + j * 16 * 128 + k * 16, 128);
                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }
            // Apply scale: 1 / sqrt(128)
            for (int t = 0; t < acc.num_elements; t++) acc.x[t] *= 0.0883883476f;
            wmma::store_matrix_sync(smem_S + wid * 16 * 64 + j * 16, acc, 64, wmma::mem_row_major);
        }
        __syncwarp();

        // row max, exp, scale O, and write P inline to smem_S
        if (lane < 16) {
            int r = wid * 16 + lane;
            float row_max = -INFINITY;
            for (int c = 0; c < 64; ++c) {
                if (kv_start + c < seq_len) {
                    row_max = fmaxf(row_max, smem_S[r * 64 + c]);
                }
            }
            float m_new = fmaxf(my_m, row_max);
            float exp_diff = expf(my_m - m_new);
            my_m = m_new;
            
            float row_sum = 0.0f;
            for (int c = 0; c < 64; ++c) {
                float p = 0.0f;
                if (kv_start + c < seq_len) {
                    p = expf(smem_S[r * 64 + c] - m_new);
                }
                row_sum += p;
                ((__nv_bfloat16*)smem_S)[r * 64 + c] = __float2bfloat16(p);
            }
            
            my_l = my_l * exp_diff + row_sum;
            
            for (int c = 0; c < 128; ++c) {
                smem_O[r * 128 + c] *= exp_diff;
            }
        }
        __syncthreads();

        // P @ V
        for (int j = 0; j < 8; ++j) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int k = 0; k < 4; ++k) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, ((__nv_bfloat16*)smem_S) + wid * 16 * 64 + k * 16, 64);
                wmma::load_matrix_sync(b_frag, smem_V + j * 16 * 64 + k * 16, 64);
                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }
            wmma::store_matrix_sync(smem_O_new + wid * 16 * 128 + j * 16, acc, 128, wmma::mem_row_major);
        }
        __syncthreads();

        // Accumulate O_new into O
        for (int i = 0; i < 16; ++i) {
            int idx = tid + i * 128;
            float4 o_val = ((float4*)smem_O)[idx];
            float4 new_val = ((float4*)smem_O_new)[idx];
            o_val.x += new_val.x;
            o_val.y += new_val.y;
            o_val.z += new_val.z;
            o_val.w += new_val.w;
            ((float4*)smem_O)[idx] = o_val;
        }
        __syncthreads();
    }

    // Export m and l metrics to shared memory for fully vectorized writing
    if (lane < 16) {
        int r = wid * 16 + lane;
        smem_l[r] = my_l;
        smem_m[r] = my_m;
    }
    __syncthreads();

    // Final normalization and write to O (bf16 output)
    for (int i = 0; i < 16; ++i) {
        int idx = tid + i * 128;
        int r = idx / 32;
        int c = (idx % 32) * 4;
        if (row_start + r < seq_len) {
            float l_val = smem_l[r];
            float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
            float4 o_val = ((float4*)smem_O)[idx];
            
            __nv_bfloat162 b01 = __floats2bfloat162_rn(o_val.x * inv_l, o_val.y * inv_l);
            __nv_bfloat162 b23 = __floats2bfloat162_rn(o_val.z * inv_l, o_val.w * inv_l);
            
            ((__nv_bfloat162*)(O_ptr + r * 128 + c))[0] = b01;
            ((__nv_bfloat162*)(O_ptr + r * 128 + c))[1] = b23;
        }
    }

    // Write final LSE
    if (tid < 64) {
        if (row_start + tid < seq_len) {
            LSE_ptr[tid] = smem_m[tid] + logf(smem_l[tid]);
        }
    }
}

namespace tvm_ffi_mha {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);

    const __nv_bfloat16* q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_data = static_cast<float*>(LSE.data_ptr());

    dim3 block(128);
    dim3 grid((S + 63) / 64, H, B);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    int dynamic_smem = 131584; // exactly 128 KB + 512 bytes

    CUDA_CHECK(cudaFuncSetAttribute(
        mha_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_smem
    ));

    mha_kernel<<<grid, block, dynamic_smem, stream>>>(
        q_data, k_data, v_data, o_data, lse_data,
        B, H, S, 128
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_mha