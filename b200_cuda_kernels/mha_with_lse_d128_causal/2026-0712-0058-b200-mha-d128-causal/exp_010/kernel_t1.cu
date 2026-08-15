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

__global__ __launch_bounds__(32, 1) void flash_attention_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    int B, int H, int S, int D) 
{
    int q_block = blockIdx.x;
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int global_q_base = (batch_idx * H + head_idx) * S + q_block * 128;

    extern __shared__ __align__(128) char smem_pool[];
    __nv_bfloat16 (*sq)[128] = (__nv_bfloat16 (*)[128])smem_pool;
    __nv_bfloat16 (*sk)[128] = (__nv_bfloat16 (*)[128])(smem_pool + 32768);
    __nv_bfloat16 (*sp)[128] = (__nv_bfloat16 (*)[128])(smem_pool + 32768); // Reuse sk
    __nv_bfloat16 (*sv)[128] = (__nv_bfloat16 (*)[128])(smem_pool + 65536);
    float (*s)[128] = (float (*)[128])(smem_pool + 98304);
    float (*so)[128] = (float (*)[128])(smem_pool + 163840);
    float *sm = (float*)(smem_pool + 229376);
    float *sl = (float*)(smem_pool + 229888);

    int tid = threadIdx.x;

    // Load Q into shared memory
    for (int i = tid; i < 128 * 128; i += 32) {
        int row = i / 128;
        int col = i % 128;
        int global_idx = global_q_base * D + col;
        if (q_block * 128 + row < S) {
            sq[row][col] = Q[global_idx];
        } else {
            sq[row][col] = __float2bfloat16(0.0f);
        }
    }

    // Initialize running max and sum
    if (tid < 128) {
        sm[tid] = -1e20f;
        sl[tid] = 0.0f;
    }

    // Initialize global output accumulation to zero
    for (int i = tid; i < 128 * 128; i += 32) {
        int row = i / 128;
        int col = i % 128;
        so[row][col] = 0.0f;
    }

    // Load valid K positions for the whole sequence into a small predicate array
    __shared__ bool valid_k_idxs[128];
    if (tid < 128) {
        valid_k_idxs[tid] = (blockIdx.x * 128 + tid < S);
    }
    __syncthreads();

    int my_q_base = tid * 4;

    float global_P_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float global_P_max[4] = {-1e20f, -1e20f, -1e20f, -1e20f};

    for (int k_block = 0; k_block <= q_block; ++k_block) {
        int global_k_base = (batch_idx * H + head_idx) * S + k_block * 128;

        // Load K and V for this block
        for (int i = tid; i < 128 * 128; i += 32) {
            int row = i / 128;
            int col = i % 128;
            int global_idx = global_k_base * D + col;
            if (k_block * 128 + row < S) {
                sk[row][col] = K[global_idx];
                sv[row][col] = V[global_idx];
            } else {
                sk[row][col] = __float2bfloat16(0.0f);
                sv[row][col] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // Compute Q @ K^T -> s
        for (int i = 0; i < 4; ++i) {
            for (int k = 0; k < 128; ++k) {
                s[my_q_base + i][k] = 0.0f;
            }
        }
        
        for (int k = 0; k < 128; ++k) {
            for (int i = 0; i < 4; ++i) {
                float sum = 0.0f;
                int q_idx = my_q_base + i;
                for (int d = 0; d < 128; d += 2) {
                    uint32_t q_packed = *(uint32_t*)&sq[q_idx][d];
                    uint32_t k_packed = *(uint32_t*)&sk[k][d];
                    uint16_t q0 = q_packed & 0xFFFF;
                    uint16_t q1 = (q_packed >> 16) & 0xFFFF;
                    uint16_t k0 = k_packed & 0xFFFF;
                    uint16_t k1 = (k_packed >> 16) & 0xFFFF;
                    sum += __bfloat162float(q0) * __bfloat162float(k0);
                    sum += __bfloat162float(q1) * __bfloat162float(k1);
                }
                s[q_idx][k] = sum;
            }
        }

        // Apply causal mask and scale factor
        float scale = 1.0f / sqrtf(D);
        float my_new_max[4] = {-1e20f, -1e20f, -1e20f, -1e20f};
        for (int i = 0; i < 4; ++i) {
            int q_idx = my_q_base + i;
            for (int k_idx = 0; k_idx < 128; ++k_idx) {
                int global_q_idx = q_block * 128 + q_idx;
                int global_k_idx = k_block * 128 + k_idx;
                if (global_k_idx > global_q_idx || !valid_k_idxs[k_idx]) {
                    s[q_idx][k_idx] = -1e20f;
                } else {
                    s[q_idx][k_idx] *= scale;
                }
                my_new_max[i] = fmaxf(my_new_max[i], s[q_idx][k_idx]);
            }
        }

        // Update global max and scale previous outputs
        float my_alpha[4];
        for (int i = 0; i < 4; ++i) {
            int q_idx = my_q_base + i;
            float new_global_max = fmaxf(my_new_max[i], global_P_max[i]);
            my_alpha[i] = expf(global_P_max[i] - new_global_max);
            
            // Soft guard against -inf early on (before any valid keys processed)
            if (new_global_max < -1e19f) {
                my_alpha[i] = 0.0f;
            }
            
            global_P_max[i] = new_global_max;
            
            // Rescale previously accumulated O
            for (int d = 0; d < 128; d += 2) {
                uint32_t val = *(uint32_t*)&so[q_idx][d];
                float o0 = __bfloat162float(val & 0xFFFF) * my_alpha[i];
                float o1 = __bfloat162float((val >> 16) & 0xFFFF) * my_alpha[i];
                *(uint32_t*)&so[q_idx][d] = ((uint32_t)__bfloat162float_to_uint(o1) << 16) | (uint32_t)__bfloat162float_to_uint(o0);
            }
        }

        // Compute P = exp(S - max), sum, and convert P to sp (bf16)
        float my_new_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = 0; i < 4; ++i) {
            int q_idx = my_q_base + i;
            for (int k_idx = 0; k_idx < 128; ++k_idx) {
                float p = expf(s[q_idx][k_idx] - global_P_max[i]);
                my_new_sum[i] += p;
                sp[q_idx][k_idx] = __float2bfloat16(p);
            }
        }

        // Broadcast sums and denom
        __shared__ float new_sum[128];
        __shared__ float denom[128];
        if (tid < 128) {
            new_sum[tid] = (k_block == q_block) ? my_new_sum[tid % 4] : 0.0f;
            denom[tid] = (k_block == q_block) ? (global_P_sum[tid % 4] + my_new_sum[tid % 4] * expf(my_new_max[tid % 4] - global_P_max[tid % 4])) : global_P_sum[tid % 4];
        }
        __syncthreads();
        
        if (tid < 128) {
            sl[tid] = denom[tid];
        }

        // Accumulate P @ V into so
        for (int i = 0; i < 4; ++i) {
            int q_idx = my_q_base + i;
            float my_beta = (k_block == q_block) ? expf(my_new_max[i] - global_P_max[i]) : 0.0f;
            for (int s_step = 0; s_step < 128; s_step++) {
                float p = __bfloat162float(sp[q_idx][s_step]) * my_beta;
                for (int d = 0; d < 128; d += 2) {
                    uint32_t v_packed = *(uint32_t*)&sv[s_step][d];
                    float v0 = __bfloat162float(v_packed & 0xFFFF);
                    float v1 = __bfloat162float((v_packed >> 16) & 0xFFFF);
                    
                    uint32_t so_packed = *(uint32_t*)&so[q_idx][d];
                    float o0 = __bfloat162float(so_packed & 0xFFFF) + p * v0;
                    float o1 = __bfloat162float((so_packed >> 16) & 0xFFFF) + p * v1;
                    *(uint32_t*)&so[q_idx][d] = ((uint32_t)__bfloat162float_to_uint(o1) << 16) | (uint32_t)__bfloat162float_to_uint(o0);
                }
            }
        }

        if (tid < 128) {
            global_P_sum[tid] = sl[tid];
        }
    }

    // Normalize O by global P sum and write to global memory
    for (int i = 0; i < 4; ++i) {
        int q_idx = my_q_base + i;
        float sum_i = global_P_sum[i];
        int global_q_idx = q_block * 128 + q_idx;
        if (global_q_idx < S) {
            for (int d = 0; d < 128; d += 2) {
                uint32_t o_packed = *(uint32_t*)&so[q_idx][d];
                float o0 = __bfloat162float(o_packed & 0xFFFF) / sum_i;
                float o1 = __bfloat162float((o_packed >> 16) & 0xFFFF) / sum_i;
                uint32_t out_packed = ((uint32_t)__bfloat162float_to_uint(o1) << 16) | (uint32_t)__bfloat162float_to_uint(o0);
                int out_idx = (batch_idx * H + head_idx) * S * D + global_q_idx * D + d;
                *(uint32_t*)&O[out_idx] = out_packed;
            }
        }
    }

    // Write LogSumExp to global memory
    if (tid < 128) {
        int global_q_idx = q_block * 128 + tid;
        if (global_q_idx < S) {
            float lse = global_P_max[tid % 4] + logf(global_P_sum[tid % 4]);
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
    dim3 block(32, 1, 1);

    // Request ~225KB of shared memory dynamically
    int smem_size = 225280; 
    
    // Pass pointers explicitly mapped over logical view layout [B, H, S, D]
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