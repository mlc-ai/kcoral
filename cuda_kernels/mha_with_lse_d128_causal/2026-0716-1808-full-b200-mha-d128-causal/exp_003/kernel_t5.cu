#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

namespace tvm_ffi_fmha {

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void cp_async_128(void* smem, const void* gmem) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" 
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__global__ __launch_bounds__(128, 1)
void fmha_4_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    int64_t S, int64_t D)
{
    int row_block = blockIdx.x;
    int bh = blockIdx.y;
    int tid = threadIdx.x;

    setmaxnreg_inc_sync_fn<248>();

    __nv_bfloat16* ptr_O_bh = O + bh * S * D;
    float* ptr_LSE_bh = LSE + bh * S;

    int s_start = row_block * 128;

    extern __shared__ char smem_pool[];
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_pool);
    uint32_t align_offset = (16 - (smem_addr % 16)) % 16;
    char* aligned_smem = smem_pool + align_offset;

    __nv_bfloat16* Q_smem = (__nv_bfloat16*)aligned_smem;
    __nv_bfloat16* K_smem = Q_smem + 128 * 128;
    __nv_bfloat16* V_smem_T = K_smem + 128 * 128;
    float* O_smem_f32 = (float*)(V_smem_T + 128 * 128);
    __nv_bfloat16* O_smem_bf16 = (__nv_bfloat16*)(O_smem_f32 + 128 * 128);
    __nv_bfloat16* P_smem = O_smem_bf16;
    
    float* m_prev_row = (float*)(P_smem + 128 * 128);
    float* l_prev_row = m_prev_row + 128;

    for (int i = tid; i < 128; i += 128) {
        m_prev_row[i] = -1e20f;
        l_prev_row[i] = 0.0f;
    }
    for (int i = tid; i < 128 * 128; i += 128) {
        O_smem_f32[i] = 0.0f;
    }
    __syncthreads();

    const __nv_bfloat16* ptr_Q_bh = Q + bh * S * D;
    for(int i = tid; i < 128 * 16; i += 128) {
        int row = i / 16;
        int col = (i % 16) * 8;
        float4 val;
        if (s_start + row < S) {
            val = *reinterpret_cast<float4*>(&ptr_Q_bh[(s_start + row) * D + col]);
        } else {
            val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        *reinterpret_cast<float4*>(&Q_smem[row * 128 + col]) = val;
    }
    __syncthreads();

    auto wgmma = [&](float (&m_out)[16 * 8], const __nv_bfloat16 (&regs_a)[16], const __nv_bfloat16 (&regs_b)[16]) {
        asm volatile("wgmma.mma.sync.aligned.p1.m16n8k16.float32 accumulator {%0-%15}, input_a{%16-%31}, input_b{%32-%47};" 
            : : "r"(m_out[0]), "r"(m_out[1]), "r"(m_out[2]), "r"(m_out[3]),
              "r"(m_out[4]), "r"(m_out[5]), "r"(m_out[6]), "r"(m_out[7]),
              "r"(m_out[8]), "r"(m_out[9]), "r"(m_out[10]), "r"(m_out[11]),
              "r"(m_out[12]), "r"(m_out[13]), "r"(m_out[14]), "r"(m_out[15]),
              "r"(regs_a[0]), "r"(regs_a[1]), "r"(regs_a[2]), "r"(regs_a[3]),
              "r"(regs_a[4]), "r"(regs_a[5]), "r"(regs_a[6]), "r"(regs_a[7]),
              "r"(regs_a[8]), "r"(regs_a[9]), "r"(regs_a[10]), "r"(regs_a[11]),
              "r"(regs_a[12]), "r"(regs_a[13]), "r"(regs_a[14]), "r"(regs_a[15]),
              "r"(regs_b[0]), "r"(regs_b[1]), "r"(regs_b[2]), "r"(regs_b[3]),
              "r"(regs_b[4]), "r"(regs_b[5]), "r"(regs_b[6]), "r"(regs_b[7]),
              "r"(regs_b[8]), "r"(regs_b[9]), "r"(regs_b[10]), "r"(regs_b[11]),
              "r"(regs_b[12]), "r"(regs_b[13]), "r"(regs_b[14]), "r"(regs_b[15])
            : "memory");
    };

    float scale = 1.0f / sqrtf((float)D);

    const __nv_bfloat16* ptr_K_bh = K + bh * S * D;
    const __nv_bfloat16* ptr_V_bh = V + bh * S * D;

    float m_out_pv[16 * 8];
    __sync_wgmma_accumulate_m16n8k16_fp32(m_out_pv, 0.0f);

    for (int j = 0; j <= row_block; ++j) {
        int kv_start = j * 128;

        for(int i = tid; i < 128 * 16; i += 128) {
            int row = i / 16;
            int col = (i % 16) * 8;
            float4 k_val, v_val;
            if (kv_start + row < S) {
                k_val = *reinterpret_cast<float4*>(&ptr_K_bh[(kv_start + row) * D + col]);
                v_val = *reinterpret_cast<float4*>(&ptr_V_bh[(kv_start + row) * D + col]);
            } else {
                k_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                v_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            *reinterpret_cast<float4*>(&K_smem[row * 128 + col]) = k_val;
            *reinterpret_cast<float4*>(&V_smem_T[row * 128 + col]) = v_val;
        }
        __syncthreads();

        float m_out_0[16 * 8];
        __sync_wgmma_accumulate_m16n8k16_fp32(m_out_0, 0.0f);

        for(int k = 0; k < 128; k += 16) { 
            __nv_bfloat16 q_regs_a0[16];
            __nv_bfloat16 k_regs_B0[16];
            __nv_bfloat16* q_ptr = Q_smem + k * 128;
            __nv_bfloat16* k_ptr = K_smem + k * 128;
            __sync_wgmma_matrix_a_A0_16x16_bf16_swizzle(q_regs_a0, q_ptr, 0);
            __sync_wgmma_matrix_b_B0_16x16_bf16_swizzle(k_regs_B0, k_ptr, 0);
            
            for(int n = 0; n < 128; n += 8) { 
                wgmma(m_out_0, q_regs_a0, k_regs_B0); 
            }
        }
        __sync_wgmma_commit_group();
        __sync_wgmma_wait_group(0);

        float thread_max = -1e20f;
        float thread_sum = 0.0f;
        
        for(int row = 0; row < 16; ++row) {
            for(int col = 0; col < 8; ++col) {
                float val = m_out_0[row * 8 + col] * scale;
                int global_col = kv_start + col;
                int global_row = s_start + (tid / 32) * 16 + row;
                if (global_col > global_row || global_col >= S) {
                    val = -1e20f;
                }
                thread_max = fmaxf(thread_max, val);
                m_out_0[row * 8 + col] = val;
            }
        }

        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, offset));
        }

        float m_new = fmaxf(m_prev_row[tid], thread_max);
        
        for(int row = 0; row < 16; ++row) {
            for(int col = 0; col < 8; ++col) {
                float val = m_out_0[row * 8 + col];
                float p = (val <= -1e20f) ? 0.0f : __expf(val - m_new);
                thread_sum += p;
                m_out_0[row * 8 + col] = p;
            }
        }

        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, offset);
        }

        float factor = 1.0f;
        if (m_prev_row[tid] <= -1e19f) {
            m_prev_row[tid] = m_new;
            l_prev_row[tid] = thread_sum;
        } else {
            factor = __expf(m_prev_row[tid] - m_new);
            m_prev_row[tid] = m_new;
            l_prev_row[tid] = l_prev_row[tid] * factor + thread_sum;
        }

        float m_out_1[16 * 8];
        #pragma unroll
        for(int i = 0; i < 16 * 8; ++i) {
            m_out_1[i] = m_out_0[i] * factor;
        }

        for(int row = 0; row < 16; ++row) {
            for(int col = 0; col < 8; ++col) {
                int idx = (tid % 32) / 2 + (col * 2);
                int offset = (tid % 32) % 2;
                P_smem[((tid / 32) * 16 + row) * 128 + idx * 2 + offset] = __float2bfloat16(m_out_1[row * 8 + col]);
            }
        }
        __syncthreads(); 
        
        if (factor != 1.0f) {
            for(int c = tid; c < 128 * 128; c += 128) {
                O_smem_f32[c] *= factor;
            }
        }
        __syncthreads();
        
        for(int k = 0; k < 128; k += 16) { 
            __nv_bfloat16 p_regs_A0[16];
            __nv_bfloat16 v_regs_B0[16];
            __nv_bfloat16* p_ptr = P_smem + k * 128;
            __nv_bfloat16* v_ptr = V_smem_T + k; 
            
            __sync_wgmma_matrix_a_A0_16x16_bf16_swizzle(p_regs_A0, p_ptr, 0);
            __sync_wgmma_matrix_b_B0_16x16_bf16_swizzle(v_regs_B0, v_ptr, 0);
            
            for(int n = 0; n < 128; n += 8) { 
                wgmma(m_out_pv, p_regs_A0, v_regs_B0);
            }
        }
        __sync_wgmma_commit_group();
        __sync_wgmma_wait_group(0);
        
        for(int row = 0; row < 16; ++row) {
            for(int col = 0; col < 8; ++col) {
                O_smem_f32[((tid / 32) * 16 + row) * 128 + (tid % 32) + col * 32] += m_out_pv[row * 8 + col];
            }
        }
        __syncthreads();
    }

    __syncthreads();
    
    for(int i = tid; i < 128 * 128; i += 128) {
        int row = i / 128;
        int col = i % 128;
        float val = O_smem_f32[i];
        if (l_prev_row[row] > 0.0f) {
            val /= l_prev_row[row];
        }
        O_smem_bf16[row * 128 + col] = __float2bfloat16(val);
    }
    __syncthreads();
    
    for(int i = tid; i < 128 * 16; i += 128) {
        int row = i / 16;
        int col = (i % 16) * 8;
        if (s_start + row < S) {
            *reinterpret_cast<float4*>(&ptr_O_bh[(s_start + row) * D + col]) = *reinterpret_cast<float4*>(&O_smem_bf16[row * 128 + col]);
        }
    }
    
    if (tid < 128 && s_start + tid < S) {
        ptr_LSE_bh[s_start + tid] = m_prev_row[tid] + logf(l_prev_row[tid]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());
    
    int64_t num_row_blocks = (S + 127) / 128;
    dim3 grid(num_row_blocks, B * H);
    dim3 block(128);
    
    int smem_size = 16 + 4 * 128 * 128 * sizeof(__nv_bfloat16) + 128 * 128 * sizeof(float) + 256;
                    
    CUDA_CHECK(cudaFuncSetAttribute(
        fmha_4_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    fmha_4_kernel<<<grid, block, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, S, D
    );
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_fmha