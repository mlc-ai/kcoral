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

__device__ __forceinline__ float2 read_bf16_vec2(const __nv_bfloat16* ptr) {
    uint16_t val = *reinterpret_cast<const uint16_t*>(ptr);
    __nv_bfloat16* ptr_bf16 = reinterpret_cast<__nv_bfloat16*>(&val);
    return make_float2(__bfloat162float(ptr_bf16[0]), __bfloat162float(ptr_bf16[1]));
}

__device__ __forceinline__ void write_bf16_vec2(__nv_bfloat16* ptr, float2 val) {
    __nv_bfloat16 bf16_val[2];
    bf16_val[0] = __float2bfloat16(val.x);
    bf16_val[1] = __float2bfloat16(val.y);
    *reinterpret_cast<uint16_t*>(ptr) = *reinterpret_cast<uint16_t*>(bf16_val);
}

__device__ __forceinline__ float2 read_f32_vec2(const float* ptr) {
    uint32_t val = *reinterpret_cast<const uint32_t*>(ptr);
    float* ptr_f32 = reinterpret_cast<float*>(&val);
    return make_float2(ptr_f32[0], ptr_f32[1]);
}

__device__ __forceinline__ void load_Q(
    const __nv_bfloat16* ptr_Q_bh, __nv_bfloat16* Q_smem,
    int s_start, int64_t S, int64_t D, int tid)
{
    if (s_start + tid < S) {
        const __nv_bfloat16* q_gmem = ptr_Q_bh + (s_start + tid) * D;
        for (int d = 0; d < 128; d += 2) {
            write_bf16_vec2(&Q_smem[tid * 128 + d], read_bf16_vec2(q_gmem + d));
        }
    } else {
        for (int d = 0; d < 128; d += 2) {
            write_bf16_vec2(&Q_smem[tid * 128 + d], make_float2(0.0f, 0.0f));
        }
    }
}

__device__ __forceinline__ void load_KV(
    const __nv_bfloat16* ptr_K_bh, const __nv_bfloat16* ptr_V_bh,
    __nv_bfloat16* K_smem, __nv_bfloat16* V_smem_T,
    int kv_start, int64_t S, int64_t D, int tid)
{
    if (kv_start + tid < S) {
        const __nv_bfloat16* k_gmem = ptr_K_bh + (kv_start + tid) * D;
        const __nv_bfloat16* v_gmem = ptr_V_bh + (kv_start + tid) * D;
        for (int d = 0; d < 128; d += 2) {
            write_bf16_vec2(&K_smem[tid * 128 + d], read_bf16_vec2(k_gmem + d));
            __nv_bfloat16* v_ptr0 = &V_smem_T[d * 128 + tid];
            __nv_bfloat16* v_ptr1 = &V_smem_T[(d + 1) * 128 + tid];
            write_bf16_vec2(v_ptr0, read_bf16_vec2(v_gmem + d));
            write_bf16_vec2(v_ptr1, read_bf16_vec2(v_gmem + d + 1));
        }
    } else {
        for (int d = 0; d < 128; d += 2) {
            write_bf16_vec2(&K_smem[tid * 128 + d], make_float2(0.0f, 0.0f));
            __nv_bfloat16* v_ptr0 = &V_smem_T[d * 128 + tid];
            __nv_bfloat16* v_ptr1 = &V_smem_T[(d + 1) * 128 + tid];
            write_bf16_vec2(v_ptr0, make_float2(0.0f, 0.0f));
            write_bf16_vec2(v_ptr1, make_float2(0.0f, 0.0f));
        }
    }
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

    const __nv_bfloat16* ptr_Q_bh = Q + bh * S * D;
    const __nv_bfloat16* ptr_K_bh = K + bh * S * D;
    const __nv_bfloat16* ptr_V_bh = V + bh * S * D;
    __nv_bfloat16* ptr_O_bh = O + bh * S * D;
    float* ptr_LSE_bh = LSE + bh * S;

    int s_start = row_block * 128;
    int global_row = s_start + tid;

    extern __shared__ char smem_pool[];
    __nv_bfloat16* Q_smem = (__nv_bfloat16*)smem_pool;
    __nv_bfloat16* K_smem = Q_smem + 128 * 128;
    __nv_bfloat16* V_smem_T = K_smem + 128 * 128;
    __nv_bfloat16* P_smem = V_smem_T + 128 * 128;
    float* O_smem_f32 = (float*)(P_smem + 128 * 128);
    float* m_row = (float*)(O_smem_f32 + 128 * 128);
    float* l_row = m_row + 128;

    if (tid < 128) {
        m_row[tid] = -1e20f;
        l_row[tid] = 0.0f;
    }
    
    for (int i = tid; i < 128 * 128; i += 128) {
        O_smem_f32[i] = 0.0f;
    }
    
    __syncthreads();

    load_Q(ptr_Q_bh, Q_smem, s_start, S, D, tid);
    __syncthreads();

    float scale = 1.0f / sqrtf((float)D);

    for (int j = 0; j <= row_block; ++j) {
        __syncthreads();

        load_KV(ptr_K_bh, ptr_V_bh, K_smem, V_smem_T, j * 128, S, D, tid);
        __syncthreads();

        float S_reg[128] = {0};
        for (int k_idx = 0; k_idx < 128; ++k_idx) {
            float s = 0;
            for (int d = 0; d < 128; ++d) {
                s += __bfloat162float(Q_smem[tid * 128 + d]) * 
                     __bfloat162float(K_smem[k_idx * 128 + d]);
            }
            S_reg[k_idx] = s * scale;
        }

        for (int col = 0; col < 128; ++col) {
            int global_col = j * 128 + col;
            if (global_col > global_row || global_col >= S) {
                S_reg[col] = -1e20f;
            }
        }

        float m_prev_tid = m_row[tid];
        float m_new = m_prev_tid;
        for (int col = 0; col < 128; ++col) {
            m_new = fmaxf(m_new, S_reg[col]);
        }

        float temp_l = 0.0f;
        for (int col = 0; col < 128; ++col) {
            float p = (S_reg[col] <= -1e20f) ? 0.0f : __expf(S_reg[col] - m_new);
            temp_l += p;
            S_reg[col] = p;
        }

        if (m_prev_tid <= -1e19f) {
            m_row[tid] = m_new;
            l_row[tid] = temp_l;
            for (int col = 0; col < 128; ++col) {
                P_smem[tid * 128 + col] = __float2bfloat16(S_reg[col]);
            }
        } else {
            float factor = __expf(m_prev_tid - m_new);
            m_row[tid] = m_new;
            l_row[tid] = l_row[tid] * factor + temp_l;
            for (int col = 0; col < 128; ++col) {
                P_smem[tid * 128 + col] = __float2bfloat16(S_reg[col] * factor);
            }
            for (int d = 0; d < 128; ++d) {
                O_smem_f32[tid * 128 + d] *= factor;
            }
        }

        __syncthreads();

        for (int d = 0; d < 128; ++d) {
            float o = O_smem_f32[tid * 128 + d];
            for (int j_idx = 0; j_idx < 128; ++j_idx) {
                o += __bfloat162float(P_smem[tid * 128 + j_idx]) * 
                     __bfloat162float(V_smem_T[d * 128 + j_idx]);
            }
            O_smem_f32[tid * 128 + d] = o;
        }
    }

    __syncthreads();
    
    for (int d = 0; d < 128; d += 2) {
        float l = l_row[tid];
        float2 o = read_f32_vec2(&O_smem_f32[tid * 128 + d]);
        o.x /= l;
        o.y /= l;
        __nv_bfloat16* o_gmem = ptr_O_bh + (s_start + tid) * D + d;
        if (s_start + tid < S) {
            write_bf16_vec2(o_gmem, o);
        }
    }

    if (tid < 128 && s_start + tid < S) {
        ptr_LSE_bh[s_start + tid] = m_row[tid] + logf(l_row[tid]);
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
    
    int smem_size = 4 * 128 * 128 * sizeof(__nv_bfloat16) + 
                    128 * 128 * sizeof(float) + 
                    2 * 128 * sizeof(float);
                    
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