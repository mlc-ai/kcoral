#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
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

namespace tvm_ffi_mha {

__device__ __forceinline__ uint32_t cvt_to_smem(const void* ptr) {
    uint32_t smem_ptr;
    asm("{\n\t"
        ".reg .u64 smem_ptr64;\n\t"
        "cvta.to.shared.u64 smem_ptr64, %1;\n\t"
        "cvt.u32.u64 %0, smem_ptr64;\n\t"
        "}"
        : "=r"(smem_ptr) : "l"(ptr));
    return smem_ptr;
}

__global__ void mha_fwd_kernel(const __nv_bfloat16* __restrict__ Q,
                               const __nv_bfloat16* __restrict__ K,
                               const __nv_bfloat16* __restrict__ V,
                               __nv_bfloat16* __restrict__ O,
                               float* __restrict__ LSE,
                               int seq_len) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_start = blockIdx.x * 64;
    
    if (q_start >= seq_len) return;
    
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    
    int64_t batch_head_offset = (int64_t)b * gridDim.y * seq_len * 128 + (int64_t)h * seq_len * 128;
    const __nv_bfloat16* q_ptr = Q + batch_head_offset;
    const __nv_bfloat16* k_ptr = K + batch_head_offset;
    const __nv_bfloat16* v_ptr = V + batch_head_offset;
    __nv_bfloat16* o_ptr = O + batch_head_offset;
    float* lse_ptr = LSE + (int64_t)b * gridDim.y * seq_len + (int64_t)h * seq_len;
    
    extern __shared__ __align__(16) char smem_dyn[];
    __nv_bfloat16 (*Q_s)[136] = (__nv_bfloat16 (*)[136])(smem_dyn);
    __nv_bfloat16 (*K_s)[136] = (__nv_bfloat16 (*)[136])(smem_dyn + 17408);
    __nv_bfloat16 (*V_s)[136] = (__nv_bfloat16 (*)[136])(smem_dyn + 34816);
    __nv_bfloat16 (*P_s)[136] = K_s; // P_s safely aliases K_s as they do not overlap in time
    
    #pragma unroll
    for (int i = threadIdx.x; i < 64 * 16; i += 128) {
        int row = i / 16;
        int col = (i % 16) * 8;
        if (q_start + row < seq_len) {
            *(int4*)&Q_s[row][col] = *(const int4*)&q_ptr[(q_start + row) * 128 + col];
        } else {
            *(int4*)&Q_s[row][col] = make_int4(0,0,0,0);
        }
    }
    
    float m0 = -INFINITY, m1 = -INFINITY;
    float l0 = 0.0f, l1 = 0.0f;
    float O_reg[16][4];
    
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        O_reg[j][0] = 0; O_reg[j][1] = 0; O_reg[j][2] = 0; O_reg[j][3] = 0;
    }
    
    const float scale = 0.0883883476f; // 1 / sqrt(128)
    
    for (int k_start = 0; k_start <= q_start + 63 && k_start < seq_len; k_start += 64) {
        #pragma unroll
        for (int i = threadIdx.x; i < 64 * 16; i += 128) {
            int row = i / 16;
            int col = (i % 16) * 8;
            if (k_start + row < seq_len) {
                *(int4*)&K_s[row][col] = *(const int4*)&k_ptr[(k_start + row) * 128 + col];
                *(int4*)&V_s[row][col] = *(const int4*)&v_ptr[(k_start + row) * 128 + col];
            } else {
                *(int4*)&K_s[row][col] = make_int4(0,0,0,0);
                *(int4*)&V_s[row][col] = make_int4(0,0,0,0);
            }
        }
        __syncthreads();
        
        float S[8][4];
        #pragma unroll
        for(int j = 0; j < 8; ++j) {
            S[j][0] = 0; S[j][1] = 0; S[j][2] = 0; S[j][3] = 0;
        }
        
        #pragma unroll
        for (int k = 0; k < 128; k += 16) {
            uint32_t q_reg[4];
            uint32_t q_addr = cvt_to_smem(&Q_s[warp_id * 16 + (lane % 16)][k]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                         : "=r"(q_reg[0]), "=r"(q_reg[1]), "=r"(q_reg[2]), "=r"(q_reg[3]) : "r"(q_addr));
            
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t k_reg[2];
                int r_k = j * 8;
                uint32_t k_addr;
                if (lane < 8) k_addr = cvt_to_smem(&K_s[r_k + lane][k]);
                else if (lane < 16) k_addr = cvt_to_smem(&K_s[r_k + lane - 8][k + 8]);
                else k_addr = 0;
                
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                             : "=r"(k_reg[0]), "=r"(k_reg[1]) : "r"(k_addr));
                             
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16 "
                             "{%0, %1, %2, %3}, "
                             "{%4, %5, %6, %7}, "
                             "{%8, %9}, "
                             "{%0, %1, %2, %3};"
                             : "+f"(S[j][0]), "+f"(S[j][1]), "+f"(S[j][2]), "+f"(S[j][3])
                             : "r"(q_reg[0]), "r"(q_reg[1]), "r"(q_reg[2]), "r"(q_reg[3]),
                               "r"(k_reg[0]), "r"(k_reg[1]));
            }
        }
        
        float max0 = -INFINITY, max1 = -INFINITY;
        
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int g_q0 = q_start + warp_id * 16 + lane / 4;
            int g_q1 = g_q0 + 8;
            int g_k0 = k_start + j * 8 + (lane % 4) * 2;
            int g_k1 = g_k0 + 1;
            
            bool mask00 = (g_k0 <= g_q0) && (g_k0 < seq_len) && (g_q0 < seq_len);
            bool mask01 = (g_k1 <= g_q0) && (g_k1 < seq_len) && (g_q0 < seq_len);
            bool mask10 = (g_k0 <= g_q1) && (g_k0 < seq_len) && (g_q1 < seq_len);
            bool mask11 = (g_k1 <= g_q1) && (g_k1 < seq_len) && (g_q1 < seq_len);
            
            S[j][0] = mask00 ? S[j][0] * scale : -INFINITY;
            S[j][1] = mask01 ? S[j][1] * scale : -INFINITY;
            S[j][2] = mask10 ? S[j][2] * scale : -INFINITY;
            S[j][3] = mask11 ? S[j][3] * scale : -INFINITY;
            
            max0 = fmaxf(max0, S[j][0]);
            max0 = fmaxf(max0, S[j][1]);
            max1 = fmaxf(max1, S[j][2]);
            max1 = fmaxf(max1, S[j][3]);
        }
        
        #pragma unroll
        for (int offset = 1; offset < 4; offset *= 2) {
            max0 = fmaxf(max0, __shfl_xor_sync(0xffffffff, max0, offset));
            max1 = fmaxf(max1, __shfl_xor_sync(0xffffffff, max1, offset));
        }
        
        float new_m0 = fmaxf(m0, max0);
        float new_m1 = fmaxf(m1, max1);
        float exp_diff0 = expf(m0 - new_m0);
        float exp_diff1 = expf(m1 - new_m1);
        
        l0 *= exp_diff0;
        l1 *= exp_diff1;
        
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            O_reg[j][0] *= exp_diff0;
            O_reg[j][1] *= exp_diff0;
            O_reg[j][2] *= exp_diff1;
            O_reg[j][3] *= exp_diff1;
        }
        
        m0 = new_m0;
        m1 = new_m1;
        
        float sum0 = 0.0f, sum1 = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            S[j][0] = expf(S[j][0] - m0);
            S[j][1] = expf(S[j][1] - m0);
            S[j][2] = expf(S[j][2] - m1);
            S[j][3] = expf(S[j][3] - m1);
            sum0 += S[j][0] + S[j][1];
            sum1 += S[j][2] + S[j][3];
        }
        
        #pragma unroll
        for (int offset = 1; offset < 4; offset *= 2) {
            sum0 += __shfl_xor_sync(0xffffffff, sum0, offset);
            sum1 += __shfl_xor_sync(0xffffffff, sum1, offset);
        }
        
        l0 += sum0;
        l1 += sum1;
        
        __syncthreads(); // Wait for all warps to finish reading K_s before overwriting with P_s
        
        int r0 = warp_id * 16 + lane / 4;
        int r1 = warp_id * 16 + lane / 4 + 8;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int c0 = j * 8 + (lane % 4) * 2;
            int c1 = c0 + 1;
            P_s[r0][c0] = __float2bfloat16(S[j][0]);
            P_s[r0][c1] = __float2bfloat16(S[j][1]);
            P_s[r1][c0] = __float2bfloat16(S[j][2]);
            P_s[r1][c1] = __float2bfloat16(S[j][3]);
        }
        
        __syncwarp();
        
        #pragma unroll
        for (int kp = 0; kp < 64; kp += 16) {
            uint32_t p_reg[4];
            uint32_t p_addr = cvt_to_smem(&P_s[warp_id * 16 + (lane % 16)][kp]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                         : "=r"(p_reg[0]), "=r"(p_reg[1]), "=r"(p_reg[2]), "=r"(p_reg[3]) : "r"(p_addr));
                         
            #pragma unroll
            for (int jv = 0; jv < 16; ++jv) {
                uint32_t v_reg[2];
                uint32_t v_addr = cvt_to_smem(&V_s[kp + (lane % 16)][jv * 8]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                             : "=r"(v_reg[0]), "=r"(v_reg[1]) : "r"(v_addr));
                             
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16 "
                             "{%0, %1, %2, %3}, "
                             "{%4, %5, %6, %7}, "
                             "{%8, %9}, "
                             "{%0, %1, %2, %3};"
                             : "+f"(O_reg[jv][0]), "+f"(O_reg[jv][1]), "+f"(O_reg[jv][2]), "+f"(O_reg[jv][3])
                             : "r"(p_reg[0]), "r"(p_reg[1]), "r"(p_reg[2]), "r"(p_reg[3]),
                               "r"(v_reg[0]), "r"(v_reg[1]));
            }
        }
        
        __syncthreads(); // ensure P_s/V_s are consumed before next iteration overwrites K_s/V_s
    }
    
    float inv_l0 = l0 > 0.0f ? 1.0f / l0 : 0.0f;
    float inv_l1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        O_reg[j][0] *= inv_l0;
        O_reg[j][1] *= inv_l0;
        O_reg[j][2] *= inv_l1;
        O_reg[j][3] *= inv_l1;
        
        int g_q0 = q_start + warp_id * 16 + lane / 4;
        int g_q1 = g_q0 + 8;
        int c0 = j * 8 + (lane % 4) * 2;
        int c1 = c0 + 1;
        
        if (g_q0 < seq_len) {
            o_ptr[g_q0 * 128 + c0] = __float2bfloat16(O_reg[j][0]);
            o_ptr[g_q0 * 128 + c1] = __float2bfloat16(O_reg[j][1]);
        }
        if (g_q1 < seq_len) {
            o_ptr[g_q1 * 128 + c0] = __float2bfloat16(O_reg[j][2]);
            o_ptr[g_q1 * 128 + c1] = __float2bfloat16(O_reg[j][3]);
        }
    }
    
    if (lane % 4 == 0) {
        int g_q0 = q_start + warp_id * 16 + lane / 4;
        int g_q1 = g_q0 + 8;
        if (g_q0 < seq_len) lse_ptr[g_q0] = m0 + logf(l0);
        if (g_q1 < seq_len) lse_ptr[g_q1] = m1 + logf(l1);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    const __nv_bfloat16* q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_data = static_cast<float*>(LSE.data_ptr());
    
    int64_t threads = 128;
    dim3 blocks((S + 63) / 64, H, B);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int smem_size = 3 * 64 * 136 * sizeof(__nv_bfloat16); // 52,224 bytes
    CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    mha_fwd_kernel<<<blocks, threads, smem_size, stream>>>(q_data, k_data, v_data, o_data, lse_data, S);
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha