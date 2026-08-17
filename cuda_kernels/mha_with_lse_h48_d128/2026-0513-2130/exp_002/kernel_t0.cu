#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
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

namespace tvm_ffi_flash_attn {

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ uint32_t smem_ptr_to_uint(const void* ptr) {
    uint32_t addr;
    asm ("{ .reg .u64 u64addr;\n"
         " cvta.to.shared.u64 u64addr, %1;\n"
         " cvt.u32.u64 %0, u64addr; }\n"
         : "=r"(addr)
         : "l"(ptr));
    return addr;
}

__global__ void flash_attn_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S) 
{
    int b = blockIdx.z;
    int h = blockIdx.y;
    int s_block = blockIdx.x;
    
    int s_start = s_block * 64;
    if (s_start >= S) return;

    int head_offset = b * H * S * 128 + h * S * 128;
    const __nv_bfloat16* q_ptr = Q + head_offset + s_start * 128;
    __nv_bfloat16* o_ptr = O + head_offset + s_start * 128;
    float* lse_ptr = LSE + b * H * S + h * S + s_start;

    extern __shared__ __align__(16) uint8_t smem_raw[];
    __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    
    __nv_bfloat16* s_Q = smem;                                  // 64 * 136 = 8704
    __nv_bfloat16* s_K = s_Q + 64 * 136;                        // 64 * 136 = 8704
    __nv_bfloat16* s_V = s_K + 64 * 136;                        // 128 * 72 = 9216
    __nv_bfloat16* s_V_temp = s_V + 128 * 72;                   // 64 * 136 = 8704
    __nv_bfloat16* s_P = s_V_temp + 64 * 136;                   // 64 * 72 = 4608

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int m_start = warp_id * 16;

    // Load Q block (64x128)
    int row = tid / 16;
    int col = (tid % 16) * 8;
    for(int i = 0; i < 8; i++) {
        int r = row + i * 8;
        if (s_start + r < S) {
            float4 val = *reinterpret_cast<const float4*>(&q_ptr[r * 128 + col]);
            *reinterpret_cast<float4*>(&s_Q[r * 136 + col]) = val;
        } else {
            *reinterpret_cast<float4*>(&s_Q[r * 136 + col]) = make_float4(0, 0, 0, 0);
        }
    }

    float O_acc[16][4];
    for(int d = 0; d < 16; d++) {
        for(int j = 0; j < 4; j++) O_acc[d][j] = 0.0f;
    }

    float m_i[2] = {-INFINITY, -INFINITY};
    float l_i[2] = {0.0f, 0.0f};

    int num_n_blocks = (S + 63) / 64;
    float scale = 0.0883883476f; // 1.0f / sqrt(128.0f)

    for (int n_block = 0; n_block < num_n_blocks; n_block++) {
        int n_start = n_block * 64;
        
        const __nv_bfloat16* k_ptr = K + head_offset + n_start * 128;
        const __nv_bfloat16* v_ptr = V + head_offset + n_start * 128;

        __syncthreads();

        for(int i = 0; i < 8; i++) {
            int r = row + i * 8;
            if (n_start + r < S) {
                float4 val_k = *reinterpret_cast<const float4*>(&k_ptr[r * 128 + col]);
                *reinterpret_cast<float4*>(&s_K[r * 136 + col]) = val_k;
                
                float4 val_v = *reinterpret_cast<const float4*>(&v_ptr[r * 128 + col]);
                *reinterpret_cast<float4*>(&s_V_temp[r * 136 + col]) = val_v;
            } else {
                *reinterpret_cast<float4*>(&s_K[r * 136 + col]) = make_float4(0, 0, 0, 0);
                *reinterpret_cast<float4*>(&s_V_temp[r * 136 + col]) = make_float4(0, 0, 0, 0);
            }
        }
        
        __syncthreads();

        // Transpose V from s_V_temp to s_V
        for(int i = 0; i < 64; i++) {
            int idx = tid * 64 + i;
            int n = idx / 128;
            int d = idx % 128;
            s_V[d * 72 + n] = s_V_temp[n * 136 + d];
        }
        
        __syncthreads();

        // Compute Q @ K^T
        float acc[8][4];
        for(int n = 0; n < 8; n++) {
            for(int j = 0; j < 4; j++) acc[n][j] = 0.0f;
        }

        for(int k_step = 0; k_step < 8; k_step++) {
            uint32_t q_regs[4];
            int mat_idx_q = lane_id / 8;
            int q_r = m_start + (mat_idx_q / 2) * 8 + (lane_id % 8);
            int q_c = k_step * 16 + (mat_idx_q % 2) * 8;
            uint32_t q_addr = smem_ptr_to_uint(&s_Q[q_r * 136 + q_c]);
            
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                         : "=r"(q_regs[0]), "=r"(q_regs[1]), "=r"(q_regs[2]), "=r"(q_regs[3])
                         : "r"(q_addr));

            for(int n_step = 0; n_step < 8; n_step++) {
                uint32_t k_regs[2];
                int mat_idx_k = (lane_id % 16) / 8;
                int k_r = n_step * 8 + (lane_id % 8);
                int k_c = k_step * 16 + mat_idx_k * 8;
                uint32_t k_addr = smem_ptr_to_uint(&s_K[k_r * 136 + k_c]);
                
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                             : "=r"(k_regs[0]), "=r"(k_regs[1])
                             : "r"(k_addr));

                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                             "{%0, %1, %2, %3}, "
                             "{%4, %5, %6, %7}, "
                             "{%8, %9}, "
                             "{%10, %11, %12, %13};"
                             : "=f"(acc[n_step][0]), "=f"(acc[n_step][1]), "=f"(acc[n_step][2]), "=f"(acc[n_step][3])
                             : "r"(q_regs[0]), "r"(q_regs[1]), "r"(q_regs[2]), "r"(q_regs[3]),
                               "r"(k_regs[0]), "r"(k_regs[1]),
                               "f"(acc[n_step][0]), "f"(acc[n_step][1]), "f"(acc[n_step][2]), "f"(acc[n_step][3]));
            }
        }

        // Mask, scale and soft-max updates
        float m_ij[2] = {-INFINITY, -INFINITY};
        int row0 = lane_id / 4;
        int row1 = lane_id / 4 + 8;
        
        for(int n = 0; n < 8; n++) {
            int g_col0 = n_start + n * 8 + (lane_id % 4) * 2;
            int g_col1 = g_col0 + 1;
            
            acc[n][0] = (g_col0 >= S) ? -INFINITY : (acc[n][0] * scale);
            acc[n][2] = (g_col0 >= S) ? -INFINITY : (acc[n][2] * scale);
            
            acc[n][1] = (g_col1 >= S) ? -INFINITY : (acc[n][1] * scale);
            acc[n][3] = (g_col1 >= S) ? -INFINITY : (acc[n][3] * scale);

            m_ij[0] = max(m_ij[0], max(acc[n][0], acc[n][1]));
            m_ij[1] = max(m_ij[1], max(acc[n][2], acc[n][3]));
        }

        for(int offset = 2; offset > 0; offset /= 2) {
            m_ij[0] = max(m_ij[0], __shfl_xor_sync(0xffffffff, m_ij[0], offset));
            m_ij[1] = max(m_ij[1], __shfl_xor_sync(0xffffffff, m_ij[1], offset));
        }

        float m_new[2];
        m_new[0] = max(m_i[0], m_ij[0]);
        m_new[1] = max(m_i[1], m_ij[1]);

        float scale_o[2];
        scale_o[0] = (m_new[0] == -INFINITY) ? 0.0f : expf(m_i[0] - m_new[0]);
        scale_o[1] = (m_new[1] == -INFINITY) ? 0.0f : expf(m_i[1] - m_new[1]);

        for(int d = 0; d < 16; d++) {
            O_acc[d][0] *= scale_o[0];
            O_acc[d][1] *= scale_o[0];
            O_acc[d][2] *= scale_o[1];
            O_acc[d][3] *= scale_o[1];
        }

        float sum_exp[2] = {0.0f, 0.0f};
        for(int n = 0; n < 8; n++) {
            acc[n][0] = (m_new[0] == -INFINITY) ? 0.0f : expf(acc[n][0] - m_new[0]);
            acc[n][1] = (m_new[0] == -INFINITY) ? 0.0f : expf(acc[n][1] - m_new[0]);
            acc[n][2] = (m_new[1] == -INFINITY) ? 0.0f : expf(acc[n][2] - m_new[1]);
            acc[n][3] = (m_new[1] == -INFINITY) ? 0.0f : expf(acc[n][3] - m_new[1]);

            sum_exp[0] += acc[n][0] + acc[n][1];
            sum_exp[1] += acc[n][2] + acc[n][3];

            int c0 = n * 8 + (lane_id % 4) * 2;
            uint32_t p0 = pack_bf16_fn(__float_as_uint(acc[n][0]), __float_as_uint(acc[n][1]));
            uint32_t p1 = pack_bf16_fn(__float_as_uint(acc[n][2]), __float_as_uint(acc[n][3]));
            
            *reinterpret_cast<uint32_t*>(&s_P[(m_start + row0) * 72 + c0]) = p0;
            *reinterpret_cast<uint32_t*>(&s_P[(m_start + row1) * 72 + c0]) = p1;
        }

        for(int offset = 2; offset > 0; offset /= 2) {
            sum_exp[0] += __shfl_xor_sync(0xffffffff, sum_exp[0], offset);
            sum_exp[1] += __shfl_xor_sync(0xffffffff, sum_exp[1], offset);
        }

        l_i[0] = (m_new[0] == -INFINITY) ? 0.0f : (l_i[0] * scale_o[0] + sum_exp[0]);
        l_i[1] = (m_new[1] == -INFINITY) ? 0.0f : (l_i[1] * scale_o[1] + sum_exp[1]);
        m_i[0] = m_new[0];
        m_i[1] = m_new[1];

        __syncwarp();

        // Compute P @ V
        for(int k_step = 0; k_step < 4; k_step++) {
            uint32_t p_regs[4];
            int mat_idx_p = lane_id / 8;
            int p_r = m_start + (mat_idx_p / 2) * 8 + (lane_id % 8);
            int p_c = k_step * 16 + (mat_idx_p % 2) * 8;
            uint32_t p_addr = smem_ptr_to_uint(&s_P[p_r * 72 + p_c]);
            
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                         : "=r"(p_regs[0]), "=r"(p_regs[1]), "=r"(p_regs[2]), "=r"(p_regs[3])
                         : "r"(p_addr));

            for(int d_step = 0; d_step < 16; d_step++) {
                uint32_t v_regs[2];
                int mat_idx_v = (lane_id % 16) / 8;
                int v_r = d_step * 8 + (lane_id % 8);
                int v_c = k_step * 16 + mat_idx_v * 8;
                uint32_t v_addr = smem_ptr_to_uint(&s_V[v_r * 72 + v_c]);
                
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                             : "=r"(v_regs[0]), "=r"(v_regs[1])
                             : "r"(v_addr));

                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                             "{%0, %1, %2, %3}, "
                             "{%4, %5, %6, %7}, "
                             "{%8, %9}, "
                             "{%10, %11, %12, %13};"
                             : "=f"(O_acc[d_step][0]), "=f"(O_acc[d_step][1]), "=f"(O_acc[d_step][2]), "=f"(O_acc[d_step][3])
                             : "r"(p_regs[0]), "r"(p_regs[1]), "r"(p_regs[2]), "r"(p_regs[3]),
                               "r"(v_regs[0]), "r"(v_regs[1]),
                               "f"(O_acc[d_step][0]), "f"(O_acc[d_step][1]), "f"(O_acc[d_step][2]), "f"(O_acc[d_step][3]));
            }
        }
    }

    if (lane_id % 4 == 0) {
        int g_row0 = s_start + m_start + row0;
        int g_row1 = s_start + m_start + row1;
        if (g_row0 < S) lse_ptr[m_start + row0] = m_i[0] + logf(l_i[0]);
        if (g_row1 < S) lse_ptr[m_start + row1] = m_i[1] + logf(l_i[1]);
    }

    __syncthreads();

    float inv_l0 = (l_i[0] == 0.0f) ? 0.0f : (1.0f / l_i[0]);
    float inv_l1 = (l_i[1] == 0.0f) ? 0.0f : (1.0f / l_i[1]);

    for(int d = 0; d < 16; d++) {
        __nv_bfloat16 o00 = __float2bfloat16(O_acc[d][0] * inv_l0);
        __nv_bfloat16 o01 = __float2bfloat16(O_acc[d][1] * inv_l0);
        __nv_bfloat16 o10 = __float2bfloat16(O_acc[d][2] * inv_l1);
        __nv_bfloat16 o11 = __float2bfloat16(O_acc[d][3] * inv_l1);

        int smem_row0 = m_start + row0;
        int smem_row1 = m_start + row1;
        int smem_col0 = d * 8 + (lane_id % 4) * 2;

        s_Q[smem_row0 * 136 + smem_col0] = o00;
        s_Q[smem_row0 * 136 + smem_col0 + 1] = o01;
        s_Q[smem_row1 * 136 + smem_col0] = o10;
        s_Q[smem_row1 * 136 + smem_col0 + 1] = o11;
    }

    __syncthreads();

    for(int i = 0; i < 8; i++) {
        int r = row + i * 8;
        if (s_start + r < S) {
            float4 val = *reinterpret_cast<float4*>(&s_Q[r * 136 + col]);
            *reinterpret_cast<float4*>(&o_ptr[r * 128 + col]) = val;
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    // int64_t D = Q.size(3); // Const=128

    const __nv_bfloat16* q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    int threads = 128;
    dim3 blocks((S + 63) / 64, H, B);
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int smem_size = 79872;
    CUDA_CHECK(cudaFuncSetAttribute(flash_attn_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    flash_attn_fwd_kernel<<<blocks, threads, smem_size, stream>>>(
        q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, B, H, S
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_flash_attn