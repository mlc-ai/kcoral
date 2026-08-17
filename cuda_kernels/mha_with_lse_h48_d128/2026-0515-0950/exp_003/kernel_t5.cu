#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        exit(1);                                                   \
    }                                                              \
} while(0)

extern __shared__ __align__(128) uint8_t smem_base_bytes[];

__global__ void __launch_bounds__(128) MHA_Kernel(
    const __nv_bfloat16* __restrict__ Q_ptr,
    const __nv_bfloat16* __restrict__ K_ptr,
    const __nv_bfloat16* __restrict__ V_ptr,
    __nv_bfloat16* __restrict__ O_ptr,
    float* __restrict__ LSE_ptr,
    int B, int H, int S) 
{
    __nv_bfloat16 (*smem_Q)[136]   = (__nv_bfloat16 (*)[136])(smem_base_bytes);
    __nv_bfloat16 (*smem_K)[136]   = (__nv_bfloat16 (*)[136])(smem_base_bytes + 34816);
    __nv_bfloat16 (*smem_V_T)[136] = (__nv_bfloat16 (*)[136])(smem_base_bytes + 69632);
    __nv_bfloat16 (*smem_P)[136]   = (__nv_bfloat16 (*)[136])(smem_base_bytes + 104448);

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;

    int b_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int s_idx = blockIdx.x;
    
    size_t batch_offset = (size_t)b_idx * H * S * 128 + (size_t)h_idx * S * 128;
    const __nv_bfloat16* Q_gmem = Q_ptr + batch_offset;
    const __nv_bfloat16* K_gmem = K_ptr + batch_offset;
    const __nv_bfloat16* V_gmem = V_ptr + batch_offset;
    __nv_bfloat16* O_gmem = O_ptr + batch_offset;
    float* LSE_gmem = LSE_ptr + (size_t)b_idx * H * S + (size_t)h_idx * S;

    int global_row = s_idx * 128;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int f4_idx = i * 128 + tid;
        int row = f4_idx / 16;
        int col_f4 = f4_idx % 16;
        if (global_row + row < S) {
            ((float4*)&smem_Q[row][col_f4 * 8])[0] = ((const float4*)Q_gmem)[(global_row + row) * 16 + col_f4];
        } else {
            ((float4*)&smem_Q[row][col_f4 * 8])[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    
    float O_acc[2][16][4];
    #pragma unroll
    for(int wm=0; wm<2; ++wm) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            O_acc[wm][i][0] = 0.0f; O_acc[wm][i][1] = 0.0f; O_acc[wm][i][2] = 0.0f; O_acc[wm][i][3] = 0.0f;
        }
    }
    
    float m[2][2] = {{-CUDART_INF_F, -CUDART_INF_F}, {-CUDART_INF_F, -CUDART_INF_F}};
    float l[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    float scale = 1.0f / sqrtf(128.0f);

    int num_chunks = (S + 127) / 128;

    for (int j_chunk = 0; j_chunk < num_chunks; ++j_chunk) {
        int j_global_row = j_chunk * 128;
        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int f4_idx = i * 128 + tid;
            int row = f4_idx / 16;
            int col_f4 = f4_idx % 16;
            if (j_global_row + row < S) {
                ((float4*)&smem_K[row][col_f4 * 8])[0] = ((const float4*)K_gmem)[(j_global_row + row) * 16 + col_f4];
            } else {
                ((float4*)&smem_K[row][col_f4 * 8])[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int f4_idx = i * 128 + tid;
            int row = f4_idx / 16;
            int col_f4 = f4_idx % 16;
            
            float4 val;
            if (j_global_row + row < S) {
                val = ((const float4*)V_gmem)[(j_global_row + row) * 16 + col_f4];
            } else {
                val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            __nv_bfloat162* h2 = (__nv_bfloat162*)&val;
            smem_V_T[col_f4 * 8 + 0][row] = h2[0].x;
            smem_V_T[col_f4 * 8 + 1][row] = h2[0].y;
            smem_V_T[col_f4 * 8 + 2][row] = h2[1].x;
            smem_V_T[col_f4 * 8 + 3][row] = h2[1].y;
            smem_V_T[col_f4 * 8 + 4][row] = h2[2].x;
            smem_V_T[col_f4 * 8 + 5][row] = h2[2].y;
            smem_V_T[col_f4 * 8 + 6][row] = h2[3].x;
            smem_V_T[col_f4 * 8 + 7][row] = h2[3].y;
        }
        
        __syncthreads();
        
        for (int wm = 0; wm < 2; ++wm) {
            float S_ij[16][4];
            #pragma unroll
            for (int t = 0; t < 16; ++t) {
                S_ij[t][0] = 0.0f; S_ij[t][1] = 0.0f; S_ij[t][2] = 0.0f; S_ij[t][3] = 0.0f;
            }
            
            uint32_t a[4];
            uint32_t b[2];
            
            int row_Q = warp * 32 + wm * 16;
            
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                uint32_t addr_A = (uint32_t)__cvta_generic_to_shared(&smem_Q[row_Q + (lane % 16)][k * 16 + (lane / 16) * 8]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr_A));
                             
                #pragma unroll
                for (int t = 0; t < 16; ++t) {
                    uint32_t addr_B = (uint32_t)__cvta_generic_to_shared(&smem_K[t * 8 + (lane % 16) % 8][k * 16 + ((lane % 16) / 8) * 8]);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                                 : "=r"(b[0]), "=r"(b[1]) : "r"(addr_B));
                                 
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                                 : "+f"(S_ij[t][0]), "+f"(S_ij[t][1]), "+f"(S_ij[t][2]), "+f"(S_ij[t][3])
                                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                                   "r"(b[0]), "r"(b[1]));
                }
            }
            
            float m_new[2] = {-CUDART_INF_F, -CUDART_INF_F};
            #pragma unroll
            for (int t = 0; t < 16; ++t) {
                int c0 = t * 8 + (lane % 4) * 2;
                int c1 = c0 + 1;
                
                S_ij[t][0] *= scale; S_ij[t][1] *= scale; S_ij[t][2] *= scale; S_ij[t][3] *= scale;
                
                if (j_global_row + c0 >= S) { S_ij[t][0] = -CUDART_INF_F; S_ij[t][2] = -CUDART_INF_F; }
                if (j_global_row + c1 >= S) { S_ij[t][1] = -CUDART_INF_F; S_ij[t][3] = -CUDART_INF_F; }
                
                m_new[0] = fmaxf(m_new[0], fmaxf(S_ij[t][0], S_ij[t][1]));
                m_new[1] = fmaxf(m_new[1], fmaxf(S_ij[t][2], S_ij[t][3]));
            }
            
            m_new[0] = fmaxf(m_new[0], __shfl_xor_sync(0xffffffff, m_new[0], 1));
            m_new[0] = fmaxf(m_new[0], __shfl_xor_sync(0xffffffff, m_new[0], 2));
            m_new[1] = fmaxf(m_new[1], __shfl_xor_sync(0xffffffff, m_new[1], 1));
            m_new[1] = fmaxf(m_new[1], __shfl_xor_sync(0xffffffff, m_new[1], 2));
            
            m_new[0] = fmaxf(m_new[0], m[wm][0]);
            m_new[1] = fmaxf(m_new[1], m[wm][1]);
            
            float exp_diff[2] = {0.0f, 0.0f};
            if (m[wm][0] != -CUDART_INF_F && m_new[0] != -CUDART_INF_F) exp_diff[0] = expf(m[wm][0] - m_new[0]);
            if (m[wm][1] != -CUDART_INF_F && m_new[1] != -CUDART_INF_F) exp_diff[1] = expf(m[wm][1] - m_new[1]);
            
            #pragma unroll
            for (int t_v = 0; t_v < 16; ++t_v) {
                O_acc[wm][t_v][0] *= exp_diff[0];
                O_acc[wm][t_v][1] *= exp_diff[0];
                O_acc[wm][t_v][2] *= exp_diff[1];
                O_acc[wm][t_v][3] *= exp_diff[1];
            }
            
            float p_sum[2] = {0.0f, 0.0f};
            
            #pragma unroll
            for (int t = 0; t < 16; ++t) {
                float p0 = m_new[0] != -CUDART_INF_F ? expf(S_ij[t][0] - m_new[0]) : 0.0f;
                float p1 = m_new[0] != -CUDART_INF_F ? expf(S_ij[t][1] - m_new[0]) : 0.0f;
                float p2 = m_new[1] != -CUDART_INF_F ? expf(S_ij[t][2] - m_new[1]) : 0.0f;
                float p3 = m_new[1] != -CUDART_INF_F ? expf(S_ij[t][3] - m_new[1]) : 0.0f;
                
                p_sum[0] += p0 + p1;
                p_sum[1] += p2 + p3;
                
                int r0 = warp * 16 + lane / 4;
                int r1 = r0 + 8;
                int c0 = t * 8 + (lane % 4) * 2;
                int c1 = c0 + 1;
                
                smem_P[r0][c0] = __float2bfloat16(p0);
                smem_P[r0][c1] = __float2bfloat16(p1);
                smem_P[r1][c0] = __float2bfloat16(p2);
                smem_P[r1][c1] = __float2bfloat16(p3);
            }
            
            p_sum[0] += __shfl_xor_sync(0xffffffff, p_sum[0], 1);
            p_sum[0] += __shfl_xor_sync(0xffffffff, p_sum[0], 2);
            p_sum[1] += __shfl_xor_sync(0xffffffff, p_sum[1], 1);
            p_sum[1] += __shfl_xor_sync(0xffffffff, p_sum[1], 2);
            
            l[wm][0] = l[wm][0] * exp_diff[0] + p_sum[0];
            l[wm][1] = l[wm][1] * exp_diff[1] + p_sum[1];
            
            m[wm][0] = m_new[0];
            m[wm][1] = m_new[1];
            
            __syncwarp();
            
            #pragma unroll
            for (int k_p = 0; k_p < 8; ++k_p) {
                uint32_t addr_A = (uint32_t)__cvta_generic_to_shared(&smem_P[warp * 16 + (lane % 16)][k_p * 16 + (lane / 16) * 8]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr_A));
                             
                #pragma unroll
                for (int t_v = 0; t_v < 16; ++t_v) {
                    uint32_t addr_B = (uint32_t)__cvta_generic_to_shared(&smem_V_T[t_v * 8 + (lane % 16) % 8][k_p * 16 + ((lane % 16) / 8) * 8]);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                                 : "=r"(b[0]), "=r"(b[1]) : "r"(addr_B));
                                 
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                                 : "+f"(O_acc[wm][t_v][0]), "+f"(O_acc[wm][t_v][1]), "+f"(O_acc[wm][t_v][2]), "+f"(O_acc[wm][t_v][3])
                                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                                   "r"(b[0]), "r"(b[1]));
                }
            }
        }
    }
    
    #pragma unroll
    for(int wm=0; wm<2; ++wm) {
        int r0 = (warp * 32 + wm * 16) + lane / 4;
        int r1 = r0 + 8;
        
        #pragma unroll
        for (int t_v = 0; t_v < 16; ++t_v) {
            int c0 = t_v * 8 + (lane % 4) * 2;
            int c1 = c0 + 1;
            
            float val00 = O_acc[wm][t_v][0] / l[wm][0];
            float val01 = O_acc[wm][t_v][1] / l[wm][0];
            float val10 = O_acc[wm][t_v][2] / l[wm][1];
            float val11 = O_acc[wm][t_v][3] / l[wm][1];
            
            if (global_row + r0 < S) {
                O_gmem[(global_row + r0) * 128 + c0] = __float2bfloat16(val00);
                O_gmem[(global_row + r0) * 128 + c1] = __float2bfloat16(val01);
            }
            if (global_row + r1 < S) {
                O_gmem[(global_row + r1) * 128 + c0] = __float2bfloat16(val10);
                O_gmem[(global_row + r1) * 128 + c1] = __float2bfloat16(val11);
            }
        }
        
        if (lane % 4 == 0) {
            if (global_row + r0 < S) {
                LSE_gmem[global_row + r0] = m[wm][0] + logf(l[wm][0]);
            }
            if (global_row + r1 < S) {
                LSE_gmem[global_row + r1] = m[wm][1] + logf(l[wm][1]);
            }
        }
    }
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    
    int blocks_x = (S + 127) / 128;
    dim3 blocks(blocks_x, H, B);
    dim3 threads(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int smem_size = 139264;
    CUDA_CHECK(cudaFuncSetAttribute(MHA_Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    MHA_Kernel<<<blocks, threads, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        B, H, S
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda