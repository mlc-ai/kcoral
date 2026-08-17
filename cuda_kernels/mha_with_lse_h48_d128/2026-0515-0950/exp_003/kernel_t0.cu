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

extern __shared__ __nv_bfloat16 smem_base[];

__global__ void MHA_Kernel(
    const __nv_bfloat16* __restrict__ Q_ptr,
    const __nv_bfloat16* __restrict__ K_ptr,
    const __nv_bfloat16* __restrict__ V_ptr,
    __nv_bfloat16* __restrict__ O_ptr,
    float* __restrict__ LSE_ptr,
    int B, int H, int S) 
{
    // Disjoint SMEM buffers to avoid overwriting during pipelined execution
    __nv_bfloat16 (*smem_Q)[136] = (__nv_bfloat16 (*)[136])(smem_base);
    __nv_bfloat16 (*smem_K)[136] = (__nv_bfloat16 (*)[136])(smem_base + 64 * 136);
    __nv_bfloat16 (*smem_V)[136] = (__nv_bfloat16 (*)[136])(smem_base + 64 * 136 * 2);
    __nv_bfloat16 (*smem_P)[72]  = (__nv_bfloat16 (*)[72])(smem_base + 64 * 136 * 3);

    int b_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int s_idx = blockIdx.x;
    int global_row = s_idx * 64;

    size_t batch_offset = (size_t)b_idx * H * S * 128 + (size_t)h_idx * S * 128;
    const __nv_bfloat16* Q_gmem = Q_ptr + batch_offset;
    const __nv_bfloat16* K_gmem = K_ptr + batch_offset;
    const __nv_bfloat16* V_gmem = V_ptr + batch_offset;
    __nv_bfloat16* O_gmem = O_ptr + batch_offset;
    
    float* LSE_gmem = LSE_ptr + (size_t)b_idx * H * S + (size_t)h_idx * S;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;

    // Load Q block (64x128) collaboratively using float4 vectorization
    for (int i = 0; i < 8; ++i) {
        int f4_idx = i * 128 + tid;
        int row = f4_idx / 16;
        int col_f4 = f4_idx % 16;
        if (global_row + row < S) {
            ((float4*)&smem_Q[row][col_f4 * 8])[0] = ((const float4*)Q_gmem)[(global_row + row) * 16 + col_f4];
        } else {
            ((float4*)&smem_Q[row][col_f4 * 8])[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    
    // Accumulators
    float O_acc[16][4];
    for (int i = 0; i < 16; ++i) {
        O_acc[i][0] = 0.0f; O_acc[i][1] = 0.0f; O_acc[i][2] = 0.0f; O_acc[i][3] = 0.0f;
    }
    
    float m[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float l[2] = {0.0f, 0.0f};
    float scale = 1.0f / sqrtf(128.0f);

    int num_j_chunks = (S + 63) / 64;
    
    for (int j_chunk = 0; j_chunk < num_j_chunks; ++j_chunk) {
        int j_global_row = j_chunk * 64;
        
        __syncthreads();
        
        // Load K block
        for (int i = 0; i < 8; ++i) {
            int f4_idx = i * 128 + tid;
            int row = f4_idx / 16;
            int col_f4 = f4_idx % 16;
            if (j_global_row + row < S) {
                ((float4*)&smem_K[row][col_f4 * 8])[0] = ((const float4*)K_gmem)[(j_global_row + row) * 16 + col_f4];
            } else {
                ((float4*)&smem_K[row][col_f4 * 8])[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        
        // Load V block
        for (int i = 0; i < 8; ++i) {
            int f4_idx = i * 128 + tid;
            int row = f4_idx / 16;
            int col_f4 = f4_idx % 16;
            if (j_global_row + row < S) {
                ((float4*)&smem_V[row][col_f4 * 8])[0] = ((const float4*)V_gmem)[(j_global_row + row) * 16 + col_f4];
            } else {
                ((float4*)&smem_V[row][col_f4 * 8])[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        
        __syncthreads();
        
        float S_ij[8][4];
        for (int t = 0; t < 8; ++t) {
            S_ij[t][0] = 0.0f; S_ij[t][1] = 0.0f; S_ij[t][2] = 0.0f; S_ij[t][3] = 0.0f;
        }
        
        uint32_t a[4];
        uint32_t b[2];
        
        // Q @ K^T computation
        for (int k = 0; k < 8; ++k) {
            // A matrix (16x16 row-major)
            uint32_t addr_A = (uint32_t)__cvta_generic_to_shared(&smem_Q[warp * 16 + (lane % 16)][k * 16 + (lane / 16) * 8]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr_A));
                         
            for (int t = 0; t < 8; ++t) {
                // B matrix (16x8 col-major, loaded perfectly from K row-major layout)
                uint32_t addr_B = (uint32_t)__cvta_generic_to_shared(&smem_K[k * 16 + (lane % 16) % 8][t * 8 + ((lane % 16) / 8) * 8]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                             : "=r"(b[0]), "=r"(b[1]) : "r"(addr_B));
                             
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                             : "+f"(S_ij[t][0]), "+f"(S_ij[t][1]), "+f"(S_ij[t][2]), "+f"(S_ij[t][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                               "r"(b[0]), "r"(b[1]));
            }
        }
        
        // Softmax and scaling logic
        float m_new[2] = {-CUDART_INF_F, -CUDART_INF_F};
        for (int t = 0; t < 8; ++t) {
            int c0 = t * 8 + (lane % 4) * 2;
            int c1 = c0 + 1;
            
            S_ij[t][0] *= scale;
            S_ij[t][1] *= scale;
            S_ij[t][2] *= scale;
            S_ij[t][3] *= scale;
            
            // Mask out of bounds elements directly across sequence length
            if (j_global_row + c0 >= S) { S_ij[t][0] = -CUDART_INF_F; S_ij[t][2] = -CUDART_INF_F; }
            if (j_global_row + c1 >= S) { S_ij[t][1] = -CUDART_INF_F; S_ij[t][3] = -CUDART_INF_F; }
            
            m_new[0] = fmaxf(m_new[0], fmaxf(S_ij[t][0], S_ij[t][1]));
            m_new[1] = fmaxf(m_new[1], fmaxf(S_ij[t][2], S_ij[t][3]));
        }
        
        // Row max reduction within the warp threads mapping to the same row
        m_new[0] = fmaxf(m_new[0], __shfl_xor_sync(0xffffffff, m_new[0], 1));
        m_new[0] = fmaxf(m_new[0], __shfl_xor_sync(0xffffffff, m_new[0], 2));
        
        m_new[1] = fmaxf(m_new[1], __shfl_xor_sync(0xffffffff, m_new[1], 1));
        m_new[1] = fmaxf(m_new[1], __shfl_xor_sync(0xffffffff, m_new[1], 2));
        
        m_new[0] = fmaxf(m_new[0], m[0]);
        m_new[1] = fmaxf(m_new[1], m[1]);
        
        float exp_diff[2] = {expf(m[0] - m_new[0]), expf(m[1] - m_new[1])};
        
        // Apply scaling factor to accrued O elements
        for (int t_v = 0; t_v < 16; ++t_v) {
            O_acc[t_v][0] *= exp_diff[0];
            O_acc[t_v][1] *= exp_diff[0];
            O_acc[t_v][2] *= exp_diff[1];
            O_acc[t_v][3] *= exp_diff[1];
        }
        
        float p_sum[2] = {0.0f, 0.0f};
        float P[8][4];
        
        for (int t = 0; t < 8; ++t) {
            P[t][0] = expf(S_ij[t][0] - m_new[0]);
            P[t][1] = expf(S_ij[t][1] - m_new[0]);
            P[t][2] = expf(S_ij[t][2] - m_new[1]);
            P[t][3] = expf(S_ij[t][3] - m_new[1]);
            
            p_sum[0] += P[t][0] + P[t][1];
            p_sum[1] += P[t][2] + P[t][3];
            
            int r0 = warp * 16 + lane / 4;
            int r1 = r0 + 8;
            int c0 = t * 8 + (lane % 4) * 2;
            int c1 = c0 + 1;
            
            // Stash P tile into SMEM (ready to be multiplied by V)
            smem_P[r0][c0] = __float2bfloat16(P[t][0]);
            smem_P[r0][c1] = __float2bfloat16(P[t][1]);
            smem_P[r1][c0] = __float2bfloat16(P[t][2]);
            smem_P[r1][c1] = __float2bfloat16(P[t][3]);
        }
        
        // Row sum reduction
        p_sum[0] += __shfl_xor_sync(0xffffffff, p_sum[0], 1);
        p_sum[0] += __shfl_xor_sync(0xffffffff, p_sum[0], 2);
        
        p_sum[1] += __shfl_xor_sync(0xffffffff, p_sum[1], 1);
        p_sum[1] += __shfl_xor_sync(0xffffffff, p_sum[1], 2);
        
        l[0] = l[0] * exp_diff[0] + p_sum[0];
        l[1] = l[1] * exp_diff[1] + p_sum[1];
        
        m[0] = m_new[0];
        m[1] = m_new[1];
        
        // P @ V computation 
        for (int k_p = 0; k_p < 4; ++k_p) {
            uint32_t addr_A = (uint32_t)__cvta_generic_to_shared(&smem_P[warp * 16 + (lane % 16)][k_p * 16 + (lane / 16) * 8]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr_A));
                         
            for (int t_v = 0; t_v < 16; ++t_v) {
                // V loaded as 16x8 row-major directly 
                uint32_t addr_B = (uint32_t)__cvta_generic_to_shared(&smem_V[k_p * 16 + (lane % 16)][t_v * 8]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                             : "=r"(b[0]), "=r"(b[1]) : "r"(addr_B));
                             
                asm volatile("mma.sync.aligned.m16n8k16.row.row.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                             : "+f"(O_acc[t_v][0]), "+f"(O_acc[t_v][1]), "+f"(O_acc[t_v][2]), "+f"(O_acc[t_v][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                               "r"(b[0]), "r"(b[1]));
            }
        }
    }
    
    int r0 = warp * 16 + lane / 4;
    int r1 = r0 + 8;
    
    // Divide the accumulated output by total sequence sum elements
    for (int t_v = 0; t_v < 16; ++t_v) {
        int c0 = t_v * 8 + (lane % 4) * 2;
        int c1 = c0 + 1;
        
        float val00 = O_acc[t_v][0] / l[0];
        float val01 = O_acc[t_v][1] / l[0];
        float val10 = O_acc[t_v][2] / l[1];
        float val11 = O_acc[t_v][3] / l[1];
        
        if (global_row + r0 < S) {
            O_gmem[(global_row + r0) * 128 + c0] = __float2bfloat16(val00);
            O_gmem[(global_row + r0) * 128 + c1] = __float2bfloat16(val01);
        }
        if (global_row + r1 < S) {
            O_gmem[(global_row + r1) * 128 + c0] = __float2bfloat16(val10);
            O_gmem[(global_row + r1) * 128 + c1] = __float2bfloat16(val11);
        }
    }
    
    // Store log sum exp block
    if (lane % 4 == 0) {
        if (global_row + r0 < S) {
            LSE_gmem[global_row + r0] = m[0] + logf(l[0]);
        }
        if (global_row + r1 < S) {
            LSE_gmem[global_row + r1] = m[1] + logf(l[1]);
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
    
    int blocks_x = (S + 63) / 64;
    dim3 blocks(blocks_x, H, B);
    dim3 threads(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    // 60KB Dynamic Shared Memory requirement for Q, K, V, P
    CUDA_CHECK(cudaFuncSetAttribute(MHA_Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 61440));
    
    MHA_Kernel<<<blocks, threads, 61440, stream>>>(
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