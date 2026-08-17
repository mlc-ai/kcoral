#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <math.h>
#include <stdio.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

__global__ void AttentionKernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S) 
{
    int bh = blockIdx.x;
    int s_block = blockIdx.y;
    int s_start = s_block * 128;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_m = warp_id * 32;
    
    // Total dynamically allocated shared memory: ~98.8 KB
    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem = (__nv_bfloat16*)smem;
    __nv_bfloat16* K_smem = Q_smem + 128 * 128;
    __nv_bfloat16* V_smem = K_smem + 64 * 128;
    float* S_smem = (float*)(V_smem + 64 * 128);
    // Pad row length to 65 to avoid 32-way bank conflicts during column-wise reads
    float* warp_S_smem = S_smem + warp_id * 32 * 65; 
    
    // Load Q tile (128 x 128)
    uint4* Q_gmem_u4 = (uint4*)(Q + bh * S * 128 + s_start * 128);
    uint4* Q_smem_u4 = (uint4*)Q_smem;
    for (int i = 0; i < 16; i++) {
        int offset = i * 128 + tid;
        int row = offset / 16;
        if (s_start + row < S) {
            Q_smem_u4[offset] = Q_gmem_u4[offset];
        } else {
            uint4 zero4 = {0, 0, 0, 0};
            Q_smem_u4[offset] = zero4;
        }
    }
    
    __syncthreads();
    
    float O_i[128];
    for (int d = 0; d < 128; d++) {
        O_i[d] = 0.0f;
    }
    
    // Using a large negative value rather than actual -INFINITY to avoid NaN during `expf` subtractions
    float m_i = -1e20f;
    float l_i = 0.0f;
    float scale = 1.0f / sqrtf(128.0f);
    
    for (int k_start = 0; k_start < S; k_start += 64) {
        
        // Load K tile (64 x 128)
        uint4* K_gmem_u4 = (uint4*)(K + bh * S * 128 + k_start * 128);
        uint4* K_smem_u4 = (uint4*)K_smem;
        for (int i = 0; i < 8; i++) {
            int offset = i * 128 + tid;
            int row = offset / 16;
            if (k_start + row < S) {
                K_smem_u4[offset] = K_gmem_u4[offset];
            } else {
                uint4 zero4 = {0, 0, 0, 0};
                K_smem_u4[offset] = zero4;
            }
        }
        
        // Load V tile (64 x 128)
        uint4* V_gmem_u4 = (uint4*)(V + bh * S * 128 + k_start * 128);
        uint4* V_smem_u4 = (uint4*)V_smem;
        for (int i = 0; i < 8; i++) {
            int offset = i * 128 + tid;
            int row = offset / 16;
            if (k_start + row < S) {
                V_smem_u4[offset] = V_gmem_u4[offset];
            } else {
                uint4 zero4 = {0, 0, 0, 0};
                V_smem_u4[offset] = zero4;
            }
        }
        
        __syncthreads();
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_S[2][4];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                wmma::fill_fragment(frag_S[i][j], 0.0f);
            }
        }
        
        // Accumulate Dot Product internally
        for (int k = 0; k < 8; k++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_Q[2];
            wmma::load_matrix_sync(frag_Q[0], &Q_smem[warp_m * 128 + k * 16], 128);
            wmma::load_matrix_sync(frag_Q[1], &Q_smem[(warp_m + 16) * 128 + k * 16], 128);
            
            // K^T is effectively accessed with col_major layout given that K is stored row_major contiguous
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> frag_K[4];
            for (int j = 0; j < 4; j++) {
                wmma::load_matrix_sync(frag_K[j], &K_smem[j * 16 * 128 + k * 16], 128);
            }
            
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 4; j++) {
                    wmma::mma_sync(frag_S[i][j], frag_Q[i], frag_K[j], frag_S[i][j]);
                }
            }
        }
        
        // Store computed values back to Shared Memory
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                wmma::store_matrix_sync(&warp_S_smem[i * 16 * 65 + j * 16], frag_S[i][j], 65, wmma::mem_row_major);
            }
        }
        
        __syncwarp();
        
        float m_curr = -1e20f;
        float row_S[64];
        int valid_k = S - k_start;
        if (valid_k > 64) valid_k = 64;
        
        // Safely extract the max-per-row and scale
        for (int j = 0; j < 64; j++) {
            if (j < valid_k) {
                row_S[j] = warp_S_smem[lane_id * 65 + j] * scale;
            } else {
                row_S[j] = -1e20f;
            }
            if (row_S[j] > m_curr) {
                m_curr = row_S[j];
            }
        }
        
        float m_new = fmaxf(m_i, m_curr);
        float scale_old = expf(m_i - m_new);
        float l_curr = 0.0f;
        
        for (int j = 0; j < 64; j++) {
            float p = expf(row_S[j] - m_new);
            row_S[j] = p;
            l_curr += p;
        }
        
        float l_new = l_i * scale_old + l_curr;
        
        for (int d = 0; d < 128; d++) {
            O_i[d] *= scale_old;
        }
        
        // Compute S * V
        for (int j = 0; j < 64; j++) {
            float p = row_S[j];
            uint4* v_ptr = (uint4*)&V_smem[j * 128];
            for (int d4 = 0; d4 < 128 / 8; d4++) {
                union { uint4 u; __nv_bfloat162 v[4]; } conv;
                conv.u = v_ptr[d4];
                
                float2 f0 = __bfloat1622float2(conv.v[0]);
                float2 f1 = __bfloat1622float2(conv.v[1]);
                float2 f2 = __bfloat1622float2(conv.v[2]);
                float2 f3 = __bfloat1622float2(conv.v[3]);
                
                int d_base = d4 * 8;
                O_i[d_base + 0] += p * f0.x;
                O_i[d_base + 1] += p * f0.y;
                O_i[d_base + 2] += p * f1.x;
                O_i[d_base + 3] += p * f1.y;
                O_i[d_base + 4] += p * f2.x;
                O_i[d_base + 5] += p * f2.y;
                O_i[d_base + 6] += p * f3.x;
                O_i[d_base + 7] += p * f3.y;
            }
        }
        
        m_i = m_new;
        l_i = l_new;
        
        __syncthreads();
    }
    
    int q_idx = s_start + warp_id * 32 + lane_id;
    if (q_idx < S) {
        float inv_l = 1.0f / l_i;
        for (int d = 0; d < 128; d++) {
            O_i[d] *= inv_l;
        }
        
        __nv_bfloat16* O_out = O + bh * S * 128 + q_idx * 128;
        for (int d4 = 0; d4 < 128 / 8; d4++) {
            float2 f0 = {O_i[d4 * 8 + 0], O_i[d4 * 8 + 1]};
            float2 f1 = {O_i[d4 * 8 + 2], O_i[d4 * 8 + 3]};
            float2 f2 = {O_i[d4 * 8 + 4], O_i[d4 * 8 + 5]};
            float2 f3 = {O_i[d4 * 8 + 6], O_i[d4 * 8 + 7]};
            
            union { __nv_bfloat162 v[4]; uint4 u; } conv;
            conv.v[0] = __float22bfloat162_rn(f0);
            conv.v[1] = __float22bfloat162_rn(f1);
            conv.v[2] = __float22bfloat162_rn(f2);
            conv.v[3] = __float22bfloat162_rn(f3);
            
            ((uint4*)O_out)[d4] = conv.u;
        }
        
        float lse = m_i + logf(l_i);
        LSE[bh * S + q_idx] = lse;
    }
}

namespace tvm_ffi_mha {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
             
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());
    
    dim3 block(128, 1, 1);
    dim3 grid(B * H, (S + 127) / 128, 1);
    int shared_mem_size = 100 * 1024;
    
    CUDA_CHECK(cudaFuncSetAttribute(AttentionKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
    
    AttentionKernel<<<grid, block, shared_mem_size, stream>>>(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, S);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

}  // namespace tvm_ffi_mha