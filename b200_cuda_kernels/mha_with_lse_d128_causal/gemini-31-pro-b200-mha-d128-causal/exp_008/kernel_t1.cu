#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <mma.h>
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

using namespace nvcuda;

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
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    
    int64_t batch_head_offset = (int64_t)b * gridDim.y * seq_len * 128 + (int64_t)h * seq_len * 128;
    const __nv_bfloat16* q_ptr = Q + batch_head_offset;
    const __nv_bfloat16* k_ptr = K + batch_head_offset;
    const __nv_bfloat16* v_ptr = V + batch_head_offset;
    __nv_bfloat16* o_ptr = O + batch_head_offset;
    float* lse_ptr = LSE + (int64_t)b * gridDim.y * seq_len + (int64_t)h * seq_len;
    
    // Allocate 114,944 bytes of dynamic shared memory. All sizes padded to avoid bank conflicts.
    extern __shared__ __align__(16) char smem_dyn[];
    __nv_bfloat16 (*Q_s)[136] = (__nv_bfloat16 (*)[136])(smem_dyn);
    __nv_bfloat16 (*K_s)[136] = (__nv_bfloat16 (*)[136])(smem_dyn + 17408);
    __nv_bfloat16 (*V_s)[136] = (__nv_bfloat16 (*)[136])(smem_dyn + 34816);
    float (*O_s)[136] = (float (*)[136])(smem_dyn + 52224);
    float (*S_s)[72] = (float (*)[72])(smem_dyn + 87040);
    __nv_bfloat16 (*P_s)[72] = (__nv_bfloat16 (*)[72])(smem_dyn + 105472);
    float *l_s = (float *)(smem_dyn + 114688);
    
    // Load Q block (64x128) into SMEM
    #pragma unroll
    for (int i = tid; i < 64 * 128 / 8; i += blockDim.x) {
        int row = i / 16;
        int col = (i % 16) * 8;
        if (q_start + row < seq_len) {
            *(int4*)&Q_s[row][col] = *(const int4*)&q_ptr[(q_start + row) * 128 + col];
        } else {
            *(int4*)&Q_s[row][col] = make_int4(0,0,0,0);
        }
    }
    
    // Zero-initialize O accumulator in SMEM
    #pragma unroll
    for (int i = tid; i < 64 * 128; i += blockDim.x) {
        O_s[i / 128][i % 128] = 0.0f;
    }
    
    // Local Softmax Statistics
    float m_i = -INFINITY;
    float l_i = 0.0f;
    
    int warp_row_s = (warp_id / 2) * 32;
    int warp_col_s = (warp_id % 2) * 32;
    
    int warp_row_o = (warp_id / 2) * 32;
    int warp_col_o = (warp_id % 2) * 64;
    
    __syncthreads();
    
    for (int k_start = 0; k_start <= q_start + 63 && k_start < seq_len; k_start += 64) {
        // Load K and V blocks (64x128) into SMEM
        #pragma unroll
        for (int i = tid; i < 64 * 128 / 8; i += blockDim.x) {
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
        
        // 1. WMMA: S = Q * K^T
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_q[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> frag_k[2];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
        
        wmma::fill_fragment(acc[0][0], 0.0f);
        wmma::fill_fragment(acc[0][1], 0.0f);
        wmma::fill_fragment(acc[1][0], 0.0f);
        wmma::fill_fragment(acc[1][1], 0.0f);
        
        for (int step = 0; step < 128; step += 16) {
            wmma::load_matrix_sync(frag_q[0], &Q_s[warp_row_s][step], 136);
            wmma::load_matrix_sync(frag_q[1], &Q_s[warp_row_s + 16][step], 136);
            
            // K is transposed automatically by loading it as col_major from its row-major storage format
            wmma::load_matrix_sync(frag_k[0], &K_s[warp_col_s][step], 136);
            wmma::load_matrix_sync(frag_k[1], &K_s[warp_col_s + 16][step], 136);
            
            wmma::mma_sync(acc[0][0], frag_q[0], frag_k[0], acc[0][0]);
            wmma::mma_sync(acc[0][1], frag_q[0], frag_k[1], acc[0][1]);
            wmma::mma_sync(acc[1][0], frag_q[1], frag_k[0], acc[1][0]);
            wmma::mma_sync(acc[1][1], frag_q[1], frag_k[1], acc[1][1]);
        }
        
        wmma::store_matrix_sync(&S_s[warp_row_s][warp_col_s], acc[0][0], 72, wmma::mem_row_major);
        wmma::store_matrix_sync(&S_s[warp_row_s][warp_col_s + 16], acc[0][1], 72, wmma::mem_row_major);
        wmma::store_matrix_sync(&S_s[warp_row_s + 16][warp_col_s], acc[1][0], 72, wmma::mem_row_major);
        wmma::store_matrix_sync(&S_s[warp_row_s + 16][warp_col_s + 16], acc[1][1], 72, wmma::mem_row_major);
        
        __syncthreads();
        
        // 2. Local Softmax
        if (tid < 64) {
            float max_val = -INFINITY;
            for (int c = 0; c < 64; ++c) {
                int g_q = q_start + tid;
                int g_k = k_start + c;
                float val = S_s[tid][c];
                
                // Causal mask & Out-of-bounds check
                if (g_k > g_q || g_k >= seq_len || g_q >= seq_len) {
                    val = -INFINITY;
                } else {
                    val *= 0.0883883476f; // Scale by 1/sqrt(128)
                }
                S_s[tid][c] = val;
                max_val = fmaxf(max_val, val);
            }
            
            float m_new = fmaxf(m_i, max_val);
            float exp_diff = (m_i == -INFINITY) ? 0.0f : expf(m_i - m_new);
            l_i *= exp_diff;
            
            // Rescale previously accumulated output O
            for (int c = 0; c < 128; ++c) {
                O_s[tid][c] *= exp_diff;
            }
            
            float sum_val = 0.0f;
            for (int c = 0; c < 64; ++c) {
                float e = 0.0f;
                if (m_new != -INFINITY) {
                    e = expf(S_s[tid][c] - m_new);
                }
                P_s[tid][c] = __float2bfloat16(e); // Store to P for the next WMMA step
                sum_val += e;
            }
            l_i += sum_val;
            m_i = m_new;
            l_s[tid] = l_i; // Write final l_i to SMEM for the normalizer step
        }
        
        __syncthreads();
        
        // 3. WMMA: O = P * V
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_p[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_v[4];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_acc[2][4];
        
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 4; ++j) {
                wmma::load_matrix_sync(o_acc[i][j], &O_s[warp_row_o + i * 16][warp_col_o + j * 16], 136, wmma::mem_row_major);
            }
        }
        
        for (int step = 0; step < 64; step += 16) {
            wmma::load_matrix_sync(frag_p[0], &P_s[warp_row_o][step], 72);
            wmma::load_matrix_sync(frag_p[1], &P_s[warp_row_o + 16][step], 72);
            
            for (int j = 0; j < 4; ++j) {
                wmma::load_matrix_sync(frag_v[j], &V_s[step][warp_col_o + j * 16], 136);
            }
            
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 4; ++j) {
                    wmma::mma_sync(o_acc[i][j], frag_p[i], frag_v[j], o_acc[i][j]);
                }
            }
        }
        
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 4; ++j) {
                wmma::store_matrix_sync(&O_s[warp_row_o + i * 16][warp_col_o + j * 16], o_acc[i][j], 136, wmma::mem_row_major);
            }
        }
        
        __syncthreads(); // Prevent overwriting V_s before it's completely consumed
    }
    
    // Normalize and Write Out O
    #pragma unroll
    for (int i = tid; i < 64 * 128; i += blockDim.x) {
        int row = i / 128;
        int col = i % 128;
        if (q_start + row < seq_len) {
            float out_val = O_s[row][col] / l_s[row];
            o_ptr[(q_start + row) * 128 + col] = __float2bfloat16(out_val);
        }
    }
    
    // Write out LSE
    if (tid < 64) {
        if (q_start + tid < seq_len) {
            lse_ptr[q_start + tid] = m_i + logf(l_i);
        }
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
    
    int smem_size = 114944;
    CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    mha_fwd_kernel<<<blocks, threads, smem_size, stream>>>(q_data, k_data, v_data, o_data, lse_data, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha