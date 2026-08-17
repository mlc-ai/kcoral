#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <float.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace mha_impl {

template<int BM, int BN, int BD, int DN>
__global__ void mha_kernel(
    const __nv_bfloat16* __restrict__ Q, 
    const __nv_bfloat16* __restrict__ K, 
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O, 
    float* __restrict__ LSE,
    int B, int H, int S, int D,
    int stride_Q_s, int stride_Q_h,
    int stride_K_s, int stride_K_h,
    int stride_V_s, int stride_V_h,
    int stride_O_s, int stride_O_h,
    int stride_LSE_s, int stride_LSE_h)
{
    constexpr int NUM_THREADS = 256;
    constexpr int ELEMS = (BM * DN) / NUM_THREADS; 
    
    const uint32_t bh_id = blockIdx.x;
    const uint32_t bm_block = blockIdx.y;
    const uint32_t dn_block = blockIdx.z;
    const uint32_t tid = threadIdx.x;
    
    const int b = bh_id / H;
    const int h = bh_id % H;
    
    const __nv_bfloat16* Q_ptr = Q + b * stride_Q_h + h * stride_Q_h;
    const __nv_bfloat16* K_ptr = K + b * stride_K_h + h * stride_K_h;
    const __nv_bfloat16* V_ptr = V + b * stride_V_h + h * stride_V_h;
    __nv_bfloat16* O_ptr = O + b * stride_O_h + h * stride_O_h;
    float* LSE_ptr = LSE + b * stride_LSE_h + h * stride_LSE_h;
    
    extern __shared__ char smem[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK = sQ + BM * BD;
    __nv_bfloat16* sV = sK + BN * BD;
    
    const int base_j = dn_block * DN + (tid % DN);
    const int base_i = tid / DN;
    const int row_stride = NUM_THREADS / DN;
    
    float o_reg[ELEMS] = {};
    float m_reg[ELEMS] = {-FLT_MAX};
    float d_reg[ELEMS] = {0.0f};
    float scores[BN] = {};
    
    const float inv_sqrt_d = rsqrtf(static_cast<float>(D));
    const int m_start = bm_block * BM;
    const bool m_valid = (m_start < S);
    
    // Load Q tile
    if (m_valid) {
        for (int idx = tid; idx < BM * BD; idx += NUM_THREADS) {
            int i = idx / BD;
            int d = idx % BD;
            int g_i = m_start + i;
            if (g_i < S && d < D) {
                sQ[idx] = Q_ptr[g_i * stride_Q_s + d];
            } else {
                sQ[idx] = __float2bfloat16(0.0f);
            }
        }
    } else {
        for (int idx = tid; idx < BM * BD; idx += NUM_THREADS) {
            sQ[idx] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();
    
    const int n_blocks = (S + BN - 1) / BN;
    
    for (int nb = 0; nb < n_blocks; ++nb) {
        int n_start = nb * BN;
        
        // Load K and V tiles
        for (int idx = tid; idx < BN * BD; idx += NUM_THREADS) {
            int n = idx / BD;
            int d = idx % BD;
            int g_n = n_start + n;
            bool valid_n = (g_n < S);
            if (valid_n && d < D) {
                sK[idx] = K_ptr[g_n * stride_K_s + d];
                sV[idx] = V_ptr[g_n * stride_V_s + d];
            } else {
                sK[idx] = __float2bfloat16(0.0f);
                sV[idx] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
        
        // Process each element owned by this thread
        #pragma unroll
        for (int e = 0; e < ELEMS; ++e) {
            int i = base_i + e * row_stride;
            if (!m_valid || (m_start + i) >= S) continue;
            
            // Compute QK^T scores for this row
            float row_max = -FLT_MAX;
            const __nv_bfloat16* q_r = sQ + i * BD;
            #pragma unroll
            for (int n = 0; n < BN; ++n) {
                float sum = 0.0f;
                const __nv_bfloat16* k_r = sK + n * BD;
                #pragma unroll
                for (int d = 0; d < BD; d += 4) {
                    sum += __bfloat162float(q_r[d])   * __bfloat162float(k_r[d]);
                    sum += __bfloat162float(q_r[d+1]) * __bfloat162float(k_r[d+1]);
                    sum += __bfloat162float(q_r[d+2]) * __bfloat162float(k_r[d+2]);
                    sum += __bfloat162float(q_r[d+3]) * __bfloat162float(k_r[d+3]);
                }
                float sc = (n_start + n < S) ? (sum * inv_sqrt_d) : (-FLT_MAX);
                scores[n] = sc;
                if (sc > row_max) row_max = sc;
            }
            
            // Online softmax update & PV accumulation
            float old_m = m_reg[e];
            float new_m = max(old_m, row_max);
            float alpha = expf(old_m - new_m);
            float new_sum = 0.0f;
            
            float o_acc = o_reg[e];
            const __nv_bfloat16* v_base = sV;
            
            #pragma unroll
            for (int n = 0; n < BN; ++n) {
                float p = expf(scores[n] - new_m);
                new_sum += p;
                
                float v_val = __bfloat162float(v_base[n * BD + base_j]);
                o_acc = fmaf(p, v_val, o_acc * alpha * d_reg[e]);
            }
            
            o_reg[e] = o_acc;
            m_reg[e] = new_m;
            d_reg[e] = d_reg[e] * alpha + new_sum;
        }
        __syncthreads();
    }
    
    // Epilogue: Normalize and store
    #pragma unroll
    for (int e = 0; e < ELEMS; ++e) {
        int i = base_i + e * row_stride;
        int g_i = m_start + i;
        if (g_i >= S) continue;
        
        if (d_reg[e] <= 0.0f) {
            O_ptr[g_i * stride_O_s + base_j] = __float2bfloat16(0.0f);
            LSE_ptr[g_i * stride_LSE_s] = -FLT_MAX;
            continue;
        }
        
        float inv_d = 1.0f / d_reg[e];
        float o_val = o_reg[e] * inv_d;
        float lse_val = m_reg[e] + logf(d_reg[e]);
        
        O_ptr[g_i * stride_O_s + base_j] = __float2bfloat16(o_val);
        
        // Each thread writes LSE for its owned rows. No conflicts since rows are partitioned.
        LSE_ptr[g_i * stride_LSE_s] = lse_val;
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    constexpr int BM = 64, BN = 32, BD = 128, DN = 32;
    
    dim3 grid(B * H, (S + BM - 1) / BM, (D + DN - 1) / DN);
    dim3 block(256);
    
    size_t smem_bytes = (BM + 2 * BN) * BD * sizeof(__nv_bfloat16);
    
    // Compute strides assuming contiguous tensors
    int stride_Q_s = static_cast<int>(D);
    int stride_Q_h = static_cast<int>(S * D);
    int stride_K_s = stride_Q_s;
    int stride_K_h = stride_Q_h;
    int stride_V_s = stride_Q_s;
    int stride_V_h = stride_Q_h;
    int stride_O_s = stride_Q_s;
    int stride_O_h = stride_Q_h;
    int stride_LSE_s = 1;
    int stride_LSE_h = static_cast<int>(S);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_kernel<BM, BN, BD, DN><<<grid, block, smem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        B, H, S, D,
        stride_Q_s, stride_Q_h,
        stride_K_s, stride_K_h,
        stride_V_s, stride_V_h,
        stride_O_s, stride_O_h,
        stride_LSE_s, stride_LSE_h
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);

} // namespace mha_impl