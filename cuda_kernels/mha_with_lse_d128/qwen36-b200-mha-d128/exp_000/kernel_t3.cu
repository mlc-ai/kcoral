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
    constexpr int ELEMS_PER_THREAD = (BM * DN) / NUM_THREADS; // 8
    
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
    
    const int base_r = tid / 32;
    const int base_c = tid % 32;
    const int r_stride = ELEMS_PER_THREAD;
    
    float o_acc[ELEMS_PER_THREAD] = {};
    float m_acc[ELEMS_PER_THREAD] = {-FLT_MAX};
    float d_acc[ELEMS_PER_THREAD] = {0.0f};
    float scores[BN] = {};
    
    const float inv_sqrt_d = rsqrtf(static_cast<float>(D));
    const int m_start = bm_block * BM;
    const int n_start_base = dn_block * DN;
    
    // Load Q tile (once)
    for (int idx = tid; idx < BM * BD; idx += NUM_THREADS) {
        int r = idx / BD;
        int d = idx % BD;
        int g_r = m_start + r;
        if (g_r < S && d < D) {
            sQ[idx] = Q_ptr[g_r * stride_Q_s + d];
        } else {
            sQ[idx] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();
    
    const int k_blocks = (D + BD - 1) / BD;
    
    for (int kb = 0; kb < k_blocks; ++kb) {
        int k_start = kb * BD;
        
        // Load K and V tiles
        for (int idx = tid; idx < BN * BD; idx += NUM_THREADS) {
            int n = idx / BD;
            int d = idx % BD;
            int g_n = kb == 0 ? (n_start_base + n) : (k_start + n); // Simplified: K/V tile along S
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
        
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
            int r = base_r + e * r_stride;
            int c = base_c;
            
            float row_max = -FLT_MAX;
            const __nv_bfloat16* q_row = sQ + r * BD;
            
            #pragma unroll
            for (int n = 0; n < BN; ++n) {
                float sum = 0.0f;
                const __nv_bfloat16* k_col = sK + n * BD;
                #pragma unroll
                for (int d = 0; d < BD; d += 4) {
                    sum += __bfloat162float(q_row[d])   * __bfloat162float(k_col[d]);
                    sum += __bfloat162float(q_row[d+1]) * __bfloat162float(k_col[d+1]);
                    sum += __bfloat162float(q_row[d+2]) * __bfloat162float(k_col[d+2]);
                    sum += __bfloat162float(q_row[d+3]) * __bfloat162float(k_col[d+3]);
                }
                float sc = (n_start_base + n < S) ? (sum * inv_sqrt_d) : (-FLT_MAX);
                scores[n] = sc;
                if (sc > row_max) row_max = sc;
            }
            
            float old_m = m_acc[e];
            float new_m = max(old_m, row_max);
            float alpha = expf(old_m - new_m);
            float new_sum = 0.0f;
            
            #pragma unroll
            for (int n = 0; n < BN; ++n) {
                float p = expf(scores[n] - new_m);
                new_sum += p;
                o_acc[e] = fmaf(p, __bfloat162float(sV[n * BD + c]), o_acc[e]);
            }
            o_acc[e] *= alpha * d_acc[e];
            m_acc[e] = new_m;
            d_acc[e] = d_acc[e] * alpha + new_sum;
        }
        __syncthreads();
    }
    
    // Epilogue
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        int r = base_r + e * r_stride;
        int c = base_c;
        int g_r = m_start + r;
        int g_c = n_start_base + c;
        
        if (g_r >= S || g_c >= D) continue;
        
        if (d_acc[e] <= 0.0f) {
            O_ptr[g_r * stride_O_s + g_c] = __float2bfloat16(0.0f);
            LSE_ptr[g_r * stride_LSE_s] = -FLT_MAX;
        } else {
            float inv_d = 1.0f / d_acc[e];
            O_ptr[g_r * stride_O_s + g_c] = __float2bfloat16(o_acc[e] * inv_d);
            LSE_ptr[g_r * stride_LSE_s] = m_acc[e] + logf(d_acc[e]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    constexpr int BM = 64, BN = 64, BD = 32, DN = 32;
    
    dim3 grid(B * H, (S + BM - 1) / BM, (D + DN - 1) / DN);
    dim3 block(256);
    size_t smem_bytes = (BM + 2 * BN) * BD * sizeof(__nv_bfloat16);
    
    int stride_s = static_cast<int>(D);
    int stride_h = static_cast<int>(S * D);
    int stride_lse_s = 1;
    int stride_lse_h = static_cast<int>(S);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_kernel<BM, BN, BD, DN><<<grid, block, smem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        B, H, S, D,
        stride_s, stride_h, stride_s, stride_h, stride_s, stride_h,
        stride_s, stride_h, stride_lse_s, stride_lse_h
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);

} // namespace mha_impl