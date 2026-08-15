#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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
    static_assert((BM * DN) % NUM_THREADS == 0, "Tile dimensions must be multiples of block size");
    constexpr int ELEMS = (BM * DN) / NUM_THREADS;

    const uint32_t bh_id = blockIdx.x;
    const uint32_t bm_idx = blockIdx.y;
    const uint32_t dn_idx = blockIdx.z;
    const uint32_t tid = threadIdx.x;

    const int b = bh_id / H;
    const int h = bh_id % H;

    const __nv_bfloat16* Q_base = Q + b * stride_Q_h + h * stride_Q_h;
    const __nv_bfloat16* K_base = K + b * stride_K_h + h * stride_K_h;
    const __nv_bfloat16* V_base = V + b * stride_V_h + h * stride_V_h;
    __nv_bfloat16* O_base = O + b * stride_O_h + h * stride_O_h;
    float* LSE_base = LSE + b * stride_LSE_h + h * stride_LSE_h;

    extern __shared__ char smem[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK = sQ + BM * D;
    __nv_bfloat16* sV = sK + BN * D;

    // Precompute row/col indices for each element handled by this thread
    int r_arr[ELEMS], c_arr[ELEMS];
    for(int e = 0; e < ELEMS; ++e) {
        int idx = tid * ELEMS + e;
        r_arr[e] = idx / DN;
        c_arr[e] = idx % DN;
    }

    // Online softmax state per element
    float o_reg[ELEMS] = {};
    float m_reg[ELEMS] = {-FLT_MAX};
    float d_reg[ELEMS] = {0.0f};
    float scores[BN] = {};

    const float inv_sqrt_d = rsqrtf(static_cast<float>(D));
    const int m_start = bm_idx * BM;
    const int n_start_base = dn_idx * DN;
    const bool m_valid = (m_start < S);
    const bool n_valid = (n_start_base < D);

    if (m_valid && n_valid) {
        // Load Q tile into shared memory
        for(int i = tid; i < BM * D; i += NUM_THREADS) {
            int r = i / D;
            int c = i % D;
            int g_r = m_start + r;
            if(g_r < S && c < D) {
                sQ[i] = Q_base[g_r * stride_Q_s + c];
            } else {
                sQ[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        const int num_s_blocks = (S + BN - 1) / BN;
        const int num_d_iters = (D + BD - 1) / BD;

        for(int sb = 0; sb < num_s_blocks; ++sb) {
            int s_offset = sb * BN;

            // Load K and V tiles into shared memory
            for(int i = tid; i < BN * D; i += NUM_THREADS) {
                int n = i / D;
                int d = i % D;
                int g_s = s_offset + n;
                bool valid_s = (g_s < S);
                if(valid_s && d < D) {
                    sK[i] = K_base[g_s * stride_K_s + d];
                    sV[i] = V_base[g_s * stride_V_s + d];
                } else {
                    sK[i] = __float2bfloat16(0.0f);
                    sV[i] = __float2bfloat16(0.0f);
                }
            }
            __syncthreads();

            #pragma unroll
            for(int e = 0; e < ELEMS; ++e) {
                int r = r_arr[e];
                int c = c_arr[e];
                
                // Compute QK^T scores and find row max
                float row_max = -FLT_MAX;
                #pragma unroll
                for(int n = 0; n < BN; ++n) {
                    float p = 0.0f;
                    const __nv_bfloat16* q_ptr = sQ + r * D;
                    const __nv_bfloat16* k_ptr = sK + n * D;
                    #pragma unroll
                    for(int di = 0; di < num_d_iters; ++di) {
                        int off = di * BD;
                        #pragma unroll
                        for(int dd = 0; dd < BD; dd += 4) {
                            p += __bfloat162float(q_ptr[off+dd])   * __bfloat162float(k_ptr[off+dd]);
                            p += __bfloat162float(q_ptr[off+dd+1]) * __bfloat162float(k_ptr[off+dd+1]);
                            p += __bfloat162float(q_ptr[off+dd+2]) * __bfloat162float(k_ptr[off+dd+2]);
                            p += __bfloat162float(q_ptr[off+dd+3]) * __bfloat162float(k_ptr[off+dd+3]);
                        }
                    }
                    float sc = (s_offset + n < S) ? (p * inv_sqrt_d) : (-FLT_MAX);
                    scores[n] = sc;
                    if(sc > row_max) row_max = sc;
                }

                // Online softmax update & PV accumulation
                float prev_m = m_reg[e];
                float prev_d = d_reg[e];
                float prev_o = o_reg[e];
                
                float new_m = max(prev_m, row_max);
                float alpha = expf(prev_m - new_m);
                float cur_sum = 0.0f;
                float cur_o = prev_o * alpha;
                
                #pragma unroll
                for(int n = 0; n < BN; ++n) {
                    float pn = expf(scores[n] - new_m);
                    cur_sum += pn;
                    cur_o += pn * __bfloat162float(sV[n * D + c]);
                }
                
                m_reg[e] = new_m;
                d_reg[e] = prev_d * alpha + cur_sum;
                o_reg[e] = cur_o;
            }
            __syncthreads();
        }

        // Epilogue: Normalize and store results
        #pragma unroll
        for(int e = 0; e < ELEMS; ++e) {
            int r = r_arr[e];
            int c = c_arr[e];
            int g_r = m_start + r;
            int g_c = n_start_base + c;
            
            if(g_r >= S || g_c >= D) continue;
            
            if(d_reg[e] > 0.0f) {
                float inv_d = 1.0f / d_reg[e];
                O_base[g_r * stride_O_s + g_c] = __float2bfloat16(o_reg[e] * inv_d);
                // Only one thread per row writes LSE to avoid contention
                if(tid % DN == 0) {
                    LSE_base[g_r * stride_LSE_s] = m_reg[e] + logf(d_reg[e]);
                }
            } else {
                O_base[g_r * stride_O_s + g_c] = __float2bfloat16(0.0f);
                if(tid % DN == 0) LSE_base[g_r * stride_LSE_s] = -FLT_MAX;
            }
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
    
    // Tiling parameters optimized for BF16 & Blackwell shared memory bandwidth
    constexpr int BM = 64, BN = 64, BD = 32, DN = 64;
    
    dim3 grid(B * H, (S + BM - 1) / BM, (D + DN - 1) / DN);
    dim3 block(256);
    size_t smem_bytes = (BM + 2 * BN) * D * sizeof(__nv_bfloat16);
    
    // Compute strides assuming standard contiguous NCHW layout: [B, H, S, D]
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
        stride_s, stride_h,
        stride_s, stride_h,
        stride_s, stride_h,
        stride_s, stride_h,
        stride_lse_s, stride_lse_h
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);

} // namespace mha_impl