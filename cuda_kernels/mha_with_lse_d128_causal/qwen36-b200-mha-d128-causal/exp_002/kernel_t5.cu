#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <cfloat>
#include <cmath>
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

namespace mha_blackwell {

constexpr int BM = 64;   // Query rows per CTA
constexpr int BN = 64;   // KV cols per tile
constexpr int BD = 128;  // Head dimension
constexpr int TPB = 128; // Threads per block
constexpr int TRR = TPB / BM;      // 2 threads per query row
constexpr int DPT = BD / TRR;      // 64 D-elements per thread

__global__ void mha_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
          __nv_bfloat16* __restrict__ O,
          float*         __restrict__ LSE,
    int B, int H, int S, int D,
    float inv_sqrt_d)
{
    extern __shared__ char smem[];
    
    __nv_bfloat16* __restrict__ sK = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* __restrict__ sV = reinterpret_cast<__nv_bfloat16*>(smem + BN * BD * sizeof(__nv_bfloat16));
    
    int tid = threadIdx.x;
    int q_local = tid / TRR;     
    int lane = tid % TRR;         
    int d_off = lane * DPT;       
    
    int bh_idx = blockIdx.x;
    int b = bh_idx / H;
    int h = bh_idx % H;
    int q_tile = blockIdx.y;
    
    int qr = q_tile * BM + q_local;
    bool valid_q = qr < S;
    
    int stride_SD = S * D;
    int stride_BHD = H * stride_SD;
    int stride_LSE = H * S;
    
    const __nv_bfloat16* Q_base = Q + b * stride_BHD + h * stride_SD;
    const __nv_bfloat16* K_base = K + b * stride_BHD + h * stride_SD;
    const __nv_bfloat16* V_base = V + b * stride_BHD + h * stride_SD;
    __nv_bfloat16* O_base = O + b * stride_BHD + h * stride_SD;
    float* LSE_base = LSE + b * stride_LSE + h * S;
    
    // Load Q fragment
    float q[DPT];
    if (valid_q) {
        const __nv_bfloat16* q_ptr = Q_base + qr * D + d_off;
        #pragma unroll
        for (int i = 0; i < DPT; ++i) {
            q[i] = __bfloat162float(q_ptr[i]) * inv_sqrt_d;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < DPT; ++i) q[i] = 0.0f;
    }
    
    float o_acc[DPT];
    #pragma unroll
    for (int i = 0; i < DPT; ++i) o_acc[i] = 0.0f;
    
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    
    int num_k_tiles = (S + BN - 1) / BN;
    unsigned mask = 0xFFFFFFFFu;
    
    for (int tk = 0; tk < num_k_tiles; ++tk) {
        int k_start = tk * BN;
        
        // Cooperative load K
        for (int i = tid; i < BN * BD; i += TPB) {
            int kn = i / BD;
            int kd = i % BD;
            int kg = k_start + kn;
            if (kg < S) {
                sK[i] = K_base[kg * D + kd];
            } else {
                sK[i] = __float2bfloat16(0.0f);
            }
        }
        
        // Cooperative load V
        for (int i = tid; i < BN * BD; i += TPB) {
            int vn = i / BD;
            int vd = i % BD;
            int vg = k_start + vn;
            if (vg < S) {
                sV[i] = V_base[vg * D + vd];
            } else {
                sV[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
        
        // Compute Q@K^T scores and find tile max
        float tile_max = -FLT_MAX;
        float scores[BK];
        
        const int k_end = min(k_start + BN, S);
        const int first_key = max(k_start, 0);
        const int last_key = valid_q ? min(k_end, qr) : 0;
        const int nk = max(0, last_key - first_key);
        
        for (int kc = 0; kc < BN; ++kc) {
            float partial = 0.0f;
            const __nv_bfloat16* kcol = sK + kc * BD + d_off;
            
            #pragma unroll
            for (int di = 0; di < DPT; di += 4) {
                partial += q[di] * __bfloat162float(kcol[di]);
                partial += q[di+1] * __bfloat162float(kcol[di+1]);
                partial += q[di+2] * __bfloat162float(kcol[di+2]);
                partial += q[di+3] * __bfloat162float(kcol[di+3]);
            }
            
            // Warp reduce (2-way)
            float score = partial + __shfl_down_sync(mask, partial, 1);
            scores[kc] = score;
            
            if (kc < nk && (score > tile_max)) {
                tile_max = score;
            }
        }
        
        // Online softmax update
        float old_max = row_max;
        if (nk > 0 && tile_max > row_max) {
            row_max = tile_max;
        }
        
        float alpha = (nk > 0 && old_max > -FLT_MAX * 0.5f && row_max != old_max) 
                      ? expf(old_max - row_max) : 1.0f;
        
        if (alpha < 1.0f) {
            row_sum *= alpha;
            #pragma unroll
            for (int i = 0; i < DPT; ++i) o_acc[i] *= alpha;
        }
        
        // Accumulate weighted V contributions
        if (nk > 0) {
            for (int kc = 0; kc < nk; ++kc) {
                float p = expf(scores[kc] - row_max);
                row_sum += p;
                
                const __nv_bfloat16* vcol = sV + kc * BD + d_off;
                #pragma unroll
                for (int di = 0; di < DPT; di += 4) {
                    o_acc[di]   += p * __bfloat162float(vcol[di]);
                    o_acc[di+1] += p * __bfloat162float(vcol[di+1]);
                    o_acc[di+2] += p * __bfloat162float(vcol[di+2]);
                    o_acc[di+3] += p * __bfloat162float(vcol[di+3]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write-back output and LSE
    if (valid_q) {
        __nv_bfloat16* o_ptr = O_base + qr * D + d_off;
        if (row_sum > 0.0f) {
            float norm = 1.0f / row_sum;
            LSE_base[qr] = row_max + logf(row_sum);
            #pragma unroll
            for (int i = 0; i < DPT; i += 4) {
                o_ptr[i]   = __float2bfloat16(o_acc[i] * norm);
                o_ptr[i+1] = __float2bfloat16(o_acc[i+1] * norm);
                o_ptr[i+2] = __float2bfloat16(o_acc[i+2] * norm);
                o_ptr[i+3] = __float2bfloat16(o_acc[i+3] * norm);
            }
        } else {
            LSE_base[qr] = -FLT_MAX;
            #pragma unroll
            for (int i = 0; i < DPT; ++i) {
                o_ptr[i] = __float2bfloat16(0.0f);
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
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());
    
    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(D));
    
    int64_t num_bh = B * H;
    int64_t num_qtiles = (S + BM - 1) / BM;
    
    dim3 grid(static_cast<unsigned int>(num_bh), static_cast<unsigned int>(num_qtiles));
    dim3 block(TPB);
    
    size_t smem_size = 2ULL * BN * BD * sizeof(__nv_bfloat16);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_causal_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S),
        static_cast<int>(D), inv_sqrt_d);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_blackwell::run);

}  // namespace mha_blackwell