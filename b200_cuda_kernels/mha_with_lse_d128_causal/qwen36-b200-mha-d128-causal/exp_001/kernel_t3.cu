#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) \
    do {                 \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1);     \
        }                \
    } while (0)

namespace attention_impl {

template <int BM, int BN, int NUM_THREADS>
__global__ void attn_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, int D,
    float scale_factor) 
{
    extern __shared__ char smem[];

    // Shared Memory Layout
    // s_Q_full: BM * D
    // s_K_tile: BN * D
    // s_V_tile: BN * D
    // s_O_acc : BM * D (FP32)
    // s_max   : BM (FP32)
    // s_sum   : BM (FP32)
    
    __nv_bfloat16* s_Q = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* s_K = s_Q + BM * D;
    __nv_bfloat16* s_V = s_K + BN * D;
    float* s_O = reinterpret_cast<float*>(s_V + BN * D);
    float* s_max = s_O + BM * D;
    float* s_sum = s_max + BM;

    int b = blockIdx.x / gridDim.y;
    int h = blockIdx.x % gridDim.y;
    int q_start = blockIdx.y * BM;
    
    int tid = threadIdx.x;
    int num_rows = min(q_start + BM, S) - q_start;

    // Initialize State
    for (int i = tid; i < num_rows; i += NUM_THREADS) {
        s_max[i] = -1e20f;
        s_sum[i] = 0.0f;
    }
    for (int i = tid; i < num_rows * D; i += NUM_THREADS) {
        s_O[i] = 0.0f;
    }
    __syncthreads();

    // Pre-calculate Base Offset for [B,H,S,D] -> ((b*H+h)*S)*D
    int64_t base_offset = ((int64_t)b * H + h) * S * D;
    
    // Load Q Full Tile: BM x D
    for (int i = tid; i < num_rows * D; i += NUM_THREADS) {
        int r = i / D;
        int c = i % D;
        if (q_start + r < S) {
             s_Q[i] = Q[base_offset + (q_start + r) * D + c];
        } else {
             s_Q[i] = __float2bfloat16(0.0f);
        }
    }
    
    // Handle partial block padding if num_rows < BM
    if (num_rows < BM) {
        for (int i = tid; i < (BM - num_rows) * D; i += NUM_THREADS) {
            int r = num_rows + (i / D);
            s_Q[i] = __float2bfloat16(0.0f); // Should remain 0 ideally, but explicit clear helps
        }
    }
    __syncthreads();

    int num_k_steps = (S + BN - 1) / BN;

    for (int k_step = 0; k_step < num_k_steps; ++k_step) {
        int k_start = k_step * BN;
        int eff_k_end = min(k_start + BN, S);
        int eff_bn = eff_k_end - k_start;

        // Load K and V Tiles: BN x D
        for (int i = tid; i < BN * D; i += NUM_THREADS) {
             int r = i / D;
             int c = i % D;
             if (r < eff_bn) {
                  int kr = k_start + r;
                  s_K[i] = K[base_offset + kr * D + c];
                  s_V[i] = V[base_offset + kr * D + c];
             } else {
                  s_K[i] = __float2bfloat16(0.0f);
                  s_V[i] = __float2bfloat16(0.0f);
             }
        }
        __syncthreads();

        // Compute QK^T, Softmax, and Accumulate O
        for (int r = tid; r < num_rows; r += NUM_THREADS) {
            float cur_max = -1e20f;
            float p_vals[64]; 
            
            // 1. Q @ K^T for this row
            for (int k_r = 0; k_r < BN; ++k_r) {
                float dot = 0.0f;
                float* q_ptr = s_Q + r * D;
                float* k_ptr = s_K + k_r * D;
                // Vectorized dot product
                #pragma unroll
                for(int d=0; d<D; d+=4){
                    dot += __bfloat162float(q_ptr[d])   * __bfloat162float(k_ptr[d]);
                    dot += __bfloat162float(q_ptr[d+1]) * __bfloat162float(k_ptr[d+1]);
                    dot += __bfloat162float(q_ptr[d+2]) * __bfloat162float(k_ptr[d+2]);
                    dot += __bfloat162float(q_ptr[d+3]) * __bfloat162float(k_ptr[d+3]);
                }
                
                int kr_global = k_start + k_r;
                int qr_global = q_start + r;
                
                if (kr_global > qr_global) {
                    dot = -1e20f; // Causal Mask
                }
                
                dot *= scale_factor;
                p_vals[k_r] = dot;
                if (dot > cur_max) cur_max = dot;
            }
            
            // 2. Online Softmax update
            float old_max = s_max[r];
            float new_max = fmaxf(old_max, cur_max);
            
            float alpha = expf(old_max - new_max);
            float beta = expf(cur_max - new_max);
            
            s_max[r] = new_max;
            
            float sum_p = 0.0f;
            float* o_ptr = s_O + r * D;
            
            // 3. O_acc += P @ V
            for(int k_r=0; k_r<BN; ++k_r){
                float p = expf(p_vals[k_r] - new_max);
                sum_p += p;
                
                float* v_ptr = s_V + k_r * D;
                #pragma unroll
                for(int d=0; d<D; d+=4){
                     float vf0 = __bfloat162float(v_ptr[d]);
                     float vf1 = __bfloat162float(v_ptr[d+1]);
                     float vf2 = __bfloat162float(v_ptr[d+2]);
                     float vf3 = __bfloat162float(v_ptr[d+3]);
                     o_ptr[d]   += p * vf0;
                     o_ptr[d+1] += p * vf1;
                     o_ptr[d+2] += p * vf2;
                     o_ptr[d+3] += p * vf3;
                }
            }
            
            float old_sum = s_sum[r];
            s_sum[r] = old_sum * alpha + sum_p * beta;
            
            // Scale existing O acc by alpha
            #pragma unroll
            for(int d=0; d<D; d+=4){
                o_ptr[d]   *= alpha;
                o_ptr[d+1] *= alpha;
                o_ptr[d+2] *= alpha;
                o_ptr[d+3] *= alpha;
            }
        }
        __syncthreads();
    }
    
    // Epilogue: Normalize and Store Output
    for (int i = tid; i < num_rows * D; i += NUM_THREADS) {
        int r = i / D;
        int c = i % D;
        
        float final_max = s_max[r];
        float final_sum = s_sum[r];
        float lse = final_max + logf(final_sum);
        float inv_sum = (final_sum > 1e-10f) ? 1.0f / final_sum : 0.0f;
        
        float o_val = s_O[i] * inv_sum;
        
        int out_idx = base_offset + (q_start + r) * D + c;
        O[out_idx] = __float2bfloat16(o_val);
        
        if (c == 0) {
            LSE[((int64_t)b * H + h) * S + (q_start + r)] = lse;
        }
    }
}

void run(tvm::ffi::TensorView Q_in, tvm::ffi::TensorView K_in, tvm::ffi::TensorView V_in, 
         tvm::ffi::TensorView O_out, tvm::ffi::TensorView LSE_out) {
    CUDA_CHECK(cudaSetDevice(Q_in.device().device_id));
    
    int64_t B = Q_in.size(0), H = Q_in.size(1), S = Q_in.size(2), D = Q_in.size(3);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q_in.device().device_type, Q_in.device().device_id));
    
    constexpr int BM = 64, BN = 64, NUM_THREADS = 256;
    
    dim3 grid(B * H, (S + BM - 1) / BM);
    dim3 block(NUM_THREADS);
    
    // Shared Memory Calculation
    // Q(BM*D) + K(BN*D) + V(BN*D) + O(BM*D float) + States(2*BM float)
    size_t smem_size = (BM * D + BN * D + BN * D) * sizeof(__nv_bfloat16) 
                     + BM * D * sizeof(float) + 2 * BM * sizeof(float);
                     
    float inv_sqrt_d = rsqrtf((float)D);
    
    // Explicit cast to void function pointer to avoid template instantiation errors during macro expansion
    void(*kernel_func)(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, 
                       __nv_bfloat16*, float*, int, int, int, int, float) = attn_fwd_kernel<BM, BN, NUM_THREADS>;

    kernel_func<<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q_in.data_ptr()),
        static_cast<const __nv_bfloat16*>(K_in.data_ptr()),
        static_cast<const __nv_bfloat16*>(V_in.data_ptr()),
        static_cast<__nv_bfloat16*>(O_out.data_ptr()),
        static_cast<float*>(LSE_out.data_ptr()),
        B, H, S, D, inv_sqrt_d
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attention_impl::run);

}  // namespace attention_impl