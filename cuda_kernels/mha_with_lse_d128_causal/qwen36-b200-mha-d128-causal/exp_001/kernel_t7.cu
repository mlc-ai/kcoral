#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) \
    do {                 \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
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
    extern __shared__ char smem_byte[];

    // Shared Memory Layout (byte offsets)
    uint32_t off_Q = 0;
    uint32_t off_K = BM * D * sizeof(__nv_bfloat16);
    uint32_t off_V = off_K + BN * D * sizeof(__nv_bfloat16);
    uint32_t off_O = off_V + BN * D * sizeof(__nv_bfloat16);
    uint32_t off_m = off_O + BM * D * sizeof(float);
    uint32_t off_s = off_m + BM * sizeof(float);

    __nv_bfloat16* s_Q = reinterpret_cast<__nv_bfloat16*>(smem_byte + off_Q);
    __nv_bfloat16* s_K = reinterpret_cast<__nv_bfloat16*>(smem_byte + off_K);
    __nv_bfloat16* s_V = reinterpret_cast<__nv_bfloat16*>(smem_byte + off_V);
    float* s_O = reinterpret_cast<float*>(smem_byte + off_O);
    float* s_max = reinterpret_cast<float*>(smem_byte + off_m);
    float* s_sum = reinterpret_cast<float*>(smem_byte + off_s);

    int bh_idx = blockIdx.x;
    int b = bh_idx / H;
    int h = bh_idx % H;
    int q_start = blockIdx.y * BM;
    
    int tid = threadIdx.x;
    int num_rows = (q_start + BM < S) ? BM : (S - q_start);
    if (num_rows <= 0) return;

    // Initialize State
    for (int i = tid; i < num_rows; i += NUM_THREADS) {
        s_max[i] = -1e20f;
        s_sum[i] = 0.0f;
    }
    for (int i = tid; i < BM * D; i += NUM_THREADS) {
        s_O[i] = 0.0f;
    }
    __syncthreads();

    int64_t base_offset = ((int64_t)b * H + h) * S * D;
    
    // Load Q Full Tile: BM x D
    for (int i = tid; i < BM * D; i += NUM_THREADS) {
        int r = i / D;
        int c = i % D;
        if (q_start + r < S) {
             s_Q[i] = Q[base_offset + (q_start + r) * D + c];
        } else {
             s_Q[i] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    int num_k_steps = (S + BN - 1) / BN;

    for (int k_step = 0; k_step < num_k_steps; ++k_step) {
        int k_start = k_step * BN;
        int eff_bn = (k_start + BN < S) ? BN : (S - k_start);

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
            
            for (int k_r = 0; k_r < BN; ++k_r) {
                float dot = 0.0f;
                __nv_bfloat16* q_ptr = s_Q + r * D;
                __nv_bfloat16* k_ptr = s_K + k_r * D;
                
                for(int d = 0; d < D; d += 4){
                    dot += __bfloat162float(q_ptr[d])   * __bfloat162float(k_ptr[d]);
                    dot += __bfloat162float(q_ptr[d+1]) * __bfloat162float(k_ptr[d+1]);
                    dot += __bfloat162float(q_ptr[d+2]) * __bfloat162float(k_ptr[d+2]);
                    dot += __bfloat162float(q_ptr[d+3]) * __bfloat162float(k_ptr[d+3]);
                }
                
                // Causal Mask
                if (k_start + k_r > q_start + r) {
                    dot = -1e20f; 
                }
                
                dot *= scale_factor;
                p_vals[k_r] = dot;
                if (dot > cur_max) cur_max = dot;
            }
            
            // Online Softmax Update
            float old_max = s_max[r];
            float new_max = (old_max > cur_max) ? old_max : cur_max;
            float alpha = expf(old_max - new_max);
            s_max[r] = new_max;
            
            float sum_p = 0.0f;
            float* o_ptr = s_O + r * D;
            
            // O_acc += P @ V chunk
            for(int k_r = 0; k_r < BN; ++k_r){
                float p = expf(p_vals[k_r] - new_max);
                sum_p += p;
                
                __nv_bfloat16* v_ptr = s_V + k_r * D;
                for(int d = 0; d < D; d += 4){
                     o_ptr[d]   += p * __bfloat162float(v_ptr[d]);
                     o_ptr[d+1] += p * __bfloat162float(v_ptr[d+1]);
                     o_ptr[d+2] += p * __bfloat162float(v_ptr[d+2]);
                     o_ptr[d+3] += p * __bfloat162float(v_ptr[d+3]);
                }
            }
            
            // Update cumulative sum & scale previous accumulation
            s_sum[r] = s_sum[r] * alpha + sum_p;
            
            for(int d = 0; d < D; d += 4){
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
        float inv_sum = (final_sum > 1e-10f) ? (1.0f / final_sum) : 0.0f;
        
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
    
    dim3 grid((unsigned int)(B * H), (unsigned int)((S + BM - 1) / BM), 1);
    dim3 block(NUM_THREADS, 1, 1);
    
    // Shared Memory Calculation: Q(BM*D) + K(BN*D) + V(BN*D) + O(BM*D float) + States(2*BM float)
    size_t smem_size = (size_t)(BM * D + BN * D + BN * D) * sizeof(__nv_bfloat16) 
                     + (size_t)(BM * D) * sizeof(float) + 2 * BM * sizeof(float);
    smem_size = (smem_size + 1023) & ~1023ULL;
                     
    float inv_sqrt_d = rsqrtf((float)D);
    
    attn_fwd_kernel<BM, BN, NUM_THREADS><<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q_in.data_ptr()),
        static_cast<const __nv_bfloat16*>(K_in.data_ptr()),
        static_cast<const __nv_bfloat16*>(V_in.data_ptr()),
        static_cast<__nv_bfloat16*>(O_out.data_ptr()),
        static_cast<float*>(LSE_out.data_ptr()),
        (int)B, (int)H, (int)S, (int)D, inv_sqrt_d
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attention_impl::run);

}  // namespace attention_impl