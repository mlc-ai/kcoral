#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
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

namespace attention_impl {

template<int BM, int BN, int BK, int NUM_THREADS>
__global__ void attn_fwd_kernel(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* O,
    float* LSE,
    int B, int H, int S, int D,
    float inv_sqrt_d)
{
    extern __shared__ char smem[];
    
    // Shared memory layout
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem + BM * BK * sizeof(__nv_bfloat16));
    __nv_bfloat16* sV = reinterpret_cast<__nv_bfloat16*>(smem + (BM * BK + BN * BK) * sizeof(__nv_bfloat16));
    float* sO = reinterpret_cast<float*>(smem + (BM * BK + BN * BK + BN * D) * sizeof(__nv_bfloat16));
    float* s_m = reinterpret_cast<float*>(smem + (BM * BK + BN * BK + BN * D) * sizeof(__nv_bfloat16) + BM * D * sizeof(float));
    float* s_s = s_m + BM;

    int b = blockIdx.x / gridDim.y;
    int h = blockIdx.x % gridDim.y;
    int q_start = blockIdx.y * BM;
    if (q_start >= S) return;
    
    int num_q_rows = min(q_start + BM, S);
    int tid = threadIdx.x;
    
    // Initialize state
    for (int i = tid; i < num_q_rows; i += NUM_THREADS) {
        s_m[i - q_start] = -1e20f;
        s_s[i - q_start] = 0.0f;
    }
    for (int i = tid; i < num_q_rows * D; i += NUM_THREADS) {
        sO[i] = 0.0f;
    }
    __syncthreads();

    // Load Q tile once
    for (int i = tid; i < BM * BK; i += NUM_THREADS) {
        int r = i / BK;
        int c = i % BK;
        if (q_start + r < S) {
            sQ[i] = Q[((int64_t)b * H + h) * S * D + (q_start + r) * D + (c)]; // Simplified stride, assuming contiguous or handled by launcher
            // Actually Q shape is [B,H,S,D]. Global index: ((b*H+h)*S + (q_start+r))*D + col
            // We need to loop over all D cols for Q, not just BK. 
            // Q tile should be BM x D. But we process K in BK chunks.
            // Let's reload Q row pieces on the fly or load full Q tile (BM x D). 
            // BM*D = 64*128 = 8KB. Fits easily.
        }
    }
    // Relocate Q load to cover full width D
    for (int i = tid; i < BM * D; i += NUM_THREADS) {
        int r = i / D;
        int c = i % D;
        if (q_start + r < S) {
            sQ[i] = Q[((int64_t)b * H + h) * S * D + (q_start + r) * D + c];
        } else {
            sQ[i] = __float2bfloat16(0.0f);
        }
    }
    // Adjust pointers due to Q size change
    sK = reinterpret_cast<__nv_bfloat16*>(smem + BM * D * sizeof(__nv_bfloat16));
    sV = reinterpret_cast<__nv_bfloat16*>(smem + (BM * D + BN * BK) * sizeof(__nv_bfloat16));
    sO = reinterpret_cast<float*>(smem + (BM * D + BN * BK + BN * D) * sizeof(__nv_bfloat16));
    s_m = reinterpret_cast<float*>(smem + (BM * D + BN * BK + BN * D) * sizeof(__nv_bfloat16) + BM * D * sizeof(float));
    s_s = s_m + BM;

    // Re-init state with corrected pointers
    for (int i = tid; i < num_q_rows; i += NUM_THREADS) {
        s_m[i - q_start] = -1e20f;
        s_s[i - q_start] = 0.0f;
    }
    for (int i = tid; i < num_q_rows * D; i += NUM_THREADS) {
        sO[i] = 0.0f;
    }
    __syncthreads();

    int num_k_steps = (S + BN - 1) / BN;
    
    for (int ks = 0; ks < num_k_steps; ++ks) {
        int k_start = ks * BN;
        int eff_bn = min(k_start + BN, S) - k_start;
        
        // Load K tile: BN x BK
        for (int i = tid; i < BN * BK; i += NUM_THREADS) {
            int r = i / BK;
            int c = i % BK;
            if (k_start + r < S) {
                sK[i] = K[((int64_t)b * H + h) * S * D + (k_start + r) * D + (c + ks*BK)];
            } else {
                sK[i] = __float2bfloat16(0.0f);
            }
        }
        
        // Load V tile: BN x D
        for (int i = tid; i < BN * D; i += NUM_THREADS) {
            int r = i / D;
            int c = i % D;
            if (k_start + r < S) {
                sV[i] = V[((int64_t)b * H + h) * S * D + (k_start + r) * D + c];
            } else {
                sV[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // Process in BK strides
        for (int bk_step = 0; bk_step < BK; bk_step += 1) { // Unrolled implicitly by compiler or kept simple
            // Each thread processes a subset of query rows
            for (int r = tid; r < num_q_rows; r += NUM_THREADS) {
                float cur_max = -1e20f;
                float p_vals[BN]; // Temporary P row
                
                int qr = q_start + r;
                float* q_row = sQ + r * D;
                
                for (int c = 0; c < eff_bn; ++c) {
                    float dot = 0.0f;
                    int kc = k_start + c;
                    float* k_row = sK + c * BK + bk_step;
                    // Manual dot product for BK=16 (here just 1 iter for simplicity, but we loop bk_step)
                    // To optimize, we accumulate across bk_step. Let's restructure slightly for correctness:
                    // We'll compute P outside inner loop over bk_step, or accumulate dot here.
                    // Simpler: compute P[c] fully before max/sum.
                }
            }
        }
        
        // Restructure for clarity and correctness:
        // Compute P[eff_bn] for each assigned row, then update stats & O
        for (int r = tid; r < num_q_rows; r += NUM_THREADS) {
            float cur_max = -1e20f;
            float p[BN];
            
            int qr = q_start + r;
            for (int c = 0; c < eff_bn; ++c) {
                float dot = 0.0f;
                float* q_ptr = sQ + r * D;
                float* k_ptr = sK + c * BK;
                for (int k = 0; k < BK; ++k) {
                    dot += __bfloat162float(q_ptr[k]) * __bfloat162float(k_ptr[k]);
                }
                
                int kc = k_start + c;
                if (kc > qr) dot = -1e20f;
                dot *= inv_sqrt_d;
                
                p[c] = dot;
                if (dot > cur_max) cur_max = dot;
            }
            
            float old_max = s_m[r - q_start];
            float new_max = fmaxf(old_max, cur_max);
            s_m[r - q_start] = new_max;
            
            float alpha = expf(old_max - new_max);
            float beta = expf(cur_max - new_max);
            
            float sum_p = 0.0f;
            float* o_row = sO + r * D;
            for (int c = 0; c < eff_bn; ++c) {
                float pc = expf(p[c] - new_max);
                sum_p += pc;
                float v_scale = pc;
                float* v_row = sV + c * D;
                for (int d = 0; d < D; d += 4) {
                    o_row[d]   += v_scale * __bfloat162float(v_row[d]);
                    o_row[d+1] += v_scale * __bfloat162float(v_row[d+1]);
                    o_row[d+2] += v_scale * __bfloat162float(v_row[d+2]);
                    o_row[d+3] += v_scale * __bfloat162float(v_row[d+3]);
                }
            }
            
            // Update other rows' O acc with alpha
            float old_sum = s_s[r - q_start];
            s_s[r - q_start] = old_sum * alpha + sum_p * beta;
            
            // Scale accumulated O by alpha for this row
            // Since other threads don't touch this row, safe to loop
            for(int d=0; d<D; d+=4){
                o_row[d]   *= alpha;
                o_row[d+1] *= alpha;
                o_row[d+2] *= alpha;
                o_row[d+3] *= alpha;
            }
        }
        __syncthreads();
    }
    
    // Epilogue
    for (int r = tid; r < num_q_rows; ++r += NUM_THREADS) {
        float rs = s_s[r - q_start];
        float rm = s_m[r - q_start];
        float lse = rm + logf(rs);
        float inv_rs = rs > 1e-10f ? 1.0f / rs : 0.0f;
        
        int g_row = (int64_t)b * H + h;
        int out_idx_base = ((int64_t)g_row * S + (q_start + r));
        
        LSE[out_idx_base] = lse;
        
        float* o_row = sO + r * D;
        __nv_bfloat16* o_out = O + out_idx_base * D;
        for (int d = 0; d < D; d += 4) {
            o_out[d]   = __float2bfloat16(o_row[d] * inv_rs);
            o_out[d+1] = __float2bfloat16(o_row[d+1] * inv_rs);
            o_out[d+2] = __float2bfloat16(o_row[d+2] * inv_rs);
            o_out[d+3] = __float2bfloat16(o_row[d+3] * inv_rs);
        }
    }
}

void run(tvm::ffi::TensorView Q_in, tvm::ffi::TensorView K_in, tvm::ffi::TensorView V_in, 
         tvm::ffi::TensorView O_out, tvm::ffi::TensorView LSE_out) {
    CUDA_CHECK(cudaSetDevice(Q_in.device().device_id));
    
    int64_t B = Q_in.size(0), H = Q_in.size(1), S = Q_in.size(2), D = Q_in.size(3);
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q_in.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K_in.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V_in.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O_out.data_ptr());
    float* LSE_data = static_cast<float*>(LSE_out.data_ptr());
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q_in.device().device_type, Q_in.device().device_id));
    
    constexpr int BM = 64, BN = 64, BK = 16, NUM_THREADS = 256;
    dim3 grid(B * H, (S + BM - 1) / BM, 1);
    dim3 block(NUM_THREADS);
    
    // SMEM calculation: Q(BM*D) + K(BN*BK) + V(BN*D) + O(BM*D) + m(BM) + s(BM)
    size_t smem_size = (BM * D + BN * BK + BN * D + BM * D) * sizeof(__nv_bfloat16) 
                     + (BM * D) * sizeof(float) + BM * 2 * sizeof(float);
    
    float inv_sqrt_d = rsqrtf((float)D);
    
    CUDA_CHECK(cudaLaunchKernel(((void(*)(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, 
                                          __nv_bfloat16*, float*, int, int, int, int, float))&attn_fwd_kernel<BM, BN, BK, NUM_THREADS>)(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S, D, inv_sqrt_d)),
        grid, block, smem_size, stream));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attention_impl::run);

}  // namespace attention_impl