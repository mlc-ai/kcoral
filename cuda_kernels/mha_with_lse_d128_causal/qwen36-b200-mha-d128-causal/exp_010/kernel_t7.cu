#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

namespace tvm_ffi_mha_causal_d128 {

// Causal FlashAttention kernel for BF16 inputs with fixed D=128
template <int BM, int BN, int D, int NUM_THREADS>
__global__ void causal_attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S,
    float inv_sqrt_d)
{
    // Shared memory layout
    extern __shared__ char shared_mem[];
    
    __nv_bfloat16* s_Q   = reinterpret_cast<__nv_bfloat16*>(shared_mem);
    __nv_bfloat16* s_K   = s_Q + BM * D;
    __nv_bfloat16* s_V   = s_K + BN * D;
    float*         s_O   = reinterpret_cast<float*>(s_V + BN * D);
    
    int tid = threadIdx.x;
    int bid_bh = blockIdx.x;
    int batch = bid_bh / H;
    int head  = bid_bh % H;
    if (batch >= B || head >= H) return;
    
    int64_t bh_off = (static_cast<int64_t>(batch) * H + head) * static_cast<int64_t>(S) * D;
    const __nv_bfloat16* Q_base = Q + bh_off;
    const __nv_bfloat16* K_base = K + bh_off;
    const __nv_bfloat16* V_base = V + bh_off;
    __nv_bfloat16* O_base      = O + bh_off;
    float* LSE_base             = LSE + (batch * H + head) * S;
    
    int num_kv_blocks = (S + BN - 1) / BN;
    
    // Process one query block assigned to this block.y
    int b_m = blockIdx.y;
    int q_start = b_m * BM;
    if (q_start >= S) return;
    int bm_actual = min(q_start + BM, S) - q_start;
    
    // Load Q tile cooperatively
    for (int idx = tid; idx < bm_actual * D; idx += NUM_THREADS) {
        int r = idx / D;
        int c = idx % D;
        s_Q[r * D + c] = Q_base[(q_start + r) * D + c];
    }
    __syncthreads();
    
    // Initialize O accumulator to zero
    for (int idx = tid; idx < bm_actual * D; idx += NUM_THREADS) {
        s_O[idx] = 0.0f;
    }
    __syncthreads();
    
    // Per-thread softmax state
    bool active = tid < bm_actual;
    float m_val = active ? -INFINITY : 0.0f;
    float l_val = active ? 0.0f       : 0.0f;
    
    // Iterate over all KV blocks
    for (int b_n = 0; b_n < num_kv_blocks; b_n++) {
        int kv_start = b_n * BN;
        int bn_actual = min(kv_start + BN, S) - kv_start;
        
        // Load K and V tiles cooperatively
        for (int idx = tid; idx < bn_actual * D; idx += NUM_THREADS) {
            int r = idx / D;
            int c = idx % D;
            s_K[r * D + c] = K_base[(kv_start + r) * D + c];
            s_V[r * D + c] = V_base[(kv_start + r) * D + c];
        }
        __syncthreads();
        
        // Step 1: compute local max over keys in this block for this query row
        float new_m = -INFINITY;
        if (active) {
            int qr = tid;
            int qp = q_start + qr;
            
            for (int kp = 0; kp < bn_actual; kp++) {
                float dot = 0.0f;
                
                #pragma unroll
                for (int dd = 0; dd < D; dd += 4) {
                    float q0 = __bfloat162float(s_Q[qr * D + dd]);
                    float q1 = __bfloat162float(s_Q[qr * D + dd + 1]);
                    float q2 = __bfloat162float(s_Q[qr * D + dd + 2]);
                    float q3 = __bfloat162float(s_Q[qr * D + dd + 3]);
                    float k0 = __bfloat162float(s_K[kp * D + dd]);
                    float k1 = __bfloat162float(s_K[kp * D + dd + 1]);
                    float k2 = __bfloat162float(s_K[kp * D + dd + 2]);
                    float k3 = __bfloat162float(s_K[kp * D + dd + 3]);
                    dot += q0*k0 + q1*k1 + q2*k2 + q3*k3;
                }
                float scaled = dot * inv_sqrt_d;
                
                if ((kv_start + kp) <= qp && scaled > new_m) {
                    new_m = scaled;
                }
            }
        }
        
        // Update running max - compare with global new_m across the warp/block
        float old_m = m_val;
        if (active && new_m > m_val) {
            m_val = new_m;
        }
        
        // Rescale previous accumulations if max changed
        if (active) {
            float alpha = expf(old_m - m_val);
            
            int o_row = tid * D;
            #pragma unroll
            for (int dd = 0; dd < D; dd++) {
                s_O[o_row + dd] *= alpha;
            }
            l_val *= alpha;
        }
        __syncthreads();
        
        // Step 2: compute attn scores and accumulate
        if (active) {
            int qr = tid;
            int qp = q_start + qr;
            int o_row = qr * D;
            float local_sum = 0.0f;
            
            for (int kp = 0; kp < bn_actual; kp++) {
                int kp_abs = kv_start + kp;
                float dot = 0.0f;
                
                #pragma unroll
                for (int dd = 0; dd < D; dd += 4) {
                    float q0 = __bfloat162float(s_Q[qr * D + dd]);
                    float q1 = __bfloat162float(s_Q[qr * D + dd + 1]);
                    float q2 = __bfloat162float(s_Q[qr * D + dd + 2]);
                    float q3 = __bfloat162float(s_Q[qr * D + dd + 3]);
                    float k0 = __bfloat162float(s_K[kp * D + dd]);
                    float k1 = __bfloat162float(s_K[kp * D + dd + 1]);
                    float k2 = __bfloat162float(s_K[kp * D + dd + 2]);
                    float k3 = __bfloat162float(s_K[kp * D + dd + 3]);
                    dot += q0*k0 + q1*k1 + q2*k2 + q3*k3;
                }
                
                if (kp_abs <= qp) {
                    float scaled = dot * inv_sqrt_d;
                    float attn = expf(scaled - m_val);
                    local_sum += attn;
                    
                    #pragma unroll
                    for (int dd = 0; dd < D; dd++) {
                        s_O[o_row + dd] += attn * __bfloat162float(s_V[kp * D + dd]);
                    }
                }
            }
            l_val += local_sum;
        }
        __syncthreads();
    }
    
    // Write final output and LSE
    if (active) {
        int out_row = q_start + tid;
        int o_row = tid * D;
        float safe_l = fmaxf(l_val, 1e-12f);
        float inv_l = 1.0f / safe_l;
        
        #pragma unroll
        for (int dd = 0; dd < D; dd++) {
            O_base[static_cast<int64_t>(out_row) * D + dd] = 
                __float2bfloat16(s_O[o_row + dd] * inv_l);
        }
        LSE_base[out_row] = m_val + logf(safe_l);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    constexpr int D = 128;
    
    const __nv_bfloat16* q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr       = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr             = static_cast<float*>(LSE.data_ptr());
    
    // Smaller tiles to fit shared memory comfortably
    constexpr int BM  = 32;
    constexpr int BN  = 64;
    constexpr int NUM_THREADS = 64;
    
    int num_q_blocks = (S + BM - 1) / BM;
    dim3 grid(static_cast<int>(B * H), num_q_blocks);
    dim3 block(NUM_THREADS);
    
    // Shared memory: Q(BM*D*2) + K(BN*D*2) + V(BN*D*2) + O(BM*D*4)
    // = 32*128*2 + 64*128*2 + 64*128*2 + 32*128*4 = 8192 + 16384 + 16384 + 16384 = 57344 bytes = 56 KB
    size_t smem_size = (BM * D + BN * D + BN * D) * sizeof(__nv_bfloat16) 
                     + BM * D * sizeof(float);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    float inv_sqrt_d = rsqrtf(static_cast<float>(D));
    
    causal_attn_kernel<BM, BN, D, NUM_THREADS><<<grid, block, smem_size, stream>>>(
        q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
        static_cast<int>(B), static_cast<int>(H),
        static_cast<int>(S),
        inv_sqrt_d
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_causal_d128::run);

}  // namespace tvm_ffi_mha_causal_d128