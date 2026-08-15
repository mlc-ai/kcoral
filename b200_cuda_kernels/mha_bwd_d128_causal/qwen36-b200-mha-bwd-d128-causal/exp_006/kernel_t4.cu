#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

namespace mha_back_clean {

__global__ void memset_bf16_kernel(__nv_bfloat16* ptr, size_t n, __nv_bfloat16 val) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = val;
}

// Kernel for dV: dV[b,h,sk,dim] = sum_{sq>=sk} P[sq,sk] * dO[b,h,sq,dim]
// Each thread computes one (b,h,sk,dim) element entirely
__global__ void mha_bwd_dV_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int d,
    size_t stride_bh_dv, size_t stride_s_dv,
    size_t stride_q_kv, size_t stride_lse
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S * d;
    if (idx >= total) return;
    
    int dim = idx % d;
    int tmp = idx / d;
    int sk = tmp % S;
    int tmp2 = tmp / S;
    int h = tmp2 % H;
    int b = tmp2 / H;
    
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    
    // Base pointers for this (b,h)
    const __nv_bfloat16* q_base = Q + (size_t)b * stride_q_kv + (size_t)h * (stride_q_kv / H);
    const __nv_bfloat16* k_base = K + (size_t)b * stride_q_kv + (size_t)h * (stride_q_kv / H);
    const __nv_bfloat16* do_base = dO + (size_t)b * stride_q_kv + (size_t)h * (stride_q_kv / H);
    const float* lse_base = L + (size_t)b * (stride_lse) + (size_t)h * (stride_lse / H);
    
    __nv_bfloat16* dv_ptr = dV + (size_t)b * stride_bh_dv + (size_t)h * (stride_bh_dv / H) + (size_t)sk * stride_s_dv;
    
    float acc = 0.0f;
    
    const __nv_bfloat16* k_row = k_base + (size_t)sk * d;
    float lse_sk = lse_base[sk]; // LSE is per-query-position
    
    // Sum over sq >= sk (causal mask)
    for (int sq = sk; sq < S; sq++) {
        const __nv_bfloat16* q_row = q_base + (size_t)sq * d;
        
        // Compute score = Q[sq].K[sk] / sqrt(d)
        float score = 0.0f;
        for (int i = 0; i < d; i += 4) {
            score += __bfloat162float(q_row[i])   * __bfloat162float(k_row[i]);
            score += __bfloat162float(q_row[i+1]) * __bfloat162float(k_row[i+1]);
            score += __bfloat162float(q_row[i+2]) * __bfloat162float(k_row[i+2]);
            score += __bfloat162float(q_row[i+3]) * __bfloat162float(k_row[i+3]);
        }
        score *= inv_sqrt_d;
        
        float p = expf(score - lse_base[sq]);
        
        acc += p * __bfloat162float(do_base[(size_t)sq * d + dim]);
    }
    
    dv_ptr[dim] = __float2bfloat16(acc);
}

// Kernel for dQ: dQ[b,h,sq,dim] = sum_{sk<=sq} dS[sq,sk] * K[b,h,sk,dim]
// Each thread computes one (b,h,sq,dim) element entirely
__global__ void mha_bwd_dQ_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    int B, int H, int S, int d,
    size_t stride_bh, size_t stride_s
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S * d;
    if (idx >= total) return;
    
    int dim = idx % d;
    int tmp = idx / d;
    int sq = tmp % S;
    int tmp2 = tmp / S;
    int h = tmp2 % H;
    int b = tmp2 / H;
    
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    
    size_t base = (size_t)b * stride_bh + (size_t)h * (stride_bh / H);
    size_t base_lse = (size_t)b * S * H + (size_t)h * S;
    
    const __nv_bfloat16* q_vec = Q + base + (size_t)sq * stride_s;
    const __nv_bfloat16* do_vec = dO + base + (size_t)sq * stride_s;
    const __nv_bfloat16* o_vec  = O  + base + (size_t)sq * stride_s;
    float lse = L[base_lse + sq];
    
    // Compute D = dot(dO[sq], O[sq])
    float D = 0.0f;
    for (int i = 0; i < d; i += 4) {
        D += __bfloat162float(do_vec[i])   * __bfloat162float(o_vec[i]);
        D += __bfloat162float(do_vec[i+1]) * __bfloat162float(o_vec[i+1]);
        D += __bfloat162float(do_vec[i+2]) * __bfloat162float(o_vec[i+2]);
        D += __bfloat162float(do_vec[i+3]) * __bfloat162float(o_vec[i+3]);
    }
    
    float dq_acc = 0.0f;
    
    // Sum over sk <= sq (causal)
    for (int sk = 0; sk <= sq; sk++) {
        const __nv_bfloat16* k_vec = K + base + (size_t)sk * stride_s;
        const __nv_bfloat16* v_vec = V + base + (size_t)sk * stride_s;
        
        // Score = Q[sq].K[sk]/sqrt(d)
        float score = 0.0f;
        for (int i = 0; i < d; i += 4) {
            score += __bfloat162float(q_vec[i])   * __bfloat162float(k_vec[i]);
            score += __bfloat162float(q_vec[i+1]) * __bfloat162float(k_vec[i+1]);
            score += __bfloat162float(q_vec[i+2]) * __bfloat162float(k_vec[i+2]);
            score += __bfloat162float(q_vec[i+3]) * __bfloat162float(k_vec[i+3]);
        }
        score *= inv_sqrt_d;
        
        float p = expf(score - lse);
        
        // dP_partial = V[sk].dO[sq]
        float dp_partial = 0.0f;
        for (int i = 0; i < d; i += 4) {
            dp_partial += __bfloat162float(v_vec[i])   * __bfloat162float(do_vec[i]);
            dp_partial += __bfloat162float(v_vec[i+1]) * __bfloat162float(do_vec[i+1]);
            dp_partial += __bfloat162float(v_vec[i+2]) * __bfloat162float(do_vec[i+2]);
            dp_partial += __bfloat162float(v_vec[i+3]) * __bfloat162float(do_vec[i+3]);
        }
        
        float ds = p * (dp_partial - D);
        
        // dQ[sq,dim] += ds * K[sk,dim]
        dq_acc += ds * __bfloat162float(k_vec[dim]);
    }
    
    dQ[base + (size_t)sq * stride_s + dim] = __float2bfloat16(dq_acc);
}

// Kernel for dK: dK[b,h,sk,dim] = sum_{sq>=sk} dS[sq,sk] * Q[b,h,sq,dim]
// Each thread computes one (b,h,sk,dim) element entirely
__global__ void mha_bwd_dK_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dK,
    int B, int H, int S, int d,
    size_t stride_bh, size_t stride_s
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S * d;
    if (idx >= total) return;
    
    int dim = idx % d;
    int tmp = idx / d;
    int sk = tmp % S;
    int tmp2 = tmp / S;
    int h = tmp2 % H;
    int b = tmp2 / H;
    
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    
    size_t base = (size_t)b * stride_bh + (size_t)h * (stride_bh / H);
    size_t base_lse = (size_t)b * S * H + (size_t)h * S;
    
    const __nv_bfloat16* k_vec = K + base + (size_t)sk * stride_s;
    
    float dk_acc = 0.0f;
    
    // Sum over sq >= sk (causal)
    for (int sq = sk; sq < S; sq++) {
        const __nv_bfloat16* q_vec = Q + base + (size_t)sq * stride_s;
        const __nv_bfloat16* do_vec = dO + base + (size_t)sq * stride_s;
        const __nv_bfloat16* o_vec  = O  + base + (size_t)sq * stride_s;
        const __nv_bfloat16* v_vec  = V  + base + (size_t)sk * stride_s;
        float lse = L[base_lse + sq];
        
        // Recompute D
        float D = 0.0f;
        for (int i = 0; i < d; i += 4) {
            D += __bfloat162float(do_vec[i])   * __bfloat162float(o_vec[i]);
            D += __bfloat162float(do_vec[i+1]) * __bfloat162float(o_vec[i+1]);
            D += __bfloat162float(do_vec[i+2]) * __bfloat162float(o_vec[i+2]);
            D += __bfloat162float(do_vec[i+3]) * __bfloat162float(o_vec[i+3]);
        }
        
        // Score
        float score = 0.0f;
        for (int i = 0; i < d; i += 4) {
            score += __bfloat162float(q_vec[i])   * __bfloat162float(k_vec[i]);
            score += __bfloat162float(q_vec[i+1]) * __bfloat162float(k_vec[i+1]);
            score += __bfloat162float(q_vec[i+2]) * __bfloat162float(k_vec[i+2]);
            score += __bfloat162float(q_vec[i+3]) * __bfloat162float(k_vec[i+3]);
        }
        score *= inv_sqrt_d;
        
        float p = expf(score - lse);
        
        float dp_partial = 0.0f;
        for (int i = 0; i < d; i += 4) {
            dp_partial += __bfloat162float(v_vec[i])   * __bfloat162float(do_vec[i]);
            dp_partial += __bfloat162float(v_vec[i+1]) * __bfloat162float(do_vec[i+1]);
            dp_partial += __bfloat162float(v_vec[i+2]) * __bfloat162float(do_vec[i+2]);
            dp_partial += __bfloat162float(v_vec[i+3]) * __bfloat162float(do_vec[i+3]);
        }
        
        float ds = p * (dp_partial - D);
        
        // dK[sk,dim] += ds * Q[sq,dim]
        dk_acc += ds * __bfloat162float(q_vec[dim]);
    }
    
    dK[base + (size_t)sk * stride_s + dim] = __float2bfloat16(dk_acc);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int d = (int)Q.size(3);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    // Contiguous layout strides: [B, H, S, d]
    size_t stride_d = 1;
    size_t stride_s = d;
    size_t stride_h = S * d;
    size_t stride_b = H * S * d;
    size_t stride_bh = stride_h;  // stride from H dimension
    size_t stride_lse = H * S;    // L is [B, H, S] contiguous
    
    size_t total_elems = (size_t)B * H * S * d;
    int threads = 512;
    int blocks = (int)((total_elems + threads - 1) / threads);
    
    // Zero outputs
    memset_bf16_kernel<<<(int)((total_elems + threads - 1) / threads), threads, 0, stream>>>(
        dQ_ptr, total_elems, __float2bfloat16(0.0f));
    CUDA_CHECK(cudaGetLastError());
    memset_bf16_kernel<<<(int)((total_elems + threads - 1) / threads), threads, 0, stream>>>(
        dK_ptr, total_elems, __float2bfloat16(0.0f));
    CUDA_CHECK(cudaGetLastError());
    memset_bf16_kernel<<<(int)((total_elems + threads - 1) / threads), threads, 0, stream>>>(
        dV_ptr, total_elems, __float2bfloat16(0.0f));
    CUDA_CHECK(cudaGetLastError());
    
    // Launch three kernels in parallel on same stream
    mha_bwd_dV_kernel<<<blocks, threads, 0, stream>>>(
        Q_ptr, K_ptr, dO_ptr, L_ptr, dV_ptr, B, H, S, d,
        stride_h, stride_d, stride_h, stride_lse);
    CUDA_CHECK(cudaGetLastError());
    
    mha_bwd_dQ_kernel<<<blocks, threads, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
        B, H, S, d, stride_h, stride_d);
    CUDA_CHECK(cudaGetLastError());
    
    mha_bwd_dK_kernel<<<blocks, threads, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr,
        B, H, S, d, stride_h, stride_d);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_back_clean::run);

}  // namespace mha_back_clean