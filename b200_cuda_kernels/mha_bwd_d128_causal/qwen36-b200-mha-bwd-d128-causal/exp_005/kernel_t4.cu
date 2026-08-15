#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__device__ __forceinline__ float bf162f(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 f2bf16(float x) {
    return __float2bfloat16(x);
}

// =============================================================================
// Kernel 1: Compute dQ[b,h,sq,dq] — one block per (b,h,sq), direct write
// =============================================================================
__global__ void mha_bwd_dQ_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    int B, int H, int S, int d)
{
    int bhsq = blockIdx.x;
    if (bhsq >= B * H * S) return;
    
    int sq = bhsq % S;
    bhsq /= S;
    int h = bhsq % H;
    int b = bhsq / H;

    float attn_scale = 1.0f / sqrtf((float)d);

    int64_t bh_off = (int64_t)(b * H + h) * (int64_t)S * d;
    const __nv_bfloat16* Qbh = Q + bh_off;
    const __nv_bfloat16* KBh = K + bh_off;
    const __nv_bfloat16* VBh = V + bh_off;
    const __nv_bfloat16* dOBh = dO + bh_off;
    __nv_bfloat16* dQBh = dQ + bh_off;
    const float* LBh = L + (b * H + h) * S;

    int tid = threadIdx.x;
    int nt = blockDim.x;
    int64_t sq_off = (int64_t)sq * d;
    float l_val = LBh[sq];

    // Shared memory for intermediate values
    extern __shared__ char smem_char[];
    float* smem_score = reinterpret_cast<float*>(smem_char);
    float* smem_dp = smem_score + S;
    float* smem_p = smem_dp + S;

    // Compute score[sk] and dp[sk] for all sk <= sq via per-thread dot product reduction
    for (int k = tid; k <= sq; k += nt) {
        int64_t sk_off = (int64_t)k * d;
        
        float s_val = 0.0f, dp_val = 0.0f;
        for (int di = tid; di < d; di += nt) {
            s_val  += bf162f(Qbh[sq_off + di]) * bf162f(KBh[sk_off + di]);
            dp_val += bf162f(dOBh[sq_off + di]) * bf162f(VBh[sk_off + di]);
        }
        
        #pragma unroll
        for (int offset = nt >> 1; offset > 0; offset >>= 1) {
            s_val  += __shfl_down_sync(0xFFFFFFFF, s_val, offset);
            dp_val += __shfl_down_sync(0xFFFFFFFF, dp_val, offset);
        }
        
        s_val *= attn_scale;
        smem_score[k] = s_val;
        smem_dp[k] = dp_val;
        smem_p[k] = expf(s_val - l_val);
    }
    __syncthreads();

    // Compute corr = sum_{k<=sq} P[k]*dp[k]
    float corr = 0.0f;
    for (int k = tid; k <= sq; k += nt) {
        corr += smem_p[k] * smem_dp[k];
    }
    #pragma unroll
    for (int offset = nt >> 1; offset > 0; offset >>= 1) {
        corr += __shfl_down_sync(0xFFFFFFFF, corr, offset);
    }

    // Write dQ[sq, dq] directly (no conflicts — each block owns one sq)
    for (int dq = tid; dq < d; dq += nt) {
        float acc = 0.0f;
        for (int k = 0; k <= sq; k++) {
            float ds = smem_p[k] * (smem_dp[k] - corr);
            acc += ds * bf162f(KBh[(int64_t)k * d + dq]);
        }
        dQBh[sq_off + dq] = f2bf16(acc * attn_scale);
    }
}

// =============================================================================
// Kernel 2: Compute dK[b,h,sk,dk] — each thread accumulates over valid sq range
// =============================================================================
__global__ void mha_bwd_dK_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dK,
    int B, int H, int S, int d)
{
    // Each thread handles one (b,h,sk,dk) element
    // Accumulate over sq from sk to S-1 (causal: sq >= sk)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_sk = B * H * S; // number of (b,h,sk) triples
    
    int bkhs = idx / d;
    int dk   = idx % d;
    if (bkhs >= total_sk) return;
    
    int sk = bkhs % S;
    bkhs /= S;
    int h = bkhs % H;
    int b = bkhs / H;

    float attn_scale = 1.0f / sqrtf((float)d);

    int64_t bh_off = (int64_t)(b * H + h) * (int64_t)S * d;
    const __nv_bfloat16* Qbh = Q + bh_off;
    const __nv_bfloat16* KBh = K + bh_off;
    const __nv_bfloat16* VBh = V + bh_off;
    const __nv_bfloat16* dOBh = dO + bh_off;
    __nv_bfloat16* dKBh = dK + bh_off;
    const float* LBh = L + (b * H + h) * S;

    int64_t sk_off = (int64_t)sk * d;
    float acc = 0.0f;

    for (int sq = sk; sq < S; sq++) {
        int64_t sq_off = (int64_t)sq * d;
        float l_val = LBh[sq];

        // Compute score and dP dot products
        float s_val = 0.0f, dp_val = 0.0f;
        for (int di = 0; di < d; di++) {
            s_val  += bf162f(Qbh[sq_off + di]) * bf162f(KBh[sk_off + di]);
            dp_val += bf162f(dOBh[sq_off + di]) * bf162f(VBh[sk_off + di]);
        }
        s_val *= attn_scale;
        float p_val = expf(s_val - l_val);
        
        // Need corr[sq] for softmax gradient
        float corr = 0.0f;
        for (int kk = 0; kk <= sq; kk++) {
            int64_t kk_off = (int64_t)kk * d;
            float s_k = 0.0f, dp_k = 0.0f;
            for (int di = 0; di < d; di++) {
                s_k  += bf162f(Qbh[sq_off + di]) * bf162f(KBh[kk_off + di]);
                dp_k += bf162f(dOBh[sq_off + di]) * bf162f(VBh[kk_off + di]);
            }
            s_k *= attn_scale;
            float p_k = expf(s_k - l_val);
            corr += p_k * dp_k;
        }
        
        float ds = p_val * (dp_val - corr);
        acc += ds * bf162f(Qbh[sq_off + dk]);
    }
    dKBh[sk_off + dk] = f2bf16(acc * attn_scale);
}

// =============================================================================
// Kernel 3: Compute dV[b,h,sv,dv] — each thread accumulates over valid sq range
// =============================================================================
__global__ void mha_bwd_dV_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int d)
{
    // Each thread handles one (b,h,sv,dv) element
    // Accumulate over sq from sv to S-1 (causal: sq >= sv)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_sv = B * H * S;
    
    int bkv = idx / d;
    int dv  = idx % d;
    if (bkv >= total_sv) return;
    
    int sv = bkv % S;
    bkv /= S;
    int h = bkv % H;
    int b = bkv / H;

    float attn_scale = 1.0f / sqrtf((float)d);

    int64_t bh_off = (int64_t)(b * H + h) * (int64_t)S * d;
    const __nv_bfloat16* Qbh = Q + bh_off;
    const __nv_bfloat16* KBh = K + bh_off;
    const __nv_bfloat16* dOBh = dO + bh_off;
    __nv_bfloat16* dVBh = dV + bh_off;
    const float* LBh = L + (b * H + h) * S;

    int64_t sv_off = (int64_t)sv * d;
    float acc = 0.0f;

    for (int sq = sv; sq < S; sq++) {
        int64_t sq_off = (int64_t)sq * d;
        float l_val = LBh[sq];

        float s_val = 0.0f;
        for (int di = 0; di < d; di++) {
            s_val += bf162f(Qbh[sq_off + di]) * bf162f(KBh[sv_off + di]);
        }
        s_val *= attn_scale;
        float p_val = expf(s_val - l_val);
        acc += p_val * bf162f(dOBh[sq_off + dv]);
    }
    dVBh[sv_off + dv] = f2bf16(acc);
}

namespace attention_mha_bwd {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d_dim = Q.size(3);

    if (B * H == 0 || S == 0) return;

    cudaStream_t stream = nullptr;

    // Launch dQ kernel: one block per (b,h,sq), d threads per block
    int64_t n_bh_sq = B * H * S;
    dim3 grid_dQ(n_bh_sq);
    dim3 block_dQ(static_cast<int>(d_dim)); // 128 threads
    size_t smem_bytes = 2 * (size_t)S * sizeof(float); // smem_score + smem_dp (+smem_p inline reuse)
    // Actually we need 3 arrays: score, dp, p each of size S
    smem_bytes = 3 * (size_t)S * sizeof(float);
    
    mha_bwd_dQ_kernel<<<grid_dQ, block_dQ, smem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        (int)B, (int)H, (int)S, (int)d_dim);
    CUDA_CHECK(cudaGetLastError());

    // Launch dK kernel: grid-stride over (b,h,sk)*d elements
    int64_t n_elem_k = B * H * S * d_dim;
    int block_sz = 256;
    int64_t n_blocks_k = (n_elem_k + block_sz - 1) / block_sz;
    dim3 grid_dK(n_blocks_k);
    dim3 block_dK(block_sz);

    mha_bwd_dK_kernel<<<grid_dK, block_dK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        (int)B, (int)H, (int)S, (int)d_dim);
    CUDA_CHECK(cudaGetLastError());

    // Launch dV kernel: grid-stride over (b,h,sv)*d elements
    dim3 grid_dV(n_blocks_k);
    dim3 block_dV(block_sz);

    mha_bwd_dV_kernel<<<grid_dV, block_dV, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        (int)B, (int)H, (int)S, (int)d_dim);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attention_mha_bwd::run);

}  // namespace attention_mha_bwd