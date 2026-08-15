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

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int d)
{
    // Each block handles one (batch, head) pair
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;
    int nt = blockDim.x;

    // Attention scale: 1/sqrt(d), matching cuDNN's attn_scale
    float attn_scale = 1.0f / sqrtf((float)d);

    // Base pointer offsets for this (b, h) pair
    int64_t bh_off = (int64_t)(b * H + h) * (int64_t)S * d;
    const __nv_bfloat16* Qbh = Q + bh_off;
    const __nv_bfloat16* KBh = K + bh_off;
    const __nv_bfloat16* VBh = V + bh_off;
    const __nv_bfloat16* dOBh = dO + bh_off;
    __nv_bfloat16* dQBh = dQ + bh_off;
    __nv_bfloat16* dKBh = dK + bh_off;
    __nv_bfloat16* dVBh = dV + bh_off;
    const float* LBh = L + (b * H + h) * S;

    // Shared memory for softmax correction term: corr[sq] = sum_{sk<=sq} P[sq,sk]*dP[sq,sk]
    extern __shared__ float s_corr[];

    // ========================================================================
    // Phase 0: Compute corr[sq] for all sq
    // corr[sq] = sum_{sk=0}^{sq} P[sq,sk] * dP[sq,sk]
    // where P[sq,sk] = exp(Q[sq,:].K[sk,:]*attn_scale - L[sq])
    //       dP[sq,sk] = dO[sq,:].V[sk,:]
    // ========================================================================
    for (int sq = tid; sq < S; sq += nt) {
        float c = 0.0f;
        float l_val = LBh[sq];
        int64_t sq_off = (int64_t)sq * d;

        // Causal: sk <= sq
        for (int sk = 0; sk <= sq; sk++) {
            int64_t sk_off = (int64_t)sk * d;

            // Combined dot products: score and dP simultaneously
            float s_val = 0.0f, dp_val = 0.0f;
            for (int di = 0; di < d; di++) {
                s_val  += bf162f(Qbh[sq_off + di]) * bf162f(KBh[sk_off + di]);
                dp_val += bf162f(dOBh[sq_off + di]) * bf162f(VBh[sk_off + di]);
            }
            s_val *= attn_scale;
            float p_val = expf(s_val - l_val);
            c += p_val * dp_val;
        }
        s_corr[sq] = c;
    }
    __syncthreads();

    int64_t nelem = (int64_t)S * d;

    // ========================================================================
    // Phase 1: Compute dQ
    // dQ[sq, dq] = attn_scale * sum_{sk=0}^{sq} dS[sq,sk] * K[sk,dq]
    // dS[sq,sk] = P[sq,sk] * (dP[sq,sk] - corr[sq])
    // ========================================================================
    for (int64_t e = tid; e < nelem; e += nt) {
        int sq = (int)(e / d);
        int dq = (int)(e % d);

        float acc = 0.0f;
        float l_val = LBh[sq];
        int64_t sq_off = (int64_t)sq * d;

        for (int sk = 0; sk <= sq; sk++) {
            int64_t sk_off = (int64_t)sk * d;

            float s_val = 0.0f, dp_val = 0.0f;
            for (int di = 0; di < d; di++) {
                s_val  += bf162f(Qbh[sq_off + di]) * bf162f(KBh[sk_off + di]);
                dp_val += bf162f(dOBh[sq_off + di]) * bf162f(VBh[sk_off + di]);
            }
            s_val *= attn_scale;
            float p_val = expf(s_val - l_val);
            float ds = p_val * (dp_val - s_corr[sq]);
            acc += ds * bf162f(KBh[sk_off + dq]);
        }
        dQBh[e] = f2bf16(acc * attn_scale);
    }

    // ========================================================================
    // Phase 2: Compute dK
    // dK[sk, dk] = attn_scale * sum_{sq=sk}^{S-1} dS[sq,sk] * Q[sq,dk]
    // dS[sq,sk] = P[sq,sk] * (dP[sq,sk] - corr[sq])
    // ========================================================================
    for (int64_t e = tid; e < nelem; e += nt) {
        int sk = (int)(e / d);
        int dk = (int)(e % d);

        float acc = 0.0f;
        int64_t sk_off = (int64_t)sk * d;

        for (int sq = sk; sq < S; sq++) {
            int64_t sq_off = (int64_t)sq * d;

            float s_val = 0.0f, dp_val = 0.0f;
            for (int di = 0; di < d; di++) {
                s_val  += bf162f(Qbh[sq_off + di]) * bf162f(KBh[sk_off + di]);
                dp_val += bf162f(dOBh[sq_off + di]) * bf162f(VBh[sk_off + di]);
            }
            s_val *= attn_scale;
            float p_val = expf(s_val - LBh[sq]);
            float ds = p_val * (dp_val - s_corr[sq]);
            acc += ds * bf162f(Qbh[sq_off + dk]);
        }
        dKBh[e] = f2bf16(acc * attn_scale);
    }

    // ========================================================================
    // Phase 3: Compute dV
    // dV[sv, dv] = sum_{sq=sv}^{S-1} P[sq,sv] * dO[sq,dv]
    // No softmax correction needed for dV
    // ========================================================================
    for (int64_t e = tid; e < nelem; e += nt) {
        int sv = (int)(e / d);
        int dv = (int)(e % d);

        float acc = 0.0f;
        int64_t sv_off = (int64_t)sv * d;

        for (int sq = sv; sq < S; sq++) {
            int64_t sq_off = (int64_t)sq * d;

            float s_val = 0.0f;
            for (int di = 0; di < d; di++) {
                s_val += bf162f(Qbh[sq_off + di]) * bf162f(KBh[sv_off + di]);
            }
            s_val *= attn_scale;
            float p_val = expf(s_val - LBh[sq]);
            acc += p_val * bf162f(dOBh[sq_off + dv]);
        }
        dVBh[e] = f2bf16(acc);
    }
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

    dim3 grid(B * H);
    dim3 block(1024);
    // Dynamic shared memory: S floats for corr[] array
    size_t smem_bytes = (size_t)S * sizeof(float);

    cudaStream_t stream = nullptr; // default stream

    mha_bwd_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        (int)B, (int)H, (int)S, (int)d_dim);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attention_mha_bwd::run);

}  // namespace attention_mha_bwd