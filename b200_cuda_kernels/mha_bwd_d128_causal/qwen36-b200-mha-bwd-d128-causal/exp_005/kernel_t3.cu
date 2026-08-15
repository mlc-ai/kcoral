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
    // Parallelism: each block handles one (b, h, sq) triple
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
    __nv_bfloat16* dKBh = dK + bh_off;
    __nv_bfloat16* dVBh = dV + bh_off;
    const float* LBh = L + (b * H + h) * S;

    int tid = threadIdx.x;
    int nt = blockDim.x;
    int64_t sq_off = (int64_t)sq * d;
    float l_val = LBh[sq];

    // Shared memory: store intermediate results for k=0..sq
    // Using sq+1 entries (causal mask ensures we only need up to sq)
    extern __shared__ char smem_char[];
    float* smem_score = reinterpret_cast<float*>(smem_char);
    float* smem_dp = smem_score + S;
    float* smem_p = smem_dp + S;

    // Phase 1: Compute score[sk] and dp[sk] for all sk <= sq via reduction
    for (int k = tid; k <= sq; k += nt) {
        int64_t sk_off = (int64_t)k * d;
        
        float s_val = 0.0f, dp_val = 0.0f;
        for (int di = tid; di < d; di += nt) {
            s_val += bf162f(Qbh[sq_off + di]) * bf162f(KBh[sk_off + di]);
            dp_val += bf162f(dOBh[sq_off + di]) * bf162f(VBh[sk_off + di]);
        }
        
        // Warp reduce
        #pragma unroll
        for (int offset = nt >> 1; offset > 0; offset >>= 1) {
            s_val += __shfl_down_sync(0xFFFFFFFF, s_val, offset);
            dp_val += __shfl_down_sync(0xFFFFFFFF, dp_val, offset);
        }
        
        s_val *= attn_scale;
        smem_score[k] = s_val;
        smem_dp[k] = dp_val;
        smem_p[k] = expf(s_val - l_val);
    }
    __syncthreads();

    // Phase 2: Compute corr = sum_{k<=sq} P[k]*dp[k]
    float corr = 0.0f;
    for (int k = tid; k <= sq; k += nt) {
        corr += smem_p[k] * smem_dp[k];
    }
    #pragma unroll
    for (int offset = nt >> 1; offset > 0; offset >>= 1) {
        corr += __shfl_down_sync(0xFFFFFFFF, corr, offset);
    }

    // Phase 3: Write dQ[sq, :]
    // dQ[sq, dq] = attn_scale * sum_{k=0}^{sq} P[k]*(dp[k]-corr)*K[k,dq]
    for (int dq = tid; dq < d; dq += nt) {
        float acc = 0.0f;
        for (int k = 0; k <= sq; k++) {
            float ds = smem_p[k] * (smem_dp[k] - corr);
            acc += ds * bf162f(KBh[(int64_t)k * d + dq]);
        }
        dQBh[sq_off + dq] = f2bf16(acc * attn_scale);
    }

    // Phase 4: Accumulate dK contribution via atomics
    // dK[sk, dk] += attn_scale * P[sq,sk]*(dP[sq,sk]-corr)*Q[sq,dk]
    for (int di = tid; di < d; di += nt) {
        float qv = bf162f(Qbh[sq_off + di]);
        for (int k = 0; k <= sq; k++) {
            float ds = smem_p[k] * (smem_dp[k] - corr);
            float val = attn_scale * ds * qv;
            float* ptr = reinterpret_cast<float*>(dKBh + (int64_t)k * d + di);
            atomicAdd(ptr, val);
        }
    }

    // Phase 5: Accumulate dV contribution via atomics
    // dV[sv, dv] += P[sq,sv] * dO[sq,dv]
    for (int di = tid; di < d; di += nt) {
        float dov = bf162f(dOBh[sq_off + di]);
        for (int k = 0; k <= sq; k++) {
            float val = smem_p[k] * dov;
            float* ptr = reinterpret_cast<float*>(dVBh + (int64_t)k * d + di);
            atomicAdd(ptr, val);
        }
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

    dim3 grid(B * H * S);
    dim3 block(static_cast<int>(d_dim));
    
    // Shared memory: 3 arrays of S floats each
    size_t smem_bytes = 3 * (size_t)S * sizeof(float);

    // Initialize dK and dV to zero
    int64_t nelem = B * H * S * d_dim;
    CUDA_CHECK(cudaMemsetAsync(static_cast<void*>(dK.data_ptr()), 0, 
                               nelem * sizeof(__nv_bfloat16), nullptr));
    CUDA_CHECK(cudaMemsetAsync(static_cast<void*>(dV.data_ptr()), 0, 
                               nelem * sizeof(__nv_bfloat16), nullptr));

    cudaStream_t stream = nullptr;

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