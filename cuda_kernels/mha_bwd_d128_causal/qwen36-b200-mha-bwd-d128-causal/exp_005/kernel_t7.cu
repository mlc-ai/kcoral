#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

__device__ __forceinline__ float bf162f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 f2bf16(float x) { return __float2bfloat16(x); }

// =============================================================================
// Kernel 1: Compute dQ and store corr for later dK computation
// 1 block per (b,h,sq), 128 threads (1 per d-dimension)
// =============================================================================
__global__ void mha_bwd_dQ_corr_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    float* __restrict__ corr_out,
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
    float* cors = corr_out + (b * H + h) * S;

    int tid = threadIdx.x;
    int nt = blockDim.x; // 128
    int nwarp = nt / 32; // 4
    int64_t sq_base = (int64_t)sq * d;
    float lse_val = LBh[sq];

    // Shared memory layout:
    // s_mem[S] + dp_mem[S] + warp_corr[4] = 2*S+4 floats
    extern __shared__ char smem_char[];
    float* s_mem          = reinterpret_cast<float*>(smem_char);
    float* dp_mem         = s_mem + S;
    float* warp_corr      = dp_mem + S;

    // Phase 1: Load dot products into SMEM for each sk <= sq
    for (int bs = 0; bs <= sq; bs += nt) {
        int ce = min(bs + nt, sq + 1);
        int sk = bs + tid;
        if (sk < ce && sk <= sq) {
            int64_t sk_base = (int64_t)sk * d;
            float s = 0.0f, dp = 0.0f;
            for (int di = 0; di < d; di++) {
                s  += bf162f(Qbh[sq_base + di]) * bf162f(KBh[sk_base + di]);
                dp += bf162f(dOBh[sq_base + di]) * bf162f(VBh[sk_base + di]);
            }
            s_mem[sk] = s * attn_scale;
            dp_mem[sk] = dp;
        }
    }
    __syncthreads();

    // Phase 2: Compute corr = sum_{sk<=sq} P[sk] * dp[sk]
    float corr_local = 0.0f;
    for (int sk = tid; sk <= sq; sk += nt) {
        float p = expf(s_mem[sk] - lse_val);
        corr_local += p * dp_mem[sk];
    }
    // Warp-level reduce
    #pragma unroll
    for (int offset = 64; offset > 0; offset >>= 1)
        corr_local += __shfl_down_sync(0xFFFFFFFF, corr_local, offset);
    // Store warp results
    if ((tid % 32) == 0) warp_corr[tid / 32] = corr_local;
    __syncthreads();
    // Block-level reduce (thread 0)
    float corr_final = 0.0f;
    if (tid == 0) {
        for (int w = 0; w < nwarp; w++) corr_final += warp_corr[w];
        cors[sq] = corr_final; // write to global
    }
    __syncthreads();
    // Broadcast corr to all threads
    corr_final = warp_corr[0];
    for (int w = 1; w < nwarp; w++) corr_final += warp_corr[w];

    // Phase 3: Compute dQ[sq, dq] = attn_scale * sum_{sk<=sq} P[sk]*(dp[sk]-corr)*K[sk,dq]
    float dQ_acc = 0.0f;
    for (int sk = 0; sk <= sq; sk++) {
        float p = expf(s_mem[sk] - lse_val);
        float ds = p * (dp_mem[sk] - corr_final);
        dQ_acc += ds * bf162f(KBh[(int64_t)sk * d + tid]);
    }
    dQBh[sq_base + tid] = f2bf16(dQ_acc * attn_scale);
}

// =============================================================================
// Kernel 2: Compute dK[b,h,sk,:] 
// 1 block per (b,h,sk), 128 threads
// dK[sk,dk] = attn_scale * sum_{sq=sk}^{S-1} P[sq,sk]*(dP[sq,sk]-corr[sq])*Q[sq,dk]
// =============================================================================
__global__ void mha_bwd_dK_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ corr_global,
    __nv_bfloat16* __restrict__ dK,
    int B, int H, int S, int d)
{
    int bhsk = blockIdx.x;
    if (bhsk >= B * H * S) return;
    int sk = bhsk % S;
    bhsk /= S;
    int h = bhsk % H;
    int b = bhsk / H;

    float attn_scale = 1.0f / sqrtf((float)d);
    int64_t bh_off = (int64_t)(b * H + h) * (int64_t)S * d;
    const __nv_bfloat16* Qbh = Q + bh_off;
    const __nv_bfloat16* KBh = K + bh_off;
    const __nv_bfloat16* VBh = V + bh_off;
    const __nv_bfloat16* dOBh = dO + bh_off;
    __nv_bfloat16* dKBh = dK + bh_off;
    const float* LBh = L + (b * H + h) * S;
    const float* cors = corr_global + (b * H + h) * S;

    int tid = threadIdx.x;
    int64_t sk_base = (int64_t)sk * d;

    // Shared memory for broadcasting dot product results
    extern __shared__ char smem_char[];
    float* s_broadcast = reinterpret_cast<float*>(smem_char);
    float* dp_broadcast = s_broadcast + 1;

    // Register accumulators for dK
    float dK_acc = 0.0f;

    for (int sq = sk; sq < S; sq++) {
        int64_t sq_base = (int64_t)sq * d;

        // Thread 0 computes both dot products, others wait
        float s = 0.0f, dp = 0.0f;
        if (tid == 0) {
            for (int di = 0; di < d; di++) {
                s  += bf162f(Qbh[sq_base + di]) * bf162f(KBh[sk_base + di]);
                dp += bf162f(dOBh[sq_base + di]) * bf162f(VBh[sk_base + di]);
            }
            s_broadcast[0] = s * attn_scale;
            dp_broadcast[0] = dp;
        }
        // Memory fence to ensure visibility
        asm volatile("fence.proxy.async;\n");

        // All threads read the results
        s = s_broadcast[0];
        dp = dp_broadcast[0];

        float p = expf(s - LBh[sq]);
        float ds = p * (dp - cors[sq]);
        dK_acc += ds * bf162f(Qbh[sq_base + tid]);
    }
    dKBh[sk_base + tid] = f2bf16(dK_acc * attn_scale);
}

// =============================================================================
// Kernel 3: Compute dV[b,h,sv,:]
// 1 block per (b,h,sv), 128 threads
// dV[sv,dv] = sum_{sq=sv}^{S-1} P[sq,sv] * dO[sq,dv]
// =============================================================================
__global__ void mha_bwd_dV_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int d)
{
    int bhsv = blockIdx.x;
    if (bhsv >= B * H * S) return;
    int sv = bhsv % S;
    bhsv /= S;
    int h = bhsv % H;
    int b = bhsv / H;

    float attn_scale = 1.0f / sqrtf((float)d);
    int64_t bh_off = (int64_t)(b * H + h) * (int64_t)S * d;
    const __nv_bfloat16* Qbh = Q + bh_off;
    const __nv_bfloat16* KBh = K + bh_off;
    const __nv_bfloat16* dOBh = dO + bh_off;
    __nv_bfloat16* dVBh = dV + bh_off;
    const float* LBh = L + (b * H + h) * S;

    int tid = threadIdx.x;
    int64_t sv_base = (int64_t)sv * d;

    extern __shared__ char smem_char[];
    float* s_broadcast = reinterpret_cast<float*>(smem_char);

    float dV_acc = 0.0f;

    for (int sq = sv; sq < S; sq++) {
        int64_t sq_base = (int64_t)sq * d;

        // Thread 0 computes dot product, others wait
        float s = 0.0f;
        if (tid == 0) {
            for (int di = 0; di < d; di++) {
                s += bf162f(Qbh[sq_base + di]) * bf162f(KBh[sv_base + di]);
            }
            s_broadcast[0] = s * attn_scale;
        }
        asm volatile("fence.proxy.async;\n");

        s = s_broadcast[0];
        float p = expf(s - LBh[sq]);
        dV_acc += p * bf162f(dOBh[sq_base + tid]);
    }
    dVBh[sv_base + tid] = f2bf16(dV_acc);
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

    int64_t n_bh_sq = B * H * S;
    dim3 grid(n_bh_sq);
    dim3 block(static_cast<int>(d_dim)); // 128 threads

    cudaStream_t stream = nullptr;

    // Allocate temporary buffer for corr: B*H*S floats
    size_t corr_bytes = (size_t)B * H * S * sizeof(float);
    float* d_corr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_corr, corr_bytes));
    CUDA_CHECK(cudaMemsetAsync(d_corr, 0, corr_bytes, stream));

    // Shared memory for kernel 1: 2*S + 4 floats
    size_t smem1 = (2 * S + 4) * sizeof(float);
    // Shared memory for kernel 2&3: 2 floats (broadcast)
    size_t smem2 = 2 * sizeof(float);

    // Kernel 1: dQ + corr
    mha_bwd_dQ_corr_kernel<<<grid, block, smem1, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        d_corr,
        (int)B, (int)H, (int)S, (int)d_dim);
    CUDA_CHECK(cudaGetLastError());

    // Kernel 2: dK
    mha_bwd_dK_kernel<<<grid, block, smem2, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        d_corr,
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        (int)B, (int)H, (int)S, (int)d_dim);
    CUDA_CHECK(cudaGetLastError());

    // Kernel 3: dV
    mha_bwd_dV_kernel<<<grid, block, smem2, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        (int)B, (int)H, (int)S, (int)d_dim);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_corr));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attention_mha_bwd::run);

}  // namespace attention_mha_bwd