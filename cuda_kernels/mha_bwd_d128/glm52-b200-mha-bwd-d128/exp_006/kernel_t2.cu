#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

namespace tvm_ffi_attn_bwd {

constexpr int D = 128;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr float SCALE = 0.08838834764831845f;

__device__ __forceinline__ void load_tile(
        const __nv_bfloat16* gptr, int gstride,
        __nv_bfloat16* sptr, int sstride,
        int rows, int cols, int row_base, int S_limit) {
    int tid = threadIdx.x;
    int nvec = cols / 8;
    int total = rows * nvec;
    for (int idx = tid; idx < total; idx += 128) {
        int r = idx / nvec;
        int c = (idx % nvec) * 8;
        float4 z = {0.0f, 0.0f, 0.0f, 0.0f};
        if (row_base + r < S_limit) {
            z = *reinterpret_cast<const float4*>(gptr + (size_t)r * gstride + c);
        }
        *reinterpret_cast<float4*>(sptr + r * sstride + c) = z;
    }
}

__global__ void attn_bwd_kernel(
        const __nv_bfloat16* __restrict__ Q,
        const __nv_bfloat16* __restrict__ K,
        const __nv_bfloat16* __restrict__ V,
        const __nv_bfloat16* __restrict__ O,
        const __nv_bfloat16* __restrict__ dO,
        const float* __restrict__ L,
        float* __restrict__ dQ_ws,
        __nv_bfloat16* __restrict__ dK,
        __nv_bfloat16* __restrict__ dV,
        int S)
{
    int kv_block = blockIdx.x;
    int bh = blockIdx.y;
    int col_base = kv_block * BN;
    if (col_base >= S) return;

    const size_t bh_off = (size_t)bh * S * D;
    const __nv_bfloat16* Q_bh = Q + bh_off;
    const __nv_bfloat16* K_bh = K + bh_off;
    const __nv_bfloat16* V_bh = V + bh_off;
    const __nv_bfloat16* O_bh = O + bh_off;
    const __nv_bfloat16* dO_bh = dO + bh_off;
    const float* L_bh = L + (size_t)bh * S;
    float* dQ_ws_bh = dQ_ws + bh_off;
    __nv_bfloat16* dK_bh = dK + bh_off;
    __nv_bfloat16* dV_bh = dV + bh_off;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sK = sQ + BM * D;
    __nv_bfloat16* sV = sK + BN * D;
    __nv_bfloat16* sdO = sV + BN * D;
    float* sS_sdP = reinterpret_cast<float*>(sdO + BM * D);
    float* sP_sdS = sS_sdP + BM * BN;
    float* sdK_acc = sP_sdS + BM * BN;
    float* sdV_acc = sdK_acc + BN * D;
    float* sDQ = sdV_acc + BN * D;
    float* sL = sDQ + BM * D;
    float* sD = sL + BM;

    int tid = threadIdx.x;

    load_tile(K_bh + (size_t)col_base * D, D, sK, D, BN, D, col_base, S);
    load_tile(V_bh + (size_t)col_base * D, D, sV, D, BN, D, col_base, S);

    for (int i = tid; i < BN * D; i += 128) {
        sdK_acc[i] = 0.0f;
        sdV_acc[i] = 0.0f;
    }
    __syncthreads();

    for (int qb = 0; qb < S; qb += BM) {
        load_tile(Q_bh + (size_t)qb * D, D, sQ, D, BM, D, qb, S);
        load_tile(dO_bh + (size_t)qb * D, D, sdO, D, BM, D, qb, S);
        for (int i = tid; i < BM; i += 128)
            sL[i] = (qb + i < S) ? L_bh[qb + i] : 0.0f;
        __syncthreads();

        // D[r] = sum_c O[qb+r,c] * dO[r,c]
        for (int r = tid; r < BM; r += 128) {
            float acc = 0.0f;
            if (qb + r < S) {
                const __nv_bfloat16* orow = O_bh + (size_t)(qb + r) * D;
                for (int c = 0; c < D; c++)
                    acc += __bfloat162float(orow[c]) * __bfloat162float(sdO[r * D + c]);
            }
            sD[r] = acc;
        }
        __syncthreads();

        // S = Q @ K^T -> sS_sdP
        for (int idx = tid; idx < BM * BN; idx += 128) {
            int m = idx / BN, n = idx % BN;
            float sum = 0.0f;
            for (int k = 0; k < D; k++)
                sum += __bfloat162float(sQ[m * D + k]) * __bfloat162float(sK[n * D + k]);
            sS_sdP[m * BN + n] = sum;
        }
        __syncthreads();

        // P = exp(S * scale - L) -> sP_sdS
        for (int idx = tid; idx < BM * BN; idx += 128) {
            int m = idx / BN, n = idx % BN;
            sP_sdS[m * BN + n] = expf(sS_sdP[m * BN + n] * SCALE - sL[m]);
        }
        __syncthreads();

        // dP = dO @ V^T -> sS_sdP (overwrite S)
        for (int idx = tid; idx < BM * BN; idx += 128) {
            int m = idx / BN, n = idx % BN;
            float sum = 0.0f;
            for (int k = 0; k < D; k++)
                sum += __bfloat162float(sdO[m * D + k]) * __bfloat162float(sV[n * D + k]);
            sS_sdP[m * BN + n] = sum;
        }
        __syncthreads();

        // dV += P^T @ dO (uses sP_sdS)
        for (int idx = tid; idx < BN * D; idx += 128) {
            int n = idx / D, d = idx % D;
            float sum = 0.0f;
            for (int m = 0; m < BM; m++)
                sum += sP_sdS[m * BN + n] * __bfloat162float(sdO[m * D + d]);
            sdV_acc[n * D + d] += sum;
        }
        __syncthreads();

        // dS = P * (dP - D) * scale -> sP_sdS (overwrite P, uses sS_sdP as dP)
        for (int idx = tid; idx < BM * BN; idx += 128) {
            int m = idx / BN, n = idx % BN;
            sP_sdS[m * BN + n] = sP_sdS[m * BN + n] * (sS_sdP[m * BN + n] - sD[m]) * SCALE;
        }
        __syncthreads();

        // dK += dS^T @ Q (uses sP_sdS as dS)
        for (int idx = tid; idx < BN * D; idx += 128) {
            int n = idx / D, d = idx % D;
            float sum = 0.0f;
            for (int m = 0; m < BM; m++)
                sum += sP_sdS[m * BN + n] * __bfloat162float(sQ[m * D + d]);
            sdK_acc[n * D + d] += sum;
        }
        __syncthreads();

        // dQ = dS @ K (uses sP_sdS as dS)
        for (int idx = tid; idx < BM * D; idx += 128) {
            int m = idx / D, d = idx % D;
            float sum = 0.0f;
            for (int n = 0; n < BN; n++)
                sum += sP_sdS[m * BN + n] * __bfloat162float(sK[n * D + d]);
            sDQ[m * D + d] = sum;
        }
        __syncthreads();

        // AtomicAdd dQ
        for (int idx = tid; idx < BM * D; idx += 128) {
            int m = idx / D, d = idx % D;
            if (qb + m < S)
                atomicAdd(&dQ_ws_bh[(size_t)(qb + m) * D + d], sDQ[m * D + d]);
        }
        __syncthreads();
    }

    // Store dK, dV
    for (int idx = tid; idx < BN * D; idx += 128) {
        int n = idx / D, d = idx % D;
        if (col_base + n < S) {
            dK_bh[(size_t)(col_base + n) * D + d] = __float2bfloat16(sdK_acc[n * D + d]);
            dV_bh[(size_t)(col_base + n) * D + d] = __float2bfloat16(sdV_acc[n * D + d]);
        }
    }
}

__global__ void convert_kernel(const float* src, __nv_bfloat16* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(src[idx]);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = (int)Q.size(0), H = (int)Q.size(1), S = (int)Q.size(2);
    int BH = B * H;

    const __nv_bfloat16* Q_p = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_p = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_p = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_p = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_p = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_p = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_p = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_p = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_p = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    size_t ws_count = (size_t)BH * S * D;
    float* dQ_ws = nullptr;
    CUDA_CHECK(cudaMallocAsync(&dQ_ws, ws_count * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dQ_ws, 0, ws_count * sizeof(float), stream));

    int num_kv = (S + BN - 1) / BN;
    dim3 grid(num_kv, BH);
    dim3 block(128);

    int smem = BM*D*2 + BN*D*2 + BN*D*2 + BM*D*2
             + BM*BN*4 + BM*BN*4
             + BN*D*4 + BN*D*4
             + BM*D*4
             + BM*4 + BM*4;
    smem = (smem + 15) & ~15;

    CUDA_CHECK(cudaFuncSetAttribute(attn_bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    attn_bwd_kernel<<<grid, block, smem, stream>>>(
        Q_p, K_p, V_p, O_p, dO_p, L_p, dQ_ws, dK_p, dV_p, S);
    CUDA_CHECK(cudaGetLastError());

    int total = (int)ws_count;
    convert_kernel<<<(total+255)/256, 256, 0, stream>>>(dQ_ws, dQ_p, total);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(dQ_ws, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_attn_bwd::run);

}  // namespace tvm_ffi_attn_bwd