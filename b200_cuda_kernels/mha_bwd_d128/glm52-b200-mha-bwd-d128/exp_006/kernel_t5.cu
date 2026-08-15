#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace tvm_ffi_attn_bwd {

constexpr int D = 128;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int NTHR = 256;
constexpr int NWARPS = NTHR / 32;
constexpr float SCALE = 0.08838834764831845f;

template <int M, int N, int K, bool A_COL, bool B_COL, bool C_COL>
__device__ void wmma_gemm(const __nv_bfloat16* A, int lda,
                          const __nv_bfloat16* B, int ldb,
                          float* C, int ldc, bool accum) {
    constexpr int TM = 16, TN = 16, TK = 16;
    constexpr int ntm = M / TM, ntn = N / TN, ntk = K / TK;
    const int warp = threadIdx.x / 32;
    const int total = ntm * ntn;
    for (int t = warp; t < total; t += NWARPS) {
        int ti = t / ntn, tj = t % ntn;
        wmma::fragment<wmma::accumulator, TM, TN, TK, float> cf;
        if (accum) {
            const float* cptr = C + (C_COL ? (tj * TN * ldc + ti * TM) : (ti * TM * ldc + tj * TN));
            wmma::load_matrix_sync(cf, cptr, ldc, C_COL ? wmma::mem_col_major : wmma::mem_row_major);
        } else {
            wmma::fill_fragment(cf, 0.0f);
        }
        for (int kk = 0; kk < ntk; kk++) {
            if constexpr (!A_COL && !B_COL) {
                wmma::fragment<wmma::matrix_a, TM, TN, TK, __nv_bfloat16, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, TM, TN, TK, __nv_bfloat16, wmma::row_major> bf;
                wmma::load_matrix_sync(af, A + ti * TM * lda + kk * TK, lda);
                wmma::load_matrix_sync(bf, B + kk * TK * ldb + tj * TN, ldb);
                wmma::mma_sync(cf, af, bf, cf);
            } else if constexpr (!A_COL && B_COL) {
                wmma::fragment<wmma::matrix_a, TM, TN, TK, __nv_bfloat16, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, TM, TN, TK, __nv_bfloat16, wmma::col_major> bf;
                wmma::load_matrix_sync(af, A + ti * TM * lda + kk * TK, lda);
                wmma::load_matrix_sync(bf, B + tj * TN * ldb + kk * TK, ldb);
                wmma::mma_sync(cf, af, bf, cf);
            } else if constexpr (A_COL && !B_COL) {
                wmma::fragment<wmma::matrix_a, TM, TN, TK, __nv_bfloat16, wmma::col_major> af;
                wmma::fragment<wmma::matrix_b, TM, TN, TK, __nv_bfloat16, wmma::row_major> bf;
                wmma::load_matrix_sync(af, A + kk * TK * lda + ti * TM, lda);
                wmma::load_matrix_sync(bf, B + kk * TK * ldb + tj * TN, ldb);
                wmma::mma_sync(cf, af, bf, cf);
            } else {
                wmma::fragment<wmma::matrix_a, TM, TN, TK, __nv_bfloat16, wmma::col_major> af;
                wmma::fragment<wmma::matrix_b, TM, TN, TK, __nv_bfloat16, wmma::col_major> bf;
                wmma::load_matrix_sync(af, A + kk * TK * lda + ti * TM, lda);
                wmma::load_matrix_sync(bf, B + tj * TN * ldb + kk * TK, ldb);
                wmma::mma_sync(cf, af, bf, cf);
            }
        }
        float* cptr = C + (C_COL ? (tj * TN * ldc + ti * TM) : (ti * TM * ldc + tj * TN));
        wmma::store_matrix_sync(cptr, cf, ldc, C_COL ? wmma::mem_col_major : wmma::mem_row_major);
    }
}

__device__ __forceinline__ void load_tile(
        const __nv_bfloat16* gptr, int64_t gstride,
        __nv_bfloat16* sptr, int sstride,
        int rows, int cols, int row_base, int S_limit) {
    int tid = threadIdx.x;
    int nvec = cols / 8;
    int total = rows * nvec;
    for (int idx = tid; idx < total; idx += NTHR) {
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
        __nv_bfloat16* __restrict__ dK_g,
        __nv_bfloat16* __restrict__ dV_g,
        int S, int H,
        int64_t stride_b, int64_t stride_h, int64_t stride_s,
        int64_t l_stride_b, int64_t l_stride_h, int64_t l_stride_s)
{
    int kv_block = blockIdx.x;
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int col_base = kv_block * BN;
    if (col_base >= S) return;

    size_t base4 = (size_t)b * stride_b + h * stride_h;
    size_t base_l = (size_t)b * l_stride_b + h * l_stride_h;

    const __nv_bfloat16* Q_bh = Q + base4;
    const __nv_bfloat16* K_bh = K + base4;
    const __nv_bfloat16* V_bh = V + base4;
    const __nv_bfloat16* O_bh = O + base4;
    const __nv_bfloat16* dO_bh = dO + base4;
    const float* L_bh = L + base_l;
    float* dQ_ws_bh = dQ_ws + base4;
    __nv_bfloat16* dK_bh = dK_g + base4;
    __nv_bfloat16* dV_bh = dV_g + base4;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sK = sQ + BM * D;
    __nv_bfloat16* sV = sK + BN * D;
    __nv_bfloat16* sdO = sV + BN * D;
    // Reuse sS buffer for sO during D computation (both 16KB)
    __nv_bfloat16* sO = reinterpret_cast<__nv_bfloat16*>(sdO + BM * D);
    float* sS = reinterpret_cast<float*>(sdO + BM * D);  // same address, reused after D
    float* sP = sS + BM * BN;
    float* sdK_acc = sP + BM * BN;
    float* sdV_acc = sdK_acc + BN * D;
    float* sDQ = sdV_acc + BN * D;
    float* sL = sDQ + BM * D;
    float* sD = sL + BM;
    __nv_bfloat16* sP_bf16 = reinterpret_cast<__nv_bfloat16*>(sD + BM);
    __nv_bfloat16* sdS_bf16 = sP_bf16 + BM * BN;

    int tid = threadIdx.x;

    // Load persistent K, V tile
    load_tile(K_bh + (size_t)col_base * stride_s, stride_s, sK, D, BN, D, col_base, S);
    load_tile(V_bh + (size_t)col_base * stride_s, stride_s, sV, D, BN, D, col_base, S);

    for (int i = tid; i < BN * D; i += NTHR) {
        sdK_acc[i] = 0.0f;
        sdV_acc[i] = 0.0f;
    }
    __syncthreads();

    for (int qb = 0; qb < S; qb += BM) {
        load_tile(Q_bh + (size_t)qb * stride_s, stride_s, sQ, D, BM, D, qb, S);
        load_tile(dO_bh + (size_t)qb * stride_s, stride_s, sdO, D, BM, D, qb, S);
        // Load O into sO (reuses sS buffer, 16KB bf16)
        load_tile(O_bh + (size_t)qb * stride_s, stride_s, sO, D, BM, D, qb, S);

        for (int i = tid; i < BM; i += NTHR)
            sL[i] = (qb + i < S) ? L_bh[(size_t)(qb + i) * l_stride_s] : 0.0f;
        __syncthreads();

        // D[r] = sum_c O[r,c] * dO[r,c] -- 4 threads per row, vectorized
        {
            int r = tid / 4;
            int sub = tid % 4;
            if (r < BM) {
                float acc = 0.0f;
                if (qb + r < S) {
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        int c = sub * 32 + i * 8;
                        float4 ov = *reinterpret_cast<float4*>(&sO[r * D + c]);
                        float4 dv = *reinterpret_cast<float4*>(&sdO[r * D + c]);
                        __nv_bfloat16* oh = reinterpret_cast<__nv_bfloat16*>(&ov);
                        __nv_bfloat16* dh = reinterpret_cast<__nv_bfloat16*>(&dv);
                        #pragma unroll
                        for (int k = 0; k < 8; k++)
                            acc += __bfloat162float(oh[k]) * __bfloat162float(dh[k]);
                    }
                }
                // Reduce 4 partial sums via shfl
                acc += __shfl_xor_sync(0xffffffff, acc, 1);
                acc += __shfl_xor_sync(0xffffffff, acc, 2);
                if (sub == 0) sD[r] = acc;
            }
        }
        __syncthreads();

        // S = Q @ K^T * scale -> sS [BM, BN] fp32 (now using sS as float, sO no longer needed)
        wmma_gemm<BM, BN, D, false, true, false>(sQ, D, sK, D, sS, BN, false);
        __syncthreads();

        // P = exp(S * scale - L) -> sP (fp32) and sP_bf16
        for (int idx = tid; idx < BM * BN; idx += NTHR) {
            int r = idx / BN, c = idx % BN;
            float p = __expf(sS[r * BN + c] * SCALE - sL[r]);
            sP[r * BN + c] = p;
            sP_bf16[r * BN + c] = __float2bfloat16(p);
        }
        __syncthreads();

        // dP = dO @ V^T -> sS (overwrite S)
        wmma_gemm<BM, BN, D, false, true, false>(sdO, D, sV, D, sS, BN, false);
        __syncthreads();

        // dV += P^T @ dO
        wmma_gemm<BN, D, BM, true, false, false>(sP_bf16, BN, sdO, D, sdV_acc, D, true);
        __syncthreads();

        // dS = P * (dP - D) -> sP (overwrite), sdS_bf16
        for (int idx = tid; idx < BM * BN; idx += NTHR) {
            int r = idx / BN, c = idx % BN;
            float p = sP[r * BN + c];
            float ds = p * (sS[r * BN + c] - sD[r]);
            sP[r * BN + c] = ds;
            sdS_bf16[r * BN + c] = __float2bfloat16(ds);
        }
        __syncthreads();

        // dK += dS^T @ Q * scale (scale applied at store)
        wmma_gemm<BN, D, BM, true, false, false>(sdS_bf16, BN, sQ, D, sdK_acc, D, true);
        __syncthreads();

        // dQ_partial = dS @ K -> sDQ (scale applied during atomicAdd)
        wmma_gemm<BM, D, BN, false, false, false>(sdS_bf16, BN, sK, D, sDQ, D, false);
        __syncthreads();

        // AtomicAdd dQ with scale
        for (int idx = tid; idx < BM * D; idx += NTHR) {
            int r = idx / D, c = idx % D;
            if (qb + r < S)
                atomicAdd(&dQ_ws_bh[(size_t)(qb + r) * stride_s + c], sDQ[r * D + c] * SCALE);
        }
        __syncthreads();
    }

    // Store dK, dV
    for (int idx = tid; idx < BN * D; idx += NTHR) {
        int r = idx / D, c = idx % D;
        if (col_base + r < S) {
            dK_bh[(size_t)(col_base + r) * stride_s + c] = __float2bfloat16(sdK_acc[r * D + c] * SCALE);
            dV_bh[(size_t)(col_base + r) * stride_s + c] = __float2bfloat16(sdV_acc[r * D + c]);
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

    int64_t stride_b = (int64_t)H * S * D;
    int64_t stride_h = (int64_t)S * D;
    int64_t stride_s = (int64_t)D;
    int64_t l_stride_b = (int64_t)H * S;
    int64_t l_stride_h = (int64_t)S;
    int64_t l_stride_s = 1;

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

    size_t ws_count = (size_t)B * H * S * D;
    float* dQ_ws = nullptr;
    CUDA_CHECK(cudaMallocAsync(&dQ_ws, ws_count * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dQ_ws, 0, ws_count * sizeof(float), stream));

    int num_kv = (S + BN - 1) / BN;
    dim3 grid(num_kv, BH);
    dim3 block(NTHR);

    int smem = BM*D*2 + BN*D*2 + BN*D*2 + BM*D*2  // sQ, sK, sV, sdO (bf16)
             + BM*BN*4 + BM*BN*4                   // sS/dP, sP/dS (fp32, sS reused for sO)
             + BN*D*4 + BN*D*4                     // sdK_acc, sdV_acc (fp32)
             + BM*D*4                               // sDQ (fp32)
             + BM*4 + BM*4                          // sL, sD (fp32)
             + BM*BN*2 + BM*BN*2;                   // sP_bf16, sdS_bf16 (bf16)
    smem = (smem + 15) & ~15;

    CUDA_CHECK(cudaFuncSetAttribute(attn_bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    attn_bwd_kernel<<<grid, block, smem, stream>>>(
        Q_p, K_p, V_p, O_p, dO_p, L_p, dQ_ws, dK_p, dV_p,
        S, H, stride_b, stride_h, stride_s,
        l_stride_b, l_stride_h, l_stride_s);
    CUDA_CHECK(cudaGetLastError());

    int total = (int)ws_count;
    convert_kernel<<<(total+255)/256, 256, 0, stream>>>(dQ_ws, dQ_p, total);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(dQ_ws, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_attn_bwd::run);

}  // namespace tvm_ffi_attn_bwd