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
constexpr float SCALE = 0.08838834764831845f;

template <int M, int N, int K, bool A_COL, bool B_COL, bool C_COL>
__device__ void wmma_gemm(const __nv_bfloat16* A, int lda,
                          const __nv_bfloat16* B, int ldb,
                          float* C, int ldc, bool accum) {
    constexpr int TM = 16, TN = 16, TK = 16;
    constexpr int ntm = M / TM, ntn = N / TN, ntk = K / TK;
    const int warp = threadIdx.x / 32;
    const int total = ntm * ntn;
    for (int t = warp; t < total; t += 4) {
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

__device__ __forceinline__ void load_tile_strided(
        const __nv_bfloat16* gptr, int64_t gstride,
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
    float* sS = reinterpret_cast<float*>(sdO + BM * D);
    float* sdP = sS + BM * BN;
    __nv_bfloat16* sP = reinterpret_cast<__nv_bfloat16*>(sdP + BM * BN);
    __nv_bfloat16* sdS = sP + BM * BN;
    float* sdK_acc = reinterpret_cast<float*>(sdS + BM * BN);
    float* sdV_acc = sdK_acc + BN * D;
    float* sDQ = sdV_acc + BN * D;
    float* sL = sDQ + BM * D;
    float* sD = sL + BM;

    int tid = threadIdx.x;

    // Load persistent K, V tile
    load_tile_strided(K_bh + (size_t)col_base * stride_s, stride_s, sK, D, BN, D, col_base, S);
    load_tile_strided(V_bh + (size_t)col_base * stride_s, stride_s, sV, D, BN, D, col_base, S);

    for (int i = tid; i < BN * D; i += 128) {
        sdK_acc[i] = 0.0f;
        sdV_acc[i] = 0.0f;
    }
    __syncthreads();

    for (int qb = 0; qb < S; qb += BM) {
        load_tile_strided(Q_bh + (size_t)qb * stride_s, stride_s, sQ, D, BM, D, qb, S);
        load_tile_strided(dO_bh + (size_t)qb * stride_s, stride_s, sdO, D, BM, D, qb, S);

        // Load LSE
        for (int i = tid; i < BM; i += 128)
            sL[i] = (qb + i < S) ? L_bh[(size_t)(qb + i) * l_stride_s] : 0.0f;
        __syncthreads();

        // D[r] = sum_c O[qb+r,c] * dO[r,c]
        for (int r = tid; r < BM; r += 128) {
            float acc = 0.0f;
            if (qb + r < S) {
                const __nv_bfloat16* orow = O_bh + (size_t)(qb + r) * stride_s;
                for (int c = 0; c < D; c++)
                    acc += __bfloat162float(orow[c]) * __bfloat162float(sdO[r * D + c]);
            }
            sD[r] = acc;
        }
        __syncthreads();

        // S = Q @ K^T -> sS
        wmma_gemm<BM, BN, D, false, true, false>(sQ, D, sK, D, sS, BN, false);
        __syncthreads();

        // P = exp(S * scale - L) -> sP
        for (int idx = tid; idx < BM * BN; idx += 128) {
            int r = idx / BN, c = idx % BN;
            sP[r * BN + c] = __float2bfloat16(expf(sS[r * BN + c] * SCALE - sL[r]));
        }
        __syncthreads();

        // dP = dO @ V^T -> sdP
        wmma_gemm<BM, BN, D, false, true, false>(sdO, D, sV, D, sdP, BN, false);
        __syncthreads();

        // dS = P * (dP - D) * scale -> sdS
        for (int idx = tid; idx < BM * BN; idx += 128) {
            int r = idx / BN, c = idx % BN;
            float p = __bfloat162float(sP[r * BN + c]);
            sdS[r * BN + c] = __float2bfloat16(p * (sdP[r * BN + c] - sD[r]) * SCALE);
        }
        __syncthreads();

        // dV += P^T @ dO
        wmma_gemm<BN, D, BM, true, false, false>(sP, BN, sdO, D, sdV_acc, D, true);
        __syncthreads();

        // dK += dS^T @ Q
        wmma_gemm<BN, D, BM, true, false, false>(sdS, BN, sQ, D, sdK_acc, D, true);
        __syncthreads();

        // dQ_partial = dS @ K
        wmma_gemm<BM, D, BN, false, false, false>(sdS, BN, sK, D, sDQ, D, false);
        __syncthreads();

        // AtomicAdd dQ
        for (int idx = tid; idx < BM * D; idx += 128) {
            int r = idx / D, c = idx % D;
            if (qb + r < S)
                atomicAdd(&dQ_ws_bh[(size_t)(qb + r) * stride_s + c], sDQ[r * D + c]);
        }
        __syncthreads();
    }

    // Store dK, dV
    for (int idx = tid; idx < BN * D; idx += 128) {
        int r = idx / D, c = idx % D;
        if (col_base + r < S) {
            dK_bh[(size_t)(col_base + r) * stride_s + c] = __float2bfloat16(sdK_acc[r * D + c]);
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

    // Extract strides - handle both [B,H,S,D] and [B,S,H,D] layouts
    const DLTensor* qt = Q;
    int64_t stride_b, stride_h, stride_s, stride_d;
    if (qt->strides) {
        stride_b = qt->strides[0];
        stride_h = qt->strides[1];
        stride_s = qt->strides[2];
        stride_d = qt->strides[3];
    } else {
        stride_b = (int64_t)H * S * D;
        stride_h = (int64_t)S * D;
        stride_s = (int64_t)D;
        stride_d = 1;
    }

    const DLTensor* lt = L;
    int64_t l_stride_b, l_stride_h, l_stride_s;
    if (lt->strides) {
        l_stride_b = lt->strides[0];
        l_stride_h = lt->strides[1];
        l_stride_s = lt->strides[2];
    } else {
        l_stride_b = (int64_t)H * S;
        l_stride_h = (int64_t)S;
        l_stride_s = 1;
    }

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

    // Allocate fp32 workspace for dQ accumulation
    size_t ws_count = (size_t)B * H * S * D;
    float* dQ_ws = nullptr;
    CUDA_CHECK(cudaMallocAsync(&dQ_ws, ws_count * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dQ_ws, 0, ws_count * sizeof(float), stream));

    int num_kv = (S + BN - 1) / BN;
    dim3 grid(num_kv, BH);
    dim3 block(128);

    int smem = BM*D*2 + BN*D*2 + BN*D*2 + BM*D*2
             + BM*BN*4 + BM*BN*4
             + BM*BN*2 + BM*BN*2
             + BN*D*4 + BN*D*4
             + BM*D*4
             + BM*4 + BM*4;
    smem = (smem + 15) & ~15;

    CUDA_CHECK(cudaFuncSetAttribute(attn_bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    attn_bwd_kernel<<<grid, block, smem, stream>>>(
        Q_p, K_p, V_p, O_p, dO_p, L_p, dQ_ws, dK_p, dV_p,
        S, H, stride_b, stride_h, stride_s,
        l_stride_b, l_stride_h, l_stride_s);
    CUDA_CHECK(cudaGetLastError());

    // Convert dQ from fp32 workspace to bf16 output
    int total = (int)ws_count;
    convert_kernel<<<(total+255)/256, 256, 0, stream>>>(dQ_ws, dQ_p, total);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(dQ_ws, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_attn_bwd::run);

}  // namespace tvm_ffi_attn_bwd