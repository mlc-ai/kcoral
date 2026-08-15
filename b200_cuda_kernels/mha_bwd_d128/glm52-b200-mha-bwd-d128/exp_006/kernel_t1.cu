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

// C[M,N] = A[M,K] @ B[K,N]
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

__device__ __forceinline__ void load_tile_bf16(
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

__device__ __forceinline__ void load_lse(const float* gptr, float* sptr, int n, int base, int S_limit) {
    int tid = threadIdx.x;
    for (int i = tid; i < n; i += 128) {
        sptr[i] = (base + i < S_limit) ? gptr[base + i] : 0.0f;
    }
}

__device__ __forceinline__ void zero_buffer(float* ptr, int n) {
    int tid = threadIdx.x;
    for (int i = tid; i < n; i += 128) ptr[i] = 0.0f;
}

__global__ void convert_f32_to_bf16_kernel(const float* src, __nv_bfloat16* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(src[idx]);
}

// Shared memory layout:
// sQ:      [BM, D]      bf16
// sK:      [BN, D]      bf16
// sV:      [BN, D]      bf16
// sdO:     [BM, D]      bf16
// sS:      [BM, BN]     f32  (S = Q@K^T)
// sdP:     [BM, BN]     f32  (dP = dO@V^T)
// sP:      [BM, BN]     bf16
// sdS:     [BM, BN]     bf16
// sdK_acc: [BN, D]      f32
// sdV_acc: [BN, D]      f32
// sDQ:     [BM, D]      f32
// sL:      [BM]         f32
// sD:      [BM]         f32
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
    // Layout with explicit offsets
    __nv_bfloat16* sQ   = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sK   = sQ + BM * D;
    __nv_bfloat16* sV   = sK + BN * D;
    __nv_bfloat16* sdO  = sV + BN * D;
    // Align sS to 16 bytes (it's already aligned since previous bf16 buffers end at 16-byte boundary)
    float* sS           = reinterpret_cast<float*>(sdO + BM * D);
    float* sdP          = sS + BM * BN;
    __nv_bfloat16* sP   = reinterpret_cast<__nv_bfloat16*>(sdP + BM * BN);
    __nv_bfloat16* sdS  = sP + BM * BN;
    float* sdK_acc      = reinterpret_cast<float*>(sdS + BM * BN);
    float* sdV_acc      = sdK_acc + BN * D;
    float* sDQ          = sdV_acc + BN * D;
    float* sL           = sDQ + BM * D;
    float* sD           = sL + BM;

    int tid = threadIdx.x;

    // Load persistent K, V tile
    load_tile_bf16(K_bh + (size_t)col_base * D, D, sK, D, BN, D, col_base, S);
    load_tile_bf16(V_bh + (size_t)col_base * D, D, sV, D, BN, D, col_base, S);

    // Zero accumulators
    zero_buffer(sdK_acc, BN * D);
    zero_buffer(sdV_acc, BN * D);
    __syncthreads();

    for (int qb = 0; qb < S; qb += BM) {
        // Load Q, dO, LSE
        load_tile_bf16(Q_bh + (size_t)qb * D, D, sQ, D, BM, D, qb, S);
        load_tile_bf16(dO_bh + (size_t)qb * D, D, sdO, D, BM, D, qb, S);
        load_lse(L_bh + qb, sL, BM, qb, S);
        __syncthreads();

        // D[r] = sum_c O[qb+r, c] * dO[r, c]
        if (tid < BM) {
            int r = tid;
            float acc = 0.0f;
            if (qb + r < S) {
                const __nv_bfloat16* orow = O_bh + (size_t)(qb + r) * D;
                for (int c = 0; c < D; c++) {
                    acc += __bfloat162float(orow[c]) * __bfloat162float(sdO[r * D + c]);
                }
            }
            sD[r] = acc;
        }
        __syncthreads();

        // S = Q @ K^T  -> sS [BM, BN] fp32
        wmma_gemm<BM, BN, D, false, true, false>(sQ, D, sK, D, sS, BN, false);
        __syncthreads();

        // P = exp(S * scale - L) -> sP [BM, BN] bf16
        for (int idx = tid; idx < BM * BN; idx += 128) {
            int r = idx / BN, c = idx % BN;
            float s = sS[r * BN + c] * SCALE;
            float p = expf(s - sL[r]);
            sP[r * BN + c] = __float2bfloat16(p);
        }
        __syncthreads();

        // dP = dO @ V^T -> sdP [BM, BN] fp32  (separate buffer from sS)
        wmma_gemm<BM, BN, D, false, true, false>(sdO, D, sV, D, sdP, BN, false);
        __syncthreads();

        // dS = P * (dP - D) * scale -> sdS [BM, BN] bf16
        for (int idx = tid; idx < BM * BN; idx += 128) {
            int r = idx / BN, c = idx % BN;
            float p = __bfloat162float(sP[r * BN + c]);
            float dp = sdP[r * BN + c];
            float ds = p * (dp - sD[r]) * SCALE;
            sdS[r * BN + c] = __float2bfloat16(ds);
        }
        __syncthreads();

        // dV += P^T @ dO   [BN, D]
        wmma_gemm<BN, D, BM, true, false, false>(sP, BN, sdO, D, sdV_acc, D, true);
        __syncthreads();

        // dK += dS^T @ Q   [BN, D]
        wmma_gemm<BN, D, BM, true, false, false>(sdS, BN, sQ, D, sdK_acc, D, true);
        __syncthreads();

        // dQ_partial = dS @ K  [BM, D]
        wmma_gemm<BM, D, BN, false, false, false>(sdS, BN, sK, D, sDQ, D, false);
        __syncthreads();

        // Atomic add dQ_partial to global workspace
        for (int idx = tid; idx < BM * D; idx += 128) {
            int r = idx / D, c = idx % D;
            if (qb + r < S) {
                atomicAdd(&dQ_ws_bh[(size_t)(qb + r) * D + c], sDQ[r * D + c]);
            }
        }
        __syncthreads();
    }

    // Store dK, dV (each location written exactly once)
    for (int idx = tid; idx < BN * D; idx += 128) {
        int r = idx / D, c = idx % D;
        if (col_base + r < S) {
            dK_bh[(size_t)(col_base + r) * D + c] = __float2bfloat16(sdK_acc[r * D + c]);
            dV_bh[(size_t)(col_base + r) * D + c] = __float2bfloat16(sdV_acc[r * D + c]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
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

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    size_t ws_count = (size_t)BH * S * D;
    float* dQ_ws = nullptr;
    CUDA_CHECK(cudaMallocAsync(&dQ_ws, ws_count * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dQ_ws, 0, ws_count * sizeof(float), stream));

    int num_kv = (S + BN - 1) / BN;
    dim3 grid(num_kv, BH);
    dim3 block(128);

    // Calculate exact shared memory size
    int smem_bytes = 0;
    smem_bytes += BM * D * 2;  // sQ
    smem_bytes += BN * D * 2;  // sK
    smem_bytes += BN * D * 2;  // sV
    smem_bytes += BM * D * 2;  // sdO
    smem_bytes += BM * BN * 4; // sS (fp32)
    smem_bytes += BM * BN * 4; // sdP (fp32)
    smem_bytes += BM * BN * 2; // sP (bf16)
    smem_bytes += BM * BN * 2; // sdS (bf16)
    smem_bytes += BN * D * 4;  // sdK_acc (fp32)
    smem_bytes += BN * D * 4;  // sdV_acc (fp32)
    smem_bytes += BM * D * 4;  // sDQ (fp32)
    smem_bytes += BM * 4;      // sL (fp32)
    smem_bytes += BM * 4;      // sD (fp32)
    // Add padding for alignment
    smem_bytes = (smem_bytes + 15) & ~15;

    CUDA_CHECK(cudaFuncSetAttribute(attn_bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    attn_bwd_kernel<<<grid, block, smem_bytes, stream>>>(
        Q_p, K_p, V_p, O_p, dO_p, L_p, dQ_ws, dK_p, dV_p, S);
    CUDA_CHECK(cudaGetLastError());

    int total_elems = (int)ws_count;
    int cthreads = 256;
    int cblocks = (total_elems + cthreads - 1) / cthreads;
    convert_f32_to_bf16_kernel<<<cblocks, cthreads, 0, stream>>>(dQ_ws, dQ_p, total_elems);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(dQ_ws, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_attn_bwd::run);

}  // namespace tvm_ffi_attn_bwd