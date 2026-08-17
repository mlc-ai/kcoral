#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
    }                                                              \
} while(0)

namespace mha {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int HD = 128;      // head dimension (fixed = 128)
constexpr int WARPS = 4;
constexpr int THREADS = WARPS * 32;

__device__ __forceinline__ void load_tile(const __nv_bfloat16* gbase, int grow0,
                                           int S, int rows, __nv_bfloat16* smem, int tid) {
    const int vecPerRow = HD / 8; // 16 (float4 = 8 bf16)
    int totalVec = rows * vecPerRow;
    for (int v = tid; v < totalVec; v += THREADS) {
        int r = v / vecPerRow;
        int c = (v % vecPerRow) * 8;
        int grow = grow0 + r;
        float4 val;
        if (grow < S) {
            val = *reinterpret_cast<const float4*>(gbase + (int64_t)grow * HD + c);
        } else {
            val = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        *reinterpret_cast<float4*>(smem + r * HD + c) = val;
    }
}

__global__ void mha_kernel(const __nv_bfloat16* __restrict__ Q,
                           const __nv_bfloat16* __restrict__ K,
                           const __nv_bfloat16* __restrict__ V,
                           __nv_bfloat16* __restrict__ O,
                           float* __restrict__ LSE,
                           int B, int H, int S, float scale) {
    extern __shared__ char smem_raw[];
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* Ks = Qs + BM * HD;
    __nv_bfloat16* Vs = Ks + BN * HD;
    float*         Ss = reinterpret_cast<float*>(Vs + BN * HD);
    __nv_bfloat16* Ps = reinterpret_cast<__nv_bfloat16*>(Ss + BM * BN);
    float*         Os = reinterpret_cast<float*>(Ps + BM * BN);
    float*       mrow = Os + BM * HD;
    float*       lrow = mrow + BM;
    float*       corr = lrow + BM;

    int b = blockIdx.z;
    int h = blockIdx.y;
    int qb = blockIdx.x;
    int q0 = qb * BM;
    int tid = threadIdx.x;
    int warp = tid >> 5;

    int64_t hbase = (int64_t)(b * H + h) * S * HD;
    const __nv_bfloat16* Qbase = Q + hbase;
    const __nv_bfloat16* Kbase = K + hbase;
    const __nv_bfloat16* Vbase = V + hbase;

    // Load Q tile (resident).
    load_tile(Qbase, q0, S, BM, Qs, tid);

    // Init accumulators.
    for (int idx = tid; idx < BM * HD; idx += THREADS) Os[idx] = 0.f;
    if (tid < BM) { mrow[tid] = -INFINITY; lrow[tid] = 0.f; }
    __syncthreads();

    int numKB = (S + BN - 1) / BN;
    for (int kbi = 0; kbi < numKB; kbi++) {
        int kb = kbi * BN;
        load_tile(Kbase, kb, S, BN, Ks, tid);
        load_tile(Vbase, kb, S, BN, Vs, tid);
        __syncthreads();

        // ---- S = Q @ K^T ----  (B operand loaded col_major => transpose of K)
        {
            int rowtile = warp; // 0..3
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[BN / 16];
            #pragma unroll
            for (int nt = 0; nt < BN / 16; nt++) wmma::fill_fragment(acc[nt], 0.f);
            #pragma unroll
            for (int k = 0; k < HD / 16; k++) {
                wmma::load_matrix_sync(a, &Qs[rowtile * 16 * HD + k * 16], HD);
                #pragma unroll
                for (int nt = 0; nt < BN / 16; nt++) {
                    wmma::load_matrix_sync(bf, &Ks[nt * 16 * HD + k * 16], HD);
                    wmma::mma_sync(acc[nt], a, bf, acc[nt]);
                }
            }
            #pragma unroll
            for (int nt = 0; nt < BN / 16; nt++)
                wmma::store_matrix_sync(&Ss[rowtile * 16 * BN + nt * 16], acc[nt], BN,
                                        wmma::mem_row_major);
        }
        __syncthreads();

        // ---- online softmax (one thread per row) ----
        if (tid < BM) {
            int i = tid;
            int qg = q0 + i;
            if (qg < S) {
                float bmax = -INFINITY;
                for (int j = 0; j < BN; j++) {
                    int jj = (j + i) & (BN - 1);          // permuted to avoid bank conflicts
                    int kg = kb + jj;
                    float v = (kg < S) ? Ss[i * BN + jj] * scale : -INFINITY;
                    bmax = fmaxf(bmax, v);
                }
                float m_old = mrow[i];
                float m_new = fmaxf(m_old, bmax);
                float c = (m_old == -INFINITY) ? 0.f : __expf(m_old - m_new);
                float sump = 0.f;
                for (int j = 0; j < BN; j++) {
                    int jj = (j + i) & (BN - 1);
                    int kg = kb + jj;
                    float v = (kg < S) ? Ss[i * BN + jj] * scale : -INFINITY;
                    float p = (v == -INFINITY) ? 0.f : __expf(v - m_new);
                    sump += p;
                    Ps[i * BN + jj] = __float2bfloat16(p);
                }
                float l_old = lrow[i];
                mrow[i] = m_new;
                lrow[i] = l_old * c + sump;
                corr[i] = c;
            } else {
                for (int j = 0; j < BN; j++) Ps[i * BN + j] = __float2bfloat16(0.f);
                corr[i] = 1.f;
            }
        }
        __syncthreads();

        // ---- rescale O accumulator by per-row correction ----
        for (int idx = tid; idx < BM * HD; idx += THREADS) {
            int r = idx / HD;
            Os[idx] *= corr[r];
        }
        __syncthreads();

        // ---- O += P @ V  (accumulate in shared via load/store accumulator frag) ----
        {
            int rowtile = warp;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            #pragma unroll
            for (int nt = 0; nt < HD / 16; nt++) {
                wmma::load_matrix_sync(acc, &Os[rowtile * 16 * HD + nt * 16], HD,
                                       wmma::mem_row_major);
                #pragma unroll
                for (int kk = 0; kk < BN / 16; kk++) {
                    wmma::load_matrix_sync(a, &Ps[rowtile * 16 * BN + kk * 16], BN);
                    wmma::load_matrix_sync(bf, &Vs[kk * 16 * HD + nt * 16], HD);
                    wmma::mma_sync(acc, a, bf, acc);
                }
                wmma::store_matrix_sync(&Os[rowtile * 16 * HD + nt * 16], acc, HD,
                                        wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    // ---- final normalization + write out ----
    if (tid < BM) {
        int i = tid;
        int qg = q0 + i;
        if (qg < S) {
            float l = lrow[i];
            float inv = (l > 0.f) ? 1.f / l : 0.f;
            __nv_bfloat16* Ob = O + hbase + (int64_t)qg * HD;
            for (int d = 0; d < HD; d++) {
                Ob[d] = __float2bfloat16(Os[i * HD + d] * inv);
            }
            LSE[(int64_t)(b * H + h) * S + qg] = mrow[i] + logf(l);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int Bn = (int)Q.size(0);
    int Hn = (int)Q.size(1);
    int Sn = (int)Q.size(2);
    int Dn = (int)Q.size(3);

    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSEp = static_cast<float*>(LSE.data_ptr());

    float scale = 1.0f / sqrtf((float)Dn);

    int numQB = (Sn + BM - 1) / BM;
    dim3 grid(numQB, Hn, Bn);
    dim3 block(THREADS);

    size_t smem =
          (size_t)BM * HD * 2   // Qs
        + (size_t)BN * HD * 2   // Ks
        + (size_t)BN * HD * 2   // Vs
        + (size_t)BM * BN * 4   // Ss
        + (size_t)BM * BN * 2   // Ps
        + (size_t)BM * HD * 4   // Os
        + (size_t)BM * 4 * 3;   // mrow,lrow,corr

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    mha_kernel<<<grid, block, smem, stream>>>(Qp, Kp, Vp, Op, LSEp, Bn, Hn, Sn, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace mha

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha::run);