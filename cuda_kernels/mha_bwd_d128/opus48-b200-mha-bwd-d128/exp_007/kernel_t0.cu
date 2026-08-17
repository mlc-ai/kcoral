#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd {

using namespace nvcuda;

constexpr int Dh = 128;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int NWARP = 4;
constexpr int NTHREAD = NWARP * 32;

// ---------------- precompute D = rowsum(O ⊙ dO) ----------------
__global__ void compute_D_kernel(const __nv_bfloat16* __restrict__ O,
                                 const __nv_bfloat16* __restrict__ dO,
                                 float* __restrict__ D,
                                 int64_t total_rows) {
    int warps_per_block = blockDim.x / 32;
    int64_t gwarp = (int64_t)blockIdx.x * warps_per_block + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (gwarp >= total_rows) return;
    const __nv_bfloat16* o = O + gwarp * Dh;
    const __nv_bfloat16* g = dO + gwarp * Dh;
    float s = 0.0f;
    for (int e = lane; e < Dh; e += 32) {
        s += __bfloat162float(o[e]) * __bfloat162float(g[e]);
    }
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffff, s, off);
    if (lane == 0) D[gwarp] = s;
}

// ---------------- load a [rows x Dh] bf16 tile into shared (zero-pad) ----------------
__device__ inline void load_tile(const __nv_bfloat16* base, int row0, int S,
                                 int rows, __nv_bfloat16* dst) {
    const int4* src4 = reinterpret_cast<const int4*>(base);
    int4* dst4 = reinterpret_cast<int4*>(dst);
    int units = rows * 16;  // 16 int4 (8 bf16) per row of 128
    for (int u = threadIdx.x; u < units; u += blockDim.x) {
        int row = u >> 4;
        int c8 = u & 15;
        int grow = row0 + row;
        int4 val;
        if (grow < S) val = src4[(int64_t)grow * 16 + c8];
        else          val = make_int4(0, 0, 0, 0);
        dst4[u] = val;
    }
}

// ---------------- generic WMMA gemm (C = A@B, or C += A@B) ----------------
template<typename LA, typename LB>
__device__ inline void wmma_gemm(const __nv_bfloat16* A, int lda,
                                 const __nv_bfloat16* B, int ldb,
                                 float* C, int ldc,
                                 int MT, int NT, int KT,
                                 bool acc_from_C) {
    const bool ARow = std::is_same<LA, wmma::row_major>::value;
    const bool BRow = std::is_same<LB, wmma::row_major>::value;
    int warp = threadIdx.x >> 5;
    for (int t = warp; t < MT * NT; t += NWARP) {
        int mt = t / NT;
        int nt = t % NT;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        if (acc_from_C)
            wmma::load_matrix_sync(acc, C + mt * 16 * ldc + nt * 16, ldc, wmma::mem_row_major);
        else
            wmma::fill_fragment(acc, 0.0f);
        for (int kt = 0; kt < KT; kt++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, LA> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, LB> b;
            int aoff = ARow ? (mt * 16 * lda + kt * 16) : (mt * 16 + kt * 16 * lda);
            int boff = BRow ? (kt * 16 * ldb + nt * 16) : (kt * 16 + nt * 16 * ldb);
            wmma::load_matrix_sync(a, A + aoff, lda);
            wmma::load_matrix_sync(b, B + boff, ldb);
            wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(C + mt * 16 * ldc + nt * 16, acc, ldc, wmma::mem_row_major);
    }
}

// ---------------- Pass: dK, dV (one CTA per KV block) ----------------
__global__ void bwd_dkv_kernel(const __nv_bfloat16* __restrict__ Q,
                               const __nv_bfloat16* __restrict__ K,
                               const __nv_bfloat16* __restrict__ V,
                               const __nv_bfloat16* __restrict__ dO,
                               const float* __restrict__ L,
                               const float* __restrict__ Dsum,
                               __nv_bfloat16* __restrict__ dK,
                               __nv_bfloat16* __restrict__ dV,
                               int S, float scale) {
    extern __shared__ char smem[];
    int jb = blockIdx.x;
    int bh = blockIdx.y;
    int n0 = jb * BN;
    const int64_t mat_off = (int64_t)bh * S * Dh;
    const __nv_bfloat16* Qb = Q + mat_off;
    const __nv_bfloat16* Kb = K + mat_off;
    const __nv_bfloat16* Vb = V + mat_off;
    const __nv_bfloat16* dOb = dO + mat_off;
    const float* Lb = L + (int64_t)bh * S;
    const float* Db = Dsum + (int64_t)bh * S;

    char* p = smem;
    __nv_bfloat16* sQ  = (__nv_bfloat16*)p; p += BM * Dh * 2;
    __nv_bfloat16* sdO = (__nv_bfloat16*)p; p += BM * Dh * 2;
    __nv_bfloat16* sK  = (__nv_bfloat16*)p; p += BN * Dh * 2;
    __nv_bfloat16* sV  = (__nv_bfloat16*)p; p += BN * Dh * 2;
    float* sScore = (float*)p; p += BM * BN * 4;
    float* sdP    = (float*)p; p += BM * BN * 4;
    __nv_bfloat16* sP  = (__nv_bfloat16*)p; p += BM * BN * 2;
    __nv_bfloat16* sdS = (__nv_bfloat16*)p; p += BM * BN * 2;
    float* dV_acc = (float*)p; p += BN * Dh * 4;
    float* dK_acc = (float*)p; p += BN * Dh * 4;
    float* sL = (float*)p; p += BM * 4;
    float* sD = (float*)p; p += BM * 4;

    load_tile(Kb, n0, S, BN, sK);
    load_tile(Vb, n0, S, BN, sV);
    for (int i = threadIdx.x; i < BN * Dh; i += blockDim.x) { dV_acc[i] = 0.0f; dK_acc[i] = 0.0f; }
    __syncthreads();

    int num_q = (S + BM - 1) / BM;
    for (int ib = 0; ib < num_q; ib++) {
        int m0 = ib * BM;
        load_tile(Qb, m0, S, BM, sQ);
        load_tile(dOb, m0, S, BM, sdO);
        for (int m = threadIdx.x; m < BM; m += blockDim.x) {
            int g = m0 + m;
            sL[m] = g < S ? Lb[g] : 0.0f;
            sD[m] = g < S ? Db[g] : 0.0f;
        }
        __syncthreads();

        // S = Q @ K^T   ;  dP = dO @ V^T
        wmma_gemm<wmma::row_major, wmma::col_major>(sQ, Dh, sK, Dh, sScore, BN, BM/16, BN/16, Dh/16, false);
        wmma_gemm<wmma::row_major, wmma::col_major>(sdO, Dh, sV, Dh, sdP, BN, BM/16, BN/16, Dh/16, false);
        __syncthreads();

        // elementwise -> P (bf16), dS' (bf16)
        for (int idx = threadIdx.x; idx < BM * BN; idx += blockDim.x) {
            int m = idx / BN;
            int n = idx % BN;
            int grow = m0 + m;
            int gcol = n0 + n;
            float P, dS;
            if (grow < S && gcol < S) {
                float sv = sScore[idx];
                P = __expf(scale * sv - sL[m]);
                float dPv = sdP[idx];
                dS = scale * P * (dPv - sD[m]);
            } else { P = 0.0f; dS = 0.0f; }
            sP[idx]  = __float2bfloat16(P);
            sdS[idx] = __float2bfloat16(dS);
        }
        __syncthreads();

        // dV += P^T @ dO   ;   dK += dS'^T @ Q
        wmma_gemm<wmma::col_major, wmma::row_major>(sP, BN, sdO, Dh, dV_acc, Dh, BN/16, Dh/16, BM/16, true);
        wmma_gemm<wmma::col_major, wmma::row_major>(sdS, BN, sQ, Dh, dK_acc, Dh, BN/16, Dh/16, BM/16, true);
        __syncthreads();
    }

    for (int idx = threadIdx.x; idx < BN * Dh; idx += blockDim.x) {
        int row = idx / Dh;
        int col = idx % Dh;
        int g = n0 + row;
        if (g < S) {
            dV[mat_off + (int64_t)g * Dh + col] = __float2bfloat16(dV_acc[idx]);
            dK[mat_off + (int64_t)g * Dh + col] = __float2bfloat16(dK_acc[idx]);
        }
    }
}

// ---------------- Pass: dQ (one CTA per Q block) ----------------
__global__ void bwd_dq_kernel(const __nv_bfloat16* __restrict__ Q,
                              const __nv_bfloat16* __restrict__ K,
                              const __nv_bfloat16* __restrict__ V,
                              const __nv_bfloat16* __restrict__ dO,
                              const float* __restrict__ L,
                              const float* __restrict__ Dsum,
                              __nv_bfloat16* __restrict__ dQ,
                              int S, float scale) {
    extern __shared__ char smem[];
    int ib = blockIdx.x;
    int bh = blockIdx.y;
    int m0 = ib * BM;
    const int64_t mat_off = (int64_t)bh * S * Dh;
    const __nv_bfloat16* Qb = Q + mat_off;
    const __nv_bfloat16* Kb = K + mat_off;
    const __nv_bfloat16* Vb = V + mat_off;
    const __nv_bfloat16* dOb = dO + mat_off;
    const float* Lb = L + (int64_t)bh * S;
    const float* Db = Dsum + (int64_t)bh * S;

    char* p = smem;
    __nv_bfloat16* sQ  = (__nv_bfloat16*)p; p += BM * Dh * 2;
    __nv_bfloat16* sdO = (__nv_bfloat16*)p; p += BM * Dh * 2;
    __nv_bfloat16* sK  = (__nv_bfloat16*)p; p += BN * Dh * 2;
    __nv_bfloat16* sV  = (__nv_bfloat16*)p; p += BN * Dh * 2;
    float* sScore = (float*)p; p += BM * BN * 4;
    float* sdP    = (float*)p; p += BM * BN * 4;
    __nv_bfloat16* sdS = (__nv_bfloat16*)p; p += BM * BN * 2;
    float* dQ_acc = (float*)p; p += BM * Dh * 4;
    float* sL = (float*)p; p += BM * 4;
    float* sD = (float*)p; p += BM * 4;

    load_tile(Qb, m0, S, BM, sQ);
    load_tile(dOb, m0, S, BM, sdO);
    for (int m = threadIdx.x; m < BM; m += blockDim.x) {
        int g = m0 + m;
        sL[m] = g < S ? Lb[g] : 0.0f;
        sD[m] = g < S ? Db[g] : 0.0f;
    }
    for (int i = threadIdx.x; i < BM * Dh; i += blockDim.x) dQ_acc[i] = 0.0f;
    __syncthreads();

    int num_kv = (S + BN - 1) / BN;
    for (int jb = 0; jb < num_kv; jb++) {
        int n0 = jb * BN;
        load_tile(Kb, n0, S, BN, sK);
        load_tile(Vb, n0, S, BN, sV);
        __syncthreads();

        wmma_gemm<wmma::row_major, wmma::col_major>(sQ, Dh, sK, Dh, sScore, BN, BM/16, BN/16, Dh/16, false);
        wmma_gemm<wmma::row_major, wmma::col_major>(sdO, Dh, sV, Dh, sdP, BN, BM/16, BN/16, Dh/16, false);
        __syncthreads();

        for (int idx = threadIdx.x; idx < BM * BN; idx += blockDim.x) {
            int m = idx / BN;
            int n = idx % BN;
            int grow = m0 + m;
            int gcol = n0 + n;
            float dS;
            if (grow < S && gcol < S) {
                float sv = sScore[idx];
                float P = __expf(scale * sv - sL[m]);
                float dPv = sdP[idx];
                dS = scale * P * (dPv - sD[m]);
            } else dS = 0.0f;
            sdS[idx] = __float2bfloat16(dS);
        }
        __syncthreads();

        // dQ += dS' @ K
        wmma_gemm<wmma::row_major, wmma::row_major>(sdS, BN, sK, Dh, dQ_acc, Dh, BM/16, Dh/16, BN/16, true);
        __syncthreads();
    }

    for (int idx = threadIdx.x; idx < BM * Dh; idx += blockDim.x) {
        int row = idx / Dh;
        int col = idx % Dh;
        int g = m0 + row;
        if (g < S)
            dQ[mat_off + (int64_t)g * Dh + col] = __float2bfloat16(dQ_acc[idx]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t BH = B * H;

    const __nv_bfloat16* Qp  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* Op  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dOp = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* Lp = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQp = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dKp = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dVp = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float scale = 1.0f / sqrtf((float)Dh);

    float* Dbuf = nullptr;
    CUDA_CHECK(cudaMalloc(&Dbuf, sizeof(float) * BH * S));

    int64_t total_rows = BH * S;
    {
        int block = 256;
        int warps_per_block = block / 32;
        int64_t grid = (total_rows + warps_per_block - 1) / warps_per_block;
        compute_D_kernel<<<(unsigned int)grid, block, 0, stream>>>(Op, dOp, Dbuf, total_rows);
        CUDA_CHECK(cudaGetLastError());
    }

    size_t smem_dkv =
        (size_t)(BM * Dh + BM * Dh + BN * Dh + BN * Dh) * 2 +
        (size_t)(BM * BN + BM * BN) * 4 +
        (size_t)(BM * BN + BM * BN) * 2 +
        (size_t)(BN * Dh + BN * Dh) * 4 +
        (size_t)(BM + BM) * 4;

    size_t smem_dq =
        (size_t)(BM * Dh + BM * Dh + BN * Dh + BN * Dh) * 2 +
        (size_t)(BM * BN + BM * BN) * 4 +
        (size_t)(BM * BN) * 2 +
        (size_t)(BM * Dh) * 4 +
        (size_t)(BM + BM) * 4;

    CUDA_CHECK(cudaFuncSetAttribute(bwd_dkv_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dkv));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dq));

    int num_q  = (int)((S + BM - 1) / BM);
    int num_kv = (int)((S + BN - 1) / BN);

    dim3 grid_dkv((unsigned int)num_kv, (unsigned int)BH);
    bwd_dkv_kernel<<<grid_dkv, NTHREAD, smem_dkv, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Dbuf, dKp, dVp, (int)S, scale);
    CUDA_CHECK(cudaGetLastError());

    dim3 grid_dq((unsigned int)num_q, (unsigned int)BH);
    bwd_dq_kernel<<<grid_dq, NTHREAD, smem_dq, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Dbuf, dQp, (int)S, scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(Dbuf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd