#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
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
constexpr int NWARP = 8;
constexpr int NTHREAD = NWARP * 32;
constexpr int LDH = Dh + 8;
constexpr int LDS = BN + 8;

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
    #pragma unroll
    for (int e = lane; e < Dh; e += 32)
        s += __bfloat162float(o[e]) * __bfloat162float(g[e]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffff, s, off);
    if (lane == 0) D[gwarp] = s;
}

__device__ inline void load_tile_pad(const __nv_bfloat16* base, int row0, int rows, int S,
                                      __nv_bfloat16* dst) {
    const int4* src4 = reinterpret_cast<const int4*>(base);
    int4* dst4 = reinterpret_cast<int4*>(dst);
    int units = rows * 16;
    for (int u = threadIdx.x; u < units; u += blockDim.x) {
        int row = u >> 4;
        int c = u & 15;
        int grow = row0 + row;
        int4 val;
        if (grow < S) val = src4[(int64_t)grow * 16 + c];
        else          val = make_int4(0, 0, 0, 0);
        dst4[row * (LDH / 8) + c] = val;
    }
}

typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> AccF;
typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> FragARow;
typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> FragACol;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> FragBRow;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> FragBCol;

// ---------------- Pass 1: dK, dV ----------------
__global__ __launch_bounds__(NTHREAD, 2)
void bwd_dkv_kernel(const __nv_bfloat16* __restrict__ Q,
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
    __nv_bfloat16* sK  = (__nv_bfloat16*)p; p += BN * LDH * 2;
    __nv_bfloat16* sV  = (__nv_bfloat16*)p; p += BN * LDH * 2;
    __nv_bfloat16* sQ  = (__nv_bfloat16*)p; p += BM * LDH * 2;
    __nv_bfloat16* sdO = (__nv_bfloat16*)p; p += BM * LDH * 2;
    __nv_bfloat16* sP  = (__nv_bfloat16*)p; p += BM * LDS * 2;
    __nv_bfloat16* sdS = (__nv_bfloat16*)p; p += BM * LDS * 2;
    float* stage = (float*)p; p += NWARP * 512 * 4;
    float* sL = (float*)p; p += BM * 4;
    float* sD = (float*)p; p += BM * 4;

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    float* stS = stage + warp * 512;
    float* stP = stS + 256;

    AccF accV[4], accK[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) { wmma::fill_fragment(accV[i], 0.0f); wmma::fill_fragment(accK[i], 0.0f); }

    load_tile_pad(Kb, n0, BN, S, sK);
    load_tile_pad(Vb, n0, BN, S, sV);
    __syncthreads();

    int num_q = (S + BM - 1) / BM;
    for (int ib = 0; ib < num_q; ib++) {
        int m0 = ib * BM;
        load_tile_pad(Qb, m0, BM, S, sQ);
        load_tile_pad(dOb, m0, BM, S, sdO);
        for (int m = threadIdx.x; m < BM; m += blockDim.x) {
            int g = m0 + m;
            sL[m] = g < S ? Lb[g] : 0.0f;
            sD[m] = g < S ? Db[g] : 0.0f;
        }
        __syncthreads();

        // Each warp: 2 score tiles -> elementwise -> P, dS
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int t = warp + 8 * i;
            int mt = t >> 2, nt = t & 3;
            AccF accS, accP;
            wmma::fill_fragment(accS, 0.0f);
            wmma::fill_fragment(accP, 0.0f);
            #pragma unroll
            for (int kt = 0; kt < Dh / 16; kt++) {
                FragARow a; FragBCol b;
                wmma::load_matrix_sync(a, sQ + mt * 16 * LDH + kt * 16, LDH);
                wmma::load_matrix_sync(b, sK + kt * 16 + nt * 16 * LDH, LDH);
                wmma::mma_sync(accS, a, b, accS);
                FragARow a2; FragBCol b2;
                wmma::load_matrix_sync(a2, sdO + mt * 16 * LDH + kt * 16, LDH);
                wmma::load_matrix_sync(b2, sV + kt * 16 + nt * 16 * LDH, LDH);
                wmma::mma_sync(accP, a2, b2, accP);
            }
            wmma::store_matrix_sync(stS, accS, 16, wmma::mem_row_major);
            wmma::store_matrix_sync(stP, accP, 16, wmma::mem_row_major);
            __syncwarp();
            for (int idx = lane; idx < 256; idx += 32) {
                int r = idx >> 4, c = idx & 15;
                int m = mt * 16 + r, n = nt * 16 + c;
                int grow = m0 + m, gcol = n0 + n;
                float P = 0.0f, dS = 0.0f;
                if (grow < S && gcol < S) {
                    P = __expf(scale * stS[idx] - sL[m]);
                    dS = scale * P * (stP[idx] - sD[m]);
                }
                sP[m * LDS + n]  = __float2bfloat16(P);
                sdS[m * LDS + n] = __float2bfloat16(dS);
            }
            __syncwarp();
        }
        __syncthreads();

        // dV += P^T @ dO ; dK += dS^T @ Q
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            #pragma unroll
            for (int kt = 0; kt < BM / 16; kt++) {
                FragACol a; FragBRow b;
                wmma::load_matrix_sync(a, sP + nt * 16 + kt * 16 * LDS, LDS);
                wmma::load_matrix_sync(b, sdO + kt * 16 * LDH + warp * 16, LDH);
                wmma::mma_sync(accV[nt], a, b, accV[nt]);
                FragACol a2; FragBRow b2;
                wmma::load_matrix_sync(a2, sdS + nt * 16 + kt * 16 * LDS, LDS);
                wmma::load_matrix_sync(b2, sQ + kt * 16 * LDH + warp * 16, LDH);
                wmma::mma_sync(accK[nt], a2, b2, accK[nt]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        wmma::store_matrix_sync(stS, accV[nt], 16, wmma::mem_row_major);
        __syncwarp();
        for (int j = lane; j < 256; j += 32) {
            int r = j >> 4, c = j & 15;
            int gn = n0 + nt * 16 + r;
            int ge = warp * 16 + c;
            if (gn < S) dV[mat_off + (int64_t)gn * Dh + ge] = __float2bfloat16(stS[j]);
        }
        __syncwarp();
    }
    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        wmma::store_matrix_sync(stS, accK[nt], 16, wmma::mem_row_major);
        __syncwarp();
        for (int j = lane; j < 256; j += 32) {
            int r = j >> 4, c = j & 15;
            int gn = n0 + nt * 16 + r;
            int ge = warp * 16 + c;
            if (gn < S) dK[mat_off + (int64_t)gn * Dh + ge] = __float2bfloat16(stS[j]);
        }
        __syncwarp();
    }
}

// ---------------- Pass 2: dQ ----------------
__global__ __launch_bounds__(NTHREAD, 2)
void bwd_dq_kernel(const __nv_bfloat16* __restrict__ Q,
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
    __nv_bfloat16* sQ  = (__nv_bfloat16*)p; p += BM * LDH * 2;
    __nv_bfloat16* sdO = (__nv_bfloat16*)p; p += BM * LDH * 2;
    __nv_bfloat16* sK  = (__nv_bfloat16*)p; p += BN * LDH * 2;
    __nv_bfloat16* sV  = (__nv_bfloat16*)p; p += BN * LDH * 2;
    __nv_bfloat16* sdS = (__nv_bfloat16*)p; p += BM * LDS * 2;
    float* stage = (float*)p; p += NWARP * 512 * 4;
    float* sL = (float*)p; p += BM * 4;
    float* sD = (float*)p; p += BM * 4;

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    float* stS = stage + warp * 512;
    float* stP = stS + 256;

    AccF accQ[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) wmma::fill_fragment(accQ[i], 0.0f);

    load_tile_pad(Qb, m0, BM, S, sQ);
    load_tile_pad(dOb, m0, BM, S, sdO);
    for (int m = threadIdx.x; m < BM; m += blockDim.x) {
        int g = m0 + m;
        sL[m] = g < S ? Lb[g] : 0.0f;
        sD[m] = g < S ? Db[g] : 0.0f;
    }
    __syncthreads();

    int num_kv = (S + BN - 1) / BN;
    for (int jb = 0; jb < num_kv; jb++) {
        int n0 = jb * BN;
        load_tile_pad(Kb, n0, BN, S, sK);
        load_tile_pad(Vb, n0, BN, S, sV);
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int t = warp + 8 * i;
            int mt = t >> 2, nt = t & 3;
            AccF accS, accP;
            wmma::fill_fragment(accS, 0.0f);
            wmma::fill_fragment(accP, 0.0f);
            #pragma unroll
            for (int kt = 0; kt < Dh / 16; kt++) {
                FragARow a; FragBCol b;
                wmma::load_matrix_sync(a, sQ + mt * 16 * LDH + kt * 16, LDH);
                wmma::load_matrix_sync(b, sK + kt * 16 + nt * 16 * LDH, LDH);
                wmma::mma_sync(accS, a, b, accS);
                FragARow a2; FragBCol b2;
                wmma::load_matrix_sync(a2, sdO + mt * 16 * LDH + kt * 16, LDH);
                wmma::load_matrix_sync(b2, sV + kt * 16 + nt * 16 * LDH, LDH);
                wmma::mma_sync(accP, a2, b2, accP);
            }
            wmma::store_matrix_sync(stS, accS, 16, wmma::mem_row_major);
            wmma::store_matrix_sync(stP, accP, 16, wmma::mem_row_major);
            __syncwarp();
            for (int idx = lane; idx < 256; idx += 32) {
                int r = idx >> 4, c = idx & 15;
                int m = mt * 16 + r, n = nt * 16 + c;
                int grow = m0 + m, gcol = n0 + n;
                float dS = 0.0f;
                if (grow < S && gcol < S) {
                    float P = __expf(scale * stS[idx] - sL[m]);
                    dS = scale * P * (stP[idx] - sD[m]);
                }
                sdS[m * LDS + n] = __float2bfloat16(dS);
            }
            __syncwarp();
        }
        __syncthreads();

        #pragma unroll
        for (int mt = 0; mt < 4; mt++) {
            #pragma unroll
            for (int kt = 0; kt < BN / 16; kt++) {
                FragARow a; FragBRow b;
                wmma::load_matrix_sync(a, sdS + mt * 16 * LDS + kt * 16, LDS);
                wmma::load_matrix_sync(b, sK + kt * 16 * LDH + warp * 16, LDH);
                wmma::mma_sync(accQ[mt], a, b, accQ[mt]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mt = 0; mt < 4; mt++) {
        wmma::store_matrix_sync(stS, accQ[mt], 16, wmma::mem_row_major);
        __syncwarp();
        for (int j = lane; j < 256; j += 32) {
            int r = j >> 4, c = j & 15;
            int gm = m0 + mt * 16 + r;
            int ge = warp * 16 + c;
            if (gm < S) dQ[mat_off + (int64_t)gm * Dh + ge] = __float2bfloat16(stS[j]);
        }
        __syncwarp();
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

    size_t smem_dkv = (size_t)(BN*LDH + BN*LDH + BM*LDH + BM*LDH)*2
                    + (size_t)(BM*LDS + BM*LDS)*2
                    + (size_t)(NWARP*512)*4
                    + (size_t)(BM + BM)*4;
    size_t smem_dq  = (size_t)(BM*LDH + BM*LDH + BN*LDH + BN*LDH)*2
                    + (size_t)(BM*LDS)*2
                    + (size_t)(NWARP*512)*4
                    + (size_t)(BM + BM)*4;

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