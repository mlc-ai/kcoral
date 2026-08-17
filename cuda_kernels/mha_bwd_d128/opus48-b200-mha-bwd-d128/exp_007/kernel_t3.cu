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

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

// async load [rows x Dh] bf16 tile into shared, zero-fill OOB rows
__device__ inline void async_load_tile(const __nv_bfloat16* base, int row0, int rows, int S,
                                        __nv_bfloat16* dst) {
    const int4* src4 = reinterpret_cast<const int4*>(base);
    int4* dst4 = reinterpret_cast<int4*>(dst);
    int units = rows * 16;
    for (int u = threadIdx.x; u < units; u += blockDim.x) {
        int row = u >> 4;
        int c8 = u & 15;
        int grow = row0 + row;
        int valid = grow < S;
        int gg = valid ? grow : 0;
        const int4* s = src4 + (int64_t)gg * 16 + c8;
        uint32_t sm = (uint32_t)__cvta_generic_to_shared(dst4 + u);
        int sz = valid ? 16 : 0;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :: "r"(sm), "l"(s), "r"(sz) : "memory");
    }
}

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

typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> AccF;
typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> FragARow;
typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> FragACol;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> FragBRow;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> FragBCol;

// ---------------- Pass 1: dK, dV ----------------
__global__ __launch_bounds__(NTHREAD)
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
    __nv_bfloat16* sK  = (__nv_bfloat16*)p; p += BN * Dh * 2;
    __nv_bfloat16* sV  = (__nv_bfloat16*)p; p += BN * Dh * 2;
    __nv_bfloat16* sQ  = (__nv_bfloat16*)p; p += 2 * BM * Dh * 2;
    __nv_bfloat16* sdO = (__nv_bfloat16*)p; p += 2 * BM * Dh * 2;
    float* sSc = (float*)p; p += BM * BN * 4;
    float* sdP = (float*)p; p += BM * BN * 4;
    __nv_bfloat16* sP  = (__nv_bfloat16*)p; p += BM * BN * 2;
    __nv_bfloat16* sdS = (__nv_bfloat16*)p; p += BM * BN * 2;
    float* sL = (float*)p; p += BM * 4;
    float* sD = (float*)p; p += BM * 4;

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    AccF accV[4], accK[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) { wmma::fill_fragment(accV[i], 0.0f); wmma::fill_fragment(accK[i], 0.0f); }

    int num_q = (S + BM - 1) / BM;

    // preload K, V and Q0, dO0
    async_load_tile(Kb, n0, BN, S, sK);
    async_load_tile(Vb, n0, BN, S, sV);
    async_load_tile(Qb, 0, BM, S, sQ);
    async_load_tile(dOb, 0, BM, S, sdO);
    cp_commit();

    for (int ib = 0; ib < num_q; ib++) {
        cp_wait0();
        __syncthreads();
        int buf = ib & 1;
        int m0 = ib * BM;
        __nv_bfloat16* Qcur = sQ + buf * BM * Dh;
        __nv_bfloat16* dOcur = sdO + buf * BM * Dh;

        // prefetch next
        if (ib + 1 < num_q) {
            int nxt = (ib + 1) & 1;
            async_load_tile(Qb, (ib + 1) * BM, BM, S, sQ + nxt * BM * Dh);
            async_load_tile(dOb, (ib + 1) * BM, BM, S, sdO + nxt * BM * Dh);
            cp_commit();
        }

        for (int m = threadIdx.x; m < BM; m += blockDim.x) {
            int g = m0 + m;
            sL[m] = g < S ? Lb[g] : 0.0f;
            sD[m] = g < S ? Db[g] : 0.0f;
        }

        // S = Q@K^T ; dP = dO@V^T
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
                wmma::load_matrix_sync(a, Qcur + mt * 16 * Dh + kt * 16, Dh);
                wmma::load_matrix_sync(b, sK + kt * 16 + nt * 16 * Dh, Dh);
                wmma::mma_sync(accS, a, b, accS);
                FragARow a2; FragBCol b2;
                wmma::load_matrix_sync(a2, dOcur + mt * 16 * Dh + kt * 16, Dh);
                wmma::load_matrix_sync(b2, sV + kt * 16 + nt * 16 * Dh, Dh);
                wmma::mma_sync(accP, a2, b2, accP);
            }
            wmma::store_matrix_sync(sSc + mt * 16 * BN + nt * 16, accS, BN, wmma::mem_row_major);
            wmma::store_matrix_sync(sdP + mt * 16 * BN + nt * 16, accP, BN, wmma::mem_row_major);
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < BM * BN; idx += blockDim.x) {
            int m = idx / BN;
            int n = idx % BN;
            int grow = m0 + m, gcol = n0 + n;
            float P = 0.0f, dS = 0.0f;
            if (grow < S && gcol < S) {
                P = __expf(scale * sSc[idx] - sL[m]);
                dS = scale * P * (sdP[idx] - sD[m]);
            }
            sP[idx]  = __float2bfloat16(P);
            sdS[idx] = __float2bfloat16(dS);
        }
        __syncthreads();

        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            #pragma unroll
            for (int kt = 0; kt < BM / 16; kt++) {
                FragACol a; FragBRow b;
                wmma::load_matrix_sync(a, sP + nt * 16 + kt * 16 * BN, BN);
                wmma::load_matrix_sync(b, dOcur + kt * 16 * Dh + warp * 16, Dh);
                wmma::mma_sync(accV[nt], a, b, accV[nt]);
                FragACol a2; FragBRow b2;
                wmma::load_matrix_sync(a2, sdS + nt * 16 + kt * 16 * BN, BN);
                wmma::load_matrix_sync(b2, Qcur + kt * 16 * Dh + warp * 16, Dh);
                wmma::mma_sync(accK[nt], a2, b2, accK[nt]);
            }
        }
    }

    __syncthreads();
    float* stage = sSc + warp * 256;
    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        wmma::store_matrix_sync(stage, accV[nt], 16, wmma::mem_row_major);
        __syncwarp();
        for (int j = lane; j < 256; j += 32) {
            int r = j >> 4, c = j & 15;
            int gn = n0 + nt * 16 + r;
            int ge = warp * 16 + c;
            if (gn < S) dV[mat_off + (int64_t)gn * Dh + ge] = __float2bfloat16(stage[j]);
        }
        __syncwarp();
    }
    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        wmma::store_matrix_sync(stage, accK[nt], 16, wmma::mem_row_major);
        __syncwarp();
        for (int j = lane; j < 256; j += 32) {
            int r = j >> 4, c = j & 15;
            int gn = n0 + nt * 16 + r;
            int ge = warp * 16 + c;
            if (gn < S) dK[mat_off + (int64_t)gn * Dh + ge] = __float2bfloat16(stage[j]);
        }
        __syncwarp();
    }
}

// ---------------- Pass 2: dQ ----------------
__global__ __launch_bounds__(NTHREAD)
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
    __nv_bfloat16* sQ  = (__nv_bfloat16*)p; p += BM * Dh * 2;
    __nv_bfloat16* sdO = (__nv_bfloat16*)p; p += BM * Dh * 2;
    __nv_bfloat16* sK  = (__nv_bfloat16*)p; p += 2 * BN * Dh * 2;
    __nv_bfloat16* sV  = (__nv_bfloat16*)p; p += 2 * BN * Dh * 2;
    float* sSc = (float*)p; p += BM * BN * 4;
    float* sdP = (float*)p; p += BM * BN * 4;
    __nv_bfloat16* sdS = (__nv_bfloat16*)p; p += BM * BN * 2;
    float* sL = (float*)p; p += BM * 4;
    float* sD = (float*)p; p += BM * 4;

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    AccF accQ[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) wmma::fill_fragment(accQ[i], 0.0f);

    for (int m = threadIdx.x; m < BM; m += blockDim.x) {
        int g = m0 + m;
        sL[m] = g < S ? Lb[g] : 0.0f;
        sD[m] = g < S ? Db[g] : 0.0f;
    }

    int num_kv = (S + BN - 1) / BN;

    async_load_tile(Qb, m0, BM, S, sQ);
    async_load_tile(dOb, m0, BM, S, sdO);
    async_load_tile(Kb, 0, BN, S, sK);
    async_load_tile(Vb, 0, BN, S, sV);
    cp_commit();

    for (int jb = 0; jb < num_kv; jb++) {
        cp_wait0();
        __syncthreads();
        int buf = jb & 1;
        int n0 = jb * BN;
        __nv_bfloat16* Kcur = sK + buf * BN * Dh;
        __nv_bfloat16* Vcur = sV + buf * BN * Dh;

        if (jb + 1 < num_kv) {
            int nxt = (jb + 1) & 1;
            async_load_tile(Kb, (jb + 1) * BN, BN, S, sK + nxt * BN * Dh);
            async_load_tile(Vb, (jb + 1) * BN, BN, S, sV + nxt * BN * Dh);
            cp_commit();
        }

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
                wmma::load_matrix_sync(a, sQ + mt * 16 * Dh + kt * 16, Dh);
                wmma::load_matrix_sync(b, Kcur + kt * 16 + nt * 16 * Dh, Dh);
                wmma::mma_sync(accS, a, b, accS);
                FragARow a2; FragBCol b2;
                wmma::load_matrix_sync(a2, sdO + mt * 16 * Dh + kt * 16, Dh);
                wmma::load_matrix_sync(b2, Vcur + kt * 16 + nt * 16 * Dh, Dh);
                wmma::mma_sync(accP, a2, b2, accP);
            }
            wmma::store_matrix_sync(sSc + mt * 16 * BN + nt * 16, accS, BN, wmma::mem_row_major);
            wmma::store_matrix_sync(sdP + mt * 16 * BN + nt * 16, accP, BN, wmma::mem_row_major);
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < BM * BN; idx += blockDim.x) {
            int m = idx / BN;
            int n = idx % BN;
            int grow = m0 + m, gcol = n0 + n;
            float dS = 0.0f;
            if (grow < S && gcol < S) {
                float P = __expf(scale * sSc[idx] - sL[m]);
                dS = scale * P * (sdP[idx] - sD[m]);
            }
            sdS[idx] = __float2bfloat16(dS);
        }
        __syncthreads();

        #pragma unroll
        for (int mt = 0; mt < 4; mt++) {
            #pragma unroll
            for (int kt = 0; kt < BN / 16; kt++) {
                FragARow a; FragBRow b;
                wmma::load_matrix_sync(a, sdS + mt * 16 * BN + kt * 16, BN);
                wmma::load_matrix_sync(b, Kcur + kt * 16 * Dh + warp * 16, Dh);
                wmma::mma_sync(accQ[mt], a, b, accQ[mt]);
            }
        }
    }

    __syncthreads();
    float* stage = sSc + warp * 256;
    #pragma unroll
    for (int mt = 0; mt < 4; mt++) {
        wmma::store_matrix_sync(stage, accQ[mt], 16, wmma::mem_row_major);
        __syncwarp();
        for (int j = lane; j < 256; j += 32) {
            int r = j >> 4, c = j & 15;
            int gm = m0 + mt * 16 + r;
            int ge = warp * 16 + c;
            if (gm < S) dQ[mat_off + (int64_t)gm * Dh + ge] = __float2bfloat16(stage[j]);
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

    size_t smem_dkv = (size_t)(BN*Dh + BN*Dh + 2*BM*Dh + 2*BM*Dh)*2
                    + (size_t)(BM*BN + BM*BN)*4
                    + (size_t)(BM*BN + BM*BN)*2
                    + (size_t)(BM + BM)*4;
    size_t smem_dq  = (size_t)(BM*Dh + BM*Dh + 2*BN*Dh + 2*BN*Dh)*2
                    + (size_t)(BM*BN + BM*BN)*4
                    + (size_t)(BM*BN)*2
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