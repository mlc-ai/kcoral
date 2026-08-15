#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
#include <math.h>
#include <stdio.h>
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
using ARowFrag = wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major>;
using AColFrag = wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::col_major>;
using BRowFrag = wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major>;
using BColFrag = wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major>;
using AccFrag  = wmma::fragment<wmma::accumulator,16,16,16,float>;

#define D_DIM 128
#define BM 64
#define BN 64

// SMEM layout sizes (bytes)
// Ks,Vs,Qs,dOs : 4 * 64*128*2 = 65536
// Ss(float)    : 64*64*4 = 16384
// Ps(bf16)     : 64*64*2 = 8192
// dSs(bf16)    : 64*64*2 = 8192
// Ls,Ds(float) : 64*4 + 64*4 = 512
static constexpr int SMEM_BYTES = 65536 + 16384 + 8192 + 8192 + 512;

// ----------------- precompute D = sum_d O*dO -----------------
__global__ void compute_D_kernel(const __nv_bfloat16* __restrict__ Og,
                                 const __nv_bfloat16* __restrict__ dOg,
                                 float* __restrict__ Dg, long long rows) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= rows) return;
    const __nv_bfloat16* o = Og + (size_t)warp * D_DIM;
    const __nv_bfloat16* g = dOg + (size_t)warp * D_DIM;
    float acc = 0.f;
    #pragma unroll
    for (int k = lane; k < D_DIM; k += 32) {
        acc += (float)o[k] * (float)g[k];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) Dg[warp] = acc;
}

__device__ inline void load_tile(__nv_bfloat16* dst, const __nv_bfloat16* g,
                                 int bh, int S, int row_start, int tid) {
    // 64 rows x 128 cols, vectorized as int4 (8 bf16)
    #pragma unroll
    for (int v = tid; v < BN * 16; v += 128) {
        int row = v >> 4;           // /16
        int vc  = v & 15;           // %16
        int col = vc * 8;
        int gr = row_start + row;
        int4 val;
        if (gr < S) val = *reinterpret_cast<const int4*>(&g[((size_t)bh * S + gr) * D_DIM + col]);
        else { val.x = 0; val.y = 0; val.z = 0; val.w = 0; }
        *reinterpret_cast<int4*>(&dst[row * D_DIM + col]) = val;
    }
}

// C[M=64][N=64] = A[64x128] @ B[64x128]^T  ; out row-major ld=64
__device__ inline void mm_AKt(const __nv_bfloat16* A, const __nv_bfloat16* B,
                              float* out, int r0) {
    AccFrag acc[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) wmma::fill_fragment(acc[j], 0.f);
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        ARowFrag af;
        wmma::load_matrix_sync(af, A + r0 * D_DIM + k * 16, D_DIM);
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            BColFrag bf;
            wmma::load_matrix_sync(bf, B + (nt * 16) * D_DIM + k * 16, D_DIM);
            wmma::mma_sync(acc[nt], af, bf, acc[nt]);
        }
    }
    #pragma unroll
    for (int nt = 0; nt < 4; nt++)
        wmma::store_matrix_sync(out + r0 * BN + nt * 16, acc[nt], BN, wmma::mem_row_major);
}

// ----------------- Pass 1: dK, dV -----------------
__global__ __launch_bounds__(128)
void bwd_dkdv_kernel(const __nv_bfloat16* __restrict__ Qg,
                     const __nv_bfloat16* __restrict__ Kg,
                     const __nv_bfloat16* __restrict__ Vg,
                     const __nv_bfloat16* __restrict__ dOg,
                     const float* __restrict__ Lg,
                     const float* __restrict__ Dg,
                     __nv_bfloat16* __restrict__ dKg,
                     __nv_bfloat16* __restrict__ dVg,
                     int S, float scale) {
    extern __shared__ char smem_raw[];
    __nv_bfloat16* Ks  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* Vs  = Ks + BN * D_DIM;
    __nv_bfloat16* Qs  = Vs + BN * D_DIM;
    __nv_bfloat16* dOs = Qs + BM * D_DIM;
    float* Ss          = reinterpret_cast<float*>(dOs + BM * D_DIM);
    __nv_bfloat16* Ps  = reinterpret_cast<__nv_bfloat16*>(Ss + BM * BN);
    __nv_bfloat16* dSs = Ps + BM * BN;
    float* Ls          = reinterpret_cast<float*>(dSs + BM * BN);
    float* Ds          = Ls + BM;
    float* obuf        = reinterpret_cast<float*>(Qs); // reuse Qs+dOs (32KB) at end

    int bh  = blockIdx.y;
    int kvb = blockIdx.x;
    int kv_start = kvb * BN;
    int tid = threadIdx.x;
    int warpId = tid >> 5;
    int r0 = warpId * 16;
    int num_q = (S + BM - 1) / BM;

    load_tile(Ks, Kg, bh, S, kv_start, tid);
    load_tile(Vs, Vg, bh, S, kv_start, tid);

    AccFrag c_dV[8], c_dK[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) { wmma::fill_fragment(c_dV[i], 0.f); wmma::fill_fragment(c_dK[i], 0.f); }

    for (int qt = kvb; qt < num_q; qt++) {
        int q_start = qt * BM;
        __syncthreads();
        load_tile(Qs, Qg, bh, S, q_start, tid);
        load_tile(dOs, dOg, bh, S, q_start, tid);
        if (tid < BM) {
            int qi = q_start + tid;
            Ls[tid] = (qi < S) ? Lg[(size_t)bh * S + qi] : 0.f;
            Ds[tid] = (qi < S) ? Dg[(size_t)bh * S + qi] : 0.f;
        }
        __syncthreads();

        // S = Q @ K^T
        mm_AKt(Qs, Ks, Ss, r0);
        __syncthreads();

        // P = exp(S*scale - L) with causal mask
        for (int idx = tid; idx < BM * BN; idx += 128) {
            int m = idx >> 6;
            int n = idx & 63;
            int qi = q_start + m, kj = kv_start + n;
            float p = 0.f;
            if (qi < S && kj < S && kj <= qi)
                p = __expf(Ss[idx] * scale - Ls[m]);
            Ps[idx] = __float2bfloat16(p);
        }
        __syncthreads();

        // dP = dO @ V^T
        mm_AKt(dOs, Vs, Ss, r0);
        __syncthreads();

        // dS = P * (dP - D)
        for (int idx = tid; idx < BM * BN; idx += 128) {
            int m = idx >> 6;
            float dp = Ss[idx];
            float p  = (float)Ps[idx];
            float ds = p * (dp - Ds[m]);
            dSs[idx] = __float2bfloat16(ds);
        }
        __syncthreads();

        // dV += P^T @ dO   (output rows = key index n = r0..)
        {
            int n0 = r0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                AColFrag af;
                wmma::load_matrix_sync(af, Ps + (k * 16) * BN + n0, BN);
                #pragma unroll
                for (int nt = 0; nt < 8; nt++) {
                    BRowFrag bf;
                    wmma::load_matrix_sync(bf, dOs + (k * 16) * D_DIM + nt * 16, D_DIM);
                    wmma::mma_sync(c_dV[nt], af, bf, c_dV[nt]);
                }
            }
        }
        // dK += dS^T @ Q
        {
            int n0 = r0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                AColFrag af;
                wmma::load_matrix_sync(af, dSs + (k * 16) * BN + n0, BN);
                #pragma unroll
                for (int nt = 0; nt < 8; nt++) {
                    BRowFrag bf;
                    wmma::load_matrix_sync(bf, Qs + (k * 16) * D_DIM + nt * 16, D_DIM);
                    wmma::mma_sync(c_dK[nt], af, bf, c_dK[nt]);
                }
            }
        }
    }

    __syncthreads();
    // write dV (no scale)
    #pragma unroll
    for (int nt = 0; nt < 8; nt++)
        wmma::store_matrix_sync(obuf + r0 * D_DIM + nt * 16, c_dV[nt], D_DIM, wmma::mem_row_major);
    __syncthreads();
    for (int v = tid; v < BN * 16; v += 128) {
        int row = v >> 4, vc = v & 15, col = vc * 8;
        int gr = kv_start + row;
        if (gr < S) {
            int4 packed;
            __nv_bfloat16* hp = reinterpret_cast<__nv_bfloat16*>(&packed);
            float* op = &obuf[row * D_DIM + col];
            #pragma unroll
            for (int e = 0; e < 8; e++) hp[e] = __float2bfloat16(op[e]);
            *reinterpret_cast<int4*>(&dVg[((size_t)bh * S + gr) * D_DIM + col]) = packed;
        }
    }
    __syncthreads();
    // write dK (scaled)
    #pragma unroll
    for (int nt = 0; nt < 8; nt++)
        wmma::store_matrix_sync(obuf + r0 * D_DIM + nt * 16, c_dK[nt], D_DIM, wmma::mem_row_major);
    __syncthreads();
    for (int v = tid; v < BN * 16; v += 128) {
        int row = v >> 4, vc = v & 15, col = vc * 8;
        int gr = kv_start + row;
        if (gr < S) {
            int4 packed;
            __nv_bfloat16* hp = reinterpret_cast<__nv_bfloat16*>(&packed);
            float* op = &obuf[row * D_DIM + col];
            #pragma unroll
            for (int e = 0; e < 8; e++) hp[e] = __float2bfloat16(op[e] * scale);
            *reinterpret_cast<int4*>(&dKg[((size_t)bh * S + gr) * D_DIM + col]) = packed;
        }
    }
}

// ----------------- Pass 2: dQ -----------------
__global__ __launch_bounds__(128)
void bwd_dq_kernel(const __nv_bfloat16* __restrict__ Qg,
                   const __nv_bfloat16* __restrict__ Kg,
                   const __nv_bfloat16* __restrict__ Vg,
                   const __nv_bfloat16* __restrict__ dOg,
                   const float* __restrict__ Lg,
                   const float* __restrict__ Dg,
                   __nv_bfloat16* __restrict__ dQg,
                   int S, float scale) {
    extern __shared__ char smem_raw[];
    __nv_bfloat16* Ks  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* Vs  = Ks + BN * D_DIM;
    __nv_bfloat16* Qs  = Vs + BN * D_DIM;
    __nv_bfloat16* dOs = Qs + BM * D_DIM;
    float* Ss          = reinterpret_cast<float*>(dOs + BM * D_DIM);
    __nv_bfloat16* Ps  = reinterpret_cast<__nv_bfloat16*>(Ss + BM * BN);
    __nv_bfloat16* dSs = Ps + BM * BN;
    float* Ls          = reinterpret_cast<float*>(dSs + BM * BN);
    float* Ds          = Ls + BM;
    float* obuf        = reinterpret_cast<float*>(Ks); // reuse Ks+Vs (32KB) at end

    int bh = blockIdx.y;
    int qb = blockIdx.x;
    int q_start = qb * BM;
    int tid = threadIdx.x;
    int warpId = tid >> 5;
    int r0 = warpId * 16;

    load_tile(Qs, Qg, bh, S, q_start, tid);
    load_tile(dOs, dOg, bh, S, q_start, tid);
    if (tid < BM) {
        int qi = q_start + tid;
        Ls[tid] = (qi < S) ? Lg[(size_t)bh * S + qi] : 0.f;
        Ds[tid] = (qi < S) ? Dg[(size_t)bh * S + qi] : 0.f;
    }

    AccFrag c_dQ[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) wmma::fill_fragment(c_dQ[i], 0.f);

    for (int j = 0; j <= qb; j++) {
        int kv_start = j * BN;
        __syncthreads();
        load_tile(Ks, Kg, bh, S, kv_start, tid);
        load_tile(Vs, Vg, bh, S, kv_start, tid);
        __syncthreads();

        // S = Q @ K^T
        mm_AKt(Qs, Ks, Ss, r0);
        __syncthreads();

        // P
        for (int idx = tid; idx < BM * BN; idx += 128) {
            int m = idx >> 6;
            int n = idx & 63;
            int qi = q_start + m, kj = kv_start + n;
            float p = 0.f;
            if (qi < S && kj < S && kj <= qi)
                p = __expf(Ss[idx] * scale - Ls[m]);
            Ps[idx] = __float2bfloat16(p);
        }
        __syncthreads();

        // dP = dO @ V^T
        mm_AKt(dOs, Vs, Ss, r0);
        __syncthreads();

        // dS = P*(dP - D)
        for (int idx = tid; idx < BM * BN; idx += 128) {
            int m = idx >> 6;
            float dp = Ss[idx];
            float p  = (float)Ps[idx];
            dSs[idx] = __float2bfloat16(p * (dp - Ds[m]));
        }
        __syncthreads();

        // dQ += dS @ K   (output rows = query m = r0..)
        {
            int m0 = r0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                ARowFrag af;
                wmma::load_matrix_sync(af, dSs + m0 * BN + k * 16, BN);
                #pragma unroll
                for (int nt = 0; nt < 8; nt++) {
                    BRowFrag bf;
                    wmma::load_matrix_sync(bf, Ks + (k * 16) * D_DIM + nt * 16, D_DIM);
                    wmma::mma_sync(c_dQ[nt], af, bf, c_dQ[nt]);
                }
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for (int nt = 0; nt < 8; nt++)
        wmma::store_matrix_sync(obuf + r0 * D_DIM + nt * 16, c_dQ[nt], D_DIM, wmma::mem_row_major);
    __syncthreads();
    for (int v = tid; v < BM * 16; v += 128) {
        int row = v >> 4, vc = v & 15, col = vc * 8;
        int gr = q_start + row;
        if (gr < S) {
            int4 packed;
            __nv_bfloat16* hp = reinterpret_cast<__nv_bfloat16*>(&packed);
            float* op = &obuf[row * D_DIM + col];
            #pragma unroll
            for (int e = 0; e < 8; e++) hp[e] = __float2bfloat16(op[e] * scale);
            *reinterpret_cast<int4*>(&dQg[((size_t)bh * S + gr) * D_DIM + col]) = packed;
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
    int d = (int)Q.size(3);
    int BH = B * H;
    if (S == 0) return;
    float scale = 1.0f / sqrtf((float)d);

    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* Op = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dOp = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* Lp = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQp = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dKp = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dVp = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* Dbuf = nullptr;
    CUDA_CHECK(cudaMallocAsync((void**)&Dbuf, sizeof(float) * (size_t)BH * S, stream));

    long long rows = (long long)BH * S;
    int blkD = 256; // 8 warps -> 8 rows per block
    long long gridD = (rows + 7) / 8;
    compute_D_kernel<<<(unsigned)gridD, blkD, 0, stream>>>(Op, dOp, Dbuf, rows);

    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute(bwd_dkdv_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
        CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
        attr_set = true;
    }

    int num_kv = (S + BN - 1) / BN;
    int num_q  = (S + BM - 1) / BM;

    dim3 grid1((unsigned)num_kv, (unsigned)BH, 1);
    dim3 grid2((unsigned)num_q, (unsigned)BH, 1);
    dim3 block(128, 1, 1);

    bwd_dkdv_kernel<<<grid1, block, SMEM_BYTES, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Dbuf, dKp, dVp, S, scale);

    bwd_dq_kernel<<<grid2, block, SMEM_BYTES, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Dbuf, dQp, S, scale);

    CUDA_CHECK(cudaFreeAsync(Dbuf, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd