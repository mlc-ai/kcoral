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
using BRowFrag = wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major>;
using BColFrag = wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major>;
using AccFrag  = wmma::fragment<wmma::accumulator,16,16,16,float>;

#define D_DIM 128
#define BM 64
#define BN 64

// ---------- precompute D = sum_d O*dO ----------
__global__ void compute_D_kernel(const __nv_bfloat16* __restrict__ Og,
                                 const __nv_bfloat16* __restrict__ dOg,
                                 float* __restrict__ Dg, long long rows) {
    long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= rows) return;
    const __nv_bfloat16* o = Og + (size_t)warp * D_DIM;
    const __nv_bfloat16* g = dOg + (size_t)warp * D_DIM;
    float acc = 0.f;
    #pragma unroll
    for (int k = lane; k < D_DIM; k += 32) acc += (float)o[k] * (float)g[k];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) Dg[warp] = acc;
}

__device__ inline void load_tile(__nv_bfloat16* dst, const __nv_bfloat16* g,
                                 int bh, int S, int row_start, int tid) {
    #pragma unroll
    for (int v = tid; v < 64 * 16; v += 128) {
        int row = v >> 4, vc = v & 15, col = vc * 8;
        int gr = row_start + row;
        int4 val;
        if (gr < S) val = *reinterpret_cast<const int4*>(&g[((size_t)bh * S + gr) * D_DIM + col]);
        else { val.x = val.y = val.z = val.w = 0; }
        *reinterpret_cast<int4*>(&dst[row * D_DIM + col]) = val;
    }
}

// C[64x64] = A[64x128] @ B[64x128]^T  (row-major out, ld=64)
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

__device__ inline void write_out(__nv_bfloat16* G, float* obuf, int bh, int S,
                                  int row_start, int tid, float mul) {
    for (int v = tid; v < 64 * 16; v += 128) {
        int row = v >> 4, vc = v & 15, col = vc * 8;
        int gr = row_start + row;
        if (gr < S) {
            int4 packed;
            __nv_bfloat16* hp = reinterpret_cast<__nv_bfloat16*>(&packed);
            float* op = &obuf[row * D_DIM + col];
            #pragma unroll
            for (int e = 0; e < 8; e++) hp[e] = __float2bfloat16(op[e] * mul);
            *reinterpret_cast<int4*>(&G[((size_t)bh * S + gr) * D_DIM + col]) = packed;
        }
    }
}

// ---------------- dV ----------------
__global__ __launch_bounds__(128)
void bwd_dv_kernel(const __nv_bfloat16* __restrict__ Qg,
                   const __nv_bfloat16* __restrict__ Kg,
                   const __nv_bfloat16* __restrict__ dOg,
                   const float* __restrict__ Lg,
                   __nv_bfloat16* __restrict__ dVg,
                   int S, float scale) {
    extern __shared__ char smem[];
    __nv_bfloat16* Ks  = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* Qs  = Ks + BN * D_DIM;
    __nv_bfloat16* dOs = Qs + BM * D_DIM;
    float* Ss          = reinterpret_cast<float*>(dOs + BM * D_DIM);
    __nv_bfloat16* Pt  = reinterpret_cast<__nv_bfloat16*>(Ss + BM * BN);
    float* Ls          = reinterpret_cast<float*>(Pt + BM * BN);
    float* obuf        = reinterpret_cast<float*>(Qs);

    int bh = blockIdx.y, kvb = blockIdx.x;
    int kv_start = kvb * BN, tid = threadIdx.x;
    int r0 = (tid >> 5) * 16;
    int num_q = (S + BM - 1) / BM;

    load_tile(Ks, Kg, bh, S, kv_start, tid);

    AccFrag c_dV[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) wmma::fill_fragment(c_dV[i], 0.f);

    for (int qt = kvb; qt < num_q; qt++) {
        int q_start = qt * BM;
        __syncthreads();
        load_tile(Qs, Qg, bh, S, q_start, tid);
        load_tile(dOs, dOg, bh, S, q_start, tid);
        if (tid < BM) { int qi = q_start + tid; Ls[tid] = (qi < S) ? Lg[(size_t)bh * S + qi] : 0.f; }
        __syncthreads();
        mm_AKt(Qs, Ks, Ss, r0);
        __syncthreads();
        for (int idx = tid; idx < BM * BN; idx += 128) {
            int m = idx >> 6, n = idx & 63;
            int qi = q_start + m, kj = kv_start + n;
            float p = 0.f;
            if (qi < S && kj < S && kj <= qi) p = __expf(Ss[idx] * scale - Ls[m]);
            Pt[n * BM + m] = __float2bfloat16(p);
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            ARowFrag aV;
            wmma::load_matrix_sync(aV, Pt + r0 * BM + k * 16, BM);
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                BRowFrag bO;
                wmma::load_matrix_sync(bO, dOs + (k * 16) * D_DIM + nt * 16, D_DIM);
                wmma::mma_sync(c_dV[nt], aV, bO, c_dV[nt]);
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for (int nt = 0; nt < 8; nt++)
        wmma::store_matrix_sync(obuf + r0 * D_DIM + nt * 16, c_dV[nt], D_DIM, wmma::mem_row_major);
    __syncthreads();
    write_out(dVg, obuf, bh, S, kv_start, tid, 1.0f);
}

// ---------------- dK ----------------
__global__ __launch_bounds__(128)
void bwd_dk_kernel(const __nv_bfloat16* __restrict__ Qg,
                   const __nv_bfloat16* __restrict__ Kg,
                   const __nv_bfloat16* __restrict__ Vg,
                   const __nv_bfloat16* __restrict__ dOg,
                   const float* __restrict__ Lg,
                   const float* __restrict__ Dg,
                   __nv_bfloat16* __restrict__ dKg,
                   int S, float scale) {
    extern __shared__ char smem[];
    __nv_bfloat16* Ks  = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* Vs  = Ks + BN * D_DIM;
    __nv_bfloat16* Qs  = Vs + BN * D_DIM;
    __nv_bfloat16* dOs = Qs + BM * D_DIM;
    float* Ss          = reinterpret_cast<float*>(dOs + BM * D_DIM);
    float* dPs         = Ss + BM * BN;
    __nv_bfloat16* dSt = reinterpret_cast<__nv_bfloat16*>(dPs + BM * BN);
    float* Ls          = reinterpret_cast<float*>(dSt + BM * BN);
    float* Ds          = Ls + BM;
    float* obuf        = reinterpret_cast<float*>(Ks);

    int bh = blockIdx.y, kvb = blockIdx.x;
    int kv_start = kvb * BN, tid = threadIdx.x;
    int r0 = (tid >> 5) * 16;
    int num_q = (S + BM - 1) / BM;

    load_tile(Ks, Kg, bh, S, kv_start, tid);
    load_tile(Vs, Vg, bh, S, kv_start, tid);

    AccFrag c_dK[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) wmma::fill_fragment(c_dK[i], 0.f);

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
        mm_AKt(Qs, Ks, Ss, r0);
        mm_AKt(dOs, Vs, dPs, r0);
        __syncthreads();
        for (int idx = tid; idx < BM * BN; idx += 128) {
            int m = idx >> 6, n = idx & 63;
            int qi = q_start + m, kj = kv_start + n;
            float p = 0.f;
            if (qi < S && kj < S && kj <= qi) p = __expf(Ss[idx] * scale - Ls[m]);
            float ds = p * (dPs[idx] - Ds[m]);
            dSt[n * BM + m] = __float2bfloat16(ds);
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            ARowFrag aK;
            wmma::load_matrix_sync(aK, dSt + r0 * BM + k * 16, BM);
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                BRowFrag bQ;
                wmma::load_matrix_sync(bQ, Qs + (k * 16) * D_DIM + nt * 16, D_DIM);
                wmma::mma_sync(c_dK[nt], aK, bQ, c_dK[nt]);
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for (int nt = 0; nt < 8; nt++)
        wmma::store_matrix_sync(obuf + r0 * D_DIM + nt * 16, c_dK[nt], D_DIM, wmma::mem_row_major);
    __syncthreads();
    write_out(dKg, obuf, bh, S, kv_start, tid, scale);
}

// ---------------- dQ ----------------
__global__ __launch_bounds__(128)
void bwd_dq_kernel(const __nv_bfloat16* __restrict__ Qg,
                   const __nv_bfloat16* __restrict__ Kg,
                   const __nv_bfloat16* __restrict__ Vg,
                   const __nv_bfloat16* __restrict__ dOg,
                   const float* __restrict__ Lg,
                   const float* __restrict__ Dg,
                   __nv_bfloat16* __restrict__ dQg,
                   int S, float scale) {
    extern __shared__ char smem[];
    __nv_bfloat16* Ks  = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* Vs  = Ks + BN * D_DIM;
    __nv_bfloat16* Qs  = Vs + BN * D_DIM;
    __nv_bfloat16* dOs = Qs + BM * D_DIM;
    float* Ss          = reinterpret_cast<float*>(dOs + BM * D_DIM);
    float* dPs         = Ss + BM * BN;
    __nv_bfloat16* dSs = reinterpret_cast<__nv_bfloat16*>(dPs + BM * BN);
    float* Ls          = reinterpret_cast<float*>(dSs + BM * BN);
    float* Ds          = Ls + BM;
    float* obuf        = reinterpret_cast<float*>(Ks);

    int bh = blockIdx.y, qb = blockIdx.x;
    int q_start = qb * BM, tid = threadIdx.x;
    int r0 = (tid >> 5) * 16;

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
        mm_AKt(Qs, Ks, Ss, r0);
        mm_AKt(dOs, Vs, dPs, r0);
        __syncthreads();
        for (int idx = tid; idx < BM * BN; idx += 128) {
            int m = idx >> 6, n = idx & 63;
            int qi = q_start + m, kj = kv_start + n;
            float p = 0.f;
            if (qi < S && kj < S && kj <= qi) p = __expf(Ss[idx] * scale - Ls[m]);
            dSs[idx] = __float2bfloat16(p * (dPs[idx] - Ds[m]));
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            ARowFrag af;
            wmma::load_matrix_sync(af, dSs + r0 * BN + k * 16, BN);
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                BRowFrag bK;
                wmma::load_matrix_sync(bK, Ks + (k * 16) * D_DIM + nt * 16, D_DIM);
                wmma::mma_sync(c_dQ[nt], af, bK, c_dQ[nt]);
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for (int nt = 0; nt < 8; nt++)
        wmma::store_matrix_sync(obuf + r0 * D_DIM + nt * 16, c_dQ[nt], D_DIM, wmma::mem_row_major);
    __syncthreads();
    write_out(dQg, obuf, bh, S, q_start, tid, scale);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = (int)Q.size(0), H = (int)Q.size(1), S = (int)Q.size(2), d = (int)Q.size(3);
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
    compute_D_kernel<<<(unsigned)((rows + 7) / 8), 256, 0, stream>>>(Op, dOp, Dbuf, rows);

    int SMEM_DV = (int)(BN*D_DIM*2 + 2*BM*D_DIM*2 + BM*BN*4 + BM*BN*2 + BM*4);
    int SMEM_DK = (int)(2*BN*D_DIM*2 + 2*BM*D_DIM*2 + 2*BM*BN*4 + BM*BN*2 + 2*BM*4);
    int SMEM_DQ = SMEM_DK;

    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute(bwd_dv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_DV));
        CUDA_CHECK(cudaFuncSetAttribute(bwd_dk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_DK));
        CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_DQ));
        attr_set = true;
    }

    int num_kv = (S + BN - 1) / BN;
    int num_q  = (S + BM - 1) / BM;
    dim3 block(128, 1, 1);

    bwd_dv_kernel<<<dim3((unsigned)num_kv, (unsigned)BH, 1), block, SMEM_DV, stream>>>(
        Qp, Kp, dOp, Lp, dVp, S, scale);
    bwd_dk_kernel<<<dim3((unsigned)num_kv, (unsigned)BH, 1), block, SMEM_DK, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Dbuf, dKp, S, scale);
    bwd_dq_kernel<<<dim3((unsigned)num_q, (unsigned)BH, 1), block, SMEM_DQ, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Dbuf, dQp, S, scale);

    CUDA_CHECK(cudaFreeAsync(Dbuf, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd