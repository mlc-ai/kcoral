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

// ---------- D_i = sum_d O*dO ----------
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
    for (int v = tid; v < 64 * 16; v += 256) {
        int row = v >> 4, vc = v & 15, col = vc * 8;
        int gr = row_start + row;
        int4 val;
        if (gr < S) val = *reinterpret_cast<const int4*>(&g[((size_t)bh * S + gr) * D_DIM + col]);
        else { val.x = val.y = val.z = val.w = 0; }
        *reinterpret_cast<int4*>(&dst[row * D_DIM + col]) = val;
    }
}

// out[r0..r0+15, 0..63] = A[..,128] @ B[64,128]^T  (per-warp; bf16 in, fp32 acc)
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

// c[4] += A[outRows x 64] @ B[64 x 128] ; split-bf16 A. g=row group, h=d-half
__device__ inline void out_mma_split(AccFrag* c, const __nv_bfloat16* Ah,
    const __nv_bfloat16* Al, const __nv_bfloat16* B, int g, int h) {
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        ARowFrag ah, al;
        wmma::load_matrix_sync(ah, Ah + (g * 16) * 64 + k * 16, 64);
        wmma::load_matrix_sync(al, Al + (g * 16) * 64 + k * 16, 64);
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            BRowFrag b;
            wmma::load_matrix_sync(b, B + (k * 16) * D_DIM + h * 64 + nt * 16, D_DIM);
            wmma::mma_sync(c[nt], ah, b, c[nt]);
            wmma::mma_sync(c[nt], al, b, c[nt]);
        }
    }
}

__device__ inline void store_frags(float* obuf, AccFrag* c, int g, int h) {
    #pragma unroll
    for (int nt = 0; nt < 4; nt++)
        wmma::store_matrix_sync(obuf + (g * 16) * D_DIM + h * 64 + nt * 16, c[nt], D_DIM, wmma::mem_row_major);
}

__device__ inline void write_out(__nv_bfloat16* G, float* obuf, int bh, int S,
                                  int row_start, int tid, float mul) {
    for (int v = tid; v < 64 * 16; v += 256) {
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
__global__ __launch_bounds__(256)
void bwd_dv_kernel(const __nv_bfloat16* __restrict__ Qg,
                   const __nv_bfloat16* __restrict__ Kg,
                   const __nv_bfloat16* __restrict__ dOg,
                   const float* __restrict__ Lg,
                   __nv_bfloat16* __restrict__ dVg,
                   int S, float scale) {
    extern __shared__ char smem[];
    __nv_bfloat16* Ks  = reinterpret_cast<__nv_bfloat16*>(smem);   // 16K
    __nv_bfloat16* Qs  = Ks + BN * D_DIM;                          // 16K
    __nv_bfloat16* dOs = Qs + BM * D_DIM;                          // 16K
    float* Ss          = reinterpret_cast<float*>(dOs + BM * D_DIM); // 16K
    __nv_bfloat16* Ph  = reinterpret_cast<__nv_bfloat16*>(Ss + BM * BN); // 8K
    __nv_bfloat16* Pl  = Ph + BM * BN;                             // 8K
    float* Ls          = reinterpret_cast<float*>(Pl + BM * BN);
    float* obuf        = reinterpret_cast<float*>(Ks);             // overlay Ks+Qs (32K)

    int bh = blockIdx.y, kvb = blockIdx.x;
    int kv_start = kvb * BN, tid = threadIdx.x;
    int w = tid >> 5, g = w & 3, h = w >> 2;
    int num_q = (S + BM - 1) / BM;

    load_tile(Ks, Kg, bh, S, kv_start, tid);

    AccFrag c[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) wmma::fill_fragment(c[i], 0.f);

    for (int qt = kvb; qt < num_q; qt++) {
        int q_start = qt * BM;
        __syncthreads();
        load_tile(Qs, Qg, bh, S, q_start, tid);
        load_tile(dOs, dOg, bh, S, q_start, tid);
        if (tid < BM) { int qi = q_start + tid; Ls[tid] = (qi < S) ? Lg[(size_t)bh * S + qi] : 0.f; }
        __syncthreads();
        if (w < 4) mm_AKt(Qs, Ks, Ss, w * 16);
        __syncthreads();
        for (int idx = tid; idx < BM * BN; idx += 256) {
            int m = idx >> 6, n = idx & 63;
            int qi = q_start + m, kj = kv_start + n;
            float p = 0.f;
            if (qi < S && kj < S && kj <= qi) p = __expf(Ss[idx] * scale - Ls[m]);
            __nv_bfloat16 ph = __float2bfloat16(p);
            Ph[n * BM + m] = ph;
            Pl[n * BM + m] = __float2bfloat16(p - (float)ph);
        }
        __syncthreads();
        out_mma_split(c, Ph, Pl, dOs, g, h);   // dV = P^T @ dO
    }
    __syncthreads();
    store_frags(obuf, c, g, h);
    __syncthreads();
    write_out(dVg, obuf, bh, S, kv_start, tid, 1.0f);
}

// ---------------- dK ----------------
__global__ __launch_bounds__(256)
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
    __nv_bfloat16* dSh = reinterpret_cast<__nv_bfloat16*>(dPs + BM * BN);
    __nv_bfloat16* dSl = dSh + BM * BN;
    float* Ls          = reinterpret_cast<float*>(dSl + BM * BN);
    float* Ds          = Ls + BM;
    float* obuf        = reinterpret_cast<float*>(Ks);             // overlay Ks+Vs (32K)

    int bh = blockIdx.y, kvb = blockIdx.x;
    int kv_start = kvb * BN, tid = threadIdx.x;
    int w = tid >> 5, g = w & 3, h = w >> 2;
    int num_q = (S + BM - 1) / BM;

    load_tile(Ks, Kg, bh, S, kv_start, tid);
    load_tile(Vs, Vg, bh, S, kv_start, tid);

    AccFrag c[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) wmma::fill_fragment(c[i], 0.f);

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
        if (w < 4) mm_AKt(Qs, Ks, Ss, w * 16);
        else       mm_AKt(dOs, Vs, dPs, (w - 4) * 16);
        __syncthreads();
        for (int idx = tid; idx < BM * BN; idx += 256) {
            int m = idx >> 6, n = idx & 63;
            int qi = q_start + m, kj = kv_start + n;
            float p = 0.f;
            if (qi < S && kj < S && kj <= qi) p = __expf(Ss[idx] * scale - Ls[m]);
            float ds = p * (dPs[idx] - Ds[m]);
            __nv_bfloat16 dh = __float2bfloat16(ds);
            dSh[n * BM + m] = dh;                       // transposed
            dSl[n * BM + m] = __float2bfloat16(ds - (float)dh);
        }
        __syncthreads();
        out_mma_split(c, dSh, dSl, Qs, g, h);   // dK = dS^T @ Q
    }
    __syncthreads();
    store_frags(obuf, c, g, h);
    __syncthreads();
    write_out(dKg, obuf, bh, S, kv_start, tid, scale);
}

// ---------------- dQ ----------------
__global__ __launch_bounds__(256)
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
    __nv_bfloat16* dSh = reinterpret_cast<__nv_bfloat16*>(dPs + BM * BN);
    __nv_bfloat16* dSl = dSh + BM * BN;
    float* Ls          = reinterpret_cast<float*>(dSl + BM * BN);
    float* Ds          = Ls + BM;
    float* obuf        = reinterpret_cast<float*>(Qs);             // overlay Qs+dOs (32K)

    int bh = blockIdx.y, qb = blockIdx.x;
    int q_start = qb * BM, tid = threadIdx.x;
    int w = tid >> 5, g = w & 3, h = w >> 2;

    load_tile(Qs, Qg, bh, S, q_start, tid);
    load_tile(dOs, dOg, bh, S, q_start, tid);
    if (tid < BM) {
        int qi = q_start + tid;
        Ls[tid] = (qi < S) ? Lg[(size_t)bh * S + qi] : 0.f;
        Ds[tid] = (qi < S) ? Dg[(size_t)bh * S + qi] : 0.f;
    }

    AccFrag c[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) wmma::fill_fragment(c[i], 0.f);

    for (int j = 0; j <= qb; j++) {
        int kv_start = j * BN;
        __syncthreads();
        load_tile(Ks, Kg, bh, S, kv_start, tid);
        load_tile(Vs, Vg, bh, S, kv_start, tid);
        __syncthreads();
        if (w < 4) mm_AKt(Qs, Ks, Ss, w * 16);
        else       mm_AKt(dOs, Vs, dPs, (w - 4) * 16);
        __syncthreads();
        for (int idx = tid; idx < BM * BN; idx += 256) {
            int m = idx >> 6, n = idx & 63;
            int qi = q_start + m, kj = kv_start + n;
            float p = 0.f;
            if (qi < S && kj < S && kj <= qi) p = __expf(Ss[idx] * scale - Ls[m]);
            float ds = p * (dPs[idx] - Ds[m]);
            __nv_bfloat16 dh = __float2bfloat16(ds);
            dSh[m * BN + n] = dh;                       // normal
            dSl[m * BN + n] = __float2bfloat16(ds - (float)dh);
        }
        __syncthreads();
        out_mma_split(c, dSh, dSl, Ks, g, h);   // dQ = dS @ K
    }
    __syncthreads();
    store_frags(obuf, c, g, h);
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

    int SMEM_DV = (int)(3*BM*D_DIM*2 + BM*BN*4 + 2*BM*BN*2 + BM*4);
    int SMEM_DK = (int)(4*BM*D_DIM*2 + 2*BM*BN*4 + 2*BM*BN*2 + 2*BM*4);
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
    dim3 block(256, 1, 1);

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