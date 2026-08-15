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
#define LDT 136          // padded leading dim for [64 x 128] tiles
#define LDS 72           // padded leading dim for [64 x 64] score buffers
#define TILE_ELEM (64*LDT)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem){
    unsigned s=(unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(s),"l"(gmem):"memory");
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n":::"memory"); }
template<int N> __device__ __forceinline__ void cp_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory"); }

__device__ inline void aload_tile(__nv_bfloat16* dst, const __nv_bfloat16* g,
                                  int bh, int S, int row_start, int tid){
    #pragma unroll
    for (int v = tid; v < 64*16; v += 256){
        int row = v >> 4, col = (v & 15) * 8;
        int gr = row_start + row;
        if (gr >= S) gr = S - 1;
        if (gr < 0) gr = 0;
        cp_async16(&dst[row*LDT + col], &g[((size_t)bh*S + gr)*D_DIM + col]);
    }
}

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

// out[r0..r0+15, 0..63] = A[..,128] @ B[64,128]^T  (per warp; bf16 in, fp32 acc)
__device__ inline void mm_AKt(const __nv_bfloat16* A, const __nv_bfloat16* B,
                              float* out, int r0) {
    AccFrag acc[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) wmma::fill_fragment(acc[j], 0.f);
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        ARowFrag af;
        wmma::load_matrix_sync(af, A + r0 * LDT + k * 16, LDT);
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            BColFrag bf;
            wmma::load_matrix_sync(bf, B + (nt * 16) * LDT + k * 16, LDT);
            wmma::mma_sync(acc[nt], af, bf, acc[nt]);
        }
    }
    #pragma unroll
    for (int nt = 0; nt < 4; nt++)
        wmma::store_matrix_sync(out + r0 * LDS + nt * 16, acc[nt], LDS, wmma::mem_row_major);
}

// dV/dK : output rows = key index n. A stored [m][n] ld LDS; read as A^T (col_major).
__device__ inline void out_mma_col(AccFrag* c, const __nv_bfloat16* Ah,
                                   const __nv_bfloat16* Al, const __nv_bfloat16* B, int g, int h) {
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        AColFrag ah, al;
        wmma::load_matrix_sync(ah, Ah + g*16 + (k*16)*LDS, LDS);
        wmma::load_matrix_sync(al, Al + g*16 + (k*16)*LDS, LDS);
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            BRowFrag b;
            wmma::load_matrix_sync(b, B + (k*16)*LDT + h*64 + nt*16, LDT);
            wmma::mma_sync(c[nt], ah, b, c[nt]);
            wmma::mma_sync(c[nt], al, b, c[nt]);
        }
    }
}

// dQ : output rows = query index m. A stored [m][n] ld LDS; read directly (row_major).
__device__ inline void out_mma_row(AccFrag* c, const __nv_bfloat16* Ah,
                                   const __nv_bfloat16* Al, const __nv_bfloat16* B, int g, int h) {
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        ARowFrag ah, al;
        wmma::load_matrix_sync(ah, Ah + (g*16)*LDS + k*16, LDS);
        wmma::load_matrix_sync(al, Al + (g*16)*LDS + k*16, LDS);
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            BRowFrag b;
            wmma::load_matrix_sync(b, B + (k*16)*LDT + h*64 + nt*16, LDT);
            wmma::mma_sync(c[nt], ah, b, c[nt]);
            wmma::mma_sync(c[nt], al, b, c[nt]);
        }
    }
}

__device__ inline void store_frags(float* obuf, AccFrag* c, int g, int h) {
    #pragma unroll
    for (int nt = 0; nt < 4; nt++)
        wmma::store_matrix_sync(obuf + (g*16)*LDT + h*64 + nt*16, c[nt], LDT, wmma::mem_row_major);
}

__device__ inline void write_out(__nv_bfloat16* G, float* obuf, int bh, int S,
                                  int row_start, int tid, float mul) {
    for (int v = tid; v < 64 * 16; v += 256) {
        int row = v >> 4, col = (v & 15) * 8;
        int gr = row_start + row;
        if (gr < S) {
            int4 packed;
            __nv_bfloat16* hp = reinterpret_cast<__nv_bfloat16*>(&packed);
            float* op = &obuf[row * LDT + col];
            #pragma unroll
            for (int e = 0; e < 8; e++) hp[e] = __float2bfloat16(op[e] * mul);
            *reinterpret_cast<int4*>(&G[((size_t)bh * S + gr) * D_DIM + col]) = packed;
        }
    }
}

// ============== merged dK + dV ==============
__global__ __launch_bounds__(256)
void bwd_dkdv_kernel(const __nv_bfloat16* __restrict__ Qg,
                     const __nv_bfloat16* __restrict__ Kg,
                     const __nv_bfloat16* __restrict__ Vg,
                     const __nv_bfloat16* __restrict__ dOg,
                     const float* __restrict__ Lg,
                     const float* __restrict__ Dg,
                     __nv_bfloat16* __restrict__ dKg,
                     __nv_bfloat16* __restrict__ dVg,
                     int S, float scale) {
    extern __shared__ char smem[];
    __nv_bfloat16* Ks  = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* Vs  = Ks + TILE_ELEM;
    __nv_bfloat16* Q0  = Vs + TILE_ELEM;
    __nv_bfloat16* Q1  = Q0 + TILE_ELEM;
    __nv_bfloat16* dO0 = Q1 + TILE_ELEM;
    __nv_bfloat16* dO1 = dO0 + TILE_ELEM;
    float* Ss          = reinterpret_cast<float*>(dO1 + TILE_ELEM);
    float* dPs         = Ss + 64 * LDS;
    __nv_bfloat16* Ph  = reinterpret_cast<__nv_bfloat16*>(dPs + 64 * LDS);
    __nv_bfloat16* Pl  = Ph + 64 * LDS;
    __nv_bfloat16* dSh = Pl + 64 * LDS;
    __nv_bfloat16* dSl = dSh + 64 * LDS;
    float* Ls          = reinterpret_cast<float*>(dSl + 64 * LDS);
    float* Ds          = Ls + BM;
    float* obuf        = reinterpret_cast<float*>(Ks);   // overlay Ks+Vs

    int bh = blockIdx.y, kvb = blockIdx.x;
    int kv_start = kvb * BN, tid = threadIdx.x;
    int w = tid >> 5, g = w & 3, h = w >> 2;
    int num_q = (S + BM - 1) / BM;
    int niter = num_q - kvb;

    aload_tile(Ks, Kg, bh, S, kv_start, tid);
    aload_tile(Vs, Vg, bh, S, kv_start, tid);
    aload_tile(Q0, Qg, bh, S, kvb*BM, tid);
    aload_tile(dO0, dOg, bh, S, kvb*BM, tid);
    cp_commit();

    AccFrag cV[4], cK[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) { wmma::fill_fragment(cV[i], 0.f); wmma::fill_fragment(cK[i], 0.f); }

    for (int i = 0; i < niter; i++) {
        int qt = kvb + i, q_start = qt * BM, cur = i & 1;
        __nv_bfloat16* Qcur = cur ? Q1 : Q0;
        __nv_bfloat16* dOcur = cur ? dO1 : dO0;

        if (i + 1 < niter) {
            int nb = cur ^ 1, qn = (qt + 1) * BM;
            __nv_bfloat16* Qn = nb ? Q1 : Q0;
            __nv_bfloat16* dOn = nb ? dO1 : dO0;
            aload_tile(Qn, Qg, bh, S, qn, tid);
            aload_tile(dOn, dOg, bh, S, qn, tid);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        if (tid < BM) {
            int qi = q_start + tid;
            Ls[tid] = (qi < S) ? Lg[(size_t)bh * S + qi] : 0.f;
            Ds[tid] = (qi < S) ? Dg[(size_t)bh * S + qi] : 0.f;
        }
        __syncthreads();

        if (w < 4) mm_AKt(Qcur, Ks, Ss, w * 16);
        else       mm_AKt(dOcur, Vs, dPs, (w - 4) * 16);
        __syncthreads();

        for (int idx = tid; idx < BM * BN; idx += 256) {
            int m = idx >> 6, n = idx & 63;
            int p_idx = m * LDS + n;
            int qi = q_start + m, kj = kv_start + n;
            float p = 0.f;
            if (qi < S && kj < S && kj <= qi) p = __expf(Ss[p_idx] * scale - Ls[m]);
            __nv_bfloat16 ph = __float2bfloat16(p);
            Ph[p_idx] = ph;
            Pl[p_idx] = __float2bfloat16(p - (float)ph);
            float ds = p * (dPs[p_idx] - Ds[m]);
            __nv_bfloat16 dh = __float2bfloat16(ds);
            dSh[p_idx] = dh;
            dSl[p_idx] = __float2bfloat16(ds - (float)dh);
        }
        __syncthreads();

        out_mma_col(cV, Ph, Pl, dOcur, g, h);   // dV = P^T @ dO
        out_mma_col(cK, dSh, dSl, Qcur, g, h);  // dK = dS^T @ Q
        __syncthreads();
    }

    store_frags(obuf, cV, g, h);
    __syncthreads();
    write_out(dVg, obuf, bh, S, kv_start, tid, 1.0f);
    __syncthreads();
    store_frags(obuf, cK, g, h);
    __syncthreads();
    write_out(dKg, obuf, bh, S, kv_start, tid, scale);
}

// ============== dQ ==============
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
    __nv_bfloat16* Qs  = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* dOs = Qs + TILE_ELEM;
    __nv_bfloat16* K0  = dOs + TILE_ELEM;
    __nv_bfloat16* K1  = K0 + TILE_ELEM;
    __nv_bfloat16* V0  = K1 + TILE_ELEM;
    __nv_bfloat16* V1  = V0 + TILE_ELEM;
    float* Ss          = reinterpret_cast<float*>(V1 + TILE_ELEM);
    float* dPs         = Ss + 64 * LDS;
    __nv_bfloat16* dSh = reinterpret_cast<__nv_bfloat16*>(dPs + 64 * LDS);
    __nv_bfloat16* dSl = dSh + 64 * LDS;
    float* Ls          = reinterpret_cast<float*>(dSl + 64 * LDS);
    float* Ds          = Ls + BM;
    float* obuf        = reinterpret_cast<float*>(K0);   // overlay K0+K1

    int bh = blockIdx.y, qb = blockIdx.x;
    int q_start = qb * BM, tid = threadIdx.x;
    int w = tid >> 5, g = w & 3, h = w >> 2;
    int niter = qb + 1;

    aload_tile(Qs, Qg, bh, S, q_start, tid);
    aload_tile(dOs, dOg, bh, S, q_start, tid);
    aload_tile(K0, Kg, bh, S, 0, tid);
    aload_tile(V0, Vg, bh, S, 0, tid);
    cp_commit();
    if (tid < BM) {
        int qi = q_start + tid;
        Ls[tid] = (qi < S) ? Lg[(size_t)bh * S + qi] : 0.f;
        Ds[tid] = (qi < S) ? Dg[(size_t)bh * S + qi] : 0.f;
    }

    AccFrag cD[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) wmma::fill_fragment(cD[i], 0.f);

    for (int i = 0; i < niter; i++) {
        int j = i, kv_start = j * BN, cur = i & 1;
        __nv_bfloat16* Kcur = cur ? K1 : K0;
        __nv_bfloat16* Vcur = cur ? V1 : V0;

        if (i + 1 < niter) {
            int nb = cur ^ 1, kn = (j + 1) * BN;
            __nv_bfloat16* Kn = nb ? K1 : K0;
            __nv_bfloat16* Vn = nb ? V1 : V0;
            aload_tile(Kn, Kg, bh, S, kn, tid);
            aload_tile(Vn, Vg, bh, S, kn, tid);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        if (w < 4) mm_AKt(Qs, Kcur, Ss, w * 16);
        else       mm_AKt(dOs, Vcur, dPs, (w - 4) * 16);
        __syncthreads();

        for (int idx = tid; idx < BM * BN; idx += 256) {
            int m = idx >> 6, n = idx & 63;
            int p_idx = m * LDS + n;
            int qi = q_start + m, kj = kv_start + n;
            float p = 0.f;
            if (qi < S && kj < S && kj <= qi) p = __expf(Ss[p_idx] * scale - Ls[m]);
            float ds = p * (dPs[p_idx] - Ds[m]);
            __nv_bfloat16 dh = __float2bfloat16(ds);
            dSh[p_idx] = dh;
            dSl[p_idx] = __float2bfloat16(ds - (float)dh);
        }
        __syncthreads();

        out_mma_row(cD, dSh, dSl, Kcur, g, h);   // dQ = dS @ K
        __syncthreads();
    }

    store_frags(obuf, cD, g, h);
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

    int SMEM_DKDV = (int)(6*TILE_ELEM*2 + 2*64*LDS*4 + 4*64*LDS*2 + 2*BM*4);
    int SMEM_DQ   = (int)(6*TILE_ELEM*2 + 2*64*LDS*4 + 2*64*LDS*2 + 2*BM*4);

    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute(bwd_dkdv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_DKDV));
        CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_DQ));
        attr_set = true;
    }

    int num_kv = (S + BN - 1) / BN;
    int num_q  = (S + BM - 1) / BM;
    dim3 block(256, 1, 1);

    bwd_dkdv_kernel<<<dim3((unsigned)num_kv, (unsigned)BH, 1), block, SMEM_DKDV, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Dbuf, dKp, dVp, S, scale);
    bwd_dq_kernel<<<dim3((unsigned)num_q, (unsigned)BH, 1), block, SMEM_DQ, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Dbuf, dQp, S, scale);

    CUDA_CHECK(cudaFreeAsync(Dbuf, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd