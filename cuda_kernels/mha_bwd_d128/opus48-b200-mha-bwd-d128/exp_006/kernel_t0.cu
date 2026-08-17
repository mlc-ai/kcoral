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

// C[M,N] = A[M,Kd](rowmajor) * Bt[N,Kd](rowmajor)^T   (i.e. A @ Bt^T)
__device__ __forceinline__ void gemm_ABt(
    const __nv_bfloat16* A, const __nv_bfloat16* Bt, float* C,
    int MT, int NT, int KT, int lda, int ldb, int ldc,
    int warp, int nwarps)
{
    int ntiles = MT * NT;
    for (int t = warp; t < ntiles; t += nwarps) {
        int mt = t / NT;
        int nt = t % NT;
        wmma::fragment<wmma::accumulator,16,16,16,float> acc;
        wmma::fill_fragment(acc, 0.0f);
        for (int k = 0; k < KT; k++) {
            wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> fb;
            wmma::load_matrix_sync(fa, A + (mt*16)*lda + k*16, lda);
            wmma::load_matrix_sync(fb, Bt + (nt*16)*ldb + k*16, ldb);
            wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(C + (mt*16)*ldc + nt*16, acc, ldc, wmma::mem_row_major);
    }
}

// D_i = sum_k dO_ik * O_ik    (one warp per row, head dim = 128)
__global__ void compute_D_kernel(const __nv_bfloat16* __restrict__ O,
                                 const __nv_bfloat16* __restrict__ dO,
                                 float* __restrict__ Dout, int total_rows)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= total_rows) return;
    const __nv_bfloat16* Op  = O  + (size_t)row * 128;
    const __nv_bfloat16* dOp = dO + (size_t)row * 128;
    float s = 0.f;
    #pragma unroll
    for (int c = lane; c < 128; c += 32)
        s += __bfloat162float(Op[c]) * __bfloat162float(dOp[c]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if (lane == 0) Dout[row] = s;
}

// ---------------- Kernel A: dK, dV ----------------
template<int BM, int BN>
__global__ __launch_bounds__(256,1) void bwd_dkdv_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ Lmat,
    const float* __restrict__ Dmat,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int S, float scale)
{
    const int D = 128;
    const int WARPS = 8;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int bh = blockIdx.y;
    int n0 = blockIdx.x * BN;

    const __nv_bfloat16* Qb  = Q  + (size_t)bh * S * D;
    const __nv_bfloat16* Kb  = K  + (size_t)bh * S * D;
    const __nv_bfloat16* Vb  = V  + (size_t)bh * S * D;
    const __nv_bfloat16* dOb = dO + (size_t)bh * S * D;
    const float* Lb = Lmat + (size_t)bh * S;
    const float* Db = Dmat + (size_t)bh * S;
    __nv_bfloat16* dKb = dK + (size_t)bh * S * D;
    __nv_bfloat16* dVb = dV + (size_t)bh * S * D;

    extern __shared__ char smem[];
    __nv_bfloat16* Ks   = (__nv_bfloat16*)smem;
    __nv_bfloat16* Vs   = Ks + BN*D;
    __nv_bfloat16* Qs   = Vs + BN*D;
    __nv_bfloat16* dOs  = Qs + BM*D;
    float* Sbuf         = (float*)(dOs + BM*D);
    __nv_bfloat16* Pbuf = (__nv_bfloat16*)(Sbuf + BM*BN);
    __nv_bfloat16* dSbuf= Pbuf + BM*BN;
    float* Ls           = (float*)(dSbuf + BM*BN);
    float* Ds           = Ls + BM;

    // load K,V tile (zero pad OOB)
    for (int idx = tid; idx < BN*D; idx += 256) {
        int r = idx / D, c = idx % D; int gr = n0 + r;
        __nv_bfloat16 z = __float2bfloat16(0.f);
        Ks[idx] = (gr < S) ? Kb[(size_t)gr*D + c] : z;
        Vs[idx] = (gr < S) ? Vb[(size_t)gr*D + c] : z;
    }

    int rt = warp & 3;    // key row tile (BN/16 == 4)
    int cg = warp >> 2;   // 0/1 col group over d (8 col tiles)
    wmma::fragment<wmma::accumulator,16,16,16,float> dV_acc[4];
    wmma::fragment<wmma::accumulator,16,16,16,float> dK_acc[4];
    #pragma unroll
    for (int c = 0; c < 4; c++) { wmma::fill_fragment(dV_acc[c],0.f); wmma::fill_fragment(dK_acc[c],0.f); }

    const int KM = BM/16; // contraction subtiles over query
    int num_q = (S + BM - 1) / BM;
    for (int qt = 0; qt < num_q; qt++) {
        int m0 = qt * BM;
        __syncthreads();
        for (int idx = tid; idx < BM*D; idx += 256) {
            int r = idx / D, c = idx % D; int gr = m0 + r;
            __nv_bfloat16 z = __float2bfloat16(0.f);
            Qs[idx]  = (gr < S) ? Qb[(size_t)gr*D + c]  : z;
            dOs[idx] = (gr < S) ? dOb[(size_t)gr*D + c] : z;
        }
        for (int r = tid; r < BM; r += 256) {
            int gr = m0 + r;
            Ls[r] = (gr < S) ? Lb[gr] : 0.f;
            Ds[r] = (gr < S) ? Db[gr] : 0.f;
        }
        __syncthreads();

        // S_raw = Q @ K^T
        gemm_ABt(Qs, Ks, Sbuf, BM/16, BN/16, D/16, D, D, BN, warp, WARPS);
        __syncthreads();

        // P = exp(scale*S - L), masked
        for (int idx = tid; idx < BM*BN; idx += 256) {
            int m = idx / BN, n = idx % BN;
            int gm = m0 + m, gn = n0 + n;
            float p = (gm < S && gn < S) ? __expf(scale * Sbuf[idx] - Ls[m]) : 0.f;
            Pbuf[idx] = __float2bfloat16(p);
        }
        __syncthreads();

        // dP = dO @ V^T
        gemm_ABt(dOs, Vs, Sbuf, BM/16, BN/16, D/16, D, D, BN, warp, WARPS);
        __syncthreads();

        // dS = P * (dP - D)
        for (int idx = tid; idx < BM*BN; idx += 256) {
            int m = idx / BN;
            float p = __bfloat162float(Pbuf[idx]);
            float ds = p * (Sbuf[idx] - Ds[m]);
            dSbuf[idx] = __float2bfloat16(ds);
        }
        __syncthreads();

        // dV += P^T @ dO ;  dK += dS^T @ Q
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            int ct = cg*4 + c;
            #pragma unroll
            for (int k = 0; k < KM; k++) {
                wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::col_major> fa;
                wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> fb;
                wmma::load_matrix_sync(fa, Pbuf + (k*16)*BN + rt*16, BN);
                wmma::load_matrix_sync(fb, dOs + (k*16)*D + ct*16, D);
                wmma::mma_sync(dV_acc[c], fa, fb, dV_acc[c]);
                wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::col_major> fa2;
                wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> fb2;
                wmma::load_matrix_sync(fa2, dSbuf + (k*16)*BN + rt*16, BN);
                wmma::load_matrix_sync(fb2, Qs + (k*16)*D + ct*16, D);
                wmma::mma_sync(dK_acc[c], fa2, fb2, dK_acc[c]);
            }
        }
    }
    __syncthreads();

    float* stage = Sbuf; // reuse
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        int ct = cg*4 + c;
        wmma::store_matrix_sync(stage + warp*256, dV_acc[c], 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            int r = e/16, cc = e%16;
            int gr = n0 + rt*16 + r; int gc = ct*16 + cc;
            if (gr < S) dVb[(size_t)gr*D + gc] = __float2bfloat16(stage[warp*256 + e]);
        }
        __syncwarp();
        wmma::store_matrix_sync(stage + warp*256, dK_acc[c], 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            int r = e/16, cc = e%16;
            int gr = n0 + rt*16 + r; int gc = ct*16 + cc;
            if (gr < S) dKb[(size_t)gr*D + gc] = __float2bfloat16(stage[warp*256 + e] * scale);
        }
        __syncwarp();
    }
}

// ---------------- Kernel B: dQ ----------------
template<int BM, int BN>
__global__ __launch_bounds__(256,1) void bwd_dq_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ Lmat,
    const float* __restrict__ Dmat,
    __nv_bfloat16* __restrict__ dQ,
    int S, float scale)
{
    const int D = 128;
    const int WARPS = 8;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int bh = blockIdx.y;
    int m0 = blockIdx.x * BM;

    const __nv_bfloat16* Qb  = Q  + (size_t)bh * S * D;
    const __nv_bfloat16* Kb  = K  + (size_t)bh * S * D;
    const __nv_bfloat16* Vb  = V  + (size_t)bh * S * D;
    const __nv_bfloat16* dOb = dO + (size_t)bh * S * D;
    const float* Lb = Lmat + (size_t)bh * S;
    const float* Db = Dmat + (size_t)bh * S;
    __nv_bfloat16* dQb = dQ + (size_t)bh * S * D;

    extern __shared__ char smem[];
    __nv_bfloat16* Ks   = (__nv_bfloat16*)smem;
    __nv_bfloat16* Vs   = Ks + BN*D;
    __nv_bfloat16* Qs   = Vs + BN*D;
    __nv_bfloat16* dOs  = Qs + BM*D;
    float* Sbuf         = (float*)(dOs + BM*D);
    float* Pf           = (float*)(Sbuf + BM*BN);
    __nv_bfloat16* dSbuf= (__nv_bfloat16*)(Pf + BM*BN);
    float* Ls           = (float*)(dSbuf + BM*BN);
    float* Ds           = Ls + BM;

    // load Q,dO,L,D once
    for (int idx = tid; idx < BM*D; idx += 256) {
        int r = idx / D, c = idx % D; int gr = m0 + r;
        __nv_bfloat16 z = __float2bfloat16(0.f);
        Qs[idx]  = (gr < S) ? Qb[(size_t)gr*D + c]  : z;
        dOs[idx] = (gr < S) ? dOb[(size_t)gr*D + c] : z;
    }
    for (int r = tid; r < BM; r += 256) {
        int gr = m0 + r;
        Ls[r] = (gr < S) ? Lb[gr] : 0.f;
        Ds[r] = (gr < S) ? Db[gr] : 0.f;
    }

    int rt = warp & 3;    // query row tile
    int cg = warp >> 2;   // col group over d
    wmma::fragment<wmma::accumulator,16,16,16,float> dQ_acc[4];
    #pragma unroll
    for (int c = 0; c < 4; c++) wmma::fill_fragment(dQ_acc[c], 0.f);

    const int KN = BN/16;
    int num_kv = (S + BN - 1) / BN;
    for (int kv = 0; kv < num_kv; kv++) {
        int n0 = kv * BN;
        __syncthreads();
        for (int idx = tid; idx < BN*D; idx += 256) {
            int r = idx / D, c = idx % D; int gr = n0 + r;
            __nv_bfloat16 z = __float2bfloat16(0.f);
            Ks[idx] = (gr < S) ? Kb[(size_t)gr*D + c] : z;
            Vs[idx] = (gr < S) ? Vb[(size_t)gr*D + c] : z;
        }
        __syncthreads();

        gemm_ABt(Qs, Ks, Sbuf, BM/16, BN/16, D/16, D, D, BN, warp, WARPS);
        __syncthreads();

        for (int idx = tid; idx < BM*BN; idx += 256) {
            int m = idx / BN, n = idx % BN;
            int gm = m0 + m, gn = n0 + n;
            Pf[idx] = (gm < S && gn < S) ? __expf(scale * Sbuf[idx] - Ls[m]) : 0.f;
        }
        __syncthreads();

        gemm_ABt(dOs, Vs, Sbuf, BM/16, BN/16, D/16, D, D, BN, warp, WARPS);
        __syncthreads();

        for (int idx = tid; idx < BM*BN; idx += 256) {
            int m = idx / BN;
            float ds = Pf[idx] * (Sbuf[idx] - Ds[m]);
            dSbuf[idx] = __float2bfloat16(ds);
        }
        __syncthreads();

        // dQ += dS @ K
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            int dt = cg*4 + c;
            #pragma unroll
            for (int k = 0; k < KN; k++) {
                wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> fb;
                wmma::load_matrix_sync(fa, dSbuf + (rt*16)*BN + k*16, BN);
                wmma::load_matrix_sync(fb, Ks + (k*16)*D + dt*16, D);
                wmma::mma_sync(dQ_acc[c], fa, fb, dQ_acc[c]);
            }
        }
    }
    __syncthreads();

    float* stage = Sbuf;
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        int dt = cg*4 + c;
        wmma::store_matrix_sync(stage + warp*256, dQ_acc[c], 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            int r = e/16, cc = e%16;
            int gr = m0 + rt*16 + r; int gc = dt*16 + cc;
            if (gr < S) dQb[(size_t)gr*D + gc] = __float2bfloat16(stage[warp*256 + e] * scale);
        }
        __syncwarp();
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV)
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int d = (int)Q.size(3);
    int BH = B * H;
    float scale = 1.0f / sqrtf((float)d);

    const __nv_bfloat16* Qp  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* Op  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dOp = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* Lp = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQp = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dKp = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dVp = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* Dscr = nullptr;
    CUDA_CHECK(cudaMallocAsync(&Dscr, (size_t)BH * S * sizeof(float), stream));

    int total_rows = BH * S;
    compute_D_kernel<<<(total_rows + 3) / 4, 128, 0, stream>>>(Op, dOp, Dscr, total_rows);
    CUDA_CHECK(cudaGetLastError());

    constexpr int BM = 64, BN = 64;
    size_t smemA = (size_t)(BN*128 + BN*128 + BM*128 + BM*128) * 2
                 + (size_t)BM*BN*4 + (size_t)BM*BN*2 + (size_t)BM*BN*2
                 + (size_t)BM*4*2;
    size_t smemB = (size_t)(BN*128 + BN*128 + BM*128 + BM*128) * 2
                 + (size_t)BM*BN*4 + (size_t)BM*BN*4 + (size_t)BM*BN*2
                 + (size_t)BM*4*2;

    CUDA_CHECK(cudaFuncSetAttribute(bwd_dkdv_kernel<BM,BN>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemA));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel<BM,BN>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemB));

    dim3 gridA((S + BN - 1) / BN, BH);
    dim3 gridB((S + BM - 1) / BM, BH);

    bwd_dkdv_kernel<BM,BN><<<gridA, 256, smemA, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Dscr, dKp, dVp, S, scale);
    CUDA_CHECK(cudaGetLastError());

    bwd_dq_kernel<BM,BN><<<gridB, 256, smemB, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Dscr, dQp, S, scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(Dscr, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd