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
using bf16 = __nv_bfloat16;

static constexpr int LDD = 136;  // padded head-dim leading (d=128 +8)
static constexpr int LDN = 72;   // padded BN leading (BN=64 +8)

// Register-blocked GEMM: C[M][N] = A[M][K] @ B[N][K]^T
template<int M,int N,int Kd,int WM,int WN>
__device__ __forceinline__ void gemm_qk(
    const bf16* A, const bf16* B, float* C,
    int lda, int ldb, int ldc, int warp)
{
    constexpr int GN = N/WN;
    constexpr int SM = WM/16, SN = WN/16, SK = Kd/16;
    int wm = warp / GN, wn = warp % GN;
    int m0 = wm*WM, n0 = wn*WN;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc[SM][SN];
    #pragma unroll
    for(int i=0;i<SM;i++)
      #pragma unroll
      for(int j=0;j<SN;j++) wmma::fill_fragment(acc[i][j],0.f);
    #pragma unroll
    for(int k=0;k<SK;k++){
      wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> af[SM];
      wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> bf[SN];
      #pragma unroll
      for(int i=0;i<SM;i++) wmma::load_matrix_sync(af[i], A + (m0+i*16)*lda + k*16, lda);
      #pragma unroll
      for(int j=0;j<SN;j++) wmma::load_matrix_sync(bf[j], B + (n0+j*16)*ldb + k*16, ldb);
      #pragma unroll
      for(int i=0;i<SM;i++)
        #pragma unroll
        for(int j=0;j<SN;j++) wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    #pragma unroll
    for(int i=0;i<SM;i++)
      #pragma unroll
      for(int j=0;j<SN;j++)
        wmma::store_matrix_sync(C + (m0+i*16)*ldc + (n0+j*16), acc[i][j], ldc, wmma::mem_row_major);
}

// D_i = sum_k dO_ik * O_ik  (one warp per row, head dim = 128)
__global__ void compute_D_kernel(const bf16* __restrict__ O,
                                 const bf16* __restrict__ dO,
                                 float* __restrict__ Dout, int total_rows)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= total_rows) return;
    const bf16* Op  = O  + (size_t)row * 128;
    const bf16* dOp = dO + (size_t)row * 128;
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
__global__ __launch_bounds__(256) void bwd_dkdv_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K,
    const bf16* __restrict__ V, const bf16* __restrict__ dO,
    const float* __restrict__ Lmat, const float* __restrict__ Dmat,
    bf16* __restrict__ dK, bf16* __restrict__ dV,
    int S, float scale)
{
    const int D = 128;
    int tid = threadIdx.x, warp = tid >> 5;
    int bh = blockIdx.y;
    int n0 = blockIdx.x * BN;

    const bf16* Qb  = Q  + (size_t)bh * S * D;
    const bf16* Kb  = K  + (size_t)bh * S * D;
    const bf16* Vb  = V  + (size_t)bh * S * D;
    const bf16* dOb = dO + (size_t)bh * S * D;
    const float* Lb = Lmat + (size_t)bh * S;
    const float* Db = Dmat + (size_t)bh * S;
    bf16* dKb = dK + (size_t)bh * S * D;
    bf16* dVb = dV + (size_t)bh * S * D;

    extern __shared__ char smem[];
    bf16* Ks   = (bf16*)smem;
    bf16* Vs   = Ks + BN*LDD;
    bf16* Qs   = Vs + BN*LDD;
    bf16* dOs  = Qs + BM*LDD;
    float* Sbuf= (float*)(dOs + BM*LDD);
    bf16* Pbuf = (bf16*)(Sbuf + BM*LDN);
    bf16* dSbuf= Pbuf + BM*LDN;
    float* Ls  = (float*)(dSbuf + BM*LDN);
    float* Ds  = Ls + BM;

    for (int idx = tid; idx < BN*D; idx += 256) {
        int r = idx / D, c = idx % D; int gr = n0 + r;
        bf16 z = __float2bfloat16(0.f);
        Ks[r*LDD + c] = (gr < S) ? Kb[(size_t)gr*D + c] : z;
        Vs[r*LDD + c] = (gr < S) ? Vb[(size_t)gr*D + c] : z;
    }

    int wn_n = warp >> 2;   // 0..1
    int wn_d = warp & 3;    // 0..3
    int n_base = wn_n*32, d_base = wn_d*32;
    wmma::fragment<wmma::accumulator,16,16,16,float> dV_acc[2][2];
    wmma::fragment<wmma::accumulator,16,16,16,float> dK_acc[2][2];
    #pragma unroll
    for(int i=0;i<2;i++)
      #pragma unroll
      for(int j=0;j<2;j++){ wmma::fill_fragment(dV_acc[i][j],0.f); wmma::fill_fragment(dK_acc[i][j],0.f); }

    const int SK = BM/16;
    int num_q = (S + BM - 1) / BM;
    for (int qt = 0; qt < num_q; qt++) {
        int m0 = qt * BM;
        __syncthreads();
        for (int idx = tid; idx < BM*D; idx += 256) {
            int r = idx / D, c = idx % D; int gr = m0 + r;
            bf16 z = __float2bfloat16(0.f);
            Qs[r*LDD + c]  = (gr < S) ? Qb[(size_t)gr*D + c]  : z;
            dOs[r*LDD + c] = (gr < S) ? dOb[(size_t)gr*D + c] : z;
        }
        for (int r = tid; r < BM; r += 256) {
            int gr = m0 + r;
            Ls[r] = (gr < S) ? Lb[gr] : 0.f;
            Ds[r] = (gr < S) ? Db[gr] : 0.f;
        }
        __syncthreads();

        gemm_qk<BM,BN,D,32,32>(Qs, Ks, Sbuf, LDD, LDD, LDN, warp);
        __syncthreads();

        for (int idx = tid; idx < BM*BN; idx += 256) {
            int m = idx / BN, n = idx % BN;
            int gm = m0 + m, gn = n0 + n;
            float p = (gm < S && gn < S) ? __expf(scale * Sbuf[m*LDN+n] - Ls[m]) : 0.f;
            Pbuf[m*LDN+n] = __float2bfloat16(p);
        }
        __syncthreads();

        gemm_qk<BM,BN,D,32,32>(dOs, Vs, Sbuf, LDD, LDD, LDN, warp);
        __syncthreads();

        for (int idx = tid; idx < BM*BN; idx += 256) {
            int m = idx / BN, n = idx % BN;
            float p = __bfloat162float(Pbuf[m*LDN+n]);
            dSbuf[m*LDN+n] = __float2bfloat16(p * (Sbuf[m*LDN+n] - Ds[m]));
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < SK; k++) {
            wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::col_major> aP[2], aS[2];
            wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> bO[2], bQ[2];
            #pragma unroll
            for (int i=0;i<2;i++) {
                wmma::load_matrix_sync(aP[i], Pbuf  + (n_base+i*16) + (k*16)*LDN, LDN);
                wmma::load_matrix_sync(aS[i], dSbuf + (n_base+i*16) + (k*16)*LDN, LDN);
            }
            #pragma unroll
            for (int j=0;j<2;j++) {
                wmma::load_matrix_sync(bO[j], dOs + (k*16)*LDD + (d_base+j*16), LDD);
                wmma::load_matrix_sync(bQ[j], Qs  + (k*16)*LDD + (d_base+j*16), LDD);
            }
            #pragma unroll
            for (int i=0;i<2;i++)
              #pragma unroll
              for (int j=0;j<2;j++) {
                wmma::mma_sync(dV_acc[i][j], aP[i], bO[j], dV_acc[i][j]);
                wmma::mma_sync(dK_acc[i][j], aS[i], bQ[j], dK_acc[i][j]);
              }
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i=0;i<2;i++)
      #pragma unroll
      for (int j=0;j<2;j++)
        wmma::store_matrix_sync(Sbuf + (n_base+i*16)*LDD + (d_base+j*16), dV_acc[i][j], LDD, wmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < BN*D; idx += 256) {
        int n = idx / D, dd = idx % D; int gr = n0 + n;
        if (gr < S) dVb[(size_t)gr*D + dd] = __float2bfloat16(Sbuf[n*LDD+dd]);
    }
    __syncthreads();
    #pragma unroll
    for (int i=0;i<2;i++)
      #pragma unroll
      for (int j=0;j<2;j++)
        wmma::store_matrix_sync(Sbuf + (n_base+i*16)*LDD + (d_base+j*16), dK_acc[i][j], LDD, wmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < BN*D; idx += 256) {
        int n = idx / D, dd = idx % D; int gr = n0 + n;
        if (gr < S) dKb[(size_t)gr*D + dd] = __float2bfloat16(Sbuf[n*LDD+dd] * scale);
    }
}

// ---------------- Kernel B: dQ ----------------
template<int BM, int BN>
__global__ __launch_bounds__(256) void bwd_dq_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K,
    const bf16* __restrict__ V, const bf16* __restrict__ dO,
    const float* __restrict__ Lmat, const float* __restrict__ Dmat,
    bf16* __restrict__ dQ, int S, float scale)
{
    const int D = 128;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int bh = blockIdx.y;
    int m0 = blockIdx.x * BM;

    const bf16* Qb  = Q  + (size_t)bh * S * D;
    const bf16* Kb  = K  + (size_t)bh * S * D;
    const bf16* Vb  = V  + (size_t)bh * S * D;
    const bf16* dOb = dO + (size_t)bh * S * D;
    const float* Lb = Lmat + (size_t)bh * S;
    const float* Db = Dmat + (size_t)bh * S;
    bf16* dQb = dQ + (size_t)bh * S * D;

    extern __shared__ char smem[];
    bf16* Qs   = (bf16*)smem;
    bf16* dOs  = Qs + BM*LDD;
    bf16* Ks   = dOs + BM*LDD;
    bf16* Vs   = Ks + BN*LDD;
    float* Sbuf= (float*)(Vs + BN*LDD);
    float* Pf  = Sbuf + BM*LDN;
    bf16* dSbuf= (bf16*)(Pf + BM*LDN);
    float* Ls  = (float*)(dSbuf + BM*LDN);
    float* Ds  = Ls + BM;
    float* stage = Ds + BM;   // 8*256

    for (int idx = tid; idx < BM*D; idx += 256) {
        int r = idx / D, c = idx % D; int gr = m0 + r;
        bf16 z = __float2bfloat16(0.f);
        Qs[r*LDD + c]  = (gr < S) ? Qb[(size_t)gr*D + c]  : z;
        dOs[r*LDD + c] = (gr < S) ? dOb[(size_t)gr*D + c] : z;
    }
    for (int r = tid; r < BM; r += 256) {
        int gr = m0 + r;
        Ls[r] = (gr < S) ? Lb[gr] : 0.f;
        Ds[r] = (gr < S) ? Db[gr] : 0.f;
    }

    int wm = warp >> 2;   // 0..1
    int wd = warp & 3;    // 0..3
    int m_base = wm*64, d_base = wd*32;
    wmma::fragment<wmma::accumulator,16,16,16,float> dQ_acc[4][2];
    #pragma unroll
    for(int i=0;i<4;i++)
      #pragma unroll
      for(int j=0;j<2;j++) wmma::fill_fragment(dQ_acc[i][j],0.f);

    const int SK = BN/16;
    int num_kv = (S + BN - 1) / BN;
    for (int kv = 0; kv < num_kv; kv++) {
        int n0 = kv * BN;
        __syncthreads();
        for (int idx = tid; idx < BN*D; idx += 256) {
            int r = idx / D, c = idx % D; int gr = n0 + r;
            bf16 z = __float2bfloat16(0.f);
            Ks[r*LDD + c] = (gr < S) ? Kb[(size_t)gr*D + c] : z;
            Vs[r*LDD + c] = (gr < S) ? Vb[(size_t)gr*D + c] : z;
        }
        __syncthreads();

        gemm_qk<BM,BN,D,32,32>(Qs, Ks, Sbuf, LDD, LDD, LDN, warp);
        __syncthreads();

        for (int idx = tid; idx < BM*BN; idx += 256) {
            int m = idx / BN, n = idx % BN;
            int gm = m0 + m, gn = n0 + n;
            Pf[m*LDN+n] = (gm < S && gn < S) ? __expf(scale * Sbuf[m*LDN+n] - Ls[m]) : 0.f;
        }
        __syncthreads();

        gemm_qk<BM,BN,D,32,32>(dOs, Vs, Sbuf, LDD, LDD, LDN, warp);
        __syncthreads();

        for (int idx = tid; idx < BM*BN; idx += 256) {
            int m = idx / BN, n = idx % BN;
            dSbuf[m*LDN+n] = __float2bfloat16(Pf[m*LDN+n] * (Sbuf[m*LDN+n] - Ds[m]));
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < SK; k++) {
            wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> aS[4];
            wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> bK[2];
            #pragma unroll
            for (int i=0;i<4;i++)
                wmma::load_matrix_sync(aS[i], dSbuf + (m_base+i*16)*LDN + k*16, LDN);
            #pragma unroll
            for (int j=0;j<2;j++)
                wmma::load_matrix_sync(bK[j], Ks + (k*16)*LDD + (d_base+j*16), LDD);
            #pragma unroll
            for (int i=0;i<4;i++)
              #pragma unroll
              for (int j=0;j<2;j++)
                wmma::mma_sync(dQ_acc[i][j], aS[i], bK[j], dQ_acc[i][j]);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i=0;i<4;i++)
      #pragma unroll
      for (int j=0;j<2;j++) {
        wmma::store_matrix_sync(stage + warp*256, dQ_acc[i][j], 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            int r = e/16, c = e%16;
            int gr = m0 + m_base + i*16 + r;
            int gc = d_base + j*16 + c;
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

    const bf16* Qp  = static_cast<const bf16*>(Q.data_ptr());
    const bf16* Kp  = static_cast<const bf16*>(K.data_ptr());
    const bf16* Vp  = static_cast<const bf16*>(V.data_ptr());
    const bf16* Op  = static_cast<const bf16*>(O.data_ptr());
    const bf16* dOp = static_cast<const bf16*>(dO.data_ptr());
    const float* Lp = static_cast<const float*>(L.data_ptr());
    bf16* dQp = static_cast<bf16*>(dQ.data_ptr());
    bf16* dKp = static_cast<bf16*>(dK.data_ptr());
    bf16* dVp = static_cast<bf16*>(dV.data_ptr());

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* Dscr = nullptr;
    CUDA_CHECK(cudaMallocAsync(&Dscr, (size_t)BH * S * sizeof(float), stream));

    int total_rows = BH * S;
    compute_D_kernel<<<(total_rows + 3) / 4, 128, 0, stream>>>(Op, dOp, Dscr, total_rows);
    CUDA_CHECK(cudaGetLastError());

    constexpr int BM = 128, BN = 64;

    size_t smemA = (size_t)(BN*LDD + BN*LDD + BM*LDD + BM*LDD) * sizeof(bf16)
                 + (size_t)(BM*LDN) * sizeof(float)
                 + (size_t)(BM*LDN + BM*LDN) * sizeof(bf16)
                 + (size_t)(BM + BM) * sizeof(float);

    size_t smemB = (size_t)(BM*LDD + BM*LDD + BN*LDD + BN*LDD) * sizeof(bf16)
                 + (size_t)(BM*LDN + BM*LDN) * sizeof(float)
                 + (size_t)(BM*LDN) * sizeof(bf16)
                 + (size_t)(BM + BM + 8*256) * sizeof(float);

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