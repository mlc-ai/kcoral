#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
    } \
} while(0)

namespace mha_bwd {

using bf16 = __nv_bfloat16;
using half_t = __half;

#define HD 128
#define BK 64
#define BQ 64
#define NT 128
#define NWARPS 4

__device__ __forceinline__ uint32_t ld2(const half_t* p){
    return *reinterpret_cast<const uint32_t*>(p);
}

// fp16 x fp16 -> f32.  C[M,N] += A[M,K]*B[N,K]  (contraction over K)
template<int M,int N,int K,int LDA,int LDB,int LDC,bool ACC>
__device__ __forceinline__ void gemm(const half_t* A, const half_t* B, float* C, int warp_id, int lane){
    const int ROWS = M / NWARPS;
    int group = lane >> 2;
    int tig = lane & 3;
    int t2 = tig * 2;
    #pragma unroll
    for (int rm = 0; rm < ROWS; rm += 16) {
        int mrow = warp_id * ROWS + rm;
        #pragma unroll
        for (int nn = 0; nn < N; nn += 8) {
            float c0,c1,c2,c3;
            if (ACC) {
                c0 = C[(mrow+group)*LDC + nn + t2];
                c1 = C[(mrow+group)*LDC + nn + t2 + 1];
                c2 = C[(mrow+group+8)*LDC + nn + t2];
                c3 = C[(mrow+group+8)*LDC + nn + t2 + 1];
            } else { c0=c1=c2=c3=0.f; }
            #pragma unroll
            for (int kk = 0; kk < K; kk += 16) {
                uint32_t a0 = ld2(&A[(mrow+group)*LDA + kk + t2]);
                uint32_t a1 = ld2(&A[(mrow+group+8)*LDA + kk + t2]);
                uint32_t a2 = ld2(&A[(mrow+group)*LDA + kk + 8 + t2]);
                uint32_t a3 = ld2(&A[(mrow+group+8)*LDA + kk + 8 + t2]);
                uint32_t b0 = ld2(&B[(nn+group)*LDB + kk + t2]);
                uint32_t b1 = ld2(&B[(nn+group)*LDB + kk + 8 + t2]);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
            C[(mrow+group)*LDC + nn + t2]       = c0;
            C[(mrow+group)*LDC + nn + t2 + 1]   = c1;
            C[(mrow+group+8)*LDC + nn + t2]     = c2;
            C[(mrow+group+8)*LDC + nn + t2 + 1] = c3;
        }
    }
}

__device__ __forceinline__ half_t bf2h(bf16 x){ return __float2half(__bfloat162float(x)); }

__global__ void compute_D_kernel(const bf16* __restrict__ O, const bf16* __restrict__ dO,
                                 float* __restrict__ Dout, int total_rows) {
    int row = blockIdx.x;
    if (row >= total_rows) return;
    int tid = threadIdx.x;
    float v = __bfloat162float(O[(size_t)row*HD + tid]) * __bfloat162float(dO[(size_t)row*HD + tid]);
    __shared__ float sm[NT];
    sm[tid] = v;
    __syncthreads();
    for (int s = NT/2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid+s];
        __syncthreads();
    }
    if (tid == 0) Dout[row] = sm[0];
}

__global__ void __launch_bounds__(NT) bwd_dkv_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
    const bf16* __restrict__ dO, const float* __restrict__ Lv, const float* __restrict__ Dv,
    bf16* __restrict__ dK, bf16* __restrict__ dV, int S, float scale)
{
    int bh = blockIdx.y;
    int kb = blockIdx.x;
    int kj0 = kb*BK;
    if (kj0 >= S) return;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    const bf16* Kp  = K  + (size_t)bh*S*HD;
    const bf16* Vp  = V  + (size_t)bh*S*HD;
    const bf16* Qp  = Q  + (size_t)bh*S*HD;
    const bf16* dOp = dO + (size_t)bh*S*HD;
    const float* Lp = Lv + (size_t)bh*S;
    const float* Dp = Dv + (size_t)bh*S;
    bf16* dKp = dK + (size_t)bh*S*HD;
    bf16* dVp = dV + (size_t)bh*S*HD;

    extern __shared__ __align__(16) char smem[];
    half_t* Ksm   = (half_t*)smem;               // [BK,HD]
    half_t* Vsm   = Ksm + BK*HD;                 // [BK,HD]
    half_t* Qsm   = Vsm + BK*HD;                 // [BQ,HD]
    half_t* QTsm  = Qsm + BQ*HD;                 // [HD,BQ]
    half_t* dOsm  = QTsm + HD*BQ;                // [BQ,HD]
    half_t* dOTsm = dOsm + BQ*HD;                // [HD,BQ]
    half_t* Ph    = dOTsm + HD*BQ;               // [BK,BQ]
    half_t* dSh   = Ph + BK*BQ;                  // [BK,BQ]
    float* Cfp    = (float*)(dSh + BK*BQ);       // [BK,BQ]
    float* dVacc  = Cfp + BK*BQ;                 // [BK,HD]
    float* dKacc  = dVacc + BK*HD;               // [BK,HD]
    float* Lsm    = dKacc + BK*HD;               // [BQ]
    float* Dsm    = Lsm + BQ;                    // [BQ]

    for (int idx = tid; idx < BK*HD; idx += NT) {
        int j = idx / HD, k = idx % HD;
        int gj = kj0 + j;
        Ksm[idx] = (gj < S) ? bf2h(Kp[(size_t)gj*HD + k]) : __float2half(0.f);
        Vsm[idx] = (gj < S) ? bf2h(Vp[(size_t)gj*HD + k]) : __float2half(0.f);
        dVacc[idx] = 0.f; dKacc[idx] = 0.f;
    }
    __syncthreads();

    int numQB = (S + BQ - 1)/BQ;
    for (int qb = kb; qb < numQB; qb++) {
        int qi0 = qb*BQ;
        for (int idx = tid; idx < BQ*HD; idx += NT) {
            int i = idx / HD, k = idx % HD;
            int gi = qi0 + i;
            half_t qv = (gi < S) ? bf2h(Qp[(size_t)gi*HD + k]) : __float2half(0.f);
            half_t dv = (gi < S) ? bf2h(dOp[(size_t)gi*HD + k]) : __float2half(0.f);
            Qsm[i*HD + k] = qv; QTsm[k*BQ + i] = qv;
            dOsm[i*HD + k] = dv; dOTsm[k*BQ + i] = dv;
        }
        for (int idx = tid; idx < BQ; idx += NT) {
            int gi = qi0 + idx;
            Lsm[idx] = (gi < S) ? Lp[gi] : 0.f;
            Dsm[idx] = (gi < S) ? Dp[gi] : 0.f;
        }
        __syncthreads();

        // S^T = K * Q^T
        gemm<BK,BQ,HD,HD,HD,BQ,false>(Ksm, Qsm, Cfp, warp_id, lane);
        __syncthreads();

        // P^T = exp(scale*S^T - L) (causal)
        for (int e = tid; e < BK*BQ; e += NT) {
            int j = e / BQ, i = e % BQ;
            int gj = kj0 + j, gq = qi0 + i;
            bool valid = (gj < S) && (gq < S) && (gj <= gq);
            float p = valid ? __expf(Cfp[e]*scale - Lsm[i]) : 0.f;
            Ph[e] = __float2half(p);
        }
        __syncthreads();

        // dP^T = V * dO^T
        gemm<BK,BQ,HD,HD,HD,BQ,false>(Vsm, dOsm, Cfp, warp_id, lane);
        __syncthreads();

        // dS^T = P^T * (dP^T - D)
        for (int e = tid; e < BK*BQ; e += NT) {
            int i = e % BQ;
            float p = __half2float(Ph[e]);
            dSh[e] = __float2half(p * (Cfp[e] - Dsm[i]));
        }
        __syncthreads();

        // dV += P^T * dO ; dK += dS^T * Q
        gemm<BK,HD,BQ,BQ,BQ,HD,true>(Ph,  dOTsm, dVacc, warp_id, lane);
        gemm<BK,HD,BQ,BQ,BQ,HD,true>(dSh, QTsm,  dKacc, warp_id, lane);
        __syncthreads();
    }

    for (int idx = tid; idx < BK*HD; idx += NT) {
        int j = idx / HD, k = idx % HD;
        int gj = kj0 + j;
        if (gj < S) {
            dKp[(size_t)gj*HD + k] = __float2bfloat16(scale * dKacc[idx]);
            dVp[(size_t)gj*HD + k] = __float2bfloat16(dVacc[idx]);
        }
    }
}

__global__ void __launch_bounds__(NT) bwd_dq_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
    const bf16* __restrict__ dO, const float* __restrict__ Lv, const float* __restrict__ Dv,
    bf16* __restrict__ dQ, int S, float scale)
{
    int bh = blockIdx.y;
    int qb = blockIdx.x;
    int qi0 = qb*BQ;
    if (qi0 >= S) return;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    const bf16* Kp  = K  + (size_t)bh*S*HD;
    const bf16* Vp  = V  + (size_t)bh*S*HD;
    const bf16* Qp  = Q  + (size_t)bh*S*HD;
    const bf16* dOp = dO + (size_t)bh*S*HD;
    const float* Lp = Lv + (size_t)bh*S;
    const float* Dp = Dv + (size_t)bh*S;
    bf16* dQp = dQ + (size_t)bh*S*HD;

    extern __shared__ __align__(16) char smem[];
    half_t* Qsm  = (half_t*)smem;              // [BQ,HD]
    half_t* dOsm = Qsm + BQ*HD;                // [BQ,HD]
    half_t* Ksm  = dOsm + BQ*HD;               // [BK,HD]
    half_t* Vsm  = Ksm + BK*HD;                // [BK,HD]
    half_t* KTsm = Vsm + BK*HD;                // [HD,BK]
    half_t* Ph   = KTsm + HD*BK;               // [BQ,BK]
    half_t* dSh  = Ph + BQ*BK;                 // [BQ,BK]
    float* Cfp   = (float*)(dSh + BQ*BK);      // [BQ,BK]
    float* dQacc = Cfp + BQ*BK;                // [BQ,HD]
    float* Lsm   = dQacc + BQ*HD;              // [BQ]
    float* Dsm   = Lsm + BQ;                   // [BQ]

    for (int idx = tid; idx < BQ*HD; idx += NT) {
        int i = idx / HD, k = idx % HD;
        int gi = qi0 + i;
        Qsm[idx]  = (gi < S) ? bf2h(Qp[(size_t)gi*HD + k]) : __float2half(0.f);
        dOsm[idx] = (gi < S) ? bf2h(dOp[(size_t)gi*HD + k]) : __float2half(0.f);
        dQacc[idx] = 0.f;
    }
    for (int idx = tid; idx < BQ; idx += NT) {
        int gi = qi0 + idx;
        Lsm[idx] = (gi < S) ? Lp[gi] : 0.f;
        Dsm[idx] = (gi < S) ? Dp[gi] : 0.f;
    }
    __syncthreads();

    for (int kb = 0; kb <= qb; kb++) {
        int kj0 = kb*BK;
        for (int idx = tid; idx < BK*HD; idx += NT) {
            int j = idx / HD, k = idx % HD;
            int gj = kj0 + j;
            half_t kv = (gj < S) ? bf2h(Kp[(size_t)gj*HD + k]) : __float2half(0.f);
            half_t vv = (gj < S) ? bf2h(Vp[(size_t)gj*HD + k]) : __float2half(0.f);
            Ksm[j*HD + k] = kv; KTsm[k*BK + j] = kv;
            Vsm[idx] = vv;
        }
        __syncthreads();

        // S = Q * K^T
        gemm<BQ,BK,HD,HD,HD,BK,false>(Qsm, Ksm, Cfp, warp_id, lane);
        __syncthreads();

        // P = exp(scale*S - L) causal
        for (int e = tid; e < BQ*BK; e += NT) {
            int i = e / BK, j = e % BK;
            int gq = qi0 + i, gj = kj0 + j;
            bool valid = (gq < S) && (gj < S) && (gj <= gq);
            float p = valid ? __expf(Cfp[e]*scale - Lsm[i]) : 0.f;
            Ph[e] = __float2half(p);
        }
        __syncthreads();

        // dP = dO * V^T
        gemm<BQ,BK,HD,HD,HD,BK,false>(dOsm, Vsm, Cfp, warp_id, lane);
        __syncthreads();

        // dS = P * (dP - D)
        for (int e = tid; e < BQ*BK; e += NT) {
            int i = e / BK;
            float p = __half2float(Ph[e]);
            dSh[e] = __float2half(p * (Cfp[e] - Dsm[i]));
        }
        __syncthreads();

        // dQ += dS * K
        gemm<BQ,HD,BK,BK,BK,HD,true>(dSh, KTsm, dQacc, warp_id, lane);
        __syncthreads();
    }

    for (int idx = tid; idx < BQ*HD; idx += NT) {
        int i = idx / HD, k = idx % HD;
        int gi = qi0 + i;
        if (gi < S) dQp[(size_t)gi*HD + k] = __float2bfloat16(scale * dQacc[idx]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = Q.size(0), H = Q.size(1), S = Q.size(2), d = Q.size(3);
    float scale = 1.0f / sqrtf((float)d);

    const bf16* Qp  = (const bf16*)Q.data_ptr();
    const bf16* Kp  = (const bf16*)K.data_ptr();
    const bf16* Vp  = (const bf16*)V.data_ptr();
    const bf16* Op  = (const bf16*)O.data_ptr();
    const bf16* dOp = (const bf16*)dO.data_ptr();
    const float* Lp = (const float*)L.data_ptr();
    bf16* dQp = (bf16*)dQ.data_ptr();
    bf16* dKp = (bf16*)dK.data_ptr();
    bf16* dVp = (bf16*)dV.data_ptr();

    cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id);

    float* Dscr = nullptr;
    CUDA_CHECK(cudaMallocAsync((void**)&Dscr, (size_t)B*H*S*sizeof(float), stream));

    int total_rows = B*H*S;
    compute_D_kernel<<<total_rows, HD, 0, stream>>>(Op, dOp, Dscr, total_rows);
    CUDA_CHECK(cudaGetLastError());

    size_t sm_dkv = (size_t)(8*BK*HD)*sizeof(half_t)
                  + (size_t)(BK*BQ)*sizeof(float)
                  + (size_t)(2*BK*HD)*sizeof(float)
                  + (size_t)(2*BQ)*sizeof(float);
    size_t sm_dq  = (size_t)(7*BK*HD)*sizeof(half_t)
                  + (size_t)(BQ*BK)*sizeof(float)
                  + (size_t)(BQ*HD)*sizeof(float)
                  + (size_t)(2*BQ)*sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute(bwd_dkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_dkv));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel,  cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_dq));

    int numKB = (S + BK - 1)/BK;
    dim3 g1(numKB, B*H);
    bwd_dkv_kernel<<<g1, NT, sm_dkv, stream>>>(Qp, Kp, Vp, dOp, Lp, Dscr, dKp, dVp, S, scale);
    CUDA_CHECK(cudaGetLastError());

    int numQB = (S + BQ - 1)/BQ;
    dim3 g2(numQB, B*H);
    bwd_dq_kernel<<<g2, NT, sm_dq, stream>>>(Qp, Kp, Vp, dOp, Lp, Dscr, dQp, S, scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(Dscr, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd