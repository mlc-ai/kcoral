#include <cuda_bf16.h>
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

#define HD 128
#define BK 64
#define BQ 64
#define NT 128
#define NWARPS 4

__device__ __forceinline__ uint32_t ld2(const bf16* p){
    return *reinterpret_cast<const uint32_t*>(p);
}

// bf16 x bf16 -> f32.  C[M,N] += A[M,K]*B[N,K]  (contraction over K)
template<int M,int N,int K,int LDA,int LDB,int LDC,bool ACC>
__device__ __forceinline__ void gemm_bf16(const bf16* A, const bf16* B, float* C, int warp_id, int lane){
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
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
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

// TF32: A is fp32 (truncated to tf32), B is bf16 (upconverted). C[M,N]+=A[M,K]*B[N,K]
template<int M,int N,int K,int LDA,int LDB,int LDC,bool ACC>
__device__ __forceinline__ void gemm_tf32(const float* A, const bf16* B, float* C, int warp_id, int lane){
    const int ROWS = M / NWARPS;
    int group = lane >> 2;
    int tid = lane & 3;
    #pragma unroll
    for (int rm = 0; rm < ROWS; rm += 16) {
        int mrow = warp_id * ROWS + rm;
        #pragma unroll
        for (int nn = 0; nn < N; nn += 8) {
            float c0,c1,c2,c3;
            if (ACC) {
                c0 = C[(mrow+group)*LDC + nn + 2*tid];
                c1 = C[(mrow+group)*LDC + nn + 2*tid + 1];
                c2 = C[(mrow+group+8)*LDC + nn + 2*tid];
                c3 = C[(mrow+group+8)*LDC + nn + 2*tid + 1];
            } else { c0=c1=c2=c3=0.f; }
            #pragma unroll
            for (int kk = 0; kk < K; kk += 8) {
                uint32_t a0 = __float_as_uint(A[(mrow+group)*LDA + kk + tid]);
                uint32_t a1 = __float_as_uint(A[(mrow+group+8)*LDA + kk + tid]);
                uint32_t a2 = __float_as_uint(A[(mrow+group)*LDA + kk + tid + 4]);
                uint32_t a3 = __float_as_uint(A[(mrow+group+8)*LDA + kk + tid + 4]);
                uint32_t b0 = __float_as_uint(__bfloat162float(B[(nn+group)*LDB + kk + tid]));
                uint32_t b1 = __float_as_uint(__bfloat162float(B[(nn+group)*LDB + kk + tid + 4]));
                asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
            C[(mrow+group)*LDC + nn + 2*tid]       = c0;
            C[(mrow+group)*LDC + nn + 2*tid + 1]   = c1;
            C[(mrow+group+8)*LDC + nn + 2*tid]     = c2;
            C[(mrow+group+8)*LDC + nn + 2*tid + 1] = c3;
        }
    }
}

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
    bf16* Ksm   = (bf16*)smem;                 // [BK,HD]
    bf16* Vsm   = Ksm + BK*HD;                 // [BK,HD]
    bf16* Qsm   = Vsm + BK*HD;                 // [BQ,HD]
    bf16* QTsm  = Qsm + BQ*HD;                 // [HD,BQ]
    bf16* dOsm  = QTsm + HD*BQ;                // [BQ,HD]
    bf16* dOTsm = dOsm + BQ*HD;                // [HD,BQ]
    float* Cfp  = (float*)(dOTsm + HD*BQ);     // [BK,BQ]
    float* Pf   = Cfp + BK*BQ;                 // [BK,BQ]
    float* dSf  = Pf + BK*BQ;                  // [BK,BQ]
    float* dVacc= dSf + BK*BQ;                 // [BK,HD]
    float* dKacc= dVacc + BK*HD;               // [BK,HD]
    float* Lsm  = dKacc + BK*HD;               // [BQ]
    float* Dsm  = Lsm + BQ;                    // [BQ]

    for (int idx = tid; idx < BK*HD; idx += NT) {
        int j = idx / HD, k = idx % HD;
        int gj = kj0 + j;
        bf16 kv = (gj < S) ? Kp[(size_t)gj*HD + k] : __float2bfloat16(0.f);
        bf16 vv = (gj < S) ? Vp[(size_t)gj*HD + k] : __float2bfloat16(0.f);
        Ksm[idx] = kv; Vsm[idx] = vv;
        dVacc[idx] = 0.f; dKacc[idx] = 0.f;
    }
    __syncthreads();

    int numQB = (S + BQ - 1)/BQ;
    for (int qb = kb; qb < numQB; qb++) {
        int qi0 = qb*BQ;
        for (int idx = tid; idx < BQ*HD; idx += NT) {
            int i = idx / HD, k = idx % HD;
            int gi = qi0 + i;
            bf16 qv = (gi < S) ? Qp[(size_t)gi*HD + k] : __float2bfloat16(0.f);
            bf16 dv = (gi < S) ? dOp[(size_t)gi*HD + k] : __float2bfloat16(0.f);
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
        gemm_bf16<BK,BQ,HD,HD,HD,BQ,false>(Ksm, Qsm, Cfp, warp_id, lane);
        __syncthreads();

        // P^T = exp(scale*S^T - L)  (causal)
        for (int e = tid; e < BK*BQ; e += NT) {
            int j = e / BQ, i = e % BQ;
            int gj = kj0 + j, gq = qi0 + i;
            bool valid = (gj < S) && (gq < S) && (gj <= gq);
            Pf[e] = valid ? __expf(Cfp[e]*scale - Lsm[i]) : 0.f;
        }
        __syncthreads();

        // dP^T = V * dO^T
        gemm_bf16<BK,BQ,HD,HD,HD,BQ,false>(Vsm, dOsm, Cfp, warp_id, lane);
        __syncthreads();

        // dS^T = P^T * (dP^T - D)
        for (int e = tid; e < BK*BQ; e += NT) {
            int i = e % BQ;
            dSf[e] = Pf[e] * (Cfp[e] - Dsm[i]);
        }
        __syncthreads();

        // dV += P^T * dO ; dK += dS^T * Q
        gemm_tf32<BK,HD,BQ,BQ,BQ,HD,true>(Pf,  dOTsm, dVacc, warp_id, lane);
        gemm_tf32<BK,HD,BQ,BQ,BQ,HD,true>(dSf, QTsm,  dKacc, warp_id, lane);
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
    bf16* Qsm  = (bf16*)smem;              // [BQ,HD]
    bf16* dOsm = Qsm + BQ*HD;              // [BQ,HD]
    bf16* Ksm  = dOsm + BQ*HD;             // [BK,HD]
    bf16* Vsm  = Ksm + BK*HD;              // [BK,HD]
    bf16* KTsm = Vsm + BK*HD;              // [HD,BK]
    float* Cfp = (float*)(KTsm + HD*BK);   // [BQ,BK]
    float* Pf  = Cfp + BQ*BK;              // [BQ,BK]
    float* dSf = Pf + BQ*BK;               // [BQ,BK]
    float* dQacc = dSf + BQ*BK;            // [BQ,HD]
    float* Lsm = dQacc + BQ*HD;            // [BQ]
    float* Dsm = Lsm + BQ;                 // [BQ]

    for (int idx = tid; idx < BQ*HD; idx += NT) {
        int i = idx / HD, k = idx % HD;
        int gi = qi0 + i;
        Qsm[idx]  = (gi < S) ? Qp[(size_t)gi*HD + k] : __float2bfloat16(0.f);
        dOsm[idx] = (gi < S) ? dOp[(size_t)gi*HD + k] : __float2bfloat16(0.f);
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
            bf16 kv = (gj < S) ? Kp[(size_t)gj*HD + k] : __float2bfloat16(0.f);
            bf16 vv = (gj < S) ? Vp[(size_t)gj*HD + k] : __float2bfloat16(0.f);
            Ksm[j*HD + k] = kv; KTsm[k*BK + j] = kv;
            Vsm[idx] = vv;
        }
        __syncthreads();

        // S = Q * K^T
        gemm_bf16<BQ,BK,HD,HD,HD,BK,false>(Qsm, Ksm, Cfp, warp_id, lane);
        __syncthreads();

        // P = exp(scale*S - L) causal
        for (int e = tid; e < BQ*BK; e += NT) {
            int i = e / BK, j = e % BK;
            int gq = qi0 + i, gj = kj0 + j;
            bool valid = (gq < S) && (gj < S) && (gj <= gq);
            Pf[e] = valid ? __expf(Cfp[e]*scale - Lsm[i]) : 0.f;
        }
        __syncthreads();

        // dP = dO * V^T
        gemm_bf16<BQ,BK,HD,HD,HD,BK,false>(dOsm, Vsm, Cfp, warp_id, lane);
        __syncthreads();

        // dS = P * (dP - D)
        for (int e = tid; e < BQ*BK; e += NT) {
            int i = e / BK;
            dSf[e] = Pf[e] * (Cfp[e] - Dsm[i]);
        }
        __syncthreads();

        // dQ += dS * K
        gemm_tf32<BQ,HD,BK,BK,BK,HD,true>(dSf, KTsm, dQacc, warp_id, lane);
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

    size_t sm_dkv = (size_t)(6*BK*HD)*sizeof(bf16)
                  + (size_t)(3*BK*BQ)*sizeof(float)
                  + (size_t)(2*BK*HD)*sizeof(float)
                  + (size_t)(2*BQ)*sizeof(float);
    size_t sm_dq  = (size_t)(5*BK*HD)*sizeof(bf16)
                  + (size_t)(3*BQ*BK)*sizeof(float)
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