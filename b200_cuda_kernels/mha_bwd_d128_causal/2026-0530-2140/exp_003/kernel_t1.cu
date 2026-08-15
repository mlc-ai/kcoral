#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
#include <math.h>
#include <stdio.h>
#include <type_traits>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)

using namespace nvcuda;

namespace mha_bwd {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int Dh = 128;
constexpr int THREADS = 256;
constexpr int NWARPS = THREADS/32;

// Generic WMMA gemm: C(MxN) (+=) A(MxK) @ B(KxN), C stored row-major fp32 in shared.
template<typename LA, typename LB>
__device__ __forceinline__ void wmma_gemm(const __nv_bfloat16* A,int lda,
        const __nv_bfloat16* B,int ldb, float* C,int ldc,
        int M,int N,int K,int warp_id,bool accum){
    constexpr bool A_row = std::is_same<LA,wmma::row_major>::value;
    constexpr bool B_row = std::is_same<LB,wmma::row_major>::value;
    int ntj=N/16, nti=M/16, total=nti*ntj;
    for(int t=warp_id; t<total; t+=NWARPS){
        int ti=t/ntj, tj=t%ntj;
        wmma::fragment<wmma::accumulator,16,16,16,float> c;
        if(accum) wmma::load_matrix_sync(c, C+(ti*16)*ldc+tj*16, ldc, wmma::mem_row_major);
        else wmma::fill_fragment(c,0.f);
        for(int kk=0; kk<K; kk+=16){
            wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,LA> a;
            wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,LB> b;
            const __nv_bfloat16* aptr = A_row ? (A+(ti*16)*lda+kk) : (A+(ti*16)+kk*lda);
            const __nv_bfloat16* bptr = B_row ? (B+kk*ldb+tj*16) : (B+kk+(tj*16)*ldb);
            wmma::load_matrix_sync(a,aptr,lda);
            wmma::load_matrix_sync(b,bptr,ldb);
            wmma::mma_sync(c,a,b,c);
        }
        wmma::store_matrix_sync(C+(ti*16)*ldc+tj*16, c, ldc, wmma::mem_row_major);
    }
}

// ---- Delta = rowsum(dO * O) ----
__global__ void compute_delta_kernel(const __nv_bfloat16* __restrict__ dO,
                                     const __nv_bfloat16* __restrict__ O,
                                     float* __restrict__ Delta, long total_rows){
    int wpb = blockDim.x/32;
    long row = (long)blockIdx.x * wpb + threadIdx.x/32;
    int lane = threadIdx.x & 31;
    if(row >= total_rows) return;
    const __nv_bfloat16* dop = dO + row*Dh;
    const __nv_bfloat16* op  = O  + row*Dh;
    float sum=0.f;
    for(int k=lane;k<Dh;k+=32)
        sum += __bfloat162float(dop[k]) * __bfloat162float(op[k]);
    for(int off=16;off>0;off>>=1) sum += __shfl_down_sync(0xffffffffu,sum,off);
    if(lane==0) Delta[row]=sum;
}

// shared layout offsets (bytes) for dkdv
constexpr int O_sQ=0;
constexpr int O_sdO=O_sQ+BM*Dh*2;
constexpr int O_sK=O_sdO+BM*Dh*2;
constexpr int O_sV=O_sK+BN*Dh*2;
constexpr int O_sScore=O_sV+BN*Dh*2;
constexpr int O_sScore2=O_sScore+BM*BN*4;
constexpr int O_sPbf=O_sScore2+BM*BN*4;
constexpr int O_sdSbf=O_sPbf+BM*BN*2;
constexpr int O_dVacc=O_sdSbf+BM*BN*2;
constexpr int O_dKacc=O_dVacc+BN*Dh*4;
constexpr int O_sL=O_dKacc+BN*Dh*4;
constexpr int O_sD=O_sL+BM*4;
constexpr int SMEM_DKDV=O_sD+BM*4;

__global__ void dkdv_kernel(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
                            const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
                            const float* __restrict__ L, const float* __restrict__ Delta,
                            __nv_bfloat16* __restrict__ dK, __nv_bfloat16* __restrict__ dV,
                            int S, float scale){
    extern __shared__ char smem[];
    __nv_bfloat16* sQ =(__nv_bfloat16*)(smem+O_sQ);
    __nv_bfloat16* sdO=(__nv_bfloat16*)(smem+O_sdO);
    __nv_bfloat16* sK =(__nv_bfloat16*)(smem+O_sK);
    __nv_bfloat16* sV =(__nv_bfloat16*)(smem+O_sV);
    float* sScore =(float*)(smem+O_sScore);
    float* sScore2=(float*)(smem+O_sScore2);
    __nv_bfloat16* sPbf =(__nv_bfloat16*)(smem+O_sPbf);
    __nv_bfloat16* sdSbf=(__nv_bfloat16*)(smem+O_sdSbf);
    float* dVacc=(float*)(smem+O_dVacc);
    float* dKacc=(float*)(smem+O_dKacc);
    float* sL=(float*)(smem+O_sL);
    float* sD=(float*)(smem+O_sD);

    int bh = blockIdx.y;
    int kv_start = blockIdx.x * BN;
    if(kv_start >= S) return;
    long base = (long)bh * S * Dh;
    int tid = threadIdx.x;
    int warp_id = tid/32;

    for(int e=tid;e<BN*Dh;e+=THREADS){
        int n=e/Dh,k=e%Dh; int krow=kv_start+n;
        if(krow<S){ sK[e]=K[base+(long)krow*Dh+k]; sV[e]=V[base+(long)krow*Dh+k]; }
        else { sK[e]=__float2bfloat16(0.f); sV[e]=__float2bfloat16(0.f); }
    }
    for(int e=tid;e<BN*Dh;e+=THREADS){ dVacc[e]=0.f; dKacc[e]=0.f; }
    __syncthreads();

    int q_start0 = kv_start; // BM==BN, aligned
    for(int q_start=q_start0; q_start<S; q_start+=BM){
        for(int e=tid;e<BM*Dh;e+=THREADS){
            int m=e/Dh,k=e%Dh; int qrow=q_start+m;
            if(qrow<S){ sQ[e]=Q[base+(long)qrow*Dh+k]; sdO[e]=dO[base+(long)qrow*Dh+k]; }
            else { sQ[e]=__float2bfloat16(0.f); sdO[e]=__float2bfloat16(0.f); }
        }
        for(int m=tid;m<BM;m+=THREADS){
            int qrow=q_start+m;
            if(qrow<S){ sL[m]=L[(long)bh*S+qrow]; sD[m]=Delta[(long)bh*S+qrow]; }
            else { sL[m]=0.f; sD[m]=0.f; }
        }
        __syncthreads();

        // S = Q @ K^T
        wmma_gemm<wmma::row_major,wmma::col_major>(sQ,Dh,sK,Dh,sScore,BN,BM,BN,Dh,warp_id,false);
        // dP = dO @ V^T
        wmma_gemm<wmma::row_major,wmma::col_major>(sdO,Dh,sV,Dh,sScore2,BN,BM,BN,Dh,warp_id,false);
        __syncthreads();

        // elementwise
        for(int e=tid;e<BM*BN;e+=THREADS){
            int m=e/BN,n=e%BN; int qrow=q_start+m,krow=kv_start+n;
            float pval=0.f, dsval=0.f;
            if(qrow<S && krow<S && krow<=qrow){
                float p=__expf(sScore[e]*scale - sL[m]);
                pval=p;
                dsval=scale*p*(sScore2[e]-sD[m]);
            }
            sPbf[e]=__float2bfloat16(pval);
            sdSbf[e]=__float2bfloat16(dsval);
        }
        __syncthreads();

        // dV += P^T @ dO ; dK += dS^T @ Q
        wmma_gemm<wmma::col_major,wmma::row_major>(sPbf,BN,sdO,Dh,dVacc,Dh,BN,Dh,BM,warp_id,true);
        wmma_gemm<wmma::col_major,wmma::row_major>(sdSbf,BN,sQ,Dh,dKacc,Dh,BN,Dh,BM,warp_id,true);
        __syncthreads();
    }

    for(int e=tid;e<BN*Dh;e+=THREADS){
        int n=e/Dh,k=e%Dh; int krow=kv_start+n;
        if(krow<S){
            dV[base+(long)krow*Dh+k]=__float2bfloat16(dVacc[e]);
            dK[base+(long)krow*Dh+k]=__float2bfloat16(dKacc[e]);
        }
    }
}

// shared layout for dq
constexpr int Q_sQ=0;
constexpr int Q_sdO=Q_sQ+BM*Dh*2;
constexpr int Q_sK=Q_sdO+BM*Dh*2;
constexpr int Q_sV=Q_sK+BN*Dh*2;
constexpr int Q_sScore=Q_sV+BN*Dh*2;
constexpr int Q_sScore2=Q_sScore+BM*BN*4;
constexpr int Q_sdSbf=Q_sScore2+BM*BN*4;
constexpr int Q_dQacc=Q_sdSbf+BM*BN*2;
constexpr int Q_sL=Q_dQacc+BM*Dh*4;
constexpr int Q_sD=Q_sL+BM*4;
constexpr int SMEM_DQ=Q_sD+BM*4;

__global__ void dq_kernel(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
                          const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
                          const float* __restrict__ L, const float* __restrict__ Delta,
                          __nv_bfloat16* __restrict__ dQ, int S, float scale){
    extern __shared__ char smem[];
    __nv_bfloat16* sQ =(__nv_bfloat16*)(smem+Q_sQ);
    __nv_bfloat16* sdO=(__nv_bfloat16*)(smem+Q_sdO);
    __nv_bfloat16* sK =(__nv_bfloat16*)(smem+Q_sK);
    __nv_bfloat16* sV =(__nv_bfloat16*)(smem+Q_sV);
    float* sScore =(float*)(smem+Q_sScore);
    float* sScore2=(float*)(smem+Q_sScore2);
    __nv_bfloat16* sdSbf=(__nv_bfloat16*)(smem+Q_sdSbf);
    float* dQacc=(float*)(smem+Q_dQacc);
    float* sL=(float*)(smem+Q_sL);
    float* sD=(float*)(smem+Q_sD);

    int bh = blockIdx.y;
    int q_start = blockIdx.x * BM;
    if(q_start >= S) return;
    long base = (long)bh * S * Dh;
    int tid = threadIdx.x;
    int warp_id = tid/32;

    for(int e=tid;e<BM*Dh;e+=THREADS){
        int m=e/Dh,k=e%Dh; int qrow=q_start+m;
        if(qrow<S){ sQ[e]=Q[base+(long)qrow*Dh+k]; sdO[e]=dO[base+(long)qrow*Dh+k]; }
        else { sQ[e]=__float2bfloat16(0.f); sdO[e]=__float2bfloat16(0.f); }
    }
    for(int m=tid;m<BM;m+=THREADS){
        int qrow=q_start+m;
        if(qrow<S){ sL[m]=L[(long)bh*S+qrow]; sD[m]=Delta[(long)bh*S+qrow]; }
        else { sL[m]=0.f; sD[m]=0.f; }
    }
    for(int e=tid;e<BM*Dh;e+=THREADS) dQacc[e]=0.f;
    __syncthreads();

    int q_end = q_start + BM - 1;
    for(int kv_start=0; kv_start<=q_end && kv_start<S; kv_start+=BN){
        for(int e=tid;e<BN*Dh;e+=THREADS){
            int n=e/Dh,k=e%Dh; int krow=kv_start+n;
            if(krow<S){ sK[e]=K[base+(long)krow*Dh+k]; sV[e]=V[base+(long)krow*Dh+k]; }
            else { sK[e]=__float2bfloat16(0.f); sV[e]=__float2bfloat16(0.f); }
        }
        __syncthreads();

        wmma_gemm<wmma::row_major,wmma::col_major>(sQ,Dh,sK,Dh,sScore,BN,BM,BN,Dh,warp_id,false);
        wmma_gemm<wmma::row_major,wmma::col_major>(sdO,Dh,sV,Dh,sScore2,BN,BM,BN,Dh,warp_id,false);
        __syncthreads();

        for(int e=tid;e<BM*BN;e+=THREADS){
            int m=e/BN,n=e%BN; int qrow=q_start+m,krow=kv_start+n;
            float dsval=0.f;
            if(qrow<S && krow<S && krow<=qrow){
                float p=__expf(sScore[e]*scale - sL[m]);
                dsval=scale*p*(sScore2[e]-sD[m]);
            }
            sdSbf[e]=__float2bfloat16(dsval);
        }
        __syncthreads();

        // dQ += dS @ K
        wmma_gemm<wmma::row_major,wmma::row_major>(sdSbf,BN,sK,Dh,dQacc,Dh,BM,Dh,BN,warp_id,true);
        __syncthreads();
    }

    for(int e=tid;e<BM*Dh;e+=THREADS){
        int m=e/Dh,k=e%Dh; int qrow=q_start+m;
        if(qrow<S) dQ[base+(long)qrow*Dh+k]=__float2bfloat16(dQacc[e]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV){
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int Bsz = Q.size(0), H = Q.size(1), S = Q.size(2), d = Q.size(3);
    int BH = Bsz*H;
    float scale = 1.0f/sqrtf((float)d);

    const __nv_bfloat16* Qp =(const __nv_bfloat16*)Q.data_ptr();
    const __nv_bfloat16* Kp =(const __nv_bfloat16*)K.data_ptr();
    const __nv_bfloat16* Vp =(const __nv_bfloat16*)V.data_ptr();
    const __nv_bfloat16* Op =(const __nv_bfloat16*)O.data_ptr();
    const __nv_bfloat16* dOp=(const __nv_bfloat16*)dO.data_ptr();
    const float* Lp =(const float*)L.data_ptr();
    __nv_bfloat16* dQp=(__nv_bfloat16*)dQ.data_ptr();
    __nv_bfloat16* dKp=(__nv_bfloat16*)dK.data_ptr();
    __nv_bfloat16* dVp=(__nv_bfloat16*)dV.data_ptr();

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* Delta=nullptr;
    CUDA_CHECK(cudaMalloc(&Delta, (size_t)BH*S*sizeof(float)));

    long total_rows = (long)BH*S;
    int wpb = 8;
    long blocks_delta = (total_rows + wpb - 1)/wpb;
    compute_delta_kernel<<<(unsigned)blocks_delta, wpb*32, 0, stream>>>(dOp, Op, Delta, total_rows);

    CUDA_CHECK(cudaFuncSetAttribute(dkdv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_DKDV));
    CUDA_CHECK(cudaFuncSetAttribute(dq_kernel,   cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_DQ));

    int num_kv = (S+BN-1)/BN;
    int num_q  = (S+BM-1)/BM;
    dim3 grid_kv(num_kv, BH);
    dim3 grid_q (num_q,  BH);

    dkdv_kernel<<<grid_kv, THREADS, SMEM_DKDV, stream>>>(Qp,Kp,Vp,dOp,Lp,Delta,dKp,dVp,S,scale);
    dq_kernel  <<<grid_q,  THREADS, SMEM_DQ,   stream>>>(Qp,Kp,Vp,dOp,Lp,Delta,dQp,S,scale);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(Delta));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd