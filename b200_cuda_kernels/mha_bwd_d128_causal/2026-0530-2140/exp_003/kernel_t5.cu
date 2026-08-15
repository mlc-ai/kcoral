#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
#include <math.h>
#include <stdio.h>
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
constexpr int LDO = Dh + 8;   // 136
constexpr int LDS = BN + 8;   // 72

// single fp32 score buffer layout
constexpr int O_sQ=0;
constexpr int O_sdO=O_sQ + BM*LDO*2;
constexpr int O_sK =O_sdO+ BM*LDO*2;
constexpr int O_sV =O_sK + BN*LDO*2;
constexpr int O_sScore =O_sV + BN*LDO*2;
constexpr int O_sPbf =O_sScore + BM*LDS*4;
constexpr int O_sdSbf=O_sPbf + BM*LDS*2;
constexpr int O_sL =O_sdSbf + BM*LDS*2;
constexpr int O_sD =O_sL + BM*4;
constexpr int SMEM = O_sD + BM*4;
constexpr int O_stage = O_sScore;  // dead after q-loop; >= BN*Dh*4 region

__device__ __forceinline__ void load_tile(const __nv_bfloat16* __restrict__ g, long base,
        int row0, int rows, int S, __nv_bfloat16* __restrict__ s, int tid){
    int total = rows*16;
    for(int i=tid;i<total;i+=THREADS){
        int r=i>>4, c=i&15; int gr=row0+r;
        int4 v;
        if(gr<S) v = *reinterpret_cast<const int4*>(g + base + (long)gr*Dh + c*8);
        else { v.x=v.y=v.z=v.w=0; }
        *reinterpret_cast<int4*>(s + r*LDO + c*8) = v;
    }
}

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
    // each lane handles 4 bf16 via int2 (vectorized): 32 lanes * 4 = 128
    int2 dv = *reinterpret_cast<const int2*>(dop + lane*4);
    int2 ov = *reinterpret_cast<const int2*>(op  + lane*4);
    __nv_bfloat16* dp=(__nv_bfloat16*)&dv; __nv_bfloat16* opp=(__nv_bfloat16*)&ov;
    #pragma unroll
    for(int j=0;j<4;j++) sum += __bfloat162float(dp[j])*__bfloat162float(opp[j]);
    for(int off=16;off>0;off>>=1) sum += __shfl_down_sync(0xffffffffu,sum,off);
    if(lane==0) Delta[row]=sum;
}

// Out[BM][LDS] = A[BM][LDO] @ B[BN][LDO]^T  (contract Dh)
__device__ __forceinline__ void gemm_ABt(const __nv_bfloat16* A, const __nv_bfloat16* B,
                                          float* Out, int warp_id){
    constexpr int nti=BM/16, ntj=BN/16, total=nti*ntj;
    for(int t=warp_id; t<total; t+=NWARPS){
        int ti=t/ntj, tj=t%ntj;
        wmma::fragment<wmma::accumulator,16,16,16,float> c;
        wmma::fill_fragment(c,0.f);
        #pragma unroll
        for(int kk=0; kk<Dh; kk+=16){
            wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
            wmma::load_matrix_sync(a, A + ti*16*LDO + kk, LDO);
            wmma::load_matrix_sync(b, B + kk + tj*16*LDO, LDO);
            wmma::mma_sync(c,a,b,c);
        }
        wmma::store_matrix_sync(Out + ti*16*LDS + tj*16, c, LDS, wmma::mem_row_major);
    }
}

// ============================= dK, dV =============================
__global__ __launch_bounds__(THREADS,2)
void dkdv_kernel(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
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
    __nv_bfloat16* sPbf =(__nv_bfloat16*)(smem+O_sPbf);
    __nv_bfloat16* sdSbf=(__nv_bfloat16*)(smem+O_sdSbf);
    float* sL=(float*)(smem+O_sL);
    float* sD=(float*)(smem+O_sD);
    float* stage=(float*)(smem+O_stage);

    int bh = blockIdx.y;
    int kv_start = blockIdx.x * BN;
    if(kv_start >= S) return;
    long base = (long)bh * S * Dh;
    int tid = threadIdx.x;
    int warp_id = tid/32;
    int rt = warp_id % 4;
    int cg = warp_id / 4;

    load_tile(K, base, kv_start, BN, S, sK, tid);
    load_tile(V, base, kv_start, BN, S, sV, tid);

    wmma::fragment<wmma::accumulator,16,16,16,float> accV[4], accK[4];
    #pragma unroll
    for(int c=0;c<4;c++){ wmma::fill_fragment(accV[c],0.f); wmma::fill_fragment(accK[c],0.f); }
    __syncthreads();

    for(int q_start=kv_start; q_start<S; q_start+=BM){
        load_tile(Q,  base, q_start, BM, S, sQ,  tid);
        load_tile(dO, base, q_start, BM, S, sdO, tid);
        for(int m=tid;m<BM;m+=THREADS){
            int qrow=q_start+m;
            if(qrow<S){ sL[m]=L[(long)bh*S+qrow]; sD[m]=Delta[(long)bh*S+qrow]; }
            else { sL[m]=0.f; sD[m]=0.f; }
        }
        __syncthreads();

        gemm_ABt(sQ, sK, sScore, warp_id);   // S = Q@K^T
        __syncthreads();
        for(int e=tid;e<BM*BN;e+=THREADS){
            int m=e/BN,n=e%BN; int qrow=q_start+m,krow=kv_start+n;
            float p=0.f;
            if(qrow<S && krow<S && krow<=qrow) p=__expf(sScore[m*LDS+n]*scale - sL[m]);
            sPbf[m*LDS+n]=__float2bfloat16(p);
        }
        __syncthreads();
        gemm_ABt(sdO, sV, sScore, warp_id);  // dP = dO@V^T (reuse buffer)
        __syncthreads();
        for(int e=tid;e<BM*BN;e+=THREADS){
            int m=e/BN,n=e%BN; int qrow=q_start+m,krow=kv_start+n;
            float ds=0.f;
            if(qrow<S && krow<S && krow<=qrow){
                float p=__bfloat162float(sPbf[m*LDS+n]);
                ds=scale*p*(sScore[m*LDS+n]-sD[m]);
            }
            sdSbf[m*LDS+n]=__float2bfloat16(ds);
        }
        __syncthreads();

        #pragma unroll
        for(int ct=0;ct<4;ct++){
            int col0 = cg*64 + ct*16;
            for(int kk=0;kk<BM;kk+=16){
                wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::col_major> aP, aS;
                wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> bO, bQ;
                wmma::load_matrix_sync(aP, sPbf  + rt*16 + kk*LDS, LDS);
                wmma::load_matrix_sync(bO, sdO   + kk*LDO + col0, LDO);
                wmma::mma_sync(accV[ct], aP, bO, accV[ct]);
                wmma::load_matrix_sync(aS, sdSbf + rt*16 + kk*LDS, LDS);
                wmma::load_matrix_sync(bQ, sQ    + kk*LDO + col0, LDO);
                wmma::mma_sync(accK[ct], aS, bQ, accK[ct]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int ct=0;ct<4;ct++)
        wmma::store_matrix_sync(stage + rt*16*Dh + cg*64 + ct*16, accV[ct], Dh, wmma::mem_row_major);
    __syncthreads();
    for(int e=tid;e<BN*Dh;e+=THREADS){
        int n=e/Dh,k=e%Dh; int krow=kv_start+n;
        if(krow<S) dV[base+(long)krow*Dh+k]=__float2bfloat16(stage[n*Dh+k]);
    }
    __syncthreads();
    #pragma unroll
    for(int ct=0;ct<4;ct++)
        wmma::store_matrix_sync(stage + rt*16*Dh + cg*64 + ct*16, accK[ct], Dh, wmma::mem_row_major);
    __syncthreads();
    for(int e=tid;e<BN*Dh;e+=THREADS){
        int n=e/Dh,k=e%Dh; int krow=kv_start+n;
        if(krow<S) dK[base+(long)krow*Dh+k]=__float2bfloat16(stage[n*Dh+k]);
    }
}

// ============================= dQ =============================
__global__ __launch_bounds__(THREADS,2)
void dq_kernel(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
               const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
               const float* __restrict__ L, const float* __restrict__ Delta,
               __nv_bfloat16* __restrict__ dQ, int S, float scale){
    extern __shared__ char smem[];
    __nv_bfloat16* sQ =(__nv_bfloat16*)(smem+O_sQ);
    __nv_bfloat16* sdO=(__nv_bfloat16*)(smem+O_sdO);
    __nv_bfloat16* sK =(__nv_bfloat16*)(smem+O_sK);
    __nv_bfloat16* sV =(__nv_bfloat16*)(smem+O_sV);
    float* sScore =(float*)(smem+O_sScore);
    __nv_bfloat16* sPbf =(__nv_bfloat16*)(smem+O_sPbf);
    __nv_bfloat16* sdSbf=(__nv_bfloat16*)(smem+O_sdSbf);
    float* sL=(float*)(smem+O_sL);
    float* sD=(float*)(smem+O_sD);
    float* stage=(float*)(smem+O_stage);

    int bh = blockIdx.y;
    int q_start = blockIdx.x * BM;
    if(q_start >= S) return;
    long base = (long)bh * S * Dh;
    int tid = threadIdx.x;
    int warp_id = tid/32;
    int rt = warp_id % 4;
    int cg = warp_id / 4;

    load_tile(Q,  base, q_start, BM, S, sQ,  tid);
    load_tile(dO, base, q_start, BM, S, sdO, tid);
    for(int m=tid;m<BM;m+=THREADS){
        int qrow=q_start+m;
        if(qrow<S){ sL[m]=L[(long)bh*S+qrow]; sD[m]=Delta[(long)bh*S+qrow]; }
        else { sL[m]=0.f; sD[m]=0.f; }
    }

    wmma::fragment<wmma::accumulator,16,16,16,float> accQ[4];
    #pragma unroll
    for(int c=0;c<4;c++) wmma::fill_fragment(accQ[c],0.f);
    __syncthreads();

    int q_end = q_start + BM - 1;
    for(int kv_start=0; kv_start<=q_end && kv_start<S; kv_start+=BN){
        load_tile(K, base, kv_start, BN, S, sK, tid);
        load_tile(V, base, kv_start, BN, S, sV, tid);
        __syncthreads();

        gemm_ABt(sQ, sK, sScore, warp_id);
        __syncthreads();
        for(int e=tid;e<BM*BN;e+=THREADS){
            int m=e/BN,n=e%BN; int qrow=q_start+m,krow=kv_start+n;
            float p=0.f;
            if(qrow<S && krow<S && krow<=qrow) p=__expf(sScore[m*LDS+n]*scale - sL[m]);
            sPbf[m*LDS+n]=__float2bfloat16(p);
        }
        __syncthreads();
        gemm_ABt(sdO, sV, sScore, warp_id);
        __syncthreads();
        for(int e=tid;e<BM*BN;e+=THREADS){
            int m=e/BN,n=e%BN; int qrow=q_start+m,krow=kv_start+n;
            float ds=0.f;
            if(qrow<S && krow<S && krow<=qrow){
                float p=__bfloat162float(sPbf[m*LDS+n]);
                ds=scale*p*(sScore[m*LDS+n]-sD[m]);
            }
            sdSbf[m*LDS+n]=__float2bfloat16(ds);
        }
        __syncthreads();

        #pragma unroll
        for(int ct=0;ct<4;ct++){
            int col0 = cg*64 + ct*16;
            for(int nn=0;nn<BN;nn+=16){
                wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b;
                wmma::load_matrix_sync(a, sdSbf + rt*16*LDS + nn, LDS);
                wmma::load_matrix_sync(b, sK + nn*LDO + col0, LDO);
                wmma::mma_sync(accQ[ct], a, b, accQ[ct]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int ct=0;ct<4;ct++)
        wmma::store_matrix_sync(stage + rt*16*Dh + cg*64 + ct*16, accQ[ct], Dh, wmma::mem_row_major);
    __syncthreads();
    for(int e=tid;e<BM*Dh;e+=THREADS){
        int m=e/Dh,k=e%Dh; int qrow=q_start+m;
        if(qrow<S) dQ[base+(long)qrow*Dh+k]=__float2bfloat16(stage[m*Dh+k]);
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

    CUDA_CHECK(cudaFuncSetAttribute(dkdv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    CUDA_CHECK(cudaFuncSetAttribute(dq_kernel,   cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));

    int num_kv = (S+BN-1)/BN;
    int num_q  = (S+BM-1)/BM;
    dim3 grid_kv(num_kv, BH);
    dim3 grid_q (num_q,  BH);

    dkdv_kernel<<<grid_kv, THREADS, SMEM, stream>>>(Qp,Kp,Vp,dOp,Lp,Delta,dKp,dVp,S,scale);
    dq_kernel  <<<grid_q,  THREADS, SMEM,   stream>>>(Qp,Kp,Vp,dOp,Lp,Delta,dQp,S,scale);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(Delta));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd