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

__device__ __forceinline__ void cp_async16(void* s, const void* g){
    uint32_t sa=(uint32_t)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(g));
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N)); }

// async load rows×Dh (contiguous in gmem) into padded smem [*][LDO]
__device__ __forceinline__ void load_tile_async(const __nv_bfloat16* __restrict__ g, long base,
        int row0, int rows, int S, __nv_bfloat16* __restrict__ s, int tid){
    int total=rows*16;
    if(row0+rows<=S){
        for(int i=tid;i<total;i+=THREADS){
            int r=i>>4,c=i&15;
            cp_async16(s+r*LDO+c*8, g+base+(long)(row0+r)*Dh+c*8);
        }
    } else {
        for(int i=tid;i<total;i+=THREADS){
            int r=i>>4,c=i&15; int gr=row0+r;
            int4 v; if(gr<S) v=*reinterpret_cast<const int4*>(g+base+(long)gr*Dh+c*8);
            else {v.x=v.y=v.z=v.w=0;}
            *reinterpret_cast<int4*>(s+r*LDO+c*8)=v;
        }
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

// dkdv smem offsets (double-buffered sQ,sdO)
constexpr int D_sQ=0;
constexpr int D_sdO=D_sQ + 2*BM*LDO*2;
constexpr int D_sK =D_sdO+ 2*BM*LDO*2;
constexpr int D_sV =D_sK + BN*LDO*2;
constexpr int D_sScore =D_sV + BN*LDO*2;
constexpr int D_sScore2=D_sScore + BM*LDS*4;
constexpr int D_sPbf =D_sScore2 + BM*LDS*4;
constexpr int D_sdSbf=D_sPbf + BM*LDS*2;
constexpr int D_sL =D_sdSbf + BM*LDS*2;
constexpr int D_sD =D_sL + BM*4;
constexpr int SMEM_DKDV = D_sD + BM*4;

__global__ void dkdv_kernel(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
                 const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
                 const float* __restrict__ L, const float* __restrict__ Delta,
                 __nv_bfloat16* __restrict__ dK, __nv_bfloat16* __restrict__ dV,
                 int S, float scale){
    extern __shared__ char smem[];
    __nv_bfloat16* sQ =(__nv_bfloat16*)(smem+D_sQ);
    __nv_bfloat16* sdO=(__nv_bfloat16*)(smem+D_sdO);
    __nv_bfloat16* sK =(__nv_bfloat16*)(smem+D_sK);
    __nv_bfloat16* sV =(__nv_bfloat16*)(smem+D_sV);
    float* sScore =(float*)(smem+D_sScore);
    float* sScore2=(float*)(smem+D_sScore2);
    __nv_bfloat16* sPbf =(__nv_bfloat16*)(smem+D_sPbf);
    __nv_bfloat16* sdSbf=(__nv_bfloat16*)(smem+D_sdSbf);
    float* sL=(float*)(smem+D_sL);
    float* sD=(float*)(smem+D_sD);
    float* stage=(float*)(smem+D_sScore);

    int bh = blockIdx.y;
    int kv_start = blockIdx.x * BN;
    if(kv_start >= S) return;
    long base = (long)bh * S * Dh;
    int tid = threadIdx.x;
    int warp_id = tid/32;
    int rt = warp_id % 4;
    int cg = warp_id / 4;

    load_tile_async(K, base, kv_start, BN, S, sK, tid);
    load_tile_async(V, base, kv_start, BN, S, sV, tid);
    cp_commit();

    wmma::fragment<wmma::accumulator,16,16,16,float> accV[4], accK[4];
    #pragma unroll
    for(int c=0;c<4;c++){ wmma::fill_fragment(accV[c],0.f); wmma::fill_fragment(accK[c],0.f); }

    int niter = (S - kv_start + BM - 1)/BM;
    int cur=0;
    // prologue: load q tile 0
    load_tile_async(Q,  base, kv_start, BM, S, sQ,  tid);
    load_tile_async(dO, base, kv_start, BM, S, sdO, tid);
    cp_commit();

    for(int it=0; it<niter; it++){
        int q_start = kv_start + it*BM;
        int qbuf = cur*(BM*LDO);
        int nbuf = (cur^1)*(BM*LDO);
        if(it+1<niter){
            int qn = kv_start + (it+1)*BM;
            load_tile_async(Q,  base, qn, BM, S, sQ + nbuf,  tid);
            load_tile_async(dO, base, qn, BM, S, sdO + nbuf, tid);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        for(int m=tid;m<BM;m+=THREADS){
            int qr=q_start+m;
            if(qr<S){ sL[m]=L[(long)bh*S+qr]; sD[m]=Delta[(long)bh*S+qr]; }
            else { sL[m]=0.f; sD[m]=0.f; }
        }
        __syncthreads();

        __nv_bfloat16* sQc=sQ+qbuf;
        __nv_bfloat16* sdOc=sdO+qbuf;

        gemm_ABt(sQc, sK, sScore, warp_id);    // S = Q@K^T
        gemm_ABt(sdOc, sV, sScore2, warp_id);  // dP = dO@V^T
        __syncthreads();

        for(int e=tid;e<BM*BN;e+=THREADS){
            int m=e/BN,n=e%BN; int qr=q_start+m,kr=kv_start+n;
            float p=0.f, ds=0.f;
            if(qr<S && kr<S && kr<=qr){
                p=__expf(sScore[m*LDS+n]*scale - sL[m]);
                ds=scale*p*(sScore2[m*LDS+n]-sD[m]);
            }
            sPbf[m*LDS+n]=__float2bfloat16(p);
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
                wmma::load_matrix_sync(bO, sdOc  + kk*LDO + col0, LDO);
                wmma::mma_sync(accV[ct], aP, bO, accV[ct]);
                wmma::load_matrix_sync(aS, sdSbf + rt*16 + kk*LDS, LDS);
                wmma::load_matrix_sync(bQ, sQc   + kk*LDO + col0, LDO);
                wmma::mma_sync(accK[ct], aS, bQ, accK[ct]);
            }
        }
        __syncthreads();
        cur^=1;
    }

    #pragma unroll
    for(int ct=0;ct<4;ct++)
        wmma::store_matrix_sync(stage + rt*16*Dh + cg*64 + ct*16, accV[ct], Dh, wmma::mem_row_major);
    __syncthreads();
    for(int e=tid;e<BN*Dh;e+=THREADS){
        int n=e/Dh,k=e%Dh; int kr=kv_start+n;
        if(kr<S) dV[base+(long)kr*Dh+k]=__float2bfloat16(stage[n*Dh+k]);
    }
    __syncthreads();
    #pragma unroll
    for(int ct=0;ct<4;ct++)
        wmma::store_matrix_sync(stage + rt*16*Dh + cg*64 + ct*16, accK[ct], Dh, wmma::mem_row_major);
    __syncthreads();
    for(int e=tid;e<BN*Dh;e+=THREADS){
        int n=e/Dh,k=e%Dh; int kr=kv_start+n;
        if(kr<S) dK[base+(long)kr*Dh+k]=__float2bfloat16(stage[n*Dh+k]);
    }
}

// dq smem offsets (double-buffered sK,sV)
constexpr int Q_sQ=0;
constexpr int Q_sdO=Q_sQ + BM*LDO*2;
constexpr int Q_sK =Q_sdO+ BM*LDO*2;
constexpr int Q_sV =Q_sK + 2*BN*LDO*2;
constexpr int Q_sScore =Q_sV + 2*BN*LDO*2;
constexpr int Q_sScore2=Q_sScore + BM*LDS*4;
constexpr int Q_sdSbf=Q_sScore2 + BM*LDS*4;
constexpr int Q_sL =Q_sdSbf + BM*LDS*2;
constexpr int Q_sD =Q_sL + BM*4;
constexpr int SMEM_DQ = Q_sD + BM*4;

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
    float* sL=(float*)(smem+Q_sL);
    float* sD=(float*)(smem+Q_sD);
    float* stage=(float*)(smem+Q_sScore);

    int bh = blockIdx.y;
    int q_start = blockIdx.x * BM;
    if(q_start >= S) return;
    long base = (long)bh * S * Dh;
    int tid = threadIdx.x;
    int warp_id = tid/32;
    int rt = warp_id % 4;
    int cg = warp_id / 4;

    load_tile_async(Q,  base, q_start, BM, S, sQ,  tid);
    load_tile_async(dO, base, q_start, BM, S, sdO, tid);
    cp_commit();
    for(int m=tid;m<BM;m+=THREADS){
        int qr=q_start+m;
        if(qr<S){ sL[m]=L[(long)bh*S+qr]; sD[m]=Delta[(long)bh*S+qr]; }
        else { sL[m]=0.f; sD[m]=0.f; }
    }

    wmma::fragment<wmma::accumulator,16,16,16,float> accQ[4];
    #pragma unroll
    for(int c=0;c<4;c++) wmma::fill_fragment(accQ[c],0.f);

    int q_end = q_start + BM - 1;
    int last_kv = q_end < S-1 ? q_end : S-1;
    int nkv = last_kv/BN + 1;
    int cur=0;

    // prologue kv tile 0
    load_tile_async(K, base, 0, BN, S, sK, tid);
    load_tile_async(V, base, 0, BN, S, sV, tid);
    cp_commit();

    for(int it=0; it<nkv; it++){
        int kv_start = it*BN;
        int kbuf = cur*(BN*LDO);
        int nbuf = (cur^1)*(BN*LDO);
        if(it+1<nkv){
            int kn=(it+1)*BN;
            load_tile_async(K, base, kn, BN, S, sK + nbuf, tid);
            load_tile_async(V, base, kn, BN, S, sV + nbuf, tid);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        __nv_bfloat16* sKc=sK+kbuf;
        __nv_bfloat16* sVc=sV+kbuf;

        gemm_ABt(sQ, sKc, sScore, warp_id);
        gemm_ABt(sdO, sVc, sScore2, warp_id);
        __syncthreads();

        for(int e=tid;e<BM*BN;e+=THREADS){
            int m=e/BN,n=e%BN; int qr=q_start+m,kr=kv_start+n;
            float ds=0.f;
            if(qr<S && kr<S && kr<=qr){
                float p=__expf(sScore[m*LDS+n]*scale - sL[m]);
                ds=scale*p*(sScore2[m*LDS+n]-sD[m]);
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
                wmma::load_matrix_sync(b, sKc + nn*LDO + col0, LDO);
                wmma::mma_sync(accQ[ct], a, b, accQ[ct]);
            }
        }
        __syncthreads();
        cur^=1;
    }

    #pragma unroll
    for(int ct=0;ct<4;ct++)
        wmma::store_matrix_sync(stage + rt*16*Dh + cg*64 + ct*16, accQ[ct], Dh, wmma::mem_row_major);
    __syncthreads();
    for(int e=tid;e<BM*Dh;e+=THREADS){
        int m=e/Dh,k=e%Dh; int qr=q_start+m;
        if(qr<S) dQ[base+(long)qr*Dh+k]=__float2bfloat16(stage[m*Dh+k]);
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