#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
#include <type_traits>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;
using bf16 = __nv_bfloat16;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd {

constexpr int D  = 128;
constexpr int BR = 64;
constexpr int BC = 64;
constexpr int NW = 8;
constexpr int NT_THREADS = NW*32;
constexpr int LDT = D + 8;
constexpr int LDS = BC + 8;
constexpr int LDB = BC + 8;
constexpr int DT16 = D/16;

template<int MT,int NT,int KT,typename AL,typename BL>
__device__ __forceinline__ void score_gemm(const bf16*A,int lda,const bf16*B,int ldb,
                                           float*C,int ldc,int warp_id){
  constexpr int MTIL=MT/16, NTIL=NT/16, NSUB=MTIL*NTIL;
  for(int s=warp_id;s<NSUB;s+=NW){
    int mt=s/NTIL, nt=s%NTIL, m0=mt*16,n0=nt*16;
    wmma::fragment<wmma::accumulator,16,16,16,float> cf;
    wmma::fill_fragment(cf,0.f);
    #pragma unroll
    for(int k0=0;k0<KT;k0+=16){
      wmma::fragment<wmma::matrix_a,16,16,16,bf16,AL> af;
      wmma::fragment<wmma::matrix_b,16,16,16,bf16,BL> bfr;
      const bf16* ap; if constexpr(std::is_same<AL,wmma::row_major>::value) ap=A+m0*lda+k0; else ap=A+m0+k0*lda;
      const bf16* bp; if constexpr(std::is_same<BL,wmma::row_major>::value) bp=B+k0*ldb+n0; else bp=B+k0+n0*ldb;
      wmma::load_matrix_sync(af,ap,lda);
      wmma::load_matrix_sync(bfr,bp,ldb);
      wmma::mma_sync(cf,af,bfr,cf);
    }
    wmma::store_matrix_sync(C+m0*ldc+n0,cf,ldc,wmma::mem_row_major);
  }
}

template<int ROWS>
__device__ __forceinline__ void load_tile(bf16* smem, const bf16* g, int row0, int S,
                                          int tid, int nthreads){
    constexpr int VEC=8, per_row=D/VEC, total=ROWS*per_row;
    for(int idx=tid; idx<total; idx+=nthreads){
        int r=idx/per_row, cv=idx%per_row;
        int gr=row0+r;
        int4 val;
        if(gr<S) val=*reinterpret_cast<const int4*>(g+(long)gr*D+cv*VEC);
        else     val=make_int4(0,0,0,0);
        *reinterpret_cast<int4*>(smem+r*LDT+cv*VEC)=val;
    }
}

__global__ void compute_D_kernel(const bf16* O, const bf16* dO, float* Dg, int total){
    int wpb = blockDim.x/32;
    int row = blockIdx.x*wpb + (threadIdx.x/32);
    if(row>=total) return;
    int lane=threadIdx.x%32;
    const bf16* Op=O+(long)row*D; const bf16* dOp=dO+(long)row*D;
    float acc=0.f;
    #pragma unroll
    for(int k=lane;k<D;k+=32) acc+=__bfloat162float(Op[k])*__bfloat162float(dOp[k]);
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
    if(lane==0) Dg[row]=acc;
}

template<int NACC>
__device__ __forceinline__ void epilogue(
    wmma::fragment<wmma::accumulator,16,16,16,float>* frag,
    float* scr, bf16* Gout, int row0, int S, int warp_id, int lane, float sc)
{
    float* wscr = scr + warp_id*256;
    #pragma unroll
    for(int i=0;i<NACC;i++){
        int tile=warp_id+i*NW; int mt=tile/DT16, nt=tile%DT16;
        wmma::store_matrix_sync(wscr, frag[i], 16, wmma::mem_row_major);
        __syncwarp();
        for(int e=lane;e<256;e+=32){
            int rr=e/16, cc=e%16;
            int gj=row0+mt*16+rr; int k=nt*16+cc;
            if(gj<S) Gout[(long)gj*D+k]=__float2bfloat16(wscr[e]*sc);
        }
        __syncwarp();
    }
}

// ---------------- Kernel 1: dK, dV ----------------
__global__ void bwd_kv_kernel(
    const bf16* Q, const bf16* K, const bf16* V, const bf16* dO,
    const float* Lg, const float* Dg, bf16* dK, bf16* dV, int S, float scale)
{
    int bh=blockIdx.x; int kvrow0=blockIdx.y*BC;
    if(kvrow0>=S) return;
    int tid=threadIdx.x, nthreads=blockDim.x, warp_id=tid/32, lane=tid%32;

    long base=(long)bh*S*D;
    const bf16* Qb=Q+base; const bf16* Kb=K+base; const bf16* Vb=V+base; const bf16* dOb=dO+base;
    const float* Lb=Lg+(long)bh*S; const float* Db=Dg+(long)bh*S;
    bf16* dKg=dK+base; bf16* dVg=dV+base;

    extern __shared__ char smem[];
    bf16* Ks =(bf16*)smem;
    bf16* Vs =Ks+BC*LDT;
    bf16* Qs =Vs+BC*LDT;
    bf16* dOs=Qs+BR*LDT;
    bf16* Pbf=dOs+BR*LDT;
    bf16* dSbf=Pbf+BR*LDS;
    float* buf0=(float*)(dSbf+BR*LDS);
    float* buf1=buf0+BR*LDB;
    float* Ls=buf1+BR*LDB;
    float* Ds=Ls+BR;

    load_tile<BC>(Ks,Kb,kvrow0,S,tid,nthreads);
    load_tile<BC>(Vs,Vb,kvrow0,S,tid,nthreads);

    constexpr int NACC=(BC/16)*(D/16)/NW;   // 4
    wmma::fragment<wmma::accumulator,16,16,16,float> dV_frag[NACC], dK_frag[NACC];
    #pragma unroll
    for(int i=0;i<NACC;i++){wmma::fill_fragment(dV_frag[i],0.f);wmma::fill_fragment(dK_frag[i],0.f);}
    __syncthreads();

    int num_q=(S+BR-1)/BR;
    for(int qt=0; qt<num_q; ++qt){
        int qrow0=qt*BR;
        load_tile<BR>(Qs,Qb,qrow0,S,tid,nthreads);
        load_tile<BR>(dOs,dOb,qrow0,S,tid,nthreads);
        for(int i=tid;i<BR;i+=nthreads){ int gi=qrow0+i; Ls[i]=(gi<S)?Lb[gi]:0.f; Ds[i]=(gi<S)?Db[gi]:0.f; }
        __syncthreads();

        score_gemm<BR,BC,D, wmma::row_major, wmma::col_major>(Qs, LDT, Ks, LDT, buf0, LDB, warp_id);
        score_gemm<BR,BC,D, wmma::row_major, wmma::col_major>(dOs,LDT, Vs, LDT, buf1, LDB, warp_id);
        __syncthreads();

        for(int idx=tid; idx<BR*BC; idx+=nthreads){
            int i=idx/BC, j=idx%BC; int gi=qrow0+i, gj=kvrow0+j;
            bool ok=(gi<S&&gj<S);
            float p = ok ? __expf(scale*buf0[i*LDB+j]-Ls[i]) : 0.f;
            float ds= ok ? p*(buf1[i*LDB+j]-Ds[i]) : 0.f;
            Pbf[i*LDS+j]=__float2bfloat16(p);
            dSbf[i*LDS+j]=__float2bfloat16(ds);
        }
        __syncthreads();

        #pragma unroll
        for(int i=0;i<NACC;i++){
            int tile=warp_id+i*NW; int mt=tile/DT16, nt=tile%DT16; int m0=mt*16,n0=nt*16;
            #pragma unroll
            for(int k0=0;k0<BR;k0+=16){
                wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::col_major> aP,aS;
                wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> bO,bQ;
                wmma::load_matrix_sync(aP, Pbf + m0 + k0*LDS, LDS);
                wmma::load_matrix_sync(bO, dOs + k0*LDT + n0, LDT);
                wmma::mma_sync(dV_frag[i], aP, bO, dV_frag[i]);
                wmma::load_matrix_sync(aS, dSbf + m0 + k0*LDS, LDS);
                wmma::load_matrix_sync(bQ, Qs + k0*LDT + n0, LDT);
                wmma::mma_sync(dK_frag[i], aS, bQ, dK_frag[i]);
            }
        }
        __syncthreads();
    }

    __syncthreads();
    epilogue<NACC>(dV_frag, buf0, dVg, kvrow0, S, warp_id, lane, 1.0f);
    __syncthreads();
    epilogue<NACC>(dK_frag, buf0, dKg, kvrow0, S, warp_id, lane, scale);
}

// ---------------- Kernel 2: dQ ----------------
__global__ void bwd_q_kernel(
    const bf16* Q, const bf16* K, const bf16* V, const bf16* dO,
    const float* Lg, const float* Dg, bf16* dQ, int S, float scale)
{
    int bh=blockIdx.x; int qrow0=blockIdx.y*BR;
    if(qrow0>=S) return;
    int tid=threadIdx.x, nthreads=blockDim.x, warp_id=tid/32, lane=tid%32;

    long base=(long)bh*S*D;
    const bf16* Qb=Q+base; const bf16* Kb=K+base; const bf16* Vb=V+base; const bf16* dOb=dO+base;
    const float* Lb=Lg+(long)bh*S; const float* Db=Dg+(long)bh*S;
    bf16* dQg=dQ+base;

    extern __shared__ char smem[];
    bf16* Qs =(bf16*)smem;
    bf16* dOs=Qs+BR*LDT;
    bf16* Ks =dOs+BR*LDT;
    bf16* Vs =Ks+BC*LDT;
    bf16* dSbf=Vs+BC*LDT;
    float* buf0=(float*)(dSbf+BR*LDS);
    float* buf1=buf0+BR*LDB;
    float* Ls=buf1+BR*LDB;
    float* Ds=Ls+BR;

    load_tile<BR>(Qs,Qb,qrow0,S,tid,nthreads);
    load_tile<BR>(dOs,dOb,qrow0,S,tid,nthreads);
    for(int i=tid;i<BR;i+=nthreads){ int gi=qrow0+i; Ls[i]=(gi<S)?Lb[gi]:0.f; Ds[i]=(gi<S)?Db[gi]:0.f; }

    constexpr int NACC=(BR/16)*(D/16)/NW;   // 4
    wmma::fragment<wmma::accumulator,16,16,16,float> dQ_frag[NACC];
    #pragma unroll
    for(int i=0;i<NACC;i++) wmma::fill_fragment(dQ_frag[i],0.f);
    __syncthreads();

    int num_kv=(S+BC-1)/BC;
    for(int kt=0; kt<num_kv; ++kt){
        int kvrow0=kt*BC;
        load_tile<BC>(Ks,Kb,kvrow0,S,tid,nthreads);
        load_tile<BC>(Vs,Vb,kvrow0,S,tid,nthreads);
        __syncthreads();

        score_gemm<BR,BC,D, wmma::row_major, wmma::col_major>(Qs, LDT, Ks, LDT, buf0, LDB, warp_id);
        score_gemm<BR,BC,D, wmma::row_major, wmma::col_major>(dOs,LDT, Vs, LDT, buf1, LDB, warp_id);
        __syncthreads();

        for(int idx=tid; idx<BR*BC; idx+=nthreads){
            int i=idx/BC, j=idx%BC; int gi=qrow0+i, gj=kvrow0+j;
            bool ok=(gi<S&&gj<S);
            float p = ok ? __expf(scale*buf0[i*LDB+j]-Ls[i]) : 0.f;
            float ds= ok ? p*(buf1[i*LDB+j]-Ds[i]) : 0.f;
            dSbf[i*LDS+j]=__float2bfloat16(ds);
        }
        __syncthreads();

        #pragma unroll
        for(int i=0;i<NACC;i++){
            int tile=warp_id+i*NW; int mt=tile/DT16, nt=tile%DT16; int m0=mt*16,n0=nt*16;
            #pragma unroll
            for(int k0=0;k0<BC;k0+=16){
                wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> aS;
                wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> bK;
                wmma::load_matrix_sync(aS, dSbf + m0*LDS + k0, LDS);
                wmma::load_matrix_sync(bK, Ks + k0*LDT + n0, LDT);
                wmma::mma_sync(dQ_frag[i], aS, bK, dQ_frag[i]);
            }
        }
        __syncthreads();
    }

    __syncthreads();
    epilogue<NACC>(dQ_frag, buf0, dQg, qrow0, S, warp_id, lane, scale);
}

static float* g_D = nullptr;
static size_t  g_Dsz = 0;

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV)
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2);
    int BH=B*H;
    float scale=1.0f/sqrtf((float)D);

    const bf16* Qp=static_cast<const bf16*>(Q.data_ptr());
    const bf16* Kp=static_cast<const bf16*>(K.data_ptr());
    const bf16* Vp=static_cast<const bf16*>(V.data_ptr());
    const bf16* Op=static_cast<const bf16*>(O.data_ptr());
    const bf16* dOp=static_cast<const bf16*>(dO.data_ptr());
    const float* Lp=static_cast<const float*>(L.data_ptr());
    bf16* dQp=static_cast<bf16*>(dQ.data_ptr());
    bf16* dKp=static_cast<bf16*>(dK.data_ptr());
    bf16* dVp=static_cast<bf16*>(dV.data_ptr());

    cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));

    size_t need=(size_t)BH*S*sizeof(float);
    if(need>g_Dsz){ if(g_D) cudaFree(g_D); CUDA_CHECK(cudaMalloc(&g_D,need)); g_Dsz=need; }

    int total_rows=BH*S;
    int dblock=128, drows=dblock/32;
    int dgrid=(total_rows+drows-1)/drows;
    compute_D_kernel<<<dgrid,dblock,0,stream>>>(Op,dOp,g_D,total_rows);
    CUDA_CHECK(cudaGetLastError());

    size_t smem1=(size_t)(2*BC*LDT + 2*BR*LDT + 2*BR*LDS)*sizeof(bf16)
               + (size_t)(2*BR*LDB + 2*BR)*sizeof(float);
    size_t smem2=(size_t)(2*BR*LDT + 2*BC*LDT + BR*LDS)*sizeof(bf16)
               + (size_t)(2*BR*LDB + 2*BR)*sizeof(float);

    static bool attr_set=false;
    if(!attr_set){
        CUDA_CHECK(cudaFuncSetAttribute(bwd_kv_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem1));
        CUDA_CHECK(cudaFuncSetAttribute(bwd_q_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem2));
        attr_set=true;
    }

    int block=NT_THREADS;
    int num_kv=(S+BC-1)/BC, num_q=(S+BR-1)/BR;

    dim3 grid1(BH,num_kv);
    bwd_kv_kernel<<<grid1,block,smem1,stream>>>(Qp,Kp,Vp,dOp,Lp,g_D,dKp,dVp,S,scale);
    CUDA_CHECK(cudaGetLastError());

    dim3 grid2(BH,num_q);
    bwd_q_kernel<<<grid2,block,smem2,stream>>>(Qp,Kp,Vp,dOp,Lp,g_D,dQp,S,scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd