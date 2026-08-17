#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;
using bf16 = __nv_bfloat16;
using AccFrag = wmma::fragment<wmma::accumulator,16,16,16,float>;

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);} }while(0)

namespace mha_bwd {

static const int LDD = 136; // padded d stride (128+8)
static const int LDS = 72;  // padded score stride (64+8)

__global__ void compute_delta_kernel(const bf16* O, const bf16* dO, float* D, int total_rows){
  int warp_in_block = threadIdx.x>>5;
  int lane = threadIdx.x&31;
  int row = blockIdx.x*(blockDim.x>>5) + warp_in_block;
  if(row>=total_rows) return;
  const bf16* o = O + (size_t)row*128;
  const bf16* g = dO + (size_t)row*128;
  float sum=0.f;
  #pragma unroll
  for(int k=lane;k<128;k+=32) sum += __bfloat162float(o[k])*__bfloat162float(g[k]);
  #pragma unroll
  for(int off=16;off>0;off>>=1) sum += __shfl_down_sync(0xffffffffu,sum,off);
  if(lane==0) D[row]=sum;
}

// load [R,128] tile into s[row*LDD + k]
__device__ __forceinline__ void load_tile(const bf16* g, bf16* s, int start, int R, int S, int bh, int tid){
  int total = R*16;
  for(int c=tid;c<total;c+=128){
    int row=c>>4; int wk=c&15;
    int gr=start+row;
    uint4 v;
    if(gr<S) v = *(const uint4*)(g + ((size_t)bh*S+gr)*128 + wk*8);
    else     v = make_uint4(0,0,0,0);
    *(uint4*)(s + row*LDD + wk*8) = v;
  }
}

// C[ROWS,COLS] = A[ROWS,128] @ B[COLS,128]^T  (contract 128)
template<int ROWS,int COLS>
__device__ __forceinline__ void mm_score(const bf16* A, const bf16* B, float* C, int warp){
  constexpr int CPW=COLS/4;
  constexpr int RT=ROWS/16;
  constexpr int CT=CPW/16;
  #pragma unroll
  for(int rt=0;rt<RT;rt++){
    #pragma unroll
    for(int ct=0;ct<CT;ct++){
      AccFrag acc; wmma::fill_fragment(acc,0.f);
      int col0=warp*CPW+ct*16;
      #pragma unroll
      for(int kt=0;kt<8;kt++){
        wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> bff;
        wmma::load_matrix_sync(af, A + (rt*16)*LDD + kt*16, LDD);
        wmma::load_matrix_sync(bff, B + col0*LDD + kt*16, LDD);
        wmma::mma_sync(acc,af,bff,acc);
      }
      wmma::store_matrix_sync(C + (rt*16)*LDS + col0, acc, LDS, wmma::mem_row_major);
    }
  }
}

// acc[ROWS,128] += A[ROWS,M2] @ B[M2,128]  (contract M2)
template<int ROWS,int M2>
__device__ __forceinline__ void mm_out(const bf16* A, const bf16* B, AccFrag(&acc)[ROWS/16][2], int warp){
  constexpr int RT=ROWS/16;
  constexpr int KT=M2/16;
  #pragma unroll
  for(int rt=0;rt<RT;rt++){
    #pragma unroll
    for(int ct=0;ct<2;ct++){
      int col0=warp*32+ct*16;
      #pragma unroll
      for(int kt=0;kt<KT;kt++){
        wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> bff;
        wmma::load_matrix_sync(af, A + (rt*16)*LDS + kt*16, LDS);
        wmma::load_matrix_sync(bff, B + (kt*16)*LDD + col0, LDD);
        wmma::mma_sync(acc[rt][ct],af,bff,acc[rt][ct]);
      }
    }
  }
}

// dK,dV kernel: block over (bh, kv block of 32 keys), loop query blocks of 64
__global__ void bwd_dkdv_kernel(const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
    const float* Lg,const float* Dg,bf16* dK,bf16* dV,int S){
  extern __shared__ char smem[];
  bf16*  Ks   = (bf16*) (smem+0);
  bf16*  Vs   = Ks   + 32*LDD;
  bf16*  Qs   = Vs   + 32*LDD;
  bf16*  dOs  = Qs   + 64*LDD;
  float* Sbuf = (float*)(dOs + 64*LDD);
  float* dPbuf= Sbuf + 32*LDS;
  bf16*  pTb  = (bf16*)(dPbuf + 32*LDS);
  bf16*  dsTb = pTb  + 32*LDS;
  float* Ls   = (float*)(dsTb + 32*LDS);
  float* Ds   = Ls + 64;
  float* outf = (float*)(smem+0);

  int bh=blockIdx.y;
  int kv_start=blockIdx.x*32;
  int tid=threadIdx.x; int warp=tid>>5;
  const float scale=0.08838834764831845f;

  load_tile(K,Ks,kv_start,32,S,bh,tid);
  load_tile(V,Vs,kv_start,32,S,bh,tid);

  AccFrag dV_acc[2][2], dK_acc[2][2];
  #pragma unroll
  for(int i=0;i<2;i++)
    #pragma unroll
    for(int j=0;j<2;j++){wmma::fill_fragment(dV_acc[i][j],0.f);wmma::fill_fragment(dK_acc[i][j],0.f);}

  int numq=(S+63)/64;
  for(int qb=0;qb<numq;qb++){
    int q_start=qb*64;
    __syncthreads();
    load_tile(Q,Qs,q_start,64,S,bh,tid);
    load_tile(dO,dOs,q_start,64,S,bh,tid);
    if(tid<64){int gm=q_start+tid; if(gm<S){Ls[tid]=Lg[(size_t)bh*S+gm];Ds[tid]=Dg[(size_t)bh*S+gm];}else{Ls[tid]=0.f;Ds[tid]=0.f;}}
    __syncthreads();
    mm_score<32,64>(Ks,Qs,Sbuf,warp);
    mm_score<32,64>(Vs,dOs,dPbuf,warp);
    __syncthreads();
    for(int idx=tid;idx<32*64;idx+=128){
      int m=idx&63; int r=idx>>6; int gm=q_start+m;
      float p=(gm<S)?__expf(scale*Sbuf[r*LDS+m]-Ls[m]):0.f;
      pTb[r*LDS+m]=__float2bfloat16(p);
      float ds=scale*p*(dPbuf[r*LDS+m]-Ds[m]);
      dsTb[r*LDS+m]=__float2bfloat16(ds);
    }
    __syncthreads();
    mm_out<32,64>(pTb,dOs,dV_acc,warp);
    mm_out<32,64>(dsTb,Qs,dK_acc,warp);
  }

  __syncthreads();
  #pragma unroll
  for(int rt=0;rt<2;rt++)
   #pragma unroll
   for(int ct=0;ct<2;ct++){int col0=warp*32+ct*16;wmma::store_matrix_sync(outf+(rt*16)*128+col0,dV_acc[rt][ct],128,wmma::mem_row_major);}
  __syncthreads();
  for(int idx=tid;idx<32*128;idx+=128){int row=idx>>7;int c=idx&127;int gr=kv_start+row;if(gr<S)dV[((size_t)bh*S+gr)*128+c]=__float2bfloat16(outf[idx]);}
  __syncthreads();
  #pragma unroll
  for(int rt=0;rt<2;rt++)
   #pragma unroll
   for(int ct=0;ct<2;ct++){int col0=warp*32+ct*16;wmma::store_matrix_sync(outf+(rt*16)*128+col0,dK_acc[rt][ct],128,wmma::mem_row_major);}
  __syncthreads();
  for(int idx=tid;idx<32*128;idx+=128){int row=idx>>7;int c=idx&127;int gr=kv_start+row;if(gr<S)dK[((size_t)bh*S+gr)*128+c]=__float2bfloat16(outf[idx]);}
}

// dQ kernel: block over (bh, q block of 32 queries), loop key blocks of 64
__global__ void bwd_dq_kernel(const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
    const float* Lg,const float* Dg,bf16* dQ,int S){
  extern __shared__ char smem[];
  bf16*  Qs   = (bf16*) (smem+0);
  bf16*  dOs  = Qs  + 32*LDD;
  bf16*  Ks   = dOs + 32*LDD;
  bf16*  Vs   = Ks  + 64*LDD;
  float* Sbuf = (float*)(Vs + 64*LDD);
  float* dPbuf= Sbuf + 32*LDS;
  bf16*  dSb  = (bf16*)(dPbuf + 32*LDS);
  float* Ls   = (float*)(dSb + 32*LDS);
  float* Ds   = Ls + 32;
  float* outf = (float*)(smem+0);

  int bh=blockIdx.y;
  int q_start=blockIdx.x*32;
  int tid=threadIdx.x; int warp=tid>>5;
  const float scale=0.08838834764831845f;

  load_tile(Q,Qs,q_start,32,S,bh,tid);
  load_tile(dO,dOs,q_start,32,S,bh,tid);
  if(tid<32){int gm=q_start+tid; if(gm<S){Ls[tid]=Lg[(size_t)bh*S+gm];Ds[tid]=Dg[(size_t)bh*S+gm];}else{Ls[tid]=0.f;Ds[tid]=0.f;}}

  AccFrag dQ_acc[2][2];
  #pragma unroll
  for(int i=0;i<2;i++)
   #pragma unroll
   for(int j=0;j<2;j++) wmma::fill_fragment(dQ_acc[i][j],0.f);

  int numk=(S+63)/64;
  for(int kb=0;kb<numk;kb++){
    int k_start=kb*64;
    __syncthreads();
    load_tile(K,Ks,k_start,64,S,bh,tid);
    load_tile(V,Vs,k_start,64,S,bh,tid);
    __syncthreads();
    mm_score<32,64>(Qs,Ks,Sbuf,warp);
    mm_score<32,64>(dOs,Vs,dPbuf,warp);
    __syncthreads();
    for(int idx=tid;idx<32*64;idx+=128){
      int q=idx>>6; int n=idx&63; int gn=k_start+n;
      float p=(gn<S)?__expf(scale*Sbuf[q*LDS+n]-Ls[q]):0.f;
      float ds=scale*p*(dPbuf[q*LDS+n]-Ds[q]);
      dSb[q*LDS+n]=__float2bfloat16(ds);
    }
    __syncthreads();
    mm_out<32,64>(dSb,Ks,dQ_acc,warp);
  }
  __syncthreads();
  #pragma unroll
  for(int rt=0;rt<2;rt++)
   #pragma unroll
   for(int ct=0;ct<2;ct++){int col0=warp*32+ct*16;wmma::store_matrix_sync(outf+(rt*16)*128+col0,dQ_acc[rt][ct],128,wmma::mem_row_major);}
  __syncthreads();
  for(int idx=tid;idx<32*128;idx+=128){int row=idx>>7;int c=idx&127;int gr=q_start+row;if(gr<S)dQ[((size_t)bh*S+gr)*128+c]=__float2bfloat16(outf[idx]);}
}

static float* g_Dbuf=nullptr;
static size_t g_Dsize=0;

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=Q.size(0),H=Q.size(1),S=Q.size(2);
  int BH=B*H;

  const bf16* Qp=(const bf16*)Q.data_ptr();
  const bf16* Kp=(const bf16*)K.data_ptr();
  const bf16* Vp=(const bf16*)V.data_ptr();
  const bf16* Op=(const bf16*)O.data_ptr();
  const bf16* dOp=(const bf16*)dO.data_ptr();
  const float* Lp=(const float*)L.data_ptr();
  bf16* dQp=(bf16*)dQ.data_ptr();
  bf16* dKp=(bf16*)dK.data_ptr();
  bf16* dVp=(bf16*)dV.data_ptr();

  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id);

  size_t need=(size_t)BH*S;
  if(need>g_Dsize){ if(g_Dbuf) cudaFree(g_Dbuf); CUDA_CHECK(cudaMalloc(&g_Dbuf,need*sizeof(float))); g_Dsize=need; }
  float* Dbuf=g_Dbuf;

  {
    int total=BH*S;
    int block=256;
    int grid=(total + (block/32) -1)/(block/32);
    compute_delta_kernel<<<grid,block,0,stream>>>(Op,dOp,Dbuf,total);
    CUDA_CHECK(cudaGetLastError());
  }

  int sh_dkdv = 2*32*LDD*2 + 2*64*LDD*2 + 2*32*LDS*4 + 2*32*LDS*2 + 128*4 + 1024;
  int sh_dq   = 2*32*LDD*2 + 2*64*LDD*2 + 2*32*LDS*4 + 1*32*LDS*2 + 64*4 + 1024;
  cudaFuncSetAttribute(bwd_dkdv_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sh_dkdv);
  cudaFuncSetAttribute(bwd_dq_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sh_dq);

  {
    dim3 grid((S+31)/32, BH);
    dim3 block(128);
    bwd_dkdv_kernel<<<grid,block,sh_dkdv,stream>>>(Qp,Kp,Vp,dOp,Lp,Dbuf,dKp,dVp,S);
    CUDA_CHECK(cudaGetLastError());
  }
  {
    dim3 grid((S+31)/32, BH);
    dim3 block(128);
    bwd_dq_kernel<<<grid,block,sh_dq,stream>>>(Qp,Kp,Vp,dOp,Lp,Dbuf,dQp,S);
    CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd