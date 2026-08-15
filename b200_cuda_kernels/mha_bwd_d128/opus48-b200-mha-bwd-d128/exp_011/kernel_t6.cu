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

static const int LDD = 136;   // stride for [*,128] tiles
static const int LDS = 72;    // stride for [32,64] score tiles
static const int OUTLD = 136; // stride for [32,128] output staging

__global__ void compute_delta_kernel(const bf16* O, const bf16* dO, float* D, int total_rows){
  int wib = threadIdx.x>>5, lane = threadIdx.x&31;
  int row = blockIdx.x*(blockDim.x>>5) + wib;
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

// load [R,128] tile into s[row*LDD + k], rows >= S masked to 0
__device__ __forceinline__ void load_tile(const bf16* g, bf16* s, int start, int R, int S, int bh, int tid){
  int total = R*16;
  for(int c=tid;c<total;c+=128){
    int row=c>>4, wk=c&15, gr=start+row;
    uint4 v;
    if(gr<S) v = *(const uint4*)(g + ((size_t)bh*S+gr)*128 + wk*8);
    else     v = make_uint4(0,0,0,0);
    *(uint4*)(s + row*LDD + wk*8) = v;
  }
}

// C[32,64] = A[32,128] @ B[64,128]^T  (contract 128), 4 warps, warp -> 16 cols
__device__ __forceinline__ void mm_score(const bf16* A,const bf16* B,float* C,int w){
  int col0 = w*16;
  AccFrag acc0, acc1; wmma::fill_fragment(acc0,0.f); wmma::fill_fragment(acc1,0.f);
  #pragma unroll
  for(int kt=0;kt<8;kt++){
    wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> bf;
    wmma::load_matrix_sync(bf, B + col0*LDD + kt*16, LDD);
    wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> af0, af1;
    wmma::load_matrix_sync(af0, A + 0*LDD + kt*16, LDD);
    wmma::load_matrix_sync(af1, A + 16*LDD + kt*16, LDD);
    wmma::mma_sync(acc0, af0, bf, acc0);
    wmma::mma_sync(acc1, af1, bf, acc1);
  }
  wmma::store_matrix_sync(C + 0*LDS + col0, acc0, LDS, wmma::mem_row_major);
  wmma::store_matrix_sync(C + 16*LDS + col0, acc1, LDS, wmma::mem_row_major);
}

// acc[2][2] += A[32,64] @ B[64,128] (contract 64), 4 warps, warp -> 32 cols
__device__ __forceinline__ void mm_out(const bf16* A,const bf16* B,AccFrag(&acc)[2][2],int w){
  int col_base = w*32;
  #pragma unroll
  for(int kt=0;kt<4;kt++){
    wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> af0, af1;
    wmma::load_matrix_sync(af0, A + 0*LDS + kt*16, LDS);
    wmma::load_matrix_sync(af1, A + 16*LDS + kt*16, LDS);
    #pragma unroll
    for(int ct=0;ct<2;ct++){
      int col0 = col_base + ct*16;
      wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> bff;
      wmma::load_matrix_sync(bff, B + (kt*16)*LDD + col0, LDD);
      wmma::mma_sync(acc[0][ct], af0, bff, acc[0][ct]);
      wmma::mma_sync(acc[1][ct], af1, bff, acc[1][ct]);
    }
  }
}

// dK,dV kernel: block over (bh, 32-key tile), loop 64-query blocks
__global__ void bwd_dkdv_kernel(const bf16* __restrict__ Q,const bf16* __restrict__ K,
    const bf16* __restrict__ V,const bf16* __restrict__ dO,
    const float* __restrict__ Lg,const float* __restrict__ Dg,
    bf16* __restrict__ dK,bf16* __restrict__ dV,int S){
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

  int bh=blockIdx.y, kv_start=blockIdx.x*32;
  int tid=threadIdx.x, w=tid>>5;
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
    mm_score(Ks,Qs,Sbuf,w);   // S^T[key,query]
    mm_score(Vs,dOs,dPbuf,w);  // dP^T[key,query]
    __syncthreads();
    for(int idx=tid;idx<32*64;idx+=128){
      int key=idx>>6, query=idx&63; int gq=q_start+query;
      float p=(gq<S)?__expf(scale*Sbuf[key*LDS+query]-Ls[query]):0.f;
      pTb[key*LDS+query]=__float2bfloat16(p);
      float ds=scale*p*(dPbuf[key*LDS+query]-Ds[query]);
      dsTb[key*LDS+query]=__float2bfloat16(ds);
    }
    __syncthreads();
    mm_out(pTb,dOs,dV_acc,w);
    mm_out(dsTb,Qs,dK_acc,w);
  }
  int col_base=w*32;
  __syncthreads();
  #pragma unroll
  for(int ct=0;ct<2;ct++){
    wmma::store_matrix_sync(outf+0*OUTLD+col_base+ct*16,dV_acc[0][ct],OUTLD,wmma::mem_row_major);
    wmma::store_matrix_sync(outf+16*OUTLD+col_base+ct*16,dV_acc[1][ct],OUTLD,wmma::mem_row_major);
  }
  __syncthreads();
  for(int idx=tid;idx<32*128;idx+=128){int row=idx>>7,c=idx&127,gr=kv_start+row;if(gr<S)dV[((size_t)bh*S+gr)*128+c]=__float2bfloat16(outf[row*OUTLD+c]);}
  __syncthreads();
  #pragma unroll
  for(int ct=0;ct<2;ct++){
    wmma::store_matrix_sync(outf+0*OUTLD+col_base+ct*16,dK_acc[0][ct],OUTLD,wmma::mem_row_major);
    wmma::store_matrix_sync(outf+16*OUTLD+col_base+ct*16,dK_acc[1][ct],OUTLD,wmma::mem_row_major);
  }
  __syncthreads();
  for(int idx=tid;idx<32*128;idx+=128){int row=idx>>7,c=idx&127,gr=kv_start+row;if(gr<S)dK[((size_t)bh*S+gr)*128+c]=__float2bfloat16(outf[row*OUTLD+c]);}
}

// dQ kernel: block over (bh, 32-query tile), loop 64-key blocks
__global__ void bwd_dq_kernel(const bf16* __restrict__ Q,const bf16* __restrict__ K,
    const bf16* __restrict__ V,const bf16* __restrict__ dO,
    const float* __restrict__ Lg,const float* __restrict__ Dg,
    bf16* __restrict__ dQ,int S){
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

  int bh=blockIdx.y, q_start=blockIdx.x*32;
  int tid=threadIdx.x, w=tid>>5;
  const float scale=0.08838834764831845f;

  load_tile(Q,Qs,q_start,32,S,bh,tid);
  load_tile(dO,dOs,q_start,32,S,bh,tid);
  if(tid<32){int gm=q_start+tid; if(gm<S){Ls[tid]=Lg[(size_t)bh*S+gm];Ds[tid]=Dg[(size_t)bh*S+gm];}else{Ls[tid]=0.f;Ds[tid]=0.f;}}

  // dQ output: [32,128] -> need only 2x2 acc per warp? warp->32 cols, rows 0,16
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
    // S[query(32),key(64)] : A=Q[32,128], B=K[64,128]^T
    mm_score(Qs,Ks,Sbuf,w);
    mm_score(dOs,Vs,dPbuf,w);
    __syncthreads();
    for(int idx=tid;idx<32*64;idx+=128){
      int query=idx>>6, key=idx&63; int gk=k_start+key;
      float p=(gk<S)?__expf(scale*Sbuf[query*LDS+key]-Ls[query]):0.f;
      float ds=scale*p*(dPbuf[query*LDS+key]-Ds[query]);
      dSb[query*LDS+key]=__float2bfloat16(ds);
    }
    __syncthreads();
    // dQ[32,128] += dS[32,64] @ K[64,128]
    mm_out(dSb,Ks,dQ_acc,w);
  }
  int col_base=w*32;
  __syncthreads();
  #pragma unroll
  for(int ct=0;ct<2;ct++){
    wmma::store_matrix_sync(outf+0*OUTLD+col_base+ct*16,dQ_acc[0][ct],OUTLD,wmma::mem_row_major);
    wmma::store_matrix_sync(outf+16*OUTLD+col_base+ct*16,dQ_acc[1][ct],OUTLD,wmma::mem_row_major);
  }
  __syncthreads();
  for(int idx=tid;idx<32*128;idx+=128){int row=idx>>7,c=idx&127,gr=q_start+row;if(gr<S)dQ[((size_t)bh*S+gr)*128+c]=__float2bfloat16(outf[row*OUTLD+c]);}
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
    int total=BH*S, block=256;
    int grid=(total + (block/32) -1)/(block/32);
    compute_delta_kernel<<<grid,block,0,stream>>>(Op,dOp,Dbuf,total);
    CUDA_CHECK(cudaGetLastError());
  }

  int sh_dkdv = (32+32+64+64)*LDD*2 + 2*32*LDS*4 + 2*32*LDS*2 + 128*4;
  int sh_dq   = (32+32+64+64)*LDD*2 + 2*32*LDS*4 + 1*32*LDS*2 + 64*4;
  cudaFuncSetAttribute(bwd_dkdv_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sh_dkdv);
  cudaFuncSetAttribute(bwd_dq_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sh_dq);

  {
    dim3 grid((S+31)/32, BH); dim3 block(128);
    bwd_dkdv_kernel<<<grid,block,sh_dkdv,stream>>>(Qp,Kp,Vp,dOp,Lp,Dbuf,dKp,dVp,S);
    CUDA_CHECK(cudaGetLastError());
  }
  {
    dim3 grid((S+31)/32, BH); dim3 block(128);
    bwd_dq_kernel<<<grid,block,sh_dq,stream>>>(Qp,Kp,Vp,dOp,Lp,Dbuf,dQp,S);
    CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd