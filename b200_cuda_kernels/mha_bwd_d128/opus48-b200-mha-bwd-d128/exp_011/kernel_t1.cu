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

// D_i = sum_k O[i,k]*dO[i,k]
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

// load [R,128] bf16 tile (row-major) into shared (rows >= S masked to 0)
__device__ __forceinline__ void load_tile(const bf16* g, bf16* s, int start, int R, int S, int bh, int tid){
  int total = R*16;
  for(int c=tid;c<total;c+=256){
    int row=c>>4; int wk=c&15;
    int gr=start+row;
    uint4 v;
    if(gr<S) v = *(const uint4*)(g + ((size_t)bh*S+gr)*128 + wk*8);
    else     v = make_uint4(0,0,0,0);
    *(uint4*)(s + row*128 + wk*8) = v;
  }
}

// C[64,64] = A[64,128] @ B[64,128]^T  (contraction over 128), 8 warps
__device__ __forceinline__ void mm_scoreT(const bf16* A, const bf16* B, float* C, int wr, int wc){
  #pragma unroll
  for(int ct=0;ct<2;ct++){
    int col0 = wc*32 + ct*16;
    AccFrag acc; wmma::fill_fragment(acc,0.f);
    #pragma unroll
    for(int kt=0;kt<8;kt++){
      wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> af;
      wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> bff;
      wmma::load_matrix_sync(af, A + wr*16*128 + kt*16, 128);
      wmma::load_matrix_sync(bff, B + col0*128 + kt*16, 128);
      wmma::mma_sync(acc,af,bff,acc);
    }
    wmma::store_matrix_sync(C + wr*16*64 + col0, acc, 64, wmma::mem_row_major);
  }
}

// acc[64,128] += A[64,64] @ B[64,128]  (contraction over 64), 8 warps
__device__ __forceinline__ void mm_out(const bf16* A, const bf16* B, AccFrag (&acc)[4], int wr, int wc){
  #pragma unroll
  for(int ct=0;ct<4;ct++){
    int col0 = wc*64 + ct*16;
    #pragma unroll
    for(int kt=0;kt<4;kt++){
      wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> af;
      wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> bff;
      wmma::load_matrix_sync(af, A + wr*16*64 + kt*16, 64);
      wmma::load_matrix_sync(bff, B + (kt*16)*128 + col0, 128);
      wmma::mma_sync(acc[ct],af,bff,acc[ct]);
    }
  }
}

// dK,dV kernel: block over (bh, kv block of 64 keys), loop query blocks of 64
__global__ void bwd_dkdv_kernel(const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
    const float* Lg,const float* Dg,bf16* dK,bf16* dV,int S){
  extern __shared__ char smem[];
  bf16*  Ks   = (bf16*) (smem+0);
  bf16*  Vs   = (bf16*) (smem+16384);
  bf16*  Qs   = (bf16*) (smem+32768);
  bf16*  dOs  = (bf16*) (smem+49152);
  float* Sbuf = (float*)(smem+65536);
  float* dPbuf= (float*)(smem+81920);
  bf16*  pTb  = (bf16*) (smem+98304);
  bf16*  dsTb = (bf16*) (smem+106496);
  float* Ls   = (float*)(smem+114688);
  float* Ds   = (float*)(smem+114944);
  float* outf = (float*)(smem+0);   // overlaps Ks/Vs (unused at epilogue)

  int bh=blockIdx.y;
  int kv_start=blockIdx.x*64;
  int tid=threadIdx.x;
  int warp=tid>>5; int wr=warp>>1; int wc=warp&1;
  const float scale=0.08838834764831845f;

  load_tile(K,Ks,kv_start,64,S,bh,tid);
  load_tile(V,Vs,kv_start,64,S,bh,tid);

  AccFrag dV_acc[4], dK_acc[4];
  #pragma unroll
  for(int i=0;i<4;i++){wmma::fill_fragment(dV_acc[i],0.f);wmma::fill_fragment(dK_acc[i],0.f);}

  int numq=(S+63)/64;
  for(int qb=0;qb<numq;qb++){
    int q_start=qb*64;
    __syncthreads();
    load_tile(Q,Qs,q_start,64,S,bh,tid);
    load_tile(dO,dOs,q_start,64,S,bh,tid);
    if(tid<64){int gm=q_start+tid; if(gm<S){Ls[tid]=Lg[(size_t)bh*S+gm];Ds[tid]=Dg[(size_t)bh*S+gm];}else{Ls[tid]=0.f;Ds[tid]=0.f;}}
    __syncthreads();
    mm_scoreT(Ks,Qs,Sbuf,wr,wc);   // S^T[key,query]
    mm_scoreT(Vs,dOs,dPbuf,wr,wc);  // dP^T[key,query]
    __syncthreads();
    for(int idx=tid;idx<64*64;idx+=256){
      int m=idx&63; int gm=q_start+m;
      float s=Sbuf[idx];
      float p=(gm<S)?__expf(scale*s-Ls[m]):0.f;
      pTb[idx]=__float2bfloat16(p);
      float dp=dPbuf[idx];
      float ds=scale*p*(dp-Ds[m]);
      dsTb[idx]=__float2bfloat16(ds);
    }
    __syncthreads();
    mm_out(pTb,dOs,dV_acc,wr,wc);
    mm_out(dsTb,Qs,dK_acc,wr,wc);
  }
  __syncthreads();
  #pragma unroll
  for(int ct=0;ct<4;ct++) wmma::store_matrix_sync(outf+wr*16*128+wc*64+ct*16,dV_acc[ct],128,wmma::mem_row_major);
  __syncthreads();
  for(int idx=tid;idx<64*128;idx+=256){int row=idx>>7;int c=idx&127;int gr=kv_start+row;if(gr<S)dV[((size_t)bh*S+gr)*128+c]=__float2bfloat16(outf[idx]);}
  __syncthreads();
  #pragma unroll
  for(int ct=0;ct<4;ct++) wmma::store_matrix_sync(outf+wr*16*128+wc*64+ct*16,dK_acc[ct],128,wmma::mem_row_major);
  __syncthreads();
  for(int idx=tid;idx<64*128;idx+=256){int row=idx>>7;int c=idx&127;int gr=kv_start+row;if(gr<S)dK[((size_t)bh*S+gr)*128+c]=__float2bfloat16(outf[idx]);}
}

// dQ kernel: block over (bh, q block of 64 queries), loop key blocks of 64
__global__ void bwd_dq_kernel(const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
    const float* Lg,const float* Dg,bf16* dQ,int S){
  extern __shared__ char smem[];
  bf16*  Qs   = (bf16*) (smem+0);
  bf16*  dOs  = (bf16*) (smem+16384);
  bf16*  Ks   = (bf16*) (smem+32768);
  bf16*  Vs   = (bf16*) (smem+49152);
  float* Sbuf = (float*)(smem+65536);
  float* dPbuf= (float*)(smem+81920);
  bf16*  dSb  = (bf16*) (smem+98304);
  float* Ls   = (float*)(smem+106496);
  float* Ds   = (float*)(smem+106752);
  float* outf = (float*)(smem+0);

  int bh=blockIdx.y;
  int q_start=blockIdx.x*64;
  int tid=threadIdx.x;
  int warp=tid>>5; int wr=warp>>1; int wc=warp&1;
  const float scale=0.08838834764831845f;

  load_tile(Q,Qs,q_start,64,S,bh,tid);
  load_tile(dO,dOs,q_start,64,S,bh,tid);
  if(tid<64){int gm=q_start+tid;if(gm<S){Ls[tid]=Lg[(size_t)bh*S+gm];Ds[tid]=Dg[(size_t)bh*S+gm];}else{Ls[tid]=0.f;Ds[tid]=0.f;}}

  AccFrag dQ_acc[4];
  #pragma unroll
  for(int i=0;i<4;i++) wmma::fill_fragment(dQ_acc[i],0.f);

  int numk=(S+63)/64;
  for(int kb=0;kb<numk;kb++){
    int k_start=kb*64;
    __syncthreads();
    load_tile(K,Ks,k_start,64,S,bh,tid);
    load_tile(V,Vs,k_start,64,S,bh,tid);
    __syncthreads();
    mm_scoreT(Qs,Ks,Sbuf,wr,wc);    // S[query,key]
    mm_scoreT(dOs,Vs,dPbuf,wr,wc);   // dP[query,key]
    __syncthreads();
    for(int idx=tid;idx<64*64;idx+=256){
      int q=idx>>6; int n=idx&63; int gn=k_start+n;
      float s=Sbuf[idx];
      float p=(gn<S)?__expf(scale*s-Ls[q]):0.f;
      float dp=dPbuf[idx];
      float ds=scale*p*(dp-Ds[q]);
      dSb[idx]=__float2bfloat16(ds);
    }
    __syncthreads();
    mm_out(dSb,Ks,dQ_acc,wr,wc);
  }
  __syncthreads();
  #pragma unroll
  for(int ct=0;ct<4;ct++) wmma::store_matrix_sync(outf+wr*16*128+wc*64+ct*16,dQ_acc[ct],128,wmma::mem_row_major);
  __syncthreads();
  for(int idx=tid;idx<64*128;idx+=256){int row=idx>>7;int c=idx&127;int gr=q_start+row;if(gr<S)dQ[((size_t)bh*S+gr)*128+c]=__float2bfloat16(outf[idx]);}
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

  int sh_dkdv=115200;
  int sh_dq=107008;
  cudaFuncSetAttribute(bwd_dkdv_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sh_dkdv);
  cudaFuncSetAttribute(bwd_dq_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sh_dq);

  {
    dim3 grid((S+63)/64, BH);
    dim3 block(256);
    bwd_dkdv_kernel<<<grid,block,sh_dkdv,stream>>>(Qp,Kp,Vp,dOp,Lp,Dbuf,dKp,dVp,S);
    CUDA_CHECK(cudaGetLastError());
  }
  {
    dim3 grid((S+63)/64, BH);
    dim3 block(256);
    bwd_dq_kernel<<<grid,block,sh_dq,stream>>>(Qp,Kp,Vp,dOp,Lp,Dbuf,dQp,S);
    CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd