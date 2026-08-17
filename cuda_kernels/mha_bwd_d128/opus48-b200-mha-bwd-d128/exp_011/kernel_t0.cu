#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);} }while(0)

namespace mha_bwd {

// D_i = sum_k O[i,k]*dO[i,k]
__global__ void compute_delta_kernel(const __nv_bfloat16* O, const __nv_bfloat16* dO, float* D, int total_rows){
  int warp_in_block = threadIdx.x/32;
  int lane = threadIdx.x%32;
  int row = blockIdx.x*(blockDim.x/32) + warp_in_block;
  if(row>=total_rows) return;
  const __nv_bfloat16* o = O + (size_t)row*128;
  const __nv_bfloat16* g = dO + (size_t)row*128;
  float sum=0.f;
  #pragma unroll
  for(int k=lane;k<128;k+=32) sum += __bfloat162float(o[k])*__bfloat162float(g[k]);
  #pragma unroll
  for(int off=16;off>0;off>>=1) sum += __shfl_down_sync(0xffffffffu,sum,off);
  if(lane==0) D[row]=sum;
}

// load [R,128] bf16 tile (row-major) from global (rows may be masked to 0)
__device__ __forceinline__ void load_tile(const __nv_bfloat16* g, __nv_bfloat16* s, int start, int R, int S, int bh, int tid){
  int total = R*16; // uint4 chunks of 8 bf16
  for(int c=tid;c<total;c+=128){
    int row=c>>4; int wk=c&15;
    int gr=start+row;
    uint4 val;
    if(gr<S) val = *(const uint4*)(g + ((size_t)bh*S+gr)*128 + wk*8);
    else     val = make_uint4(0,0,0,0);
    *(uint4*)(s + row*128 + wk*8) = val;
  }
}

// C[ROWS,COLS] = A[ROWS,128] * B[COLS,128]^T  (contraction over 128)
template<int ROWS,int COLS>
__device__ __forceinline__ void mm_score(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, int warp){
  constexpr int CPW=COLS/4;
  constexpr int RT=ROWS/16;
  constexpr int CT=CPW/16;
  #pragma unroll
  for(int rt=0;rt<RT;rt++){
    #pragma unroll
    for(int ct=0;ct<CT;ct++){
      wmma::fragment<wmma::accumulator,16,16,16,float> acc;
      wmma::fill_fragment(acc,0.f);
      int col0=warp*CPW+ct*16;
      #pragma unroll
      for(int kt=0;kt<8;kt++){
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> bf;
        wmma::load_matrix_sync(af, A + (rt*16)*128 + kt*16, 128);
        wmma::load_matrix_sync(bf, B + col0*128 + kt*16, 128);
        wmma::mma_sync(acc,af,bf,acc);
      }
      wmma::store_matrix_sync(C + (rt*16)*COLS + col0, acc, COLS, wmma::mem_row_major);
    }
  }
}

// acc[ROWS,128] += A[ROWS,M2] * B[M2,128]  (contraction over M2)
template<int ROWS,int M2>
__device__ __forceinline__ void mm_out(const __nv_bfloat16* A, const __nv_bfloat16* B,
   wmma::fragment<wmma::accumulator,16,16,16,float>(&acc)[ROWS/16][2], int warp){
  constexpr int RT=ROWS/16;
  constexpr int KT=M2/16;
  #pragma unroll
  for(int rt=0;rt<RT;rt++){
    #pragma unroll
    for(int ct=0;ct<2;ct++){
      int col0=warp*32+ct*16;
      #pragma unroll
      for(int kt=0;kt<KT;kt++){
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> bf;
        wmma::load_matrix_sync(af, A + (rt*16)*M2 + kt*16, M2);
        wmma::load_matrix_sync(bf, B + (kt*16)*128 + col0, 128);
        wmma::mma_sync(acc[rt][ct],af,bf,acc[rt][ct]);
      }
    }
  }
}

// dK,dV kernel: block over (bh, kv_block of 32 keys), loop query blocks of 64
__global__ void bwd_dkdv_kernel(const __nv_bfloat16* Q,const __nv_bfloat16* K,const __nv_bfloat16* V,
    const __nv_bfloat16* dO,const float* Lg,const float* Dg,__nv_bfloat16* dK,__nv_bfloat16* dV,int S){
  extern __shared__ char smem[];
  __nv_bfloat16* Ks=(__nv_bfloat16*)smem;
  __nv_bfloat16* Vs=Ks+32*128;
  __nv_bfloat16* Qs=Vs+32*128;
  __nv_bfloat16* dOs=Qs+64*128;
  float* sTf=(float*)(dOs+64*128);
  float* pTf=sTf+32*64;
  __nv_bfloat16* pTb=(__nv_bfloat16*)(pTf+32*64);
  __nv_bfloat16* dsTb=pTb+32*64;
  float* outf=(float*)(dsTb+32*64);
  float* Ls=outf+32*128;
  float* Ds=Ls+64;

  int bh=blockIdx.y;
  int kv_start=blockIdx.x*32;
  int tid=threadIdx.x;
  int warp=tid/32;
  float scale=rsqrtf(128.f);

  load_tile(K,Ks,kv_start,32,S,bh,tid);
  load_tile(V,Vs,kv_start,32,S,bh,tid);

  wmma::fragment<wmma::accumulator,16,16,16,float> dV_acc[2][2];
  wmma::fragment<wmma::accumulator,16,16,16,float> dK_acc[2][2];
  #pragma unroll
  for(int rt=0;rt<2;rt++)
    #pragma unroll
    for(int ct=0;ct<2;ct++){ wmma::fill_fragment(dV_acc[rt][ct],0.f); wmma::fill_fragment(dK_acc[rt][ct],0.f);}

  __syncthreads();

  int numq=(S+63)/64;
  for(int qb=0;qb<numq;qb++){
    int q_start=qb*64;
    __syncthreads();
    load_tile(Q,Qs,q_start,64,S,bh,tid);
    load_tile(dO,dOs,q_start,64,S,bh,tid);
    if(tid<64){int gm=q_start+tid; if(gm<S){Ls[tid]=Lg[(size_t)bh*S+gm];Ds[tid]=Dg[(size_t)bh*S+gm];}else{Ls[tid]=0.f;Ds[tid]=0.f;}}
    __syncthreads();
    mm_score<32,64>(Ks,Qs,sTf,warp);          // S^T[key,query]
    __syncthreads();
    for(int idx=tid;idx<32*64;idx+=128){int m=idx&63;int gm=q_start+m;float s=sTf[idx];float p=(gm<S)?__expf(scale*s-Ls[m]):0.f;pTf[idx]=p;pTb[idx]=__float2bfloat16(p);}
    __syncthreads();
    mm_score<32,64>(Vs,dOs,sTf,warp);          // dP^T[key,query]
    __syncthreads();
    for(int idx=tid;idx<32*64;idx+=128){int m=idx&63;float dp=sTf[idx];float p=pTf[idx];float ds=scale*p*(dp-Ds[m]);dsTb[idx]=__float2bfloat16(ds);}
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
  for(int idx=tid;idx<32*128;idx+=128){int row=idx>>7;int k=idx&127;int gr=kv_start+row;if(gr<S)dV[((size_t)bh*S+gr)*128+k]=__float2bfloat16(outf[idx]);}
  __syncthreads();
  #pragma unroll
  for(int rt=0;rt<2;rt++)
   #pragma unroll
   for(int ct=0;ct<2;ct++){int col0=warp*32+ct*16;wmma::store_matrix_sync(outf+(rt*16)*128+col0,dK_acc[rt][ct],128,wmma::mem_row_major);}
  __syncthreads();
  for(int idx=tid;idx<32*128;idx+=128){int row=idx>>7;int k=idx&127;int gr=kv_start+row;if(gr<S)dK[((size_t)bh*S+gr)*128+k]=__float2bfloat16(outf[idx]);}
}

// dQ kernel: block over (bh, q_block of 32 queries), loop key blocks of 64
__global__ void bwd_dq_kernel(const __nv_bfloat16* Q,const __nv_bfloat16* K,const __nv_bfloat16* V,
    const __nv_bfloat16* dO,const float* Lg,const float* Dg,__nv_bfloat16* dQ,int S){
  extern __shared__ char smem[];
  __nv_bfloat16* Ks=(__nv_bfloat16*)smem;
  __nv_bfloat16* Vs=Ks+64*128;
  __nv_bfloat16* Qs=Vs+64*128;
  __nv_bfloat16* dOs=Qs+32*128;
  float* sf=(float*)(dOs+32*128);
  float* pf=sf+32*64;
  __nv_bfloat16* dSb=(__nv_bfloat16*)(pf+32*64);
  float* outf=(float*)(dSb+32*64);
  float* Ls=outf+32*128;
  float* Ds=Ls+32;

  int bh=blockIdx.y;
  int q_start=blockIdx.x*32;
  int tid=threadIdx.x;
  int warp=tid/32;
  float scale=rsqrtf(128.f);

  load_tile(Q,Qs,q_start,32,S,bh,tid);
  load_tile(dO,dOs,q_start,32,S,bh,tid);
  if(tid<32){int gm=q_start+tid; if(gm<S){Ls[tid]=Lg[(size_t)bh*S+gm];Ds[tid]=Dg[(size_t)bh*S+gm];}else{Ls[tid]=0.f;Ds[tid]=0.f;}}

  wmma::fragment<wmma::accumulator,16,16,16,float> dQ_acc[2][2];
  #pragma unroll
  for(int rt=0;rt<2;rt++)
   #pragma unroll
   for(int ct=0;ct<2;ct++) wmma::fill_fragment(dQ_acc[rt][ct],0.f);

  __syncthreads();

  int numk=(S+63)/64;
  for(int kb=0;kb<numk;kb++){
    int k_start=kb*64;
    __syncthreads();
    load_tile(K,Ks,k_start,64,S,bh,tid);
    load_tile(V,Vs,k_start,64,S,bh,tid);
    __syncthreads();
    mm_score<32,64>(Qs,Ks,sf,warp);            // S[query,key]
    __syncthreads();
    for(int idx=tid;idx<32*64;idx+=128){int m=idx>>6;int n=idx&63;int gn=k_start+n;float s=sf[idx];float p=(gn<S)?__expf(scale*s-Ls[m]):0.f;pf[idx]=p;}
    __syncthreads();
    mm_score<32,64>(dOs,Vs,sf,warp);            // dP[query,key]
    __syncthreads();
    for(int idx=tid;idx<32*64;idx+=128){int m=idx>>6;float dp=sf[idx];float p=pf[idx];float ds=scale*p*(dp-Ds[m]);dSb[idx]=__float2bfloat16(ds);}
    __syncthreads();
    mm_out<32,64>(dSb,Ks,dQ_acc,warp);
  }
  __syncthreads();
  #pragma unroll
  for(int rt=0;rt<2;rt++)
   #pragma unroll
   for(int ct=0;ct<2;ct++){int col0=warp*32+ct*16;wmma::store_matrix_sync(outf+(rt*16)*128+col0,dQ_acc[rt][ct],128,wmma::mem_row_major);}
  __syncthreads();
  for(int idx=tid;idx<32*128;idx+=128){int row=idx>>7;int k=idx&127;int gr=q_start+row;if(gr<S)dQ[((size_t)bh*S+gr)*128+k]=__float2bfloat16(outf[idx]);}
}

static float* g_Dbuf=nullptr;
static size_t g_Dsize=0;

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=Q.size(0),H=Q.size(1),S=Q.size(2);
  int BH=B*H;

  const __nv_bfloat16* Qp=(const __nv_bfloat16*)Q.data_ptr();
  const __nv_bfloat16* Kp=(const __nv_bfloat16*)K.data_ptr();
  const __nv_bfloat16* Vp=(const __nv_bfloat16*)V.data_ptr();
  const __nv_bfloat16* Op=(const __nv_bfloat16*)O.data_ptr();
  const __nv_bfloat16* dOp=(const __nv_bfloat16*)dO.data_ptr();
  const float* Lp=(const float*)L.data_ptr();
  __nv_bfloat16* dQp=(__nv_bfloat16*)dQ.data_ptr();
  __nv_bfloat16* dKp=(__nv_bfloat16*)dK.data_ptr();
  __nv_bfloat16* dVp=(__nv_bfloat16*)dV.data_ptr();

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

  int sh_dkdv=90624;
  int sh_dq=86272;
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