#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)

namespace mha_bwd {

constexpr int D  = 128;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int THREADS = 256; // 8 warps

using tvm::ffi::TensorView;

__device__ __forceinline__ uint32_t pk(__nv_bfloat16 a, __nv_bfloat16 b){
  __nv_bfloat162 v = __halves2bfloat162(a,b);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ void mma(float&d0,float&d1,float&d2,float&d3,
    uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

// A[m][k] = base[(rb+m)*ld + (kb+k)]
__device__ __forceinline__ void ldA(const __nv_bfloat16* base,int rb,int kb,int ld,int lane,
    uint32_t&a0,uint32_t&a1,uint32_t&a2,uint32_t&a3){
  int gid=lane>>2, kc=(lane&3)*2;
  const __nv_bfloat16* p0=base+(int64_t)(rb+gid)*ld+kb;
  const __nv_bfloat16* p1=base+(int64_t)(rb+gid+8)*ld+kb;
  a0=*reinterpret_cast<const uint32_t*>(p0+kc);
  a2=*reinterpret_cast<const uint32_t*>(p0+kc+8);
  a1=*reinterpret_cast<const uint32_t*>(p1+kc);
  a3=*reinterpret_cast<const uint32_t*>(p1+kc+8);
}
// B[k][n] = base[(nb+n)*ld + (kb+k)]   (col-major operand from row-major [n][k] storage)
__device__ __forceinline__ void ldBcol(const __nv_bfloat16* base,int nb,int kb,int ld,int lane,
    uint32_t&b0,uint32_t&b1){
  int gid=lane>>2, kc=(lane&3)*2;
  const __nv_bfloat16* p=base+(int64_t)(nb+gid)*ld+kb;
  b0=*reinterpret_cast<const uint32_t*>(p+kc);
  b1=*reinterpret_cast<const uint32_t*>(p+kc+8);
}
// B[k][n] = base[(kb+k)*ld + (nb+n)]   (k indexes rows of base)
__device__ __forceinline__ void ldBrow(const __nv_bfloat16* base,int nb,int kb,int ld,int lane,
    uint32_t&b0,uint32_t&b1){
  int gid=lane>>2, kc=(lane&3)*2;
  int n=nb+gid;
  b0=pk(base[(int64_t)(kb+kc)*ld+n],   base[(int64_t)(kb+kc+1)*ld+n]);
  b1=pk(base[(int64_t)(kb+kc+8)*ld+n], base[(int64_t)(kb+kc+9)*ld+n]);
}

__device__ __forceinline__ void loadTile(__nv_bfloat16* smem,const __nv_bfloat16* gptr,
    int row_start,int S,int tid){
  const int VPR = D*2/16; // 16 int4 per row
  #pragma unroll
  for(int idx=tid; idx<64*VPR; idx+=THREADS){
    int r=idx/VPR, v=idx%VPR;
    int gp=row_start+r;
    int4 data;
    if(gp<S) data=reinterpret_cast<const int4*>(gptr+(int64_t)gp*D)[v];
    else data=make_int4(0,0,0,0);
    reinterpret_cast<int4*>(smem+r*D)[v]=data;
  }
}

__global__ void delta_kernel(const __nv_bfloat16* __restrict__ O,
                             const __nv_bfloat16* __restrict__ dOg,
                             float* __restrict__ Delta,int64_t total_rows){
  int64_t row=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=total_rows) return;
  const __nv_bfloat16* o=O+row*D;
  const __nv_bfloat16* g=dOg+row*D;
  float s=0.f;
  #pragma unroll
  for(int c=0;c<D;c++) s+=__bfloat162float(o[c])*__bfloat162float(g[c]);
  Delta[row]=s;
}

// ===================== dK, dV kernel (KV outer) =====================
__global__ void dkv_kernel(
   const __nv_bfloat16* __restrict__ Q,const __nv_bfloat16* __restrict__ K,
   const __nv_bfloat16* __restrict__ V,const __nv_bfloat16* __restrict__ dOg,
   const float* __restrict__ Lmat,const float* __restrict__ Delta,
   __nv_bfloat16* __restrict__ dK,__nv_bfloat16* __restrict__ dV,
   int S,int H,float scale)
{
  extern __shared__ char smem[];
  __nv_bfloat16* sK=(__nv_bfloat16*)smem;
  __nv_bfloat16* sV=sK+BN*D;
  __nv_bfloat16* sQ=sV+BN*D;
  __nv_bfloat16* sdO=sQ+BM*D;
  __nv_bfloat16* sPt=sdO+BM*D;     // [BN][BM]
  __nv_bfloat16* sdSt=sPt+BN*BM;   // [BN][BM]
  float* sL=(float*)(sdSt+BN*BM);
  float* sD=sL+BM;

  int kvb=blockIdx.x,h=blockIdx.y,b=blockIdx.z;
  int kv_start=kvb*BN;
  if(kv_start>=S) return;

  int64_t head_off=((int64_t)(b*H+h))*S*D;
  const __nv_bfloat16* Qh=Q+head_off;
  const __nv_bfloat16* Kh=K+head_off;
  const __nv_bfloat16* Vh=V+head_off;
  const __nv_bfloat16* dOh=dOg+head_off;
  const float* Lh=Lmat+(int64_t)(b*H+h)*S;
  const float* Dh=Delta+(int64_t)(b*H+h)*S;
  __nv_bfloat16* dKh=dK+head_off;
  __nv_bfloat16* dVh=dV+head_off;

  int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
  int gid=lane>>2, t=lane&3;

  int row_base=(warp&3)*16;   // key rows (dV/dK)
  int col_base=(warp>>2)*64;  // d cols (dV/dK)
  int s_row=(warp&3)*16;      // key rows (score)
  int s_qb=(warp>>2)*32;      // query cols (score)

  loadTile(sK,Kh,kv_start,S,tid);
  loadTile(sV,Vh,kv_start,S,tid);

  float dVa[8][4],dKa[8][4];
  #pragma unroll
  for(int i=0;i<8;i++)
    #pragma unroll
    for(int j=0;j<4;j++){dVa[i][j]=0.f;dKa[i][j]=0.f;}

  __syncthreads();

  for(int q_start=kv_start;q_start<S;q_start+=BM){
    loadTile(sQ,Qh,q_start,S,tid);
    loadTile(sdO,dOh,q_start,S,tid);
    if(tid<BM){int qp=q_start+tid; sL[tid]=(qp<S)?Lh[qp]:0.f; sD[tid]=(qp<S)?Dh[qp]:0.f;}
    __syncthreads();

    // ---- S^T, dP^T ----
    float St[4][4],dPt[4][4];
    #pragma unroll
    for(int i=0;i<4;i++)
      #pragma unroll
      for(int j=0;j<4;j++){St[i][j]=0.f;dPt[i][j]=0.f;}
    #pragma unroll
    for(int ks=0;ks<8;ks++){
      int kk=ks*16;
      uint32_t aK0,aK1,aK2,aK3,aV0,aV1,aV2,aV3;
      ldA(sK,s_row,kk,D,lane,aK0,aK1,aK2,aK3);
      ldA(sV,s_row,kk,D,lane,aV0,aV1,aV2,aV3);
      #pragma unroll
      for(int nt=0;nt<4;nt++){
        int nb=s_qb+nt*8;
        uint32_t bQ0,bQ1,bO0,bO1;
        ldBcol(sQ,nb,kk,D,lane,bQ0,bQ1);
        ldBcol(sdO,nb,kk,D,lane,bO0,bO1);
        mma(St[nt][0],St[nt][1],St[nt][2],St[nt][3],aK0,aK1,aK2,aK3,bQ0,bQ1);
        mma(dPt[nt][0],dPt[nt][1],dPt[nt][2],dPt[nt][3],aV0,aV1,aV2,aV3,bO0,bO1);
      }
    }
    // ---- elementwise -> sPt, sdSt ----
    #pragma unroll
    for(int nt=0;nt<4;nt++){
      int key0=s_row+gid, key1=key0+8;
      int q0=s_qb+nt*8+2*t, q1=q0+1;
      float p,ds;
      int qg,kg;
      qg=q_start+q0; kg=kv_start+key0;
      p=(qg<S&&kg<S&&qg>=kg)?__expf(scale*St[nt][0]-sL[q0]):0.f; ds=p*(dPt[nt][0]-sD[q0]);
      sPt[key0*BM+q0]=__float2bfloat16(p); sdSt[key0*BM+q0]=__float2bfloat16(ds);
      qg=q_start+q1; kg=kv_start+key0;
      p=(qg<S&&kg<S&&qg>=kg)?__expf(scale*St[nt][1]-sL[q1]):0.f; ds=p*(dPt[nt][1]-sD[q1]);
      sPt[key0*BM+q1]=__float2bfloat16(p); sdSt[key0*BM+q1]=__float2bfloat16(ds);
      qg=q_start+q0; kg=kv_start+key1;
      p=(qg<S&&kg<S&&qg>=kg)?__expf(scale*St[nt][2]-sL[q0]):0.f; ds=p*(dPt[nt][2]-sD[q0]);
      sPt[key1*BM+q0]=__float2bfloat16(p); sdSt[key1*BM+q0]=__float2bfloat16(ds);
      qg=q_start+q1; kg=kv_start+key1;
      p=(qg<S&&kg<S&&qg>=kg)?__expf(scale*St[nt][3]-sL[q1]):0.f; ds=p*(dPt[nt][3]-sD[q1]);
      sPt[key1*BM+q1]=__float2bfloat16(p); sdSt[key1*BM+q1]=__float2bfloat16(ds);
    }
    __syncthreads();

    // ---- dV += P^T@dO ; dK += dS^T@Q (contract over queries) ----
    #pragma unroll
    for(int ks=0;ks<4;ks++){
      int kk=ks*16;
      uint32_t aP0,aP1,aP2,aP3,aS0,aS1,aS2,aS3;
      ldA(sPt,row_base,kk,BM,lane,aP0,aP1,aP2,aP3);
      ldA(sdSt,row_base,kk,BM,lane,aS0,aS1,aS2,aS3);
      #pragma unroll
      for(int nt=0;nt<8;nt++){
        int nb=col_base+nt*8;
        uint32_t bO0,bO1,bQ0,bQ1;
        ldBrow(sdO,nb,kk,D,lane,bO0,bO1);
        ldBrow(sQ,nb,kk,D,lane,bQ0,bQ1);
        mma(dVa[nt][0],dVa[nt][1],dVa[nt][2],dVa[nt][3],aP0,aP1,aP2,aP3,bO0,bO1);
        mma(dKa[nt][0],dKa[nt][1],dKa[nt][2],dKa[nt][3],aS0,aS1,aS2,aS3,bQ0,bQ1);
      }
    }
    __syncthreads();
  }

  // write
  #pragma unroll
  for(int nt=0;nt<8;nt++){
    int c0=col_base+nt*8+2*t, c1=c0+1;
    int key0=kv_start+row_base+gid, key1=key0+8;
    if(key0<S){
      dVh[(int64_t)key0*D+c0]=__float2bfloat16(dVa[nt][0]);
      dVh[(int64_t)key0*D+c1]=__float2bfloat16(dVa[nt][1]);
      dKh[(int64_t)key0*D+c0]=__float2bfloat16(scale*dKa[nt][0]);
      dKh[(int64_t)key0*D+c1]=__float2bfloat16(scale*dKa[nt][1]);
    }
    if(key1<S){
      dVh[(int64_t)key1*D+c0]=__float2bfloat16(dVa[nt][2]);
      dVh[(int64_t)key1*D+c1]=__float2bfloat16(dVa[nt][3]);
      dKh[(int64_t)key1*D+c0]=__float2bfloat16(scale*dKa[nt][2]);
      dKh[(int64_t)key1*D+c1]=__float2bfloat16(scale*dKa[nt][3]);
    }
  }
}

// ===================== dQ kernel (Q outer) =====================
__global__ void dq_kernel(
   const __nv_bfloat16* __restrict__ Q,const __nv_bfloat16* __restrict__ K,
   const __nv_bfloat16* __restrict__ V,const __nv_bfloat16* __restrict__ dOg,
   const float* __restrict__ Lmat,const float* __restrict__ Delta,
   __nv_bfloat16* __restrict__ dQ,int S,int H,float scale)
{
  extern __shared__ char smem[];
  __nv_bfloat16* sQ=(__nv_bfloat16*)smem;
  __nv_bfloat16* sdO=sQ+BM*D;
  __nv_bfloat16* sK=sdO+BM*D;
  __nv_bfloat16* sV=sK+BN*D;
  __nv_bfloat16* sdS=sV+BN*D;   // [BM][BN]
  float* sL=(float*)(sdS+BM*BN);
  float* sD=sL+BM;

  int qb=blockIdx.x,h=blockIdx.y,b=blockIdx.z;
  int q_start=qb*BM;
  if(q_start>=S) return;

  int64_t head_off=((int64_t)(b*H+h))*S*D;
  const __nv_bfloat16* Qh=Q+head_off;
  const __nv_bfloat16* Kh=K+head_off;
  const __nv_bfloat16* Vh=V+head_off;
  const __nv_bfloat16* dOh=dOg+head_off;
  const float* Lh=Lmat+(int64_t)(b*H+h)*S;
  const float* Dh=Delta+(int64_t)(b*H+h)*S;
  __nv_bfloat16* dQh=dQ+head_off;

  int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
  int gid=lane>>2, t=lane&3;

  int row_base=(warp&3)*16;   // query rows (dQ)
  int col_base=(warp>>2)*64;  // d cols (dQ)
  int s_qrow=(warp&3)*16;     // query rows (score)
  int s_kb=(warp>>2)*32;      // key cols (score)

  loadTile(sQ,Qh,q_start,S,tid);
  loadTile(sdO,dOh,q_start,S,tid);
  if(tid<BM){int qp=q_start+tid; sL[tid]=(qp<S)?Lh[qp]:0.f; sD[tid]=(qp<S)?Dh[qp]:0.f;}

  float dQa[8][4];
  #pragma unroll
  for(int i=0;i<8;i++)
    #pragma unroll
    for(int j=0;j<4;j++) dQa[i][j]=0.f;

  __syncthreads();

  for(int kv_start=0;kv_start<=q_start;kv_start+=BN){
    loadTile(sK,Kh,kv_start,S,tid);
    loadTile(sV,Vh,kv_start,S,tid);
    __syncthreads();

    // ---- S, dP ----
    float Sm[4][4],dP[4][4];
    #pragma unroll
    for(int i=0;i<4;i++)
      #pragma unroll
      for(int j=0;j<4;j++){Sm[i][j]=0.f;dP[i][j]=0.f;}
    #pragma unroll
    for(int ks=0;ks<8;ks++){
      int kk=ks*16;
      uint32_t aQ0,aQ1,aQ2,aQ3,aO0,aO1,aO2,aO3;
      ldA(sQ,s_qrow,kk,D,lane,aQ0,aQ1,aQ2,aQ3);
      ldA(sdO,s_qrow,kk,D,lane,aO0,aO1,aO2,aO3);
      #pragma unroll
      for(int nt=0;nt<4;nt++){
        int nb=s_kb+nt*8;
        uint32_t bK0,bK1,bV0,bV1;
        ldBcol(sK,nb,kk,D,lane,bK0,bK1);
        ldBcol(sV,nb,kk,D,lane,bV0,bV1);
        mma(Sm[nt][0],Sm[nt][1],Sm[nt][2],Sm[nt][3],aQ0,aQ1,aQ2,aQ3,bK0,bK1);
        mma(dP[nt][0],dP[nt][1],dP[nt][2],dP[nt][3],aO0,aO1,aO2,aO3,bV0,bV1);
      }
    }
    // ---- elementwise -> sdS ----
    #pragma unroll
    for(int nt=0;nt<4;nt++){
      int q0=s_qrow+gid, q1=q0+8;
      int k0=s_kb+nt*8+2*t, k1=k0+1;
      float p,ds;int qg,kg;
      qg=q_start+q0; kg=kv_start+k0;
      p=(qg<S&&kg<S&&qg>=kg)?__expf(scale*Sm[nt][0]-sL[q0]):0.f; ds=p*(dP[nt][0]-sD[q0]);
      sdS[q0*BN+k0]=__float2bfloat16(ds);
      qg=q_start+q0; kg=kv_start+k1;
      p=(qg<S&&kg<S&&qg>=kg)?__expf(scale*Sm[nt][1]-sL[q0]):0.f; ds=p*(dP[nt][1]-sD[q0]);
      sdS[q0*BN+k1]=__float2bfloat16(ds);
      qg=q_start+q1; kg=kv_start+k0;
      p=(qg<S&&kg<S&&qg>=kg)?__expf(scale*Sm[nt][2]-sL[q1]):0.f; ds=p*(dP[nt][2]-sD[q1]);
      sdS[q1*BN+k0]=__float2bfloat16(ds);
      qg=q_start+q1; kg=kv_start+k1;
      p=(qg<S&&kg<S&&qg>=kg)?__expf(scale*Sm[nt][3]-sL[q1]):0.f; ds=p*(dP[nt][3]-sD[q1]);
      sdS[q1*BN+k1]=__float2bfloat16(ds);
    }
    __syncthreads();

    // ---- dQ += dS@K (contract over keys) ----
    #pragma unroll
    for(int ks=0;ks<4;ks++){
      int kk=ks*16;
      uint32_t aD0,aD1,aD2,aD3;
      ldA(sdS,row_base,kk,BN,lane,aD0,aD1,aD2,aD3);
      #pragma unroll
      for(int nt=0;nt<8;nt++){
        int nb=col_base+nt*8;
        uint32_t bK0,bK1;
        ldBrow(sK,nb,kk,D,lane,bK0,bK1);
        mma(dQa[nt][0],dQa[nt][1],dQa[nt][2],dQa[nt][3],aD0,aD1,aD2,aD3,bK0,bK1);
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for(int nt=0;nt<8;nt++){
    int c0=col_base+nt*8+2*t, c1=c0+1;
    int q0=q_start+row_base+gid, q1=q0+8;
    if(q0<S){
      dQh[(int64_t)q0*D+c0]=__float2bfloat16(scale*dQa[nt][0]);
      dQh[(int64_t)q0*D+c1]=__float2bfloat16(scale*dQa[nt][1]);
    }
    if(q1<S){
      dQh[(int64_t)q1*D+c0]=__float2bfloat16(scale*dQa[nt][2]);
      dQh[(int64_t)q1*D+c1]=__float2bfloat16(scale*dQa[nt][3]);
    }
  }
}

void run(TensorView Q,TensorView K,TensorView V,TensorView O,TensorView dO,TensorView L,
         TensorView dQ,TensorView dK,TensorView dV){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0),H=(int)Q.size(1),S=(int)Q.size(2);
  float scale=1.0f/sqrtf((float)D);
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));

  auto Qp=(const __nv_bfloat16*)Q.data_ptr();
  auto Kp=(const __nv_bfloat16*)K.data_ptr();
  auto Vp=(const __nv_bfloat16*)V.data_ptr();
  auto Op=(const __nv_bfloat16*)O.data_ptr();
  auto dOp=(const __nv_bfloat16*)dO.data_ptr();
  auto Lp=(const float*)L.data_ptr();
  auto dQp=(__nv_bfloat16*)dQ.data_ptr();
  auto dKp=(__nv_bfloat16*)dK.data_ptr();
  auto dVp=(__nv_bfloat16*)dV.data_ptr();

  int64_t totalRows=(int64_t)B*H*S;
  float* Delta=nullptr;
  CUDA_CHECK(cudaMalloc(&Delta,totalRows*sizeof(float)));

  {
    int tpb=256; int64_t bl=(totalRows+tpb-1)/tpb;
    delta_kernel<<<(unsigned)bl,tpb,0,stream>>>(Op,dOp,Delta,totalRows);
  }

  size_t smem_dkv=(size_t)(2*BN+2*BM)*D*sizeof(__nv_bfloat16)+(size_t)2*BN*BM*sizeof(__nv_bfloat16)+2*BM*sizeof(float);
  size_t smem_dq =(size_t)(2*BM+2*BN)*D*sizeof(__nv_bfloat16)+(size_t)BM*BN*sizeof(__nv_bfloat16)+2*BM*sizeof(float);
  CUDA_CHECK(cudaFuncSetAttribute(dkv_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_dkv));
  CUDA_CHECK(cudaFuncSetAttribute(dq_kernel ,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_dq));

  int num=(S+BN-1)/BN;
  dim3 grid((unsigned)num,(unsigned)H,(unsigned)B);
  dkv_kernel<<<grid,THREADS,smem_dkv,stream>>>(Qp,Kp,Vp,dOp,Lp,Delta,dKp,dVp,S,H,scale);
  dq_kernel <<<grid,THREADS,smem_dq ,stream>>>(Qp,Kp,Vp,dOp,Lp,Delta,dQp,S,H,scale);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFree(Delta));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd