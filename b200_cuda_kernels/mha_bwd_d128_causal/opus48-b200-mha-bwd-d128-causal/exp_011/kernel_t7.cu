#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);exit(1);} } while(0)

namespace mha_bwd {
using bf16 = __nv_bfloat16;
using tvm::ffi::TensorView;

constexpr int HD = 128;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int NTHREAD = 256;
constexpr int LDD = HD + 8;   // 136
constexpr int LDN = 64 + 8;   // 72

__device__ __forceinline__ uint32_t ld_u32(const void* p){ return *reinterpret_cast<const uint32_t*>(p); }

__device__ __forceinline__ void ldm_x4(const half* S,int ld,int r0,int c0,int lane,uint32_t a[4]){
  int quad=lane>>3, r=lane&7;
  int rr=r0+((quad&1)?8:0)+r;
  int cc=c0+((quad>=2)?8:0);
  uint32_t addr=(uint32_t)__cvta_generic_to_shared(&S[rr*ld+cc]);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
    :"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(addr));
}
__device__ __forceinline__ void ldm_x2t(const half* S,int ld,int r0,int c0,int lane,uint32_t b[2]){
  int mmat=(lane>>3)&1, mrow=lane&7;
  int rr=r0+mmat*8+mrow;
  uint32_t addr=(uint32_t)__cvta_generic_to_shared(&S[rr*ld+c0]);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
    :"=r"(b[0]),"=r"(b[1]):"r"(addr));
}
__device__ __forceinline__ void loadB1(const half* S,int ld,int k0,int n0,int lane,uint32_t b[2]){
  int gid=lane>>2, tig=lane&3;
  b[0]=ld_u32(&S[(n0+gid)*ld + k0+2*tig]);
  b[1]=ld_u32(&S[(n0+gid)*ld + k0+8+2*tig]);
}
__device__ __forceinline__ void mma_f16(float c[4],const uint32_t a[4],const uint32_t b[2]){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
    :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

__device__ __forceinline__ void load_cvt8(half* dst,const bf16* src,bool valid){
  half tmp[8];
  if(valid){
    int4 raw=*(const int4*)src;
    const bf16* b=(const bf16*)&raw;
    #pragma unroll
    for(int e=0;e<8;e++) tmp[e]=__float2half(__bfloat162float(b[e]));
  } else {
    #pragma unroll
    for(int e=0;e<8;e++) tmp[e]=__float2half(0.f);
  }
  *(int4*)dst=*(int4*)tmp;
}

__global__ void delta_kernel(const bf16* __restrict__ dO,const bf16* __restrict__ O,
                             float* __restrict__ Delta,int64_t total_rows){
  int64_t row=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(row<total_rows){
    const bf16* d=dO+row*HD; const bf16* o=O+row*HD;
    float acc=0.f;
    #pragma unroll
    for(int e=0;e<HD;e++) acc+=__bfloat162float(d[e])*__bfloat162float(o[e]);
    Delta[row]=acc;
  }
}

__launch_bounds__(256,2)
__global__ void dkv_kernel(
    const bf16* __restrict__ Q,const bf16* __restrict__ K,const bf16* __restrict__ V,
    const bf16* __restrict__ dO,const float* __restrict__ L,const float* __restrict__ Delta,
    bf16* __restrict__ dK,bf16* __restrict__ dV,int S,float scale){
  extern __shared__ char smem[];
  half* sK=(half*)smem;
  half* sV=sK+BN*LDD;
  half* sQ=sV+BN*LDD;
  half* sdO=sQ+BM*LDD;
  half* sP=sdO+BM*LDD;
  half* sSS=sP+BN*LDN;
  float* sL=(float*)(sSS+BN*LDN);
  float* sD=sL+BM;

  int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
  int gid=lane>>2, tig=lane&3;
  int m_sub=warp&3, hlf=warp>>2;
  int kb=blockIdx.x, bh=blockIdx.y;
  int k0=kb*BN;
  int64_t base=(int64_t)bh*S*HD, Lbase=(int64_t)bh*S;
  const bf16* Kb=K+base; const bf16* Vb=V+base;
  const bf16* Qb=Q+base; const bf16* dOb=dO+base;

  #pragma unroll
  for(int c=0;c<4;c++){
    int idx=tid+c*256; int row=idx>>4, col8=idx&15; int gk=k0+row;
    load_cvt8(&sK[row*LDD+col8*8], &Kb[(int64_t)gk*HD+col8*8], gk<S);
    load_cvt8(&sV[row*LDD+col8*8], &Vb[(int64_t)gk*HD+col8*8], gk<S);
  }

  float dv[8][4]; float dk[8][4];
  #pragma unroll
  for(int i=0;i<8;i++)
    #pragma unroll
    for(int j=0;j<4;j++){ dv[i][j]=0.f; dk[i][j]=0.f; }

  int num_q=(S+BM-1)/BM;
  for(int qb=kb; qb<num_q; ++qb){
    int q0=qb*BM;
    __syncthreads();
    #pragma unroll
    for(int c=0;c<4;c++){
      int idx=tid+c*256; int row=idx>>4, col8=idx&15; int gq=q0+row;
      load_cvt8(&sQ[row*LDD+col8*8], &Qb[(int64_t)gq*HD+col8*8], gq<S);
      load_cvt8(&sdO[row*LDD+col8*8], &dOb[(int64_t)gq*HD+col8*8], gq<S);
    }
    if(tid<BM){ int gq=q0+tid; sL[tid]=(gq<S)?L[Lbase+gq]:0.f; sD[tid]=(gq<S)?Delta[Lbase+gq]:0.f; }
    __syncthreads();

    float sfrag[4][4]; float dpfrag[4][4];
    #pragma unroll
    for(int nt=0;nt<4;nt++)
      #pragma unroll
      for(int c=0;c<4;c++){ sfrag[nt][c]=0.f; dpfrag[nt][c]=0.f; }
    #pragma unroll
    for(int kt=0;kt<8;kt++){
      uint32_t aK[4],aV[4];
      ldm_x4(sK,LDD,16*m_sub,16*kt,lane,aK);
      ldm_x4(sV,LDD,16*m_sub,16*kt,lane,aV);
      #pragma unroll
      for(int nt=0;nt<4;nt++){
        uint32_t bQ[2],bO[2];
        loadB1(sQ,LDD,16*kt,32*hlf+8*nt,lane,bQ);
        loadB1(sdO,LDD,16*kt,32*hlf+8*nt,lane,bO);
        mma_f16(sfrag[nt],aK,bQ);
        mma_f16(dpfrag[nt],aV,bO);
      }
    }
    #pragma unroll
    for(int nt=0;nt<4;nt++){
      #pragma unroll
      for(int cc=0;cc<4;cc++){
        int i=32*hlf+8*nt+2*tig+(cc&1);
        int j=16*m_sub+gid+((cc>=2)?8:0);
        int gq=q0+i, gk=k0+j;
        bool valid=(gq<S)&&(gk<S)&&(gq>=gk);
        float pt = valid? __expf(scale*sfrag[nt][cc]-sL[i]) : 0.f;
        float ds = valid? pt*(dpfrag[nt][cc]-sD[i]) : 0.f;
        int off=j*LDN+i;
        sP[off]=__float2half(pt); sSS[off]=__float2half(ds);
      }
    }
    __syncthreads();
    #pragma unroll
    for(int kt=0;kt<4;kt++){
      uint32_t aP[4],aS[4];
      ldm_x4(sP,LDN,16*m_sub,16*kt,lane,aP);
      ldm_x4(sSS,LDN,16*m_sub,16*kt,lane,aS);
      #pragma unroll
      for(int nt=0;nt<8;nt++){
        uint32_t bO[2],bQ[2];
        ldm_x2t(sdO,LDD,16*kt,64*hlf+8*nt,lane,bO);
        ldm_x2t(sQ ,LDD,16*kt,64*hlf+8*nt,lane,bQ);
        mma_f16(dv[nt],aP,bO);
        mma_f16(dk[nt],aS,bQ);
      }
    }
  }
  #pragma unroll
  for(int nt=0;nt<8;nt++){
    #pragma unroll
    for(int cc=0;cc<4;cc++){
      int j=16*m_sub+gid+((cc>=2)?8:0);
      int e=64*hlf+8*nt+2*tig+(cc&1);
      int gk=k0+j;
      if(gk<S){
        dK[base+(int64_t)gk*HD+e]=__float2bfloat16(dk[nt][cc]*scale);
        dV[base+(int64_t)gk*HD+e]=__float2bfloat16(dv[nt][cc]);
      }
    }
  }
}

__launch_bounds__(256,2)
__global__ void dq_kernel(
    const bf16* __restrict__ Q,const bf16* __restrict__ K,const bf16* __restrict__ V,
    const bf16* __restrict__ dO,const float* __restrict__ L,const float* __restrict__ Delta,
    bf16* __restrict__ dQ,int S,float scale){
  extern __shared__ char smem[];
  half* sQ=(half*)smem;
  half* sdO=sQ+BM*LDD;
  half* sK=sdO+BM*LDD;
  half* sV=sK+BN*LDD;
  half* sSS=sV+BN*LDD;
  float* sL=(float*)(sSS+BM*LDN);
  float* sD=sL+BM;

  int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
  int gid=lane>>2, tig=lane&3;
  int m_sub=warp&3, hlf=warp>>2;
  int qb=blockIdx.x, bh=blockIdx.y;
  int q0=qb*BM;
  int64_t base=(int64_t)bh*S*HD, Lbase=(int64_t)bh*S;
  const bf16* Kb=K+base; const bf16* Vb=V+base;
  const bf16* Qb=Q+base; const bf16* dOb=dO+base;

  #pragma unroll
  for(int c=0;c<4;c++){
    int idx=tid+c*256; int row=idx>>4, col8=idx&15; int gq=q0+row;
    load_cvt8(&sQ[row*LDD+col8*8], &Qb[(int64_t)gq*HD+col8*8], gq<S);
    load_cvt8(&sdO[row*LDD+col8*8], &dOb[(int64_t)gq*HD+col8*8], gq<S);
  }
  if(tid<BM){ int gq=q0+tid; sL[tid]=(gq<S)?L[Lbase+gq]:0.f; sD[tid]=(gq<S)?Delta[Lbase+gq]:0.f; }

  float dq[8][4];
  #pragma unroll
  for(int i=0;i<8;i++)
    #pragma unroll
    for(int j=0;j<4;j++) dq[i][j]=0.f;

  for(int kb=0;kb<=qb;++kb){
    int k0=kb*BN;
    __syncthreads();
    #pragma unroll
    for(int c=0;c<4;c++){
      int idx=tid+c*256; int row=idx>>4, col8=idx&15; int gk=k0+row;
      load_cvt8(&sK[row*LDD+col8*8], &Kb[(int64_t)gk*HD+col8*8], gk<S);
      load_cvt8(&sV[row*LDD+col8*8], &Vb[(int64_t)gk*HD+col8*8], gk<S);
    }
    __syncthreads();

    float sfrag[4][4]; float dpfrag[4][4];
    #pragma unroll
    for(int nt=0;nt<4;nt++)
      #pragma unroll
      for(int c=0;c<4;c++){ sfrag[nt][c]=0.f; dpfrag[nt][c]=0.f; }
    #pragma unroll
    for(int kt=0;kt<8;kt++){
      uint32_t aQ[4],aO[4];
      ldm_x4(sQ,LDD,16*m_sub,16*kt,lane,aQ);
      ldm_x4(sdO,LDD,16*m_sub,16*kt,lane,aO);
      #pragma unroll
      for(int nt=0;nt<4;nt++){
        uint32_t bK[2],bV[2];
        loadB1(sK,LDD,16*kt,32*hlf+8*nt,lane,bK);
        loadB1(sV,LDD,16*kt,32*hlf+8*nt,lane,bV);
        mma_f16(sfrag[nt],aQ,bK);
        mma_f16(dpfrag[nt],aO,bV);
      }
    }
    #pragma unroll
    for(int nt=0;nt<4;nt++){
      #pragma unroll
      for(int cc=0;cc<4;cc++){
        int i=16*m_sub+gid+((cc>=2)?8:0);
        int j=32*hlf+8*nt+2*tig+(cc&1);
        int gq=q0+i, gk=k0+j;
        bool valid=(gq<S)&&(gk<S)&&(gq>=gk);
        float pt = valid? __expf(scale*sfrag[nt][cc]-sL[i]) : 0.f;
        float ds = valid? pt*(dpfrag[nt][cc]-sD[i]) : 0.f;
        sSS[i*LDN+j]=__float2half(ds);
      }
    }
    __syncthreads();
    #pragma unroll
    for(int kt=0;kt<4;kt++){
      uint32_t aS[4];
      ldm_x4(sSS,LDN,16*m_sub,16*kt,lane,aS);
      #pragma unroll
      for(int nt=0;nt<8;nt++){
        uint32_t bK[2];
        ldm_x2t(sK,LDD,16*kt,64*hlf+8*nt,lane,bK);
        mma_f16(dq[nt],aS,bK);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int nt=0;nt<8;nt++){
    #pragma unroll
    for(int cc=0;cc<4;cc++){
      int i=16*m_sub+gid+((cc>=2)?8:0);
      int e=64*hlf+8*nt+2*tig+(cc&1);
      int gq=q0+i;
      if(gq<S) dQ[base+(int64_t)gq*HD+e]=__float2bfloat16(dq[nt][cc]*scale);
    }
  }
}

void run(TensorView Q,TensorView K,TensorView V,TensorView O,TensorView dO,TensorView L,
         TensorView dQ,TensorView dK,TensorView dV){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2);
  int64_t BH=(int64_t)B*H;

  const bf16* Qp=(const bf16*)Q.data_ptr();
  const bf16* Kp=(const bf16*)K.data_ptr();
  const bf16* Vp=(const bf16*)V.data_ptr();
  const bf16* Op=(const bf16*)O.data_ptr();
  const bf16* dOp=(const bf16*)dO.data_ptr();
  const float* Lp=(const float*)L.data_ptr();
  bf16* dQp=(bf16*)dQ.data_ptr();
  bf16* dKp=(bf16*)dK.data_ptr();
  bf16* dVp=(bf16*)dV.data_ptr();

  float scale=1.0f/sqrtf((float)HD);
  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id);

  float* Delta=nullptr;
  CUDA_CHECK(cudaMallocAsync(&Delta,(size_t)BH*S*sizeof(float),stream));

  int64_t total_rows=BH*S;
  int dthreads=256;
  int64_t dblocks=(total_rows+dthreads-1)/dthreads;
  delta_kernel<<<(unsigned)dblocks,dthreads,0,stream>>>(dOp,Op,Delta,total_rows);
  CUDA_CHECK(cudaGetLastError());

  int smem_dkv = (4*BN*LDD + 2*BN*LDN)*(int)sizeof(half) + 2*BM*(int)sizeof(float);
  int smem_dq  = (4*BN*LDD + BM*LDN)*(int)sizeof(half) + 2*BM*(int)sizeof(float);
  CUDA_CHECK(cudaFuncSetAttribute(dkv_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem_dkv));
  CUDA_CHECK(cudaFuncSetAttribute(dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,smem_dq));

  int num_kv=(S+BN-1)/BN;
  int num_q =(S+BM-1)/BM;
  dim3 gkv((unsigned)num_kv,(unsigned)BH);
  dim3 gq ((unsigned)num_q, (unsigned)BH);

  dkv_kernel<<<gkv,NTHREAD,smem_dkv,stream>>>(Qp,Kp,Vp,dOp,Lp,Delta,dKp,dVp,S,scale);
  CUDA_CHECK(cudaGetLastError());
  dq_kernel<<<gq,NTHREAD,smem_dq,stream>>>(Qp,Kp,Vp,dOp,Lp,Delta,dQp,S,scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFreeAsync(Delta,stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd