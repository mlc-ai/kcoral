#include <cuda_bf16.h>
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
constexpr int BM = 64;   // queries per block
constexpr int BN = 64;   // keys per block
constexpr int NTHREAD = 256;

__device__ __forceinline__ uint32_t ld_u32(const void* p){ return *reinterpret_cast<const uint32_t*>(p); }

__device__ __forceinline__ void loadA(const bf16* S,int ld,int row0,int col0,int gid,int tig,uint32_t a[4]){
  a[0]=ld_u32(&S[(row0+gid)*ld + col0+2*tig]);
  a[1]=ld_u32(&S[(row0+gid+8)*ld + col0+2*tig]);
  a[2]=ld_u32(&S[(row0+gid)*ld + col0+8+2*tig]);
  a[3]=ld_u32(&S[(row0+gid+8)*ld + col0+8+2*tig]);
}
// contiguous B (contract inner dim): b[k][n] = SRC[n][k]
__device__ __forceinline__ void loadB1(const bf16* S,int ld,int k0,int n0,int gid,int tig,uint32_t b[2]){
  b[0]=ld_u32(&S[(n0+gid)*ld + k0+2*tig]);
  b[1]=ld_u32(&S[(n0+gid)*ld + k0+8+2*tig]);
}
// strided B (contract outer dim): b[k][n] = SRC[k][n]
__device__ __forceinline__ void loadB2(const bf16* S,int ld,int k0,int n0,int gid,int tig,uint32_t b[2]){
  uint16_t lo0=*(const uint16_t*)&S[(k0+2*tig)*ld + n0+gid];
  uint16_t hi0=*(const uint16_t*)&S[(k0+2*tig+1)*ld + n0+gid];
  b[0]=(uint32_t)lo0 | ((uint32_t)hi0<<16);
  uint16_t lo1=*(const uint16_t*)&S[(k0+8+2*tig)*ld + n0+gid];
  uint16_t hi1=*(const uint16_t*)&S[(k0+9+2*tig)*ld + n0+gid];
  b[1]=(uint32_t)lo1 | ((uint32_t)hi1<<16);
}
__device__ __forceinline__ void mma16816(float c[4],const uint32_t a[4],const uint32_t b[2]){
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
    :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
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

__launch_bounds__(256)
__global__ void dkv_kernel(
    const bf16* __restrict__ Q,const bf16* __restrict__ K,const bf16* __restrict__ V,
    const bf16* __restrict__ dO,const float* __restrict__ L,const float* __restrict__ Delta,
    bf16* __restrict__ dK,bf16* __restrict__ dV,int S,float scale){
  extern __shared__ char smem[];
  bf16* sK=(bf16*)smem;
  bf16* sV=sK+BN*HD;
  bf16* sQ=sV+BN*HD;
  bf16* sdO=sQ+BM*HD;
  bf16* sP=sdO+BM*HD;
  bf16* sdS=sP+BN*BM;
  float* sL=(float*)(sdS+BN*BM);
  float* sD=sL+BM;

  int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
  int gid=lane>>2, tig=lane&3;
  int m_sub=warp&3, half=warp>>2;
  int kb=blockIdx.x, bh=blockIdx.y;
  int k0=kb*BN;
  int64_t base=(int64_t)bh*S*HD, Lbase=(int64_t)bh*S;
  const bf16* Kb=K+base; const bf16* Vb=V+base;
  const bf16* Qb=Q+base; const bf16* dOb=dO+base;

  // load K,V
  #pragma unroll
  for(int c=0;c<4;c++){
    int idx=tid+c*256; int row=idx>>4, col8=idx&15; int gk=k0+row;
    int4 vk = (gk<S)? *(const int4*)&Kb[(int64_t)gk*HD+col8*8] : make_int4(0,0,0,0);
    int4 vv = (gk<S)? *(const int4*)&Vb[(int64_t)gk*HD+col8*8] : make_int4(0,0,0,0);
    *(int4*)&sK[row*HD+col8*8]=vk;
    *(int4*)&sV[row*HD+col8*8]=vv;
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
      int4 vq =(gq<S)? *(const int4*)&Qb[(int64_t)gq*HD+col8*8] : make_int4(0,0,0,0);
      int4 vo =(gq<S)? *(const int4*)&dOb[(int64_t)gq*HD+col8*8] : make_int4(0,0,0,0);
      *(int4*)&sQ[row*HD+col8*8]=vq;
      *(int4*)&sdO[row*HD+col8*8]=vo;
    }
    if(tid<BM){ int gq=q0+tid; sL[tid]=(gq<S)?L[Lbase+gq]:0.f; sD[tid]=(gq<S)?Delta[Lbase+gq]:0.f; }
    __syncthreads();

    // mma1: S^T = K@Q^T ; mma2: dP^T = V@dO^T
    float sfrag[4][4]; float dpfrag[4][4];
    #pragma unroll
    for(int nt=0;nt<4;nt++)
      #pragma unroll
      for(int c=0;c<4;c++){ sfrag[nt][c]=0.f; dpfrag[nt][c]=0.f; }
    #pragma unroll
    for(int kt=0;kt<8;kt++){
      uint32_t aK[4],aV[4];
      loadA(sK,HD,16*m_sub,16*kt,gid,tig,aK);
      loadA(sV,HD,16*m_sub,16*kt,gid,tig,aV);
      #pragma unroll
      for(int nt=0;nt<4;nt++){
        uint32_t bQ[2],bO[2];
        loadB1(sQ,HD,16*kt,32*half+8*nt,gid,tig,bQ);
        loadB1(sdO,HD,16*kt,32*half+8*nt,gid,tig,bO);
        mma16816(sfrag[nt],aK,bQ);
        mma16816(dpfrag[nt],aV,bO);
      }
    }
    // elementwise -> sP, sdS
    #pragma unroll
    for(int nt=0;nt<4;nt++){
      #pragma unroll
      for(int cc=0;cc<4;cc++){
        int i=32*half+8*nt+2*tig+(cc&1);
        int j=16*m_sub+gid+((cc>=2)?8:0);
        int gq=q0+i, gk=k0+j;
        bool valid=(gq<S)&&(gk<S)&&(gq>=gk);
        float pt = valid? __expf(scale*sfrag[nt][cc]-sL[i]) : 0.f;
        float ds = valid? pt*(dpfrag[nt][cc]-sD[i]) : 0.f;
        sP[j*BM+i]=__float2bfloat16(pt);
        sdS[j*BM+i]=__float2bfloat16(ds);
      }
    }
    __syncthreads();
    // mma3: dV += P^T@dO ; mma4: dK += dS^T@Q
    #pragma unroll
    for(int kt=0;kt<4;kt++){
      uint32_t aP[4],aS[4];
      loadA(sP,BM,16*m_sub,16*kt,gid,tig,aP);
      loadA(sdS,BM,16*m_sub,16*kt,gid,tig,aS);
      #pragma unroll
      for(int nt=0;nt<8;nt++){
        uint32_t bO[2],bQ[2];
        loadB2(sdO,HD,16*kt,64*half+8*nt,gid,tig,bO);
        loadB2(sQ,HD,16*kt,64*half+8*nt,gid,tig,bQ);
        mma16816(dv[nt],aP,bO);
        mma16816(dk[nt],aS,bQ);
      }
    }
  }
  // write out
  #pragma unroll
  for(int nt=0;nt<8;nt++){
    #pragma unroll
    for(int cc=0;cc<4;cc++){
      int j=16*m_sub+gid+((cc>=2)?8:0);
      int e=64*half+8*nt+2*tig+(cc&1);
      int gk=k0+j;
      if(gk<S){
        dK[base+(int64_t)gk*HD+e]=__float2bfloat16(dk[nt][cc]*scale);
        dV[base+(int64_t)gk*HD+e]=__float2bfloat16(dv[nt][cc]);
      }
    }
  }
}

__launch_bounds__(256)
__global__ void dq_kernel(
    const bf16* __restrict__ Q,const bf16* __restrict__ K,const bf16* __restrict__ V,
    const bf16* __restrict__ dO,const float* __restrict__ L,const float* __restrict__ Delta,
    bf16* __restrict__ dQ,int S,float scale){
  extern __shared__ char smem[];
  bf16* sQ=(bf16*)smem;
  bf16* sdO=sQ+BM*HD;
  bf16* sK=sdO+BM*HD;
  bf16* sV=sK+BN*HD;
  bf16* sdS=sV+BN*HD;
  float* sL=(float*)(sdS+BM*BN);
  float* sD=sL+BM;

  int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
  int gid=lane>>2, tig=lane&3;
  int m_sub=warp&3, half=warp>>2;
  int qb=blockIdx.x, bh=blockIdx.y;
  int q0=qb*BM;
  int64_t base=(int64_t)bh*S*HD, Lbase=(int64_t)bh*S;
  const bf16* Kb=K+base; const bf16* Vb=V+base;
  const bf16* Qb=Q+base; const bf16* dOb=dO+base;

  #pragma unroll
  for(int c=0;c<4;c++){
    int idx=tid+c*256; int row=idx>>4, col8=idx&15; int gq=q0+row;
    int4 vq=(gq<S)? *(const int4*)&Qb[(int64_t)gq*HD+col8*8] : make_int4(0,0,0,0);
    int4 vo=(gq<S)? *(const int4*)&dOb[(int64_t)gq*HD+col8*8] : make_int4(0,0,0,0);
    *(int4*)&sQ[row*HD+col8*8]=vq;
    *(int4*)&sdO[row*HD+col8*8]=vo;
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
      int4 vk=(gk<S)? *(const int4*)&Kb[(int64_t)gk*HD+col8*8] : make_int4(0,0,0,0);
      int4 vv=(gk<S)? *(const int4*)&Vb[(int64_t)gk*HD+col8*8] : make_int4(0,0,0,0);
      *(int4*)&sK[row*HD+col8*8]=vk;
      *(int4*)&sV[row*HD+col8*8]=vv;
    }
    __syncthreads();

    // mma1: S = Q@K^T ; mma2: dP = dO@V^T
    float sfrag[4][4]; float dpfrag[4][4];
    #pragma unroll
    for(int nt=0;nt<4;nt++)
      #pragma unroll
      for(int c=0;c<4;c++){ sfrag[nt][c]=0.f; dpfrag[nt][c]=0.f; }
    #pragma unroll
    for(int kt=0;kt<8;kt++){
      uint32_t aQ[4],aO[4];
      loadA(sQ,HD,16*m_sub,16*kt,gid,tig,aQ);
      loadA(sdO,HD,16*m_sub,16*kt,gid,tig,aO);
      #pragma unroll
      for(int nt=0;nt<4;nt++){
        uint32_t bK[2],bV[2];
        loadB1(sK,HD,16*kt,32*half+8*nt,gid,tig,bK);
        loadB1(sV,HD,16*kt,32*half+8*nt,gid,tig,bV);
        mma16816(sfrag[nt],aQ,bK);
        mma16816(dpfrag[nt],aO,bV);
      }
    }
    // elementwise -> sdS[i][j]
    #pragma unroll
    for(int nt=0;nt<4;nt++){
      #pragma unroll
      for(int cc=0;cc<4;cc++){
        int i=16*m_sub+gid+((cc>=2)?8:0);   // query
        int j=32*half+8*nt+2*tig+(cc&1);    // key
        int gq=q0+i, gk=k0+j;
        bool valid=(gq<S)&&(gk<S)&&(gq>=gk);
        float pt = valid? __expf(scale*sfrag[nt][cc]-sL[i]) : 0.f;
        float ds = valid? pt*(dpfrag[nt][cc]-sD[i]) : 0.f;
        sdS[i*BN+j]=__float2bfloat16(ds);
      }
    }
    __syncthreads();
    // mma3: dQ += dS@K
    #pragma unroll
    for(int kt=0;kt<4;kt++){
      uint32_t aS[4];
      loadA(sdS,BN,16*m_sub,16*kt,gid,tig,aS);
      #pragma unroll
      for(int nt=0;nt<8;nt++){
        uint32_t bK[2];
        loadB2(sK,HD,16*kt,64*half+8*nt,gid,tig,bK);
        mma16816(dq[nt],aS,bK);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int nt=0;nt<8;nt++){
    #pragma unroll
    for(int cc=0;cc<4;cc++){
      int i=16*m_sub+gid+((cc>=2)?8:0);
      int e=64*half+8*nt+2*tig+(cc&1);
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

  int smem_dkv = (4*BN*HD + 2*BN*BM)*(int)sizeof(bf16) + 2*BM*(int)sizeof(float);
  int smem_dq  = (4*BN*HD + BM*BN)*(int)sizeof(bf16) + 2*BM*(int)sizeof(float);
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