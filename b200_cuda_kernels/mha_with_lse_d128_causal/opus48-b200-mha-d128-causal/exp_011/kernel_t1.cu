#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_causal {

constexpr int BM=64, BN=64, Dh=128, SD=136, NW=4, NT=128;

__device__ __forceinline__ uint32_t smem_addr(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldm_x4(uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3,uint32_t a){
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a));
}
__device__ __forceinline__ void ldm_x2(uint32_t&r0,uint32_t&r1,uint32_t a){
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];":"=r"(r0),"=r"(r1):"r"(a));
}
__device__ __forceinline__ void ldm_x2t(uint32_t&r0,uint32_t&r1,uint32_t a){
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];":"=r"(r0),"=r"(r1):"r"(a));
}
__device__ __forceinline__ void cp16(uint32_t d,const void*s){
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(d),"l"(s));
}
__device__ __forceinline__ void cpcommit(){ asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cpwait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N)); }

__device__ __forceinline__ uint32_t pack2(float a,float b){
  __nv_bfloat162 v=__floats2bfloat162_rn(a,b); return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void mma(float&d0,float&d1,float&d2,float&d3,
  uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1,
  float c0,float c1,float c2,float c3){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
   :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
   :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"f"(c0),"f"(c1),"f"(c2),"f"(c3));
}

template<int ROWS>
__device__ __forceinline__ void load_tile(__nv_bfloat16* dst,const __nv_bfloat16* src,int row0,int S,int tid){
  constexpr int ITERS = ROWS*16/NT;
  #pragma unroll
  for(int it=0;it<ITERS;it++){
    int i=tid+it*NT; int r=i>>4, chunk=i&15;
    int grow=row0+r;
    uint32_t d=smem_addr(&dst[r*SD+chunk*8]);
    if(grow<S) cp16(d,&src[(size_t)grow*Dh+chunk*8]);
    else *reinterpret_cast<uint4*>(&dst[r*SD+chunk*8])=make_uint4(0,0,0,0);
  }
}

__global__ __launch_bounds__(NT,2)
void attn_kernel(const __nv_bfloat16* Q,const __nv_bfloat16* K,const __nv_bfloat16* V,
                 __nv_bfloat16* O,float* LSE,int S){
  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* Qs=smem;
  __nv_bfloat16* Kbuf[2]={Qs+BM*SD, Qs+BM*SD+BN*SD};
  __nv_bfloat16* Vbuf[2]={Qs+BM*SD+2*BN*SD, Qs+BM*SD+3*BN*SD};

  int bh=blockIdx.y;
  int q0=blockIdx.x*BM;
  if(q0>=S) return;
  int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
  int groupID=lane>>2, tig=lane&3;

  const __nv_bfloat16* Qb=Q+(size_t)bh*S*Dh;
  const __nv_bfloat16* Kb=K+(size_t)bh*S*Dh;
  const __nv_bfloat16* Vb=V+(size_t)bh*S*Dh;

  load_tile<BM>(Qs,Qb,q0,S,tid);
  cpcommit();

  const float scale=0.08838834764831845f;
  const float log2e=1.4426950408889634f;
  const float scale2=scale*log2e;
  const float ln2=0.6931471805599453f;
  const float NEG=-1e30f;

  int maxkv=q0+BM-1; if(maxkv>S-1) maxkv=S-1;
  int num_kt=maxkv/BN+1;

  load_tile<BN>(Kbuf[0],Kb,0,S,tid);
  load_tile<BN>(Vbuf[0],Vb,0,S,tid);
  cpcommit();

  cpwait<1>();
  __syncthreads();

  uint32_t Qf[8][4];
  #pragma unroll
  for(int kk=0;kk<8;kk++){
    int a_row=lane&15; int a_col=(lane&16)?8:0;
    uint32_t ad=smem_addr(&Qs[(warp*16+a_row)*SD + kk*16 + a_col]);
    ldm_x4(Qf[kk][0],Qf[kk][1],Qf[kk][2],Qf[kk][3],ad);
  }

  float Oacc[16][4];
  #pragma unroll
  for(int i=0;i<16;i++){Oacc[i][0]=0;Oacc[i][1]=0;Oacc[i][2]=0;Oacc[i][3]=0;}
  float m_lo=NEG,m_hi=NEG,l_lo=0,l_hi=0;
  int qrow_lo=q0+warp*16+groupID;
  int qrow_hi=qrow_lo+8;

  for(int kt=0;kt<num_kt;kt++){
    int buf=kt&1;
    if(kt+1<num_kt){
      int nb=(kt+1)&1;
      load_tile<BN>(Kbuf[nb],Kb,(kt+1)*BN,S,tid);
      load_tile<BN>(Vbuf[nb],Vb,(kt+1)*BN,S,tid);
      cpcommit();
      cpwait<1>();
    } else {
      cpwait<0>();
    }
    __syncthreads();

    __nv_bfloat16* Ks=Kbuf[buf];
    __nv_bfloat16* Vs=Vbuf[buf];
    int kv0=kt*BN;

    float Sr[8][4];
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      float a0=0,a1=0,a2=0,a3=0;
      #pragma unroll
      for(int kk=0;kk<8;kk++){
        int idx=lane&15; int n=idx&7; int dd=(idx&8)?8:0;
        uint32_t bd=smem_addr(&Ks[(nt*8+n)*SD + kk*16 + dd]);
        uint32_t b0,b1; ldm_x2(b0,b1,bd);
        mma(a0,a1,a2,a3, Qf[kk][0],Qf[kk][1],Qf[kk][2],Qf[kk][3], b0,b1, a0,a1,a2,a3);
      }
      Sr[nt][0]=a0;Sr[nt][1]=a1;Sr[nt][2]=a2;Sr[nt][3]=a3;
    }

    bool need_mask=(kv0+BN-1>=q0)||(kv0+BN>S);
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      if(need_mask){
        int c0=kv0+nt*8+tig*2, c1=c0+1;
        Sr[nt][0]=(c0<=qrow_lo&&c0<S)?Sr[nt][0]*scale2:NEG;
        Sr[nt][1]=(c1<=qrow_lo&&c1<S)?Sr[nt][1]*scale2:NEG;
        Sr[nt][2]=(c0<=qrow_hi&&c0<S)?Sr[nt][2]*scale2:NEG;
        Sr[nt][3]=(c1<=qrow_hi&&c1<S)?Sr[nt][3]*scale2:NEG;
      }else{
        Sr[nt][0]*=scale2;Sr[nt][1]*=scale2;Sr[nt][2]*=scale2;Sr[nt][3]*=scale2;
      }
    }

    float rmax_lo=NEG,rmax_hi=NEG;
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      rmax_lo=fmaxf(rmax_lo,fmaxf(Sr[nt][0],Sr[nt][1]));
      rmax_hi=fmaxf(rmax_hi,fmaxf(Sr[nt][2],Sr[nt][3]));
    }
    rmax_lo=fmaxf(rmax_lo,__shfl_xor_sync(-1u,rmax_lo,1));
    rmax_lo=fmaxf(rmax_lo,__shfl_xor_sync(-1u,rmax_lo,2));
    rmax_hi=fmaxf(rmax_hi,__shfl_xor_sync(-1u,rmax_hi,1));
    rmax_hi=fmaxf(rmax_hi,__shfl_xor_sync(-1u,rmax_hi,2));

    float mn_lo=fmaxf(m_lo,rmax_lo), mn_hi=fmaxf(m_hi,rmax_hi);
    float al_lo=exp2f(m_lo-mn_lo), al_hi=exp2f(m_hi-mn_hi);

    float rs_lo=0,rs_hi=0;
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      float p0=exp2f(Sr[nt][0]-mn_lo); rs_lo+=p0; Sr[nt][0]=p0;
      float p1=exp2f(Sr[nt][1]-mn_lo); rs_lo+=p1; Sr[nt][1]=p1;
      float p2=exp2f(Sr[nt][2]-mn_hi); rs_hi+=p2; Sr[nt][2]=p2;
      float p3=exp2f(Sr[nt][3]-mn_hi); rs_hi+=p3; Sr[nt][3]=p3;
    }
    rs_lo+=__shfl_xor_sync(-1u,rs_lo,1); rs_lo+=__shfl_xor_sync(-1u,rs_lo,2);
    rs_hi+=__shfl_xor_sync(-1u,rs_hi,1); rs_hi+=__shfl_xor_sync(-1u,rs_hi,2);

    l_lo=al_lo*l_lo+rs_lo; l_hi=al_hi*l_hi+rs_hi;
    m_lo=mn_lo; m_hi=mn_hi;

    #pragma unroll
    for(int dt=0;dt<16;dt++){
      Oacc[dt][0]*=al_lo;Oacc[dt][1]*=al_lo;Oacc[dt][2]*=al_hi;Oacc[dt][3]*=al_hi;
    }

    uint32_t Af[4][4];
    #pragma unroll
    for(int ks=0;ks<4;ks++){
      Af[ks][0]=pack2(Sr[2*ks][0],Sr[2*ks][1]);
      Af[ks][1]=pack2(Sr[2*ks][2],Sr[2*ks][3]);
      Af[ks][2]=pack2(Sr[2*ks+1][0],Sr[2*ks+1][1]);
      Af[ks][3]=pack2(Sr[2*ks+1][2],Sr[2*ks+1][3]);
    }
    #pragma unroll
    for(int dt=0;dt<16;dt++){
      float c0=Oacc[dt][0],c1=Oacc[dt][1],c2=Oacc[dt][2],c3=Oacc[dt][3];
      #pragma unroll
      for(int ks=0;ks<4;ks++){
        uint32_t vd=smem_addr(&Vs[(ks*16+(lane&15))*SD + dt*8]);
        uint32_t b0,b1; ldm_x2t(b0,b1,vd);
        mma(c0,c1,c2,c3, Af[ks][0],Af[ks][1],Af[ks][2],Af[ks][3], b0,b1, c0,c1,c2,c3);
      }
      Oacc[dt][0]=c0;Oacc[dt][1]=c1;Oacc[dt][2]=c2;Oacc[dt][3]=c3;
    }
    __syncthreads();
  }

  float inv_lo=1.f/l_lo, inv_hi=1.f/l_hi;
  size_t ob=(size_t)bh*S*Dh;
  #pragma unroll
  for(int dt=0;dt<16;dt++){
    int col0=dt*8+tig*2, col1=col0+1;
    if(qrow_lo<S){
      O[ob+(size_t)qrow_lo*Dh+col0]=__float2bfloat16(Oacc[dt][0]*inv_lo);
      O[ob+(size_t)qrow_lo*Dh+col1]=__float2bfloat16(Oacc[dt][1]*inv_lo);
    }
    if(qrow_hi<S){
      O[ob+(size_t)qrow_hi*Dh+col0]=__float2bfloat16(Oacc[dt][2]*inv_hi);
      O[ob+(size_t)qrow_hi*Dh+col1]=__float2bfloat16(Oacc[dt][3]*inv_hi);
    }
  }
  if(tig==0){
    if(qrow_lo<S) LSE[(size_t)bh*S+qrow_lo]=m_lo*ln2+logf(l_lo);
    if(qrow_hi<S) LSE[(size_t)bh*S+qrow_hi]=m_hi*ln2+logf(l_hi);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int64_t B = Q.size(0);
  int64_t Hh = Q.size(1);
  int64_t S = Q.size(2);

  const __nv_bfloat16* q = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* k = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* v = static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* o = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* lse = static_cast<float*>(LSE.data_ptr());

  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  dim3 grid((unsigned)((S + BM - 1)/BM), (unsigned)(B*Hh));
  dim3 block(NT);
  size_t smem = (size_t)(BM + 4*BN) * SD * sizeof(__nv_bfloat16);

  static bool attr_set = false;
  if (!attr_set){
    cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_set = true;
  }

  attn_kernel<<<grid, block, smem, stream>>>(q, k, v, o, lse, (int)S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_causal::run);

}  // namespace mha_causal