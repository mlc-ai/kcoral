#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)

namespace mha_kernel {

#define BM 128
#define BN 64
#define HDIM 128
#define QS 136
#define KS 136
#define VS 136

__device__ __forceinline__ void mma16816(float* d, uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3, uint32_t b0,uint32_t b1){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t a, uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3){
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a));
}
__device__ __forceinline__ uint32_t pack2f(float x,float y){
  __nv_bfloat16 a=__float2bfloat16(x), b=__float2bfloat16(y);
  uint16_t ai=*reinterpret_cast<uint16_t*>(&a), bi=*reinterpret_cast<uint16_t*>(&b);
  return (uint32_t)ai | ((uint32_t)bi<<16);
}
__device__ __forceinline__ void cpasync16(void* dst,const void* src){
  unsigned d=(unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(d),"l"(src):"memory");
}
__device__ __forceinline__ void commit(){asm volatile("cp.async.commit_group;\n":::"memory");}
template<int N> __device__ __forceinline__ void cpwait(){asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory");}

__global__ __launch_bounds__(256,2) void kern(
   const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
   const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O,
   float* __restrict__ LSE, int S, int H){
  int qcta=blockIdx.x*BM, h=blockIdx.y, b=blockIdx.z;
  int64_t bh=(int64_t)b*H+h;
  const __nv_bfloat16* Qb=Q+bh*(int64_t)S*HDIM;
  const __nv_bfloat16* Kb=K+bh*(int64_t)S*HDIM;
  const __nv_bfloat16* Vb=V+bh*(int64_t)S*HDIM;
  __nv_bfloat16* Ob=O+bh*(int64_t)S*HDIM;

  extern __shared__ __nv_bfloat16 sm[];
  __nv_bfloat16* Qs=sm;
  __nv_bfloat16* Ks=Qs + BM*QS;      // 2 buffers of BN*KS
  __nv_bfloat16* Vs=Ks + 2*BN*KS;    // 2 buffers of BN*VS

  int tid=threadIdx.x, w=tid>>5, lane=tid&31, gid=lane>>2, tig=lane&3;
  const float sc = 0.08838834764831845f * 1.4426950408889634f;
  const float NEG=-1e30f;

  // load Q once
  #pragma unroll
  for(int i=tid;i<BM*HDIM/8;i+=256){
    int elt=i*8; int r=elt>>7; int d=elt&127; int q=qcta+r; int qq=q<S?q:S-1;
    cpasync16(&Qs[r*QS+d], &Qb[(int64_t)qq*HDIM+d]);
  }

  int nblk=(S+BN-1)/BN;
  // prefetch block 0 K/V
  {
    int kb=0;
    for(int i=tid;i<BN*HDIM/8;i+=256){
      int elt=i*8; int key=elt>>7; int d=elt&127; int kg=kb+key; int kk=kg<S?kg:S-1;
      cpasync16(&Ks[key*KS+d], &Kb[(int64_t)kk*HDIM+d]);
      cpasync16(&Vs[key*VS+d], &Vb[(int64_t)kk*HDIM+d]);
    }
  }
  commit();

  float o[16][4];
  #pragma unroll
  for(int i=0;i<16;i++){o[i][0]=o[i][1]=o[i][2]=o[i][3]=0.f;}
  float m0=NEG,m1=NEG,l0=0.f,l1=0.f;

  for(int j=0;j<nblk;j++){
    int cur=j&1;
    __nv_bfloat16* Kc=Ks + cur*BN*KS;
    __nv_bfloat16* Vc=Vs + cur*BN*VS;
    // prefetch next block
    if(j+1<nblk){
      int nb=(j+1)&1; int kb=(j+1)*BN;
      __nv_bfloat16* Kn=Ks + nb*BN*KS;
      __nv_bfloat16* Vn=Vs + nb*BN*VS;
      for(int i=tid;i<BN*HDIM/8;i+=256){
        int elt=i*8; int key=elt>>7; int d=elt&127; int kg=kb+key; int kk=kg<S?kg:S-1;
        cpasync16(&Kn[key*KS+d], &Kb[(int64_t)kk*HDIM+d]);
        cpasync16(&Vn[key*VS+d], &Vb[(int64_t)kk*HDIM+d]);
      }
      commit();
      cpwait<1>();
    } else {
      cpwait<0>();
    }
    __syncthreads();

    int kb=j*BN;
    // ---- S = Q @ K^T ----
    float s[8][4];
    #pragma unroll
    for(int nt=0;nt<8;nt++){s[nt][0]=s[nt][1]=s[nt][2]=s[nt][3]=0.f;}
    #pragma unroll
    for(int kt=0;kt<8;kt++){
      int qr0=16*w+gid, qr1=qr0+8, dcol=16*kt+2*tig;
      uint32_t a0=*reinterpret_cast<const uint32_t*>(&Qs[qr0*QS+dcol]);
      uint32_t a1=*reinterpret_cast<const uint32_t*>(&Qs[qr1*QS+dcol]);
      uint32_t a2=*reinterpret_cast<const uint32_t*>(&Qs[qr0*QS+dcol+8]);
      uint32_t a3=*reinterpret_cast<const uint32_t*>(&Qs[qr1*QS+dcol+8]);
      #pragma unroll
      for(int nt=0;nt<8;nt++){
        int krow=8*nt+gid;
        uint32_t b0=*reinterpret_cast<const uint32_t*>(&Kc[krow*KS+dcol]);
        uint32_t b1=*reinterpret_cast<const uint32_t*>(&Kc[krow*KS+dcol+8]);
        mma16816(s[nt],a0,a1,a2,a3,b0,b1);
      }
    }
    // scale + mask
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      s[nt][0]*=sc;s[nt][1]*=sc;s[nt][2]*=sc;s[nt][3]*=sc;
      int kA=kb+8*nt+2*tig, kB=kA+1;
      if(kA>=S){s[nt][0]=NEG;s[nt][2]=NEG;}
      if(kB>=S){s[nt][1]=NEG;s[nt][3]=NEG;}
    }
    float bm0=NEG,bm1=NEG;
    #pragma unroll
    for(int nt=0;nt<8;nt++){bm0=fmaxf(bm0,fmaxf(s[nt][0],s[nt][1]));bm1=fmaxf(bm1,fmaxf(s[nt][2],s[nt][3]));}
    bm0=fmaxf(bm0,__shfl_xor_sync(~0u,bm0,1)); bm0=fmaxf(bm0,__shfl_xor_sync(~0u,bm0,2));
    bm1=fmaxf(bm1,__shfl_xor_sync(~0u,bm1,1)); bm1=fmaxf(bm1,__shfl_xor_sync(~0u,bm1,2));
    float nm0=fmaxf(m0,bm0), nm1=fmaxf(m1,bm1);
    float corr0=exp2f(m0-nm0), corr1=exp2f(m1-nm1);
    float sum0=0.f,sum1=0.f;
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      float e0=exp2f(s[nt][0]-nm0),e1=exp2f(s[nt][1]-nm0),e2=exp2f(s[nt][2]-nm1),e3=exp2f(s[nt][3]-nm1);
      s[nt][0]=e0;s[nt][1]=e1;s[nt][2]=e2;s[nt][3]=e3; sum0+=e0+e1; sum1+=e2+e3;
    }
    sum0+=__shfl_xor_sync(~0u,sum0,1); sum0+=__shfl_xor_sync(~0u,sum0,2);
    sum1+=__shfl_xor_sync(~0u,sum1,1); sum1+=__shfl_xor_sync(~0u,sum1,2);
    l0=l0*corr0+sum0; l1=l1*corr1+sum1; m0=nm0; m1=nm1;
    #pragma unroll
    for(int nt=0;nt<16;nt++){o[nt][0]*=corr0;o[nt][1]*=corr0;o[nt][2]*=corr1;o[nt][3]*=corr1;}

    // ---- O += P @ V ----   (V loaded transposed via ldmatrix.trans)
    #pragma unroll
    for(int kt=0;kt<4;kt++){
      uint32_t a0=pack2f(s[2*kt][0],s[2*kt][1]);
      uint32_t a1=pack2f(s[2*kt][2],s[2*kt][3]);
      uint32_t a2=pack2f(s[2*kt+1][0],s[2*kt+1][1]);
      uint32_t a3=pack2f(s[2*kt+1][2],s[2*kt+1][3]);
      int k0=16*kt;
      #pragma unroll
      for(int d0=0;d0<128;d0+=16){
        int krow=k0+(lane&15);
        int dcol=d0+((lane>=16)?8:0);
        uint32_t addr=(uint32_t)__cvta_generic_to_shared(&Vc[krow*VS+dcol]);
        uint32_t r0,r1,r2,r3;
        ldmatrix_x4_trans(addr,r0,r1,r2,r3);
        int nt=d0>>3;
        mma16816(o[nt],a0,a1,a2,a3,r0,r1);
        mma16816(o[nt+1],a0,a1,a2,a3,r2,r3);
      }
    }
    __syncthreads();
  }

  int q0=qcta+16*w+gid, q1=q0+8;
  float inv0=(l0>0.f)?1.f/l0:0.f, inv1=(l1>0.f)?1.f/l1:0.f;
  #pragma unroll
  for(int nt=0;nt<16;nt++){
    int d=8*nt+2*tig;
    if(q0<S){Ob[(int64_t)q0*HDIM+d]=__float2bfloat16(o[nt][0]*inv0); Ob[(int64_t)q0*HDIM+d+1]=__float2bfloat16(o[nt][1]*inv0);}
    if(q1<S){Ob[(int64_t)q1*HDIM+d]=__float2bfloat16(o[nt][2]*inv1); Ob[(int64_t)q1*HDIM+d+1]=__float2bfloat16(o[nt][3]*inv1);}
  }
  if(tig==0){
    const float LN2=0.6931471805599453f;
    if(q0<S) LSE[bh*(int64_t)S+q0]=m0*LN2+logf(l0);
    if(q1<S) LSE[bh*(int64_t)S+q1]=m1*LN2+logf(l1);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bsz=(int)Q.size(0), Hsz=(int)Q.size(1), Ssz=(int)Q.size(2);

  const __nv_bfloat16* Qp=static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp=static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp=static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
  float* Lp=static_cast<float*>(LSE.data_ptr());

  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  int smem = (BM*QS + 2*BN*KS + 2*BN*VS)*(int)sizeof(__nv_bfloat16);
  static bool attr_set=false;
  if(!attr_set){
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr_set=true;
  }

  dim3 grid((Ssz+BM-1)/BM, Hsz, Bsz);
  kern<<<grid,256,smem,stream>>>(Qp,Kp,Vp,Op,Lp,Ssz,Hsz);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel