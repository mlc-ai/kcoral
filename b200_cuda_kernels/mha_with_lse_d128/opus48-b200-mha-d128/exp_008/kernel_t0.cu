#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} }while(0)

namespace mha_kernel_ns {

#define HD  128
#define BM  64
#define BN  64
#define SN  8    // BN/8  (n-tiles for QK output)
#define DK  8    // D/16  (k-tiles for QK contraction)
#define PVK 4    // BN/16 (k-tiles for PV contraction)
#define ON  16   // D/8   (n-tiles for PV output)

__device__ __forceinline__ void mma_m16n8k16(
    float &d0,float &d1,float &d2,float &d3,
    uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,
    uint32_t b0,uint32_t b1){
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

__device__ __forceinline__ uint32_t pack2(float x,float y){
  __nv_bfloat16 a=__float2bfloat16(x), b=__float2bfloat16(y);
  uint16_t ai=*reinterpret_cast<uint16_t*>(&a);
  uint16_t bi=*reinterpret_cast<uint16_t*>(&b);
  return (uint32_t)ai | ((uint32_t)bi<<16);
}

__device__ __forceinline__ float gmax(float v){
  v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,1));
  v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,2));
  return v;
}
__device__ __forceinline__ float gsum(float v){
  v+=__shfl_xor_sync(0xffffffff,v,1);
  v+=__shfl_xor_sync(0xffffffff,v,2);
  return v;
}

extern __shared__ __align__(16) char smem_raw[];

__global__ void mha_kernel(const __nv_bfloat16* __restrict__ Q,
                           const __nv_bfloat16* __restrict__ K,
                           const __nv_bfloat16* __restrict__ V,
                           __nv_bfloat16* __restrict__ O,
                           float* __restrict__ LSE,
                           int S, float scale){
  int bh=blockIdx.y;
  int qtile_base=blockIdx.x*BM;

  const __nv_bfloat16* Qp=Q+(size_t)bh*S*HD;
  const __nv_bfloat16* Kp=K+(size_t)bh*S*HD;
  const __nv_bfloat16* Vp=V+(size_t)bh*S*HD;
  __nv_bfloat16* Op=O+(size_t)bh*S*HD;
  float* LSEp=LSE+(size_t)bh*S;

  __nv_bfloat16* Qs=(__nv_bfloat16*)smem_raw;
  __nv_bfloat16* Ks=Qs+BM*HD;
  __nv_bfloat16* Vs=Ks+BN*HD;

  int tid=threadIdx.x;
  int warp=tid>>5, lane=tid&31;
  int groupID=lane>>2, thr=lane&3;
  int warp_row=warp*16;
  int row_a=warp_row+groupID;
  int row_b=warp_row+groupID+8;

  // load Q tile
  for(int idx=tid; idx<BM*16; idx+=blockDim.x){
    int r=idx>>4, q=idx&15;
    int gr=qtile_base+r;
    int4 v=(gr<S)? ((const int4*)Qp)[(size_t)gr*16+q] : make_int4(0,0,0,0);
    ((int4*)Qs)[r*16+q]=v;
  }
  __syncthreads();

  // preload Q fragments (reused across all KV tiles)
  uint32_t qfrag[DK][4];
  #pragma unroll
  for(int kk=0;kk<DK;kk++){
    int c=16*kk+thr*2;
    qfrag[kk][0]=*(const uint32_t*)&Qs[row_a*HD + c];
    qfrag[kk][1]=*(const uint32_t*)&Qs[row_b*HD + c];
    qfrag[kk][2]=*(const uint32_t*)&Qs[row_a*HD + c+8];
    qfrag[kk][3]=*(const uint32_t*)&Qs[row_b*HD + c+8];
  }

  float m_a=-INFINITY,m_b=-INFINITY,l_a=0.f,l_b=0.f;
  float Ofrag[ON][4];
  #pragma unroll
  for(int nt=0;nt<ON;nt++){Ofrag[nt][0]=0;Ofrag[nt][1]=0;Ofrag[nt][2]=0;Ofrag[nt][3]=0;}

  int num_kv=(S+BN-1)/BN;
  for(int kv=0; kv<num_kv; kv++){
    int kv_base=kv*BN;
    __syncthreads();
    for(int idx=tid; idx<BN*16; idx+=blockDim.x){
      int r=idx>>4, q=idx&15;
      int gr=kv_base+r;
      int4 vk=(gr<S)? ((const int4*)Kp)[(size_t)gr*16+q] : make_int4(0,0,0,0);
      int4 vv=(gr<S)? ((const int4*)Vp)[(size_t)gr*16+q] : make_int4(0,0,0,0);
      ((int4*)Ks)[r*16+q]=vk;
      ((int4*)Vs)[r*16+q]=vv;
    }
    __syncthreads();

    // S = Q @ K^T
    float Sacc[SN][4];
    #pragma unroll
    for(int nt=0;nt<SN;nt++){
      int key=nt*8+groupID;
      float s0=0,s1=0,s2=0,s3=0;
      #pragma unroll
      for(int kk=0;kk<DK;kk++){
        int c=16*kk+thr*2;
        uint32_t rb0=*(const uint32_t*)&Ks[key*HD + c];
        uint32_t rb1=*(const uint32_t*)&Ks[key*HD + c+8];
        mma_m16n8k16(s0,s1,s2,s3, qfrag[kk][0],qfrag[kk][1],qfrag[kk][2],qfrag[kk][3], rb0,rb1);
      }
      Sacc[nt][0]=s0;Sacc[nt][1]=s1;Sacc[nt][2]=s2;Sacc[nt][3]=s3;
    }
    // scale + mask OOB keys
    #pragma unroll
    for(int nt=0;nt<SN;nt++){
      int keycol=kv_base+nt*8+thr*2;
      #pragma unroll
      for(int e=0;e<4;e++){
        float val=Sacc[nt][e]*scale;
        int kidx=keycol+(e&1);
        if(kidx>=S) val=-INFINITY;
        Sacc[nt][e]=val;
      }
    }
    // row max over this tile
    float mt_a=-INFINITY,mt_b=-INFINITY;
    #pragma unroll
    for(int nt=0;nt<SN;nt++){
      mt_a=fmaxf(mt_a,fmaxf(Sacc[nt][0],Sacc[nt][1]));
      mt_b=fmaxf(mt_b,fmaxf(Sacc[nt][2],Sacc[nt][3]));
    }
    mt_a=gmax(mt_a); mt_b=gmax(mt_b);
    float new_m_a=fmaxf(m_a,mt_a), new_m_b=fmaxf(m_b,mt_b);
    float corr_a=__expf(m_a-new_m_a), corr_b=__expf(m_b-new_m_b);
    float sum_a=0,sum_b=0;
    #pragma unroll
    for(int nt=0;nt<SN;nt++){
      float p0=__expf(Sacc[nt][0]-new_m_a);
      float p1=__expf(Sacc[nt][1]-new_m_a);
      float p2=__expf(Sacc[nt][2]-new_m_b);
      float p3=__expf(Sacc[nt][3]-new_m_b);
      Sacc[nt][0]=p0;Sacc[nt][1]=p1;Sacc[nt][2]=p2;Sacc[nt][3]=p3;
      sum_a+=p0+p1; sum_b+=p2+p3;
    }
    sum_a=gsum(sum_a); sum_b=gsum(sum_b);
    l_a=l_a*corr_a+sum_a; l_b=l_b*corr_b+sum_b;
    m_a=new_m_a; m_b=new_m_b;
    #pragma unroll
    for(int nt=0;nt<ON;nt++){
      Ofrag[nt][0]*=corr_a;Ofrag[nt][1]*=corr_a;Ofrag[nt][2]*=corr_b;Ofrag[nt][3]*=corr_b;
    }
    // build A fragments (P) for PV
    uint32_t Afrag[PVK][4];
    #pragma unroll
    for(int kk=0;kk<PVK;kk++){
      Afrag[kk][0]=pack2(Sacc[2*kk][0],Sacc[2*kk][1]);
      Afrag[kk][1]=pack2(Sacc[2*kk][2],Sacc[2*kk][3]);
      Afrag[kk][2]=pack2(Sacc[2*kk+1][0],Sacc[2*kk+1][1]);
      Afrag[kk][3]=pack2(Sacc[2*kk+1][2],Sacc[2*kk+1][3]);
    }
    // O += P @ V
    #pragma unroll
    for(int nt=0;nt<ON;nt++){
      int col=8*nt+groupID;
      float o0=Ofrag[nt][0],o1=Ofrag[nt][1],o2=Ofrag[nt][2],o3=Ofrag[nt][3];
      #pragma unroll
      for(int kk=0;kk<PVK;kk++){
        int k0=16*kk+thr*2;
        uint32_t rb0=(uint32_t)(*(const uint16_t*)&Vs[k0*HD+col]) |
                     ((uint32_t)(*(const uint16_t*)&Vs[(k0+1)*HD+col])<<16);
        uint32_t rb1=(uint32_t)(*(const uint16_t*)&Vs[(k0+8)*HD+col]) |
                     ((uint32_t)(*(const uint16_t*)&Vs[(k0+9)*HD+col])<<16);
        mma_m16n8k16(o0,o1,o2,o3, Afrag[kk][0],Afrag[kk][1],Afrag[kk][2],Afrag[kk][3], rb0,rb1);
      }
      Ofrag[nt][0]=o0;Ofrag[nt][1]=o1;Ofrag[nt][2]=o2;Ofrag[nt][3]=o3;
    }
  }

  // finalize
  float inv_a=1.f/l_a, inv_b=1.f/l_b;
  int gr_a=qtile_base+row_a, gr_b=qtile_base+row_b;
  #pragma unroll
  for(int nt=0;nt<ON;nt++){
    int col=8*nt+thr*2;
    if(gr_a<S){
      Op[(size_t)gr_a*HD+col]  =__float2bfloat16(Ofrag[nt][0]*inv_a);
      Op[(size_t)gr_a*HD+col+1]=__float2bfloat16(Ofrag[nt][1]*inv_a);
    }
    if(gr_b<S){
      Op[(size_t)gr_b*HD+col]  =__float2bfloat16(Ofrag[nt][2]*inv_b);
      Op[(size_t)gr_b*HD+col+1]=__float2bfloat16(Ofrag[nt][3]*inv_b);
    }
  }
  if(thr==0){
    if(gr_a<S) LSEp[gr_a]=m_a+logf(l_a);
    if(gr_b<S) LSEp[gr_b]=m_b+logf(l_b);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int64_t B=Q.size(0),H=Q.size(1),S=Q.size(2);
  const __nv_bfloat16* Qp=(const __nv_bfloat16*)Q.data_ptr();
  const __nv_bfloat16* Kp=(const __nv_bfloat16*)K.data_ptr();
  const __nv_bfloat16* Vp=(const __nv_bfloat16*)V.data_ptr();
  __nv_bfloat16* Op=(__nv_bfloat16*)O.data_ptr();
  float* LSEp=(float*)LSE.data_ptr();

  int num_qtiles=(int)((S+BM-1)/BM);
  dim3 grid(num_qtiles, (unsigned)(B*H));
  dim3 block(128);
  size_t smem=(size_t)(BM*HD+2*BN*HD)*sizeof(__nv_bfloat16); // 48KB
  cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  float scale=1.0f/sqrtf((float)HD);
  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id);
  mha_kernel<<<grid,block,smem,stream>>>(Qp,Kp,Vp,Op,LSEp,(int)S,scale);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel_ns::run);

} // namespace mha_kernel_ns