#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s @%s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} }while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char*s; cuGetErrorString(_e,&s); \
  fprintf(stderr,"CU %s @%s:%d\n",s,__FILE__,__LINE__); exit(1);} }while(0)

namespace mha_kernel {

constexpr int Dc=128, BM=128, BN=64, THREADS=128, DH=64;

__device__ __forceinline__ uint64_t mkdesc(uint32_t a,uint32_t lbo,uint32_t sbo,int sw){
  uint64_t d=0;
  d |= (uint64_t)((a&0x3FFFF)>>4);
  d |= (uint64_t)((lbo&0x3FFFF)>>4)<<16;
  d |= (uint64_t)((sbo&0x3FFFF)>>4)<<32;
  d |= (uint64_t)1<<46;
  d |= (uint64_t)sw<<61;
  return d;
}
__device__ __forceinline__ uint32_t pk(float a,float b){
  __nv_bfloat16 x=__float2bfloat16(a), y=__float2bfloat16(b);
  return (uint32_t)*(uint16_t*)&x | ((uint32_t)*(uint16_t*)&y<<16);
}
__device__ __forceinline__ void tmem_ld_x32(uint32_t a,uint32_t* r){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
   "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
   "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
   :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
    "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
    "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
    "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
   :"r"(a));
}
__device__ __forceinline__ void tmem_st_x32(uint32_t a,const uint32_t* r){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], "
   "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
   "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
   ::"r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),
    "r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),"r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]),
    "r"(r[16]),"r"(r[17]),"r"(r[18]),"r"(r[19]),"r"(r[20]),"r"(r[21]),"r"(r[22]),"r"(r[23]),
    "r"(r[24]),"r"(r[25]),"r"(r[26]),"r"(r[27]),"r"(r[28]),"r"(r[29]),"r"(r[30]),"r"(r[31]),
    "r"(a):"memory");
}
__device__ __forceinline__ void wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;":::"memory"); }
__device__ __forceinline__ void fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }
__device__ __forceinline__ void fence_before(){ asm volatile("tcgen05.fence::before_thread_sync;":::"memory"); }
__device__ __forceinline__ void umma_cg1(uint32_t dt,uint64_t a,uint64_t b,uint32_t id,uint32_t acc){
  asm volatile("{.reg .pred p; setp.ne.b32 p,%4,0; tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;}"
   ::"r"(dt),"l"(a),"l"(b),"r"(id),"r"(acc));
}
__device__ __forceinline__ void commit_cg1(uint64_t* bar){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"::"r"(a));
}
__device__ __forceinline__ void init_bar(uint64_t* b,uint32_t c){
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c));
}
__device__ __forceinline__ void fence_bar_init(){ asm volatile("fence.mbarrier_init.release.cluster;":::"memory"); }
__device__ __forceinline__ void arrive_expect(uint64_t* b,uint32_t tx){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
   ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(tx):"memory");
}
__device__ __forceinline__ void bar_wait(uint64_t* b,uint32_t ph){
  asm volatile("{.reg .pred P; W_%=: mbarrier.try_wait.parity.shared.b64 P,[%0],%1; @!P bra W_%=;}"
   ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph));
}
__device__ __forceinline__ void fence_pa(){ asm volatile("fence.proxy.async;":::"memory"); }
__device__ __forceinline__ void tma_ld(const CUtensorMap* d,uint64_t* bar,void* smem,int32_t c0,int32_t c1){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4}],[%2];"
   ::"r"((uint32_t)__cvta_generic_to_shared(smem)),"l"((uint64_t)d),
     "r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c0),"r"(c1):"memory");
}

__global__ __launch_bounds__(THREADS) void mha_fwd(
    const __grid_constant__ CUtensorMap tmaQ,
    const __grid_constant__ CUtensorMap tmaK,
    const __grid_constant__ CUtensorMap tmaV,
    __nv_bfloat16* __restrict__ O, float* __restrict__ LSE,
    int S,int H){

  extern __shared__ char raw[];
  uint32_t raw_s=(uint32_t)__cvta_generic_to_shared(raw);
  uint32_t base_s=(raw_s+1023u)&~1023u;
  char* sm = raw + (base_s - raw_s);
  __nv_bfloat16* Qs[2]={(__nv_bfloat16*)(sm+0),(__nv_bfloat16*)(sm+16384)};
  __nv_bfloat16* Ks[2]={(__nv_bfloat16*)(sm+32768),(__nv_bfloat16*)(sm+40960)};
  __nv_bfloat16* Vs[2]={(__nv_bfloat16*)(sm+49152),(__nv_bfloat16*)(sm+57344)};
  __nv_bfloat16* Ps=(__nv_bfloat16*)(sm+65536);
  uint64_t* bar_load=(uint64_t*)(sm+81920);
  uint64_t* bar_mma =(uint64_t*)(sm+81928);
  uint32_t* tmem_ptr=(uint32_t*)(sm+81936);

  const int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
  const int b=blockIdx.z, h=blockIdx.y, q0=blockIdx.x*BM;
  const long bh=(long)(b*H+h);
  const int row=warp*32+lane, query=q0+row;
  const float scale=0.08838834764831845f;

  if(tid==0){ init_bar(bar_load,1); init_bar(bar_mma,1); fence_bar_init(); }
  if(tid<32){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(tmem_ptr);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(a),"r"(256));
  }
  __syncthreads();
  uint32_t tmem=*tmem_ptr;
  if(tid<32) asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");

  uint32_t cQ0=(uint32_t)__cvta_generic_to_shared(Qs[0]);
  uint32_t cQ1=(uint32_t)__cvta_generic_to_shared(Qs[1]);
  uint32_t cK0=(uint32_t)__cvta_generic_to_shared(Ks[0]);
  uint32_t cK1=(uint32_t)__cvta_generic_to_shared(Ks[1]);
  uint32_t cV0=(uint32_t)__cvta_generic_to_shared(Vs[0]);
  uint32_t cV1=(uint32_t)__cvta_generic_to_shared(Vs[1]);
  uint32_t cP =(uint32_t)__cvta_generic_to_shared(Ps);

  uint32_t idesc_qk=(1u<<4)|(1u<<7)|(1u<<10)|((64u>>3)<<17)|((128u>>4)<<24);
  uint32_t idesc_pv=idesc_qk|(1u<<16);

  if(tid==0){
    arrive_expect(bar_load,32768u);
    tma_ld(&tmaQ,bar_load,Qs[0],0,(int)(bh*S+q0));
    tma_ld(&tmaQ,bar_load,Qs[1],64,(int)(bh*S+q0));
  }
  uint32_t lph=0, mph=0;
  bar_wait(bar_load,lph); lph^=1;

  float m_i=-INFINITY, l_i=0.f;
  int qmax=q0+BM-1; if(qmax>=S) qmax=S-1;
  int num_kt=qmax/BN+1;

  uint32_t taddrS=tmem+((uint32_t)(warp*32)<<16);
  uint32_t taddrO=tmem+((uint32_t)(warp*32)<<16)+64u;

  float s[64];
  uint32_t r[32];

  for(int tile=0;tile<num_kt;tile++){
    int kbase=tile*BN;
    if(tid==0){
      arrive_expect(bar_load,32768u);
      int rk=(int)(bh*S+kbase);
      tma_ld(&tmaK,bar_load,Ks[0],0,rk);
      tma_ld(&tmaK,bar_load,Ks[1],64,rk);
      tma_ld(&tmaV,bar_load,Vs[0],0,rk);
      tma_ld(&tmaV,bar_load,Vs[1],64,rk);
    }
    bar_wait(bar_load,lph); lph^=1;

    if(tid==0){
      #pragma unroll
      for(int sl=0;sl<8;sl++){
        uint32_t aq=(sl<4?cQ0:cQ1)+(uint32_t)((sl&3)*32);
        uint32_t ak=(sl<4?cK0:cK1)+(uint32_t)((sl&3)*32);
        umma_cg1(tmem, mkdesc(aq,1,1024,2), mkdesc(ak,1,1024,2), idesc_qk, sl?1:0);
      }
      commit_cg1(bar_mma);
    }
    bar_wait(bar_mma,mph); mph^=1;
    fence_after();

    tmem_ld_x32(taddrS,r); wait_ld();
    #pragma unroll
    for(int i=0;i<32;i++) s[i]=__uint_as_float(r[i]);
    tmem_ld_x32(taddrS+32,r); wait_ld();
    #pragma unroll
    for(int i=0;i<32;i++) s[32+i]=__uint_as_float(r[i]);

    #pragma unroll
    for(int c=0;c<64;c++){
      float v=s[c]*scale;
      int key=kbase+c;
      if(key>query || key>=S) v=-INFINITY;
      s[c]=v;
    }
    float tm=-INFINITY;
    #pragma unroll
    for(int c=0;c<64;c++) tm=fmaxf(tm,s[c]);
    float mnew=fmaxf(m_i,tm);
    float alpha=__expf(m_i-mnew);
    float ls=0.f;
    #pragma unroll
    for(int c=0;c<64;c++){ float p=__expf(s[c]-mnew); s[c]=p; ls+=p; }
    l_i=l_i*alpha+ls;
    m_i=mnew;

    uint4* P4=(uint4*)Ps;
    #pragma unroll
    for(int x=0;x<8;x++){
      uint4 v;
      v.x=pk(s[x*8+0],s[x*8+1]); v.y=pk(s[x*8+2],s[x*8+3]);
      v.z=pk(s[x*8+4],s[x*8+5]); v.w=pk(s[x*8+6],s[x*8+7]);
      P4[row*8 + ((row&7)^x)] = v;
    }
    fence_pa();

    if(tile>0){
      #pragma unroll
      for(int c=0;c<4;c++){
        tmem_ld_x32(taddrO+c*32,r); wait_ld();
        #pragma unroll
        for(int j=0;j<32;j++) r[j]=__float_as_uint(__uint_as_float(r[j])*alpha);
        tmem_st_x32(taddrO+c*32,r);
      }
      wait_st();
    }
    fence_before();
    __syncthreads();
    fence_after();

    if(tid==0){
      #pragma unroll
      for(int dh=0;dh<2;dh++){
        uint32_t dt=tmem+64u+(uint32_t)(dh*64);
        uint32_t cv=(dh?cV1:cV0);
        #pragma unroll
        for(int sl=0;sl<4;sl++){
          uint32_t ap=cP+(uint32_t)(sl*32);
          uint32_t av=cv+(uint32_t)(sl*2048);
          umma_cg1(dt, mkdesc(ap,1,1024,2), mkdesc(av,8192,1024,2), idesc_pv, (tile==0&&sl==0)?0:1);
        }
      }
      commit_cg1(bar_mma);
    }
    bar_wait(bar_mma,mph); mph^=1;
    fence_after();
  }

  float inv=(l_i>0.f)?1.0f/l_i:0.f;
  __nv_bfloat16* Og=O+(size_t)bh*S*Dc;
  #pragma unroll
  for(int c=0;c<4;c++){
    tmem_ld_x32(taddrO+c*32,r); wait_ld();
    __nv_bfloat16 tmp[32];
    #pragma unroll
    for(int j=0;j<32;j++) tmp[j]=__float2bfloat16(__uint_as_float(r[j])*inv);
    if(query<S){
      uint4* dst=(uint4*)(Og+(size_t)query*Dc+c*32);
      uint4* src=(uint4*)tmp;
      dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3];
    }
  }
  if(query<S) LSE[(size_t)bh*S+query]=m_i+logf(l_i);

  if(tid<32) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(tmem),"r"(256));
}

static CUresult make_tma(CUtensorMap* m,void* p,uint64_t inner,uint64_t outer,uint32_t bi,uint32_t bo){
  uint64_t gd[2]={inner,outer}; uint64_t gs[1]={inner*2};
  uint32_t bd[2]={bi,bo}; uint32_t es[2]={1,1};
  return cuTensorMapEncodeTiled(m,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,p,gd,gs,bd,es,
    CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bsz=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2);
  uint64_t rows=(uint64_t)Bsz*H*S;
  void* qp=Q.data_ptr(); void* kp=K.data_ptr(); void* vp=V.data_ptr();

  CUtensorMap tmaQ,tmaK,tmaV;
  CU_CHECK(make_tma(&tmaQ,qp,Dc,rows,DH,BM));
  CU_CHECK(make_tma(&tmaK,kp,Dc,rows,DH,BN));
  CU_CHECK(make_tma(&tmaV,vp,Dc,rows,DH,BN));

  __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
  float* Lp=static_cast<float*>(LSE.data_ptr());

  dim3 grid((S+BM-1)/BM, H, Bsz);
  dim3 block(THREADS);
  size_t smem=84*1024;

  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
  static bool once=false;
  if(!once){ cudaFuncSetAttribute(mha_fwd,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem); once=true; }

  mha_fwd<<<grid,block,smem,stream>>>(tmaQ,tmaK,tmaV,Op,Lp,S,H);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel