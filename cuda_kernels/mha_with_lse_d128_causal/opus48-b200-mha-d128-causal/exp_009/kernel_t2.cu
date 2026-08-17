#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do{cudaError_t _e=(call);if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s @%s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);exit(1);}}while(0)
#define CU_CHECK(call) do{CUresult _e=(call);if(_e!=CUDA_SUCCESS){const char*s;cuGetErrorString(_e,&s);fprintf(stderr,"CU %s @%s:%d\n",s,__FILE__,__LINE__);exit(1);}}while(0)

namespace mha_kernel {
constexpr int D=128, BM=128, BN=128;

__device__ __forceinline__ void init_bar(uint64_t* b,uint32_t c){asm volatile("mbarrier.init.shared.b64 [%0],%1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c));}
__device__ __forceinline__ void fence_bar_init(){asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}
__device__ __forceinline__ void bar_arrive_tx(uint64_t* b,uint32_t tx){asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(tx):"memory");}
__device__ __forceinline__ void bar_wait(uint64_t* b,uint32_t ph){asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph));}
__device__ __forceinline__ void tma_load(const CUtensorMap* d,uint64_t* bar,void* smem,int32_t c0,int32_t c1){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4}],[%2];"
    ::"r"((uint32_t)__cvta_generic_to_shared(smem)),"l"((uint64_t)d),"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c0),"r"(c1):"memory");}
__device__ __forceinline__ void fence_proxy_async(){asm volatile("fence.proxy.async;":::"memory");}
__device__ __forceinline__ void tcg_fence_before(){asm volatile("tcgen05.fence::before_thread_sync;":::"memory");}
__device__ __forceinline__ void tcg_fence_after(){asm volatile("tcgen05.fence::after_thread_sync;":::"memory");}

__device__ __forceinline__ void tmem_alloc(uint32_t* dst,int nc){asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;"::"r"((uint32_t)__cvta_generic_to_shared(dst)),"r"(nc));}
__device__ __forceinline__ void tmem_dealloc(uint32_t a,int nc){asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"::"r"(a),"r"(nc));}
__device__ __forceinline__ void umma(uint32_t td,uint64_t da,uint64_t db,uint32_t id,uint32_t acc){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\ntcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"::"r"(td),"l"(da),"l"(db),"r"(id),"r"(acc));}
__device__ __forceinline__ void umma_commit(uint64_t* b){asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"::"r"((uint32_t)__cvta_generic_to_shared(b)));}
__device__ __forceinline__ void tmem_ld4(uint32_t col,uint32_t*r0,uint32_t*r1,uint32_t*r2,uint32_t*r3){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];":"=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3):"r"(col));}
__device__ __forceinline__ void tmem_st4(uint32_t col,uint32_t v0,uint32_t v1,uint32_t v2,uint32_t v3){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0],{%1,%2,%3,%4};"::"r"(col),"r"(v0),"r"(v1),"r"(v2),"r"(v3):"memory");}
__device__ __forceinline__ void tmem_wait_ld(){asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");}
__device__ __forceinline__ void tmem_wait_st(){asm volatile("tcgen05.wait::st.sync.aligned;":::"memory");}

__device__ __forceinline__ uint64_t make_desc(void* p,uint32_t lbo,uint32_t sbo){
  uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
  d|=(uint64_t)(a&0x3FFFF)>>4;
  d|=(uint64_t)((lbo&0x3FFFF)>>4)<<16;
  d|=(uint64_t)((sbo&0x3FFFF)>>4)<<32;
  d|=(uint64_t)1<<46;
  d|=(uint64_t)2<<61;
  return d;
}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M,uint32_t N,uint32_t aT,uint32_t bT){
  uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10); d|=(aT<<15); d|=(bT<<16); d|=((N/8)<<17); d|=((M/16)<<24); return d;}
__device__ __forceinline__ uint32_t packh(__nv_bfloat16 a,__nv_bfloat16 b){__nv_bfloat162 v=__halves2bfloat162(a,b);return *reinterpret_cast<uint32_t*>(&v);}

__global__ __launch_bounds__(128) void attn(
    const __grid_constant__ CUtensorMap tmaQ,
    const __grid_constant__ CUtensorMap tmaK,
    const __grid_constant__ CUtensorMap tmaV,
    __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int S, int H)
{
  int b=blockIdx.z,h=blockIdx.y,qtile=blockIdx.x,bh=b*H+h;
  int q_start=qtile*BM, tid=threadIdx.x, q=q_start+tid;

  extern __shared__ char smem_raw[];
  char* sm=(char*)(((uintptr_t)smem_raw+1023)&~(uintptr_t)1023);
  __nv_bfloat16* Qh=(__nv_bfloat16*)(sm+0);
  __nv_bfloat16* Kh=(__nv_bfloat16*)(sm+32768);
  __nv_bfloat16* Vh=(__nv_bfloat16*)(sm+65536);
  char* Pb=sm+98304;
  uint64_t* bar_tma=(uint64_t*)(sm+131072);
  uint64_t* bar_mma=(uint64_t*)(sm+131072+8);
  uint32_t* tmem_ptr=(uint32_t*)(sm+131072+16);

  if(tid==0){ init_bar(bar_tma,1); init_bar(bar_mma,1); fence_bar_init(); }
  if(tid<32){ tmem_alloc(tmem_ptr,256); }
  __syncthreads();
  uint32_t tmem_base=*tmem_ptr;
  uint32_t Sc=tmem_base, Oc=tmem_base+128;
  uint32_t idesc_qk=make_idesc(128,128,0,0);
  uint32_t idesc_pv=make_idesc(128,64,0,1);

  int phase_tma=0,phase_mma=0;
  const float scale=rsqrtf(128.0f);

  if(tid==0){
    bar_arrive_tx(bar_tma,2*16384);
    int r=bh*S+q_start;
    tma_load(&tmaQ,bar_tma,Qh+0,0,r);
    tma_load(&tmaQ,bar_tma,Qh+8192,64,r);
  }
  bar_wait(bar_tma,phase_tma); phase_tma^=1;

  float m_prev=-INFINITY,l_prev=0.f,corr=0.f;
  int num_kv=qtile+1, cap=(S+BN-1)/BN; if(num_kv>cap)num_kv=cap;

  for(int kvt=0;kvt<num_kv;kvt++){
    int kv_start=kvt*BN; bool first=(kvt==0);
    __syncthreads();
    if(tid==0){
      bar_arrive_tx(bar_tma,4*16384);
      int r=bh*S+kv_start;
      tma_load(&tmaK,bar_tma,Kh+0,0,r);
      tma_load(&tmaK,bar_tma,Kh+8192,64,r);
      tma_load(&tmaV,bar_tma,Vh+0,0,r);
      tma_load(&tmaV,bar_tma,Vh+8192,64,r);
    }
    bar_wait(bar_tma,phase_tma); phase_tma^=1;

    if(tid==0){
      #pragma unroll
      for(int j=0;j<8;j++){
        void* ap=sm+0+(j/4)*16384+(j%4)*32;
        void* bp=sm+32768+(j/4)*16384+(j%4)*32;
        umma(Sc,make_desc(ap,0,1024),make_desc(bp,0,1024),idesc_qk,j==0?0:1);
      }
      umma_commit(bar_mma);
    }
    bar_wait(bar_mma,phase_mma); phase_mma^=1;

    float new_m,sum;
    {
      uint32_t su[128];
      #pragma unroll
      for(int c=0;c<128;c+=4) tmem_ld4(Sc+c,&su[c],&su[c+1],&su[c+2],&su[c+3]);
      tmem_wait_ld();
      float rmax=-INFINITY;
      #pragma unroll
      for(int kk=0;kk<128;kk++){
        int key=kv_start+kk; float v=__uint_as_float(su[kk]);
        if(key>q||key>=S) v=-INFINITY;
        su[kk]=__float_as_uint(v); rmax=fmaxf(rmax,v);
      }
      rmax*=scale;
      new_m=fmaxf(m_prev,rmax);
      corr=(m_prev==-INFINITY)?0.f:__expf(m_prev-new_m);
      sum=0.f;
      int r8=tid&7, rb=tid>>3;
      #pragma unroll
      for(int c=0;c<2;c++){
        #pragma unroll
        for(int x=0;x<8;x++){
          __align__(16) __nv_bfloat16 tmp[8];
          #pragma unroll
          for(int e=0;e<8;e++){
            int kk=c*64+x*8+e;
            float p=__expf(__uint_as_float(su[kk])*scale-new_m);
            sum+=p; tmp[e]=__float2bfloat16(p);
          }
          uint32_t off=c*16384+rb*1024+r8*128+((r8^x)*16);
          *reinterpret_cast<int4*>(Pb+off)=*reinterpret_cast<int4*>(tmp);
        }
      }
    }
    l_prev=l_prev*corr+sum; m_prev=new_m;

    if(!first){
      uint32_t ou[128];
      #pragma unroll
      for(int c=0;c<128;c+=4) tmem_ld4(Oc+c,&ou[c],&ou[c+1],&ou[c+2],&ou[c+3]);
      tmem_wait_ld();
      #pragma unroll
      for(int c=0;c<128;c++) ou[c]=__float_as_uint(__uint_as_float(ou[c])*corr);
      #pragma unroll
      for(int c=0;c<128;c+=4) tmem_st4(Oc+c,ou[c],ou[c+1],ou[c+2],ou[c+3]);
      tmem_wait_st();
    }
    tcg_fence_before();
    __syncthreads();

    if(tid==0){
      fence_proxy_async();
      tcg_fence_after();
      #pragma unroll
      for(int dc=0;dc<2;dc++){
        #pragma unroll
        for(int j=0;j<8;j++){
          void* ap=sm+98304+(j/4)*16384+(j%4)*32;
          void* bp=sm+65536+dc*16384+j*2048;
          uint32_t acc=(first&&j==0)?0:1;
          umma(Oc+dc*64,make_desc(ap,0,1024),make_desc(bp,16384,1024),idesc_pv,acc);
        }
      }
      umma_commit(bar_mma);
    }
    bar_wait(bar_mma,phase_mma); phase_mma^=1;
  }

  uint32_t ou[128];
  #pragma unroll
  for(int c=0;c<128;c+=4) tmem_ld4(Oc+c,&ou[c],&ou[c+1],&ou[c+2],&ou[c+3]);
  tmem_wait_ld();
  float l=l_prev, inv=(l>0.f)?(1.f/l):0.f;
  if(q<S){
    __nv_bfloat16* Orow=O+(long)bh*S*D+(long)q*D;
    #pragma unroll
    for(int c=0;c<128;c+=2){
      __nv_bfloat16 a=__float2bfloat16(__uint_as_float(ou[c])*inv);
      __nv_bfloat16 bb=__float2bfloat16(__uint_as_float(ou[c+1])*inv);
      *reinterpret_cast<uint32_t*>(Orow+c)=packh(a,bb);
    }
    LSE[(long)bh*S+q]=(l>0.f)?(m_prev+__logf(l)):(-INFINITY);
  }
  __syncthreads();
  if(tid<32) tmem_dealloc(tmem_base,256);
}

CUresult make_tma(CUtensorMap* d,void* ptr,uint64_t rows){
  uint64_t gd[2]={128,rows}; uint64_t gs[1]={128*2}; uint32_t bd[2]={64,128}; uint32_t es[2]={1,1};
  return cuTensorMapEncodeTiled(d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,ptr,gd,gs,bd,es,
    CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q,tvm::ffi::TensorView K,tvm::ffi::TensorView V,
         tvm::ffi::TensorView O,tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bsz=Q.size(0),Hh=Q.size(1),S=Q.size(2);
  uint64_t rows=(uint64_t)Bsz*Hh*S;
  void* Qp=Q.data_ptr(); void* Kp=K.data_ptr(); void* Vp=V.data_ptr();
  __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
  float* Lp=static_cast<float*>(LSE.data_ptr());

  CUtensorMap tQ,tK,tV;
  CU_CHECK(make_tma(&tQ,Qp,rows));
  CU_CHECK(make_tma(&tK,Kp,rows));
  CU_CHECK(make_tma(&tV,Vp,rows));

  int nq=(S+BM-1)/BM;
  dim3 grid(nq,Hh,Bsz), block(128);
  int smem=133120;
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
  CUDA_CHECK(cudaFuncSetAttribute(attn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  attn<<<grid,block,smem,stream>>>(tQ,tK,tV,Op,Lp,S,Hh);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);
}  // namespace mha_kernel