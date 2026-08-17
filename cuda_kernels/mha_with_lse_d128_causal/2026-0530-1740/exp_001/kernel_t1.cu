#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);} } while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char*s; cuGetErrorString(_e,&s); \
    fprintf(stderr,"CU error %s at %s:%d\n",s,__FILE__,__LINE__);} } while(0)

namespace mha_kernel {

constexpr int Dd=128;

__device__ __forceinline__ uint32_t csmem(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ float ex2f(float x){ float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y; }
__device__ __forceinline__ uint32_t pack2bf(float a,float b){ __nv_bfloat162 v=__floats2bfloat162_rn(a,b); return *reinterpret_cast<uint32_t*>(&v); }

__device__ __forceinline__ void mbar_init(uint64_t* b,uint32_t c){ asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(csmem(b)),"r"(c)); }
__device__ __forceinline__ void mbar_arrive_expect(uint64_t* b,uint32_t tx){ asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(csmem(b)),"r"(tx):"memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t* b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"::"r"(csmem(b)),"r"(ph):"memory"); }

__device__ __forceinline__ void tma3d(const CUtensorMap* d,uint64_t* bar,void* smem,int c0,int c1,int c2){
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4,%5}],[%2];"
    ::"r"(csmem(smem)),"l"((uint64_t)d),"r"(csmem(bar)),"r"(c0),"r"(c1),"r"(c2):"memory"); }

__device__ __forceinline__ void tmem_alloc1(uint32_t* dst,int n){ asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;"::"r"(csmem(dst)),"r"(n)); }
__device__ __forceinline__ void tmem_dealloc1(uint32_t a,int n){ asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"::"r"(a),"r"(n)); }

__device__ __forceinline__ void umma1(uint32_t td,uint64_t da,uint64_t db,uint32_t id,uint32_t acc){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\ntcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    ::"r"(td),"l"(da),"l"(db),"r"(id),"r"(acc):"memory"); }
__device__ __forceinline__ void umma_commit1(uint64_t* bar){ asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"::"r"(csmem(bar)):"memory"); }

__device__ __forceinline__ void tmem_ld_x8(uint32_t a,uint32_t* r){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
    :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]):"r"(a)); }
__device__ __forceinline__ void tmem_wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void fence_async(){ asm volatile("fence.proxy.async.shared::cta;":::"memory"); }

__device__ __forceinline__ uint64_t smem_desc(const void* ptr,uint32_t lbo,uint32_t sbo){
  uint64_t d=0; uint32_t a=csmem(ptr);
  d|=(uint64_t)((a&0x3FFFF)>>4);
  d|=(uint64_t)((lbo&0x3FFFF)>>4)<<16;
  d|=(uint64_t)((sbo&0x3FFFF)>>4)<<32;
  d|=(uint64_t)1<<46;
  d|=(uint64_t)2<<61;
  return d; }
__device__ __forceinline__ uint32_t instr_desc(uint32_t M,uint32_t N,uint32_t amaj,uint32_t bmaj){
  uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10);
  d|=(amaj<<15); d|=(bmaj<<16); d|=((N>>3)<<17); d|=((M>>4)<<24); return d; }

__global__ void __launch_bounds__(128,1) attn(
    const __grid_constant__ CUtensorMap descQ,
    const __grid_constant__ CUtensorMap descK,
    const __grid_constant__ CUtensorMap descV,
    __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int B,int H,int S){

  const float SCALE2 = 0.08838834764831845f*1.4426950408889634f;
  const float LN2 = 0.6931471805599453f;

  extern __shared__ __align__(1024) char smem[];
  __nv_bfloat16* Qs=(__nv_bfloat16*)(smem+0);
  __nv_bfloat16* Ks=(__nv_bfloat16*)(smem+32768);
  __nv_bfloat16* Vs=(__nv_bfloat16*)(smem+65536);
  __nv_bfloat16* Ps=(__nv_bfloat16*)(smem+98304);
  uint64_t* bars=(uint64_t*)(smem+131072);
  uint64_t* bar_q=&bars[0]; uint64_t* bar_kv=&bars[1]; uint64_t* bar_s=&bars[2]; uint64_t* bar_o=&bars[3];
  uint32_t* tmem_ptr=(uint32_t*)(smem+131072+64);

  int tid=threadIdx.x;
  int warp=tid>>5;
  int bh=blockIdx.z*H+blockIdx.y;
  int qb=blockIdx.x*128;
  int qg=qb+tid;

  if(tid==0){ mbar_init(bar_q,1); mbar_init(bar_kv,1); mbar_init(bar_s,1); mbar_init(bar_o,1); }
  if(warp==0){ tmem_alloc1(tmem_ptr,256); }
  __syncthreads();
  uint32_t tmem_base=*tmem_ptr;
  uint32_t tmem_S=tmem_base;
  uint32_t tmem_O=tmem_base+128;
  uint32_t laneoff=((uint32_t)warp*32u)<<16;

  uint32_t idesc_qk=instr_desc(128,128,0,0);
  uint32_t idesc_pv=instr_desc(128,64,0,1);

  // Load Q once
  if(tid==0){
    mbar_arrive_expect(bar_q,32768);
    tma3d(&descQ,bar_q,Qs+0,    0,  qb, bh);
    tma3d(&descQ,bar_q,Qs+8192, 64, qb, bh);
  }
  mbar_wait(bar_q,0);
  __syncthreads();

  float Oreg[128];
  #pragma unroll
  for(int i=0;i<128;i++) Oreg[i]=0.f;
  float m=-1e30f, l=0.f;

  int qmax=qb+127; if(qmax>S-1) qmax=S-1;
  int kv_last=qmax/128;

  for(int kbi=0;kbi<=kv_last;kbi++){
    int kb=kbi*128;
    uint32_t ph=(uint32_t)(kbi&1);

    if(tid==0){
      mbar_arrive_expect(bar_kv,65536);
      tma3d(&descK,bar_kv,Ks+0,    0,  kb, bh);
      tma3d(&descK,bar_kv,Ks+8192, 64, kb, bh);
      tma3d(&descV,bar_kv,Vs+0,    0,  kb, bh);
      tma3d(&descV,bar_kv,Vs+8192, 64, kb, bh);
    }
    mbar_wait(bar_kv,ph);

    if(tid==0){
      #pragma unroll
      for(int s=0;s<8;s++){
        int atom=s/4, sub=s%4;
        uint64_t da=smem_desc(Qs+atom*8192+sub*16,16,1024);
        uint64_t db=smem_desc(Ks+atom*8192+sub*16,16,1024);
        umma1(tmem_S,da,db,idesc_qk,s>0?1:0);
      }
      umma_commit1(bar_s);
    }
    mbar_wait(bar_s,ph);
    __syncthreads();

    // pass1: rowmax
    float tmax=-1e30f;
    for(int c=0;c<16;c++){
      uint32_t taddr=tmem_S+laneoff+(uint32_t)(c*8);
      uint32_t r[8]; tmem_ld_x8(taddr,r); tmem_wait_ld();
      #pragma unroll
      for(int j=0;j<8;j++){
        int key=kb+c*8+j;
        float sv=__uint_as_float(r[j])*SCALE2;
        if(key>qg||key>=S) sv=-1e30f;
        tmax=fmaxf(tmax,sv);
      }
    }
    float newm=fmaxf(m,tmax);
    float corr=ex2f(m-newm);
    #pragma unroll
    for(int i=0;i<128;i++) Oreg[i]*=corr;
    l*=corr; m=newm;

    // pass2: exp -> P -> SMEM
    for(int c=0;c<16;c++){
      uint32_t taddr=tmem_S+laneoff+(uint32_t)(c*8);
      uint32_t r[8]; tmem_ld_x8(taddr,r); tmem_wait_ld();
      float pv[8]; float psum=0.f;
      #pragma unroll
      for(int j=0;j<8;j++){
        int key=kb+c*8+j;
        float sv=__uint_as_float(r[j])*SCALE2;
        float p=(key>qg||key>=S)?0.0f:ex2f(sv-newm);
        pv[j]=p; psum+=p;
      }
      l+=psum;
      int atom=c>>3, cia=c&7;
      int idx=atom*8192 + tid*64 + (((tid&7)^cia)*8);
      int4 v=make_int4((int)pack2bf(pv[0],pv[1]),(int)pack2bf(pv[2],pv[3]),
                       (int)pack2bf(pv[4],pv[5]),(int)pack2bf(pv[6],pv[7]));
      *reinterpret_cast<int4*>(Ps+idx)=v;
    }
    fence_async();
    __syncthreads();

    if(tid==0){
      #pragma unroll
      for(int h=0;h<2;h++){
        #pragma unroll
        for(int s=0;s<8;s++){
          int atom=s/4, sub=s%4;
          uint64_t da=smem_desc(Ps+atom*8192+sub*16,16,1024);
          uint64_t db=smem_desc(Vs+h*8192+s*1024,16384,1024);
          umma1(tmem_O+(uint32_t)(h*64),da,db,idesc_pv,s>0?1:0);
        }
      }
      umma_commit1(bar_o);
    }
    mbar_wait(bar_o,ph);
    __syncthreads();

    // read O_partial -> Oreg
    #pragma unroll
    for(int c=0;c<16;c++){
      uint32_t taddr=tmem_O+laneoff+(uint32_t)(c*8);
      uint32_t r[8]; tmem_ld_x8(taddr,r); tmem_wait_ld();
      #pragma unroll
      for(int j=0;j<8;j++) Oreg[c*8+j]+=__uint_as_float(r[j]);
    }
    __syncthreads();
  }

  float inv=1.0f/l;
  if(qg<S){
    __nv_bfloat16* Og=O+(size_t)bh*S*128+(size_t)qg*128;
    #pragma unroll
    for(int d=0;d<128;d+=2){
      __nv_bfloat162 v=__floats2bfloat162_rn(Oreg[d]*inv,Oreg[d+1]*inv);
      *reinterpret_cast<__nv_bfloat162*>(Og+d)=v;
    }
    LSE[(size_t)bh*S+qg]=m*LN2+logf(l);
  }

  __syncthreads();
  if(warp==0) tmem_dealloc1(tmem_base,256);
}

static CUresult make_tma(CUtensorMap* d,void* ptr,int B,int H,int S){
  uint64_t gdim[3]={(uint64_t)Dd,(uint64_t)S,(uint64_t)(B*H)};
  uint64_t gstr[2]={(uint64_t)Dd*2,(uint64_t)S*Dd*2};
  uint32_t bdim[3]={64,128,1};
  uint32_t estr[3]={1,1,1};
  return cuTensorMapEncodeTiled(d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,ptr,gdim,gstr,bdim,estr,
    CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q,tvm::ffi::TensorView K,tvm::ffi::TensorView V,
         tvm::ffi::TensorView O,tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2);

  void* Qp=Q.data_ptr(); void* Kp=K.data_ptr(); void* Vp=V.data_ptr();
  __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
  float* LSEp=static_cast<float*>(LSE.data_ptr());

  CUtensorMap dQ,dK,dV;
  CU_CHECK(make_tma(&dQ,Qp,B,H,S));
  CU_CHECK(make_tma(&dK,Kp,B,H,S));
  CU_CHECK(make_tma(&dV,Vp,B,H,S));

  int nq=(S+127)/128;
  dim3 grid(nq,H,B); dim3 block(128);
  size_t smem=131072+1024;
  static bool init=false;
  cudaFuncSetAttribute(attn,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
  init=true;(void)init;

  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
  attn<<<grid,block,smem,stream>>>(dQ,dK,dV,Op,LSEp,B,H,S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel