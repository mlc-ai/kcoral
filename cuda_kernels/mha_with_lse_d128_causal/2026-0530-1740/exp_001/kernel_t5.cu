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
__device__ __forceinline__ void tmem_st_x8(uint32_t a,const uint32_t* r){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0],{%1,%2,%3,%4,%5,%6,%7,%8};"
    ::"r"(a),"r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]):"memory"); }
__device__ __forceinline__ void tmem_wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void tmem_wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;":::"memory"); }
__device__ __forceinline__ void fence_async(){ asm volatile("fence.proxy.async.shared::cta;":::"memory"); }
__device__ __forceinline__ void tc_fence_before(){ asm volatile("tcgen05.fence::before_thread_sync;":::"memory"); }
__device__ __forceinline__ void tc_fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }

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

__device__ __forceinline__ void do_QK(__nv_bfloat16* Qs,__nv_bfloat16* Ksb,uint32_t tmemS,uint32_t idesc){
  #pragma unroll
  for(int s=0;s<8;s++){ int atom=s/4,sub=s%4;
    uint64_t da=smem_desc(Qs+atom*8192+sub*16,16,1024);
    uint64_t db=smem_desc(Ksb+atom*8192+sub*16,16,1024);
    umma1(tmemS,da,db,idesc,(s==0)?0u:1u);
  }
}
__device__ __forceinline__ void do_PV(__nv_bfloat16* Psb,__nv_bfloat16* Vsb,uint32_t tmemO,uint32_t idesc,bool clear){
  #pragma unroll
  for(int h=0;h<2;h++)
    #pragma unroll
    for(int s=0;s<8;s++){ int atom=s/4,sub=s%4;
      uint64_t da=smem_desc(Psb+atom*8192+sub*16,16,1024);
      uint64_t db=smem_desc(Vsb+h*8192+s*1024,16384,1024);
      umma1(tmemO+(uint32_t)(h*64),da,db,idesc,(clear&&s==0)?0u:1u);
    }
}

__global__ void __launch_bounds__(128,1) attn(
    const __grid_constant__ CUtensorMap descQ,
    const __grid_constant__ CUtensorMap descK,
    const __grid_constant__ CUtensorMap descV,
    __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int B,int H,int S){

  const float SCALE2 = 0.08838834764831845f*1.4426950408889634f;
  const float LN2 = 0.6931471805599453f;

  extern __shared__ __align__(1024) char smem[];
  __nv_bfloat16* Qs =(__nv_bfloat16*)(smem+0);
  __nv_bfloat16* Ks[2]={(__nv_bfloat16*)(smem+32768),(__nv_bfloat16*)(smem+65536)};
  __nv_bfloat16* Vs[2]={(__nv_bfloat16*)(smem+98304),(__nv_bfloat16*)(smem+131072)};
  __nv_bfloat16* Ps[2]={(__nv_bfloat16*)(smem+163840),(__nv_bfloat16*)(smem+196608)};
  uint64_t* bars=(uint64_t*)(smem+229376);
  uint32_t* tmem_ptr=(uint32_t*)(smem+229376+64);
  // bars: 0=q 1=k0 2=k1 3=v0 4=v1 5=s0 6=s1 7=o

  int tid=threadIdx.x;
  int warp=tid>>5;
  int bh=blockIdx.z*H+blockIdx.y;
  int qb=blockIdx.x*128;
  int qg=qb+tid;

  if(tid==0){ for(int k=0;k<8;k++) mbar_init(&bars[k],1); }
  if(warp==0){ tmem_alloc1(tmem_ptr,512); }
  __syncthreads();
  uint32_t tmem_base=*tmem_ptr;
  uint32_t tmem_S[2]={tmem_base+0,tmem_base+128};
  uint32_t tmem_O=tmem_base+256;
  uint32_t laneoff=((uint32_t)warp*32u)<<16;

  uint32_t idesc_qk=instr_desc(128,128,0,0);
  uint32_t idesc_pv=instr_desc(128,64,0,1);

  int qmax=qb+127; if(qmax>S-1) qmax=S-1;
  int n=qmax/128+1;

  uint32_t ph[8]={0,0,0,0,0,0,0,0};

  // ---- prologue ----
  if(tid==0){
    mbar_arrive_expect(&bars[0],32768);
    tma3d(&descQ,&bars[0],Qs,0,qb,bh);
    tma3d(&descQ,&bars[0],Qs+8192,64,qb,bh);
    mbar_arrive_expect(&bars[1],32768);
    tma3d(&descK,&bars[1],Ks[0],0,0,bh);
    tma3d(&descK,&bars[1],Ks[0]+8192,64,0,bh);
    mbar_arrive_expect(&bars[3],32768);
    tma3d(&descV,&bars[3],Vs[0],0,0,bh);
    tma3d(&descV,&bars[3],Vs[0]+8192,64,0,bh);
    if(n>1){
      mbar_arrive_expect(&bars[2],32768);
      tma3d(&descK,&bars[2],Ks[1],0,128,bh);
      tma3d(&descK,&bars[2],Ks[1]+8192,64,128,bh);
    }
    mbar_wait(&bars[0],0);
    mbar_wait(&bars[1],ph[1]); ph[1]^=1;
    do_QK(Qs,Ks[0],tmem_S[0],idesc_qk);
    umma_commit1(&bars[5]);
  }

  float m=-1e30f, l=0.f;

  for(int i=0;i<n;i++){
    int cur=i&1, nb=(i+1)&1, kb=i*128;
    bool diag=(i==n-1);

    // (1) wait QK[i]
    mbar_wait(&bars[5+cur],ph[5+cur]); ph[5+cur]^=1;

    // (2) issue QK[i+1] (overlaps softmax) + safe K prefetch
    if(tid==0){
      if(i+1<n){
        mbar_wait(&bars[1+nb],ph[1+nb]); ph[1+nb]^=1;
        do_QK(Qs,Ks[nb],tmem_S[nb],idesc_qk);
        umma_commit1(&bars[5+nb]);
      }
      if(i+2<n){
        mbar_arrive_expect(&bars[1+cur],32768);
        tma3d(&descK,&bars[1+cur],Ks[cur],0,(i+2)*128,bh);
        tma3d(&descK,&bars[1+cur],Ks[cur]+8192,64,(i+2)*128,bh);
      }
    }

    // (3) softmax[i] : batched TMEM read, max, exp -> P[cur]
    float Sr[128];
    #pragma unroll
    for(int c=0;c<16;c++) tmem_ld_x8(tmem_S[cur]+laneoff+(uint32_t)(c*8),(uint32_t*)&Sr[c*8]);
    tmem_wait_ld();

    float tmax=-1e30f;
    if(!diag){
      #pragma unroll
      for(int t=0;t<128;t++){ float v=Sr[t]*SCALE2; Sr[t]=v; tmax=fmaxf(tmax,v); }
    } else {
      #pragma unroll
      for(int c=0;c<16;c++)
        #pragma unroll
        for(int j=0;j<8;j++){
          int t=c*8+j; int key=kb+t;
          float v=Sr[t]*SCALE2;
          v=(key<=qg && key<S)? v : -1e30f;
          Sr[t]=v; tmax=fmaxf(tmax,v);
        }
    }
    float newm=fmaxf(m,tmax);
    float corr=ex2f(m-newm);
    unsigned rv=__ballot_sync(0xffffffffu, corr<1.0f);

    float l_add=0.f;
    #pragma unroll
    for(int c=0;c<16;c++){
      float p0=ex2f(Sr[c*8+0]-newm), p1=ex2f(Sr[c*8+1]-newm);
      float p2=ex2f(Sr[c*8+2]-newm), p3=ex2f(Sr[c*8+3]-newm);
      float p4=ex2f(Sr[c*8+4]-newm), p5=ex2f(Sr[c*8+5]-newm);
      float p6=ex2f(Sr[c*8+6]-newm), p7=ex2f(Sr[c*8+7]-newm);
      l_add += (p0+p1)+(p2+p3)+(p4+p5)+(p6+p7);
      int atom=c>>3, cia=c&7;
      int idx=atom*8192 + tid*64 + (((tid&7)^cia)*8);
      int4 v=make_int4((int)pack2bf(p0,p1),(int)pack2bf(p2,p3),
                       (int)pack2bf(p4,p5),(int)pack2bf(p6,p7));
      *reinterpret_cast<int4*>(Ps[cur]+idx)=v;
    }
    l=l*corr+l_add; m=newm;

    // (4) wait previous PV done; safe V prefetch; rescale O (batched)
    if(i>0){ mbar_wait(&bars[7],ph[7]); ph[7]^=1; }
    if(tid==0 && i+1<n){
      mbar_arrive_expect(&bars[3+nb],32768);
      tma3d(&descV,&bars[3+nb],Vs[nb],0,(i+1)*128,bh);
      tma3d(&descV,&bars[3+nb],Vs[nb]+8192,64,(i+1)*128,bh);
    }
    if(i>0 && rv){
      uint32_t Or[128];
      #pragma unroll
      for(int c=0;c<16;c++) tmem_ld_x8(tmem_O+laneoff+(uint32_t)(c*8),&Or[c*8]);
      tmem_wait_ld();
      #pragma unroll
      for(int t=0;t<128;t++) Or[t]=__float_as_uint(__uint_as_float(Or[t])*corr);
      #pragma unroll
      for(int c=0;c<16;c++) tmem_st_x8(tmem_O+laneoff+(uint32_t)(c*8),&Or[c*8]);
      tmem_wait_st();
    }

    // (5) fence + issue PV[i]
    tc_fence_before();
    fence_async();
    __syncthreads();
    if(tid==0){
      mbar_wait(&bars[3+cur],ph[3+cur]); ph[3+cur]^=1;
      tc_fence_after();
      do_PV(Ps[cur],Vs[cur],tmem_O,idesc_pv,(i==0));
      umma_commit1(&bars[7]);
    }
  }

  // ---- final PV ----
  mbar_wait(&bars[7],ph[7]); ph[7]^=1;

  // ---- epilogue (batched read) ----
  float inv=(l>0.f)?(1.0f/l):0.0f;
  uint32_t Or[128];
  #pragma unroll
  for(int c=0;c<16;c++) tmem_ld_x8(tmem_O+laneoff+(uint32_t)(c*8),&Or[c*8]);
  tmem_wait_ld();
  if(qg<S){
    __nv_bfloat16* Og=O+(size_t)bh*S*128+(size_t)qg*128;
    #pragma unroll
    for(int c=0;c<16;c++)
      #pragma unroll
      for(int j=0;j<8;j+=2){
        __nv_bfloat162 v=__floats2bfloat162_rn(__uint_as_float(Or[c*8+j])*inv,__uint_as_float(Or[c*8+j+1])*inv);
        *reinterpret_cast<__nv_bfloat162*>(Og+c*8+j)=v;
      }
    LSE[(size_t)bh*S+qg]=m*LN2+logf(l);
  }

  __syncthreads();
  if(warp==0) tmem_dealloc1(tmem_base,512);
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
  size_t smem=229376+256;
  cudaFuncSetAttribute(attn,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);

  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
  attn<<<grid,block,smem,stream>>>(dQ,dK,dV,Op,LSEp,B,H,S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel