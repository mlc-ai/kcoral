#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} }while(0)

namespace mha_causal {

constexpr int D=128, BM=128, BN=64, THREADS=128, NB=2;
constexpr float LN2=0.6931471805599453f;
constexpr unsigned FULL=0xffffffffu;

__device__ __forceinline__ float ex2(float x){ float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y; }
__device__ __forceinline__ uint32_t sa(const void*p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void init_bar(uint64_t*b,int c){ asm volatile("mbarrier.init.shared.b64 [%0],%1;"::"r"(sa(b)),"r"(c)); }
__device__ __forceinline__ void fence_bi(){ asm volatile("fence.mbarrier_init.release.cluster;":::"memory"); }
__device__ __forceinline__ void bar_wait(uint64_t*b,uint32_t ph){ asm volatile("{ .reg .pred P; W%=: mbarrier.try_wait.parity.shared.b64 P,[%0],%1; @!P bra W%=; }"::"r"(sa(b)),"r"(ph)); }
__device__ __forceinline__ void fence_async(){ asm volatile("fence.proxy.async;":::"memory"); }
__device__ __forceinline__ void tf_before(){ asm volatile("tcgen05.fence::before_thread_sync;":::"memory"); }
__device__ __forceinline__ void tf_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;":::"memory"); }
template<int N> __device__ __forceinline__ void cp_wait(){ asm volatile("cp.async.wait_group %0;"::"n"(N):"memory"); }
__device__ __forceinline__ void cpa16(uint32_t dst,const void*src,int bytes){ asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;\n"::"r"(dst),"l"(src),"r"(bytes):"memory"); }
__device__ __forceinline__ void tmem_alloc1(uint32_t*dst,int n){ asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;"::"r"(sa(dst)),"r"(n)); }
__device__ __forceinline__ void tmem_dealloc1(uint32_t a,int n){ asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"::"r"(a),"r"(n)); }
__device__ __forceinline__ void umma1(uint32_t d,uint64_t a,uint64_t b,uint32_t id,int acc){
  asm volatile("{ .reg .pred p; setp.ne.b32 p,%4,0; tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p; }"::"r"(d),"l"(a),"l"(b),"r"(id),"r"(acc)); }
__device__ __forceinline__ void commit1(uint64_t*b){ asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"::"r"(sa(b)):"memory"); }
__device__ __forceinline__ void wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;":::"memory"); }
__device__ __forceinline__ void ld16(uint32_t a,uint32_t*r){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},[%16];"
   :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
    "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]):"r"(a)); }
__device__ __forceinline__ void st16(uint32_t a,const uint32_t*r){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0],{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
   ::"r"(a),"r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),
     "r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),"r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]):"memory"); }
__device__ __forceinline__ uint64_t mkdesc(const __nv_bfloat16*p,uint32_t lbo,uint32_t sbo){
  uint64_t d=0; uint32_t a=sa(p);
  d|=(uint64_t)((a&0x3FFFF)>>4); d|=(uint64_t)((lbo&0x3FFFF)>>4)<<16;
  d|=(uint64_t)((sbo&0x3FFFF)>>4)<<32; d|=(uint64_t)1<<46; return d; }
__device__ __forceinline__ uint32_t mkidesc(uint32_t M,uint32_t N,uint32_t am,uint32_t bm){
  uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10); d|=(am<<15); d|=(bm<<16);
  d|=((N>>3)<<17); d|=((M>>4)<<24); return d; }

__global__ __launch_bounds__(THREADS,2) void attn(
   const __nv_bfloat16* __restrict__ Qg,const __nv_bfloat16* __restrict__ Kg,
   const __nv_bfloat16* __restrict__ Vg,__nv_bfloat16* __restrict__ Og,
   float* __restrict__ LSEg,int S,float sl2)
{
  extern __shared__ __align__(1024) __nv_bfloat16 smem[];
  __nv_bfloat16* Qs=smem;
  __nv_bfloat16* Ks=Qs+BM*D;
  __nv_bfloat16* Vs=Ks+NB*BN*D;
  __nv_bfloat16* Ps=Vs+NB*BN*D;
  __shared__ uint64_t bar[2];
  __shared__ uint32_t tmem_slot;

  int tid=threadIdx.x, q0=blockIdx.x*BM, bh=blockIdx.y, qg=q0+tid;

  if(tid==0){ init_bar(&bar[0],1); init_bar(&bar[1],1); fence_bi(); }
  __syncthreads();
  if(tid<32) tmem_alloc1(&tmem_slot,256);
  __syncthreads();
  uint32_t tmem=tmem_slot, Saddr=tmem, Oaddr=tmem+64;
  uint32_t idQK=mkidesc(128,64,0,0), idPV=mkidesc(128,128,0,1);

  const __nv_bfloat16* Qbh=Qg+(int64_t)bh*S*D;
  const __nv_bfloat16* Kbh=Kg+(int64_t)bh*S*D;
  const __nv_bfloat16* Vbh=Vg+(int64_t)bh*S*D;

  int last_key=min(q0+BM-1,S-1), ntiles=last_key/BN+1;

  auto load_kv=[&](int buf,int j){
    __nv_bfloat16* Kb=Ks+buf*(BN*D); __nv_bfloat16* Vb=Vs+buf*(BN*D);
    for(int c=tid;c<(BN*D)/8;c+=THREADS){
      int n=c/(D/8), dg=c%(D/8), gk=j*BN+n, sgk=gk<S?gk:S-1, by=gk<S?16:0;
      cpa16(sa(Kb+dg*512+n*8), Kbh+(int64_t)sgk*D+dg*8, by);
      cpa16(sa(Vb+dg*512+n*8), Vbh+(int64_t)sgk*D+dg*8, by);
    }
  };

  // ---- prologue: Q + KV[0] ----
  for(int c=tid;c<(BM*D)/8;c+=THREADS){
    int m=c/(D/8), dg=c%(D/8), gm=q0+m, sgm=gm<S?gm:S-1, by=gm<S?16:0;
    cpa16(sa(Qs+dg*1024+m*8), Qbh+(int64_t)sgm*D+dg*8, by);
  }
  load_kv(0,0);
  cp_commit();

  float m_run=-INFINITY, l_run=0.f;
  uint32_t phqk=0, phpv=0;
  bool pv_pending=false;

  for(int t=0;t<ntiles;t++){
    int cur=t&1;
    __nv_bfloat16* Kb=Ks+cur*(BN*D);
    __nv_bfloat16* Vb=Vs+cur*(BN*D);

    cp_wait<0>();
    __syncthreads();
    fence_async();

    // ---- QK[t] ----
    if(tid==0){
      #pragma unroll
      for(int k=0;k<8;k++){
        uint64_t da=mkdesc(Qs+2*k*1024,2048,128);
        uint64_t db=mkdesc(Kb+2*k*512,1024,128);
        umma1(Saddr,da,db,idQK,k>0);
      }
      commit1(&bar[0]);
    }
    bar_wait(&bar[0], phqk); phqk^=1;

    // ---- read S[t], rowmax ----
    uint32_t s[64];
    ld16(Saddr+0 ,s   ); ld16(Saddr+16,s+16); ld16(Saddr+32,s+32); ld16(Saddr+48,s+48);
    wait_ld();
    float sv[64]; float lm=-INFINITY; int kv=t*BN;
    #pragma unroll
    for(int i=0;i<64;i++){
      int key=kv+i; float v=__int_as_float(s[i])*sl2;
      if(key>qg||key>=S) v=-INFINITY;
      sv[i]=v; lm=fmaxf(lm,v);
    }
    float m_old=m_run, m_new=fmaxf(m_old,lm), cor=ex2(m_old-m_new);

    // ---- wait PV[t-1] (frees V[(t-1)&1]) ----
    if(pv_pending){ bar_wait(&bar[1], phpv); phpv^=1; pv_pending=false; }

    // ---- prefetch KV[t+1] now that its buffer is free ----
    if(t+1<ntiles){ load_kv((t+1)&1, t+1); cp_commit(); }

    // ---- guarded O rescale ----
    if(t>0 && __any_sync(FULL, m_new>m_old)){
      #pragma unroll
      for(int d=0;d<D;d+=16){
        uint32_t o[16]; ld16(Oaddr+d,o); wait_ld();
        #pragma unroll
        for(int i=0;i<16;i++) o[i]=__float_as_int(__int_as_float(o[i])*cor);
        st16(Oaddr+d,o);
      }
      wait_st();
    }
    l_run*=cor;

    // ---- exp -> P, rowsum ----
    float rs=0.f;
    #pragma unroll
    for(int i=0;i<64;i++){
      float p=ex2(sv[i]-m_new); rs+=p;
      Ps[(i>>3)*1024 + tid*8 + (i&7)]=__float2bfloat16(p);
    }
    l_run+=rs; m_run=m_new;

    tf_before();
    __syncthreads();
    fence_async();
    if(tid==0){
      tf_after();
      #pragma unroll
      for(int k=0;k<4;k++){
        uint64_t da=mkdesc(Ps+2*k*1024,2048,128);
        uint64_t db=mkdesc(Vb+k*128,128,1024);
        umma1(Oaddr,da,db,idPV,(t==0&&k==0)?0:1);
      }
      commit1(&bar[1]);
    }
    pv_pending=true;
  }

  if(pv_pending){ bar_wait(&bar[1], phpv); phpv^=1; }

  // ---- epilogue ----
  float inv=1.0f/l_run;
  int64_t obase=(int64_t)bh*S*D+(int64_t)qg*D;
  #pragma unroll
  for(int d=0;d<D;d+=16){
    uint32_t o[16]; ld16(Oaddr+d,o); wait_ld();
    if(qg<S){
      #pragma unroll
      for(int i=0;i<16;i++) Og[obase+d+i]=__float2bfloat16(__int_as_float(o[i])*inv);
    }
  }
  if(qg<S) LSEg[(int64_t)bh*S+qg]=m_run*LN2+logf(l_run);

  __syncthreads();
  if(tid<32) tmem_dealloc1(tmem,256);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0),H=(int)Q.size(1),S=(int)Q.size(2),Dd=(int)Q.size(3);
  int BH=B*H;
  float scale=1.0f/sqrtf((float)Dd);
  float sl2=scale*1.4426950408889634f;

  dim3 grid((S+BM-1)/BM, BH, 1);
  dim3 block(THREADS);
  size_t shmem=(size_t)(BM*D + NB*BN*D + NB*BN*D + BM*BN)*sizeof(__nv_bfloat16);
  CUDA_CHECK(cudaFuncSetAttribute(attn,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shmem));

  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
  attn<<<grid,block,shmem,stream>>>(
     static_cast<const __nv_bfloat16*>(Q.data_ptr()),
     static_cast<const __nv_bfloat16*>(K.data_ptr()),
     static_cast<const __nv_bfloat16*>(V.data_ptr()),
     static_cast<__nv_bfloat16*>(O.data_ptr()),
     static_cast<float*>(LSE.data_ptr()), S, sl2);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_causal::run);

}  // namespace mha_causal