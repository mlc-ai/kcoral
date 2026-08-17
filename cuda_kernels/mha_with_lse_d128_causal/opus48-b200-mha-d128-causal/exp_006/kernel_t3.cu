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
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char*s; cuGetErrorString(_e,&s); \
  fprintf(stderr,"CU %s at %s:%d\n",s,__FILE__,__LINE__); exit(1);} }while(0)

namespace mha_causal {

constexpr int D=128, BM=128, BN=64;
constexpr int THREADS=128;

__device__ __forceinline__ float ex2(float x){ float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y; }

__device__ __forceinline__ void init_bar(uint64_t*b,int c){ asm volatile("mbarrier.init.shared.b64 [%0],%1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c)); }
__device__ __forceinline__ void fence_bar_init(){ asm volatile("fence.mbarrier_init.release.cluster;":::"memory"); }
__device__ __forceinline__ void arrive_expect(uint64_t*b,uint32_t tx){ asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(tx):"memory"); }
__device__ __forceinline__ void bar_wait(uint64_t*b,uint32_t ph){ asm volatile("{ .reg .pred P; W%=: mbarrier.try_wait.parity.shared.b64 P,[%0],%1; @!P bra W%=; }"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph)); }
__device__ __forceinline__ void fence_async(){ asm volatile("fence.proxy.async;":::"memory"); }

__device__ __forceinline__ void tma3d(const CUtensorMap*d,uint64_t*bar,void*smem,int c0,int c1,int c2){
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4,%5}],[%2];"
    ::"r"((uint32_t)__cvta_generic_to_shared(smem)),"l"((uint64_t)d),
      "r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c0),"r"(c1),"r"(c2):"memory");
}

__device__ __forceinline__ void tmem_alloc1(uint32_t*dst,int n){ asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;"::"r"((uint32_t)__cvta_generic_to_shared(dst)),"r"(n)); }
__device__ __forceinline__ void tmem_dealloc1(uint32_t a,int n){ asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"::"r"(a),"r"(n)); }
__device__ __forceinline__ void umma1(uint32_t d,uint64_t a,uint64_t b,uint32_t id,int acc){
  asm volatile("{ .reg .pred p; setp.ne.b32 p,%4,0; tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p; }"
    ::"r"(d),"l"(a),"l"(b),"r"(id),"r"(acc));
}
__device__ __forceinline__ void commit1(uint64_t*b){ asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"::"r"((uint32_t)__cvta_generic_to_shared(b))); }
__device__ __forceinline__ void wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;":::"memory"); }

__device__ __forceinline__ void ld16(uint32_t a,uint32_t*r){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},[%16];"
   :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
    "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]):"r"(a));
}
__device__ __forceinline__ void st16(uint32_t a,const uint32_t*r){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0],{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
   ::"r"(a),"r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),
     "r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),"r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]):"memory");
}

__device__ __forceinline__ uint64_t mkdesc(const __nv_bfloat16*p,uint32_t lbo,uint32_t sbo){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
  uint64_t d=0;
  d|=(uint64_t)((a&0x3FFFF)>>4);
  d|=(uint64_t)((lbo&0x3FFFF)>>4)<<16;
  d|=(uint64_t)((sbo&0x3FFFF)>>4)<<32;
  d|=(uint64_t)1<<46;
  return d;
}
__device__ __forceinline__ uint32_t mkidesc(uint32_t M,uint32_t N,uint32_t am,uint32_t bm){
  uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10); d|=(am<<15); d|=(bm<<16);
  d|=((N>>3)<<17); d|=((M>>4)<<24); return d;
}

__global__ __launch_bounds__(THREADS) void attn(
   const __grid_constant__ CUtensorMap dQ, const __grid_constant__ CUtensorMap dK,
   const __grid_constant__ CUtensorMap dV, __nv_bfloat16* __restrict__ O,
   float* __restrict__ LSE, int S, float sl2)
{
  extern __shared__ __align__(1024) __nv_bfloat16 smem[];
  __nv_bfloat16* Qs=smem;
  __nv_bfloat16* Ks=Qs+BM*D;
  __nv_bfloat16* Vs=Ks+BN*D;
  __nv_bfloat16* Ps=Vs+BN*D;
  __shared__ uint64_t bar[4];
  __shared__ uint32_t tmem_slot;

  int tid=threadIdx.x;
  int q0=blockIdx.x*BM;
  int bh=blockIdx.y;
  int qg=q0+tid;

  if(tid==0){ init_bar(&bar[0],1); init_bar(&bar[1],1); init_bar(&bar[2],1); init_bar(&bar[3],1); fence_bar_init(); }
  __syncthreads();
  if(tid<32) tmem_alloc1(&tmem_slot,256);
  __syncthreads();
  uint32_t tmem=tmem_slot;
  uint32_t Saddr=tmem, Oaddr=tmem+64;

  uint32_t idQK=mkidesc(128,64,0,0);
  uint32_t idPV=mkidesc(128,128,0,1);

  // load Q once
  if(tid==0){ arrive_expect(&bar[3], BM*D*2); tma3d(&dQ,&bar[3],Qs,0,q0,bh); }
  bar_wait(&bar[3],0);
  __syncthreads();

  float m_run=-INFINITY, l_run=0.f;
  int last_key=min(q0+BM-1,S-1);
  int ntiles=last_key/BN+1;

  for(int t=0;t<ntiles;t++){
    int kv=t*BN;
    if(tid==0){ arrive_expect(&bar[0], 2*BN*D*2); tma3d(&dK,&bar[0],Ks,0,kv,bh); tma3d(&dV,&bar[0],Vs,0,kv,bh); }
    bar_wait(&bar[0], t&1);
    __syncthreads();

    if(tid==0){
      #pragma unroll
      for(int k=0;k<8;k++){
        uint64_t da=mkdesc(Qs+k*16,16,2048);
        uint64_t db=mkdesc(Ks+k*16,16,2048);
        umma1(Saddr,da,db,idQK,k>0);
      }
      commit1(&bar[1]);
    }
    bar_wait(&bar[1], t&1);

    // pass1: rowmax
    float lm=-INFINITY;
    #pragma unroll
    for(int c=0;c<BN;c+=16){
      uint32_t r[16]; ld16(Saddr+c,r); wait_ld();
      #pragma unroll
      for(int i=0;i<16;i++){
        int key=kv+c+i; float v=__int_as_float(r[i])*sl2;
        if(key>qg||key>=S) v=-INFINITY;
        lm=fmaxf(lm,v);
      }
    }
    float m_new=fmaxf(m_run,lm);
    float cor=ex2(m_run-m_new);

    if(t>0){
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

    // pass2: P
    float rs=0.f;
    #pragma unroll
    for(int c=0;c<BN;c+=16){
      uint32_t r[16]; ld16(Saddr+c,r); wait_ld();
      #pragma unroll
      for(int i=0;i<16;i++){
        int key=kv+c+i; float v=__int_as_float(r[i])*sl2;
        if(key>qg||key>=S) v=-INFINITY;
        float p=ex2(v-m_new); rs+=p;
        Ps[tid*BN+c+i]=__float2bfloat16(p);
      }
    }
    l_run+=rs; m_run=m_new;

    __syncthreads();
    fence_async();

    if(tid==0){
      #pragma unroll
      for(int k=0;k<4;k++){
        uint64_t da=mkdesc(Ps+k*16,16,1024);
        uint64_t db=mkdesc(Vs+k*16*D,2048,16);
        umma1(Oaddr,da,db,idPV,(t==0&&k==0)?0:1);
      }
      commit1(&bar[2]);
    }
    bar_wait(&bar[2], t&1);
    __syncthreads();
  }

  // epilogue: always read O (warp-collective), guard stores
  float inv=1.0f/l_run;
  #pragma unroll
  for(int d=0;d<D;d+=16){
    uint32_t o[16]; ld16(Oaddr+d,o); wait_ld();
    if(qg<S){
      #pragma unroll
      for(int i=0;i<16;i++)
        O[(int64_t)bh*S*D+(int64_t)qg*D+d+i]=__float2bfloat16(__int_as_float(o[i])*inv);
    }
  }
  if(qg<S) LSE[(int64_t)bh*S+qg]=m_run*0.6931471805599453f+logf(l_run);

  __syncthreads();
  if(tid<32) tmem_dealloc1(tmem,256);
}

static CUresult make3d(CUtensorMap*d,void*ptr,uint64_t Dd,uint64_t Sd,uint64_t BHd,uint32_t boxSeq){
  uint64_t gd[3]={Dd,Sd,BHd};
  uint64_t gs[2]={Dd*2, Dd*Sd*2};
  uint32_t bd[3]={(uint32_t)Dd,boxSeq,1};
  uint32_t es[3]={1,1,1};
  return cuTensorMapEncodeTiled(d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,ptr,gd,gs,bd,es,
     CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
     CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0),H=(int)Q.size(1),S=(int)Q.size(2),Dd=(int)Q.size(3);
  int BH=B*H;
  float scale=1.0f/sqrtf((float)Dd);
  float sl2=scale*1.4426950408889634f;

  CUtensorMap dQ,dK,dV;
  CU_CHECK(make3d(&dQ,Q.data_ptr(),Dd,S,BH,BM));
  CU_CHECK(make3d(&dK,K.data_ptr(),Dd,S,BH,BN));
  CU_CHECK(make3d(&dV,V.data_ptr(),Dd,S,BH,BN));

  dim3 grid((S+BM-1)/BM, BH, 1);
  dim3 block(THREADS);
  size_t shmem=(size_t)(BM*D+BN*D+BN*D+BM*BN)*sizeof(__nv_bfloat16);
  CUDA_CHECK(cudaFuncSetAttribute(attn,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shmem));

  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
  attn<<<grid,block,shmem,stream>>>(dQ,dK,dV,
     static_cast<__nv_bfloat16*>(O.data_ptr()),
     static_cast<float*>(LSE.data_ptr()), S, sl2);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_causal::run);

}  // namespace mha_causal