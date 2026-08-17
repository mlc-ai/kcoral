#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} }while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s; cuGetErrorString(_e,&s); \
  fprintf(stderr,"CU error %s at %s:%d\n",s,__FILE__,__LINE__); exit(1);} }while(0)

namespace mha_kernel {

constexpr int BM=128, BN=128, NT=128;
constexpr float SCALE=0.08838834764831845f;   // 1/sqrt(128)
constexpr float LOG2E=1.4426950408889634f;

// ---------- device helpers ----------
__device__ __forceinline__ void init_bar(uint64_t* b,uint32_t c){
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c));}
__device__ __forceinline__ void fence_bar_init(){asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory");}
__device__ __forceinline__ void bar_arrive_expect(uint64_t* b,uint32_t tx){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(tx):"memory");}
__device__ __forceinline__ void bar_wait(uint64_t* b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph));}
__device__ __forceinline__ void fence_proxy_async(){asm volatile("fence.proxy.async;\n":::"memory");}
__device__ __forceinline__ float ex2(float x){float y;asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
__device__ __forceinline__ uint32_t pack_bf16(uint32_t fa,uint32_t fb){
  __nv_bfloat16 a=__float2bfloat16(__uint_as_float(fa)); __nv_bfloat16 b=__float2bfloat16(__uint_as_float(fb));
  uint32_t r; asm("mov.b32 %0,{%1,%2};":"=r"(r):"h"(*(uint16_t*)&a),"h"(*(uint16_t*)&b)); return r;}

__device__ __forceinline__ void tma_load(const CUtensorMap* d,uint64_t* bar,void* smem,int c0,int c1){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4}],[%2];"
    ::"r"((uint32_t)__cvta_generic_to_shared(smem)),"l"((uint64_t)d),
      "r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c0),"r"(c1):"memory");}

__device__ __forceinline__ uint64_t smem_desc(void* p,uint32_t lbo,uint32_t sbo){
  uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
  d|=(uint64_t)(a&0x3FFFF)>>4;
  d|=(uint64_t)((lbo&0x3FFFF)>>4)<<16;
  d|=(uint64_t)((sbo&0x3FFFF)>>4)<<32;
  d|=(uint64_t)1<<46;
  d|=(uint64_t)2<<61;
  return d;}
__device__ __forceinline__ uint32_t idesc(uint32_t M,uint32_t N,uint32_t am,uint32_t bm){
  uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10); d|=(am<<15); d|=(bm<<16); d|=((N/8)<<17); d|=((M/16)<<24); return d;}

__device__ __forceinline__ void tmem_alloc(uint32_t* dst,int n){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(dst)),"r"(n));}
__device__ __forceinline__ void tmem_dealloc(uint32_t a,int n){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(a),"r"(n));}
__device__ __forceinline__ void umma(uint32_t tc,uint64_t da,uint64_t db,uint32_t id,uint32_t acc){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    ::"r"(tc),"l"(da),"l"(db),"r"(id),"r"(acc));}
__device__ __forceinline__ void umma_commit(uint64_t* b){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)));}
__device__ __forceinline__ void tc_fence_after(){asm volatile("tcgen05.fence::after_thread_sync;":::"memory");}
__device__ __forceinline__ void tc_fence_before(){asm volatile("tcgen05.fence::before_thread_sync;":::"memory");}
__device__ __forceinline__ void wait_ld(){asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");}
__device__ __forceinline__ void wait_st(){asm volatile("tcgen05.wait::st.sync.aligned;":::"memory");}
__device__ __forceinline__ void ld8(uint32_t c,uint32_t*r){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
    :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]):"r"(c));}
__device__ __forceinline__ void st8(uint32_t c,uint32_t*r){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0],{%1,%2,%3,%4,%5,%6,%7,%8};"
    ::"r"(c),"r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]):"memory");}

// ---------- kernel ----------
__global__ __launch_bounds__(128) void attn(
    const __grid_constant__ CUtensorMap tmaQ,
    const __grid_constant__ CUtensorMap tmaK,
    const __grid_constant__ CUtensorMap tmaV,
    __nv_bfloat16* __restrict__ O, float* __restrict__ LSE,
    int B,int H,int S,int num_qtiles){
  extern __shared__ __align__(1024) char smem[];
  __nv_bfloat16* Qs=(__nv_bfloat16*)(smem+0);
  __nv_bfloat16* Ks=(__nv_bfloat16*)(smem+32768);
  __nv_bfloat16* Vs=(__nv_bfloat16*)(smem+65536);
  __nv_bfloat16* Ps=(__nv_bfloat16*)(smem+98304);
  uint64_t* bars=(uint64_t*)(smem+131072);
  uint32_t* tmem_base_s=(uint32_t*)(smem+131072+64);

  int tid=threadIdx.x, warp=tid>>5;
  int qtile=blockIdx.x % num_qtiles;
  int bh=blockIdx.x / num_qtiles;
  int q0=qtile*BM;
  int row=tid;
  int gq=q0+row;
  long rowbase=(long)bh*S;

  if(tid==0){ for(int i=0;i<4;i++) init_bar(&bars[i],1); fence_bar_init(); }
  if(warp==0) tmem_alloc(tmem_base_s,256);
  __syncthreads();
  uint32_t tb=*tmem_base_s;

  uint32_t id_qk=idesc(128,128,0,0);
  uint32_t id_pv=idesc(128,64,0,1);

  // load Q
  if(tid==0){
    bar_arrive_expect(&bars[0],32768);
    for(int a=0;a<2;a++) tma_load(&tmaQ,&bars[0],(char*)Qs+a*16384,64*a,(int)(rowbase+q0));
  }
  bar_wait(&bars[0],0);

  int q_last=min(q0+BM-1,S-1);
  int num_kb=q_last/BN+1;

  float m_reg=-1e30f, l_reg=0.f;
  uint32_t ph_kv=0, ph_s=0, ph_o=0;

  for(int kb=0;kb<num_kb;kb++){
    int k0=kb*BN;
    // load K,V
    if(tid==0){
      bar_arrive_expect(&bars[1],65536);
      for(int a=0;a<2;a++) tma_load(&tmaK,&bars[1],(char*)Ks+a*16384,64*a,(int)(rowbase+k0));
      for(int a=0;a<2;a++) tma_load(&tmaV,&bars[1],(char*)Vs+a*16384,64*a,(int)(rowbase+k0));
    }
    bar_wait(&bars[1],ph_kv&1); ph_kv++;
    // QK^T
    if(tid==0){
      for(int k=0;k<8;k++){int a=k>>2,kl=k&3;
        uint64_t da=smem_desc((char*)Qs+a*16384+kl*32,1,1024);
        uint64_t db=smem_desc((char*)Ks+a*16384+kl*32,1,1024);
        umma(tb,da,db,id_qk,k==0?0:1);}
      umma_commit(&bars[2]);
    }
    bar_wait(&bars[2],ph_s&1); ph_s++;
    tc_fence_after();

    // ---- softmax ----
    // pass1: rowmax
    float rmax=-1e30f;
    #pragma unroll
    for(int col=0;col<128;col+=8){
      uint32_t r[8]; ld8(tb+col,r); wait_ld();
      #pragma unroll
      for(int j=0;j<8;j++){
        float v=__uint_as_float(r[j])*SCALE;
        int gk=k0+col+j;
        if(gk>gq||gk>=S) v=-1e30f;
        rmax=fmaxf(rmax,v);
      }
    }
    float m_old=m_reg;
    float m_new=fmaxf(m_old,rmax);
    float corr=ex2((m_old-m_new)*LOG2E);
    // pass2: exp -> P, rowsum
    float rsum=0.f;
    #pragma unroll
    for(int col=0;col<128;col+=8){
      uint32_t r[8]; ld8(tb+col,r); wait_ld();
      float p[8];
      #pragma unroll
      for(int j=0;j<8;j++){
        float v=__uint_as_float(r[j])*SCALE;
        int gk=k0+col+j;
        if(gk>gq||gk>=S) v=-1e30f;
        float e=ex2((v-m_new)*LOG2E);
        p[j]=e; rsum+=e;
      }
      int cc=col>>3, a=cc>>3, c=cc&7;
      uint32_t w0=pack_bf16(__float_as_uint(p[0]),__float_as_uint(p[1]));
      uint32_t w1=pack_bf16(__float_as_uint(p[2]),__float_as_uint(p[3]));
      uint32_t w2=pack_bf16(__float_as_uint(p[4]),__float_as_uint(p[5]));
      uint32_t w3=pack_bf16(__float_as_uint(p[6]),__float_as_uint(p[7]));
      uint4* dst=(uint4*)&Ps[a*8192 + row*64 + (c^(row&7))*8];
      *dst=make_uint4(w0,w1,w2,w3);
    }
    l_reg=l_reg*corr+rsum;
    m_reg=m_new;
    // O rescale (kb>0)
    if(kb>0){
      #pragma unroll
      for(int col=0;col<128;col+=8){
        uint32_t r[8]; ld8(tb+128+col,r); wait_ld();
        #pragma unroll
        for(int j=0;j<8;j++) r[j]=__float_as_uint(__uint_as_float(r[j])*corr);
        st8(tb+128+col,r);
      }
      wait_st();
    }
    // fences before PV
    fence_proxy_async();
    if(kb>0) tc_fence_before();
    __syncthreads();
    // PV
    if(tid==0){
      if(kb>0) tc_fence_after();
      for(int nc=0;nc<2;nc++)
        for(int k=0;k<8;k++){int a=k>>2,kl=k&3;
          uint64_t da=smem_desc((char*)Ps+a*16384+kl*32,1,1024);
          uint64_t db=smem_desc((char*)Vs+nc*16384+k*2048,16384,1024);
          uint32_t acc=(kb==0&&k==0)?0:1;
          umma(tb+128+nc*64,da,db,id_pv,acc);}
      umma_commit(&bars[3]);
    }
    bar_wait(&bars[3],ph_o&1); ph_o++;
  }

  // ---- epilogue ----
  tc_fence_after();
  float inv=1.0f/l_reg;
  #pragma unroll
  for(int col=0;col<128;col+=8){
    uint32_t r[8]; ld8(tb+128+col,r); wait_ld();
    if(gq<S){
      uint32_t w0=pack_bf16(__float_as_uint(__uint_as_float(r[0])*inv),__float_as_uint(__uint_as_float(r[1])*inv));
      uint32_t w1=pack_bf16(__float_as_uint(__uint_as_float(r[2])*inv),__float_as_uint(__uint_as_float(r[3])*inv));
      uint32_t w2=pack_bf16(__float_as_uint(__uint_as_float(r[4])*inv),__float_as_uint(__uint_as_float(r[5])*inv));
      uint32_t w3=pack_bf16(__float_as_uint(__uint_as_float(r[6])*inv),__float_as_uint(__uint_as_float(r[7])*inv));
      *(uint4*)&O[(rowbase+gq)*128 + col]=make_uint4(w0,w1,w2,w3);
    }
  }
  if(gq<S) LSE[rowbase+gq]=m_reg+logf(l_reg);

  __syncthreads();
  if(warp==0) tmem_dealloc(tb,256);
}

static CUresult make_tma(CUtensorMap* d,void* ptr,uint64_t outer){
  cuuint64_t gdim[2]={(cuuint64_t)128,(cuuint64_t)outer};
  cuuint64_t gstr[1]={(cuuint64_t)128*2};
  cuuint32_t bdim[2]={64,128};
  cuuint32_t estr[2]={1,1};
  return cuTensorMapEncodeTiled(d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,ptr,gdim,gstr,bdim,estr,
    CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q,tvm::ffi::TensorView K,tvm::ffi::TensorView V,
         tvm::ffi::TensorView O,tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2);
  __nv_bfloat16* Qp=(__nv_bfloat16*)Q.data_ptr();
  __nv_bfloat16* Kp=(__nv_bfloat16*)K.data_ptr();
  __nv_bfloat16* Vp=(__nv_bfloat16*)V.data_ptr();
  __nv_bfloat16* Op=(__nv_bfloat16*)O.data_ptr();
  float* Lp=(float*)LSE.data_ptr();

  uint64_t outer=(uint64_t)B*H*S;
  CUtensorMap dQ,dK,dV;
  CU_CHECK(make_tma(&dQ,Qp,outer));
  CU_CHECK(make_tma(&dK,Kp,outer));
  CU_CHECK(make_tma(&dV,Vp,outer));

  int num_qtiles=(S+BM-1)/BM;
  dim3 grid(B*H*num_qtiles), block(NT);
  size_t smem=131072+2048;

  static bool set=false;
  if(!set){ CUDA_CHECK(cudaFuncSetAttribute(attn,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem)); set=true; }

  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id);
  attn<<<grid,block,smem,stream>>>(dQ,dK,dV,Op,Lp,B,H,S,num_qtiles);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel