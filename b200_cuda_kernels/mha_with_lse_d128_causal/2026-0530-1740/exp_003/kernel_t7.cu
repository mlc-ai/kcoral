#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s @ %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);} } while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){const char* s; cuGetErrorString(_e,&s); fprintf(stderr,"CU %s @ %s:%d\n",s,__FILE__,__LINE__);} } while(0)

namespace flash_attn {

constexpr int BM=128, BN=128, HD=128;

__device__ __forceinline__ float ex2(float x){ float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y; }
__device__ __forceinline__ float exp2_poly(float x){
  x=fmaxf(x,-126.0f);
  float fn=floorf(x);
  int n=(int)fn;
  float f=x-fn;
  float p=0.0771f; p=p*f+0.2276f; p=p*f+0.6951f; p=p*f+1.0f;
  int bits=__float_as_int(p)+(n<<23);
  return __int_as_float(bits);
}
__device__ __forceinline__ void mbar_init(uint64_t* b,uint32_t c){ asm volatile("mbarrier.init.shared.b64 [%0],%1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c)); }
__device__ __forceinline__ void mbar_fence(){ asm volatile("fence.mbarrier_init.release.cluster;":::"memory"); }
__device__ __forceinline__ void mbar_expect(uint64_t* b,uint32_t tx){ asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(tx):"memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t* b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph));
}
__device__ __forceinline__ void tma_load(const CUtensorMap* d,uint64_t* bar,void* smem,int c0,int c1){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4}],[%2];"
    ::"r"((uint32_t)__cvta_generic_to_shared(smem)),"l"((uint64_t)d),"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c0),"r"(c1):"memory");
}
__device__ __forceinline__ void fence_async_shared(){ asm volatile("fence.proxy.async.shared::cta;":::"memory"); }
__device__ __forceinline__ uint64_t make_desc_sw(void* ptr,uint32_t sbo){
  uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(ptr);
  d |= (uint64_t)((a&0x3FFFF)>>4);
  d |= (uint64_t)((sbo&0x3FFFF)>>4)<<32;
  d |= (uint64_t)1<<46;
  d |= (uint64_t)2<<61;
  return d;
}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M,uint32_t N){
  uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10); d|=((N/8)<<17); d|=((M/16)<<24); return d;
}
__device__ __forceinline__ void umma(uint32_t tmem,uint64_t da,uint64_t db,uint32_t idesc,uint32_t accum){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\ntcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    ::"r"(tmem),"l"(da),"l"(db),"r"(idesc),"r"(accum));
}
__device__ __forceinline__ void umma_commit(uint64_t* bar){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"::"r"((uint32_t)__cvta_generic_to_shared(bar)));
}
__device__ __forceinline__ void tmem_ld4(uint32_t addr,uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];":"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(addr));
}
__device__ __forceinline__ void tmem_st4(uint32_t addr,uint32_t r0,uint32_t r1,uint32_t r2,uint32_t r3){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0],{%1,%2,%3,%4};"::"r"(addr),"r"(r0),"r"(r1),"r"(r2),"r"(r3));
}
__device__ __forceinline__ void tmem_wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void tmem_wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;":::"memory"); }
__device__ __forceinline__ void tcg_fence_before(){ asm volatile("tcgen05.fence::before_thread_sync;":::"memory"); }
__device__ __forceinline__ void tcg_fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }

__global__ __launch_bounds__(128) void fa_kernel(
    const __grid_constant__ CUtensorMap tmaQ,
    const __grid_constant__ CUtensorMap tmaK,
    const __grid_constant__ CUtensorMap tmaV,
    __nv_bfloat16* __restrict__ O, float* __restrict__ LSE,
    int H,int S){

  int qtile=blockIdx.x, h=blockIdx.y, b=blockIdx.z;
  int qstart=qtile*BM;
  if(qstart>=S) return;
  int tid=threadIdx.x, warp=tid>>5;
  int64_t headrow=(int64_t)(b*H+h)*S;

  extern __shared__ __align__(1024) char smem[];
  __nv_bfloat16* Qb[2]={(__nv_bfloat16*)(smem+0),(__nv_bfloat16*)(smem+16384)};
  __nv_bfloat16* Kb[2]={(__nv_bfloat16*)(smem+32768),(__nv_bfloat16*)(smem+49152)};
  __nv_bfloat16* sV =(__nv_bfloat16*)(smem+65536);
  __nv_bfloat16* Vtb[2]={(__nv_bfloat16*)(smem+98304),(__nv_bfloat16*)(smem+114688)};
  __nv_bfloat16* Pb[2]={(__nv_bfloat16*)(smem+131072),(__nv_bfloat16*)(smem+147456)};
  uint64_t* bar_q  =(uint64_t*)(smem+163840);
  uint64_t* bar_tma=(uint64_t*)(smem+163848);
  uint64_t* bar_mma=(uint64_t*)(smem+163856);
  uint32_t* tmem_base_smem=(uint32_t*)(smem+163864);

  if(tid==0){ mbar_init(bar_q,1); mbar_init(bar_tma,1); mbar_init(bar_mma,1); mbar_fence(); }
  __syncthreads();

  if(warp==0){
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;"
      ::"r"((uint32_t)__cvta_generic_to_shared(tmem_base_smem)),"r"(256));
  }
  __syncthreads();
  uint32_t base=*tmem_base_smem;
  uint32_t S_addr=base, O_addr=base+128;

  const float scale_log2=0.08838834764831843f*1.4426950408889634f;
  const uint32_t idesc=make_idesc(128,128);

  if(tid==0){
    mbar_expect(bar_q,32768);
    tma_load(&tmaQ,bar_q,Qb[0],0,(int)(headrow+qstart));
    tma_load(&tmaQ,bar_q,Qb[1],64,(int)(headrow+qstart));
  }
  mbar_wait(bar_q,0);

  int qmax=qstart+BM-1; if(qmax>S-1) qmax=S-1;
  int kt_last=qmax/BN;

  float m_run=-1e38f, l_run=0.f, corr=1.f;
  uint32_t ph_tma=0, ph_mma=0;
  int m=tid;

  for(int kt=0; kt<=kt_last; ++kt){
    int kbase=kt*BN;
    if(tid==0){
      mbar_expect(bar_tma,65536);
      tma_load(&tmaK,bar_tma,Kb[0],0,(int)(headrow+kbase));
      tma_load(&tmaK,bar_tma,Kb[1],64,(int)(headrow+kbase));
      tma_load(&tmaV,bar_tma,sV,0,(int)(headrow+kbase));
    }
    mbar_wait(bar_tma,ph_tma); ph_tma^=1;

    {
      int d=tid;
      #pragma unroll
      for(int blk=0;blk<2;++blk){
        __nv_bfloat16* Vd=Vtb[blk];
        #pragma unroll
        for(int ki=0;ki<8;++ki){
          int key_base=blk*64+ki*8;
          int phys=(d&7)^ki;
          __nv_bfloat16* dst=Vd+(d*8+phys)*8;
          #pragma unroll
          for(int j=0;j<8;++j) dst[j]=sV[(key_base+j)*128+d];
        }
      }
    }
    __syncthreads();

    if(tid==0){
      #pragma unroll
      for(int c=0;c<8;++c){
        int blk=c>>2, cib=c&3;
        uint64_t da=make_desc_sw(Qb[blk]+cib*16,1024);
        uint64_t db=make_desc_sw(Kb[blk]+cib*16,1024);
        umma(S_addr,da,db,idesc,c==0?0:1);
      }
      umma_commit(bar_mma);
    }
    mbar_wait(bar_mma,ph_mma); ph_mma^=1;

    float srow[128];
    #pragma unroll
    for(int c=0;c<128;c+=4){
      uint32_t r0,r1,r2,r3; tmem_ld4(S_addr+c,r0,r1,r2,r3);
      srow[c]=__uint_as_float(r0); srow[c+1]=__uint_as_float(r1);
      srow[c+2]=__uint_as_float(r2); srow[c+3]=__uint_as_float(r3);
    }
    tmem_wait_ld();

    int qpos=qstart+m;
    float m_local=-1e38f;
    #pragma unroll
    for(int k=0;k<128;++k){
      float v=srow[k]*scale_log2;
      int keypos=kbase+k;
      if(keypos>qpos || keypos>=S) v=-1e38f;
      srow[k]=v; m_local=fmaxf(m_local,v);
    }
    float m_new=fmaxf(m_run,m_local);
    corr=ex2(m_run-m_new);
    // split exp: even k -> hardware MUFU ex2, odd k -> FMA software poly
    float rowsum=0.f;
    #pragma unroll
    for(int k=0;k<128;k+=2){
      float p0=ex2(srow[k]-m_new);
      float p1=exp2_poly(srow[k+1]-m_new);
      srow[k]=p0; srow[k+1]=p1; rowsum+=p0+p1;
    }
    l_run=l_run*corr+rowsum;
    m_run=m_new;

    #pragma unroll
    for(int blk=0;blk<2;++blk){
      __nv_bfloat16* Pd=Pb[blk];
      #pragma unroll
      for(int ki=0;ki<8;++ki){
        int key_base=blk*64+ki*8;
        int phys=(m&7)^ki;
        __nv_bfloat16* dst=Pd+(m*8+phys)*8;
        #pragma unroll
        for(int j=0;j<8;++j) dst[j]=__float2bfloat16(srow[key_base+j]);
      }
    }
    fence_async_shared();
    __syncthreads();

    if(kt>0){
      #pragma unroll
      for(int c=0;c<128;c+=4){
        uint32_t r0,r1,r2,r3; tmem_ld4(O_addr+c,r0,r1,r2,r3);
        float f0=__uint_as_float(r0)*corr, f1=__uint_as_float(r1)*corr;
        float f2=__uint_as_float(r2)*corr, f3=__uint_as_float(r3)*corr;
        tmem_st4(O_addr+c,__float_as_uint(f0),__float_as_uint(f1),__float_as_uint(f2),__float_as_uint(f3));
      }
      tmem_wait_st();
      tcg_fence_before();
    }
    __syncthreads();

    if(tid==0){
      tcg_fence_after();
      #pragma unroll
      for(int c=0;c<8;++c){
        int blk=c>>2, cib=c&3;
        uint64_t da=make_desc_sw(Pb[blk]+cib*16,1024);
        uint64_t db=make_desc_sw(Vtb[blk]+cib*16,1024);
        umma(O_addr,da,db,idesc,(kt==0&&c==0)?0:1);
      }
      umma_commit(bar_mma);
    }
    mbar_wait(bar_mma,ph_mma); ph_mma^=1;
    __syncthreads();
  }

  int qpos=qstart+m;
  float inv=1.0f/l_run;
  float oreg[128];
  #pragma unroll
  for(int c=0;c<128;c+=4){
    uint32_t r0,r1,r2,r3; tmem_ld4(O_addr+c,r0,r1,r2,r3);
    oreg[c]=__uint_as_float(r0); oreg[c+1]=__uint_as_float(r1);
    oreg[c+2]=__uint_as_float(r2); oreg[c+3]=__uint_as_float(r3);
  }
  tmem_wait_ld();
  __syncthreads();

  if(warp==0){
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"::"r"(base),"r"(256));
  }

  if(qpos<S){
    __nv_bfloat16* op=O+(headrow+qpos)*HD;
    #pragma unroll
    for(int k=0;k<128;++k) op[k]=__float2bfloat16(oreg[k]*inv);
    LSE[headrow+qpos]=0.6931471805599453f*(m_run+log2f(l_run));
  }
}

static CUresult make_tma_sw(CUtensorMap* m,void* ptr,uint64_t inner,uint64_t outer,uint32_t box_inner){
  uint64_t gdim[2]={inner,outer};
  uint64_t gstr[1]={inner*2};
  uint32_t bdim[2]={box_inner,128};
  uint32_t estr[2]={1,1};
  return cuTensorMapEncodeTiled(m,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,ptr,gdim,gstr,bdim,estr,
    CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
static CUresult make_tma_none(CUtensorMap* m,void* ptr,uint64_t inner,uint64_t outer){
  uint64_t gdim[2]={inner,outer};
  uint64_t gstr[1]={inner*2};
  uint32_t bdim[2]={(uint32_t)inner,128};
  uint32_t estr[2]={1,1};
  return cuTensorMapEncodeTiled(m,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,ptr,gdim,gstr,bdim,estr,
    CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2);
  __nv_bfloat16* qp=static_cast<__nv_bfloat16*>(Q.data_ptr());
  __nv_bfloat16* kp=static_cast<__nv_bfloat16*>(K.data_ptr());
  __nv_bfloat16* vp=static_cast<__nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* op=static_cast<__nv_bfloat16*>(O.data_ptr());
  float* lp=static_cast<float*>(LSE.data_ptr());

  uint64_t outer=(uint64_t)B*H*S;
  CUtensorMap tQ,tK,tV;
  CU_CHECK(make_tma_sw(&tQ,qp,HD,outer,64));
  CU_CHECK(make_tma_sw(&tK,kp,HD,outer,64));
  CU_CHECK(make_tma_none(&tV,vp,HD,outer));

  int smem=163872;
  static bool set=false;
  if(!set){ cudaFuncSetAttribute(fa_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem); set=true; }

  int nq=(S+BM-1)/BM;
  dim3 grid(nq,H,B); dim3 block(128);
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
  fa_kernel<<<grid,block,smem,stream>>>(tQ,tK,tV,op,lp,H,S);
  CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_attn::run);

}  // namespace flash_attn