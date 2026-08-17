#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using bf16 = __nv_bfloat16;

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__);} } while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ \
  const char* s; cuGetErrorString(_e,&s); fprintf(stderr,"CU error %s at %s:%d\n", s, __FILE__, __LINE__);} } while(0)

namespace mha_kernel {

constexpr int BM=128, BN=128, D=128, THREADS=128;
constexpr int TILE_BYTES=32768;
constexpr int O_BARL=0, O_BARM=8, O_TPS=16;
constexpr int O_PQ=1024;
constexpr int O_PK =O_PQ + 32768;
constexpr int O_PVT=O_PK + 32768;
constexpr int O_PP =O_PVT+ 32768;
constexpr int O_KS =O_PP + 32768;
constexpr int O_VS =O_KS + 32768;
constexpr int SHMEM=O_VS + 32768;

__device__ __forceinline__ void init_bar(uint64_t* b,uint32_t c){
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c));
}
__device__ __forceinline__ void fence_bar_init(){ asm volatile("fence.mbarrier_init.release.cluster;":::"memory"); }
__device__ __forceinline__ void arrive_tx(uint64_t* b,uint32_t tx){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(tx):"memory");
}
__device__ __forceinline__ void bar_wait(uint64_t* b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph));
}
__device__ __forceinline__ void tma_load(const CUtensorMap* d,uint64_t* bar,void* smem,int c0,int c1){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3,%4}], [%2];"
    ::"r"((uint32_t)__cvta_generic_to_shared(smem)),"l"((uint64_t)d),
      "r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c0),"r"(c1):"memory");
}
__device__ __forceinline__ void fence_pa(){ asm volatile("fence.proxy.async;":::"memory"); }
__device__ __forceinline__ void tmem_alloc1(uint32_t* dst,int n){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(dst)),"r"(n));
}
__device__ __forceinline__ void tmem_dealloc1(uint32_t a,int n){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(a),"r"(n));
}
__device__ __forceinline__ void tmem_relinquish1(){ asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"); }
__device__ __forceinline__ void umma1(uint32_t td,uint64_t da,uint64_t db,uint32_t id,uint32_t acc){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
   "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
   ::"r"(td),"l"(da),"l"(db),"r"(id),"r"(acc));
}
__device__ __forceinline__ void umma_commit1(uint64_t* bar){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
    ::"r"((uint32_t)__cvta_generic_to_shared(bar)));
}
__device__ __forceinline__ void wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;":::"memory"); }
__device__ __forceinline__ void fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }
__device__ __forceinline__ void fence_before(){ asm volatile("tcgen05.fence::before_thread_sync;":::"memory"); }

__device__ __forceinline__ void ld32(uint32_t a,uint32_t* r){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {"
  "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
  "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
  :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
   "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
   "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
   "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
   :"r"(a));
}
__device__ __forceinline__ void st32(uint32_t a,uint32_t* r){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {"
  "%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
  "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};"
  ::"r"(a),"r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),
   "r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),"r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]),
   "r"(r[16]),"r"(r[17]),"r"(r[18]),"r"(r[19]),"r"(r[20]),"r"(r[21]),"r"(r[22]),"r"(r[23]),
   "r"(r[24]),"r"(r[25]),"r"(r[26]),"r"(r[27]),"r"(r[28]),"r"(r[29]),"r"(r[30]),"r"(r[31]));
}
__device__ __forceinline__ uint64_t desc(const void* p){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
  uint64_t d=(uint64_t)((a&0x3FFFFu)>>4);
  d|=((uint64_t)((2048u&0x3FFFFu)>>4))<<16;  // LBO=2048
  d|=((uint64_t)((128u &0x3FFFFu)>>4))<<32;  // SBO=128
  d|=((uint64_t)1)<<46;                       // version, no swizzle
  return d;
}

__global__ __launch_bounds__(128) void attn(
    const __grid_constant__ CUtensorMap tmaQ,
    const __grid_constant__ CUtensorMap tmaK,
    const __grid_constant__ CUtensorMap tmaV,
    bf16* __restrict__ O, float* __restrict__ LSE, int B,int H,int S){

  extern __shared__ char smem[];
  uint64_t* bar_load=reinterpret_cast<uint64_t*>(smem+O_BARL);
  uint64_t* bar_mma =reinterpret_cast<uint64_t*>(smem+O_BARM);
  uint32_t* tmem_ps =reinterpret_cast<uint32_t*>(smem+O_TPS);
  bf16* PQ =reinterpret_cast<bf16*>(smem+O_PQ);
  bf16* PK =reinterpret_cast<bf16*>(smem+O_PK);
  bf16* PVt=reinterpret_cast<bf16*>(smem+O_PVT);
  bf16* PP =reinterpret_cast<bf16*>(smem+O_PP);
  bf16* KS =reinterpret_cast<bf16*>(smem+O_KS);
  bf16* VS =reinterpret_cast<bf16*>(smem+O_VS);

  int tid=threadIdx.x;
  int warp=tid>>5;
  uint32_t lane_off=(uint32_t)(warp*32)<<16;
  int qb=blockIdx.x, h=blockIdx.y, b=blockIdx.z;
  int qs=qb*BM;
  long bh=(long)(b*H+h);
  long row_base=bh*(long)S;
  bf16* Obase=O+row_base*D;
  float* LSEbase=LSE+row_base;

  const float scale=0.08838834764831843f;
  const uint32_t idesc=(1u<<4)|(1u<<7)|(1u<<10)|((BN/8)<<17)|((BM/16)<<24);

  if(tid==0){ init_bar(bar_load,1); init_bar(bar_mma,1); }
  if(warp==0) tmem_alloc1(tmem_ps,256);
  fence_bar_init();
  __syncthreads();
  uint32_t tb=*tmem_ps;
  if(warp==0) tmem_relinquish1();

  uint32_t ph_load=0, ph_mma=0;

  // ---- load Q into KS(staging), repack to PQ ----
  if(tid==0){ arrive_tx(bar_load,TILE_BYTES); tma_load(&tmaQ,bar_load,KS,0,(int)(row_base+qs)); }
  bar_wait(bar_load,ph_load); ph_load^=1;
  __syncthreads();
  #pragma unroll
  for(int i=tid;i<2048;i+=THREADS){
    int m=i>>4, dc=i&15;
    reinterpret_cast<int4*>(PQ)[dc*128+m]=reinterpret_cast<int4*>(KS)[m*16+dc];
  }
  __syncthreads();
  if(tid==0) fence_pa();

  int gq=qs+tid;
  int q_max=(qs+BM-1<S-1)?(qs+BM-1):(S-1);
  int kvb_max=q_max/BN;

  float m_i=-1e30f, l_i=0.0f, corr=0.0f;

  for(int kvb=0; kvb<=kvb_max; ++kvb){
    int kv_start=kvb*BN;
    if(tid==0){
      arrive_tx(bar_load,2*TILE_BYTES);
      tma_load(&tmaK,bar_load,KS,0,(int)(row_base+kv_start));
      tma_load(&tmaV,bar_load,VS,0,(int)(row_base+kv_start));
    }
    bar_wait(bar_load,ph_load); ph_load^=1;
    __syncthreads();

    // repack K -> PK (vectorized)
    #pragma unroll
    for(int i=tid;i<2048;i+=THREADS){
      int n=i>>4, dc=i&15;
      reinterpret_cast<int4*>(PK)[dc*128+n]=reinterpret_cast<int4*>(KS)[n*16+dc];
    }
    // transpose-pack V -> PVt
    #pragma unroll 4
    for(int idx=tid;idx<BN*D;idx+=THREADS){
      int kv=idx>>7, d=idx&127;
      PVt[(kv>>3)*1024 + d*8 + (kv&7)]=VS[idx];
    }
    __syncthreads();
    if(tid==0) fence_pa();

    // ---- QK^T -> S (TMEM cols 0..127) ----
    if(tid==0){
      #pragma unroll
      for(int s=0;s<8;s++)
        umma1(tb, desc(PQ+2*s*1024), desc(PK+2*s*1024), idesc, s>0?1:0);
      umma_commit1(bar_mma);
    }
    bar_wait(bar_mma,ph_mma); ph_mma^=1;
    fence_after();

    // ---- online softmax ----
    float rowmax=-1e30f;
    #pragma unroll
    for(int c=0;c<4;c++){
      uint32_t r[32]; ld32(tb+lane_off+c*32,r); wait_ld();
      #pragma unroll
      for(int j=0;j<32;j++){
        int gk=kv_start+c*32+j;
        float s=(gk<=gq)? __uint_as_float(r[j])*scale : -1e30f;
        rowmax=fmaxf(rowmax,s);
      }
    }
    float m_new=fmaxf(m_i,rowmax);
    corr=__expf(m_i-m_new);
    float sum=0.0f;
    #pragma unroll
    for(int c=0;c<4;c++){
      uint32_t r[32]; ld32(tb+lane_off+c*32,r); wait_ld();
      #pragma unroll
      for(int j=0;j<32;j++){
        int col=c*32+j;
        int gk=kv_start+col;
        float s=(gk<=gq)? __uint_as_float(r[j])*scale : -1e30f;
        float p=__expf(s-m_new);
        sum+=p;
        PP[(col>>3)*1024 + tid*8 + (col&7)]=__float2bfloat16(p);
      }
    }
    l_i=l_i*corr+sum;
    m_i=m_new;
    __syncthreads();

    // ---- rescale O accumulator (TMEM cols 128..255) ----
    if(kvb>0){
      #pragma unroll
      for(int c=0;c<4;c++){
        uint32_t r[32]; ld32(tb+lane_off+128+c*32,r); wait_ld();
        #pragma unroll
        for(int j=0;j<32;j++) r[j]=__float_as_uint(__uint_as_float(r[j])*corr);
        st32(tb+lane_off+128+c*32,r);
      }
      wait_st();
      fence_before();
    }
    __syncthreads();

    // ---- P @ V -> O ----
    if(tid==0){
      fence_after(); fence_pa();
      #pragma unroll
      for(int s=0;s<8;s++)
        umma1(tb+128, desc(PP+2*s*1024), desc(PVt+2*s*1024), idesc, (kvb>0||s>0)?1:0);
      umma_commit1(bar_mma);
    }
    bar_wait(bar_mma,ph_mma); ph_mma^=1;
    fence_after();
    __syncthreads();
  }

  // ---- epilogue ----
  float inv=(gq<S)?(1.0f/l_i):0.0f;
  #pragma unroll
  for(int c=0;c<4;c++){
    uint32_t r[32]; ld32(tb+lane_off+128+c*32,r); wait_ld();
    if(gq<S){
      #pragma unroll
      for(int j=0;j<32;j++)
        Obase[(long)gq*D + c*32 + j]=__float2bfloat16(__uint_as_float(r[j])*inv);
    }
  }
  if(gq<S) LSEbase[gq]=m_i+logf(l_i);

  __syncthreads();
  if(warp==0) tmem_dealloc1(tb,256);
}

static CUresult make_tma(CUtensorMap* m,void* p,uint64_t outer){
  uint64_t gdim[2]={(uint64_t)D,outer};
  uint64_t gstr[1]={(uint64_t)D*2};
  uint32_t bdim[2]={(uint32_t)D,128};
  uint32_t estr[2]={1,1};
  return cuTensorMapEncodeTiled(m,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,p,gdim,gstr,bdim,estr,
    CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bv=Q.size(0), Hv=Q.size(1), Sv=Q.size(2);
  uint64_t outer=(uint64_t)Bv*Hv*Sv;

  void* Qp=Q.data_ptr(); void* Kp=K.data_ptr(); void* Vp=V.data_ptr();
  bf16* Op=static_cast<bf16*>(O.data_ptr());
  float* Lp=static_cast<float*>(LSE.data_ptr());

  CUtensorMap tmaQ,tmaK,tmaV;
  CU_CHECK(make_tma(&tmaQ,Qp,outer));
  CU_CHECK(make_tma(&tmaK,Kp,outer));
  CU_CHECK(make_tma(&tmaV,Vp,outer));

  int numQB=(Sv+BM-1)/BM;
  dim3 grid(numQB,Hv,Bv);
  dim3 block(THREADS);

  cudaStream_t stream=static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  CUDA_CHECK(cudaFuncSetAttribute(attn,
      cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM));
  attn<<<grid, block, SHMEM, stream>>>(tmaQ,tmaK,tmaV,Op,Lp,Bv,Hv,Sv);
  CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel