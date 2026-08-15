#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} }while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ \
  const char* s; cuGetErrorString(_e,&s); fprintf(stderr,"CU error %s at %s:%d\n",s,__FILE__,__LINE__); exit(1);} }while(0)

namespace mha_tc {

constexpr int BM=128, BN=128, D=128;
constexpr int TILE=16384;      // elems per KxD tile (2 blocks * 128*64)
constexpr int BLK=8192;        // elems per 64-wide block
#define SCALE (0.08838834764831843f*1.4426950408889634f)
#define LN2   0.6931471805599453f

__device__ __forceinline__ float ex2(float x){ float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y; }

__device__ __forceinline__ void init_bar(uint64_t* b,uint32_t c){
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c)); }
__device__ __forceinline__ void fence_bar_init(){ asm volatile("fence.mbarrier_init.release.cluster;":::"memory"); }
__device__ __forceinline__ void bar_wait(uint64_t* b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph)); }
__device__ __forceinline__ void mbar_arrive_expect(uint64_t* b,uint32_t tx){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(tx):"memory"); }
__device__ __forceinline__ void fence_async(){ asm volatile("fence.proxy.async;":::"memory"); }

__device__ __forceinline__ void tma_load(const CUtensorMap* d,uint64_t* bar,void* smem,int c0,int c1){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4}],[%2];"
    ::"r"((uint32_t)__cvta_generic_to_shared(smem)),"l"((uint64_t)d),
      "r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c0),"r"(c1):"memory"); }

__device__ __forceinline__ void tmem_alloc(uint32_t* d,int n){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(d)),"r"(n)); }
__device__ __forceinline__ void tmem_dealloc(uint32_t a,int n){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(a),"r"(n)); }
__device__ __forceinline__ void tmem_ld32(uint32_t t,uint32_t* r){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
    :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]):"r"(t)); }
__device__ __forceinline__ void tmem_st32(uint32_t t,const uint32_t* r){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
    ::"r"(t),"r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]):"memory"); }
__device__ __forceinline__ void ld_wait(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void st_wait(){ asm volatile("tcgen05.wait::st.sync.aligned;":::"memory"); }
__device__ __forceinline__ void fence_before(){ asm volatile("tcgen05.fence::before_thread_sync;":::"memory"); }
__device__ __forceinline__ void fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }

__device__ __forceinline__ void umma(uint32_t td,uint64_t da,uint64_t db,uint32_t id,uint32_t acc){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\ntcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    ::"r"(td),"l"(da),"l"(db),"r"(id),"r"(acc)); }
__device__ __forceinline__ void umma_commit(uint64_t* b){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)):"memory"); }

__device__ __forceinline__ uint64_t smem_desc(void* p,uint32_t lbo,uint32_t sbo){
  uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
  d |= (uint64_t)(a & 0x3FFFF) >> 4;
  d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
  d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)2 << 61;
  return d;
}
__device__ __forceinline__ uint32_t instr_desc(uint32_t am,uint32_t bm){
  uint32_t d=0;
  d |= (1u<<4); d |= (1u<<7); d |= (1u<<10);
  d |= (am<<15); d |= (bm<<16);
  d |= ((uint32_t)(BN>>3)<<17);
  d |= ((uint32_t)(BM>>4)<<24);
  return d;
}

__global__ __launch_bounds__(128) void attn(
    const __grid_constant__ CUtensorMap dQ,
    const __grid_constant__ CUtensorMap dK,
    const __grid_constant__ CUtensorMap dV,
    __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int S){
  extern __shared__ char smem_raw[];
  uintptr_t ap=((uintptr_t)smem_raw + 1023)&~(uintptr_t)1023;
  char* smem=(char*)ap;
  __nv_bfloat16* Qsh =(__nv_bfloat16*)smem;
  __nv_bfloat16* Ksh0=Qsh +TILE;
  __nv_bfloat16* Ksh1=Ksh0+TILE;
  __nv_bfloat16* Vsh0=Ksh1+TILE;
  __nv_bfloat16* Vsh1=Vsh0+TILE;
  __nv_bfloat16* Psh =Vsh1+TILE;
  uint64_t* barS=(uint64_t*)(smem + 6*TILE*2);
  uint64_t* bar_q=&barS[0], *bar_load0=&barS[1], *bar_load1=&barS[2], *bar_qk=&barS[3], *bar_pv=&barS[4];
  uint32_t* tmem_ptr=(uint32_t*)&barS[5];

  int tid=threadIdx.x; int warp=tid>>5;
  int q_start=blockIdx.x*BM;
  int bh=blockIdx.y;
  __nv_bfloat16* Og=O+(int64_t)bh*S*D;
  float* LSEg=LSE+(int64_t)bh*S;
  int row_base=bh*S;
  uint32_t lane_hi=(uint32_t)warp<<21;

  if(warp==0) tmem_alloc(tmem_ptr,256);
  if(tid==0){ init_bar(bar_q,1); init_bar(bar_load0,1); init_bar(bar_load1,1);
              init_bar(bar_qk,1); init_bar(bar_pv,1); }
  fence_bar_init();
  __syncthreads();
  uint32_t tmem_base=*tmem_ptr;
  uint32_t tmem_S=tmem_base;
  uint32_t tmem_O=tmem_base+128;

  uint32_t idesc_qk=instr_desc(0,0);
  uint32_t idesc_pv=instr_desc(0,1);

  int num_kv=(S+BN-1)/BN;

  // prologue: load Q and K/V tile 0
  if(tid==0){
    mbar_arrive_expect(bar_q,32768);
    tma_load(&dQ,bar_q,Qsh+0,   0, row_base+q_start);
    tma_load(&dQ,bar_q,Qsh+BLK, 64,row_base+q_start);
    mbar_arrive_expect(bar_load0,65536);
    tma_load(&dK,bar_load0,Ksh0+0,   0, row_base+0);
    tma_load(&dK,bar_load0,Ksh0+BLK, 64,row_base+0);
    tma_load(&dV,bar_load0,Vsh0+0,   0, row_base+0);
    tma_load(&dV,bar_load0,Vsh0+BLK, 64,row_base+0);
  }
  bar_wait(bar_q,0);

  uint32_t ph_load[2]={0,0}, ph_qk=0, ph_pv=0;
  float m=-1e30f, l=0.f;

  for(int kv=0; kv<num_kv; kv++){
    int cur=kv&1; int kv_start=kv*BN;
    __nv_bfloat16* Kc = cur? Ksh1:Ksh0;
    __nv_bfloat16* Vc = cur? Vsh1:Vsh0;
    uint64_t* blc = cur? bar_load1:bar_load0;

    bar_wait(blc, ph_load[cur]); ph_load[cur]^=1;

    // QK^T -> tmem_S
    if(tid==0){
      #pragma unroll
      for(int kk=0;kk<8;kk++){
        int block=kk>>2, sub=kk&3;
        uint64_t da=smem_desc(Qsh+block*BLK+sub*16,16,1024);
        uint64_t db=smem_desc(Kc +block*BLK+sub*16,16,1024);
        umma(tmem_S,da,db,idesc_qk,kk!=0);
      }
      umma_commit(bar_qk);
    }
    bar_wait(bar_qk,ph_qk); ph_qk^=1;

    // row max (each thread owns full row = tid)
    float rmax=-1e30f;
    #pragma unroll
    for(int c=0;c<128;c+=8){
      uint32_t r[8]; tmem_ld32(tmem_S+lane_hi+(uint32_t)c,r); ld_wait();
      #pragma unroll
      for(int j=0;j<8;j++){ if(kv_start+c+j<S){ float s=__uint_as_float(r[j])*SCALE; rmax=fmaxf(rmax,s);} }
    }
    float mnew=fmaxf(m,rmax);
    float corr=ex2(m-mnew);

    // rescale O in TMEM (needs prev PV done)
    if(kv>0){
      bar_wait(bar_pv,ph_pv); ph_pv^=1;
      #pragma unroll
      for(int c=0;c<128;c+=8){
        uint32_t r[8]; tmem_ld32(tmem_O+lane_hi+(uint32_t)c,r); ld_wait();
        #pragma unroll
        for(int j=0;j<8;j++){ r[j]=__float_as_uint(__uint_as_float(r[j])*corr); }
        tmem_st32(tmem_O+lane_hi+(uint32_t)c,r);
      }
      st_wait(); fence_before();
    }

    // prefetch next tile (buffer now guaranteed free)
    if(kv+1<num_kv && tid==0){
      int nb=(kv+1)&1;
      uint64_t* bl = nb? bar_load1:bar_load0;
      __nv_bfloat16* Kt = nb? Ksh1:Ksh0;
      __nv_bfloat16* Vt = nb? Vsh1:Vsh0;
      int c1=row_base+(kv+1)*BN;
      mbar_arrive_expect(bl,65536);
      tma_load(&dK,bl,Kt+0,   0, c1);
      tma_load(&dK,bl,Kt+BLK, 64,c1);
      tma_load(&dV,bl,Vt+0,   0, c1);
      tma_load(&dV,bl,Vt+BLK, 64,c1);
    }

    // P = exp2(S - mnew), sum, write to Psh (swizzled, K-major)
    float sum=0.f;
    #pragma unroll
    for(int c=0;c<128;c+=8){
      uint32_t r[8]; tmem_ld32(tmem_S+lane_hi+(uint32_t)c,r); ld_wait();
      union{uint4 u; __nv_bfloat16 h[8];} pk;
      #pragma unroll
      for(int j=0;j<8;j++){
        float p=0.f;
        if(kv_start+c+j<S){ p=ex2(__uint_as_float(r[j])*SCALE-mnew); sum+=p; }
        pk.h[j]=__float2bfloat16(p);
      }
      int block=c>>6; int col8=(c&63)>>3; int phys=(tid&7)^col8;
      int elem=block*BLK + tid*64 + phys*8;
      *reinterpret_cast<uint4*>(&Psh[elem])=pk.u;
    }
    l = l*corr + sum;
    m = mnew;

    fence_before();
    __syncthreads();
    if(tid==0){
      fence_after();
      fence_async();
      #pragma unroll
      for(int kk=0;kk<8;kk++){
        int block=kk>>2, sub=kk&3;
        uint64_t da=smem_desc(Psh+block*BLK+sub*16,16,1024);
        uint64_t db=smem_desc(Vc +kk*1024,16384,1024);
        umma(tmem_O,da,db,idesc_pv,kv>0);
      }
      umma_commit(bar_pv);
    }
  }

  bar_wait(bar_pv,ph_pv); ph_pv^=1;

  // epilogue
  int gr=q_start+tid;
  float inv = l>0.f? 1.f/l : 0.f;
  #pragma unroll
  for(int c=0;c<128;c+=8){
    uint32_t r[8]; tmem_ld32(tmem_O+lane_hi+(uint32_t)c,r); ld_wait();
    union{uint4 u; __nv_bfloat16 h[8];} pk;
    #pragma unroll
    for(int j=0;j<8;j++) pk.h[j]=__float2bfloat16(__uint_as_float(r[j])*inv);
    if(gr<S) *reinterpret_cast<uint4*>(&Og[(int64_t)gr*D + c])=pk.u;
  }
  if(gr<S) LSEg[gr]=m*LN2 + logf(l);

  __syncthreads();
  if(warp==0) tmem_dealloc(tmem_base,256);
}

static CUtensorMap make_desc(void* ptr, uint64_t rows){
  CUtensorMap d;
  uint64_t gdim[2]={ (uint64_t)D, rows };
  uint64_t gstride[1]={ (uint64_t)D*2 };
  uint32_t bdim[2]={ 64, 128 };
  uint32_t estride[2]={1,1};
  CU_CHECK(cuTensorMapEncodeTiled(&d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr,
      gdim, gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return d;
}

void run(tvm::ffi::TensorView Q,tvm::ffi::TensorView K,tvm::ffi::TensorView V,
         tvm::ffi::TensorView O,tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bsz=(int)Q.size(0), Hsz=(int)Q.size(1), S=(int)Q.size(2);
  void* Qp=const_cast<void*>(Q.data_ptr());
  void* Kp=const_cast<void*>(K.data_ptr());
  void* Vp=const_cast<void*>(V.data_ptr());
  __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
  float* Lp=static_cast<float*>(LSE.data_ptr());

  uint64_t rows=(uint64_t)Bsz*Hsz*S;
  CUtensorMap dQ=make_desc(Qp,rows);
  CUtensorMap dK=make_desc(Kp,rows);
  CUtensorMap dV=make_desc(Vp,rows);

  int nqt=(S+BM-1)/BM;
  dim3 grid(nqt,Bsz*Hsz);
  int smem = 6*TILE*2 + 5*8 + 8 + 1024;  // buffers + barriers + tmem_ptr + align slack

  static bool set=false;
  if(!set){ CUDA_CHECK(cudaFuncSetAttribute(attn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem)); set=true; }

  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
  attn<<<grid,128,smem,stream>>>(dQ,dK,dV,Op,Lp,S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_tc::run);

}  // namespace mha_tc