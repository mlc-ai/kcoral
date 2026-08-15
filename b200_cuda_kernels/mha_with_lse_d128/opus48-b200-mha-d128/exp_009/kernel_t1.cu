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

namespace mha_tc {

constexpr int BM=128, BN=128, D=128;
#define SCALE (0.08838834764831843f*1.4426950408889634f)
#define LN2   0.6931471805599453f

__device__ __forceinline__ float ex2(float x){ float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y; }

__device__ __forceinline__ void init_bar(uint64_t* b,uint32_t c){
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c)); }
__device__ __forceinline__ void fence_bar_init(){ asm volatile("fence.mbarrier_init.release.cluster;":::"memory"); }
__device__ __forceinline__ void bar_wait(uint64_t* b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph)); }
__device__ __forceinline__ void fence_async(){ asm volatile("fence.proxy.async;":::"memory"); }

__device__ __forceinline__ void tmem_alloc(uint32_t* d,int n){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(d)),"r"(n)); }
__device__ __forceinline__ void tmem_dealloc(uint32_t a,int n){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(a),"r"(n)); }
__device__ __forceinline__ void tmem_ld32(uint32_t t,uint32_t* r){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
    :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]):"r"(t)); }
__device__ __forceinline__ void ld_wait(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }

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
  d |= ((uint32_t)(BN>>3)<<17);   // N=128 -> but set per-use below
  d |= ((uint32_t)(BM>>4)<<24);
  return d;
}

__device__ __forceinline__ void load_tile(__nv_bfloat16* dst,const __nv_bfloat16* g,int row_start,int S,int tid){
  for(int i=tid;i<128*16;i+=128){
    int row=i>>4; int d8=i&15; int d=d8<<3;
    int block=d>>6; int col8=(d&63)>>3;
    int gs=row_start+row;
    uint4 v;
    if(gs<S) v=*reinterpret_cast<const uint4*>(&g[(int64_t)gs*D + d]);
    else v=make_uint4(0,0,0,0);
    int phys=(row&7)^col8;
    int elem = block*(128*64) + row*64 + phys*8;
    *reinterpret_cast<uint4*>(&dst[elem]) = v;
  }
}

__global__ __launch_bounds__(128) void attn(
    const __nv_bfloat16* __restrict__ Q,const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,__nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,int S){
  extern __shared__ char smem[];
  __nv_bfloat16* Qsh=reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Ksh=Qsh+16384;
  __nv_bfloat16* Vsh=Ksh+16384;
  __nv_bfloat16* Psh=Vsh+16384;
  uint64_t* bar_qk=reinterpret_cast<uint64_t*>(Psh+16384);
  uint64_t* bar_pv=bar_qk+1;
  uint32_t* tmem_ptr=reinterpret_cast<uint32_t*>(bar_pv+1);

  int tid=threadIdx.x; int warp=tid>>5;
  int q_start=blockIdx.x*BM;
  int bh=blockIdx.y;
  int64_t base=(int64_t)bh*S*D;
  const __nv_bfloat16* Qg=Q+base; const __nv_bfloat16* Kg=K+base;
  const __nv_bfloat16* Vg=V+base; __nv_bfloat16* Og=O+base;
  float* LSEg=LSE+(int64_t)bh*S;

  if(warp==0) tmem_alloc(tmem_ptr,256);
  if(tid==0){ init_bar(bar_qk,1); init_bar(bar_pv,1); }
  fence_bar_init();
  __syncthreads();
  uint32_t tmem_base=*tmem_ptr;
  uint32_t tmem_S=tmem_base;
  uint32_t tmem_O=tmem_base+128;
  uint32_t lane_hi=((uint32_t)(warp*32))<<16;

  uint32_t idesc_qk=instr_desc(0,0);
  uint32_t idesc_pv=instr_desc(0,1);

  load_tile(Qsh,Qg,q_start,S,tid);
  __syncthreads();

  int num_kv=(S+BN-1)/BN;
  uint32_t ph_qk=0, ph_pv=0;

  float m=-1e30f, l=0.f;

  // ---------- PASS A: statistics (m,l) ----------
  for(int kv=0;kv<num_kv;kv++){
    int kv_start=kv*BN;
    __syncthreads();
    load_tile(Ksh,Kg,kv_start,S,tid);
    __syncthreads();
    fence_async();
    if(tid==0){
      #pragma unroll
      for(int kk=0;kk<8;kk++){
        int block=kk>>2, sub=kk&3;
        uint64_t da=smem_desc(Qsh+block*(128*64)+sub*16,16,1024);
        uint64_t db=smem_desc(Ksh+block*(128*64)+sub*16,16,1024);
        umma(tmem_S,da,db,idesc_qk,kk!=0);
      }
      umma_commit(bar_qk);
    }
    bar_wait(bar_qk,ph_qk); ph_qk^=1;

    float rmax=-1e30f;
    for(int c=0;c<128;c+=8){
      uint32_t r[8]; tmem_ld32(tmem_S+lane_hi+c,r); ld_wait();
      #pragma unroll
      for(int j=0;j<8;j++){ if(kv_start+c+j<S){ float s=__uint_as_float(r[j])*SCALE; rmax=fmaxf(rmax,s);} }
    }
    float mn=fmaxf(m,rmax);
    float corr=ex2(m-mn); m=mn;
    l*=corr;
    float sum=0.f;
    for(int c=0;c<128;c+=8){
      uint32_t r[8]; tmem_ld32(tmem_S+lane_hi+c,r); ld_wait();
      #pragma unroll
      for(int j=0;j<8;j++){ if(kv_start+c+j<S){ sum+=ex2(__uint_as_float(r[j])*SCALE-m);} }
    }
    l+=sum;
  }

  // ---------- PASS B: output ----------
  for(int kv=0;kv<num_kv;kv++){
    int kv_start=kv*BN;
    __syncthreads();
    load_tile(Ksh,Kg,kv_start,S,tid);
    load_tile(Vsh,Vg,kv_start,S,tid);
    __syncthreads();
    fence_async();
    if(tid==0){
      #pragma unroll
      for(int kk=0;kk<8;kk++){
        int block=kk>>2, sub=kk&3;
        uint64_t da=smem_desc(Qsh+block*(128*64)+sub*16,16,1024);
        uint64_t db=smem_desc(Ksh+block*(128*64)+sub*16,16,1024);
        umma(tmem_S,da,db,idesc_qk,kk!=0);
      }
      umma_commit(bar_qk);
    }
    bar_wait(bar_qk,ph_qk); ph_qk^=1;

    // read S -> P (exp), write Psh (K-major swizzled)
    for(int c=0;c<128;c+=8){
      uint32_t r[8]; tmem_ld32(tmem_S+lane_hi+c,r); ld_wait();
      union{ uint4 u; __nv_bfloat16 h[8]; } pk;
      #pragma unroll
      for(int j=0;j<8;j++){
        float p=0.f;
        if(kv_start+c+j<S) p=ex2(__uint_as_float(r[j])*SCALE-m);
        pk.h[j]=__float2bfloat16(p);
      }
      int block=c>>6; int col8=(c&63)>>3; int phys=(tid&7)^col8;
      int elem=block*(128*64)+tid*64+phys*8;
      *reinterpret_cast<uint4*>(&Psh[elem])=pk.u;
    }
    __syncthreads();
    fence_async();
    if(tid==0){
      #pragma unroll
      for(int kk=0;kk<8;kk++){
        int block=kk>>2, sub=kk&3;
        uint64_t da=smem_desc(Psh+block*(128*64)+sub*16,16,1024);
        uint64_t db=smem_desc(Vsh+kk*16*64,16384,1024);
        umma(tmem_O,da,db,idesc_pv,(kv==0&&kk==0)?0:1);
      }
      umma_commit(bar_pv);
    }
    bar_wait(bar_pv,ph_pv); ph_pv^=1;
  }

  // ---------- epilogue ----------
  int gr=q_start+tid;
  float inv=1.f/l;
  for(int c=0;c<128;c+=8){
    uint32_t r[8]; tmem_ld32(tmem_O+lane_hi+c,r); ld_wait();
    union{ uint4 u; __nv_bfloat16 h[8]; } pk;
    #pragma unroll
    for(int j=0;j<8;j++) pk.h[j]=__float2bfloat16(__uint_as_float(r[j])*inv);
    if(gr<S) *reinterpret_cast<uint4*>(&Og[(int64_t)gr*D + c])=pk.u;
  }
  if(gr<S) LSEg[gr]=m*LN2 + logf(l);

  __syncthreads();
  if(warp==0) tmem_dealloc(tmem_base,256);
}

void run(tvm::ffi::TensorView Q,tvm::ffi::TensorView K,tvm::ffi::TensorView V,
         tvm::ffi::TensorView O,tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bsz=(int)Q.size(0), Hsz=(int)Q.size(1), S=(int)Q.size(2);
  const __nv_bfloat16* Qp=static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp=static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp=static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
  float* Lp=static_cast<float*>(LSE.data_ptr());

  int nqt=(S+BM-1)/BM;
  dim3 grid(nqt,Bsz*Hsz);
  int smem=131072 + 1024;

  static bool set=false;
  if(!set){ CUDA_CHECK(cudaFuncSetAttribute(attn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem)); set=true; }

  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
  attn<<<grid,128,smem,stream>>>(Qp,Kp,Vp,Op,Lp,S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_tc::run);

}  // namespace mha_tc