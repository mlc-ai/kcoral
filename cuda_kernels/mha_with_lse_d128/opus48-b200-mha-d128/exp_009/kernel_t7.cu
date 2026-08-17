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

constexpr int BM=64, BN=128, D=128;
constexpr int BLK_Q=64*64;    // 4096
constexpr int BLK_KV=128*64;  // 8192
constexpr int BLK_P=64*64;    // 4096
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

__device__ __forceinline__ void tmem_ld_x32(uint32_t t,uint32_t* r){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
    :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
     "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
     "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
     "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
    :"r"(t)); }

__device__ __forceinline__ void tmem_st_x32(uint32_t t,const uint32_t* r){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
    "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
    "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};"
    ::"r"(t),
     "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),
     "r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),"r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]),
     "r"(r[16]),"r"(r[17]),"r"(r[18]),"r"(r[19]),"r"(r[20]),"r"(r[21]),"r"(r[22]),"r"(r[23]),
     "r"(r[24]),"r"(r[25]),"r"(r[26]),"r"(r[27]),"r"(r[28]),"r"(r[29]),"r"(r[30]),"r"(r[31])
    :"memory"); }

__device__ __forceinline__ void ld_wait(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void st_wait(){ asm volatile("tcgen05.wait::st.sync.aligned;":::"memory"); }

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

__global__ __launch_bounds__(128,2) void attn(
    const __grid_constant__ CUtensorMap dQ,
    const __grid_constant__ CUtensorMap dK,
    const __grid_constant__ CUtensorMap dV,
    __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int S){
  extern __shared__ char smem_raw[];
  uintptr_t ap=((uintptr_t)smem_raw + 1023)&~(uintptr_t)1023;
  __nv_bfloat16* Qsh=(__nv_bfloat16*)ap;
  __nv_bfloat16* Ksh=Qsh+2*BLK_Q;
  __nv_bfloat16* Vsh=Ksh+2*BLK_KV;
  __nv_bfloat16* Psh=Vsh+2*BLK_KV;
  uint64_t* bars=(uint64_t*)(Psh+2*BLK_P);
  uint64_t* bar_q=&bars[0], *bar_load=&bars[1], *bar_qk=&bars[2], *bar_pv=&bars[3];
  uint32_t* tmem_ptr=(uint32_t*)&bars[4];

  int tid=threadIdx.x; int warp=tid>>5;
  int q_start=blockIdx.x*BM;
  int bh=blockIdx.y;
  __nv_bfloat16* Og=O+(int64_t)bh*S*D;
  float* LSEg=LSE+(int64_t)bh*S;
  int row_base=bh*S;
  uint32_t lane_hi=(uint32_t)warp<<21;

  if(warp==0) tmem_alloc(tmem_ptr,256);
  if(tid==0){ init_bar(bar_q,1); init_bar(bar_load,1); init_bar(bar_qk,1); init_bar(bar_pv,1); }
  fence_bar_init();
  __syncthreads();
  uint32_t tmem_base=*tmem_ptr;
  uint32_t tmem_S=tmem_base;
  uint32_t tmem_O=tmem_base+128;

  uint32_t idesc_qk=instr_desc(0,0);
  uint32_t idesc_pv=instr_desc(0,1);

  int num_kv=(S+BN-1)/BN;

  if(tid==0){
    mbar_arrive_expect(bar_q,2*BLK_Q*2);
    tma_load(&dQ,bar_q,Qsh+0,     0, row_base+q_start);
    tma_load(&dQ,bar_q,Qsh+BLK_Q, 64,row_base+q_start);
  }
  bar_wait(bar_q,0);

  uint32_t ph_load=0, ph_qk=0, ph_pv=0;
  float m=-1e30f, l=0.f;

  for(int kv=0; kv<num_kv; kv++){
    int kv_start=kv*BN;

    if(kv>0){ bar_wait(bar_pv,ph_pv); ph_pv^=1; }   // PV(kv-1) done -> buffers free, O stable

    if(tid==0){
      mbar_arrive_expect(bar_load,4*BLK_KV*2);
      tma_load(&dK,bar_load,Ksh+0,      0, row_base+kv_start);
      tma_load(&dK,bar_load,Ksh+BLK_KV, 64,row_base+kv_start);
      tma_load(&dV,bar_load,Vsh+0,      0, row_base+kv_start);
      tma_load(&dV,bar_load,Vsh+BLK_KV, 64,row_base+kv_start);
    }
    bar_wait(bar_load,ph_load); ph_load^=1;

    if(tid==0){
      #pragma unroll
      for(int kk=0;kk<8;kk++){
        int block=kk>>2, sub=kk&3;
        uint64_t da=smem_desc(Qsh+block*BLK_Q +sub*16,16,1024);
        uint64_t db=smem_desc(Ksh+block*BLK_KV+sub*16,16,1024);
        umma(tmem_S,da,db,idesc_qk,kk!=0);
      }
      umma_commit(bar_qk);
    }
    bar_wait(bar_qk,ph_qk); ph_qk^=1;

    if(warp<2){
      float rmax=-1e30f;
      #pragma unroll
      for(int c=0;c<128;c+=32){
        uint32_t r[32]; tmem_ld_x32(tmem_S+lane_hi+(uint32_t)c,r); ld_wait();
        #pragma unroll
        for(int j=0;j<32;j++){ if(kv_start+c+j<S){ float s=__uint_as_float(r[j])*SCALE; rmax=fmaxf(rmax,s);} }
      }
      float mnew=fmaxf(m,rmax);
      float corr=ex2(m-mnew);

      if(kv>0){
        #pragma unroll
        for(int c=0;c<128;c+=32){
          uint32_t r[32]; tmem_ld_x32(tmem_O+lane_hi+(uint32_t)c,r); ld_wait();
          #pragma unroll
          for(int j=0;j<32;j++){ r[j]=__float_as_uint(__uint_as_float(r[j])*corr); }
          tmem_st_x32(tmem_O+lane_hi+(uint32_t)c,r);
        }
        st_wait();
      }

      float sum=0.f;
      #pragma unroll
      for(int c=0;c<128;c+=32){
        uint32_t r[32]; tmem_ld_x32(tmem_S+lane_hi+(uint32_t)c,r); ld_wait();
        #pragma unroll
        for(int cc=0; cc<32; cc+=8){
          int col=c+cc;
          union{uint4 u; __nv_bfloat16 h[8];} pk;
          #pragma unroll
          for(int j=0;j<8;j++){
            float p=0.f;
            if(kv_start+col+j<S){ p=ex2(__uint_as_float(r[cc+j])*SCALE-mnew); sum+=p; }
            pk.h[j]=__float2bfloat16(p);
          }
          int block=col>>6; int col8=(col&63)>>3; int phys=(tid&7)^col8;
          int elem=block*BLK_P + tid*64 + phys*8;
          *reinterpret_cast<uint4*>(&Psh[elem])=pk.u;
        }
      }
      l = l*corr + sum;
      m = mnew;
    }

    fence_async();
    __syncthreads();
    if(tid==0){
      #pragma unroll
      for(int kk=0;kk<8;kk++){
        int block=kk>>2, sub=kk&3;
        uint64_t da=smem_desc(Psh+block*BLK_P+sub*16,16,1024);
        uint64_t db=smem_desc(Vsh+kk*1024,16384,1024);
        umma(tmem_O,da,db,idesc_pv,(kv==0&&kk==0)?0:1);
      }
      umma_commit(bar_pv);
    }
  }

  bar_wait(bar_pv,ph_pv); ph_pv^=1;

  if(warp<2){
    int gr=q_start+tid;
    float inv = l>0.f? 1.f/l : 0.f;
    #pragma unroll
    for(int c=0;c<128;c+=32){
      uint32_t r[32]; tmem_ld_x32(tmem_O+lane_hi+(uint32_t)c,r); ld_wait();
      #pragma unroll
      for(int cc=0; cc<32; cc+=8){
        union{uint4 u; __nv_bfloat16 h[8];} pk;
        #pragma unroll
        for(int j=0;j<8;j++) pk.h[j]=__float2bfloat16(__uint_as_float(r[cc+j])*inv);
        if(gr<S) *reinterpret_cast<uint4*>(&Og[(int64_t)gr*D + c+cc])=pk.u;
      }
    }
    if(gr<S) LSEg[gr]=m*LN2 + logf(l);
  }

  __syncthreads();
  if(warp==0) tmem_dealloc(tmem_base,256);
}

static CUtensorMap make_desc(void* ptr, uint64_t rows, uint32_t box_rows){
  CUtensorMap d;
  uint64_t gdim[2]={ (uint64_t)D, rows };
  uint64_t gstride[1]={ (uint64_t)D*2 };
  uint32_t bdim[2]={ 64, box_rows };
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
  CUtensorMap dQ=make_desc(Qp,rows,BM);
  CUtensorMap dK=make_desc(Kp,rows,BN);
  CUtensorMap dV=make_desc(Vp,rows,BN);

  int nqt=(S+BM-1)/BM;
  dim3 grid(nqt,Bsz*Hsz);
  int smem = (2*BLK_Q + 4*BLK_KV + 2*BLK_P)*2 + 64 + 1024;

  static bool set=false;
  if(!set){ CUDA_CHECK(cudaFuncSetAttribute(attn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem)); set=true; }

  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
  attn<<<grid,128,smem,stream>>>(dQ,dK,dV,Op,Lp,S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_tc::run);

}  // namespace mha_tc