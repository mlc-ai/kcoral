#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do{ cudaError_t e=(call); if(e!=cudaSuccess){ fprintf(stderr,"CUDA %s %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); exit(1);} }while(0)
#define CU_CHECK(call) do{ CUresult r=(call); if(r!=CUDA_SUCCESS){ const char* s; cuGetErrorString(r,&s); fprintf(stderr,"CU %s %s:%d\n",s,__FILE__,__LINE__); exit(1);} }while(0)

namespace mha {

constexpr int D=128, BM=128, BN=128;
constexpr int CHUNK=128*64*2;

__device__ __forceinline__ void init_bar(uint64_t* b,uint32_t c){ asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c)); }
__device__ __forceinline__ void bar_ae(uint64_t* b,uint32_t bytes){ asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(bytes):"memory"); }
__device__ __forceinline__ void bar_arrive(uint64_t* b){ asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"::"r"((uint32_t)__cvta_generic_to_shared(b)):"memory"); }
__device__ __forceinline__ void bar_wait(uint64_t* b,uint32_t ph){ asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph)); }
__device__ __forceinline__ void tma2d(const CUtensorMap* d,uint64_t* b,void* s,int c0,int c1){ asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4}],[%2];"::"r"((uint32_t)__cvta_generic_to_shared(s)),"l"((uint64_t)d),"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c0),"r"(c1):"memory"); }
__device__ __forceinline__ uint64_t sdesc(void* p,uint32_t lbo,uint32_t sbo){ uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(p); d|=(uint64_t)((a&0x3FFFF)>>4); d|=(uint64_t)((lbo&0x3FFFF)>>4)<<16; d|=(uint64_t)((sbo&0x3FFFF)>>4)<<32; d|=(uint64_t)1<<46; d|=(uint64_t)2<<61; return d; }
__device__ __forceinline__ uint32_t idesc(uint32_t M,uint32_t N,uint32_t am,uint32_t bm){ uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10); d|=(am<<15); d|=(bm<<16); d|=((N>>3)<<17); d|=((M>>4)<<24); return d; }
__device__ __forceinline__ void tmem_alloc(uint32_t* dst,int n){ asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(dst)),"r"(n)); }
__device__ __forceinline__ void tmem_dealloc(uint32_t a,int n){ asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"::"r"(a),"r"(n)); }
__device__ __forceinline__ void tmem_relinq(){ asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"); }
__device__ __forceinline__ void umma1(uint32_t c,uint64_t da,uint64_t db,uint32_t id,uint32_t acc){ asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\ntcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"::"r"(c),"l"(da),"l"(db),"r"(id),"r"(acc)); }
__device__ __forceinline__ void umma_commit(uint64_t* b){ asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"::"r"((uint32_t)__cvta_generic_to_shared(b))); }
__device__ __forceinline__ void ld8(uint32_t col,uint32_t* r){ asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]):"r"(col)); }
__device__ __forceinline__ void wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void fb(){ asm volatile("tcgen05.fence::before_thread_sync;":::"memory"); }
__device__ __forceinline__ void fa(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }
__device__ __forceinline__ void fpa(){ asm volatile("fence.proxy.async;":::"memory"); }
__device__ __forceinline__ float ex2(float x){ float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y; }
__device__ __forceinline__ uint32_t f2bf2(float a,float b){ __nv_bfloat162 t=__floats2bfloat162_rn(a,b); return *reinterpret_cast<uint32_t*>(&t);}

__global__ __launch_bounds__(160,1)
void kernel(const __grid_constant__ CUtensorMap descQ,
            const __grid_constant__ CUtensorMap descK,
            const __grid_constant__ CUtensorMap descV,
            __nv_bfloat16* O, float* LSE, int B,int H,int S){
  extern __shared__ char raw[];
  uint32_t shoff=(uint32_t)__cvta_generic_to_shared(raw);
  uint32_t pad=(1024-(shoff&1023))&1023;
  char* sb=raw+pad;
  uint64_t* bars=(uint64_t*)(sb+14*CHUNK);
  uint64_t* barQ=&bars[0];
  uint64_t* barKload[2]={&bars[1],&bars[2]};
  uint64_t* barVload[2]={&bars[3],&bars[4]};
  uint64_t* barS[2]={&bars[5],&bars[6]};
  uint64_t* barSfree[2]={&bars[7],&bars[8]};
  uint64_t* barP[2]={&bars[9],&bars[10]};
  uint64_t* barPV[2]={&bars[11],&bars[12]};
  uint64_t* barFinal=&bars[13];
  uint32_t* tmptr=(uint32_t*)(&bars[16]);

  int tid=threadIdx.x, warp=tid>>5;
  int qblock=blockIdx.x,h=blockIdx.y,b=blockIdx.z;
  int qbase=qblock*BM;
  long rowbase=(long)(b*H+h)*S;
  int nblk=(S+BN-1)/BN;
  const float scale2=rsqrtf((float)D)*1.4426950408889634f;

  if(tid==0){
    init_bar(barQ,1);
    init_bar(barKload[0],1); init_bar(barKload[1],1);
    init_bar(barVload[0],1); init_bar(barVload[1],1);
    init_bar(barS[0],1); init_bar(barS[1],1);
    init_bar(barSfree[0],128); init_bar(barSfree[1],128);
    init_bar(barP[0],128); init_bar(barP[1],128);
    init_bar(barPV[0],1); init_bar(barPV[1],1);
    init_bar(barFinal,1);
    asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  }
  if(warp==4) tmem_alloc(tmptr,512);
  __syncthreads();
  uint32_t tbase=tmptr[0];
  uint32_t Ocol=tbase+256;
  if(warp==4) tmem_relinq();

  uint32_t idQK=idesc(128,128,0,0);
  uint32_t idPV=idesc(128,128,0,1);

  if(warp==4){
    if(tid==128){
      int cpK[2]={0,0},cpV[2]={0,0},cpSfree[2]={0,0},cpSK[2]={0,0},cpP[2]={0,0},cpPVv[2]={0,0};
      // prologue
      bar_ae(barQ,2*CHUNK);
      tma2d(&descQ,barQ,sb+0*CHUNK,0,(int)(rowbase+qbase));
      tma2d(&descQ,barQ,sb+1*CHUNK,64,(int)(rowbase+qbase));
      bar_ae(barKload[0],2*CHUNK);
      tma2d(&descK,barKload[0],sb+2*CHUNK,0,(int)rowbase);
      tma2d(&descK,barKload[0],sb+3*CHUNK,64,(int)rowbase);
      bar_ae(barVload[0],2*CHUNK);
      tma2d(&descV,barVload[0],sb+6*CHUNK,0,(int)rowbase);
      tma2d(&descV,barVload[0],sb+7*CHUNK,64,(int)rowbase);
      if(nblk>1){
        bar_ae(barKload[1],2*CHUNK);
        tma2d(&descK,barKload[1],sb+4*CHUNK,0,(int)(rowbase+BN));
        tma2d(&descK,barKload[1],sb+5*CHUNK,64,(int)(rowbase+BN));
        bar_ae(barVload[1],2*CHUNK);
        tma2d(&descV,barVload[1],sb+8*CHUNK,0,(int)(rowbase+BN));
        tma2d(&descV,barVload[1],sb+9*CHUNK,64,(int)(rowbase+BN));
      }
      bar_wait(barQ,0);
      bar_wait(barKload[0],cpK[0]); cpK[0]^=1;
      fa();
      for(int ks=0;ks<8;ks++){
        void* ap=sb+(ks/4)*CHUNK+(ks%4)*32;
        void* bp=sb+(2+ks/4)*CHUNK+(ks%4)*32;
        umma1(tbase+0,sdesc(ap,1,1024),sdesc(bp,1,1024),idQK,ks?1:0);
      }
      umma_commit(barS[0]);

      for(int i=0;i<nblk;i++){
        int s=i&1, sn=(i+1)&1;
        // QK(i+1)
        if(i+1<nblk){
          bar_wait(barKload[sn],cpK[sn]); cpK[sn]^=1;
          if(i>=1){ bar_wait(barSfree[sn],cpSfree[sn]); cpSfree[sn]^=1; }
          fa();
          int Kc=2+2*sn;
          for(int ks=0;ks<8;ks++){
            void* ap=sb+(ks/4)*CHUNK+(ks%4)*32;
            void* bp=sb+(Kc+ks/4)*CHUNK+(ks%4)*32;
            umma1(tbase+sn*128,sdesc(ap,1,1024),sdesc(bp,1,1024),idQK,ks?1:0);
          }
          umma_commit(barS[sn]);
        }
        // K prefetch (i+2)
        if(i+2<nblk){
          bar_wait(barS[s],cpSK[s]); cpSK[s]^=1;
          int Kc=2+2*s;
          bar_ae(barKload[s],2*CHUNK);
          tma2d(&descK,barKload[s],sb+Kc*CHUNK,0,(int)(rowbase+(long)(i+2)*BN));
          tma2d(&descK,barKload[s],sb+(Kc+1)*CHUNK,64,(int)(rowbase+(long)(i+2)*BN));
        }
        // PV(i)
        bar_wait(barP[s],cpP[s]); cpP[s]^=1;
        bar_wait(barVload[s],cpV[s]); cpV[s]^=1;
        fa();
        int Pc=10+2*s, Vc=6+2*s;
        for(int ks=0;ks<8;ks++){
          void* ap=sb+(Pc+ks/4)*CHUNK+(ks%4)*32;
          void* bp=sb+Vc*CHUNK+2048*ks;
          umma1(Ocol,sdesc(ap,1,1024),sdesc(bp,16384,1024),idPV,(i==0&&ks==0)?0:1);
        }
        if(i<nblk-1) umma_commit(barPV[s]); else umma_commit(barFinal);
        // V prefetch (i+2)
        if(i+2<nblk){
          bar_wait(barPV[s],cpPVv[s]); cpPVv[s]^=1;
          int Vc2=6+2*s;
          bar_ae(barVload[s],2*CHUNK);
          tma2d(&descV,barVload[s],sb+Vc2*CHUNK,0,(int)(rowbase+(long)(i+2)*BN));
          tma2d(&descV,barVload[s],sb+(Vc2+1)*CHUNK,64,(int)(rowbase+(long)(i+2)*BN));
        }
      }
    }
  } else {
    int row=qbase+tid;
    float l=0.f;
    int spS[2]={0,0}, spPV[2]={0,0};
    for(int i=0;i<nblk;i++){
      int s=i&1;
      int kvbase=i*BN;
      bool full=(kvbase+BN<=S);
      bar_wait(barS[s],spS[s]); spS[s]^=1;
      if(i>=2){ bar_wait(barPV[s],spPV[s]); spPV[s]^=1; }
      fa();
      uint32_t Scur=tbase+s*128;
      int Pbase=10+2*s;
      float rs=0.f;
      for(int c=0;c<128;c+=32){
        uint32_t r[32];
        ld8(Scur+c,r);ld8(Scur+c+8,r+8);ld8(Scur+c+16,r+16);ld8(Scur+c+24,r+24);
        wait_ld();
        #pragma unroll
        for(int sub=0;sub<4;sub++){
          int subc=c+sub*8;
          float pv[8];
          if(full){
            #pragma unroll
            for(int j=0;j<8;j++){ float p=ex2(__uint_as_float(r[sub*8+j])*scale2); rs+=p; pv[j]=p; }
          } else {
            #pragma unroll
            for(int j=0;j<8;j++){ float p; if(kvbase+subc+j<S) p=ex2(__uint_as_float(r[sub*8+j])*scale2); else p=0.f; rs+=p; pv[j]=p; }
          }
          int nch=subc>>6, x=(subc&63)>>3;
          char* pp=sb+(Pbase+nch)*CHUNK + tid*128 + (((tid&7)^x)*16);
          uint4 out; out.x=f2bf2(pv[0],pv[1]);out.y=f2bf2(pv[2],pv[3]);out.z=f2bf2(pv[4],pv[5]);out.w=f2bf2(pv[6],pv[7]);
          *reinterpret_cast<uint4*>(pp)=out;
        }
      }
      l+=rs;
      bar_arrive(barSfree[s]);
      fpa();
      bar_arrive(barP[s]);
    }
    bar_wait(barFinal,0);
    fa();
    float inv=1.f/l;
    for(int c=0;c<128;c+=32){
      uint32_t r[32];
      ld8(Ocol+c,r);ld8(Ocol+c+8,r+8);ld8(Ocol+c+16,r+16);ld8(Ocol+c+24,r+24);
      wait_ld();
      if(row<S){
        #pragma unroll
        for(int sub=0;sub<4;sub++){
          float ov[8];
          #pragma unroll
          for(int j=0;j<8;j++) ov[j]=__uint_as_float(r[sub*8+j])*inv;
          uint4 out; out.x=f2bf2(ov[0],ov[1]);out.y=f2bf2(ov[2],ov[3]);out.z=f2bf2(ov[4],ov[5]);out.w=f2bf2(ov[6],ov[7]);
          *reinterpret_cast<uint4*>(&O[(rowbase+row)*D+c+sub*8])=out;
        }
      }
    }
    if(row<S) LSE[rowbase+row]=logf(l);
  }
  __syncthreads();
  if(warp==4) tmem_dealloc(tbase,512);
}

CUresult make_tma(CUtensorMap* d,void* p,uint64_t inner,uint64_t outer){
  uint64_t gd[2]={inner,outer}; uint64_t gs[1]={inner*2}; uint32_t bd[2]={64,128}; uint32_t es[2]={1,1};
  return cuTensorMapEncodeTiled(d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,p,gd,gs,bd,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2);
  __nv_bfloat16* Qp=(__nv_bfloat16*)Q.data_ptr();
  __nv_bfloat16* Kp=(__nv_bfloat16*)K.data_ptr();
  __nv_bfloat16* Vp=(__nv_bfloat16*)V.data_ptr();
  __nv_bfloat16* Op=(__nv_bfloat16*)O.data_ptr();
  float* LSEp=(float*)LSE.data_ptr();

  CUtensorMap dQ,dK,dV;
  uint64_t outer=(uint64_t)B*H*S;
  CU_CHECK(make_tma(&dQ,Qp,D,outer));
  CU_CHECK(make_tma(&dK,Kp,D,outer));
  CU_CHECK(make_tma(&dV,Vp,D,outer));

  int smem=14*CHUNK+2048;
  cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
  dim3 grid((S+BM-1)/BM,H,B); dim3 block(160);
  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id);
  kernel<<<grid,block,smem,stream>>>(dQ,dK,dV,Op,LSEp,B,H,S);
  CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha::run);

}  // namespace mha