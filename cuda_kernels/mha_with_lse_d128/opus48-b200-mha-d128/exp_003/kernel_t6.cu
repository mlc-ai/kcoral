#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)

namespace mha_kernel {

#define BM 128
#define BN 64
#define HD 128
#define KSTR 520
#define VSTR 520
#define KBUF 8320   // 16*520
#define PBUF 8192

__device__ __forceinline__ void tmem_alloc1(uint32_t* dst,int n){uint32_t a=(uint32_t)__cvta_generic_to_shared(dst);asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(a),"r"(n));}
__device__ __forceinline__ void tmem_dealloc1(uint32_t a,int n){asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(a),"r"(n));}
__device__ __forceinline__ void tmem_relinquish1(){asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");}
__device__ __forceinline__ void umma1(uint32_t d,uint64_t da,uint64_t db,uint32_t id,uint32_t ac){
  asm volatile("{\n.reg .pred p;\n setp.ne.b32 p,%4,0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"::"r"(d),"l"(da),"l"(db),"r"(id),"r"(ac));}
__device__ __forceinline__ void umma_commit1(uint64_t* bar){uint32_t a=(uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"::"r"(a));}
__device__ __forceinline__ void tmem_ld32(uint32_t t,uint32_t* r){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
    :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
     "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
     "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
     "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31]):"r"(t));}
__device__ __forceinline__ void tmem_wait_ld(){asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");}
__device__ __forceinline__ void tc_fence_before(){asm volatile("tcgen05.fence::before_thread_sync;":::"memory");}
__device__ __forceinline__ void tc_fence_after(){asm volatile("tcgen05.fence::after_thread_sync;":::"memory");}
__device__ __forceinline__ uint64_t make_desc(void* p,uint32_t lbo,uint32_t sbo){
  uint64_t d=0;uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
  d|=(uint64_t)((a&0x3FFFF)>>4); d|=(uint64_t)((lbo&0x3FFFF)>>4)<<16; d|=(uint64_t)((sbo&0x3FFFF)>>4)<<32;
  d|=(uint64_t)1<<46; return d;}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M,uint32_t N,uint32_t bmaj){
  uint32_t d=0;d|=(1u<<4);d|=(1u<<7);d|=(1u<<10);d|=(bmaj<<16);d|=((N/8)<<17);d|=((M/16)<<24);return d;}
__device__ __forceinline__ void bar_init(uint64_t* b,uint32_t c){asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c));}
__device__ __forceinline__ void bar_fence_init(){asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}
__device__ __forceinline__ void bar_wait(uint64_t* b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph));}
__device__ __forceinline__ void fence_async(){asm volatile("fence.proxy.async;":::"memory");}
__device__ __forceinline__ float ex2(float x){float y;asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
__device__ __forceinline__ void cpasync16(void* dst,const void* src){unsigned d=(unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(d),"l"(src):"memory");}
__device__ __forceinline__ void cp_commit(){asm volatile("cp.async.commit_group;":::"memory");}
template<int N> __device__ __forceinline__ void cpwait(){asm volatile("cp.async.wait_group %0;"::"n"(N):"memory");}
__device__ __forceinline__ uint32_t pack2(float x,float y){
  __nv_bfloat16 a=__float2bfloat16(x),b=__float2bfloat16(y);
  uint16_t ai=*reinterpret_cast<uint16_t*>(&a),bi=*reinterpret_cast<uint16_t*>(&b);
  return (uint32_t)ai|((uint32_t)bi<<16);}

__device__ void prefetch_tile(int t,int S,const __nv_bfloat16* Kb,const __nv_bfloat16* Vb,__nv_bfloat16* Kt,__nv_bfloat16* Vt){
  int tid=threadIdx.x; int buf=t&3; int base=t*BN;
  __nv_bfloat16* Kd=Kt+buf*KBUF; __nv_bfloat16* Vd=Vt+buf*KBUF;
  for(int i=tid;i<64*16;i+=128){
    int key=i>>4; int dg=i&15; int d0=dg*8; int gk=base+key; int kk=gk<S?gk:(S>0?S-1:0);
    cpasync16(&Kd[dg*KSTR+key*8], &Kb[(int64_t)kk*HD+d0]);
    cpasync16(&Vd[dg*VSTR+key*8], &Vb[(int64_t)kk*HD+d0]);
  }
  cp_commit();
}
__device__ void issue_qk(uint32_t tb,__nv_bfloat16* Qs,__nv_bfloat16* Kc,uint32_t id,uint64_t* mbar){
  tc_fence_before();
  #pragma unroll
  for(int ks=0;ks<8;ks++){ umma1(tb, make_desc(&Qs[2048*ks],2048,128), make_desc(&Kc[1040*ks],1040,128), id, ks!=0); }
  umma_commit1(mbar);
}
__device__ void issue_pv(uint32_t tb,__nv_bfloat16* Pc,__nv_bfloat16* Vc,uint32_t id,uint64_t* mbar){
  tc_fence_before();
  #pragma unroll
  for(int ks=0;ks<4;ks++){ umma1(tb+64, make_desc(&Pc[2048*ks],2048,128), make_desc(&Vc[128*ks],128,1040), id, ks!=0); }
  umma_commit1(mbar);
}
__device__ float softmax_compute(uint32_t* r0,uint32_t* r1,int kb,int S,float sc,float NEG,float& m_old,float& l,__nv_bfloat16* Pd){
  int tid=threadIdx.x;
  float s[64];
  #pragma unroll
  for(int i=0;i<32;i++){ s[i]=__uint_as_float(r0[i]); s[32+i]=__uint_as_float(r1[i]); }
  float mb=NEG;
  #pragma unroll
  for(int i=0;i<64;i++){ int gk=kb+i; float x=(gk<S)? s[i]*sc:NEG; s[i]=x; mb=fmaxf(mb,x); }
  float m_new=fmaxf(m_old,mb);
  float corr=ex2(m_old-m_new);
  float sum=0.f;
  #pragma unroll
  for(int i=0;i<64;i++){ float p=ex2(s[i]-m_new); s[i]=p; sum+=p; }
  l=l*corr+sum; m_old=m_new;
  #pragma unroll
  for(int g=0;g<8;g++){
    uint4 pk; pk.x=pack2(s[8*g+0],s[8*g+1]); pk.y=pack2(s[8*g+2],s[8*g+3]);
    pk.z=pack2(s[8*g+4],s[8*g+5]); pk.w=pack2(s[8*g+6],s[8*g+7]);
    *reinterpret_cast<uint4*>(&Pd[g*1024+tid*8])=pk;
  }
  return corr;
}

__global__ __launch_bounds__(128,1) void kern(
   const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
   const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O,
   float* __restrict__ LSE, int S, int H){
  int qcta=blockIdx.x*BM, h=blockIdx.y, b=blockIdx.z;
  int64_t bh=(int64_t)b*H+h;
  const __nv_bfloat16* Qb=Q+bh*(int64_t)S*HD;
  const __nv_bfloat16* Kb=K+bh*(int64_t)S*HD;
  const __nv_bfloat16* Vb=V+bh*(int64_t)S*HD;
  __nv_bfloat16* Ob=O+bh*(int64_t)S*HD;

  extern __shared__ char smem[];
  uint64_t* mbar=(uint64_t*)smem;              // [0]=qk [1]=pv
  uint32_t* tmem_slot=(uint32_t*)(smem+16);
  __nv_bfloat16* Qs=(__nv_bfloat16*)(smem+32);
  __nv_bfloat16* Ktiles=Qs+16384;
  __nv_bfloat16* Vtiles=Ktiles+4*KBUF;
  __nv_bfloat16* Ps=Vtiles+4*KBUF;

  int tid=threadIdx.x;
  const float sc=rsqrtf((float)HD)*1.4426950408889634f;
  const float NEG=-1e30f;
  const float LN2=0.6931471805599453f;

  if(tid==0){ bar_init(&mbar[0],1); bar_init(&mbar[1],1); }
  bar_fence_init();
  if(tid<32) tmem_alloc1(tmem_slot,256);
  __syncthreads();
  uint32_t tb=*tmem_slot;
  if(tid<32) tmem_relinquish1();

  // load Q
  for(int i=tid;i<128*16;i+=128){
    int m=i>>4; int dg=i&15; int d0=dg*8; int q=qcta+m; int qq=q<S?q:(S>0?S-1:0);
    cpasync16(&Qs[dg*1024+m*8], &Qb[(int64_t)qq*HD+d0]);
  }
  cp_commit();

  int nblk=(S+BN-1)/BN;
  uint32_t idesc_qk=make_idesc(BM,BN,0);
  uint32_t idesc_pv=make_idesc(BM,HD,1);
  uint32_t qkph=0, pvph=0;

  float Oacc[HD];
  #pragma unroll
  for(int d=0;d<HD;d++) Oacc[d]=0.f;
  float m_old=NEG, l=0.f, corrC=0.f;

  // prologue prefetch tiles 0,1,2
  if(0<nblk) prefetch_tile(0,S,Kb,Vb,Ktiles,Vtiles);
  if(1<nblk) prefetch_tile(1,S,Kb,Vb,Ktiles,Vtiles);
  if(2<nblk) prefetch_tile(2,S,Kb,Vb,Ktiles,Vtiles);
  cpwait<2>();  __syncthreads();

  uint32_t r0[32], r1[32];
  // issue QK_0
  if(tid==0){ fence_async(); issue_qk(tb, Qs, Ktiles+0, idesc_qk, &mbar[0]); }
  // A(0)
  bar_wait(&mbar[0], qkph); qkph^=1;
  tc_fence_after();
  tmem_ld32(tb,r0); tmem_ld32(tb+32,r1); tmem_wait_ld();
  if(1<nblk){
    cpwait<1>(); __syncthreads();
    if(tid==0){ fence_async(); issue_qk(tb, Qs, Ktiles+(1&3)*KBUF, idesc_qk, &mbar[0]); }
  }
  corrC = softmax_compute(r0,r1, 0, S, sc, NEG, m_old, l, Ps+0);

  for(int j=0;j<nblk;j++){
    int cur=j&1;
    if(j+3<nblk) prefetch_tile(j+3,S,Kb,Vb,Ktiles,Vtiles);

    __syncthreads();
    if(tid==0){ fence_async(); issue_pv(tb, Ps+cur*PBUF, Vtiles+(j&3)*KBUF, idesc_pv, &mbar[1]); }

    float corrN=corrC;
    if(j+1<nblk){
      bar_wait(&mbar[0], qkph); qkph^=1;
      tc_fence_after();
      tmem_ld32(tb,r0); tmem_ld32(tb+32,r1); tmem_wait_ld();
      if(j+2<nblk){
        if(j+3<nblk) cpwait<1>(); else cpwait<0>();
        __syncthreads();
        if(tid==0){ fence_async(); issue_qk(tb, Qs, Ktiles+((j+2)&3)*KBUF, idesc_qk, &mbar[0]); }
      }
      corrN = softmax_compute(r0,r1, (j+1)*BN, S, sc, NEG, m_old, l, Ps+((j+1)&1)*PBUF);
    }

    bar_wait(&mbar[1], pvph); pvph^=1;
    tc_fence_after();
    uint32_t p0[32],p1[32],p2[32],p3[32];
    tmem_ld32(tb+64,p0); tmem_ld32(tb+96,p1); tmem_ld32(tb+128,p2); tmem_ld32(tb+160,p3);
    tmem_wait_ld();
    #pragma unroll
    for(int i=0;i<32;i++){
      Oacc[i]    = Oacc[i]*corrC    + __uint_as_float(p0[i]);
      Oacc[32+i] = Oacc[32+i]*corrC + __uint_as_float(p1[i]);
      Oacc[64+i] = Oacc[64+i]*corrC + __uint_as_float(p2[i]);
      Oacc[96+i] = Oacc[96+i]*corrC + __uint_as_float(p3[i]);
    }
    corrC = corrN;
  }

  int row=qcta+tid;
  if(row<S){
    float inv=(l>0.f)? 1.f/l : 0.f;
    #pragma unroll
    for(int d=0;d<HD;d++) Ob[(int64_t)row*HD+d]=__float2bfloat16(Oacc[d]*inv);
    LSE[bh*(int64_t)S+row]=m_old*LN2 + logf(l);
  }

  __syncthreads();
  if(tid<32) tmem_dealloc1(tb,256);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bsz=(int)Q.size(0), Hsz=(int)Q.size(1), Ssz=(int)Q.size(2);
  const __nv_bfloat16* Qp=static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp=static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp=static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
  float* Lp=static_cast<float*>(LSE.data_ptr());
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  int smem = 32 + (16384 + 4*KBUF + 4*KBUF + 2*PBUF)*(int)sizeof(__nv_bfloat16);
  static bool set=false;
  if(!set){ cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); set=true; }

  dim3 grid((Ssz+BM-1)/BM, Hsz, Bsz);
  kern<<<grid,128,smem,stream>>>(Qp,Kp,Vp,Op,Lp,Ssz,Hsz);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel