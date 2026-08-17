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

constexpr int BM=128, BN=64, NT=128;
constexpr float SCALE_L2 = 0.08838834764831845f * 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;

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
__device__ __forceinline__ void umma_ss(uint32_t tc,uint64_t da,uint64_t db,uint32_t id,uint32_t acc){
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

// SMEM offsets
constexpr int OFF_Q=0;
constexpr int OFF_K=32768;         // 3*16384
constexpr int OFF_V=32768+49152;   // 81920
constexpr int OFF_P=81920+49152;   // 131072 ; 2*16384
constexpr int OFF_BAR=131072+32768;// 163840

__device__ __forceinline__ void load_kv(const CUtensorMap* tK,const CUtensorMap* tV,uint64_t* bar,char* Ks,char* Vs,int keyrow){
  bar_arrive_expect(bar,32768); // K:2*8192 + V:2*8192 = 32768
  for(int a=0;a<2;a++) tma_load(tK,bar,Ks+a*8192,64*a,keyrow);
  for(int a=0;a<2;a++) tma_load(tV,bar,Vs+a*8192,64*a,keyrow);
}

__global__ __launch_bounds__(128) void attn(
    const __grid_constant__ CUtensorMap tmaQ,
    const __grid_constant__ CUtensorMap tmaK,
    const __grid_constant__ CUtensorMap tmaV,
    __nv_bfloat16* __restrict__ O, float* __restrict__ LSE,
    int B,int H,int S,int num_qtiles){
  extern __shared__ __align__(1024) char smem[];
  char* Qs=smem+OFF_Q;
  char* Ksb=smem+OFF_K;   // stage j: +j*16384 ; halves +0,+8192
  char* Vsb=smem+OFF_V;
  char* Psb=smem+OFF_P;   // buf s: +s*16384
  uint64_t* bars=(uint64_t*)(smem+OFF_BAR);
  uint32_t* tmemptr=(uint32_t*)(smem+OFF_BAR+64);

  uint64_t* bar_q =&bars[0];
  uint64_t* bar_kv[3]={&bars[1],&bars[2],&bars[3]};
  uint64_t* bar_qk[2]={&bars[4],&bars[5]};
  uint64_t* bar_pv=&bars[6];

  int tid=threadIdx.x, warp=tid>>5;
  int qtile=blockIdx.x % num_qtiles;
  int bh=blockIdx.x / num_qtiles;
  int q0=qtile*BM;
  if(q0>=S) return;
  int row=tid;
  int gq=q0+row;
  long rowbase=(long)bh*S;

  if(tid==0){ for(int i=0;i<7;i++) init_bar(&bars[i],1); fence_bar_init(); }
  if(warp==0) tmem_alloc(tmemptr,256);
  __syncthreads();
  uint32_t tb=*tmemptr;
  uint32_t OO=tb+128;       // O at col 128..255

  uint32_t id_qk=idesc(128,64,0,0);
  uint32_t id_pv=idesc(128,64,0,1);

  int q_last=min(q0+BM-1,S-1);
  int N=q_last/BN+1;

  uint32_t p_kv[3]={0,0,0}, p_qk[2]={0,0}, p_pv=0;

  // prologue: load Q + blocks 0,1 ; issue QK[0]
  if(tid==0){
    bar_arrive_expect(bar_q,32768);
    for(int a=0;a<2;a++) tma_load(&tmaQ,bar_q,Qs+a*16384,64*a,(int)(rowbase+q0));
    int pre=min(2,N);
    for(int j=0;j<pre;j++)
      load_kv(&tmaK,&tmaV,bar_kv[j],Ksb+j*16384,Vsb+j*16384,(int)(rowbase+j*BN));
  }
  bar_wait(bar_q,0);
  bar_wait(bar_kv[0],p_kv[0]); p_kv[0]^=1;
  if(tid==0){
    for(int k=0;k<8;k++){int a=k>>2,kl=k&3;
      uint64_t da=smem_desc(Qs+a*16384+kl*32,1,1024);
      uint64_t db=smem_desc(Ksb+a*8192+kl*32,1,1024);
      umma_ss(tb+0,da,db,id_qk,k==0?0:1);}
    umma_commit(bar_qk[0]);
  }

  float m_reg=-1e30f, l_reg=0.f;

  for(int kb=0;kb<N;kb++){
    int s=kb&1, ns=1-s, s3=kb%3;
    int k0=kb*BN;
    bool masked=(kb==qtile)||(k0+BN>S);
    uint32_t Scol=tb + s*64;
    char* Pcur=Psb + s*16384;

    // wait QK[kb]
    bar_wait(bar_qk[s],p_qk[s]); p_qk[s]^=1;
    tc_fence_after();

    // issue QK[kb+1] (overlaps softmax + PV[kb-1])
    if(kb+1<N){
      int st=(kb+1)%3;
      bar_wait(bar_kv[st],p_kv[st]); p_kv[st]^=1;
      if(tid==0){
        char* Kn=Ksb+st*16384;
        uint32_t nScol=tb + ns*64;
        for(int k=0;k<8;k++){int a=k>>2,kl=k&3;
          uint64_t da=smem_desc(Qs+a*16384+kl*32,1,1024);
          uint64_t db=smem_desc(Kn+a*8192+kl*32,1,1024);
          umma_ss(nScol,da,db,id_qk,k==0?0:1);}
        umma_commit(bar_qk[ns]);
      }
    }

    // softmax pass1: rowmax
    float rmax=-1e30f;
    #pragma unroll
    for(int grp=0;grp<64;grp+=32){
      uint32_t r[32];
      ld8(Scol+grp,r); ld8(Scol+grp+8,r+8); ld8(Scol+grp+16,r+16); ld8(Scol+grp+24,r+24);
      wait_ld();
      #pragma unroll
      for(int idx=0;idx<32;idx++){
        float v=__uint_as_float(r[idx])*SCALE_L2;
        if(masked){int gk=k0+grp+idx; if(gk>gq||gk>=S) v=-1e30f;}
        rmax=fmaxf(rmax,v);
      }
    }
    float m_old=m_reg;
    float m_new=fmaxf(m_old,rmax);
    float corr=(kb==0)?1.0f:ex2(m_old-m_new);

    // softmax pass2: P=2^(S-m)
    float rsum=0.f;
    #pragma unroll
    for(int grp=0;grp<64;grp+=32){
      uint32_t r[32];
      ld8(Scol+grp,r); ld8(Scol+grp+8,r+8); ld8(Scol+grp+16,r+16); ld8(Scol+grp+24,r+24);
      wait_ld();
      #pragma unroll
      for(int g=0;g<4;g++){
        int col=grp+g*8;
        float p[8];
        #pragma unroll
        for(int j=0;j<8;j++){
          float v=__uint_as_float(r[g*8+j])*SCALE_L2;
          if(masked){int gk=k0+col+j; if(gk>gq||gk>=S) v=-1e30f;}
          float e=ex2(v-m_new); p[j]=e; rsum+=e;
        }
        int gg=(grp>>3)+g;
        uint32_t w0=pack_bf16(__float_as_uint(p[0]),__float_as_uint(p[1]));
        uint32_t w1=pack_bf16(__float_as_uint(p[2]),__float_as_uint(p[3]));
        uint32_t w2=pack_bf16(__float_as_uint(p[4]),__float_as_uint(p[5]));
        uint32_t w3=pack_bf16(__float_as_uint(p[6]),__float_as_uint(p[7]));
        *(uint4*)&((__nv_bfloat16*)Pcur)[row*64 + (gg^(row&7))*8]=make_uint4(w0,w1,w2,w3);
      }
    }
    l_reg=l_reg*corr+rsum;
    m_reg=m_new;

    // wait PV[kb-1] & rescale O
    if(kb>0){
      bar_wait(bar_pv,p_pv); p_pv^=1;
      tc_fence_after();
      #pragma unroll
      for(int c=0;c<128;c+=32){
        uint32_t r[32];
        ld8(OO+c,r); ld8(OO+c+8,r+8); ld8(OO+c+16,r+16); ld8(OO+c+24,r+24);
        wait_ld();
        #pragma unroll
        for(int i=0;i<32;i++) r[i]=__float_as_uint(__uint_as_float(r[i])*corr);
        st8(OO+c,r); st8(OO+c+8,r+8); st8(OO+c+16,r+16); st8(OO+c+24,r+24);
      }
      wait_st();
      tc_fence_before();
    }

    fence_proxy_async();
    __syncthreads();

    // issue PV[kb]
    if(tid==0){
      tc_fence_after();
      char* Vstg=Vsb+s3*16384;
      #pragma unroll
      for(int nc=0;nc<2;nc++)
        #pragma unroll
        for(int k=0;k<4;k++){
          uint64_t da=smem_desc(Pcur+k*32,1,1024);
          uint64_t db=smem_desc(Vstg+nc*8192+k*2048,8192,1024);
          uint32_t acc=(kb==0&&k==0)?0:1;
          umma_ss(OO+nc*64,da,db,id_pv,acc);
        }
      umma_commit(bar_pv);
    }

    // prefetch block kb+2 into stage (kb+2)%3 (safe: PV[kb-1] done)
    if(tid==0 && kb+2<N){
      int st=(kb+2)%3;
      load_kv(&tmaK,&tmaV,bar_kv[st],Ksb+st*16384,Vsb+st*16384,(int)(rowbase+(kb+2)*BN));
    }
  }

  // wait last PV
  bar_wait(bar_pv,p_pv); p_pv^=1;
  tc_fence_after();

  // epilogue
  float inv=(l_reg>0.f)?1.0f/l_reg:0.f;
  #pragma unroll
  for(int c=0;c<128;c+=32){
    uint32_t r[32];
    ld8(OO+c,r); ld8(OO+c+8,r+8); ld8(OO+c+16,r+16); ld8(OO+c+24,r+24);
    wait_ld();
    if(gq<S){
      #pragma unroll
      for(int g=0;g<4;g++){
        int col=c+g*8;
        uint32_t w0=pack_bf16(__float_as_uint(__uint_as_float(r[g*8+0])*inv),__float_as_uint(__uint_as_float(r[g*8+1])*inv));
        uint32_t w1=pack_bf16(__float_as_uint(__uint_as_float(r[g*8+2])*inv),__float_as_uint(__uint_as_float(r[g*8+3])*inv));
        uint32_t w2=pack_bf16(__float_as_uint(__uint_as_float(r[g*8+4])*inv),__float_as_uint(__uint_as_float(r[g*8+5])*inv));
        uint32_t w3=pack_bf16(__float_as_uint(__uint_as_float(r[g*8+6])*inv),__float_as_uint(__uint_as_float(r[g*8+7])*inv));
        *(uint4*)&O[(rowbase+gq)*128 + col]=make_uint4(w0,w1,w2,w3);
      }
    }
  }
  if(gq<S) LSE[rowbase+gq]=m_reg*LN2 + logf(l_reg);

  __syncthreads();
  if(warp==0) tmem_dealloc(tb,256);
}

static CUresult make_tma(CUtensorMap* d,void* ptr,uint64_t outer,uint32_t box_outer){
  cuuint64_t gdim[2]={(cuuint64_t)128,(cuuint64_t)outer};
  cuuint64_t gstr[1]={(cuuint64_t)128*2};
  cuuint32_t bdim[2]={64,box_outer};
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
  CU_CHECK(make_tma(&dQ,Qp,outer,128));
  CU_CHECK(make_tma(&dK,Kp,outer,64));
  CU_CHECK(make_tma(&dV,Vp,outer,64));

  int num_qtiles=(S+BM-1)/BM;
  dim3 grid(B*H*num_qtiles), block(NT);
  size_t smem=OFF_BAR+256;

  static bool set=false;
  if(!set){ CUDA_CHECK(cudaFuncSetAttribute(attn,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem)); set=true; }

  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id);
  attn<<<grid,block,smem,stream>>>(dQ,dK,dV,Op,Lp,B,H,S,num_qtiles);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel