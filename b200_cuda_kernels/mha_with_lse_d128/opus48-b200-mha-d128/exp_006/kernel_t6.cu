#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s; cuGetErrorString(_e,&s); \
  fprintf(stderr,"CU error %s at %s:%d\n",s,__FILE__,__LINE__); exit(1);} } while(0)

namespace mha_kernel {

using bf16 = __nv_bfloat16;
constexpr int D=128, BM=128, BN=128;

__device__ __forceinline__ void init_bar(uint64_t* b,uint32_t c){
  asm volatile("mbarrier.init.shared.b64 [%0],%1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c));}
__device__ __forceinline__ void fence_bar_init(){asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}
__device__ __forceinline__ void bar_expect(uint64_t* b,uint32_t tx){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(tx):"memory");}
__device__ __forceinline__ void bar_wait(uint64_t* b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph));}
__device__ __forceinline__ void fence_proxy_async(){asm volatile("fence.proxy.async;":::"memory");}

__device__ __forceinline__ void tma_load(const CUtensorMap* d,uint64_t* bar,void* smem,int c0,int c1){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4}],[%2];"
    ::"r"((uint32_t)__cvta_generic_to_shared(smem)),"l"((uint64_t)d),
      "r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c0),"r"(c1):"memory");}

__device__ __forceinline__ void tmem_alloc(uint32_t* dst,int n){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(dst)),"r"(n));}
__device__ __forceinline__ void tmem_dealloc(uint32_t a,int n){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"::"r"(a),"r"(n));}
__device__ __forceinline__ void umma(uint32_t td,uint64_t da,uint64_t db,uint32_t id,uint32_t acc){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    ::"r"(td),"l"(da),"l"(db),"r"(id),"r"(acc));}
__device__ __forceinline__ void umma_commit(uint64_t* b){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)));}
__device__ __forceinline__ void tmem_wait(){asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");}

__device__ __forceinline__ uint64_t smdesc(void* p,uint32_t lbo,uint32_t sbo){
  uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
  d|=(uint64_t)((a&0x3FFFF)>>4);
  d|=(uint64_t)(((lbo&0x3FFFF)>>4))<<16;
  d|=(uint64_t)(((sbo&0x3FFFF)>>4))<<32;
  d|=(uint64_t)1<<46; d|=(uint64_t)2<<61; return d;}
__device__ __forceinline__ uint32_t idesc(uint32_t M,uint32_t N){
  uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10);
  d|=((N/8)<<17); d|=((M/16)<<24); return d;}

// batched TMEM load of 32 columns [col..col+31] of this thread's row -> raw[0..31]
__device__ __forceinline__ void load32(uint32_t tb, int col, uint32_t* raw){
  #pragma unroll
  for(int j=0;j<8;j++){
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];"
      :"=r"(raw[j*4]),"=r"(raw[j*4+1]),"=r"(raw[j*4+2]),"=r"(raw[j*4+3])
      :"r"(tb+col+j*4));
  }
  tmem_wait();
}

__global__ __launch_bounds__(128)
void attn(const __grid_constant__ CUtensorMap dQ,
          const __grid_constant__ CUtensorMap dK,
          const __grid_constant__ CUtensorMap dV,
          bf16* __restrict__ O, float* __restrict__ LSE,
          int S, float scale){
  int bh=blockIdx.y, q_start=blockIdx.x*BM;
  int tid=threadIdx.x, warp=tid>>5;

  extern __shared__ __align__(1024) char smem[];
  bf16* Q0=(bf16*)(smem+0);
  bf16* Q1=(bf16*)(smem+16384);
  bf16* K0=(bf16*)(smem+32768);
  bf16* K1=(bf16*)(smem+49152);
  bf16* Vp=(bf16*)(smem+65536);
  bf16* Vt0=(bf16*)(smem+98304);
  bf16* Vt1=(bf16*)(smem+114688);
  bf16* P0=(bf16*)(smem+131072);
  bf16* P1=(bf16*)(smem+147456);
  uint64_t* bar_q=(uint64_t*)(smem+163840);
  uint64_t* bar_ld=(uint64_t*)(smem+163848);
  uint64_t* bar_mma=(uint64_t*)(smem+163856);
  uint32_t* tmem_ptr=(uint32_t*)(smem+163864);

  bf16* Qreg[2]={Q0,Q1};
  bf16* Kreg[2]={K0,K1};
  bf16* Preg[2]={P0,P1};
  bf16* Vtreg[2]={Vt0,Vt1};

  if(tid==0){ init_bar(bar_q,1); init_bar(bar_ld,1); init_bar(bar_mma,1); }
  fence_bar_init();
  __syncthreads();
  if(warp==0) tmem_alloc(tmem_ptr,256);
  __syncthreads();
  uint32_t tb=*tmem_ptr;

  int rowg = q_start + tid;
  bool row_valid = rowg < S;

  if(tid==0){
    bar_expect(bar_q,32768);
    tma_load(&dQ,bar_q,Q0,0,bh*S+q_start);
    tma_load(&dQ,bar_q,Q1,64,bh*S+q_start);
  }
  bar_wait(bar_q,0);

  float acc[D];
  #pragma unroll
  for(int c=0;c<D;c++) acc[c]=0.f;
  float m=-1e30f, l=0.f;

  uint32_t id1=idesc(128,128);
  int num_kv=(S+BN-1)/BN;
  uint32_t ph_ld=0, ph_mma=0;

  for(int kv=0;kv<num_kv;kv++){
    int kvs=kv*BN;
    if(tid==0){
      bar_expect(bar_ld,65536);
      tma_load(&dK,bar_ld,K0,0,bh*S+kvs);
      tma_load(&dK,bar_ld,K1,64,bh*S+kvs);
      tma_load(&dV,bar_ld,Vp,0,bh*S+kvs);
    }
    bar_wait(bar_ld,ph_ld); ph_ld^=1;

    // transpose Vp[kv,d] -> Vt[d,kv] swizzled ; thread handles d=tid
    {
      int d=tid;
      #pragma unroll
      for(int kv0=0;kv0<128;kv0+=8){
        bf16 pb[8];
        #pragma unroll
        for(int i=0;i<8;i++) pb[i]=Vp[(kv0+i)*128 + d];
        int region=kv0/64, local=kv0%64, chunk=local/8;
        char* base=(char*)Vtreg[region];
        uint4* dst=(uint4*)(base + d*128 + ((d&7)^chunk)*16);
        uint4 v; v.x=*(uint32_t*)&pb[0]; v.y=*(uint32_t*)&pb[2];
        v.z=*(uint32_t*)&pb[4]; v.w=*(uint32_t*)&pb[6];
        *dst=v;
      }
    }
    __syncthreads();

    // MMA1: S = Q @ K^T
    if(tid==0){
      #pragma unroll
      for(int k=0;k<8;k++){
        int r=k/4, loc=(k%4)*32;
        uint64_t da=smdesc((char*)Qreg[r]+loc,16,1024);
        uint64_t db=smdesc((char*)Kreg[r]+loc,16,1024);
        umma(tb, da, db, id1, k==0?0:1);
      }
      umma_commit(bar_mma);
    }
    bar_wait(bar_mma,ph_mma); ph_mma^=1;

    // ---- softmax (thread tid owns row tid) ----
    float m_old=m;
    float rmax=-1e30f;
    uint32_t raw[32];
    #pragma unroll
    for(int chunk=0;chunk<4;chunk++){
      load32(tb, chunk*32, raw);
      #pragma unroll
      for(int i=0;i<32;i++){
        int g=kvs+chunk*32+i;
        if(g<S) rmax=fmaxf(rmax,__uint_as_float(raw[i])*scale);
      }
    }
    float m_new=fmaxf(m_old,rmax);
    float corr=__expf(m_old-m_new);
    float rsum=0.f;
    #pragma unroll
    for(int chunk=0;chunk<4;chunk++){
      load32(tb, chunk*32, raw);
      #pragma unroll
      for(int grp=0;grp<4;grp++){
        bf16 pb[8];
        #pragma unroll
        for(int i=0;i<8;i++){
          int c=chunk*32+grp*8+i, g=kvs+c;
          float p=(g<S)?__expf(__uint_as_float(raw[grp*8+i])*scale-m_new):0.f;
          rsum+=p; pb[i]=__float2bfloat16(p);
        }
        int c0=chunk*32+grp*8;
        int region=c0/64, local=c0%64, chunkw=local/8;
        char* base=(char*)Preg[region];
        uint4* dst=(uint4*)(base + tid*128 + ((tid&7)^chunkw)*16);
        uint4 pv; pv.x=*(uint32_t*)&pb[0]; pv.y=*(uint32_t*)&pb[2];
        pv.z=*(uint32_t*)&pb[4]; pv.w=*(uint32_t*)&pb[6];
        *dst=pv;
      }
    }
    m=m_new; l=l*corr+rsum;
    __syncthreads();
    fence_proxy_async();
    __syncthreads();

    // MMA2: deltaO = P @ Vt
    if(tid==0){
      #pragma unroll
      for(int k=0;k<8;k++){
        int r=k/4, loc=(k%4)*32;
        uint64_t da=smdesc((char*)Preg[r]+loc,16,1024);
        uint64_t db=smdesc((char*)Vtreg[r]+loc,16,1024);
        umma(tb+128, da, db, id1, k==0?0:1);
      }
      umma_commit(bar_mma);
    }
    bar_wait(bar_mma,ph_mma); ph_mma^=1;

    // O = O*corr + deltaO  (batched read)
    #pragma unroll
    for(int chunk=0;chunk<4;chunk++){
      load32(tb+128, chunk*32, raw);
      #pragma unroll
      for(int i=0;i<32;i++){
        int c=chunk*32+i;
        acc[c]=acc[c]*corr+__uint_as_float(raw[i]);
      }
    }
    __syncthreads();
  }

  if(row_valid){
    float inv=1.f/l;
    bf16* Ob=O+(size_t)(bh*S+rowg)*D;
    #pragma unroll
    for(int c=0;c<D;c++) Ob[c]=__float2bfloat16(acc[c]*inv);
    LSE[(size_t)bh*S+rowg]=m+logf(l);
  }
  __syncthreads();
  if(warp==0) tmem_dealloc(tb,256);
}

static CUresult mk_tma(CUtensorMap* d,void* p,uint64_t inner,uint64_t outer,
                       uint32_t binner,uint32_t bouter,CUtensorMapSwizzle sw){
  uint64_t gd[2]={inner,outer};
  uint64_t gs[1]={inner*2};
  uint32_t bd[2]={binner,bouter};
  uint32_t es[2]={1,1};
  return cuTensorMapEncodeTiled(d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,p,gd,gs,bd,es,
    CU_TENSOR_MAP_INTERLEAVE_NONE,sw,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=Q.size(0), H=Q.size(1), S=Q.size(2);
  float scale=1.0f/sqrtf((float)D);
  bf16* Qp=(bf16*)Q.data_ptr(); bf16* Kp=(bf16*)K.data_ptr(); bf16* Vp=(bf16*)V.data_ptr();
  bf16* Op=(bf16*)O.data_ptr(); float* Lp=(float*)LSE.data_ptr();

  uint64_t rows=(uint64_t)B*H*S;
  CUtensorMap dQ,dK,dV;
  CU_CHECK(mk_tma(&dQ,Qp,D,rows,64,BM,CU_TENSOR_MAP_SWIZZLE_128B));
  CU_CHECK(mk_tma(&dK,Kp,D,rows,64,BN,CU_TENSOR_MAP_SWIZZLE_128B));
  CU_CHECK(mk_tma(&dV,Vp,D,rows,128,BN,CU_TENSOR_MAP_SWIZZLE_NONE));

  dim3 grid((S+BM-1)/BM, B*H);
  dim3 block(128);
  size_t smem=164096;
  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id);
  CUDA_CHECK(cudaFuncSetAttribute(attn,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
  attn<<<grid,block,smem,stream>>>(dQ,dK,dV,Op,Lp,S,scale);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel