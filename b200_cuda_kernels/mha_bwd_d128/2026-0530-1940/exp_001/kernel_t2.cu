#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do{ cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} }while(0)
#define CU_CHECK(call) do{ CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s; cuGetErrorString(_e,&s); \
  fprintf(stderr,"CU error %s at %s:%d\n",s,__FILE__,__LINE__); exit(1);} }while(0)

namespace mha_bwd {
using bf16 = __nv_bfloat16;

// ---------------- device helpers ----------------
__device__ __forceinline__ uint32_t cvta(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void init_bar(uint64_t* b, uint32_t c){
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"::"r"(cvta(b)),"r"(c)); }
__device__ __forceinline__ void fence_bar_init(){ asm volatile("fence.mbarrier_init.release.cluster;":::"memory"); }
__device__ __forceinline__ void arrive_expect_tx(uint64_t* b, uint32_t tx){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"(cvta(b)),"r"(tx):"memory"); }
__device__ __forceinline__ void bar_wait(uint64_t* b, uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"
    ::"r"(cvta(b)),"r"(ph)); }
__device__ __forceinline__ void fence_async(){ asm volatile("fence.proxy.async;":::"memory"); }

__device__ __forceinline__ void tma_load(const CUtensorMap* d, uint64_t* bar, void* smem, int c0, int c1){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4}],[%2];"
    ::"r"(cvta(smem)),"l"((uint64_t)d),"r"(cvta(&bar[0])),"r"(c0),"r"(c1):"memory"); }

__device__ __forceinline__ void tmem_alloc(uint32_t* dst,int n){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(cvta(dst)),"r"(n)); }
__device__ __forceinline__ void tmem_dealloc(uint32_t a,int n){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(a),"r"(n)); }
__device__ __forceinline__ void umma(uint32_t td,uint64_t da,uint64_t db,uint32_t id,uint32_t acc){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    ::"r"(td),"l"(da),"l"(db),"r"(id),"r"(acc)); }
__device__ __forceinline__ void umma_commit(uint64_t* bar){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"::"r"(cvta(&bar[0]))); }
__device__ __forceinline__ void tmem_ld_x4(uint32_t a,uint32_t*r0,uint32_t*r1,uint32_t*r2,uint32_t*r3){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];"
    :"=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3):"r"(a)); }
__device__ __forceinline__ void tmem_wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }

__device__ __forceinline__ uint64_t mk_desc(uint32_t saddr, uint32_t sbo){
  uint64_t d=0;
  d |= (uint64_t)((saddr&0x3FFFFu)>>4);
  d |= (uint64_t)((16u&0x3FFFFu)>>4)<<16;      // LBO=16
  d |= (uint64_t)((sbo&0x3FFFFu)>>4)<<32;       // SBO
  d |= (uint64_t)1<<46;                         // const 0b001
  return d;                                     // swizzle = 0 (none)
}
__device__ __forceinline__ uint32_t mk_idesc(int M,int N){
  uint32_t d=0;
  d |= (1u<<4);            // D = FP32
  d |= (1u<<7);            // A = BF16
  d |= (1u<<10);           // B = BF16
  d |= ((uint32_t)(N>>3)<<17);
  d |= ((uint32_t)(M>>4)<<24);
  return d;
}

// transpose src[64][128] -> dst[128][64]
__device__ __forceinline__ void transpose_64x128(const bf16* src, bf16* dst){
  for(int idx=threadIdx.x; idx<64*128; idx+=128){
    int r=idx>>7, c=idx&127;
    dst[c*64+r]=src[r*128+c];
  }
}

// ---------------- delta kernel: D = rowsum(dO*O) ----------------
__global__ void delta_kernel(const bf16* __restrict__ O, const bf16* __restrict__ dO,
                             float* __restrict__ Delta, long total){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x;
  if(idx>=total) return;
  const int4* o4=reinterpret_cast<const int4*>(O+idx*128);
  const int4* g4=reinterpret_cast<const int4*>(dO+idx*128);
  float acc=0.f;
  #pragma unroll
  for(int j=0;j<16;j++){
    int4 ov=o4[j], gv=g4[j];
    const bf16* op=reinterpret_cast<const bf16*>(&ov);
    const bf16* gp=reinterpret_cast<const bf16*>(&gv);
    #pragma unroll
    for(int k=0;k<8;k++) acc+=__bfloat162float(op[k])*__bfloat162float(gp[k]);
  }
  Delta[idx]=acc;
}

// ---------------- dK / dV kernel ----------------
// BN=128 (KV rows = M), BM=64 (Q tile), d=128
__global__ __launch_bounds__(128) void kv_kernel(
    const __grid_constant__ CUtensorMap tmK,
    const __grid_constant__ CUtensorMap tmV,
    const __grid_constant__ CUtensorMap tmQ,
    const __grid_constant__ CUtensorMap tmdO,
    const float* __restrict__ Lg, const float* __restrict__ Dg,
    bf16* __restrict__ dKo, bf16* __restrict__ dVo, int S, float scale)
{
  constexpr int BN=128, BM=64, D=128;
  extern __shared__ __align__(1024) char smem[];
  bf16* sK   = (bf16*)(smem);            // [128][128]
  bf16* sV   = sK + 128*128;             // [128][128]
  bf16* sQ   = sV + 128*128;             // [64][128]
  bf16* sdO  = sQ + 64*128;              // [64][128]
  bf16* sQT  = sdO + 64*128;             // [128][64]
  bf16* sdOT = sQT + 128*64;             // [128][64]
  bf16* sPt  = sdOT + 128*64;            // [128][64]
  bf16* sdSt = sPt + 128*64;             // [128][64]
  float* sL  = (float*)(sdSt + 128*64);  // [64]
  float* sD  = sL + 64;                  // [64]
  uint64_t* mbT=(uint64_t*)(sD+64);
  uint64_t* mbM=mbT+1;
  uint32_t* tmemp=(uint32_t*)(mbM+1);

  int tid=threadIdx.x;
  int bh=blockIdx.y, kv0=blockIdx.x*BN;
  const float* Lb=Lg+(size_t)bh*S;
  const float* Db=Dg+(size_t)bh*S;
  bf16* dKb=dKo+(size_t)bh*S*128;
  bf16* dVb=dVo+(size_t)bh*S*128;

  if(threadIdx.x<32) tmem_alloc(tmemp, 512);
  if(tid==0){ init_bar(mbT,1); init_bar(mbM,1); }
  fence_bar_init();
  __syncthreads();
  uint32_t tb=*tmemp;

  // regions (columns): S^T=0, dV=64, dP^T=192, dK=256
  const uint32_t R_S=tb+0, R_dV=tb+64, R_dP=tb+192, R_dK=tb+256;

  int pT=0,pM=0;
  // load K,V
  if(tid==0){ arrive_expect_tx(mbT,2*BN*D*2);
    tma_load(&tmK,mbT,sK,0,bh*S+kv0); tma_load(&tmV,mbT,sV,0,bh*S+kv0); }
  bar_wait(mbT,pT); pT^=1;

  int nQ=S/BM;
  for(int j=0;j<nQ;j++){
    int q0=j*BM;
    if(tid==0){ arrive_expect_tx(mbT,2*BM*D*2);
      tma_load(&tmQ,mbT,sQ,0,bh*S+q0); tma_load(&tmdO,mbT,sdO,0,bh*S+q0); }
    bar_wait(mbT,pT); pT^=1;
    for(int i=tid;i<BM;i+=128){ sL[i]=Lb[q0+i]; sD[i]=Db[q0+i]; }
    __syncthreads();
    transpose_64x128(sQ,sQT);
    transpose_64x128(sdO,sdOT);
    __syncthreads();
    fence_async();

    if(tid==0){
      uint32_t aK=cvta(sK),aQ=cvta(sQ),aV=cvta(sV),adO=cvta(sdO);
      uint32_t id1=mk_idesc(BN,BM);
      for(int k=0;k<8;k++)
        umma(R_S, mk_desc(aK+32*k,2048), mk_desc(aQ+32*k,2048), id1, k==0?0:1);
      for(int k=0;k<8;k++)
        umma(R_dP, mk_desc(aV+32*k,2048), mk_desc(adO+32*k,2048), id1, k==0?0:1);
      umma_commit(mbM);
    }
    bar_wait(mbM,pM); pM^=1;

    // elementwise: row bn=tid, col bm
    {
      for(int col=0;col<BM;col+=4){
        uint32_t s0,s1,s2,s3,p0,p1,p2,p3;
        tmem_ld_x4(R_S+col,&s0,&s1,&s2,&s3);
        tmem_ld_x4(R_dP+col,&p0,&p1,&p2,&p3);
        tmem_wait_ld();
        float Sv[4]={__uint_as_float(s0),__uint_as_float(s1),__uint_as_float(s2),__uint_as_float(s3)};
        float Pv[4]={__uint_as_float(p0),__uint_as_float(p1),__uint_as_float(p2),__uint_as_float(p3)};
        #pragma unroll
        for(int i=0;i<4;i++){
          float pt=__expf(scale*Sv[i]-sL[col+i]);
          float ds=pt*(Pv[i]-sD[col+i]);
          sPt [tid*BM+col+i]=__float2bfloat16(pt);
          sdSt[tid*BM+col+i]=__float2bfloat16(ds);
        }
      }
    }
    __syncthreads();
    fence_async();

    if(tid==0){
      uint32_t aPt=cvta(sPt),adOT=cvta(sdOT),aSt=cvta(sdSt),aQT=cvta(sQT);
      uint32_t id2=mk_idesc(BN,D);
      for(int k=0;k<4;k++)
        umma(R_dV, mk_desc(aPt+32*k,1024), mk_desc(adOT+32*k,1024), id2, (j==0&&k==0)?0:1);
      for(int k=0;k<4;k++)
        umma(R_dK, mk_desc(aSt+32*k,1024), mk_desc(aQT+32*k,1024), id2, (j==0&&k==0)?0:1);
      umma_commit(mbM);
    }
    bar_wait(mbM,pM); pM^=1;
    __syncthreads();
  }

  // epilogue
  int grow=kv0+tid;
  for(int col=0;col<128;col+=4){
    uint32_t r0,r1,r2,r3;
    tmem_ld_x4(R_dV+col,&r0,&r1,&r2,&r3); tmem_wait_ld();
    if(grow<S){
      bf16* g=dVb+(size_t)grow*128+col;
      g[0]=__float2bfloat16(__uint_as_float(r0));
      g[1]=__float2bfloat16(__uint_as_float(r1));
      g[2]=__float2bfloat16(__uint_as_float(r2));
      g[3]=__float2bfloat16(__uint_as_float(r3));
    }
  }
  for(int col=0;col<128;col+=4){
    uint32_t r0,r1,r2,r3;
    tmem_ld_x4(R_dK+col,&r0,&r1,&r2,&r3); tmem_wait_ld();
    if(grow<S){
      bf16* g=dKb+(size_t)grow*128+col;
      g[0]=__float2bfloat16(scale*__uint_as_float(r0));
      g[1]=__float2bfloat16(scale*__uint_as_float(r1));
      g[2]=__float2bfloat16(scale*__uint_as_float(r2));
      g[3]=__float2bfloat16(scale*__uint_as_float(r3));
    }
  }
  __syncthreads();
  if(threadIdx.x<32) tmem_dealloc(tb,512);
}

// ---------------- dQ kernel ----------------
// BM=128 (Q rows = M), BN=64 (KV tile), d=128
__global__ __launch_bounds__(128) void dq_kernel(
    const __grid_constant__ CUtensorMap tmQ,
    const __grid_constant__ CUtensorMap tmdO,
    const __grid_constant__ CUtensorMap tmK,
    const __grid_constant__ CUtensorMap tmV,
    const float* __restrict__ Lg, const float* __restrict__ Dg,
    bf16* __restrict__ dQo, int S, float scale)
{
  constexpr int BM=128, BN=64, D=128;
  extern __shared__ __align__(1024) char smem[];
  bf16* sQ  = (bf16*)(smem);          // [128][128]
  bf16* sdO = sQ + 128*128;           // [128][128]
  bf16* sK  = sdO + 128*128;          // [64][128]
  bf16* sV  = sK + 64*128;            // [64][128]
  bf16* sKT = sV + 64*128;            // [128][64]
  bf16* sdS = sKT + 128*64;           // [128][64]
  float* sL = (float*)(sdS + 128*64); // [128]
  float* sD = sL + 128;               // [128]
  uint64_t* mbT=(uint64_t*)(sD+128);
  uint64_t* mbM=mbT+1;
  uint32_t* tmemp=(uint32_t*)(mbM+1);

  int tid=threadIdx.x;
  int bh=blockIdx.y, q0=blockIdx.x*BM;
  const float* Lb=Lg+(size_t)bh*S;
  const float* Db=Dg+(size_t)bh*S;
  bf16* dQb=dQo+(size_t)bh*S*128;

  if(threadIdx.x<32) tmem_alloc(tmemp,256);
  if(tid==0){ init_bar(mbT,1); init_bar(mbM,1); }
  fence_bar_init();
  __syncthreads();
  uint32_t tb=*tmemp;
  const uint32_t R_S=tb+0, R_dQ=tb+64, R_dP=tb+192;

  int pT=0,pM=0;
  if(tid==0){ arrive_expect_tx(mbT,2*BM*D*2);
    tma_load(&tmQ,mbT,sQ,0,bh*S+q0); tma_load(&tmdO,mbT,sdO,0,bh*S+q0); }
  bar_wait(mbT,pT); pT^=1;
  for(int i=tid;i<BM;i+=128){ sL[i]=Lb[q0+i]; sD[i]=Db[q0+i]; }
  __syncthreads();

  int nK=S/BN;
  for(int kb=0;kb<nK;kb++){
    int kv0=kb*BN;
    if(tid==0){ arrive_expect_tx(mbT,2*BN*D*2);
      tma_load(&tmK,mbT,sK,0,bh*S+kv0); tma_load(&tmV,mbT,sV,0,bh*S+kv0); }
    bar_wait(mbT,pT); pT^=1;
    transpose_64x128(sK,sKT);
    __syncthreads();
    fence_async();

    if(tid==0){
      uint32_t aQ=cvta(sQ),aK=cvta(sK),adO=cvta(sdO),aV=cvta(sV);
      uint32_t id1=mk_idesc(BM,BN);
      for(int k=0;k<8;k++)
        umma(R_S, mk_desc(aQ+32*k,2048), mk_desc(aK+32*k,2048), id1, k==0?0:1);
      for(int k=0;k<8;k++)
        umma(R_dP, mk_desc(adO+32*k,2048), mk_desc(aV+32*k,2048), id1, k==0?0:1);
      umma_commit(mbM);
    }
    bar_wait(mbM,pM); pM^=1;

    // elementwise: row bm=tid, col bn
    {
      float l=sL[tid], dd=sD[tid];
      for(int col=0;col<BN;col+=4){
        uint32_t s0,s1,s2,s3,p0,p1,p2,p3;
        tmem_ld_x4(R_S+col,&s0,&s1,&s2,&s3);
        tmem_ld_x4(R_dP+col,&p0,&p1,&p2,&p3);
        tmem_wait_ld();
        float Sv[4]={__uint_as_float(s0),__uint_as_float(s1),__uint_as_float(s2),__uint_as_float(s3)};
        float Pv[4]={__uint_as_float(p0),__uint_as_float(p1),__uint_as_float(p2),__uint_as_float(p3)};
        #pragma unroll
        for(int i=0;i<4;i++){
          float pt=__expf(scale*Sv[i]-l);
          float ds=pt*(Pv[i]-dd);
          sdS[tid*BN+col+i]=__float2bfloat16(ds);
        }
      }
    }
    __syncthreads();
    fence_async();

    if(tid==0){
      uint32_t aS=cvta(sdS),aKT=cvta(sKT);
      uint32_t id2=mk_idesc(BM,D);
      for(int k=0;k<4;k++)
        umma(R_dQ, mk_desc(aS+32*k,1024), mk_desc(aKT+32*k,1024), id2, (kb==0&&k==0)?0:1);
      umma_commit(mbM);
    }
    bar_wait(mbM,pM); pM^=1;
    __syncthreads();
  }

  int grow=q0+tid;
  for(int col=0;col<128;col+=4){
    uint32_t r0,r1,r2,r3;
    tmem_ld_x4(R_dQ+col,&r0,&r1,&r2,&r3); tmem_wait_ld();
    if(grow<S){
      bf16* g=dQb+(size_t)grow*128+col;
      g[0]=__float2bfloat16(scale*__uint_as_float(r0));
      g[1]=__float2bfloat16(scale*__uint_as_float(r1));
      g[2]=__float2bfloat16(scale*__uint_as_float(r2));
      g[3]=__float2bfloat16(scale*__uint_as_float(r3));
    }
  }
  __syncthreads();
  if(threadIdx.x<32) tmem_dealloc(tb,256);
}

// ---------------- host ----------------
static CUresult make_tma(CUtensorMap* d, void* g, uint64_t inner, uint64_t outer,
                         uint32_t bi, uint32_t bo){
  uint64_t gdim[2]={inner,outer};
  uint64_t gstr[1]={inner*2};
  uint32_t bdim[2]={bi,bo};
  uint32_t estr[2]={1,1};
  return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, g, gdim, gstr, bdim, estr,
    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

static float* g_D=nullptr; static size_t g_Dsz=0;

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=Q.size(0), H=Q.size(1), S=Q.size(2);
  long BH=(long)B*H;
  float scale=1.0f/sqrtf(128.0f);

  bf16* Qp=(bf16*)Q.data_ptr();  bf16* Kp=(bf16*)K.data_ptr();  bf16* Vp=(bf16*)V.data_ptr();
  bf16* Op=(bf16*)O.data_ptr();  bf16* dOp=(bf16*)dO.data_ptr();
  float* Lp=(float*)L.data_ptr();
  bf16* dQp=(bf16*)dQ.data_ptr(); bf16* dKp=(bf16*)dK.data_ptr(); bf16* dVp=(bf16*)dV.data_ptr();

  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id);

  size_t need=sizeof(float)*BH*S;
  if(need>g_Dsz){ if(g_D) cudaFree(g_D); CUDA_CHECK(cudaMalloc(&g_D,need)); g_Dsz=need; }

  long total=BH*S;
  delta_kernel<<<(total+255)/256,256,0,stream>>>(Op,dOp,g_D,total);

  uint64_t inner=128, outer=(uint64_t)BH*S;
  CUtensorMap kvK,kvV,kvQ,kvdO, dqQ,dqdO,dqK,dqV;
  CU_CHECK(make_tma(&kvK ,Kp ,inner,outer,128,128));
  CU_CHECK(make_tma(&kvV ,Vp ,inner,outer,128,128));
  CU_CHECK(make_tma(&kvQ ,Qp ,inner,outer,128,64));
  CU_CHECK(make_tma(&kvdO,dOp,inner,outer,128,64));
  CU_CHECK(make_tma(&dqQ ,Qp ,inner,outer,128,128));
  CU_CHECK(make_tma(&dqdO,dOp,inner,outer,128,128));
  CU_CHECK(make_tma(&dqK ,Kp ,inner,outer,128,64));
  CU_CHECK(make_tma(&dqV ,Vp ,inner,outer,128,64));

  size_t smemKV=165000, smemDQ=133000;
  CUDA_CHECK(cudaFuncSetAttribute((const void*)kv_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemKV));
  CUDA_CHECK(cudaFuncSetAttribute((const void*)dq_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemDQ));

  dim3 gridKV(S/128, (unsigned)BH);
  dim3 gridDQ(S/128, (unsigned)BH);

  kv_kernel<<<gridKV,128,smemKV,stream>>>(kvK,kvV,kvQ,kvdO,Lp,g_D,dKp,dVp,S,scale);
  dq_kernel<<<gridDQ,128,smemDQ,stream>>>(dqQ,dqdO,dqK,dqV,Lp,g_D,dQp,S,scale);

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd