#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} }while(0)

namespace mha_bwd {
using bf16 = __nv_bfloat16;

// ---------------- tcgen05 helpers (cta_group::1) ----------------
__device__ __forceinline__ uint64_t make_desc(const void* p, uint32_t lbo, uint32_t sbo, uint32_t swz){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
  uint64_t d=0;
  d |= (uint64_t)((a>>4)&0x3FFF);
  d |= (uint64_t)(((lbo>>4)&0x3FFF))<<16;
  d |= (uint64_t)(((sbo>>4)&0x3FFF))<<32;
  d |= (uint64_t)1<<46;
  d |= (uint64_t)swz<<61;
  return d;
}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M,uint32_t N){
  uint32_t d=0;
  d|=(1u<<4);   // D=FP32
  d|=(1u<<7);   // A=BF16
  d|=(1u<<10);  // B=BF16
  d|=((N>>3)<<17);
  d|=((M>>4)<<24);
  return d;
}
__device__ __forceinline__ void umma1(uint32_t tmd,uint64_t da,uint64_t db,uint32_t idesc,uint32_t acc){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    ::"r"(tmd),"l"(da),"l"(db),"r"(idesc),"r"(acc):"memory");
}
__device__ __forceinline__ void umma_commit1(uint64_t* bar){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"::"r"(a):"memory");
}
__device__ __forceinline__ void tmem_alloc1(uint32_t* dst,int n){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(dst);
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(a),"r"(n));
}
__device__ __forceinline__ void tmem_dealloc1(uint32_t addr,int n){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(addr),"r"(n));
}
__device__ __forceinline__ void init_bar(uint64_t* bar,uint32_t c){
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c));
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar,int phase){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"
    ::"r"(a),"r"(phase):"memory");
}
__device__ __forceinline__ void fence_pa(){ asm volatile("fence.proxy.async;\n":::"memory"); }
__device__ __forceinline__ void ld4(uint32_t col,uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];"
    :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(col));
}
__device__ __forceinline__ void wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }

// ---------------- tiled loads (no-swizzle K-major core-matrix layout) ----------------
// layout offset(m,k,rows) = (k/8)*(rows*8) + (m/8)*64 + (m%8)*8 + (k%8)
__device__ __forceinline__ void load_norm(bf16* dst,const bf16* gbh,int row0,int rows,int S){
  int t=threadIdx.x; int total=rows*16;
  for(int i=t;i<total;i+=128){
    int r=i>>4, cc=i&15; int gr=row0+r;
    uint4 v; if(gr<S) v=*(const uint4*)(gbh+(size_t)gr*128+cc*8); else v=make_uint4(0,0,0,0);
    int base=cc*(rows*8)+(r>>3)*64+(r&7)*8;
    *(uint4*)(dst+base)=v;
  }
}
// transpose+tile: operand [rows=128(d)][K=ncols(seq)], stored = original[seq][d]
__device__ __forceinline__ void load_trans(bf16* dst,const bf16* gbh,int row0,int ncols,int S){
  int t=threadIdx.x; int total=ncols*16;
  for(int i=t;i<total;i+=128){
    int r=i>>4, cc=i&15; int gr=row0+r;
    uint4 v; if(gr<S) v=*(const uint4*)(gbh+(size_t)gr*128+cc*8); else v=make_uint4(0,0,0,0);
    bf16* pv=(bf16*)&v;
    int base=(r>>3)*1024 + cc*64 + (r&7);
    #pragma unroll
    for(int j=0;j<8;j++) dst[base+j*8]=pv[j];
  }
}

// ---------------- delta ----------------
__global__ void compute_delta(const bf16* O,const bf16* dO,float* Dg,long long rows){
  int warp=threadIdx.x>>5, lane=threadIdx.x&31;
  long long row=(long long)blockIdx.x*(blockDim.x>>5)+warp;
  if(row>=rows) return;
  const bf16* o=O+row*128; const bf16* g=dO+row*128;
  float s=0.f;
  #pragma unroll
  for(int k=lane;k<128;k+=32) s+=__bfloat162float(o[k])*__bfloat162float(g[k]);
  #pragma unroll
  for(int off=16;off>0;off>>=1) s+=__shfl_down_sync(0xffffffffu,s,off);
  if(lane==0) Dg[row]=s;
}

// ---------------- dK/dV kernel ----------------
// BN(kv)=128 rows, BM(q)=64 tile
__global__ __launch_bounds__(128,1) void bwd_dkdv(
  const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
  const float* Lg,const float* Dg,bf16* dK,bf16* dV,int S,float scale){
  extern __shared__ char smem[];
  bf16* Knorm =(bf16*)(smem+0);       // [128][128]
  bf16* Vnorm =(bf16*)(smem+32768);   // [128][128]
  bf16* Qnorm =(bf16*)(smem+65536);   // [64][128]
  bf16* dOnorm=(bf16*)(smem+81920);   // [64][128]
  bf16* QT    =(bf16*)(smem+98304);   // [128(d)][64(q)]
  bf16* dOT   =(bf16*)(smem+114688);  // [128(d)][64(q)]
  bf16* PTsm  =(bf16*)(smem+131072);  // [128(kv)][64(q)]
  bf16* dSTsm =(bf16*)(smem+147456);  // [128(kv)][64(q)]
  float* Lsm  =(float*)(smem+163840);
  float* Dsm  =(float*)(smem+164096);
  __shared__ __align__(8) uint64_t bar;
  __shared__ uint32_t tmbase_s;

  int t=threadIdx.x, bh=blockIdx.y, kv0=blockIdx.x*128;
  const bf16* Kbh=K+(size_t)bh*S*128; const bf16* Vbh=V+(size_t)bh*S*128;
  const bf16* Qbh=Q+(size_t)bh*S*128; const bf16* dObh=dO+(size_t)bh*S*128;
  const float* Lbh=Lg+(size_t)bh*S; const float* Dbh=Dg+(size_t)bh*S;

  if(t<32) tmem_alloc1(&tmbase_s,512);
  if(t==0) init_bar(&bar,1);
  __syncthreads();
  uint32_t tb=tmbase_s; int phase=0;
  const uint32_t DV_OFF=0,DK_OFF=128,ST_OFF=256,DPT_OFF=320;
  uint32_t idS=make_idesc(128,64), idG=make_idesc(128,128);

  load_norm(Knorm,Kbh,kv0,128,S);
  load_norm(Vnorm,Vbh,kv0,128,S);
  __syncthreads();

  int nq=(S+63)/64;
  for(int qt=0;qt<nq;qt++){
    int q0=qt*64;
    load_norm(Qnorm,Qbh,q0,64,S);
    load_norm(dOnorm,dObh,q0,64,S);
    load_trans(QT,Qbh,q0,64,S);
    load_trans(dOT,dObh,q0,64,S);
    for(int i=t;i<64;i+=128){ int gq=q0+i; Lsm[i]=(gq<S)?Lbh[gq]:0.f; Dsm[i]=(gq<S)?Dbh[gq]:0.f; }
    __syncthreads();
    fence_pa();
    __syncthreads();
    if(t==0){
      #pragma unroll
      for(int k=0;k<8;k++){
        uint64_t da=make_desc(Knorm+k*(128*16),128*16,128,0);
        uint64_t db=make_desc(Qnorm+k*(64*16),64*16,128,0);
        umma1(tb+ST_OFF,da,db,idS,k>0?1u:0u);
      }
      #pragma unroll
      for(int k=0;k<8;k++){
        uint64_t da=make_desc(Vnorm+k*(128*16),128*16,128,0);
        uint64_t db=make_desc(dOnorm+k*(64*16),64*16,128,0);
        umma1(tb+DPT_OFF,da,db,idS,k>0?1u:0u);
      }
      umma_commit1(&bar);
    }
    mbar_wait(&bar,phase); phase^=1;

    for(int c=0;c<64;c+=4){
      uint32_t s0,s1,s2,s3,p0,p1,p2,p3;
      ld4(tb+ST_OFF+c,s0,s1,s2,s3);
      ld4(tb+DPT_OFF+c,p0,p1,p2,p3);
      wait_ld();
      float sc[4]={__uint_as_float(s0),__uint_as_float(s1),__uint_as_float(s2),__uint_as_float(s3)};
      float pc[4]={__uint_as_float(p0),__uint_as_float(p1),__uint_as_float(p2),__uint_as_float(p3)};
      #pragma unroll
      for(int i=0;i<4;i++){
        int q=c+i;
        float pt=__expf(scale*sc[i]-Lsm[q]);
        float ds=scale*pt*(pc[i]-Dsm[q]);
        int off=(q>>3)*1024+(t>>3)*64+(t&7)*8+(q&7);
        PTsm[off]=__float2bfloat16(pt);
        dSTsm[off]=__float2bfloat16(ds);
      }
    }
    __syncthreads();
    fence_pa();
    __syncthreads();
    if(t==0){
      #pragma unroll
      for(int k=0;k<4;k++){
        uint64_t da=make_desc(PTsm+k*(128*16),128*16,128,0);
        uint64_t db=make_desc(dOT+k*(128*16),128*16,128,0);
        umma1(tb+DV_OFF,da,db,idG,(qt==0&&k==0)?0u:1u);
      }
      #pragma unroll
      for(int k=0;k<4;k++){
        uint64_t da=make_desc(dSTsm+k*(128*16),128*16,128,0);
        uint64_t db=make_desc(QT+k*(128*16),128*16,128,0);
        umma1(tb+DK_OFF,da,db,idG,(qt==0&&k==0)?0u:1u);
      }
      umma_commit1(&bar);
    }
    mbar_wait(&bar,phase); phase^=1;
    __syncthreads();
  }

  int gr=kv0+t;
  for(int c=0;c<128;c+=4){
    uint32_t r0,r1,r2,r3; ld4(tb+DV_OFF+c,r0,r1,r2,r3); wait_ld();
    if(gr<S){ bf16* o=dV+(size_t)bh*S*128+(size_t)gr*128+c;
      o[0]=__float2bfloat16(__uint_as_float(r0)); o[1]=__float2bfloat16(__uint_as_float(r1));
      o[2]=__float2bfloat16(__uint_as_float(r2)); o[3]=__float2bfloat16(__uint_as_float(r3)); }
  }
  for(int c=0;c<128;c+=4){
    uint32_t r0,r1,r2,r3; ld4(tb+DK_OFF+c,r0,r1,r2,r3); wait_ld();
    if(gr<S){ bf16* o=dK+(size_t)bh*S*128+(size_t)gr*128+c;
      o[0]=__float2bfloat16(__uint_as_float(r0)); o[1]=__float2bfloat16(__uint_as_float(r1));
      o[2]=__float2bfloat16(__uint_as_float(r2)); o[3]=__float2bfloat16(__uint_as_float(r3)); }
  }
  __syncthreads();
  if(t<32) tmem_dealloc1(tb,512);
}

// ---------------- dQ kernel ----------------
// BQ(q)=128 rows, BKV(kv)=64 tile
__global__ __launch_bounds__(128,1) void bwd_dq(
  const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
  const float* Lg,const float* Dg,bf16* dQ,int S,float scale){
  extern __shared__ char smem[];
  bf16* Qnorm =(bf16*)(smem+0);       // [128][128]
  bf16* dOnorm=(bf16*)(smem+32768);   // [128][128]
  bf16* Knorm =(bf16*)(smem+65536);   // [64][128]
  bf16* Vnorm =(bf16*)(smem+81920);   // [64][128]
  bf16* KT    =(bf16*)(smem+98304);   // [128(d)][64(kv)]
  bf16* dSsm  =(bf16*)(smem+114688);  // [128(q)][64(kv)]
  float* Lsm  =(float*)(smem+131072);
  float* Dsm  =(float*)(smem+131584);
  __shared__ __align__(8) uint64_t bar;
  __shared__ uint32_t tmbase_s;

  int t=threadIdx.x, bh=blockIdx.y, q0=blockIdx.x*128;
  const bf16* Kbh=K+(size_t)bh*S*128; const bf16* Vbh=V+(size_t)bh*S*128;
  const bf16* Qbh=Q+(size_t)bh*S*128; const bf16* dObh=dO+(size_t)bh*S*128;
  const float* Lbh=Lg+(size_t)bh*S; const float* Dbh=Dg+(size_t)bh*S;

  if(t<32) tmem_alloc1(&tmbase_s,256);
  if(t==0) init_bar(&bar,1);
  __syncthreads();
  uint32_t tb=tmbase_s; int phase=0;
  const uint32_t DQ_OFF=0,S_OFF=128,DP_OFF=192;
  uint32_t idS=make_idesc(128,64), idG=make_idesc(128,128);

  load_norm(Qnorm,Qbh,q0,128,S);
  load_norm(dOnorm,dObh,q0,128,S);
  for(int i=t;i<128;i+=128){ int gq=q0+i; Lsm[i]=(gq<S)?Lbh[gq]:0.f; Dsm[i]=(gq<S)?Dbh[gq]:0.f; }
  __syncthreads();

  int nkv=(S+63)/64;
  for(int kt=0;kt<nkv;kt++){
    int kv0=kt*64;
    load_norm(Knorm,Kbh,kv0,64,S);
    load_norm(Vnorm,Vbh,kv0,64,S);
    load_trans(KT,Kbh,kv0,64,S);
    __syncthreads();
    fence_pa();
    __syncthreads();
    if(t==0){
      #pragma unroll
      for(int k=0;k<8;k++){
        uint64_t da=make_desc(Qnorm+k*(128*16),128*16,128,0);
        uint64_t db=make_desc(Knorm+k*(64*16),64*16,128,0);
        umma1(tb+S_OFF,da,db,idS,k>0?1u:0u);
      }
      #pragma unroll
      for(int k=0;k<8;k++){
        uint64_t da=make_desc(dOnorm+k*(128*16),128*16,128,0);
        uint64_t db=make_desc(Vnorm+k*(64*16),64*16,128,0);
        umma1(tb+DP_OFF,da,db,idS,k>0?1u:0u);
      }
      umma_commit1(&bar);
    }
    mbar_wait(&bar,phase); phase^=1;

    float L=Lsm[t], D=Dsm[t];
    for(int c=0;c<64;c+=4){
      uint32_t s0,s1,s2,s3,p0,p1,p2,p3;
      ld4(tb+S_OFF+c,s0,s1,s2,s3);
      ld4(tb+DP_OFF+c,p0,p1,p2,p3);
      wait_ld();
      float sc[4]={__uint_as_float(s0),__uint_as_float(s1),__uint_as_float(s2),__uint_as_float(s3)};
      float pc[4]={__uint_as_float(p0),__uint_as_float(p1),__uint_as_float(p2),__uint_as_float(p3)};
      #pragma unroll
      for(int i=0;i<4;i++){
        int kv=c+i;
        float p=__expf(scale*sc[i]-L);
        float ds=scale*p*(pc[i]-D);
        int off=(kv>>3)*1024+(t>>3)*64+(t&7)*8+(kv&7);
        dSsm[off]=__float2bfloat16(ds);
      }
    }
    __syncthreads();
    fence_pa();
    __syncthreads();
    if(t==0){
      #pragma unroll
      for(int k=0;k<4;k++){
        uint64_t da=make_desc(dSsm+k*(128*16),128*16,128,0);
        uint64_t db=make_desc(KT+k*(128*16),128*16,128,0);
        umma1(tb+DQ_OFF,da,db,idG,(kt==0&&k==0)?0u:1u);
      }
      umma_commit1(&bar);
    }
    mbar_wait(&bar,phase); phase^=1;
    __syncthreads();
  }

  int gq=q0+t;
  for(int c=0;c<128;c+=4){
    uint32_t r0,r1,r2,r3; ld4(tb+DQ_OFF+c,r0,r1,r2,r3); wait_ld();
    if(gq<S){ bf16* o=dQ+(size_t)bh*S*128+(size_t)gq*128+c;
      o[0]=__float2bfloat16(__uint_as_float(r0)); o[1]=__float2bfloat16(__uint_as_float(r1));
      o[2]=__float2bfloat16(__uint_as_float(r2)); o[3]=__float2bfloat16(__uint_as_float(r3)); }
  }
  __syncthreads();
  if(t<32) tmem_dealloc1(tb,256);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int64_t B=Q.size(0),H=Q.size(1),S=Q.size(2);
  int BH=