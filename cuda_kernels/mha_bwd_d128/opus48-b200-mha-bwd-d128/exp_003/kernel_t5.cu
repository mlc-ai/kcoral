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
static const float L2E=1.4426950408889634f;

__device__ __forceinline__ float fast_exp2f(float x){ float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y; }
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
__device__ __forceinline__ uint32_t make_idesc(uint32_t M,uint32_t N,uint32_t am,uint32_t bm){
  uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10);
  d|=(am<<15); d|=(bm<<16); d|=((N>>3)<<17); d|=((M>>4)<<24); return d;
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
__device__ __forceinline__ void cpasync16(void* d, const void* s, bool pred){
  uint32_t sd=(uint32_t)__cvta_generic_to_shared(d); int sz=pred?16:0;
  asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;\n"::"r"(sd),"l"(s),"r"(sz):"memory");
}
__device__ __forceinline__ void cpasync_commit(){ asm volatile("cp.async.commit_group;\n":::"memory"); }
__device__ __forceinline__ void cpasync_wait0(){ asm volatile("cp.async.wait_group 0;\n":::"memory"); }
__device__ __forceinline__ void cpasync_wait1(){ asm volatile("cp.async.wait_group 1;\n":::"memory"); }
__device__ __forceinline__ void store8(bf16* dst,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,
                                       uint32_t b0,uint32_t b1,uint32_t b2,uint32_t b3){
  bf16 o[8];
  o[0]=__float2bfloat16(__uint_as_float(a0)); o[1]=__float2bfloat16(__uint_as_float(a1));
  o[2]=__float2bfloat16(__uint_as_float(a2)); o[3]=__float2bfloat16(__uint_as_float(a3));
  o[4]=__float2bfloat16(__uint_as_float(b0)); o[5]=__float2bfloat16(__uint_as_float(b1));
  o[6]=__float2bfloat16(__uint_as_float(b2)); o[7]=__float2bfloat16(__uint_as_float(b3));
  *(uint4*)dst=*(const uint4*)o;
}
// core-matrix K-major(K=d) layout, 256 threads
__device__ __forceinline__ void ld_tile(bf16* dst,const bf16* gbh,int row0,int rows,int S){
  for(int i=threadIdx.x;i<rows*16;i+=256){
    int r=i>>4, cc=i&15; int gr=row0+r; bool ok=gr<S; int gg=ok?gr:0;
    int base=cc*(rows*8)+(r>>3)*64+(r&7)*8;
    cpasync16(dst+base, gbh+(size_t)gg*128+cc*8, ok);
  }
}

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

// ================= dK/dV (kv tile=128, q tile=64) =================
__global__ __launch_bounds__(256,1) void bwd_dkdv(
  const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
  const float* Lg,const float* Dg,bf16* dK,bf16* dV,int S,float scale){
  extern __shared__ char smem[];
  bf16* Qn[3]={(bf16*)(smem+0),(bf16*)(smem+16384),(bf16*)(smem+32768)};
  bf16* dOn[3]={(bf16*)(smem+49152),(bf16*)(smem+65536),(bf16*)(smem+81920)};
  bf16* Kb=(bf16*)(smem+98304);
  bf16* Vb=(bf16*)(smem+131072);
  bf16* PT=(bf16*)(smem+163840);
  bf16* dS=(bf16*)(smem+180224);
  float* Lsm[3]={(float*)(smem+196608),(float*)(smem+196864),(float*)(smem+197120)};
  float* Dsm[3]={(float*)(smem+197376),(float*)(smem+197632),(float*)(smem+197888)};
  __shared__ __align__(8) uint64_t bar1,bar2;
  __shared__ uint32_t tmbase_s;

  int tid=threadIdx.x, bh=blockIdx.y, kv0=blockIdx.x*128;
  const bf16* Kbh=K+(size_t)bh*S*128; const bf16* Vbh=V+(size_t)bh*S*128;
  const bf16* Qbh=Q+(size_t)bh*S*128; const bf16* dObh=dO+(size_t)bh*S*128;
  const float* Lbh=Lg+(size_t)bh*S; const float* Dbh=Dg+(size_t)bh*S;
  float scale_l2=scale*L2E;

  if(tid<32) tmem_alloc1(&tmbase_s,512);
  if(tid==0){ init_bar(&bar1,1); init_bar(&bar2,1); }
  __syncthreads();
  uint32_t tb=tmbase_s; int p1=0,p2=0;
  uint32_t idS=make_idesc(128,64,0,0), idG=make_idesc(128,128,0,1);
  int nq=(S+63)/64;

  ld_tile(Kb,Kbh,kv0,128,S);
  ld_tile(Vb,Vbh,kv0,128,S);
  ld_tile(Qn[0],Qbh,0,64,S);
  ld_tile(dOn[0],dObh,0,64,S);
  for(int i=tid;i<64;i+=256){ Lsm[0][i]=(i<S)?Lbh[i]*L2E:0.f; Dsm[0][i]=(i<S)?Dbh[i]:0.f; }
  cpasync_commit();
  if(nq>1){
    ld_tile(Qn[1],Qbh,64,64,S); ld_tile(dOn[1],dObh,64,64,S);
    for(int i=tid;i<64;i+=256){ int gr=64+i; Lsm[1][i]=(gr<S)?Lbh[gr]*L2E:0.f; Dsm[1][i]=(gr<S)?Dbh[gr]:0.f; }
    cpasync_commit();
    cpasync_wait1();
  } else cpasync_wait0();
  __syncthreads(); fence_pa();
  if(tid==0){
    #pragma unroll
    for(int k=0;k<8;k++) umma1(tb+256, make_desc(Kb+k*2048,2048,128,0), make_desc(Qn[0]+k*1024,1024,128,0), idS, k>0);
    #pragma unroll
    for(int k=0;k<8;k++) umma1(tb+384, make_desc(Vb+k*2048,2048,128,0), make_desc(dOn[0]+k*1024,1024,128,0), idS, k>0);
    umma_commit1(&bar1);
  }

  for(int i=0;i<nq;i++){
    int qr=i%3, stb=i&1;
    int nqr=(i+1)%3, nstb=(i+1)&1;
    int q0i=i*64;
    mbar_wait(&bar1,p1); p1^=1;
    if(i+1<nq){
      cpasync_wait0(); __syncthreads(); fence_pa();
      if(tid==0){
        uint32_t sto=256+nstb*64, dpto=384+nstb*64;
        #pragma unroll
        for(int k=0;k<8;k++) umma1(tb+sto, make_desc(Kb+k*2048,2048,128,0), make_desc(Qn[nqr]+k*1024,1024,128,0), idS, k>0);
        #pragma unroll
        for(int k=0;k<8;k++) umma1(tb+dpto, make_desc(Vb+k*2048,2048,128,0), make_desc(dOn[nqr]+k*1024,1024,128,0), idS, k>0);
        umma_commit1(&bar1);
      }
    }
    if(i>0){ mbar_wait(&bar2,p2); p2^=1; }
    if(i+2<nq){
      int lr=(i+2)%3, q0=(i+2)*64;
      ld_tile(Qn[lr],Qbh,q0,64,S); ld_tile(dOn[lr],dObh,q0,64,S);
      for(int j=tid;j<64;j+=256){ int gr=q0+j; Lsm[lr][j]=(gr<S)?Lbh[gr]*L2E:0.f; Dsm[lr][j]=(gr<S)?Dbh[gr]:0.f; }
      cpasync_commit();
    }
    { // softmax(i)
      int wg=tid>>7, row=tid&127, qc0=wg*32;
      uint32_t sto=256+stb*64, dpto=384+stb*64;
      float* Lb=Lsm[qr]; float* Db=Dsm[qr];
      for(int c=qc0;c<qc0+32;c+=4){
        uint32_t sr[4],pr[4];
        ld4(tb+sto+c,sr[0],sr[1],sr[2],sr[3]);
        ld4(tb+dpto+c,pr[0],pr[1],pr[2],pr[3]);
        wait_ld();
        #pragma unroll
        for(int idx=0;idx<4;idx++){
          int q=c+idx; int gq=q0i+q;
          float ss=__uint_as_float(sr[idx]), pp=__uint_as_float(pr[idx]);
          float pt=(gq<S)?fast_exp2f(scale_l2*ss - Lb[q]):0.f;
          float ds=(gq<S)?scale*pt*(pp-Db[q]):0.f;
          int off=(q>>3)*1024+(row>>3)*64+(row&7)*8+(q&7);
          PT[off]=__float2bfloat16(pt);
          dS[off]=__float2bfloat16(ds);
        }
      }
    }
    __syncthreads(); fence_pa();
    if(tid==0){
      #pragma unroll
      for(int k=0;k<4;k++) umma1(tb+0,   make_desc(PT+k*2048,2048,128,0), make_desc(dOn[qr]+k*128,128,1024,0), idG, (i==0&&k==0)?0u:1u);
      #pragma unroll
      for(int k=0;k<4;k++) umma1(tb+128, make_desc(dS+k*2048,2048,128,0), make_desc(Qn[qr]+k*128,128,1024,0), idG, (i==0&&k==0)?0u:1u);
      umma_commit1(&bar2);
    }
  }
  mbar_wait(&bar2,p2); p2^=1;
  __syncthreads();

  if(tid<128){
    int gr=kv0+tid;
    for(int c=0;c<128;c+=8){
      uint32_t a0,a1,a2,a3,b0,b1,b2,b3;
      ld4(tb+0+c,a0,a1,a2,a3); ld4(tb+0+c+4,b0,b1,b2,b3); wait_ld();
      if(gr<S) store8(dV+(size_t)bh*S*128+(size_t)gr*128+c,a0,a1,a2,a3,b0,b1,b2,b3);
    }
    for(int c=0;c<128;c+=8){
      uint32_t a0,a1,a2,a3,b0,b1,b2,b3;
      ld4(tb+128+c,a0,a1,a2,a3); ld4(tb+128+c+4,b0,b1,b2,b3); wait_ld();
      if(gr<S) store8(dK+(size_t)bh*S*128+(size_t)gr*128+c,a0,a1,a2,a3,b0,b1,b2,b3);
    }
  }
  __syncthreads();
  if(tid<32) tmem_dealloc1(tb,512);
}

// ================= dQ (q tile=128, kv tile=64) =================
__global__ __launch_bounds__(256,1) void bwd_dq(
  const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
  const float* Lg,const float* Dg,bf16* dQ,int S,float scale){
  extern __shared__ char smem[];
  bf16* Qn =(bf16*)(smem+0);
  bf16* dOn=(bf16*)(smem+32768);
  bf16* Kb[3]={(bf16*)(smem+65536),(bf16*)(smem+81920),(bf16*)(smem+98304)};
  bf16* Vb[3]={(bf16*)(smem+114688),(bf16*)(smem+131072),(bf16*)(smem+147456)};
  bf16* dS=(bf16*)(smem+163840);
  float* Lsm=(float*)(smem+180224);
  float* Dsm=(float*)(smem+180736);
  __shared__ __align__(8) uint64_t bar1,bar2;
  __shared__ uint32_t tmbase_s;

  int tid=threadIdx.x, bh=blockIdx.y, q0=blockIdx.x*128;
  const bf16* Kbh=K+(size_t)bh*S*128; const bf16* Vbh=V+(size_t)bh*S*128;
  const bf16* Qbh=Q+(size_t)bh*S*128; const bf16* dObh=dO+(size_t)bh*S*128;
  const float* Lbh=Lg+(size_t)bh*S; const float* Dbh=Dg+(size_t)bh*S;
  float scale_l2=scale*L2E;

  if(tid<32) tmem_alloc1(&tmbase_s,512);
  if(tid==0){ init_bar(&bar1,1); init_bar(&bar2,1); }
  __syncthreads();
  uint32_t tb=tmbase_s; int p1=0,p2=0;
  uint32_t idS=make_idesc(128,64,0,0), idG=make_idesc(128,128,0,1);
  int nkv=(S+63)/64;

  ld_tile(Qn,Qbh,q0,128,S);
  ld_tile(dOn,dObh,q0,128,S);
  for(int j=tid;j<128;j+=256){ int gr=q0+j; Lsm[j]=(gr<S)?Lbh[gr]*L2E:0.f; Dsm[j]=(gr<S)?Dbh[gr]:0.f; }
  ld_tile(Kb[0],Kbh,0,64,S);
  ld_tile(Vb[0],Vbh,0,64,S);
  cpasync_commit();
  if(nkv>1){
    ld_tile(Kb[1],Kbh,64,64,S); ld_tile(Vb[1],Vbh,64,64,S);
    cpasync_commit(); cpasync_wait1();
  } else cpasync_wait0();
  __syncthreads(); fence_pa();
  if(tid==0){
    #pragma unroll
    for(int k=0;k<8;k++) umma1(tb+128, make_desc(Qn+k*2048,2048,128,0), make_desc(Kb[0]+k*1024,1024,128,0), idS, k>0);
    #pragma unroll
    for(int k=0;k<8;k++) umma1(tb+256, make_desc(dOn+k*2048,2048,128,0), make_desc(Vb[0]+k*1024,1024,128,0), idS, k>0);
    umma_commit1(&bar1);
  }

  for(int i=0;i<nkv;i++){
    int kr=i%3, sb=i&1;
    int nkr=(i+1)%3, nsb=(i+1)&1;
    int kv0i=i*64;
    mbar_wait(&bar1,p1); p1^=1;
    if(i+1<nkv){
      cpasync_wait0(); __syncthreads(); fence_pa();
      if(tid==0){
        uint32_t so=128+nsb*64, dpo=256+nsb*64;
        #pragma unroll
        for(int k=0;k<8;k++) umma1(tb+so, make_desc(Qn+k*2048,2048,128,0), make_desc(Kb[nkr]+k*1024,1024,128,0), idS, k>0);
        #pragma unroll
        for(int k=0;k<8;k++) umma1(tb+dpo, make_desc(dOn+k*2048,2048,128,0), make_desc(Vb[nkr]+k*1024,1024,128,0), idS, k>0);
        umma_commit1(&bar1);
      }
    }
    if(i>0){ mbar_wait(&bar2,p2); p2^=1; }
    if(i+2<nkv){
      int lr=(i+2)%3, kv0=(i+2)*64;
      ld_tile(Kb[lr],Kbh,kv0,64,S); ld_tile(Vb[lr],Vbh,kv0,64,S);
      cpasync_commit();
    }
    { // softmax(i)
      int wg=tid>>7, row=tid&127, kc0=wg*32;
      uint32_t so=128+sb*64, dpo=256+sb*64;
      float L=Lsm[row], D=Dsm[row];
      for(int c=kc0;c<kc0+32;c+=4){
        uint32_t sr[4],pr[4];
        ld4(tb+so+c,sr[0],sr[1],sr[2],sr[3]);
        ld4(tb+dpo+c,pr[0],pr[1],pr[2],pr[3]);
        wait_ld();
        #pragma unroll
        for(int idx=0;idx<4;idx++){
          int kv=c+idx; int gkv=kv0i+kv;
          float ss=__uint_as_float(sr[idx]), pp=__uint_as_float(pr[idx]);
          float pt=(gkv<S)?fast_exp2f(scale_l2*ss - L):0.f;
          float ds=(gkv<S)?scale*pt*(pp-D):0.f;
          int off=(kv>>3)*1024+(row>>3)*64+(row&7)*8+(kv&7);
          dS[off]=__float2bfloat16(ds);
        }
      }
    }
    __syncthreads(); fence_pa();
    if(tid==0){
      #pragma unroll
      for(int k=0;k<4;k++) umma1(tb+0, make_desc(dS+k*2048,2048,128,0), make_desc(Kb[kr]+k*128,128,1024,0), idG, (i==0&&k==0)?0u:1u);
      umma_commit1(&bar2);
    }
  }
  mbar_wait(&bar2,p2); p2^=1;
  __syncthreads();

  if(tid<128){
    int gq=q0+tid;
    for(int c=0;c<128;c+=8){
      uint32_t a0,a1,a2,a3,b0,b1,b2,b3;
      ld4(tb+0+c,a0,a1,a2,a3); ld4(tb+0+c+4,b0,b1,b2,b3); wait_ld();
      if(gq<S) store8(dQ+(size_t)bh*S*128+(size_t)gq*128+c,a0,a1,a2,a3,b0,b1,b2,b3);
    }
  }
  __syncthreads();
  if(tid<32) tmem_dealloc1(tb,512);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int64_t B=Q.size(0),H=Q.size(1),S=Q.size(2);
  int BH=(int)(B*H);
  float scale=1.0f/sqrtf(128.0f);

  const bf16* Qp=static_cast<const bf16*>(Q.data_ptr());
  const bf16* Kp=static_cast<const bf16*>(K.data_ptr());
  const bf16* Vp=static_cast<const bf16*>(V.data_ptr());
  const bf16* Op=static_cast<const bf16*>(O.data_ptr());
  const bf16* dOp=static_cast<const bf16*>(dO.data_ptr());
  const float* Lp=static_cast<const float*>(L.data_ptr());
  bf16* dQp=static_cast<bf16*>(dQ.data_ptr());
  bf16* dKp=static_cast<bf16*>(dK.data_ptr());
  bf16* dVp=static_cast<bf16*>(dV.data_ptr());

  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));

  long long rows=(long long)BH*S;
  float* Dg=nullptr;
  CUDA_CHECK(cudaMallocAsync(&Dg,sizeof(float)*rows,stream));

  {
    int threads=128; long long rpb=threads/32;
    long long blocks=(rows+rpb-1)/rpb;
    compute_delta<<<(unsigned)blocks,threads,0,stream>>>(Op,dOp,Dg,rows);
    CUDA_CHECK(cudaGetLastError());
  }

  const int DKDV_SMEM=198400;
  const int DQ_SMEM=181504;
  static bool attr_set=false;
  if(!attr_set){
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dkdv,cudaFuncAttributeMaxDynamicSharedMemorySize,DKDV_SMEM));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dq,  cudaFuncAttributeMaxDynamicSharedMemorySize,DQ_SMEM));
    attr_set=true;
  }

  int nkv_blocks=(int)((S+127)/128);
  int nq_blocks =(int)((S+127)/128);
  dim3 g1((unsigned)nkv_blocks,(unsigned)BH);
  dim3 g2((unsigned)nq_blocks,(unsigned)BH);

  bwd_dkdv<<<g1,256,DKDV_SMEM,stream>>>(Qp,Kp,Vp,dOp,Lp,Dg,dKp,dVp,(int)S,scale);
  CUDA_CHECK(cudaGetLastError());
  bwd_dq<<<g2,256,DQ_SMEM,stream>>>(Qp,Kp,Vp,dOp,Lp,Dg,dQp,(int)S,scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFreeAsync(Dg,stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd