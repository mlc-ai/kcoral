#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd {
typedef __nv_bfloat16 bf16;

static const int SMEM_BYTES = 200704;

__device__ __forceinline__ void init_bar(uint64_t* b,uint32_t c){
  asm volatile("mbarrier.init.shared.b64 [%0],%1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c));
}
__device__ __forceinline__ void bar_wait(uint64_t* b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph));
}
__device__ __forceinline__ void alloc_cg1(uint32_t* d,int n){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(d)),"r"(n));
}
__device__ __forceinline__ void dealloc_cg1(uint32_t a,int n){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"::"r"(a),"r"(n));
}
__device__ __forceinline__ void relinquish_cg1(){
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void commit_cg1(uint64_t* b){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)));
}
__device__ __forceinline__ void umma(uint32_t tm,uint64_t da,uint64_t db,uint32_t id,uint32_t acc){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    ::"r"(tm),"l"(da),"l"(db),"r"(id),"r"(acc));
}
__device__ __forceinline__ void ld4nw(uint32_t col,uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];"
    :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(col));
}
__device__ __forceinline__ void ldfence(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void fproxy(){ asm volatile("fence.proxy.async;":::"memory"); }

__device__ __forceinline__ uint64_t mkdesc(void* p){
  uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
  d |= (uint64_t)((a&0x3FFFF)>>4);
  d |= (uint64_t)0<<16;
  d |= (uint64_t)((1024&0x3FFFF)>>4)<<32;
  d |= (uint64_t)1<<46;
  d |= (uint64_t)2<<61;
  return d;
}
__device__ __forceinline__ uint32_t mkid(uint32_t M,uint32_t N,uint32_t am,uint32_t bm){
  uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10);
  d|=(am<<15); d|=(bm<<16); d|=((N/8)<<17); d|=((M/16)<<24);
  return d;
}

__global__ void compute_D_kernel(const bf16* dO,const bf16* O,float* D,long total){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total){ long b=idx*128; float a=0.f;
    #pragma unroll
    for(int c=0;c<128;c++) a+=(float)dO[b+c]*(float)O[b+c];
    D[idx]=a; }
}

__device__ __forceinline__ void loadtile(bf16* sX,const bf16* Xb,int seqbase,int S){
  int tid=threadIdx.x;
  #pragma unroll
  for(int e=tid;e<128*16;e+=128){
    int r=e>>4; int ch=e&15; int panel=ch>>3; int chunk=ch&7;
    int gr=seqbase+r;
    int4 v; if(gr<S) v=((const int4*)(Xb+(long)gr*128))[ch]; else v=make_int4(0,0,0,0);
    int off=panel*8192 + r*64 + (((r&7)^chunk)<<3);
    ((int4*)(sX+off))[0]=v;
  }
}
__device__ __forceinline__ int swz(int row,int col){
  int panel=col>>6, c=col&63; return panel*8192 + row*64 + (((row&7)^(c>>3))<<3) + (c&7);
}

// ============ dK,dV kernel (KV outer) ============
__global__ __launch_bounds__(128) void dkdv_kernel(
  const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
  const float* Lp,const float* Dp, bf16* dKo, bf16* dVo,int S,int H,float scale)
{
  extern __shared__ char smem[];
  bf16* sK=(bf16*)(smem+0); bf16* sV=(bf16*)(smem+32768);
  bf16* sQ=(bf16*)(smem+65536); bf16* sdO=(bf16*)(smem+98304);
  bf16* sPT=(bf16*)(smem+131072); bf16* sDST=(bf16*)(smem+163840);
  float* sL=(float*)(smem+196608); float* sD=(float*)(smem+197120);
  uint64_t* mbar=(uint64_t*)(smem+197632); uint32_t* stb=(uint32_t*)(smem+197648);

  int tid=threadIdx.x, warp=tid>>5;
  int j=blockIdx.x,h=blockIdx.y,b=blockIdx.z;
  int bh=b*H+h; long base=(long)bh*S*128;
  const bf16 *Kb=K+base,*Vb=V+base,*Qb=Q+base,*dOb=dO+base;
  int kvbase=j*128; int numQ=(S+127)/128;

  if(tid==0) init_bar(mbar,1);
  if(warp==0) alloc_cg1(stb,512);
  __syncthreads();
  uint32_t tb=*stb;
  if(warp==0) relinquish_cg1();
  uint32_t Scol=tb+0, dVcol=tb+128, dKcol=tb+256;
  uint32_t idS=mkid(128,128,0,0), idW=mkid(128,64,0,1);

  loadtile(sK,Kb,kvbase,S);
  loadtile(sV,Vb,kvbase,S);
  __syncthreads();
  int ph=0;
  int rv=(kvbase+tid)<S;

  for(int i=0;i<numQ;i++){
    int qb=i*128;
    loadtile(sQ,Qb,qb,S);
    loadtile(sdO,dOb,qb,S);
    { int q=tid; int gq=qb+q; sL[q]=(gq<S)?Lp[(long)bh*S+gq]:1e30f; sD[q]=(gq<S)?Dp[(long)bh*S+gq]:0.f; }
    __syncthreads();

    // G1: S^T = K @ Q^T
    if(tid==0){ fproxy();
      #pragma unroll
      for(int s=0;s<8;s++){int p=s>>2,w=s&3;
        umma(Scol, mkdesc(sK+p*8192+w*16), mkdesc(sQ+p*8192+w*16), idS, s==0?0:1);}
      commit_cg1(mbar);
    }
    bar_wait(mbar,ph); ph^=1;

    // P^T
    #pragma unroll
    for(int b32=0;b32<128;b32+=32){
      uint32_t r[32];
      #pragma unroll
      for(int k=0;k<32;k+=4) ld4nw(Scol+b32+k,r[k],r[k+1],r[k+2],r[k+3]);
      ldfence();
      #pragma unroll
      for(int k=0;k<32;k++){int q=b32+k; float s=__uint_as_float(r[k]);
        float p=0.f; if(rv&&(qb+q)<S) p=__expf(scale*s-sL[q]);
        sPT[swz(tid,q)]=__float2bfloat16(p);}
    }
    __syncthreads();

    // G3: dP^T = V @ dO^T
    if(tid==0){
      #pragma unroll
      for(int s=0;s<8;s++){int p=s>>2,w=s&3;
        umma(Scol, mkdesc(sV+p*8192+w*16), mkdesc(sdO+p*8192+w*16), idS, s==0?0:1);}
      commit_cg1(mbar);
    }
    bar_wait(mbar,ph); ph^=1;

    // dS^T = scale*P^T*(dP^T - D)
    #pragma unroll
    for(int b32=0;b32<128;b32+=32){
      uint32_t r[32];
      #pragma unroll
      for(int k=0;k<32;k+=4) ld4nw(Scol+b32+k,r[k],r[k+1],r[k+2],r[k+3]);
      ldfence();
      #pragma unroll
      for(int k=0;k<32;k++){int q=b32+k; int off=swz(tid,q); float p=(float)sPT[off];
        float dp=__uint_as_float(r[k]); float ds=0.f;
        if(rv&&(qb+q)<S) ds=scale*p*(dp-sD[q]);
        sDST[off]=__float2bfloat16(ds);}
    }
    __syncthreads();

    // G5 dV += P^T@dO ; G6 dK += dS^T@Q
    if(tid==0){ fproxy();
      #pragma unroll
      for(int dpn=0;dpn<2;dpn++)
        #pragma unroll
        for(int s=0;s<8;s++)
          umma(dVcol+dpn*64, mkdesc(sPT+(s>>2)*8192+(s&3)*16), mkdesc(sdO+dpn*8192+s*1024),
               idW, (i==0&&s==0)?0:1);
      #pragma unroll
      for(int dpn=0;dpn<2;dpn++)
        #pragma unroll
        for(int s=0;s<8;s++)
          umma(dKcol+dpn*64, mkdesc(sDST+(s>>2)*8192+(s&3)*16), mkdesc(sQ+dpn*8192+s*1024),
               idW, (i==0&&s==0)?0:1);
      commit_cg1(mbar);
    }
    bar_wait(mbar,ph); ph^=1;
    __syncthreads();
  }

  { int gk=kvbase+tid;
    if(gk<S){
      for(int c=0;c<128;c+=4){uint32_t r0,r1,r2,r3; ld4nw(dVcol+c,r0,r1,r2,r3); ldfence();
        dVo[base+(long)gk*128+c+0]=__float2bfloat16(__uint_as_float(r0));
        dVo[base+(long)gk*128+c+1]=__float2bfloat16(__uint_as_float(r1));
        dVo[base+(long)gk*128+c+2]=__float2bfloat16(__uint_as_float(r2));
        dVo[base+(long)gk*128+c+3]=__float2bfloat16(__uint_as_float(r3));}
      for(int c=0;c<128;c+=4){uint32_t r0,r1,r2,r3; ld4nw(dKcol+c,r0,r1,r2,r3); ldfence();
        dKo[base+(long)gk*128+c+0]=__float2bfloat16(__uint_as_float(r0));
        dKo[base+(long)gk*128+c+1]=__float2bfloat16(__uint_as_float(r1));
        dKo[base+(long)gk*128+c+2]=__float2bfloat16(__uint_as_float(r2));
        dKo[base+(long)gk*128+c+3]=__float2bfloat16(__uint_as_float(r3));}
    }
  }
  ldfence(); __syncthreads();
  if(warp==0) dealloc_cg1(tb,512);
}

// ============ dQ kernel (Q outer) ============
__global__ __launch_bounds__(128) void dq_kernel(
  const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
  const float* Lp,const float* Dp, bf16* dQo,int S,int H,float scale)
{
  extern __shared__ char smem[];
  bf16* sQ=(bf16*)(smem+0); bf16* sdO=(bf16*)(smem+32768);
  bf16* sK=(bf16*)(smem+65536); bf16* sV=(bf16*)(smem+98304);
  bf16* sPl=(bf16*)(smem+131072); bf16* sDS=(bf16*)(smem+163840);
  float* sL=(float*)(smem+196608); float* sD=(float*)(smem+197120);
  uint64_t* mbar=(uint64_t*)(smem+197632); uint32_t* stb=(uint32_t*)(smem+197648);

  int tid=threadIdx.x, warp=tid>>5;
  int iq=blockIdx.x,h=blockIdx.y,b=blockIdx.z;
  int bh=b*H+h; long base=(long)bh*S*128;
  const bf16 *Kb=K+base,*Vb=V+base,*Qb=Q+base,*dOb=dO+base;
  int qb=iq*128; int numKV=(S+127)/128;

  if(tid==0) init_bar(mbar,1);
  if(warp==0) alloc_cg1(stb,256);
  __syncthreads();
  uint32_t tb=*stb;
  if(warp==0) relinquish_cg1();
  uint32_t Scol=tb+0, dQcol=tb+128;
  uint32_t idS=mkid(128,128,0,0), idQ=mkid(128,64,0,1);

  loadtile(sQ,Qb,qb,S);
  loadtile(sdO,dOb,qb,S);
  { int q=tid; int gq=qb+q; sL[q]=(gq<S)?Lp[(long)bh*S+gq]:1e30f; sD[q]=(gq<S)?Dp[(long)bh*S+gq]:0.f; }
  __syncthreads();
  int ph=0;
  int qv=(qb+tid)<S;
  float Lloc=sL[tid], Dloc=sD[tid];

  for(int j=0;j<numKV;j++){
    int kvb=j*128;
    loadtile(sK,Kb,kvb,S);
    loadtile(sV,Vb,kvb,S);
    __syncthreads();

    // G1 S = Q@K^T
    if(tid==0){ fproxy();
      #pragma unroll
      for(int s=0;s<8;s++){int p=s>>2,w=s&3;
        umma(Scol, mkdesc(sQ+p*8192+w*16), mkdesc(sK+p*8192+w*16), idS, s==0?0:1);}
      commit_cg1(mbar);
    }
    bar_wait(mbar,ph); ph^=1;

    // P
    #pragma unroll
    for(int b32=0;b32<128;b32+=32){
      uint32_t r[32];
      #pragma unroll
      for(int k=0;k<32;k+=4) ld4nw(Scol+b32+k,r[k],r[k+1],r[k+2],r[k+3]);
      ldfence();
      #pragma unroll
      for(int k=0;k<32;k++){int kv=b32+k; float s=__uint_as_float(r[k]);
        float p=0.f; if(qv&&(kvb+kv)<S) p=__expf(scale*s-Lloc);
        sPl[tid*128+kv]=__float2bfloat16(p);}
    }
    __syncthreads();

    // G3 dP = dO@V^T
    if(tid==0){
      #pragma unroll
      for(int s=0;s<8;s++){int p=s>>2,w=s&3;
        umma(Scol, mkdesc(sdO+p*8192+w*16), mkdesc(sV+p*8192+w*16), idS, s==0?0:1);}
      commit_cg1(mbar);
    }
    bar_wait(mbar,ph); ph^=1;

    // dS = scale*P*(dP - D)
    #pragma unroll
    for(int b32=0;b32<128;b32+=32){
      uint32_t r[32];
      #pragma unroll
      for(int k=0;k<32;k+=4) ld4nw(Scol+b32+k,r[k],r[k+1],r[k+2],r[k+3]);
      ldfence();
      #pragma unroll
      for(int k=0;k<32;k++){int kv=b32+k; float p=(float)sPl[tid*128+kv];
        float dp=__uint_as_float(r[k]); float ds=0.f;
        if(qv&&(kvb+kv)<S) ds=scale*p*(dp-Dloc);
        sDS[swz(tid,kv)]=__float2bfloat16(ds);}
    }
    __syncthreads();

    // G6 dQ += dS@K
    if(tid==0){ fproxy();
      #pragma unroll
      for(int dpn=0;dpn<2;dpn++)
        #pragma unroll
        for(int s=0;s<8;s++)
          umma(dQcol+dpn*64, mkdesc(sDS+(s>>2)*8192+(s&3)*16), mkdesc(sK+dpn*8192+s*1024),
               idQ, (j==0&&s==0)?0:1);
      commit_cg1(mbar);
    }
    bar_wait(mbar,ph); ph^=1;
    __syncthreads();
  }

  { int gq=qb+tid;
    if(gq<S){
      for(int c=0;c<128;c+=4){uint32_t r0,r1,r2,r3; ld4nw(dQcol+c,r0,r1,r2,r3); ldfence();
        dQo[base+(long)gq*128+c+0]=__float2bfloat16(__uint_as_float(r0));
        dQo[base+(long)gq*128+c+1]=__float2bfloat16(__uint_as_float(r1));
        dQo[base+(long)gq*128+c+2]=__float2bfloat16(__uint_as_float(r2));
        dQo[base+(long)gq*128+c+3]=__float2bfloat16(__uint_as_float(r3));}
    }
  }
  ldfence(); __syncthreads();
  if(warp==0) dealloc_cg1(tb,256);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int64_t B=Q.size(0),H=Q.size(1),S=Q.size(2),d=Q.size(3);
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
  if(S<=0) return;

  long rows=B*H*S;
  float* Dbuf=nullptr;
  CUDA_CHECK(cudaMallocAsync(&Dbuf,rows*sizeof(float),stream));
  { int t=256; long bl=(rows+t-1)/t; compute_D_kernel<<<bl,t,0,stream>>>(dOp,Op,Dbuf,rows);
    CUDA_CHECK(cudaGetLastError()); }

  static bool set=false;
  if(!set){ CUDA_CHECK(cudaFuncSetAttribute(dkdv_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_BYTES));
            CUDA_CHECK(cudaFuncSetAttribute(dq_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_BYTES)); set=true; }

  int nblk=(int)((S+127)/128);
  dim3 grid(nblk,(unsigned)H,(unsigned)B);
  float scale=1.0f/sqrtf((float)d);

  dkdv_kernel<<<grid,128,SMEM_BYTES,stream>>>(Qp,Kp,Vp,dOp,Lp,Dbuf,dKp,dVp,(int)S,(int)H,scale);
  CUDA_CHECK(cudaGetLastError());
  dq_kernel<<<grid,128,SMEM_BYTES,stream>>>(Qp,Kp,Vp,dOp,Lp,Dbuf,dQp,(int)S,(int)H,scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFreeAsync(Dbuf,stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd