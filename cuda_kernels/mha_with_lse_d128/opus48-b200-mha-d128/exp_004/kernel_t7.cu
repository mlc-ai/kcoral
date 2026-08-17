#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);} }while(0)

namespace mha_kernel {

#define BM 128
#define BN 128
#define DH 128
#define LOG2E 1.4426950408889634f

__device__ __forceinline__ uint32_t cvta_s(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ float fexp2(float x){ float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y; }
__device__ __forceinline__ uint64_t make_desc(uint32_t addr, uint32_t lbo, uint32_t sbo){
  uint64_t d=0;
  d |= (uint64_t)((addr & 0x3FFFFu) >> 4);
  d |= (uint64_t)((lbo  & 0x3FFFFu) >> 4) << 16;
  d |= (uint64_t)((sbo  & 0x3FFFFu) >> 4) << 32;
  d |= (uint64_t)1 << 46;
  return d;
}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N){
  uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10); d|=((N/8)<<17); d|=((M/16)<<24); return d;
}
__device__ __forceinline__ void umma_cg1(uint32_t td, uint64_t da, uint64_t db, uint32_t id, uint32_t acc){
  asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
    :: "r"(td),"l"(da),"l"(db),"r"(id),"r"(acc):"memory");
}
__device__ __forceinline__ void umma_commit(uint64_t* b){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"::"r"(cvta_s(b)):"memory");
}
__device__ __forceinline__ void tmem_ld4(uint32_t a, float& x,float& y,float& z,float& w){
  uint32_t r0,r1,r2,r3;
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];":"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a));
  x=__uint_as_float(r0);y=__uint_as_float(r1);z=__uint_as_float(r2);w=__uint_as_float(r3);
}
__device__ __forceinline__ void tmem_st4(uint32_t a, float x,float y,float z,float w){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
    ::"r"(a),"r"(__float_as_uint(x)),"r"(__float_as_uint(y)),"r"(__float_as_uint(z)),"r"(__float_as_uint(w)):"memory");
}
__device__ __forceinline__ void wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;":::"memory"); }
__device__ __forceinline__ void fa(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }
__device__ __forceinline__ void fb(){ asm volatile("tcgen05.fence::before_thread_sync;":::"memory"); }
__device__ __forceinline__ void fpa(){ asm volatile("fence.proxy.async;":::"memory"); }
__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t c){ asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"(cvta_s(b)),"r"(c)); }
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t ph){
  asm volatile("{\n.reg .pred P;\nWL_%=:\n mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n @!P bra WL_%=;\n}\n"::"r"(cvta_s(b)),"r"(ph));
}
__device__ __forceinline__ void tmem_alloc(uint32_t* d,int n){ asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(cvta_s(d)),"r"(n)); }
__device__ __forceinline__ void tmem_dealloc(uint32_t a,int n){ asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(a),"r"(n)); }

__global__ __launch_bounds__(128,2) void attn(
   const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
   const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O,
   float* __restrict__ LSE, int S)
{
  extern __shared__ char smem[];
  __nv_bfloat16* sQ = (__nv_bfloat16*)smem;      // [DH/8][BM][8]
  __nv_bfloat16* sK = sQ + BM*DH;                // [DH/8][BN][8]  (reused as sP)
  __nv_bfloat16* sVt= sK + BN*DH;                // [BN/8][DH][8]
  __nv_bfloat16* sP = sK;
  uint64_t* bar = (uint64_t*)(sVt + BN*DH);
  uint32_t* tmemp = (uint32_t*)(bar + 4);

  int tid = threadIdx.x, warp = tid>>5;
  int q_start = blockIdx.x*BM, bh = blockIdx.y;
  const float scale = rsqrtf((float)DH);

  const __nv_bfloat16* Qh=Q+(int64_t)bh*S*DH;
  const __nv_bfloat16* Kh=K+(int64_t)bh*S*DH;
  const __nv_bfloat16* Vh=V+(int64_t)bh*S*DH;
  __nv_bfloat16* Oh=O+(int64_t)bh*S*DH;
  float* LSEh=LSE+(int64_t)bh*S;

  uint64_t* barQK=&bar[0]; uint64_t* barPV=&bar[1];
  if(tid==0){ mbar_init(barQK,1); mbar_init(barPV,1); }
  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  __syncthreads();
  if(warp==0) tmem_alloc(tmemp,256);
  __syncthreads();
  uint32_t base_t=tmemp[0];
  uint32_t tmem_S=base_t;
  uint32_t tmem_O=base_t+128;

  #pragma unroll
  for(int idx=tid; idx<BM*(DH/8); idx+=128){
    int m=idx/(DH/8), kc=idx%(DH/8); int qg=q_start+m;
    float4 v=make_float4(0,0,0,0);
    if(qg<S) v=*reinterpret_cast<const float4*>(&Qh[(int64_t)qg*DH+kc*8]);
    *reinterpret_cast<float4*>(&sQ[kc*BM*8+m*8])=v;
  }
  __syncthreads();

  const uint32_t idesc_qk=make_idesc(BM,BN);
  const uint32_t idesc_pv=make_idesc(BM,DH);

  float m_i=-1e30f, l_i=0.f;
  uint32_t qk_par=0, pv_par=0;
  int num_kv=(S+BN-1)/BN;

  for(int kv=0; kv<num_kv; kv++){
    int k_start=kv*BN;
    #pragma unroll
    for(int idx=tid; idx<BN*(DH/8); idx+=128){
      int n=idx/(DH/8), kc=idx%(DH/8); int kg=k_start+n;
      float4 v=make_float4(0,0,0,0);
      if(kg<S) v=*reinterpret_cast<const float4*>(&Kh[(int64_t)kg*DH+kc*8]);
      *reinterpret_cast<float4*>(&sK[kc*BN*8+n*8])=v;
    }
    #pragma unroll
    for(int idx=tid; idx<BN*(DH/8); idx+=128){
      int key=idx/(DH/8), dc=idx%(DH/8), d0=dc*8; int kg=k_start+key;
      float4 v=make_float4(0,0,0,0);
      if(kg<S) v=*reinterpret_cast<const float4*>(&Vh[(int64_t)kg*DH+d0]);
      __nv_bfloat16* vb=reinterpret_cast<__nv_bfloat16*>(&v);
      int base=(key/8)*(DH*8)+(key%8);
      #pragma unroll
      for(int i=0;i<8;i++) sVt[base+(d0+i)*8]=vb[i];
    }
    __syncthreads();

    if(tid==0){
      fpa();
      uint32_t qb=cvta_s(sQ), kb=cvta_s(sK);
      #pragma unroll
      for(int i=0;i<DH/16;i++){
        uint64_t da=make_desc(qb+i*(BM*32), BM*16,128);
        uint64_t db=make_desc(kb+i*(BN*32), BN*16,128);
        umma_cg1(tmem_S, da,db, idesc_qk, i==0?0:1);
      }
      umma_commit(barQK);
    }
    mbar_wait(barQK, qk_par); qk_par^=1;
    fa();

    float s[BN];
    #pragma unroll
    for(int c=0;c<BN;c+=4) tmem_ld4(tmem_S+c, s[c],s[c+1],s[c+2],s[c+3]);
    wait_ld();

    float mx=-1e30f;
    #pragma unroll
    for(int k=0;k<BN;k++){ int kg=k_start+k; float v=(kg<S)? s[k]*scale : -1e30f; s[k]=v; mx=fmaxf(mx,v); }
    float m_new=fmaxf(m_i,mx);
    float corr=fexp2((m_i-m_new)*LOG2E);
    float lsum=0.f;
    #pragma unroll
    for(int k=0;k<BN;k++){ float p=fexp2((s[k]-m_new)*LOG2E); s[k]=p; lsum+=p; }
    l_i=l_i*corr+lsum; m_i=m_new;

    #pragma unroll
    for(int kc=0;kc<BN/8;kc++){
      __nv_bfloat16 pb[8];
      #pragma unroll
      for(int i=0;i<8;i++) pb[i]=__float2bfloat16(s[kc*8+i]);
      *reinterpret_cast<float4*>(&sP[kc*BM*8 + tid*8]) = *reinterpret_cast<float4*>(pb);
    }

    bool did=false;
    if(kv>0){
      bool need = !__all_sync(0xffffffff, corr>0.99998f);
      if(need){
        #pragma unroll
        for(int half=0; half<2; half++){
          float o[64];
          #pragma unroll
          for(int c=0;c<64;c+=4) tmem_ld4(tmem_O+half*64+c, o[c],o[c+1],o[c+2],o[c+3]);
          wait_ld();
          #pragma unroll
          for(int c=0;c<64;c++) o[c]*=corr;
          #pragma unroll
          for(int c=0;c<64;c+=4) tmem_st4(tmem_O+half*64+c, o[c],o[c+1],o[c+2],o[c+3]);
        }
        wait_st(); did=true;
      }
    }
    fpa();
    if(did) fb();
    __syncthreads();

    if(tid==0){
      fa();
      uint32_t pb=cvta_s(sP), vb=cvta_s(sVt);
      #pragma unroll
      for(int j=0;j<BN/16;j++){
        uint64_t da=make_desc(pb+j*(BM*32), BM*16,128);
        uint64_t db=make_desc(vb+j*(DH*32), DH*16,128);
        umma_cg1(tmem_O, da,db, idesc_pv, (kv==0&&j==0)?0:1);
      }
      umma_commit(barPV);
    }
    mbar_wait(barPV, pv_par); pv_par^=1;
    fa();
    __syncthreads();
  }

  float linv=(l_i>0.f)?1.f/l_i:0.f;
  int row=q_start+tid;
  #pragma unroll
  for(int c=0;c<DH;c+=4){
    float o0,o1,o2,o3;
    tmem_ld4(tmem_O+c,o0,o1,o2,o3);
    wait_ld();
    if(row<S){
      __nv_bfloat16 ob[4];
      ob[0]=__float2bfloat16(o0*linv); ob[1]=__float2bfloat16(o1*linv);
      ob[2]=__float2bfloat16(o2*linv); ob[3]=__float2bfloat16(o3*linv);
      *reinterpret_cast<uint2*>(&Oh[(int64_t)row*DH + c]) = *reinterpret_cast<uint2*>(ob);
    }
  }
  if(row<S) LSEh[row]=m_i+logf(l_i);
  __syncthreads();
  if(warp==0) tmem_dealloc(base_t,256);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bsz=(int)Q.size(0), Hn=(int)Q.size(1), S=(int)Q.size(2);

  const __nv_bfloat16* Qp=static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp=static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp=static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
  float* LSEp=static_cast<float*>(LSE.data_ptr());

  dim3 grid((S+BM-1)/BM, Bsz*Hn);
  dim3 block(128);
  size_t smem = (size_t)(BM*DH + BN*DH + BN*DH)*sizeof(__nv_bfloat16) + 4*sizeof(uint64_t) + 16;
  cudaFuncSetAttribute(attn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  attn<<<grid, block, smem, stream>>>(Qp,Kp,Vp,Op,LSEp,S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel