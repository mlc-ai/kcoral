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
#define BN 64
#define DH 128
#define LOG2E 1.4426950408889634f
#define QSZ (BM*DH)
#define SKO (BN*DH)

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
__device__ __forceinline__ void nbar(int id,int cnt){ asm volatile("barrier.sync %0, %1;"::"r"(id),"r"(cnt):"memory"); }

__global__ __launch_bounds__(256,1) void attn(
   const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
   const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O,
   float* __restrict__ LSE, int S)
{
  extern __shared__ char smem[];
  __nv_bfloat16* sQ0=(__nv_bfloat16*)smem;
  __nv_bfloat16* sQ1=sQ0+QSZ;
  __nv_bfloat16* sK =sQ1+QSZ;          // single buffer
  __nv_bfloat16* sVt=sK +SKO;          // 2*SKO
  __nv_bfloat16* sP0=sVt+2*SKO;        // 2*SKO
  __nv_bfloat16* sP1=sP0+2*SKO;        // 2*SKO
  uint64_t* bar=(uint64_t*)(sP1+2*SKO);
  uint32_t* tmemp=(uint32_t*)(bar+4);

  int tid=threadIdx.x;
  int wg = tid>>7;          // 0 or 1
  int ltid = tid&127;       // 0..127 in WG
  int bx = blockIdx.x, bh = blockIdx.y;
  const float scale = rsqrtf((float)DH);

  const __nv_bfloat16* Qh=Q+(int64_t)bh*S*DH;
  const __nv_bfloat16* Kh=K+(int64_t)bh*S*DH;
  const __nv_bfloat16* Vh=V+(int64_t)bh*S*DH;
  __nv_bfloat16* Oh=O+(int64_t)bh*S*DH;
  float* LSEh=LSE+(int64_t)bh*S;

  uint64_t* bQK0=&bar[0]; uint64_t* bQK1=&bar[1]; uint64_t* bPV0=&bar[2]; uint64_t* bPV1=&bar[3];
  if(tid==0){ mbar_init(bQK0,1);mbar_init(bQK1,1);mbar_init(bPV0,1);mbar_init(bPV1,1); }
  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  __syncthreads();
  if(tid<32) tmem_alloc(tmemp,512);
  __syncthreads();
  uint32_t bt=tmemp[0];
  uint32_t tS_wg=(wg==0)?(bt+0):(bt+192);
  uint32_t tO_wg=(wg==0)?(bt+64):(bt+256);
  __nv_bfloat16* sP_wg=(wg==0)?sP0:sP1;

  // load Q0,Q1
  #pragma unroll
  for(int idx=tid; idx<BM*(DH/8); idx+=256){
    int m=idx/(DH/8), kc=idx%(DH/8);
    int q0=bx*256+m, q1=bx*256+128+m;
    float4 v0=make_float4(0,0,0,0), v1=make_float4(0,0,0,0);
    if(q0<S) v0=*reinterpret_cast<const float4*>(&Qh[(int64_t)q0*DH+kc*8]);
    if(q1<S) v1=*reinterpret_cast<const float4*>(&Qh[(int64_t)q1*DH+kc*8]);
    *reinterpret_cast<float4*>(&sQ0[kc*BM*8+m*8])=v0;
    *reinterpret_cast<float4*>(&sQ1[kc*BM*8+m*8])=v1;
  }
  __syncthreads();

  const uint32_t idesc_qk=make_idesc(BM,BN);
  const uint32_t idesc_pv=make_idesc(BM,DH);
  int N=(S+BN-1)/BN;

  float m_i=-1e30f, l_i=0.f;
  uint32_t qk0p=0,qk1p=0,pv0p=0,pv1p=0;

  for(int j=0;j<N;j++){
    int buf=j&1; int ks=j*BN;
    // load K(j) single buffer, V(j) transposed into sVt[buf]
    #pragma unroll
    for(int idx=tid; idx<BN*(DH/8); idx+=256){
      int n=idx/(DH/8), kc=idx%(DH/8); int kg=ks+n;
      float4 v=make_float4(0,0,0,0);
      if(kg<S) v=*reinterpret_cast<const float4*>(&Kh[(int64_t)kg*DH+kc*8]);
      *reinterpret_cast<float4*>(&sK[kc*BN*8+n*8])=v;
    }
    #pragma unroll
    for(int idx=tid; idx<BN*(DH/8); idx+=256){
      int key=idx/(DH/8), dc=idx%(DH/8), d0=dc*8; int kg=ks+key;
      float4 v=make_float4(0,0,0,0);
      if(kg<S) v=*reinterpret_cast<const float4*>(&Vh[(int64_t)kg*DH+d0]);
      __nv_bfloat16* vb=reinterpret_cast<__nv_bfloat16*>(&v);
      int base=(key/8)*(DH*8)+(key%8);
      #pragma unroll
      for(int i=0;i<8;i++) sVt[buf*SKO + base+(d0+i)*8]=vb[i];
    }
    __syncthreads();

    // issue QK0 -> S0 ; QK1 -> S1
    if(tid==0){
      fpa();
      uint32_t q0=cvta_s(sQ0), q1=cvta_s(sQ1), kb=cvta_s(sK);
      #pragma unroll
      for(int i=0;i<DH/16;i++){
        uint64_t da=make_desc(q0+i*(BM*32),BM*16,128);
        uint64_t db=make_desc(kb+i*(BN*32),BN*16,128);
        umma_cg1(bt+0,da,db,idesc_qk,i==0?0:1);
      }
      umma_commit(bQK0);
      #pragma unroll
      for(int i=0;i<DH/16;i++){
        uint64_t da=make_desc(q1+i*(BM*32),BM*16,128);
        uint64_t db=make_desc(kb+i*(BN*32),BN*16,128);
        umma_cg1(bt+192,da,db,idesc_qk,i==0?0:1);
      }
      umma_commit(bQK1);
    }

    // softmax(wg): overlaps the other tile's MMA
    if(wg==0){ mbar_wait(bQK0,qk0p); qk0p^=1; } else { mbar_wait(bQK1,qk1p); qk1p^=1; }
    fa();
    float s[64];
    #pragma unroll
    for(int c=0;c<64;c+=4) tmem_ld4(tS_wg+c, s[c],s[c+1],s[c+2],s[c+3]);
    wait_ld();
    float mx=-1e30f;
    #pragma unroll
    for(int k=0;k<64;k++){ int kg=ks+k; float v=(kg<S)? s[k]*scale : -1e30f; s[k]=v; mx=fmaxf(mx,v); }
    float m_new=fmaxf(m_i,mx);
    float corr=fexp2((m_i-m_new)*LOG2E);
    float lsum=0.f;
    #pragma unroll
    for(int k=0;k<64;k++){ float p=fexp2((s[k]-m_new)*LOG2E); s[k]=p; lsum+=p; }
    l_i=l_i*corr+lsum; m_i=m_new;
    #pragma unroll
    for(int kc=0;kc<8;kc++){
      __nv_bfloat16 pb[8];
      #pragma unroll
      for(int i=0;i<8;i++) pb[i]=__float2bfloat16(s[kc*8+i]);
      *reinterpret_cast<float4*>(&sP_wg[buf*SKO + kc*BM*8 + ltid*8]) = *reinterpret_cast<float4*>(pb);
    }

    // correction of O (needs prev PV done)
    if(j>0){
      if(wg==0){ mbar_wait(bPV0,pv0p); pv0p^=1; } else { mbar_wait(bPV1,pv1p); pv1p^=1; }
      bool need = !__all_sync(0xffffffff, corr>0.99998f);
      if(need){
        fa();
        #pragma unroll
        for(int b0=0;b0<DH;b0+=32){
          float o[32];
          #pragma unroll
          for(int c=0;c<32;c+=4) tmem_ld4(tO_wg+b0+c, o[c],o[c+1],o[c+2],o[c+3]);
          wait_ld();
          #pragma unroll
          for(int c=0;c<32;c++) o[c]*=corr;
          #pragma unroll
          for(int c=0;c<32;c+=4) tmem_st4(tO_wg+b0+c, o[c],o[c+1],o[c+2],o[c+3]);
        }
        wait_st();
      }
    }

    // issue PV(wg) (P0/PV0 by tid0 so it overlaps WG1's softmax)
    fpa();
    fb();
    nbar(1+wg,128);
    if(ltid==0){
      fa();
      uint32_t pb=cvta_s(sP_wg + buf*SKO);
      uint32_t vb=cvta_s(sVt + buf*SKO);
      #pragma unroll
      for(int jj=0;jj<BN/16;jj++){
        uint64_t da=make_desc(pb+jj*(BM*32),BM*16,128);
        uint64_t db=make_desc(vb+jj*(DH*32),DH*16,128);
        umma_cg1(tO_wg,da,db,idesc_pv,(j==0)?0:1);
      }
      if(wg==0) umma_commit(bPV0); else umma_commit(bPV1);
    }
  }

  // epilogue
  if(wg==0){ mbar_wait(bPV0,pv0p); } else { mbar_wait(bPV1,pv1p); }
  fa();
  float linv=(l_i>0.f)?1.f/l_i:0.f;
  int row=bx*256 + wg*128 + ltid;
  #pragma unroll
  for(int c=0;c<DH;c+=4){
    float o0,o1,o2,o3;
    tmem_ld4(tO_wg+c,o0,o1,o2,o3);
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
  if(tid<32) tmem_dealloc(bt,512);
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

  dim3 grid((S+255)/256, Bsz*Hn);
  dim3 block(256);
  size_t smem = (size_t)(2*QSZ + SKO + 2*SKO + 2*SKO + 2*SKO)*sizeof(__nv_bfloat16) + 128;
  cudaFuncSetAttribute(attn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  attn<<<grid, block, smem, stream>>>(Qp,Kp,Vp,Op,LSEp,S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel