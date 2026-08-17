#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do{ cudaError_t e=(call); if(e!=cudaSuccess){ fprintf(stderr,"CUDA %s %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); exit(1);} }while(0)
#define CU_CHECK(call) do{ CUresult e=(call); if(e!=CUDA_SUCCESS){ const char* s; cuGetErrorString(e,&s); fprintf(stderr,"CU %s %s:%d\n",s,__FILE__,__LINE__); exit(1);} }while(0)

namespace mha {

constexpr int BM=128, BN=64, D=128;

__device__ __forceinline__ uint64_t smem_desc(void* p, uint32_t lbo, uint32_t sbo){
  uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
  d |= (uint64_t)(a & 0x3FFFF) >> 4;
  d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
  d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)2 << 61;
  return d;
}
__device__ __forceinline__ uint32_t instr_desc(int M,int N,int aMaj,int bMaj){
  uint32_t d=0;
  d |= (1u<<4); d |= (1u<<7); d |= (1u<<10);
  d |= ((uint32_t)aMaj<<15); d |= ((uint32_t)bMaj<<16);
  d |= ((uint32_t)(N>>3)<<17); d |= ((uint32_t)(M>>4)<<24);
  return d;
}
__device__ __forceinline__ void mbar_init(uint64_t* b,uint32_t c){
  asm volatile("mbarrier.init.shared.b64 [%0],%1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c));
}
__device__ __forceinline__ void mbar_arrive_expect(uint64_t* b,uint32_t tx){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(tx):"memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nWAIT_%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra WAIT_%=;\n}\n"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph));
}
__device__ __forceinline__ void fence_smem_barrier_init(){ asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory"); }
__device__ __forceinline__ void fence_proxy_async(){ asm volatile("fence.proxy.async;\n":::"memory"); }
__device__ __forceinline__ void tcg_fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }
__device__ __forceinline__ void tcg_fence_before(){ asm volatile("tcgen05.fence::before_thread_sync;":::"memory"); }

__device__ __forceinline__ void tmem_alloc1(uint32_t* dst,int n){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(dst);
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(a),"r"(n));
}
__device__ __forceinline__ void tmem_dealloc1(uint32_t addr,int n){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(addr),"r"(n));
}
__device__ __forceinline__ void umma1(uint32_t d,uint64_t a,uint64_t b,uint32_t idesc,uint32_t accum){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    ::"r"(d),"l"(a),"l"(b),"r"(idesc),"r"(accum));
}
__device__ __forceinline__ void commit1(uint64_t* bar){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"::"r"(a));
}
__device__ __forceinline__ void tma_2d(const CUtensorMap* d, uint64_t* bar, void* smem, int c0, int c1){
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
    :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
       "l"((uint64_t)d),
       "r"((uint32_t)__cvta_generic_to_shared(bar)),
       "r"(c0),"r"(c1):"memory");
}
// issue-only TMEM ops; waits batched separately
__device__ __forceinline__ void tmem_ld8(uint32_t addr, float* out){
  uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
    :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7):"r"(addr));
  out[0]=__uint_as_float(r0);out[1]=__uint_as_float(r1);out[2]=__uint_as_float(r2);out[3]=__uint_as_float(r3);
  out[4]=__uint_as_float(r4);out[5]=__uint_as_float(r5);out[6]=__uint_as_float(r6);out[7]=__uint_as_float(r7);
}
__device__ __forceinline__ void tmem_st8(uint32_t addr, const float* in){
  uint32_t r0=__float_as_uint(in[0]),r1=__float_as_uint(in[1]),r2=__float_as_uint(in[2]),r3=__float_as_uint(in[3]);
  uint32_t r4=__float_as_uint(in[4]),r5=__float_as_uint(in[5]),r6=__float_as_uint(in[6]),r7=__float_as_uint(in[7]);
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0],{%1,%2,%3,%4,%5,%6,%7,%8};"
    ::"r"(addr),"r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(r4),"r"(r5),"r"(r6),"r"(r7));
}
__device__ __forceinline__ void tmem_wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void tmem_wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;":::"memory"); }
__device__ __forceinline__ uint32_t pack2bf16(float a,float b){
  __nv_bfloat162 v=__floats2bfloat162_rn(a,b);
  return *reinterpret_cast<uint32_t*>(&v);
}

__global__ void __launch_bounds__(128,2) attn(
  const __grid_constant__ CUtensorMap tmaQ,
  const __grid_constant__ CUtensorMap tmaK,
  const __grid_constant__ CUtensorMap tmaV,
  __nv_bfloat16* O, float* LSE, int B,int H,int S)
{
  int tid=threadIdx.x;
  int warp=tid>>5;
  int b=blockIdx.z, h=blockIdx.y, qtile=blockIdx.x;
  int row=qtile*BM+tid;
  int bh=b*H+h;
  int row_q=qtile*BM;

  extern __shared__ char smem_raw[];
  uint32_t raw_off=(uint32_t)__cvta_generic_to_shared(smem_raw);
  uint32_t pad=((1024 - (raw_off & 1023)) & 1023);
  char* buf=smem_raw+pad;
  char* Q0=buf+0;      char* Q1=buf+16384;
  char* K0=buf+32768;  char* K1=buf+40960;
  char* V0=buf+49152;  char* V1=buf+57344;
  char* Pbuf=buf+65536;
  uint64_t* mbar=(uint64_t*)(buf+81920);
  uint32_t* tmem_ptr=(uint32_t*)(buf+81984);
  uint64_t* mbar_q=&mbar[0]; uint64_t* mbar_kv=&mbar[1]; uint64_t* mbar_s=&mbar[2]; uint64_t* mbar_o=&mbar[3];

  if(tid==0){
    mbar_init(mbar_q,1); mbar_init(mbar_kv,1); mbar_init(mbar_s,1); mbar_init(mbar_o,1);
    fence_smem_barrier_init();
  }
  __syncthreads();
  if(warp==0) tmem_alloc1(tmem_ptr,256);
  __syncthreads();
  uint32_t tbase=*tmem_ptr;
  uint32_t s_base=tbase;
  uint32_t o_base=tbase+64;

  uint32_t idesc_qk=instr_desc(128,64,0,0);
  uint32_t idesc_pv=instr_desc(128,64,0,1);

  float m_i=-INFINITY, l_i=0.f;
  const float scale=rsqrtf((float)D);
  uint32_t loff=((uint32_t)(warp*32))<<16;

  if(tid==0){
    mbar_arrive_expect(mbar_q,32768);
    tma_2d(&tmaQ,mbar_q,Q0,0,bh*S+row_q);
    tma_2d(&tmaQ,mbar_q,Q1,64,bh*S+row_q);
    mbar_wait(mbar_q,0);
  }

  int nt=(S+BN-1)/BN;
  for(int t=0;t<nt;t++){
    if(tid==0){
      mbar_arrive_expect(mbar_kv,32768);
      tma_2d(&tmaK,mbar_kv,K0,0,bh*S+t*BN);
      tma_2d(&tmaK,mbar_kv,K1,64,bh*S+t*BN);
      tma_2d(&tmaV,mbar_kv,V0,0,bh*S+t*BN);
      tma_2d(&tmaV,mbar_kv,V1,64,bh*S+t*BN);
      mbar_wait(mbar_kv,t&1);
      #pragma unroll
      for(int k=0;k<8;k++){
        char* qp=(k<4?Q0:Q1)+(k&3)*32;
        char* kp=(k<4?K0:K1)+(k&3)*32;
        umma1(s_base, smem_desc(qp,0,1024), smem_desc(kp,0,1024), idesc_qk, k>0?1u:0u);
      }
      commit1(mbar_s);
    }
    mbar_wait(mbar_s,t&1);
    tcg_fence_after();

    // ---- read S (one wait) ----
    uint32_t sa=s_base+loff;
    float s[64];
    #pragma unroll
    for(int c=0;c<64;c+=8) tmem_ld8(sa+c,&s[c]);
    tmem_wait_ld();

    float m_t=-INFINITY;
    #pragma unroll
    for(int c=0;c<64;c++){
      int gk=t*BN+c;
      float v=(gk<S)? s[c]*scale : -INFINITY;
      s[c]=v; m_t=fmaxf(m_t,v);
    }
    float m_old=m_i;
    float m_new=fmaxf(m_old,m_t);
    float corr=__expf(m_old-m_new);
    float psum=0.f;
    #pragma unroll
    for(int c=0;c<64;c++){ float p=__expf(s[c]-m_new); s[c]=p; psum+=p; }
    l_i=l_i*corr+psum;
    m_i=m_new;

    // ---- correction (batched waits, chunks of 32) ----
    if(t>0){
      uint32_t oa=o_base+loff;
      #pragma unroll
      for(int g=0;g<128;g+=32){
        float ob[32];
        #pragma unroll
        for(int c=0;c<32;c+=8) tmem_ld8(oa+g+c,&ob[c]);
        tmem_wait_ld();
        #pragma unroll
        for(int i=0;i<32;i++) ob[i]*=corr;
        #pragma unroll
        for(int c=0;c<32;c+=8) tmem_st8(oa+g+c,&ob[c]);
        tmem_wait_st();
      }
      tcg_fence_before();
    }

    // ---- pack P to smem ----
    #pragma unroll
    for(int cc=0;cc<8;cc++){
      uint4 pk;
      pk.x=pack2bf16(s[8*cc+0],s[8*cc+1]);
      pk.y=pack2bf16(s[8*cc+2],s[8*cc+3]);
      pk.z=pack2bf16(s[8*cc+4],s[8*cc+5]);
      pk.w=pack2bf16(s[8*cc+6],s[8*cc+7]);
      char* dst=Pbuf + tid*128 + (((tid&7)^cc)*16);
      *reinterpret_cast<uint4*>(dst)=pk;
    }
    fence_proxy_async();
    __syncthreads();

    if(tid==0){
      tcg_fence_after();
      #pragma unroll
      for(int c=0;c<2;c++){
        char* vatom=(c==0?V0:V1);
        #pragma unroll
        for(int j=0;j<4;j++){
          char* pp=Pbuf+j*32;
          char* vp=vatom+j*2048;
          uint32_t accum=(t>0||j>0)?1u:0u;
          umma1(o_base+c*64, smem_desc(pp,0,1024), smem_desc(vp,8192,1024), idesc_pv, accum);
        }
      }
      commit1(mbar_o);
    }
    mbar_wait(mbar_o,t&1);
    __syncthreads();
  }

  tcg_fence_after();
  float inv=(l_i>0.f)?(1.f/l_i):0.f;
  uint32_t oa=o_base+loff;
  __nv_bfloat16* Og=O+((long)bh*S+row)*D;
  #pragma unroll
  for(int g=0;g<128;g+=32){
    float ob[32];
    #pragma unroll
    for(int c=0;c<32;c+=8) tmem_ld8(oa+g+c,&ob[c]);
    tmem_wait_ld();
    if(row<S){
      #pragma unroll
      for(int c=0;c<32;c+=8){
        uint4 pk;
        pk.x=pack2bf16(ob[c+0]*inv,ob[c+1]*inv);
        pk.y=pack2bf16(ob[c+2]*inv,ob[c+3]*inv);
        pk.z=pack2bf16(ob[c+4]*inv,ob[c+5]*inv);
        pk.w=pack2bf16(ob[c+6]*inv,ob[c+7]*inv);
        *reinterpret_cast<uint4*>(Og+g+c)=pk;
      }
    }
  }
  if(row<S) LSE[(long)bh*S+row]=m_i+logf(l_i);

  __syncthreads();
  if(warp==0) tmem_dealloc1(tbase,256);
}

static CUresult make_tma(CUtensorMap* d, void* p, long rows, uint32_t box1){
  uint64_t gdim[2]={(uint64_t)D,(uint64_t)rows};
  uint64_t gstr[1]={(uint64_t)D*2};
  uint32_t bdim[2]={64, box1};
  uint32_t estr[2]={1,1};
  return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, p, gdim, gstr, bdim, estr,
    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2);
  __nv_bfloat16* Qp=static_cast<__nv_bfloat16*>(Q.data_ptr());
  __nv_bfloat16* Kp=static_cast<__nv_bfloat16*>(K.data_ptr());
  __nv_bfloat16* Vp=static_cast<__nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
  float* Lp=static_cast<float*>(LSE.data_ptr());

  long rows=(long)B*H*S;
  CUtensorMap tmaQ,tmaK,tmaV;
  CU_CHECK(make_tma(&tmaQ,Qp,rows,128));
  CU_CHECK(make_tma(&tmaK,Kp,rows,64));
  CU_CHECK(make_tma(&tmaV,Vp,rows,64));

  int smem=84000;
  CUDA_CHECK(cudaFuncSetAttribute((void*)attn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

  dim3 grid((S+BM-1)/BM, H, B), block(128);
  cudaStream_t st=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
  attn<<<grid,block,smem,st>>>(tmaQ,tmaK,tmaV,Op,Lp,B,H,S);
  CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha::run);

}  // namespace mha