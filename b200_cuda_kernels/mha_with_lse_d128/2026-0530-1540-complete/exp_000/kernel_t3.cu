#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <cstdint>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);} }while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ \
  const char* s; cuGetErrorString(_e,&s); fprintf(stderr,"CU error %s at %s:%d\n",s,__FILE__,__LINE__);} }while(0)

namespace mha {

__device__ __forceinline__ void init_bar(uint64_t* bar, uint32_t cnt){
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"
    :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(cnt));
}
__device__ __forceinline__ void fence_bar_init(){ asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory"); }
__device__ __forceinline__ void arrive_expect_tx(uint64_t* bar, uint32_t tx){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
    :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx) : "memory");
}
__device__ __forceinline__ void bar_wait(uint64_t* bar, uint32_t phase){
  asm volatile("{\n.reg .pred P;\nWAIT_%=:\n"
    "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
    "@!P bra WAIT_%=;\n}\n"
    :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1){
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
    :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
       "l"((uint64_t)d),
       "r"((uint32_t)__cvta_generic_to_shared(bar)),
       "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void fence_proxy_async(){ asm volatile("fence.proxy.async;\n":::"memory"); }
__device__ __forceinline__ void t5_fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }
__device__ __forceinline__ void t5_fence_before(){ asm volatile("tcgen05.fence::before_thread_sync;":::"memory"); }

__device__ __forceinline__ void umma1(uint32_t d_tmem, uint64_t da, uint64_t db, uint32_t idesc, uint32_t accum){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    :: "r"(d_tmem),"l"(da),"l"(db),"r"(idesc),"r"(accum));
}
__device__ __forceinline__ void commit1(uint64_t* bar){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
    :: "r"((uint32_t)__cvta_generic_to_shared(bar)));
}
__device__ __forceinline__ void alloc1(uint32_t* dst, int ncols){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;"
    :: "r"((uint32_t)__cvta_generic_to_shared(dst)),"r"(ncols));
}
__device__ __forceinline__ void dealloc1(uint32_t addr,int ncols){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;":: "r"(addr),"r"(ncols));
}

__device__ __forceinline__ uint64_t make_desc(const __nv_bfloat16* ptr){
  uint32_t a = (uint32_t)__cvta_generic_to_shared(ptr);
  uint64_t d=0;
  d |= (uint64_t)((a & 0x3FFFF) >> 4);
  d |= (uint64_t)((1024u) >> 4) << 32;   // SBO = 1024
  d |= (uint64_t)1 << 46;                // version SM100
  d |= (uint64_t)2 << 61;                // 128B swizzle
  return d;
}
__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M,uint32_t N,uint32_t bmaj){
  uint32_t d=0;
  d|=(1u<<4); d|=(1u<<7); d|=(1u<<10);
  d|=(bmaj<<16);
  d|=((N/8)<<17);
  d|=((M/16)<<24);
  return d;
}

#define LD32(a,o,col) asm volatile( \
"tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];" \
: "=f"(a[o+0]),"=f"(a[o+1]),"=f"(a[o+2]),"=f"(a[o+3]),"=f"(a[o+4]),"=f"(a[o+5]),"=f"(a[o+6]),"=f"(a[o+7]), \
  "=f"(a[o+8]),"=f"(a[o+9]),"=f"(a[o+10]),"=f"(a[o+11]),"=f"(a[o+12]),"=f"(a[o+13]),"=f"(a[o+14]),"=f"(a[o+15]), \
  "=f"(a[o+16]),"=f"(a[o+17]),"=f"(a[o+18]),"=f"(a[o+19]),"=f"(a[o+20]),"=f"(a[o+21]),"=f"(a[o+22]),"=f"(a[o+23]), \
  "=f"(a[o+24]),"=f"(a[o+25]),"=f"(a[o+26]),"=f"(a[o+27]),"=f"(a[o+28]),"=f"(a[o+29]),"=f"(a[o+30]),"=f"(a[o+31]) \
: "r"(col))

#define ST32(a,o,col) asm volatile( \
"tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" \
:: "r"(col), "f"(a[o+0]),"f"(a[o+1]),"f"(a[o+2]),"f"(a[o+3]),"f"(a[o+4]),"f"(a[o+5]),"f"(a[o+6]),"f"(a[o+7]), \
  "f"(a[o+8]),"f"(a[o+9]),"f"(a[o+10]),"f"(a[o+11]),"f"(a[o+12]),"f"(a[o+13]),"f"(a[o+14]),"f"(a[o+15]), \
  "f"(a[o+16]),"f"(a[o+17]),"f"(a[o+18]),"f"(a[o+19]),"f"(a[o+20]),"f"(a[o+21]),"f"(a[o+22]),"f"(a[o+23]), \
  "f"(a[o+24]),"f"(a[o+25]),"f"(a[o+26]),"f"(a[o+27]),"f"(a[o+28]),"f"(a[o+29]),"f"(a[o+30]),"f"(a[o+31]) : "memory")

#define WAIT_LD asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory")
#define WAIT_ST asm volatile("tcgen05.wait::st.sync.aligned;":::"memory")

__device__ __forceinline__ void load_KV(const CUtensorMap* dK, const CUtensorMap* dV,
        uint64_t* bar, __nv_bfloat16* sK, __nv_bfloat16* sV, int kvrow){
  arrive_expect_tx(bar, 65536);
  tma_load_2d(dK, bar, sK,        0,  kvrow);
  tma_load_2d(dK, bar, sK+8192,  64,  kvrow);
  tma_load_2d(dV, bar, sV,        0,  kvrow);
  tma_load_2d(dV, bar, sV+8192,  64,  kvrow);
}

__device__ __forceinline__ float do_softmax(uint32_t colS, int k_start, int S, float scale,
        __nv_bfloat16* sP, float& m_run, float& l_run, bool first){
  int tid=threadIdx.x;
  float s[128];
  LD32(s,0,colS); LD32(s,32,colS+32); LD32(s,64,colS+64); LD32(s,96,colS+96);
  WAIT_LD;
  float mt=-1e30f;
  #pragma unroll
  for(int c=0;c<128;c++){
    float v=s[c]*scale;
    if(k_start+c>=S) v=-1e30f;
    s[c]=v; mt=fmaxf(mt,v);
  }
  float m_old=m_run;
  float m_new=fmaxf(m_old,mt);
  float sum=0.f;
  #pragma unroll
  for(int c=0;c<128;c++){ float p=__expf(s[c]-m_new); s[c]=p; sum+=p; }
  float corr;
  if(first){ corr=1.f; m_run=m_new; l_run=sum; }
  else { corr=__expf(m_old-m_new); l_run=l_run*corr+sum; m_run=m_new; }
  int row=tid;
  #pragma unroll
  for(int region=0;region<2;region++){
    #pragma unroll
    for(int chunk=0;chunk<8;chunk++){
      int swz=(row&7)^chunk;
      __align__(16) __nv_bfloat16 vals[8];
      #pragma unroll
      for(int j=0;j<8;j++) vals[j]=__float2bfloat16(s[region*64+chunk*8+j]);
      *reinterpret_cast<int4*>(&sP[region*8192+row*64+swz*8])=*reinterpret_cast<int4*>(vals);
    }
  }
  return corr;
}

constexpr int BM=128, BN=128;

__global__ __launch_bounds__(128) void attn_kernel(
    const __grid_constant__ CUtensorMap descQ,
    const __grid_constant__ CUtensorMap descK,
    const __grid_constant__ CUtensorMap descV,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, float scale)
{
  extern __shared__ char smem_raw[];
  uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem_raw);
  uint32_t off = ((uint32_t)0 - sb) & 1023u;
  __nv_bfloat16* base = reinterpret_cast<__nv_bfloat16*>(smem_raw + off);
  __nv_bfloat16* sQ = base;
  __nv_bfloat16* sK[2] = { base+16384, base+32768 };
  __nv_bfloat16* sV[2] = { base+49152, base+65536 };
  __nv_bfloat16* sP[2] = { base+81920, base+98304 };
  uint64_t* bars = reinterpret_cast<uint64_t*>(base + 114688);
  uint32_t* tmem_addr = reinterpret_cast<uint32_t*>(bars + 6);

  uint64_t* mbarQ   = &bars[0];
  uint64_t* mbarLD[2]= { &bars[1], &bars[2] };
  uint64_t* mbarM1[2]= { &bars[3], &bars[4] };
  uint64_t* mbarM2  = &bars[5];

  const int tid = threadIdx.x;
  const int qtile = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int bh = b*H + h;
  const int bh_row = bh*S;
  const int q_start = qtile*BM;
  const int qrow = bh_row + q_start;
  const uint32_t idesc1 = make_instr_desc(128,128,0);
  const uint32_t idesc2 = make_instr_desc(128,64,1);

  if(tid==0){
    init_bar(mbarQ,1); init_bar(mbarLD[0],1); init_bar(mbarLD[1],1);
    init_bar(mbarM1[0],1); init_bar(mbarM1[1],1); init_bar(mbarM2,1);
    fence_bar_init();
  }
  __syncthreads();
  if(tid<32) alloc1(tmem_addr, 512);
  __syncthreads();
  uint32_t tmem_base = *tmem_addr;
  uint32_t colS[2] = { tmem_base, tmem_base+128 };
  uint32_t colO = tmem_base + 256;

  const int N = (S + BN - 1)/BN;
  float m_run=-1e30f, l_run=0.f, corr_pending=1.f;
  uint32_t pld[2]={0,0}, pm1[2]={0,0}, pm2=0;

  // ---- prologue ----
  if(tid==0){
    arrive_expect_tx(mbarQ, 32768);
    tma_load_2d(&descQ, mbarQ, sQ,       0, qrow);
    tma_load_2d(&descQ, mbarQ, sQ+8192, 64, qrow);
    load_KV(&descK,&descV, mbarLD[0], sK[0], sV[0], bh_row+0);
    if(N>1) load_KV(&descK,&descV, mbarLD[1], sK[1], sV[1], bh_row+128);
  }
  bar_wait(mbarQ,0);
  bar_wait(mbarLD[0], pld[0]); pld[0]^=1;
  if(tid==0){
    #pragma unroll
    for(int s=0;s<8;s++){
      int rg=s>>2, ko=(s&3)*16;
      umma1(colS[0], make_desc(sQ+rg*8192+ko), make_desc(sK[0]+rg*8192+ko), idesc1, s==0?0u:1u);
    }
    commit1(mbarM1[0]);
  }
  bar_wait(mbarM1[0], pm1[0]); pm1[0]^=1; t5_fence_after();
  corr_pending = do_softmax(colS[0], 0, S, scale, sP[0], m_run, l_run, true);

  // ---- main loop ----
  for(int it=0; it<N; it++){
    int cur=it&1, nxt=(it+1)&1;

    // issue QK(it+1) -> colS[nxt]
    if(it+1<N){
      bar_wait(mbarLD[nxt], pld[nxt]); pld[nxt]^=1;
      if(tid==0){
        #pragma unroll
        for(int s=0;s<8;s++){
          int rg=s>>2, ko=(s&3)*16;
          umma1(colS[nxt], make_desc(sQ+rg*8192+ko), make_desc(sK[nxt]+rg*8192+ko), idesc1, s==0?0u:1u);
        }
        commit1(mbarM1[nxt]);
      }
    }

    // rescale O by corr (overlaps QK on tensor core)
    if(it>0){
      float o[128];
      LD32(o,0,colO); LD32(o,32,colO+32); LD32(o,64,colO+64); LD32(o,96,colO+96);
      WAIT_LD;
      #pragma unroll
      for(int c=0;c<128;c++) o[c]*=corr_pending;
      ST32(o,0,colO); ST32(o,32,colO+32); ST32(o,64,colO+64); ST32(o,96,colO+96);
      WAIT_ST;
    }

    fence_proxy_async();
    t5_fence_before();
    __syncthreads();

    // PV(it): O += P[cur] @ V[cur]   (P K-major, V MN-major)
    if(tid==0){
      t5_fence_after();
      #pragma unroll
      for(int hh=0;hh<2;hh++){
        #pragma unroll
        for(int s=0;s<8;s++){
          int rg=s>>2, ko=(s&3)*16;
          uint32_t accum=(it==0 && s==0)?0u:1u;
          umma1(colO+hh*64,
                make_desc(sP[cur]+rg*8192+ko),
                make_desc(sV[cur]+hh*8192+s*1024),
                idesc2, accum);
        }
      }
      commit1(mbarM2);
    }

    // softmax(it+1) overlaps PV(it)
    if(it+1<N){
      bar_wait(mbarM1[nxt], pm1[nxt]); pm1[nxt]^=1; t5_fence_after();
      corr_pending = do_softmax(colS[nxt], (it+1)*128, S, scale, sP[nxt], m_run, l_run, false);
    }

    bar_wait(mbarM2, pm2); pm2^=1;

    // prefetch load for tile it+2 into buffer cur (free now)
    if(it+2<N && tid==0){
      load_KV(&descK,&descV, mbarLD[cur], sK[cur], sV[cur], bh_row+(it+2)*128);
    }
  }

  // ---- epilogue ----
  t5_fence_after();
  {
    float o[128];
    LD32(o,0,colO); LD32(o,32,colO+32); LD32(o,64,colO+64); LD32(o,96,colO+96);
    WAIT_LD;
    int grow = q_start + tid;
    if(grow < S){
      float inv = 1.0f / l_run;
      size_t ob = (size_t)bh*S*128 + (size_t)grow*128;
      #pragma unroll
      for(int c=0;c<128;c+=8){
        __align__(16) __nv_bfloat16 vals[8];
        #pragma unroll
        for(int j=0;j<8;j++) vals[j]=__float2bfloat16(o[c+j]*inv);
        *reinterpret_cast<int4*>(O+ob+c) = *reinterpret_cast<int4*>(vals);
      }
      LSE[(size_t)bh*S + grow] = m_run + logf(l_run);
    }
  }
  __syncthreads();
  if(tid<32) dealloc1(tmem_base, 512);
}

static CUresult make_tma_2d(CUtensorMap* d, void* gptr, uint64_t inner, uint64_t outer,
                            uint32_t bin, uint32_t bout, CUtensorMapSwizzle swz){
  uint64_t gdim[2]={inner, outer};
  uint64_t gstr[1]={inner*2};
  uint32_t bdim[2]={bin, bout};
  uint32_t estr[2]={1,1};
  return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gptr, gdim, gstr,
      bdim, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2), Dd=(int)Q.size(3);
  __nv_bfloat16* Qp=(__nv_bfloat16*)Q.data_ptr();
  __nv_bfloat16* Kp=(__nv_bfloat16*)K.data_ptr();
  __nv_bfloat16* Vp=(__nv_bfloat16*)V.data_ptr();
  __nv_bfloat16* Op=(__nv_bfloat16*)O.data_ptr();
  float* Lp=(float*)LSE.data_ptr();

  uint64_t outer=(uint64_t)B*H*S;
  CUtensorMap dQ,dK,dV;
  CU_CHECK(make_tma_2d(&dQ, Qp, 128, outer, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B));
  CU_CHECK(make_tma_2d(&dK, Kp, 128, outer, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B));
  CU_CHECK(make_tma_2d(&dV, Vp, 128, outer, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B));

  int ntiles=(S+BM-1)/BM;
  dim3 grid(ntiles, H, B);
  dim3 block(128);
  size_t smem = (size_t)114688*2 + 1024 + 256;

  cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

  float scale = 1.0f/sqrtf((float)Dd);
  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id);
  attn_kernel<<<grid, block, smem, stream>>>(dQ, dK, dV, Op, Lp, B, H, S, scale);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha::run);

}  // namespace mha