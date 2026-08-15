#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);} }while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s; cuGetErrorString(_e,&s); \
  fprintf(stderr,"CU error %s at %s:%d\n",s,__FILE__,__LINE__);} }while(0)

namespace mha_kernel {

__device__ __forceinline__ void init_bar(uint64_t* bar, uint32_t c){
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c));
}
__device__ __forceinline__ void fence_bar_init(){ asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory"); }
__device__ __forceinline__ void arrive_expect_tx(uint64_t* bar, uint32_t tx){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(tx):"memory");
}
__device__ __forceinline__ void bar_wait(uint64_t* bar, uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"
    ::"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(ph));
}
__device__ __forceinline__ void fence_proxy_async(){ asm volatile("fence.proxy.async;\n":::"memory"); }
__device__ __forceinline__ void tcg_fence_before(){ asm volatile("tcgen05.fence::before_thread_sync;":::"memory"); }
__device__ __forceinline__ void tcg_fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }
__device__ __forceinline__ void tcg_wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void tcg_wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;":::"memory"); }

__device__ __forceinline__ void tma_load_2d(const CUtensorMap* d, uint64_t* bar, void* smem, int c0, int c1){
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
    ::"r"((uint32_t)__cvta_generic_to_shared(smem)),"l"((uint64_t)d),
      "r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c0),"r"(c1):"memory");
}
__device__ __forceinline__ void tmem_alloc_cg1(uint32_t* dst, int ncols){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(dst)),"r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc_cg1(uint32_t addr, int ncols){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(addr),"r"(ncols));
}
__device__ __forceinline__ void umma_cg1(uint32_t td, uint64_t da, uint64_t db, uint32_t id, uint32_t acc){
  asm volatile("{.reg .pred p; setp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;}\n"
    ::"r"(td),"l"(da),"l"(db),"r"(id),"r"(acc));
}
__device__ __forceinline__ void umma_commit_cg1(uint64_t* bar){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
    ::"r"((uint32_t)__cvta_generic_to_shared(bar)));
}
__device__ __forceinline__ void tmem_ld4(uint32_t taddr, uint32_t* r0,uint32_t* r1,uint32_t* r2,uint32_t* r3){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
    :"=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3):"r"(taddr));
}
__device__ __forceinline__ void tmem_st4(uint32_t taddr, uint32_t r0,uint32_t r1,uint32_t r2,uint32_t r3){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
    ::"r"(taddr),"r"(r0),"r"(r1),"r"(r2),"r"(r3):"memory");
}
__device__ __forceinline__ uint64_t make_desc(void* p, uint32_t lbo, uint32_t sbo){
  uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
  d |= (uint64_t)(a & 0x3FFFF) >> 4;
  d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
  d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)2 << 61;
  return d;
}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N){
  uint32_t d=0;
  d |= (1u<<4); d |= (1u<<7); d |= (1u<<10);
  d |= ((N/8)<<17); d |= ((M/16)<<24);
  return d;
}

__device__ __forceinline__ void do_qk(uint32_t Saddr, __nv_bfloat16* Q0, __nv_bfloat16* Qh,
                                       __nv_bfloat16* K0, __nv_bfloat16* Kh, uint32_t idesc){
  #pragma unroll
  for (int kt=0; kt<8; kt++){
    int box=kt>>2, kl=kt&3;
    __nv_bfloat16* aptr=(box?Qh:Q0)+kl*16;
    __nv_bfloat16* bptr=(box?Kh:K0)+kl*16;
    umma_cg1(Saddr, make_desc(aptr,16,1024), make_desc(bptr,16,1024), idesc, kt==0?0:1);
  }
}
__device__ __forceinline__ void do_pv(uint32_t Oaddr, __nv_bfloat16* Psmem, __nv_bfloat16* Vt,
                                       uint32_t idesc, bool first){
  #pragma unroll
  for (int kt=0; kt<4; kt++){
    __nv_bfloat16* aptr=Psmem+kt*16;
    __nv_bfloat16* bptr=Vt+kt*16;
    umma_cg1(Oaddr, make_desc(aptr,16,1024), make_desc(bptr,16,1024), idesc, (first&&kt==0)?0:1);
  }
}

__global__ __launch_bounds__(128) void attn(
    const __grid_constant__ CUtensorMap tmaQ,
    const __grid_constant__ CUtensorMap tmaK,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int H){

  int qb = blockIdx.x;
  int h  = blockIdx.y;
  int b  = blockIdx.z;
  int tid = threadIdx.x;
  int warp = tid>>5;

  extern __shared__ char smem_raw[];
  uintptr_t base = ((uintptr_t)smem_raw + 1023ull) & ~1023ull;
  __nv_bfloat16* Qsmem = (__nv_bfloat16*)base;     // 16384
  __nv_bfloat16* Ksmem0= Qsmem + 16384;            // 8192
  __nv_bfloat16* Ksmem1= Ksmem0 + 8192;            // 8192
  __nv_bfloat16* Vt    = Ksmem1 + 8192;            // 8192
  __nv_bfloat16* Psmem = Vt + 8192;                // 8192
  uint64_t* bars = (uint64_t*)(Psmem + 8192);
  uint64_t* barK = &bars[0];   // [0],[1]
  uint64_t* barQ = &bars[2];   // [2],[3]
  uint64_t* barP = &bars[4];
  uint64_t* barQl= &bars[5];

  __shared__ uint32_t tmem_addr_smem;

  if (tid==0){ for(int i=0;i<6;i++) init_bar(&bars[i],1); }
  fence_bar_init();
  __syncthreads();
  if (warp==0) tmem_alloc_cg1(&tmem_addr_smem, 256);
  __syncthreads();
  uint32_t tmem_base = tmem_addr_smem;
  uint32_t O_taddr = tmem_base;
  uint32_t S0_taddr = tmem_base + 128;
  uint32_t S1_taddr = tmem_base + 192;

  uint32_t idesc_qk = make_idesc(128,64);
  uint32_t idesc_pv = make_idesc(128,128);

  int phK[2]={0,0}, phQ[2]={0,0}, ppv=0;
  long bh = (long)(b*H+h);
  float scale = rsqrtf(128.0f);
  int qpos_i = qb*128 + tid;

  int last_kb = 2*qb+1;
  int maxkb = (S+63)/64 - 1;
  if (last_kb > maxkb) last_kb = maxkb;

  int row_q = (int)(bh*S + (long)qb*128);
  if (tid==0){
    arrive_expect_tx(barQl, 128*128*2);
    tma_load_2d(&tmaQ,barQl,Qsmem,        0, row_q);
    tma_load_2d(&tmaQ,barQl,Qsmem+128*64, 64,row_q);
  }
  if (tid==0){
    arrive_expect_tx(&barK[0], 16384);
    int rk = (int)(bh*S);
    tma_load_2d(&tmaK,&barK[0],Ksmem0,       0, rk);
    tma_load_2d(&tmaK,&barK[0],Ksmem0+64*64, 64,rk);
  }
  bar_wait(barQl,0);
  bar_wait(&barK[0], phK[0]); phK[0]^=1;
  if (tid==0){ do_qk(S0_taddr, Qsmem, Qsmem+128*64, Ksmem0, Ksmem0+64*64, idesc_qk); umma_commit_cg1(&barQ[0]); }

  float m_run = -INFINITY, l_run = 0.f;

  for (int kb=0; kb<=last_kb; kb++){
    uint32_t curS = (kb&1)? S1_taddr : S0_taddr;

    if (kb+1<=last_kb && tid==0){
      int nb=(kb+1)&1;
      __nv_bfloat16* Kb = nb?Ksmem1:Ksmem0;
      int rk=(int)(bh*S + (long)(kb+1)*64);
      arrive_expect_tx(&barK[nb], 16384);
      tma_load_2d(&tmaK,&barK[nb],Kb,       0, rk);
      tma_load_2d(&tmaK,&barK[nb],Kb+64*64, 64,rk);
    }

    // V transpose (global -> shared swizzled), overlaps with QK matmul
    for (int j=tid; j<1024; j+=128){
      int bn = j>>4, dch=j&15, d=dch<<3;
      int kpos = kb*64+bn;
      int4 data;
      if (kpos<S) data = *(const int4*)(V + (bh*S + kpos)*128 + d);
      else data = make_int4(0,0,0,0);
      const __nv_bfloat16* dv=(const __nv_bfloat16*)&data;
      #pragma unroll
      for (int e=0;e<8;e++){
        int dd=d+e;
        int phys = dd*64 + (((dd&7)^(bn>>3))<<3) + (bn&7);
        Vt[phys]=dv[e];
      }
    }

    bar_wait(&barQ[kb&1], phQ[kb&1]); phQ[kb&1]^=1;
    tcg_fence_after();

    // batch read S row (single wait)
    float val[64];
    #pragma unroll
    for (int i=0;i<16;i++){
      tmem_ld4(curS+i*4,
        reinterpret_cast<uint32_t*>(&val[i*4+0]),
        reinterpret_cast<uint32_t*>(&val[i*4+1]),
        reinterpret_cast<uint32_t*>(&val[i*4+2]),
        reinterpret_cast<uint32_t*>(&val[i*4+3]));
    }
    tcg_wait_ld();

    // issue QK[kb+1] (overlaps softmax)
    if (kb+1<=last_kb){
      int nb=(kb+1)&1;
      bar_wait(&barK[nb], phK[nb]); phK[nb]^=1;
      if (tid==0){
        __nv_bfloat16* Kb = nb?Ksmem1:Ksmem0;
        uint32_t nextS = nb? S1_taddr : S0_taddr;
        do_qk(nextS, Qsmem, Qsmem+128*64, Kb, Kb+64*64, idesc_qk);
        umma_commit_cg1(&barQ[nb]);
      }
    }

    // softmax
    float m_blk=-INFINITY;
    #pragma unroll
    for (int c=0;c<64;c++){
      int kpos=kb*64+c;
      float s;
      if (kpos<S && kpos<=qpos_i){ s=val[c]*scale; m_blk=fmaxf(m_blk,s); }
      else s=-INFINITY;
      val[c]=s;
    }
    float m_old=m_run;
    float m_new=fmaxf(m_old,m_blk);
    float corr=__expf(m_old-m_new);
    float sum_blk=0.f;
    #pragma unroll
    for (int c=0;c<64;c++){
      float p=(val[c]==-INFINITY)?0.f:__expf(val[c]-m_new);
      val[c]=p; sum_blk+=p;
    }
    l_run = l_run*corr + sum_blk;
    m_run = m_new;

    // write P (bf16, swizzle along bn)
    #pragma unroll
    for (int xc=0; xc<8; xc++){
      int phys = tid*64 + (((tid&7)^xc)<<3);
      __nv_bfloat16 tmp[8];
      #pragma unroll
      for (int e=0;e<8;e++) tmp[e]=__float2bfloat16(val[xc*8+e]);
      *(int4*)(Psmem+phys) = *(const int4*)tmp;
    }

    __syncthreads();
    fence_proxy_async();

    // per-warp rescale vote (no extra CTA syncs). corr==1 for non-bumped rows -> no-op.
    if (kb>0){
      bool need = (m_blk > m_old);
      unsigned vote = __ballot_sync(0xffffffffu, need);
      if (vote){
        tcg_fence_after();
        #pragma unroll
        for (int chunk=0; chunk<8; chunk++){
          uint32_t r[16];
          #pragma unroll
          for (int t=0;t<4;t++)
            tmem_ld4(O_taddr+(chunk*16+t*4), &r[t*4+0],&r[t*4+1],&r[t*4+2],&r[t*4+3]);
          tcg_wait_ld();
          #pragma unroll
          for (int t=0;t<4;t++)
            tmem_st4(O_taddr+(chunk*16+t*4),
              __float_as_uint(__uint_as_float(r[t*4+0])*corr),
              __float_as_uint(__uint_as_float(r[t*4+1])*corr),
              __float_as_uint(__uint_as_float(r[t*4+2])*corr),
              __float_as_uint(__uint_as_float(r[t*4+3])*corr));
        }
        tcg_wait_st();
      }
    }
    tcg_fence_before();
    __syncthreads();

    if (tid==0){
      tcg_fence_after();
      do_pv(O_taddr, Psmem, Vt, idesc_pv, kb==0);
      umma_commit_cg1(barP);
    }
    bar_wait(barP, ppv); ppv^=1;
    __syncthreads();
  }

  // epilogue: batch read O row
  tcg_fence_after();
  float inv = (qpos_i<S && l_run>0.f) ? (1.0f/l_run) : 0.f;
  for (int chunk=0; chunk<8; chunk++){
    uint32_t r[16];
    #pragma unroll
    for (int t=0;t<4;t++)
      tmem_ld4(O_taddr+(chunk*16+t*4), &r[t*4+0],&r[t*4+1],&r[t*4+2],&r[t*4+3]);
    tcg_wait_ld();
    if (qpos_i<S){
      __nv_bfloat16* op = O + (bh*S + qpos_i)*128 + chunk*16;
      #pragma unroll
      for (int e=0;e<16;e++) op[e]=__float2bfloat16(__uint_as_float(r[e])*inv);
    }
  }
  if (qpos_i<S) LSE[bh*S + qpos_i] = m_run + logf(l_run);

  __syncthreads();
  if (warp==0) tmem_dealloc_cg1(tmem_base, 256);
}

CUresult make_tma(CUtensorMap* d, void* p, uint64_t inner, uint64_t outer,
                  uint32_t binner, uint32_t bouter){
  uint64_t gdim[2]={inner,outer};
  uint64_t gstr[1]={inner*2};
  uint32_t bdim[2]={binner,bouter};
  uint32_t estr[2]={1,1};
  return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, p, gdim, gstr,
    bdim, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bsz = Q.size(0);
  int H   = Q.size(1);
  int S   = Q.size(2);

  __nv_bfloat16* Qp=(__nv_bfloat16*)Q.data_ptr();
  __nv_bfloat16* Kp=(__nv_bfloat16*)K.data_ptr();
  __nv_bfloat16* Vp=(__nv_bfloat16*)V.data_ptr();
  __nv_bfloat16* Op=(__nv_bfloat16*)O.data_ptr();
  float* Lp=(float*)LSE.data_ptr();

  uint64_t BHS = (uint64_t)Bsz*H*S;
  CUtensorMap tmaQ, tmaK;
  CU_CHECK(make_tma(&tmaQ, Qp, 128, BHS, 64, 128));
  CU_CHECK(make_tma(&tmaK, Kp, 128, BHS, 64, 64));

  int numqb=(S+127)/128;
  dim3 grid(numqb, H, Bsz), block(128);
  int smem_bytes = 102400;
  CUDA_CHECK(cudaFuncSetAttribute(attn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  attn<<<grid,block,smem_bytes,stream>>>(tmaQ,tmaK,Vp,Op,Lp,S,H);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

} // namespace mha_kernel