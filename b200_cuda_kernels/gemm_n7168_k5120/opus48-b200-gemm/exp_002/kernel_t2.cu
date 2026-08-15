#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s; cuGetErrorString(_e,&s); \
  fprintf(stderr,"CU error %s at %s:%d\n",s,__FILE__,__LINE__); exit(1);} } while(0)

namespace gemm_ns {

constexpr int BM=256, BN=256, BK=64, KS=BK/16, PIPE=3;
constexpr int SBO=8*BK*2; // 1024

// ---------- mbarrier ----------
__device__ __forceinline__ void init_bar(uint64_t* bar, uint32_t cnt){
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(cnt));
}
__device__ __forceinline__ void fence_bar_init(){ asm volatile("fence.mbarrier_init.release.cluster;":::"memory"); }
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t tx){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(tx):"memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t ph){
  asm volatile("{\n.reg .pred P;\nLW_%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra LW_%=;\n}\n"
    ::"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(ph));
}
__device__ __forceinline__ void mbar_arrive_cluster(uint64_t* bar, uint32_t target){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(bar), r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(r):"r"(a),"r"(target));
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"::"r"(r));
}
__device__ __forceinline__ void cluster_sync(){
  asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n":::"memory");
}
// ---------- TMA ----------
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* d, uint64_t* bar, void* smem, int c0, int c1){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3,%4}], [%2];"
    ::"r"((uint32_t)__cvta_generic_to_shared(smem)),"l"((uint64_t)d),
      "r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c0),"r"(c1):"memory");
}
__device__ __forceinline__ void tma_load_mc_2d(const CUtensorMap* d, uint64_t* bar, void* smem, int c0, int c1, uint16_t mask){
  uint64_t ch=0;
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%4,%5}], [%2], %3, %6;"
    ::"r"((uint32_t)__cvta_generic_to_shared(smem)),"l"((uint64_t)d),
      "r"((uint32_t)__cvta_generic_to_shared(bar)),"h"(mask),"r"(c0),"r"(c1),"l"(ch):"memory");
}
// ---------- tcgen05 ----------
__device__ __forceinline__ void tmem_alloc_cg1(uint32_t* dst, int ncols){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(dst)),"r"(ncols));
}
__device__ __forceinline__ void tmem_relinquish_cg1(){
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void tmem_dealloc_cg1(uint32_t addr, int ncols){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(addr),"r"(ncols));
}
__device__ __forceinline__ void umma_cg1(uint32_t tmem, uint64_t da, uint64_t db, uint32_t idesc, uint32_t accum){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    ::"r"(tmem),"l"(da),"l"(db),"r"(idesc),"r"(accum));
}
__device__ __forceinline__ void umma_commit_cg1(uint64_t* bar){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
    ::"r"((uint32_t)__cvta_generic_to_shared(bar)));
}
__device__ __forceinline__ void tmem_ld_x8(uint32_t col,
   uint32_t*r0,uint32_t*r1,uint32_t*r2,uint32_t*r3,uint32_t*r4,uint32_t*r5,uint32_t*r6,uint32_t*r7){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
    :"=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),"=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7):"r"(col));
}
__device__ __forceinline__ void tmem_wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void tc_fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }

// ---------- descriptors ----------
__device__ __forceinline__ uint64_t make_smem_desc(void* ptr){
  uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(ptr);
  d |= (uint64_t)(a & 0x3FFFF) >> 4;
  d |= (uint64_t)((1u & 0x3FFFF) >> 4) << 16;
  d |= (uint64_t)((SBO & 0x3FFFF) >> 4) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)2 << 61;   // 128B swizzle
  return d;
}
__device__ __forceinline__ uint32_t make_instr_desc(){
  uint32_t d=0;
  d |= (1u<<4);              // c fp32
  d |= (1u<<7);              // a bf16
  d |= (1u<<10);             // b bf16
  d |= (32u<<17);            // N=256 -> N/8
  d |= (8u<<24);             // M=128 -> M/16
  return d;
}

__global__ __launch_bounds__(128)
void gemm_kernel(const __grid_constant__ CUtensorMap tmaA,
                 const __grid_constant__ CUtensorMap tmaB,
                 __nv_bfloat16* C, int M, int N, int K){
  extern __shared__ char smem_raw[];
  uint32_t raw_s = (uint32_t)__cvta_generic_to_shared(smem_raw);
  uint32_t pad = ((raw_s + 1023u) & ~1023u) - raw_s;
  char* base = smem_raw + pad;

  __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(base);
  __nv_bfloat16* Bs = reinterpret_cast<__nv_bfloat16*>(base + PIPE*BM*BK*2);
  uint64_t* full  = reinterpret_cast<uint64_t*>(base + PIPE*BM*BK*2 + PIPE*BN*BK*2);
  uint64_t* empty = full + PIPE;
  uint64_t* bmma  = empty + PIPE;
  uint32_t* tmem_sm = reinterpret_cast<uint32_t*>(bmma + PIPE);

  int tid = threadIdx.x;
  int rank = blockIdx.x & 1;
  int cluster_n = blockIdx.x >> 1;
  int cluster_m = blockIdx.y;
  int Mbase = cluster_m * 512;
  int Nbase = cluster_n * 256;
  int m_start = Mbase + rank*256;
  int num_kb = K / BK;

  if (tid==0){
    for(int s=0;s<PIPE;s++){ init_bar(&full[s],1); init_bar(&empty[s],2); init_bar(&bmma[s],1); }
    fence_bar_init();
  }
  if (tid < 32){ tmem_alloc_cg1(tmem_sm, 512); tmem_relinquish_cg1(); }
  __syncthreads();
  cluster_sync();

  uint32_t tmem_base = *tmem_sm;
  uint32_t tmem_col0 = tmem_base & 0xFFFF;
  uint32_t idesc = make_instr_desc();
  uint32_t txb = (uint32_t)(BM*BK + BN*BK)*2; // 64KB

  if (tid==0){
    // consumer: issue UMMAs
    for (int kb=0; kb<num_kb; kb++){
      int s=kb%PIPE; uint32_t fp=(kb/PIPE)&1u;
      mbar_wait(&full[s], fp);
      __nv_bfloat16* Ap = As + s*BM*BK;
      __nv_bfloat16* Bp = Bs + s*BN*BK;
      #pragma unroll
      for(int j=0;j<KS;j++){
        uint32_t accum = (kb==0&&j==0)?0u:1u;
        uint64_t db  = make_smem_desc(Bp + 16*j);
        uint64_t da0 = make_smem_desc(Ap + 16*j);
        uint64_t da1 = make_smem_desc(Ap + 128*BK + 16*j);
        umma_cg1(tmem_base,     da0, db, idesc, accum);
        umma_cg1(tmem_base+256, da1, db, idesc, accum);
      }
      umma_commit_cg1(&bmma[s]);
      if (kb>=1){
        int pk=kb-1, ps=pk%PIPE; uint32_t pp=(pk/PIPE)&1u;
        mbar_wait(&bmma[ps], pp);
        mbar_arrive_cluster(&empty[ps], 0);
        mbar_arrive_cluster(&empty[ps], 1);
      }
    }
    { int pk=num_kb-1, ps=pk%PIPE; uint32_t pp=(pk/PIPE)&1u;
      mbar_wait(&bmma[ps], pp);
      mbar_arrive_cluster(&empty[ps], 0);
      mbar_arrive_cluster(&empty[ps], 1);
    }
  } else if (tid==32){
    // producer: TMA loads (A own, B multicast half)
    for (int kb=0; kb<num_kb; kb++){
      int s=kb%PIPE;
      if (kb>=PIPE){ uint32_t ep=((kb/PIPE)-1u)&1u; mbar_wait(&empty[s], ep); }
      mbar_arrive_expect_tx(&full[s], txb);
      int koff=kb*BK;
      __nv_bfloat16* Ap = As + s*BM*BK;
      __nv_bfloat16* Bp = Bs + s*BN*BK;
      tma_load_2d(&tmaA, &full[s], Ap, koff, m_start);
      int c1B = Nbase + rank*128;
      tma_load_mc_2d(&tmaB, &full[s], Bp + (rank*128)*BK, koff, c1B, (uint16_t)0x3);
    }
  }

  __syncthreads();
  cluster_sync();
  tc_fence_after();

  // ---- epilogue: two 128x256 sub-tiles ----
  __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(base);
  int warp = tid/32, lane = tid%32;
  for (int sub=0; sub<2; sub++){
    uint32_t tcol = tmem_col0 + sub*256;
    int rowbase = m_start + sub*128;
    for (int col=0; col<256; col+=8){
      uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
      tmem_ld_x8(tcol+col,&r0,&r1,&r2,&r3,&r4,&r5,&r6,&r7);
      tmem_wait_ld();
      int b = tid*256 + col;
      smem_out[b+0]=__float2bfloat16(__uint_as_float(r0));
      smem_out[b+1]=__float2bfloat16(__uint_as_float(r1));
      smem_out[b+2]=__float2bfloat16(__uint_as_float(r2));
      smem_out[b+3]=__float2bfloat16(__uint_as_float(r3));
      smem_out[b+4]=__float2bfloat16(__uint_as_float(r4));
      smem_out[b+5]=__float2bfloat16(__uint_as_float(r5));
      smem_out[b+6]=__float2bfloat16(__uint_as_float(r6));
      smem_out[b+7]=__float2bfloat16(__uint_as_float(r7));
    }
    __syncthreads();
    for (int step=0; step<32; step++){
      int row = step*4 + warp;
      int grow = rowbase + row;
      int col = lane*8;
      int gcol = Nbase + col;
      if (grow < M){
        int4 v = *reinterpret_cast<int4*>(&smem_out[row*256 + col]);
        *reinterpret_cast<int4*>(&C[(long)grow*N + gcol]) = v;
      }
    }
    __syncthreads();
  }
  if (tid<32){ tmem_dealloc_cg1(tmem_base, 512); }
}

static CUresult make_tma(CUtensorMap* d, void* ptr, uint64_t inner, uint64_t outer,
                         uint32_t box_inner, uint32_t box_outer){
  uint64_t gdim[2] = {inner, outer};
  uint64_t gstride[1] = {inner*2};
  uint32_t bdim[2] = {box_inner, box_outer};
  uint32_t estride[2] = {1,1};
  return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr,
      gdim, gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
  CUDA_CHECK(cudaSetDevice(A.device().device_id));
  int M = (int)A.size(0), K = (int)A.size(1), N = (int)B.size(0);
  __nv_bfloat16* Ap = static_cast<__nv_bfloat16*>(A.data_ptr());
  __nv_bfloat16* Bp = static_cast<__nv_bfloat16*>(B.data_ptr());
  __nv_bfloat16* Cp = static_cast<__nv_bfloat16*>(C.data_ptr());

  CUtensorMap tmaA, tmaB;
  CU_CHECK(make_tma(&tmaA, Ap, K, M, BK, 256));  // A: box (BK, 256 rows)
  CU_CHECK(make_tma(&tmaB, Bp, K, N, BK, 128));  // B: box (BK, 128 cols)

  int smem_bytes = 2*PIPE*BM*BK*2 + 1024 + 256; // As+Bs (BM=BN=256) + pad + bars
  CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  int clusters_n = N/256;
  int clusters_m = (M + 511)/512;

  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(2*clusters_n, clusters_m, 1);
  config.blockDim = dim3(128);
  config.dynamicSmemBytes = smem_bytes;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  config.attrs = attrs;
  config.numAttrs = 1;

  CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tmaA, tmaB, Cp, M, N, K));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_ns::run);

}  // namespace gemm_ns