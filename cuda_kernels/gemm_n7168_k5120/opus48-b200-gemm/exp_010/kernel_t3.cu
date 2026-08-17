#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); exit(1);} } while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s; cuGetErrorString(_e,&s); \
  fprintf(stderr,"CU error %s at %s:%d\n", s, __FILE__, __LINE__); exit(1);} } while(0)

namespace gemm_n7168_k5120 {

using bf16 = __nv_bfloat16;

constexpr int BM_CTA = 128;   // M rows per CTA
constexpr int BN_CTA = 128;   // N cols loaded per CTA (combined N=256)
constexpr int BN = 256;       // combined N
constexpr int BK = 64;
constexpr int NSTAGES = 6;
constexpr int TMEM_COLS = 256;
constexpr int A_TILE = BM_CTA*BK;   // 8192
constexpr int B_TILE = BN_CTA*BK;   // 8192
constexpr int TX_BYTES = 2*(A_TILE + B_TILE)*2; // both CTAs -> leader barrier = 65536

__device__ __forceinline__ uint32_t saddr(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ bool elect_one(){
  uint32_t pred;
  asm volatile("{\n.reg .pred p;\nelect.sync _|p, 0xFFFFFFFF;\nselp.b32 %0,1,0,p;\n}\n" : "=r"(pred));
  return pred!=0;
}
__device__ __forceinline__ uint32_t cluster_rank(){
  uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r;
}
__device__ __forceinline__ void cluster_sync(){
  asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}

__device__ __forceinline__ uint64_t make_desc(void* ptr){
  uint64_t d=0;
  uint32_t addr = saddr(ptr);
  d |= (uint64_t)((addr & 0x3FFFF) >> 4);
  d |= (uint64_t)((1024u & 0x3FFFF) >> 4) << 32;  // SBO=1024
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)2 << 61;   // SWIZZLE_128B
  return d;
}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N){
  uint32_t d=0;
  d |= (1u<<4); d |= (1u<<7); d |= (1u<<10);
  d |= ((N/8)<<17); d |= ((M/16)<<24);
  return d;
}

__device__ __forceinline__ void umma_cg2(uint32_t tmem_c, uint64_t da, uint64_t db, uint32_t idesc, uint32_t accum){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::2.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    :: "r"(tmem_c),"l"(da),"l"(db),"r"(idesc),"r"(accum));
}
__device__ __forceinline__ void umma_commit_2sm(uint64_t* bar){
  uint32_t a = saddr(bar);
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0],%1;"
    :: "r"(a),"h"((uint16_t)0x3));
}
__device__ __forceinline__ void tmem_alloc_cg2(uint32_t* dst,int n){
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0],%1;" :: "r"(saddr(dst)),"r"(n));
}
__device__ __forceinline__ void tmem_relinquish_cg2(){
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
}
__device__ __forceinline__ void tmem_dealloc_cg2(uint32_t addr,int n){
  asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0,%1;" :: "r"(addr),"r"(n));
}
__device__ __forceinline__ void tcgen05_fence_after(){
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void bar_init(uint64_t* b,uint32_t c){ asm volatile("mbarrier.init.shared.b64 [%0],%1;"::"r"(saddr(b)),"r"(c)); }
__device__ __forceinline__ void bar_arrive_expect(uint64_t* b,uint32_t tx){ asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(saddr(b)),"r"(tx):"memory"); }
__device__ __forceinline__ void bar_wait(uint64_t* b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nWW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra WW%=;\n}\n"::"r"(saddr(b)),"r"(ph));
}
__device__ __forceinline__ void fence_bar_init(){ asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory"); }

__device__ __forceinline__ void tma_load_cg2(const CUtensorMap* d, uint64_t* bar, void* smem, int c0, int c1){
  uint32_t sa = saddr(smem);
  uint32_t ba = saddr(bar) & 0xFEFFFFFF;
  asm volatile(
    "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
    " [%0],[%1,{%2,%3}],[%4];"
    :: "r"(sa),"l"((uint64_t)d),"r"(c0),"r"(c1),"r"(ba):"memory");
}

__global__ __launch_bounds__(128) void gemm_kernel(
   const __grid_constant__ CUtensorMap tmaA,
   const __grid_constant__ CUtensorMap tmaB,
   bf16* __restrict__ C, int M, int N, int K)
{
  extern __shared__ char smem_raw[];
  uintptr_t pbase = ((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023;
  bf16* sA = (bf16*)pbase;
  bf16* sB = sA + NSTAGES*A_TILE;
  uint64_t* full = (uint64_t*)(sB + NSTAGES*B_TILE);
  uint64_t* empty = full + NSTAGES;
  uint64_t* mma_done = empty + NSTAGES;
  uint32_t* tmem_base = (uint32_t*)(mma_done + 1);
  bf16* sC = sA;

  int tid = threadIdx.x;
  int warpid = tid>>5;
  int rank = cluster_rank();
  int num_k = K / BK;
  int mt = blockIdx.y;
  int nt = blockIdx.x / 2;

  int m_base = mt*256 + rank*128;   // A load row / output row base
  int n_base = nt*256 + rank*128;   // B load N base
  int n_out  = nt*256;              // output N base (full 256)

  if (tid < NSTAGES){ bar_init(&full[tid],1); bar_init(&empty[tid],1); }
  if (tid==0){ bar_init(mma_done,1); }
  if (warpid==0){ tmem_alloc_cg2(tmem_base, TMEM_COLS); tmem_relinquish_cg2(); }
  fence_bar_init();
  __syncthreads();
  cluster_sync();

  uint32_t tmem_c = *tmem_base;
  uint32_t idesc = make_idesc(256, 256);

  bool w0_lead = false, w1_lead = false;
  if (warpid==0) w0_lead = elect_one();
  if (warpid==1) w1_lead = elect_one();

  if (warpid==1 && w1_lead){
    // producer (both CTAs)
    for (int i=0;i<num_k;i++){
      int s = i % NSTAGES;
      if (i >= NSTAGES){
        uint32_t eph = ((i/NSTAGES)-1)&1;
        bar_wait(&empty[s], eph);
      }
      if (rank==0) bar_arrive_expect(&full[s], TX_BYTES);
      int koff = i*BK;
      tma_load_cg2(&tmaA, &full[s], sA + s*A_TILE, koff, m_base);
      tma_load_cg2(&tmaB, &full[s], sB + s*B_TILE, koff, n_base);
    }
  } else if (warpid==0 && rank==0 && w0_lead){
    // consumer (leader issues combined MMA)
    for (int i=0;i<num_k;i++){
      int s = i % NSTAGES;
      uint32_t fph = (i/NSTAGES)&1;
      bar_wait(&full[s], fph);
      bf16* bufA = sA + s*A_TILE;
      bf16* bufB = sB + s*B_TILE;
      #pragma unroll
      for (int ks=0; ks<BK/16; ks++){
        uint64_t da = make_desc(bufA + ks*16);
        uint64_t db = make_desc(bufB + ks*16);
        uint32_t accum = (i==0 && ks==0)?0:1;
        umma_cg2(tmem_c, da, db, idesc, accum);
      }
      umma_commit_2sm(&empty[s]);
    }
    umma_commit_2sm(mma_done);
  }

  bar_wait(mma_done, 0);
  __syncthreads();
  tcgen05_fence_after();

  // Epilogue phase 1: TMEM -> sC
  for (int col=0; col<BN; col+=4){
    uint32_t r0,r1,r2,r3;
    uint32_t taddr = tmem_c + col;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];"
      : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(taddr));
    asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
    int b = tid*BN + col;
    sC[b+0]=__float2bfloat16(__uint_as_float(r0));
    sC[b+1]=__float2bfloat16(__uint_as_float(r1));
    sC[b+2]=__float2bfloat16(__uint_as_float(r2));
    sC[b+3]=__float2bfloat16(__uint_as_float(r3));
  }
  __syncthreads();
  cluster_sync();
  if (warpid==0){ tmem_dealloc_cg2(tmem_c, TMEM_COLS); }

  // Epilogue phase 2: sC -> global (uint4 coalesced)
  int total_vec = (BM_CTA*BN)/8; // 4096
  for (int v=tid; v<total_vec; v+=128){
    int row = v/(BN/8);
    int cc  = (v%(BN/8))*8;
    int gr = m_base + row;
    int gc = n_out + cc;
    if (gr < M){
      uint4 data = *reinterpret_cast<uint4*>(&sC[row*BN + cc]);
      *reinterpret_cast<uint4*>(&C[(size_t)gr*N + gc]) = data;
    }
  }
}

static CUresult make_tma_desc(CUtensorMap* d, void* g, uint64_t inner, uint64_t outer, uint32_t bi, uint32_t bo){
  uint64_t gd[2]={inner,outer};
  uint64_t gs[1]={inner*2};
  uint32_t bd[2]={bi,bo};
  uint32_t es[2]={1,1};
  return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, g, gd, gs, bd, es,
     CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
     CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
  CUDA_CHECK(cudaSetDevice(A.device().device_id));
  int64_t M = A.size(0);
  int64_t K = A.size(1);
  int64_t N = B.size(0);
  bf16* Ap = static_cast<bf16*>(A.data_ptr());
  bf16* Bp = static_cast<bf16*>(B.data_ptr());
  bf16* Cp = static_cast<bf16*>(C.data_ptr());
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

  CUtensorMap dA, dB;
  CU_CHECK(make_tma_desc(&dA, Ap, (uint64_t)K, (uint64_t)M, 64, 128));
  CU_CHECK(make_tma_desc(&dB, Bp, (uint64_t)K, (uint64_t)N, 64, 128));

  size_t bar_bytes = (size_t)(2*NSTAGES+1)*8 + 64;
  size_t smem = 1024 + (size_t)NSTAGES*(A_TILE+B_TILE)*2 + bar_bytes;
  CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

  int numN = (int)((N+255)/256);
  int numM = (int)((M+255)/256);

  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(2*numN, numM, 1);
  config.blockDim = dim3(128);
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 2;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  config.attrs = attr;
  config.numAttrs = 1;

  CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, dA, dB, Cp, (int)M, (int)N, (int)K));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120::run);

}  // namespace gemm_n7168_k5120