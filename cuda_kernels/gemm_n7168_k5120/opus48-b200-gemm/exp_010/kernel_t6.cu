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

constexpr int BK = 64;
constexpr int NSTAGES = 4;
constexpr int A_TILE = 128*BK;   // 8192 (128 M per CTA)
constexpr int B_TILE = 128*BK;   // 8192 (128 N per CTA, per subtile)
constexpr int TX_BYTES = 2*(A_TILE + 2*B_TILE)*2; // both CTAs, A+B0+B1 = 98304
constexpr int TMEM_COLS = 512;
constexpr int GROUP_M = 8;

__device__ __forceinline__ uint32_t saddr(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ bool elect_one(){
  uint32_t pred;
  asm volatile("{\n.reg .pred p;\nelect.sync _|p, 0xFFFFFFFF;\nselp.b32 %0,1,0,p;\n}\n" : "=r"(pred));
  return pred!=0;
}
__device__ __forceinline__ uint32_t cluster_rank(){ uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void cluster_sync(){ asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory"); }

__device__ __forceinline__ uint64_t make_desc(void* ptr){
  uint64_t d=0; uint32_t addr = saddr(ptr);
  d |= (uint64_t)((addr & 0x3FFFF) >> 4);
  d |= (uint64_t)((1024u & 0x3FFFF) >> 4) << 32;  // SBO=1024
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)2 << 61;   // SWIZZLE_128B
  return d;
}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N){
  uint32_t d=0; d |= (1u<<4); d |= (1u<<7); d |= (1u<<10);
  d |= ((N/8)<<17); d |= ((M/16)<<24); return d;
}
__device__ __forceinline__ void umma_cg2(uint32_t tmem_c, uint64_t da, uint64_t db, uint32_t idesc, uint32_t accum){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::2.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    :: "r"(tmem_c),"l"(da),"l"(db),"r"(idesc),"r"(accum));
}
__device__ __forceinline__ void umma_commit_2sm(uint64_t* bar){
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0],%1;"
    :: "r"(saddr(bar)),"h"((uint16_t)0x3));
}
__device__ __forceinline__ void tmem_alloc_cg2(uint32_t* dst,int n){
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0],%1;" :: "r"(saddr(dst)),"r"(n));
}
__device__ __forceinline__ void tmem_relinquish_cg2(){ asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;"); }
__device__ __forceinline__ void tmem_dealloc_cg2(uint32_t addr,int n){ asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0,%1;" :: "r"(addr),"r"(n)); }
__device__ __forceinline__ void tcgen05_fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }

__device__ __forceinline__ void bar_init(uint64_t* b,uint32_t c){ asm volatile("mbarrier.init.shared.b64 [%0],%1;"::"r"(saddr(b)),"r"(c)); }
__device__ __forceinline__ void bar_arrive_expect(uint64_t* b,uint32_t tx){ asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(saddr(b)),"r"(tx):"memory"); }
__device__ __forceinline__ void bar_arrive(uint64_t* b){ asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(saddr(b)):"memory"); }
__device__ __forceinline__ void bar_arrive_cluster(uint64_t* b,uint32_t tgt){
  uint32_t a=saddr(b),ra; asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(ra):"r"(a),"r"(tgt));
  asm volatile("mbarrier.arrive.shared::cluster.b64 _,[%0];"::"r"(ra));
}
__device__ __forceinline__ void bar_wait(uint64_t* b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nWW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra WW%=;\n}\n"::"r"(saddr(b)),"r"(ph));
}
__device__ __forceinline__ void fence_bar_init(){ asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory"); }
__device__ __forceinline__ void named_bar(int id,int cnt){ asm volatile("barrier.sync.aligned %0, %1;" :: "r"(id),"r"(cnt)); }

__device__ __forceinline__ void tma_load_cg2(const CUtensorMap* d, uint64_t* bar, void* smem, int c0, int c1){
  uint32_t sa=saddr(smem), ba=saddr(bar)&0xFEFFFFFF;
  asm volatile(
    "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
    " [%0],[%1,{%2,%3}],[%4];"
    :: "r"(sa),"l"((uint64_t)d),"r"(c0),"r"(c1),"r"(ba):"memory");
}
__device__ __forceinline__ uint32_t pack2(float a, float b){
  __nv_bfloat162 v = __floats2bfloat162_rn(a,b);
  return *reinterpret_cast<uint32_t*>(&v);
}

// Grouped tile rasterization for L2 reuse. Maps linear index t -> (m, ng).
__device__ __forceinline__ void decode_tile(int t, int numM, int numNg, int& m, int& ng){
  int tiles_per_group = GROUP_M * numNg;
  int group_id = t / tiles_per_group;
  int first_m = group_id * GROUP_M;
  int gsize = numM - first_m; if (gsize > GROUP_M) gsize = GROUP_M;
  int idx = t - group_id * tiles_per_group;
  m  = first_m + (idx % gsize);
  ng = idx / gsize;
}

__global__ __launch_bounds__(192) void gemm_kernel(
   const __grid_constant__ CUtensorMap tmaA,
   const __grid_constant__ CUtensorMap tmaB,
   bf16* __restrict__ C, int M, int N, int K, int total, int num_clusters, int numM, int numNg)
{
  extern __shared__ char smem_raw[];
  uintptr_t pbase = ((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023;
  bf16* sA  = (bf16*)pbase;
  bf16* sB0 = sA  + NSTAGES*A_TILE;
  bf16* sB1 = sB0 + NSTAGES*B_TILE;
  uint64_t* full = (uint64_t*)(sB1 + NSTAGES*B_TILE);
  uint64_t* empty = full + NSTAGES;
  uint64_t* mma_done = empty + NSTAGES;
  uint64_t* tmem_free = mma_done + 1;
  uint32_t* tmem_base = (uint32_t*)(tmem_free + 1);

  int tid = threadIdx.x;
  int warpid = tid>>5;
  int rank = cluster_rank();
  int num_k = K / BK;
  int cluster_id = blockIdx.x >> 1;

  if (tid < NSTAGES){ bar_init(&full[tid],1); bar_init(&empty[tid],1); }
  if (tid==0){ bar_init(mma_done,1); bar_init(tmem_free,2); }
  if (warpid==0){ tmem_alloc_cg2(tmem_base, TMEM_COLS); tmem_relinquish_cg2(); }
  fence_bar_init();
  __syncthreads();
  cluster_sync();

  uint32_t tmem_c = *tmem_base;
  uint32_t idesc = make_idesc(256, 256);

  if (warpid < 4){
    // ---------------- Epilogue ----------------
    int gl=0;
    for (int t = cluster_id; t < total; t += num_clusters, gl++){
      int mtile, ngtile; decode_tile(t, numM, numNg, mtile, ngtile);
      int m_base = mtile*256 + rank*128;
      int n_out0 = ngtile*512;
      bar_wait(mma_done, gl&1);
      tcgen05_fence_after();
      uint32_t row = m_base + tid;
      bool valid = row < M;
      #pragma unroll
      for (int sub=0; sub<2; sub++){
        uint32_t cbase = sub*256;
        int n_out = n_out0 + sub*256;
        bf16* crow = C + (size_t)row*N + n_out;
        #pragma unroll
        for (int col=0; col<256; col+=32){
          uint32_t rr[32];
          #pragma unroll
          for (int j=0;j<4;j++){
            uint32_t ta = tmem_c + cbase + col + j*8;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
              : "=r"(rr[j*8+0]),"=r"(rr[j*8+1]),"=r"(rr[j*8+2]),"=r"(rr[j*8+3]),
                "=r"(rr[j*8+4]),"=r"(rr[j*8+5]),"=r"(rr[j*8+6]),"=r"(rr[j*8+7]) : "r"(ta));
          }
          asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
          #pragma unroll
          for (int j=0;j<4;j++){
            uint4 v;
            v.x = pack2(__uint_as_float(rr[j*8+0]),__uint_as_float(rr[j*8+1]));
            v.y = pack2(__uint_as_float(rr[j*8+2]),__uint_as_float(rr[j*8+3]));
            v.z = pack2(__uint_as_float(rr[j*8+4]),__uint_as_float(rr[j*8+5]));
            v.w = pack2(__uint_as_float(rr[j*8+6]),__uint_as_float(rr[j*8+7]));
            if (valid) *reinterpret_cast<uint4*>(crow+col+j*8) = v;
          }
        }
      }
      named_bar(1, 128);
      if (tid==0){
        if (rank==0) bar_arrive(tmem_free);
        else         bar_arrive_cluster(tmem_free, 0);
      }
    }
  } else if (warpid == 4){
    // ---------------- MMA (leader) ----------------
    if (rank==0 && elect_one()){
      int gl=0, kk=0;
      for (int t = cluster_id; t < total; t += num_clusters, gl++){
        if (gl>=1){ bar_wait(tmem_free, (gl-1)&1); }
        for (int i=0;i<num_k;i++,kk++){
          int s = kk % NSTAGES;
          uint32_t fph = (kk/NSTAGES)&1;
          bar_wait(&full[s], fph);
          bf16* bA  = sA  + s*A_TILE;
          bf16* bB0 = sB0 + s*B_TILE;
          bf16* bB1 = sB1 + s*B_TILE;
          #pragma unroll
          for (int ks=0; ks<BK/16; ks++){
            uint64_t da  = make_desc(bA  + ks*16);
            uint64_t db0 = make_desc(bB0 + ks*16);
            uint64_t db1 = make_desc(bB1 + ks*16);
            uint32_t acc = (i==0 && ks==0)?0:1;
            umma_cg2(tmem_c,     da, db0, idesc, acc);
            umma_cg2(tmem_c+256, da, db1, idesc, acc);
          }
          umma_commit_2sm(&empty[s]);
        }
        umma_commit_2sm(mma_done);
      }
    }
  } else {
    // ---------------- TMA producer (both CTAs) ----------------
    if (elect_one()){
      int gl=0, kk=0;
      for (int t = cluster_id; t < total; t += num_clusters, gl++){
        int mtile, ngtile; decode_tile(t, numM, numNg, mtile, ngtile);
        int m_base = mtile*256 + rank*128;
        int n_b0   = ngtile*512 + rank*128;
        int n_b1   = ngtile*512 + 256 + rank*128;
        for (int i=0;i<num_k;i++,kk++){
          int s = kk % NSTAGES;
          if (kk >= NSTAGES){ uint32_t eph = ((kk/NSTAGES)-1)&1; bar_wait(&empty[s], eph); }
          if (rank==0) bar_arrive_expect(&full[s], TX_BYTES);
          int koff = i*BK;
          tma_load_cg2(&tmaA, &full[s], sA  + s*A_TILE, koff, m_base);
          tma_load_cg2(&tmaB, &full[s], sB0 + s*B_TILE, koff, n_b0);
          tma_load_cg2(&tmaB, &full[s], sB1 + s*B_TILE, koff, n_b1);
        }
      }
    }
  }

  __syncthreads();
  cluster_sync();
  if (warpid==0){ tmem_dealloc_cg2(tmem_c, TMEM_COLS); }
}

static CUresult make_tma_desc(CUtensorMap* d, void* g, uint64_t inner, uint64_t outer, uint32_t bi, uint32_t bo){
  uint64_t gd[2]={inner,outer}; uint64_t gs[1]={inner*2};
  uint32_t bd[2]={bi,bo}; uint32_t es[2]={1,1};
  return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, g, gd, gs, bd, es,
     CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
     CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
  CUDA_CHECK(cudaSetDevice(A.device().device_id));
  int64_t M = A.size(0), K = A.size(1), N = B.size(0);
  bf16* Ap = static_cast<bf16*>(A.data_ptr());
  bf16* Bp = static_cast<bf16*>(B.data_ptr());
  bf16* Cp = static_cast<bf16*>(C.data_ptr());
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

  CUtensorMap dA, dB;
  CU_CHECK(make_tma_desc(&dA, Ap, (uint64_t)K, (uint64_t)M, 64, 128));
  CU_CHECK(make_tma_desc(&dB, Bp, (uint64_t)K, (uint64_t)N, 64, 128));

  size_t smem = 1024 + (size_t)NSTAGES*(A_TILE+2*B_TILE)*2 + (size_t)(2*NSTAGES+2)*8 + 64;
  CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

  int numNg = (int)((N+511)/512);
  int numM  = (int)((M+255)/256);
  int total = numM*numNg;

  int dev; CUDA_CHECK(cudaGetDevice(&dev));
  int smcount; CUDA_CHECK(cudaDeviceGetAttribute(&smcount, cudaDevAttrMultiProcessorCount, dev));
  int numCTA = smcount; if (numCTA & 1) numCTA--;
  if (numCTA/2 > total) numCTA = total*2;
  if (numCTA < 2) numCTA = 2;
  int num_clusters = numCTA/2;

  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(numCTA, 1, 1);
  config.blockDim = dim3(192);
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 2;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  config.attrs = attr;
  config.numAttrs = 1;

  CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, dA, dB, Cp, (int)M, (int)N, (int)K, total, num_clusters, numM, numNg));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120::run);

}  // namespace gemm_n7168_k5120