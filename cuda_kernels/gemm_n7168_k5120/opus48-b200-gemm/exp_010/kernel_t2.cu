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

constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 64;
constexpr int NSTAGES = 4;
constexpr int TMEM_COLS = 256;
constexpr int A_TILE = BM*BK;     // 8192 elems
constexpr int B_TILE = BN*BK;     // 16384 elems
constexpr int TX_BYTES = (A_TILE + B_TILE)*2; // 49152

__device__ __forceinline__ uint32_t saddr(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ bool elect_one(){
  uint32_t pred;
  asm volatile("{\n.reg .pred p;\nelect.sync _|p, 0xFFFFFFFF;\nselp.b32 %0,1,0,p;\n}\n" : "=r"(pred));
  return pred!=0;
}

// K-major, 128B swizzle descriptor. K-slice selected by advancing ptr by 16 bf16.
__device__ __forceinline__ uint64_t make_desc(void* ptr){
  uint64_t d=0;
  uint32_t addr = saddr(ptr);
  d |= (uint64_t)((addr & 0x3FFFF) >> 4);            // start address
  d |= (uint64_t)((1024u & 0x3FFFF) >> 4) << 32;     // SBO = 1024
  d |= (uint64_t)1 << 46;                            // fixed 001
  d |= (uint64_t)2 << 61;                            // SWIZZLE_128B
  return d;
}

__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N){
  uint32_t d=0;
  d |= (1u<<4);    // D FP32
  d |= (1u<<7);    // A BF16
  d |= (1u<<10);   // B BF16
  d |= ((N/8)<<17);
  d |= ((M/16)<<24);
  return d;
}

__device__ __forceinline__ void umma_cg1(uint32_t tmem_c, uint64_t da, uint64_t db, uint32_t idesc, uint32_t accum){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    :: "r"(tmem_c),"l"(da),"l"(db),"r"(idesc),"r"(accum));
}
__device__ __forceinline__ void umma_commit_cg1(uint64_t* bar){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(saddr(bar)));
}
__device__ __forceinline__ void tmem_alloc_cg1(uint32_t* dst,int n){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;" :: "r"(saddr(dst)),"r"(n));
}
__device__ __forceinline__ void tmem_relinquish_cg1(){
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void tmem_dealloc_cg1(uint32_t addr,int n){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;" :: "r"(addr),"r"(n));
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

__device__ __forceinline__ void tma_load(const CUtensorMap* d, uint64_t* bar, void* smem, int c0, int c1){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4}],[%2];"
    :: "r"(saddr(smem)),"l"((uint64_t)d),"r"(saddr(bar)),"r"(c0),"r"(c1):"memory");
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
  bf16* sC = sA; // reuse for epilogue staging

  int tid = threadIdx.x;
  int warpid = tid>>5;
  int num_k = K / BK;
  int m_block = blockIdx.y * BM;
  int n_block = blockIdx.x * BN;

  if (tid < NSTAGES){ bar_init(&full[tid],1); bar_init(&empty[tid],1); }
  if (tid==0){ bar_init(mma_done,1); }
  if (warpid==0){ tmem_alloc_cg1(tmem_base, TMEM_COLS); tmem_relinquish_cg1(); }
  fence_bar_init();
  __syncthreads();

  uint32_t tmem_c = *tmem_base;
  uint32_t idesc = make_idesc(BM, BN);
  bool elected = elect_one();

  if (warpid==1 && elected){
    // producer: TMA loads
    for (int i=0;i<num_k;i++){
      int s = i % NSTAGES;
      if (i >= NSTAGES){
        uint32_t eph = ((i/NSTAGES)-1)&1;
        bar_wait(&empty[s], eph);
      }
      bar_arrive_expect(&full[s], TX_BYTES);
      int koff = i*BK;
      tma_load(&tmaA, &full[s], sA + s*A_TILE, koff, m_block);
      tma_load(&tmaB, &full[s], sB + s*B_TILE, koff, n_block);
    }
  } else if (warpid==0 && elected){
    // consumer: UMMA
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
        umma_cg1(tmem_c, da, db, idesc, accum);
      }
      umma_commit_cg1(&empty[s]);
    }
    umma_commit_cg1(mma_done);
    bar_wait(mma_done, 0);
  }
  __syncthreads();
  tcgen05_fence_after();

  // Epilogue phase 1: TMEM -> sC (each thread owns row = tid)
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

  // Epilogue phase 2: coalesced sC -> C
  int total_vec = (BM*BN)/4;
  for (int v=tid; v<total_vec; v+=128){
    int row = v/(BN/4);
    int cc  = (v%(BN/4))*4;
    int gr = m_block + row;
    int gc = n_block + cc;
    if (gr < M){
      uint2 data = *reinterpret_cast<uint2*>(&sC[row*BN + cc]);
      *reinterpret_cast<uint2*>(&C[(size_t)gr*N + gc]) = data;
    }
  }
  __syncthreads();
  if (warpid==0){ tmem_dealloc_cg1(tmem_c, TMEM_COLS); }
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
  CU_CHECK(make_tma_desc(&dA, Ap, (uint64_t)K, (uint64_t)M, 64, BM)); // A [M,K]
  CU_CHECK(make_tma_desc(&dB, Bp, (uint64_t)K, (uint64_t)N, 64, BN)); // B [N,K]

  size_t smem = 1024 + (size_t)NSTAGES*(A_TILE+B_TILE)*2 + 256;
  CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

  dim3 grid((int)((N+BN-1)/BN), (int)((M+BM-1)/BM));
  dim3 block(128);
  gemm_kernel<<<grid, block, smem, stream>>>(dA, dB, Cp, (int)M, (int)N, (int)K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120::run);

}  // namespace gemm_n7168_k5120