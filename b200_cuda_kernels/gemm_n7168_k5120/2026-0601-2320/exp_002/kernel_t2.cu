#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

namespace gemm_n7168_k5120 {

constexpr int ROWS = 128;          // per-CTA A rows / B rows loaded
constexpr int NTILE = 256;         // full output N
constexpr int BK = 64;
constexpr int STAGES = 6;
constexpr int KKN = BK / 16;       // 4 MMAs per K-chunk
constexpr int STAGE_ELEMS = ROWS * BK;            // bf16 elements per A or B stage tile
constexpr int LOAD_BYTES = ROWS * BK * 2;         // 16384
constexpr uint32_t TX_TOTAL = 4u * LOAD_BYTES;    // 65536 (both CTAs A+B)

__device__ __forceinline__ void init_bar(uint64_t* bar, uint32_t count){
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"
    :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}
__device__ __forceinline__ void bar_arrive_expect(uint64_t* bar, uint32_t tx){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
    :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx) : "memory");
}
__device__ __forceinline__ void bar_wait(uint64_t* bar, uint32_t phase){
  asm volatile("{\n.reg .pred P;\nLAB_%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra LAB_%=;\n}\n"
    :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}
__device__ __forceinline__ uint32_t map_to_cta(uint32_t a, uint32_t cta){
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(cta));
  return r;
}
__device__ __forceinline__ void tma_cg2(const CUtensorMap* d, uint32_t bar_addr, void* smem, int c0, int c1){
  asm volatile(
    "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
    " [%0], [%1, {%2, %3}], [%4];"
    :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"((uint64_t)d),
       "r"(c0), "r"(c1), "r"(bar_addr) : "memory");
}
__device__ __forceinline__ uint64_t make_desc(void* smem, uint32_t lbo, uint32_t sbo){
  uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(smem);
  d |= (uint64_t)((a & 0x3FFFF) >> 4);
  d |= ((uint64_t)((lbo & 0x3FFFF)>>4))<<16;
  d |= ((uint64_t)((sbo & 0x3FFFF)>>4))<<32;
  d |= (uint64_t)1<<46;
  d |= (uint64_t)2<<61;   // 128B swizzle
  return d;
}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N){
  uint32_t d=0;
  d |= (1u<<4);    // D = F32
  d |= (1u<<7);    // A = BF16
  d |= (1u<<10);   // B = BF16
  d |= ((N>>3)<<17);
  d |= ((M>>4)<<24);
  return d;
}
__device__ __forceinline__ void tmem_alloc(uint32_t* dst, int ncols){
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
    :: "r"((uint32_t)__cvta_generic_to_shared(dst)), "r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc(uint32_t addr, int ncols){
  asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}
__device__ __forceinline__ void umma_cg2(uint32_t tmem_c, uint64_t da, uint64_t db, uint32_t idesc, uint32_t accum){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::2.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    :: "r"(tmem_c), "l"(da), "l"(db), "r"(idesc), "r"(accum));
}
__device__ __forceinline__ void umma_commit_2sm(uint64_t* bar){
  uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
    :: "r"(a), "h"((uint16_t)0x3));
}
__device__ __forceinline__ void cluster_sync(){
  asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}

__global__ void __launch_bounds__(128) gemm(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C, int M, int N, int K)
{
  extern __shared__ char raw[];
  uintptr_t al = ((uintptr_t)raw + 1023) & ~(uintptr_t)1023;
  char* smem = (char*)al;
  __nv_bfloat16* smemA = (__nv_bfloat16*)smem;
  __nv_bfloat16* smemB = smemA + STAGES*STAGE_ELEMS;
  char* barbase = (char*)(smemB + STAGES*STAGE_ELEMS);
  uint64_t* full = (uint64_t*)barbase;
  uint64_t* empty = full + STAGES;
  uint64_t* mma_final = empty + STAGES;
  uint32_t* tmem_smem = (uint32_t*)(mma_final + 1);

  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int NK = K / BK;

  int rank   = blockIdx.x & 1;       // cluster_ctarank
  int n_tile = blockIdx.x >> 1;
  int m_sup  = blockIdx.y;
  int m_base = m_sup * 256;
  int n_base = n_tile * 256;
  int my_m = m_base + rank * ROWS;   // this CTA's A rows
  int my_n = n_base + rank * ROWS;   // this CTA's B rows

  uint32_t idesc = make_idesc(256, 256);

  if (tid == 0){
    #pragma unroll
    for (int s=0;s<STAGES;++s){ init_bar(&full[s],1); init_bar(&empty[s],1); }
    init_bar(mma_final,1);
  }
  if (warp==0) tmem_alloc(tmem_smem, NTILE);
  cluster_sync();
  uint32_t tmem_base = *tmem_smem;

  // ---- Producer (both CTAs): load own A and B, signal leader's full[] ----
  if (warp==0 && lane==0){
    int ph[STAGES];
    #pragma unroll
    for(int s=0;s<STAGES;++s) ph[s]=0;
    for (int k=0;k<NK;++k){
      int s=k%STAGES;
      if (k>=STAGES){ bar_wait(&empty[s], ph[s]); ph[s]^=1; }
      uint32_t rbar = map_to_cta((uint32_t)__cvta_generic_to_shared(&full[s]), 0);
      if (rank==0) bar_arrive_expect(&full[s], TX_TOTAL);
      int koff = k*BK;
      tma_cg2(&tma_A, rbar, smemA + s*STAGE_ELEMS, koff, my_m);
      tma_cg2(&tma_B, rbar, smemB + s*STAGE_ELEMS, koff, my_n);
    }
  }
  // ---- Consumer / MMA (leader only) ----
  else if (rank==0 && warp==1 && lane==0){
    int ph[STAGES];
    #pragma unroll
    for(int s=0;s<STAGES;++s) ph[s]=0;
    for (int k=0;k<NK;++k){
      int s=k%STAGES;
      bar_wait(&full[s], ph[s]); ph[s]^=1;
      uint64_t da = make_desc(smemA + s*STAGE_ELEMS, 1, 1024);
      uint64_t db = make_desc(smemB + s*STAGE_ELEMS, 1, 1024);
      #pragma unroll
      for (int kk=0;kk<KKN;++kk){
        umma_cg2(tmem_base, da+(uint64_t)(2*kk), db+(uint64_t)(2*kk), idesc, (k==0&&kk==0)?0u:1u);
      }
      umma_commit_2sm(&empty[s]);
    }
    umma_commit_2sm(&mma_final);
  }

  bar_wait(&mma_final, 0);
  __syncthreads();

  // ---- Epilogue: TMEM -> SMEM -> coalesced global ----
  __nv_bfloat16* smem_out = (__nv_bfloat16*)smemA;
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  #pragma unroll
  for (int col=0; col<NTILE; col+=8){
    uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
    uint32_t a = tmem_base + (uint32_t)col;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
      : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(a));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    int b = tid*NTILE + col;
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
  #pragma unroll
  for (int step=0; step<ROWS/4; ++step){
    int row = step*4 + warp;
    int grow = my_m + row;
    int gcol = n_base + lane*8;
    if (grow < M){
      uint4 dd = *reinterpret_cast<uint4*>(&smem_out[row*NTILE + lane*8]);
      *reinterpret_cast<uint4*>(C + (long long)grow*N + gcol) = dd;
    }
  }
  __syncthreads();
  cluster_sync();
  if (warp==0) tmem_dealloc(tmem_base, NTILE);
}

static CUresult make_tma(CUtensorMap* d, void* gptr, uint64_t inner, uint64_t outer, uint32_t binner, uint32_t bouter){
  uint64_t gdim[2] = {inner, outer};
  uint64_t gstr[1] = {inner*2};
  uint32_t bdim[2] = {binner, bouter};
  uint32_t estr[2] = {1,1};
  return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gptr, gdim, gstr, bdim, estr,
    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
  cudaSetDevice(A.device().device_id);
  int M = (int)A.size(0);
  int K = (int)A.size(1);
  int N = (int)B.size(0);
  __nv_bfloat16* a = (__nv_bfloat16*)A.data_ptr();
  __nv_bfloat16* b = (__nv_bfloat16*)B.data_ptr();
  __nv_bfloat16* c = (__nv_bfloat16*)C.data_ptr();
  CUtensorMap tA, tB;
  make_tma(&tA, a, (uint64_t)K, (uint64_t)M, BK, ROWS);
  make_tma(&tB, b, (uint64_t)K, (uint64_t)N, BK, ROWS);

  int n_tiles = N / 256;
  int m_super = (M + 255) / 256;
  dim3 grid(2*n_tiles, m_super, 1);
  dim3 block(128);
  size_t smem_bytes = (size_t)2*STAGES*STAGE_ELEMS*2 + (size_t)(2*STAGES+1)*8 + 8 + 1024;
  cudaFuncSetAttribute(gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);

  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(A.device().device_type, A.device().device_id);

  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem_bytes;
  config.stream = stream;
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = 2;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;
  config.attrs = attribute;
  config.numAttrs = 1;
  cudaLaunchKernelEx(&config, gemm, tA, tB, c, M, N, K);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120::run);

}  // namespace gemm_n7168_k5120