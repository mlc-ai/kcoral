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

constexpr int BM=128, BN=256, BK=64, KS=BK/16, PIPE=4;
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
// ---------- TMA ----------
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* d, uint64_t* bar, void* smem, int c0, int c1){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3,%4}], [%2];"
    ::"r"((uint32_t)__cvta_generic_to_shared(smem)),"l"((uint64_t)d),
      "r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c0),"r"(c1):"memory");
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
__device__ __forceinline__ void tmem_load_x4(uint32_t col, uint32_t*r0,uint32_t*r1,uint32_t*r2,uint32_t*r3){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];"
    :"=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3):"r"(col));
}
__device__ __forceinline__ void tmem_wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void tc_fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }

// ---------- descriptors ----------
__device__ __forceinline__ uint64_t make_smem_desc(void* ptr){
  uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(ptr);
  d |= (uint64_t)(a & 0x3FFFF) >> 4;
  d |= (uint64_t)((1u & 0x3FFFF) >> 4) << 16;      // LBO unused
  d |= (uint64_t)((SBO & 0x3FFFF) >> 4) << 32;     // SBO
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)2 << 61;   // 128B swizzle
  return d;
}
__device__ __forceinline__ uint32_t make_instr_desc(){
  uint32_t d=0;
  d |= (1u<<4);              // c fp32
  d |= (1u<<7);              // a bf16
  d |= (1u<<10);             // b bf16
  d |= ((uint32_t)(BN/8)<<17);
  d |= ((uint32_t)(BM/16)<<24);
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
  uint64_t* bar_full = reinterpret_cast<uint64_t*>(base + PIPE*BM*BK*2 + PIPE*BN*BK*2);
  uint64_t* bar_mma  = bar_full + PIPE;
  uint64_t* bar_epi  = bar_mma + PIPE;
  uint32_t* tmem_sm  = reinterpret_cast<uint32_t*>(bar_epi + 1);

  int tid = threadIdx.x;
  int m_block = blockIdx.y;
  int n_block = blockIdx.x;
  int num_kb = K / BK;

  if (tid == 0){
    for (int s=0;s<PIPE;s++){ init_bar(&bar_full[s],1); init_bar(&bar_mma[s],1); }
    init_bar(bar_epi,1);
    fence_bar_init();
  }
  if (tid < 32){
    tmem_alloc_cg1(tmem_sm, BN);
    tmem_relinquish_cg1();
  }
  __syncthreads();

  uint32_t tmem_base = *tmem_sm;
  uint32_t tmem_col0 = tmem_base & 0xFFFF;
  uint32_t idesc = make_instr_desc();
  uint32_t txb = (BM*BK + BN*BK)*2;

  if (tid == 0){
    // consumer: MMA
    for (int kb=0; kb<num_kb; kb++){
      int s = kb % PIPE;
      uint32_t parity = (kb / PIPE) & 1u;
      mbar_wait(&bar_full[s], parity);
      __nv_bfloat16* Ap = As + s*BM*BK;
      __nv_bfloat16* Bp = Bs + s*BN*BK;
      #pragma unroll
      for (int j=0;j<KS;j++){
        uint64_t da = make_smem_desc(Ap + 16*j);
        uint64_t db = make_smem_desc(Bp + 16*j);
        uint32_t accum = (kb==0 && j==0) ? 0u : 1u;
        umma_cg1(tmem_base, da, db, idesc, accum);
      }
      umma_commit_cg1(&bar_mma[s]);
    }
    umma_commit_cg1(bar_epi);
    mbar_wait(bar_epi, 0);
  } else if (tid == 32){
    // producer: TMA
    for (int kb=0; kb<num_kb; kb++){
      int s = kb % PIPE;
      if (kb >= PIPE){
        uint32_t mp = ((kb / PIPE) - 1u) & 1u;
        mbar_wait(&bar_mma[s], mp);
      }
      mbar_arrive_expect_tx(&bar_full[s], txb);
      tma_load_2d(&tmaA, &bar_full[s], As + s*BM*BK, kb*BK, m_block*BM);
      tma_load_2d(&tmaB, &bar_full[s], Bs + s*BN*BK, kb*BK, n_block*BN);
    }
  }

  __syncthreads();
  tc_fence_after();

  // ---- epilogue ----
  __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(base);
  int row = tid;
  #pragma unroll
  for (int col=0; col<BN; col+=4){
    uint32_t r0,r1,r2,r3;
    tmem_load_x4(tmem_col0 + col, &r0,&r1,&r2,&r3);
    tmem_wait_ld();
    int b = row*BN + col;
    smem_out[b+0] = __float2bfloat16(__uint_as_float(r0));
    smem_out[b+1] = __float2bfloat16(__uint_as_float(r1));
    smem_out[b+2] = __float2bfloat16(__uint_as_float(r2));
    smem_out[b+3] = __float2bfloat16(__uint_as_float(r3));
  }
  __syncthreads();

  int warp = tid/32, lane = tid%32;
  #pragma unroll
  for (int step=0; step<BM/4; step++){
    int r = step*4 + warp;
    int grow = m_block*BM + r;
    int col = lane*8;
    int gcol = n_block*BN + col;
    if (grow < M){
      int4 v = *reinterpret_cast<int4*>(&smem_out[r*BN + col]);
      *reinterpret_cast<int4*>(&C[(long)grow*N + gcol]) = v;
    }
  }

  __syncthreads();
  if (tid < 32){ tmem_dealloc_cg1(tmem_base, BN); }
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
  CU_CHECK(make_tma(&tmaA, Ap, K, M, BK, BM));
  CU_CHECK(make_tma(&tmaB, Bp, K, N, BK, BN));

  int smem_bytes = 1024 + PIPE*(BM*BK + BN*BK)*2 + 512;
  CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);
  dim3 block(128);
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
  gemm_kernel<<<grid, block, smem_bytes, stream>>>(tmaA, tmaB, Cp, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_ns::run);

}  // namespace gemm_ns