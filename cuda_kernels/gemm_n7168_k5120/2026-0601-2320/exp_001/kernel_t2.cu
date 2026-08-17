#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s; cuGetErrorString(_e,&s); \
  fprintf(stderr,"CU error %s at %s:%d\n",s,__FILE__,__LINE__); exit(1);} } while(0)

namespace gemm_n7168_k5120 {

constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 64;
constexpr int STAGES = 4;
constexpr int A_BYTES = BM*BK*2;          // 16384
constexpr int B_BYTES = BN*BK*2;          // 32768
constexpr int STAGE_BYTES = A_BYTES + B_BYTES; // 49152
constexpr int STAGES_BYTES = STAGES*STAGE_BYTES; // 196608

// ---------- mbarrier helpers ----------
__device__ __forceinline__ void init_bar(uint64_t* bar, uint32_t count){
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"
    :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}
__device__ __forceinline__ void fence_bar_init(){
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void arrive_expect_tx(uint64_t* bar, uint32_t tx){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
    :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx) : "memory");
}
__device__ __forceinline__ void bar_wait(uint64_t* bar, uint32_t phase){
  asm volatile(
    "{\n .reg .pred P;\n WAITLBL_%=:\n"
    "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
    "@!P bra WAITLBL_%=;\n }\n"
    :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1){
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
    :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"((uint64_t)d),
       "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(c0), "r"(c1) : "memory");
}

// ---------- tcgen05 (cta_group::1) helpers ----------
__device__ __forceinline__ void tmem_alloc_cg1(uint32_t* dst, int ncols){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(dst);
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a),"r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc_cg1(uint32_t addr,int ncols){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr),"r"(ncols));
}
__device__ __forceinline__ void tmem_relinquish_cg1(){
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void umma_f16_cg1(uint32_t tmem_c,uint64_t da,uint64_t db,uint32_t idesc,uint32_t accum){
  asm volatile("{\n.reg .pred p;\n setp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    :: "r"(tmem_c),"l"(da),"l"(db),"r"(idesc),"r"(accum));
}
__device__ __forceinline__ void umma_commit_cg1(uint64_t* bar){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}
__device__ __forceinline__ void tcgen05_fence_before(){ asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tcgen05_fence_after(){  asm volatile("tcgen05.fence::after_thread_sync;"  ::: "memory"); }

// ---------- descriptors ----------
__device__ __forceinline__ uint64_t make_desc(void* ptr, uint32_t sbo){
  uint32_t addr=(uint32_t)__cvta_generic_to_shared(ptr);
  uint64_t d=0;
  d |= (uint64_t)((addr & 0x3FFFF) >> 4);            // start address
  d |= (uint64_t)0 << 16;                            // LBO (unused, K-major swizzle)
  d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;       // SBO
  d |= (uint64_t)1 << 46;                            // version SM100
  d |= (uint64_t)((addr >> 7) & 7) << 49;            // base offset
  d |= (uint64_t)2 << 61;                            // 128B swizzle
  return d;
}
__device__ __forceinline__ uint32_t make_instr_desc(int M,int N){
  uint32_t d=0;
  d |= (1u<<4);                 // dtype FP32
  d |= (1u<<7);                 // atype BF16
  d |= (1u<<10);                // btype BF16
  d |= ((uint32_t)(N/8)<<17);   // N
  d |= ((uint32_t)(M/16)<<24);  // M
  return d;
}

__global__ void __launch_bounds__(128,1) gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* __restrict__ C, int M, int N, int K){

  extern __shared__ __align__(1024) char smem_raw[];
  uint64_t* full     = (uint64_t*)(smem_raw + STAGES_BYTES);
  uint64_t* empty    = full + STAGES;
  uint64_t* mma_done = empty + STAGES;
  uint32_t* tmem_ptr = (uint32_t*)(mma_done + 1);

  const int tid  = threadIdx.x;
  const int warp = tid >> 5;
  const int m_block = blockIdx.y;
  const int n_block = blockIdx.x;
  const int NUMK = K / BK;

  if (tid==0){
    for(int s=0;s<STAGES;s++){ init_bar(&full[s],1); init_bar(&empty[s],1); }
    init_bar(mma_done,1);
    fence_bar_init();
  }
  if (warp==0) tmem_alloc_cg1(tmem_ptr, BN);
  __syncthreads();
  uint32_t tmem_base = tmem_ptr[0];
  if (warp==0) tmem_relinquish_cg1();

  const uint32_t idesc = make_instr_desc(BM, BN);

  if (tid==0){
    // Producer: issue TMA loads
    for(int k=0;k<NUMK;k++){
      int s=k%STAGES;
      if(k>=STAGES){ uint32_t ph=((k/STAGES)-1)&1; bar_wait(&empty[s],ph); }
      arrive_expect_tx(&full[s], STAGE_BYTES);
      char* aptr = smem_raw + s*STAGE_BYTES;
      char* bptr = aptr + A_BYTES;
      tma_load_2d(&tma_A, &full[s], aptr, k*BK, m_block*BM);
      tma_load_2d(&tma_B, &full[s], bptr, k*BK, n_block*BN);
    }
  } else if (tid==32){
    // Consumer: issue MMAs
    for(int k=0;k<NUMK;k++){
      int s=k%STAGES;
      uint32_t ph=(k/STAGES)&1;
      bar_wait(&full[s],ph);
      char* aptr = smem_raw + s*STAGE_BYTES;
      char* bptr = aptr + A_BYTES;
      #pragma unroll
      for(int j=0;j<4;j++){
        uint64_t da = make_desc(aptr + 32*j, 1024);
        uint64_t db = make_desc(bptr + 32*j, 1024);
        uint32_t accum = (k==0 && j==0)?0:1;
        umma_f16_cg1(tmem_base, da, db, idesc, accum);
      }
      umma_commit_cg1(&empty[s]);
    }
    tcgen05_fence_before();
    umma_commit_cg1(mma_done);
    bar_wait(mma_done, 0);
  }

  __syncthreads();
  tcgen05_fence_after();

  // ---- Epilogue ----
  __nv_bfloat16* smem_out = (__nv_bfloat16*)smem_raw;
  #pragma unroll
  for(int col=0; col<BN; col+=8){
    uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
      : "r"(tmem_base+(uint32_t)col));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    int base = tid*BN + col;
    smem_out[base+0]=__float2bfloat16(__uint_as_float(r0));
    smem_out[base+1]=__float2bfloat16(__uint_as_float(r1));
    smem_out[base+2]=__float2bfloat16(__uint_as_float(r2));
    smem_out[base+3]=__float2bfloat16(__uint_as_float(r3));
    smem_out[base+4]=__float2bfloat16(__uint_as_float(r4));
    smem_out[base+5]=__float2bfloat16(__uint_as_float(r5));
    smem_out[base+6]=__float2bfloat16(__uint_as_float(r6));
    smem_out[base+7]=__float2bfloat16(__uint_as_float(r7));
  }
  __syncthreads();

  // Coalesced SMEM -> global
  {
    int w = tid>>5, l = tid&31;
    #pragma unroll
    for(int rr=0;rr<32;rr++){
      int row = w*32 + rr;
      int gr  = m_block*BM + row;
      if(gr < M){
        int gc = n_block*BN + l*8;
        uint4 v = *reinterpret_cast<uint4*>(&smem_out[row*BN + l*8]);
        *reinterpret_cast<uint4*>(&C[(size_t)gr*N + gc]) = v;
      }
    }
  }
  __syncthreads();
  if (warp==0) tmem_dealloc_cg1(tmem_base, BN);
}

static CUresult make_tma_desc(CUtensorMap* d, void* gptr, uint64_t inner, uint64_t outer,
                              uint32_t box_inner, uint32_t box_outer){
  uint64_t globalDim[2]    = {inner, outer};
  uint64_t globalStrides[1]= {inner*2};
  uint32_t boxDim[2]       = {box_inner, box_outer};
  uint32_t elementStrides[2]={1,1};
  return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gptr,
     globalDim, globalStrides, boxDim, elementStrides,
     CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
     CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
  CUDA_CHECK(cudaSetDevice(A.device().device_id));
  int M=(int)A.size(0), K=(int)A.size(1), N=(int)B.size(0);
  __nv_bfloat16* a=static_cast<__nv_bfloat16*>(A.data_ptr());
  __nv_bfloat16* b=static_cast<__nv_bfloat16*>(B.data_ptr());
  __nv_bfloat16* c=static_cast<__nv_bfloat16*>(C.data_ptr());

  CUtensorMap tA, tB;
  CU_CHECK(make_tma_desc(&tA, a, K, M, BK, BM));
  CU_CHECK(make_tma_desc(&tB, b, K, N, BK, BN));

  dim3 grid(N/BN, (M+BM-1)/BM);
  dim3 block(128);
  int smem_bytes = STAGES_BYTES + 1024;

  static bool set=false;
  if(!set){ CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)); set=true; }

  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
  gemm_kernel<<<grid, block, smem_bytes, stream>>>(tA, tB, c, M, N, K);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120::run);

}  // namespace gemm_n7168_k5120