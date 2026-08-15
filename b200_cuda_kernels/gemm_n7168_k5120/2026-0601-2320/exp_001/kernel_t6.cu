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

constexpr int BM_CTA = 128;
constexpr int BN_CTA = 128;
constexpr int BM_TOT = 256;
constexpr int BN_TOT = 256;     // = NCOLS
constexpr int NCOLS  = 256;
constexpr int BK     = 64;
constexpr int STAGES = 4;
constexpr int A_BYTES = BM_CTA*BK*2;       // 16384
constexpr int B_BYTES = BN_CTA*BK*2;       // 16384
constexpr int STAGE_BYTES = A_BYTES + B_BYTES; // 32768
constexpr int TOTAL_TX = 2*STAGE_BYTES;
constexpr int SMEM_OUT_BYTES = BM_CTA*NCOLS*2; // 65536
constexpr int GROUP_M = 8;

__device__ __forceinline__ void init_bar(uint64_t* b, uint32_t c){
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"
    :: "r"((uint32_t)__cvta_generic_to_shared(b)), "r"(c));
}
__device__ __forceinline__ void fence_bar_init(){
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void arrive_expect_tx(uint64_t* b, uint32_t tx){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
    :: "r"((uint32_t)__cvta_generic_to_shared(b)), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b){
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
    :: "r"((uint32_t)__cvta_generic_to_shared(b)) : "memory");
}
__device__ __forceinline__ void mbar_arrive_remote(uint64_t* b, uint32_t tgt){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(b), ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(a), "r"(tgt));
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(ra));
}
__device__ __forceinline__ void bar_wait(uint64_t* b, uint32_t ph){
  asm volatile("{\n .reg .pred P;\n L_%=:\n"
    "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
    "@!P bra L_%=;\n }\n"
    :: "r"((uint32_t)__cvta_generic_to_shared(b)), "r"(ph));
}
__device__ __forceinline__ void tma_load_2d_cg2(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1){
  uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
  uint32_t ba = (uint32_t)__cvta_generic_to_shared(bar) & 0xFEFFFFFF;
  asm volatile(
    "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
    " [%0], [%1, {%2, %3}], [%4];"
    :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
}
__device__ __forceinline__ void tmem_alloc(uint32_t* dst, int ncols){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(dst);
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a),"r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc(uint32_t addr,int ncols){
  asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(addr),"r"(ncols));
}
__device__ __forceinline__ void tmem_relinquish(){
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
}
__device__ __forceinline__ void umma_f16_cg2(uint32_t tc,uint64_t da,uint64_t db,uint32_t idesc,uint32_t accum){
  asm volatile("{\n.reg .pred p;\n setp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::2.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    :: "r"(tc),"l"(da),"l"(db),"r"(idesc),"r"(accum));
}
__device__ __forceinline__ void umma_commit_2sm(uint64_t* bar){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
    :: "r"(a),"h"((uint16_t)0x3));
}
__device__ __forceinline__ void tc_fence_before(){ asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after(){  asm volatile("tcgen05.fence::after_thread_sync;"  ::: "memory"); }
__device__ __forceinline__ void cluster_sync(){
  asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}
__device__ __forceinline__ void nbar_sync(int id,int cnt){
  asm volatile("barrier.sync.aligned %0, %1;" :: "r"(id),"r"(cnt));
}
__device__ __forceinline__ uint64_t make_desc(void* ptr, uint32_t sbo){
  uint32_t addr=(uint32_t)__cvta_generic_to_shared(ptr);
  uint64_t d=0;
  d |= (uint64_t)((addr & 0x3FFFF) >> 4);
  d |= (uint64_t)0 << 16;
  d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)((addr >> 7) & 7) << 49;
  d |= (uint64_t)2 << 61;       // 128B swizzle
  return d;
}
__device__ __forceinline__ uint32_t make_idesc(int M,int N){
  uint32_t d=0;
  d |= (1u<<4);                 // dtype FP32
  d |= (1u<<7);                 // atype BF16
  d |= (1u<<10);                // btype BF16
  d |= ((uint32_t)(N/8)<<17);
  d |= ((uint32_t)(M/16)<<24);
  return d;
}
__device__ __forceinline__ void tile_coords(int idx,int m_clusters,int n_clusters,int& mc,int& nc){
  int num_in_group = GROUP_M * n_clusters;
  int group = idx / num_in_group;
  int first_m = group * GROUP_M;
  int gsize = m_clusters - first_m; if(gsize>GROUP_M) gsize=GROUP_M; if(gsize<1) gsize=1;
  int r = idx % num_in_group;
  mc = first_m + (r % gsize);
  nc = r / gsize;
}

__global__ void __launch_bounds__(256,1) gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* __restrict__ C, int M, int N, int K,
    int m_clusters, int n_clusters, int total_tiles){

  extern __shared__ __align__(1024) char smem[];
  char* stages = smem;
  __nv_bfloat16* smem_out = (__nv_bfloat16*)(smem + STAGES*STAGE_BYTES);
  uint64_t* bar = (uint64_t*)(smem + STAGES*STAGE_BYTES + SMEM_OUT_BYTES);
  uint64_t* full      = bar;            // [STAGES]
  uint64_t* empty     = full + STAGES;  // [STAGES]
  uint64_t* mma_ready = empty + STAGES; // [2]
  uint64_t* buf_free  = mma_ready + 2;  // [2]
  uint32_t* tmem_ptr  = (uint32_t*)(buf_free + 2);

  const int tid  = threadIdx.x;
  const int warp = tid >> 5;
  const int rank = blockIdx.x;        // 0=leader
  const int slot = blockIdx.y;
  const int NUM_SLOTS = gridDim.y;
  const int NUMK = K / BK;            // assumes NUMK % STAGES == 0
  const int LASTS = STAGES - 1;       // last-k stage index (since NUMK%STAGES==0)

  if (tid==0){
    for(int s=0;s<STAGES;s++){ init_bar(&full[s],1); init_bar(&empty[s],1); }
    init_bar(&mma_ready[0],1); init_bar(&mma_ready[1],1);
    init_bar(&buf_free[0],2);  init_bar(&buf_free[1],2);
    fence_bar_init();
  }
  __syncthreads();
  if (warp==0) tmem_alloc(tmem_ptr, NCOLS);
  __syncthreads();
  uint32_t tmem_base = tmem_ptr[0];
  if (warp==0) tmem_relinquish();
  cluster_sync();

  const uint32_t idesc = make_idesc(BM_TOT, BN_TOT);

  if (tid==0){
    // ---- Producer (both CTAs) ----
    int it=0;
    for(int idx=slot; idx<total_tiles; idx+=NUM_SLOTS){
      int mc,nc; tile_coords(idx,m_clusters,n_clusters,mc,nc);
      int m_base = mc*BM_TOT + rank*BM_CTA;
      int b_base = nc*BN_TOT + rank*BN_CTA;
      for(int k=0;k<NUMK;k++){
        int s=it%STAGES, c=it/STAGES;
        if(it>=STAGES) bar_wait(&empty[s], (uint32_t)((c-1)&1));
        if(rank==0) arrive_expect_tx(&full[s], TOTAL_TX);
        char* aptr = stages + s*STAGE_BYTES;
        char* bptr = aptr + A_BYTES;
        tma_load_2d_cg2(&tma_A, &full[s], aptr, k*BK, m_base);
        tma_load_2d_cg2(&tma_B, &full[s], bptr, k*BK, b_base);
        it++;
      }
    }
  } else if (tid==32 && rank==0){
    // ---- MMA issuer (leader only) ----
    int it=0, j=0;
    for(int idx=slot; idx<total_tiles; idx+=NUM_SLOTS){
      int buf = j & 1;
      if(j>=2) bar_wait(&buf_free[buf], (uint32_t)(((j>>1)-1)&1));
      for(int k=0;k<NUMK;k++){
        int s=it%STAGES, c=it/STAGES;
        bar_wait(&full[s], (uint32_t)(c&1));
        char* aptr = stages + s*STAGE_BYTES;
        char* bptr = aptr + A_BYTES;
        #pragma unroll
        for(int jj=0;jj<BK/16;jj++){
          uint64_t da = make_desc(aptr+32*jj,1024);
          uint64_t db = make_desc(bptr+32*jj,1024);
          uint32_t accum = (k==0 && jj==0)?0:1;
          umma_f16_cg2(tmem_base + buf*256, da, db, idesc, accum);
        }
        if(k < NUMK-1){
          umma_commit_2sm(&empty[s]);
        } else {
          tc_fence_before();
          umma_commit_2sm(&mma_ready[buf]);
        }
        it++;
      }
      j++;
    }
  } else if (tid>=128){
    // ---- Epilogue warpgroup (128 threads) ----
    int local = tid - 128;
    int j=0;
    for(int idx=slot; idx<total_tiles; idx+=NUM_SLOTS){
      int buf = j & 1;
      int mc,nc; tile_coords(idx,m_clusters,n_clusters,mc,nc);
      int m_base = mc*BM_TOT + rank*BM_CTA;
      int n_base = nc*BN_TOT;

      bar_wait(&mma_ready[buf], (uint32_t)((j>>1)&1));
      tc_fence_after();
      // promptly free the tile's last-k SMEM stage (this CTA)
      if(local==0) mbar_arrive(&empty[LASTS]);

      // TMEM[buf] -> smem_out
      #pragma unroll
      for(int col=0; col<NCOLS; col+=8){
        uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
          : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
          : "r"(tmem_base + (uint32_t)(buf*256 + col)));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        int base = local*NCOLS + col;
        smem_out[base+0]=__float2bfloat16(__uint_as_float(r0));
        smem_out[base+1]=__float2bfloat16(__uint_as_float(r1));
        smem_out[base+2]=__float2bfloat16(__uint_as_float(r2));
        smem_out[base+3]=__float2bfloat16(__uint_as_float(r3));
        smem_out[base+4]=__float2bfloat16(__uint_as_float(r4));
        smem_out[base+5]=__float2bfloat16(__uint_as_float(r5));
        smem_out[base+6]=__float2bfloat16(__uint_as_float(r6));
        smem_out[base+7]=__float2bfloat16(__uint_as_float(r7));
      }
      nbar_sync(1,128);
      // TMEM buffer fully read -> free it
      if(local==0){
        tc_fence_before();
        if(rank==0) mbar_arrive(&buf_free[buf]);
        else        mbar_arrive_remote(&buf_free[buf], 0);
      }
      // smem_out -> global (coalesced)
      const int U4_PER_ROW = NCOLS/8;
      const int TOTAL_U4   = BM_CTA*U4_PER_ROW;
      #pragma unroll 4
      for(int i=local;i<TOTAL_U4;i+=128){
        int row  = i / U4_PER_ROW;
        int col8 = (i % U4_PER_ROW)*8;
        int gr = m_base + row;
        if(gr < M){
          uint4 v = *reinterpret_cast<uint4*>(&smem_out[row*NCOLS + col8]);
          *reinterpret_cast<uint4*>(&C[(size_t)gr*N + n_base + col8]) = v;
        }
      }
      nbar_sync(1,128);
      j++;
    }
  }

  __syncthreads();
  cluster_sync();
  if (warp==0) tmem_dealloc(tmem_base, NCOLS);
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
  CU_CHECK(make_tma_desc(&tA, a, K, M, BK, BM_CTA));
  CU_CHECK(make_tma_desc(&tB, b, K, N, BK, BN_CTA));

  int n_clusters = N / BN_TOT;
  int m_clusters = (M + BM_TOT - 1) / BM_TOT;
  int total_tiles = m_clusters * n_clusters;

  int numSM=148;
  cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, A.device().device_id);
  int num_slots = numSM/2;
  if (num_slots < 1) num_slots = 1;
  if (num_slots > total_tiles) num_slots = total_tiles;

  dim3 grid(2, num_slots, 1);
  dim3 block(256);
  int smem_bytes = STAGES*STAGE_BYTES + SMEM_OUT_BYTES + 1024;

  static bool set=false;
  if(!set){ CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)); set=true; }

  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

  cudaLaunchConfig_t config={};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem_bytes;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tA, tB, c, M, N, K,
                                m_clusters, n_clusters, total_tiles));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120::run);

}  // namespace gemm_n7168_k5120