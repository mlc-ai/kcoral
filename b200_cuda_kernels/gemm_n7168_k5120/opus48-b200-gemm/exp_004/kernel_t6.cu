#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)
static inline void cu_check(CUresult e,const char* f,int l){ if(e!=CUDA_SUCCESS){ const char* s=nullptr; cuGetErrorString(e,&s); fprintf(stderr,"CU error %s at %s:%d\n", s?s:"?", f, l); exit(1);} }
#define CU_CHECK(call) cu_check((call), __FILE__, __LINE__)

namespace gemm_n7168_k5120 {

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count){
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),"r"(count));
}
__device__ __forceinline__ void fence_smem_barrier_init_fn(){ asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory"); }
__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),"r"(tx):"memory");
}
__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar){
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])):"memory");
}
__device__ __forceinline__ void mbarrier_arrive_cluster_fn(uint64_t* bar, uint32_t target_cta){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(&bar[0]); uint32_t ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra):"r"(a),"r"(target_cta));
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"::"r"(ra));
}
__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase){
  asm volatile("{\n.reg .pred P;\nWAIT_%=:\n"
    "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
    "@!P bra WAIT_%=;\n}\n" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),"r"(phase));
}
__device__ __forceinline__ void cluster_sync_fn(){ asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n":::"memory"); }
__device__ __forceinline__ void named_barrier_sync_fn(int id,int cnt){ asm volatile("barrier.sync.aligned %0, %1;"::"r"(id),"r"(cnt)); }

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1){
  uint32_t sa=(uint32_t)__cvta_generic_to_shared(smem);
  uint32_t ba=(uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF;
  asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
    :: "r"(sa),"l"((uint64_t)d),"r"(c0),"r"(c1),"r"(ba):"memory");
}
__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst, int ncols){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(dst);
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a),"r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr,int ncols){
  asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(addr),"r"(ncols));
}
__device__ __forceinline__ uint64_t make_smem_desc_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo){
  uint64_t d=0; uint32_t addr=(uint32_t)__cvta_generic_to_shared(smem_ptr);
  d |= (uint64_t)(addr & 0x3FFFF) >> 4;
  d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
  d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)2 << 61;
  return d;
}
__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N){
  uint32_t d=0;
  d |= (1u << 4); d |= (1u << 7); d |= (1u << 10);
  d |= ((N/8) << 17); d |= ((M/16) << 24);
  return d;
}
__device__ __forceinline__ void umma_f16_cg2_fn(uint32_t tmem_c, uint64_t da, uint64_t db, uint32_t idesc, uint32_t accum){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
    "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
    :: "r"(tmem_c),"l"(da),"l"(db),"r"(idesc),"r"(accum));
}
__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(&bar[0]);
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
    :: "r"(a),"h"((uint16_t)0x3));
}
__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0,uint32_t* r1,uint32_t* r2,uint32_t* r3,
    uint32_t* r4,uint32_t* r5,uint32_t* r6,uint32_t* r7){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
    : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),"=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}
__device__ __forceinline__ void tmem_load_fence_fn(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }

constexpr int BM=128, BK=64;
constexpr int NACC=2;                    // per cluster N = 512
constexpr int A_TILE=BM*BK*2;            // 16384
constexpr int B_TILE=128*BK*2;           // 16384 per B-quarter
constexpr int STAGE=A_TILE+NACC*B_TILE;  // 49152
constexpr int S=3;
constexpr int STG=BM*256*2;              // 65536 (one accumulator staging)
constexpr int GROUP_M=8;

__device__ __forceinline__ void map_tile(int tile,int num_m,int num_n,int& mc,int& nc){
  int num_pid_in_group=GROUP_M*num_n;
  int group_id=tile/num_pid_in_group;
  int first_m=group_id*GROUP_M;
  int gsm=num_m-first_m; if(gsm>GROUP_M) gsm=GROUP_M;
  int idx=tile%num_pid_in_group;
  mc=first_m + (idx%gsm);
  nc=idx/gsm;
}

__global__ __launch_bounds__(256,1) void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* __restrict__ C, int M, int N, int K,
    int num_m, int num_n, int num_tiles, int num_slots){
  const int tid=threadIdx.x;
  const int rank=blockIdx.x & 1;
  const bool is_leader=(rank==0);
  const int slot=blockIdx.y;
  const int num_kt=K/BK;

  extern __shared__ __align__(1024) uint8_t smem[];
  __nv_bfloat16* stg=(__nv_bfloat16*)(smem + S*STAGE);
  uint64_t* full_bar=(uint64_t*)(smem + S*STAGE + STG);
  uint64_t* empty_bar=full_bar + S;
  uint64_t* mma_done=empty_bar + S;
  uint64_t* tmem_free=mma_done + 1;
  uint32_t* tmem_ptr=(uint32_t*)(tmem_free + 1);

  if(tid==0){
    for(int s=0;s<S;s++){ init_smem_barrier_fn(&full_bar[s],1); init_smem_barrier_fn(&empty_bar[s],1); }
    init_smem_barrier_fn(&mma_done[0],1);
    init_smem_barrier_fn(&tmem_free[0],2);
    fence_smem_barrier_init_fn();
  }
  __syncthreads();
  if(tid<32) tmem_alloc_fn(tmem_ptr, 512);
  __syncthreads();
  cluster_sync_fn();
  uint32_t tmem_base=tmem_ptr[0];
  uint32_t idesc=make_instr_desc_fn(256,256);
  const uint32_t tx=2*NACC*A_TILE + 2*A_TILE; // 6 loads * 16KB

  if(tid==0){
    // ---- Producer (both CTAs) ----
    int sp=0;
    for(int tile=slot; tile<num_tiles; tile+=num_slots){
      int mc,nc; map_tile(tile,num_m,num_n,mc,nc);
      int m_row=mc*256 + rank*BM;
      int b0=nc*512 + rank*128;
      int b1=nc*512 + 256 + rank*128;
      for(int kt=0; kt<num_kt; kt++){
        int s=sp%S;
        if(sp>=S){ int ph=((sp/S)-1)&1; mbarrier_wait_fn(&empty_bar[s], ph); }
        if(is_leader) mbarrier_arrive_and_expect_tx_fn(&full_bar[s], tx);
        __nv_bfloat16* Ab=(__nv_bfloat16*)(smem + s*STAGE);
        __nv_bfloat16* B0b=Ab + BM*BK;
        __nv_bfloat16* B1b=B0b + 128*BK;
        tma_load_2d_cg2_fn(&tma_A, &full_bar[s], Ab,  kt*BK, m_row);
        tma_load_2d_cg2_fn(&tma_B, &full_bar[s], B0b, kt*BK, b0);
        tma_load_2d_cg2_fn(&tma_B, &full_bar[s], B1b, kt*BK, b1);
        sp++;
      }
    }
  } else if(tid==32 && is_leader){
    // ---- MMA issuer (leader only) ----
    int sp=0, i=0, tfp=0;
    for(int tile=slot; tile<num_tiles; tile+=num_slots){
      if(i>=1){ mbarrier_wait_fn(&tmem_free[0], tfp); tfp^=1; }
      for(int kt=0; kt<num_kt; kt++){
        int s=sp%S;
        int ph=(sp/S)&1;
        mbarrier_wait_fn(&full_bar[s], ph);
        __nv_bfloat16* Ab=(__nv_bfloat16*)(smem + s*STAGE);
        __nv_bfloat16* B0b=Ab + BM*BK;
        __nv_bfloat16* B1b=B0b + 128*BK;
        #pragma unroll
        for(int j=0;j<BK/16;j++){
          uint64_t da =make_smem_desc_fn(Ab  + 16*j, 1, 1024);
          uint64_t db0=make_smem_desc_fn(B0b + 16*j, 1, 1024);
          uint64_t db1=make_smem_desc_fn(B1b + 16*j, 1, 1024);
          uint32_t accum=(kt==0 && j==0)?0:1;
          umma_f16_cg2_fn(tmem_base,       da, db0, idesc, accum);
          umma_f16_cg2_fn(tmem_base + 256, da, db1, idesc, accum);
        }
        umma_commit_2sm_fn(&empty_bar[s]);
        if(kt==num_kt-1) umma_commit_2sm_fn(&mma_done[0]);
        sp++;
      }
      i++;
    }
  } else if(tid>=128){
    // ---- Epilogue warpgroup (both CTAs) ----
    int te=tid-128;
    int warp=te/32, lane=te%32;
    int mdp=0;
    for(int tile=slot; tile<num_tiles; tile+=num_slots){
      mbarrier_wait_fn(&mma_done[0], mdp); mdp^=1;
      int mc,nc; map_tile(tile,num_m,num_n,mc,nc);
      int m_row=mc*256 + rank*BM;
      #pragma unroll
      for(int acc=0; acc<NACC; acc++){
        uint32_t cbase=tmem_base + acc*256;
        #pragma unroll
        for(int col=0; col<256; col+=32){
          uint32_t r[32];
          tmem_load_8x_fn(cbase+col+0,  &r[0],&r[1],&r[2],&r[3],&r[4],&r[5],&r[6],&r[7]);
          tmem_load_8x_fn(cbase+col+8,  &r[8],&r[9],&r[10],&r[11],&r[12],&r[13],&r[14],&r[15]);
          tmem_load_8x_fn(cbase+col+16, &r[16],&r[17],&r[18],&r[19],&r[20],&r[21],&r[22],&r[23]);
          tmem_load_8x_fn(cbase+col+24, &r[24],&r[25],&r[26],&r[27],&r[28],&r[29],&r[30],&r[31]);
          tmem_load_fence_fn();
          int base=te*256 + col;
          #pragma unroll
          for(int q=0;q<32;q++) stg[base+q]=__float2bfloat16(__uint_as_float(r[q]));
        }
        named_barrier_sync_fn(1,128);
        if(acc==NACC-1 && te==0){
          if(is_leader) mbarrier_arrive_fn(&tmem_free[0]);
          else          mbarrier_arrive_cluster_fn(&tmem_free[0], 0);
        }
        int n_base=nc*512 + acc*256;
        #pragma unroll
        for(int step=0; step<BM/4; step++){
          int row=step*4 + warp;
          int grow=m_row + row;
          int gcol=n_base + lane*8;
          if(grow<M && gcol+8<=N){
            uint4 data=*reinterpret_cast<uint4*>(&stg[row*256 + lane*8]);
            *reinterpret_cast<uint4*>(&C[(int64_t)grow*N + gcol]) = data;
          }
        }
        named_barrier_sync_fn(1,128);
      }
    }
  }

  __syncthreads();
  cluster_sync_fn();
  if(tid<32) tmem_dealloc_fn(tmem_base, 512);
}

CUresult make_tma_2d(CUtensorMap* d, void* g, uint64_t inner, uint64_t outer, uint32_t bi, uint32_t bo){
  uint64_t globalDim[2]={inner,outer};
  uint64_t globalStrides[1]={inner*2};
  uint32_t boxDim[2]={bi,bo};
  uint32_t elemStrides[2]={1,1};
  return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, g,
    globalDim, globalStrides, boxDim, elemStrides,
    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
  CUDA_CHECK(cudaSetDevice(A.device().device_id));
  int64_t M=A.size(0), K=A.size(1), N=B.size(0);
  __nv_bfloat16* a=static_cast<__nv_bfloat16*>(A.data_ptr());
  __nv_bfloat16* b=static_cast<__nv_bfloat16*>(B.data_ptr());
  __nv_bfloat16* c=static_cast<__nv_bfloat16*>(C.data_ptr());

  CUtensorMap tA,tB;
  CU_CHECK(make_tma_2d(&tA, a, K, M, BK, BM));
  CU_CHECK(make_tma_2d(&tB, b, K, N, BK, 128));

  int smem_bytes = S*STAGE + STG + 256;
  CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  int num_m=(int)((M+255)/256);
  int num_n=(int)((N+511)/512);
  int num_tiles=num_m*num_n;

  int sm_count=0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, A.device().device_id));
  int num_slots=sm_count/2;
  if(num_slots>num_tiles) num_slots=num_tiles;
  if(num_slots<1) num_slots=1;

  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

  cudaLaunchConfig_t config={};
  config.gridDim=dim3(2, num_slots, 1);
  config.blockDim=dim3(256,1,1);
  config.dynamicSmemBytes=smem_bytes;
  config.stream=stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id=cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x=2;
  attrs[0].val.clusterDim.y=1;
  attrs[0].val.clusterDim.z=1;
  config.attrs=attrs;
  config.numAttrs=1;
  CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tA, tB, c, (int)M, (int)N, (int)K, num_m, num_n, num_tiles, num_slots));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120::run);

}  // namespace gemm_n7168_k5120