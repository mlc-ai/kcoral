#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

namespace gemm_n7168_k5120 {

constexpr int ROWS = 128;
constexpr int NTILE = 256;
constexpr int BK = 64;
constexpr int STAGES = 4;
constexpr int KKN = BK / 16;
constexpr int STAGE_ELEMS = ROWS * BK;            // 8192
constexpr int LOAD_BYTES = ROWS * BK * 2;         // 16384
constexpr uint32_t TX_TOTAL = 4u * LOAD_BYTES;    // 65536
constexpr int GROUP_M = 8;

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
__device__ __forceinline__ void arrive_cluster(uint64_t* bar, uint32_t cta){
  uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
  uint32_t r = map_to_cta(a, cta);
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(r) : "memory");
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
  d |= (uint64_t)2<<61;
  return d;
}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N){
  uint32_t d=0;
  d |= (1u<<4); d |= (1u<<7); d |= (1u<<10);
  d |= ((N>>3)<<17); d |= ((M>>4)<<24);
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
__device__ __forceinline__ void named_bar(int id, int cnt){
  asm volatile("barrier.sync.aligned %0, %1;" :: "r"(id), "r"(cnt));
}
__device__ __forceinline__ void fence_after(){
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__global__ void __launch_bounds__(256) gemm(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C, int M, int N, int K)
{
  extern __shared__ char raw[];
  uintptr_t al = ((uintptr_t)raw + 1023) & ~(uintptr_t)1023;
  char* smem = (char*)al;
  __nv_bfloat16* smemA = (__nv_bfloat16*)smem;
  __nv_bfloat16* smemB = smemA + STAGES*STAGE_ELEMS;
  __nv_bfloat16* smem_out = smemB + STAGES*STAGE_ELEMS;
  char* barbase = (char*)(smem_out + ROWS*NTILE);
  uint64_t* full = (uint64_t*)barbase;
  uint64_t* empty = full + STAGES;
  uint64_t* tmem_full = empty + STAGES;
  uint64_t* tmem_empty = tmem_full + 2;
  uint32_t* tmem_smem = (uint32_t*)(tmem_empty + 2);

  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int NK = K / BK;

  int rank = blockIdx.x & 1;
  int n_tiles = N >> 8;
  int m_super = (M + 255) >> 8;
  int total = m_super * n_tiles;
  int num_clusters = gridDim.x >> 1;
  int cluster_id = blockIdx.x >> 1;

  uint32_t idesc = make_idesc(256, 256);

  if (tid == 0){
    #pragma unroll
    for (int s=0;s<STAGES;++s){ init_bar(&full[s],1); init_bar(&empty[s],1); }
    init_bar(&tmem_full[0],1); init_bar(&tmem_full[1],1);
    init_bar(&tmem_empty[0],2); init_bar(&tmem_empty[1],2);
  }
  if (warp==0) tmem_alloc(tmem_smem, 512);
  __syncthreads();
  cluster_sync();
  uint32_t tmem_base = *tmem_smem;

  // group rasterization: returns (m_base, n_base) for global tile id
  auto raster = [&](int t, int& mb, int& nb){
    int gs = GROUP_M * n_tiles;
    int g = t / gs;
    int first_m = g * GROUP_M;
    int rows = m_super - first_m; if (rows > GROUP_M) rows = GROUP_M;
    int in = t - g*gs;
    int mt = first_m + (in % rows);
    int nt = in / rows;
    mb = mt*256; nb = nt*256;
  };

  if (tid < 128){
    // ---- Producer (warp 0, lane 0; both CTAs) ----
    if (warp==0 && lane==0){
      uint32_t pc = 0;
      uint32_t phe[STAGES];
      #pragma unroll
      for(int s=0;s<STAGES;++s) phe[s]=0;
      for (int tg = cluster_id; tg < total; tg += num_clusters){
        int mb, nb; raster(tg, mb, nb);
        int my_m = mb + rank*ROWS;
        int my_n = nb + rank*ROWS;
        for (int k=0;k<NK;++k){
          int s = pc % STAGES;
          if (pc >= STAGES){ bar_wait(&empty[s], phe[s]); phe[s]^=1; }
          uint32_t rbar = map_to_cta((uint32_t)__cvta_generic_to_shared(&full[s]), 0);
          if (rank==0) bar_arrive_expect(&full[s], TX_TOTAL);
          int koff = k*BK;
          tma_cg2(&tma_A, rbar, smemA + s*STAGE_ELEMS, koff, my_m);
          tma_cg2(&tma_B, rbar, smemB + s*STAGE_ELEMS, koff, my_n);
          pc++;
        }
      }
    }
    // ---- MMA (warp 1, lane 0; leader only) ----
    else if (rank==0 && warp==1 && lane==0){
      uint32_t cc = 0;
      uint32_t phf[STAGES];
      #pragma unroll
      for(int s=0;s<STAGES;++s) phf[s]=0;
      uint32_t te_ph[2]={0,0};
      int ti = 0;
      for (int tg = cluster_id; tg < total; tg += num_clusters){
        int buf = ti & 1;
        if (ti >= 2){ bar_wait(&tmem_empty[buf], te_ph[buf]); te_ph[buf]^=1; fence_after(); }
        uint32_t tc = tmem_base + (uint32_t)(buf*256);
        for (int k=0;k<NK;++k){
          int s = cc % STAGES;
          bar_wait(&full[s], phf[s]); phf[s]^=1;
          uint64_t da = make_desc(smemA + s*STAGE_ELEMS, 1, 1024);
          uint64_t db = make_desc(smemB + s*STAGE_ELEMS, 1, 1024);
          #pragma unroll
          for (int kk=0;kk<KKN;++kk){
            umma_cg2(tc, da+(uint64_t)(2*kk), db+(uint64_t)(2*kk), idesc, (k==0&&kk==0)?0u:1u);
          }
          umma_commit_2sm(&empty[s]);
          cc++;
        }
        umma_commit_2sm(&tmem_full[buf]);
        ti++;
      }
    }
  } else {
    // ---- Epilogue warpgroup (tid 128..255; both CTAs) ----
    int local = tid - 128;
    int wig = local >> 5;
    int ln  = local & 31;
    uint32_t tf_ph[2]={0,0};
    int ti = 0;
    for (int tg = cluster_id; tg < total; tg += num_clusters){
      int mb, nb; raster(tg, mb, nb);
      int my_m = mb + rank*ROWS;
      int buf = ti & 1;
      bar_wait(&tmem_full[buf], tf_ph[buf]); tf_ph[buf]^=1;
      fence_after();
      uint32_t colbase = tmem_base + (uint32_t)(buf*256);
      #pragma unroll
      for (int base=0; base<NTILE; base+=32){
        uint32_t r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;
        uint32_t r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31;
        uint32_t a0 = colbase + (uint32_t)base;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
          : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(a0));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
          : "=r"(r8),"=r"(r9),"=r"(r10),"=r"(r11),"=r"(r12),"=r"(r13),"=r"(r14),"=r"(r15) : "r"(a0+8));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
          : "=r"(r16),"=r"(r17),"=r"(r18),"=r"(r19),"=r"(r20),"=r"(r21),"=r"(r22),"=r"(r23) : "r"(a0+16));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
          : "=r"(r24),"=r"(r25),"=r"(r26),"=r"(r27),"=r"(r28),"=r"(r29),"=r"(r30),"=r"(r31) : "r"(a0+24));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        int b = local*NTILE + base;
        smem_out[b+0]=__float2bfloat16(__uint_as_float(r0));
        smem_out[b+1]=__float2bfloat16(__uint_as_float(r1));
        smem_out[b+2]=__float2bfloat16(__uint_as_float(r2));
        smem_out[b+3]=__float2bfloat16(__uint_as_float(r3));
        smem_out[b+4]=__float2bfloat16(__uint_as_float(r4));
        smem_out[b+5]=__float2bfloat16(__uint_as_float(r5));
        smem_out[b+6]=__float2bfloat16(__uint_as_float(r6));
        smem_out[b+7]=__float2bfloat16(__uint_as_float(r7));
        smem_out[b+8]=__float2bfloat16(__uint_as_float(r8));
        smem_out[b+9]=__float2bfloat16(__uint_as_float(r9));
        smem_out[b+10]=__float2bfloat16(__uint_as_float(r10));
        smem_out[b+11]=__float2bfloat16(__uint_as_float(r11));
        smem_out[b+12]=__float2bfloat16(__uint_as_float(r12));
        smem_out[b+13]=__float2bfloat16(__uint_as_float(r13));
        smem_out[b+14]=__float2bfloat16(__uint_as_float(r14));
        smem_out[b+15]=__float2bfloat16(__uint_as_float(r15));
        smem_out[b+16]=__float2bfloat16(__uint_as_float(r16));
        smem_out[b+17]=__float2bfloat16(__uint_as_float(r17));
        smem_out[b+18]=__float2bfloat16(__uint_as_float(r18));
        smem_out[b+19]=__float2bfloat16(__uint_as_float(r19));
        smem_out[b+20]=__float2bfloat16(__uint_as_float(r20));
        smem_out[b+21]=__float2bfloat16(__uint_as_float(r21));
        smem_out[b+22]=__float2bfloat16(__uint_as_float(r22));
        smem_out[b+23]=__float2bfloat16(__uint_as_float(r23));
        smem_out[b+24]=__float2bfloat16(__uint_as_float(r24));
        smem_out[b+25]=__float2bfloat16(__uint_as_float(r25));
        smem_out[b+26]=__float2bfloat16(__uint_as_float(r26));
        smem_out[b+27]=__float2bfloat16(__uint_as_float(r27));
        smem_out[b+28]=__float2bfloat16(__uint_as_float(r28));
        smem_out[b+29]=__float2bfloat16(__uint_as_float(r29));
        smem_out[b+30]=__float2bfloat16(__uint_as_float(r30));
        smem_out[b+31]=__float2bfloat16(__uint_as_float(r31));
      }
      named_bar(1, 128);
      if (local==0) arrive_cluster(&tmem_empty[buf], 0);
      #pragma unroll
      for (int step=0; step<ROWS/4; ++step){
        int row = step*4 + wig;
        int grow = my_m + row;
        int gcol = nb + ln*8;
        if (grow < M){
          uint4 dd = *reinterpret_cast<uint4*>(&smem_out[row*NTILE + ln*8]);
          *reinterpret_cast<uint4*>(C + (long long)grow*N + gcol) = dd;
        }
      }
      named_bar(1, 128);
      ti++;
    }
  }

  __syncthreads();
  cluster_sync();
  if (warp==0) tmem_dealloc(tmem_base, 512);
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

  int dev; cudaGetDevice(&dev);
  int sm; cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
  int n_tiles = N / 256;
  int m_super = (M + 255) / 256;
  int total = n_tiles * m_super;
  int num_clusters = sm / 2;
  if (num_clusters > total) num_clusters = total;
  if (num_clusters < 1) num_clusters = 1;

  dim3 grid(2*num_clusters, 1, 1);
  dim3 block(256);
  size_t smem_bytes = (size_t)3*ROWS*NTILE*2 + 12*8 + 8 + 1024;
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