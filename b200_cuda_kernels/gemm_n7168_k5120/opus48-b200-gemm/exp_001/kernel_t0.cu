#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)

namespace gemm_kernel {

#define BM 128
#define BN 128
#define BK 64
#define BKP 72          // padded K pitch (conflict-free ldmatrix)
#define STAGES 3
#define THREADS 256
#define WARPS_M 2
#define WARPS_N 4

__device__ __forceinline__ uint32_t smem_u32(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp_async_16(uint32_t s, const void* g){
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g));
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_wait(){ asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void ldm_x4(uint32_t a, uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3){
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a));
}
__device__ __forceinline__ void ldm_x2(uint32_t a, uint32_t&r0,uint32_t&r1){
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(r0),"=r"(r1):"r"(a));
}
__device__ __forceinline__ void mma_16816(float&c0,float&c1,float&c2,float&c3,
   uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3, uint32_t b0,uint32_t b1){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

__device__ __forceinline__ void load_stage(
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* sA, __nv_bfloat16* sB,
    int buf, int k0, int bm, int bn, int M, int K, int tid){
  #pragma unroll
  for(int i=0;i<4;i++){
    int c = tid + i*THREADS;
    int row = c>>3;
    int cc  = c&7;
    int col = cc*8;
    // A tile
    int gArow = bm*BM + row;
    uint32_t sa = smem_u32(&sA[buf*BM*BKP + row*BKP + col]);
    if(gArow < M){
      const __nv_bfloat16* gp = A + (size_t)gArow*K + (k0 + col);
      cp_async_16(sa, gp);
    }
    // B tile
    int gBrow = bn*BN + row;
    uint32_t sb = smem_u32(&sB[buf*BN*BKP + row*BKP + col]);
    const __nv_bfloat16* gpb = B + (size_t)gBrow*K + (k0 + col);
    cp_async_16(sb, gpb);
  }
  cp_commit();
}

__global__ __launch_bounds__(THREADS) void gemm_kernel_fn(
   const __nv_bfloat16* __restrict__ A,
   const __nv_bfloat16* __restrict__ B,
   __nv_bfloat16* __restrict__ C,
   int M, int N, int K){

  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* sA = smem;
  __nv_bfloat16* sB = smem + STAGES*BM*BKP;

  int tid = threadIdx.x;
  int warp = tid>>5, lane = tid&31;
  int warp_m = warp / WARPS_N;   // 0..1
  int warp_n = warp % WARPS_N;   // 0..3

  int bn = blockIdx.x;
  int bm = blockIdx.y;

  int numK = K / BK;

  float acc[4][4][4];
  #pragma unroll
  for(int i=0;i<4;i++)
    #pragma unroll
    for(int j=0;j<4;j++)
      #pragma unroll
      for(int k=0;k<4;k++) acc[i][j][k]=0.f;

  // prologue
  #pragma unroll
  for(int s=0;s<STAGES-1;s++){
    if(s < numK) load_stage(A,B,sA,sB,s,s*BK,bm,bn,M,K,tid);
  }

  for(int kt=0; kt<numK; kt++){
    cp_wait<STAGES-2>();
    __syncthreads();

    int cur = kt % STAGES;
    uint32_t fragA[4][4];
    uint32_t fragB[4][2];

    #pragma unroll
    for(int kk=0;kk<4;kk++){
      #pragma unroll
      for(int mt=0;mt<4;mt++){
        int m_row = warp_m*64 + mt*16 + (lane&15);
        int k_idx = kk*16 + ((lane>>4)&1)*8;
        uint32_t a = smem_u32(&sA[cur*BM*BKP + m_row*BKP + k_idx]);
        ldm_x4(a, fragA[mt][0],fragA[mt][1],fragA[mt][2],fragA[mt][3]);
      }
      #pragma unroll
      for(int nt=0;nt<4;nt++){
        int n_row = warp_n*32 + nt*8 + (lane&7);
        int k_idx = kk*16 + (((lane&15)>>3)&1)*8;
        uint32_t b = smem_u32(&sB[cur*BN*BKP + n_row*BKP + k_idx]);
        ldm_x2(b, fragB[nt][0], fragB[nt][1]);
      }
      #pragma unroll
      for(int mt=0;mt<4;mt++)
        #pragma unroll
        for(int nt=0;nt<4;nt++)
          mma_16816(acc[mt][nt][0],acc[mt][nt][1],acc[mt][nt][2],acc[mt][nt][3],
             fragA[mt][0],fragA[mt][1],fragA[mt][2],fragA[mt][3], fragB[nt][0],fragB[nt][1]);
    }

    int nk = kt + (STAGES-1);
    if(nk < numK){
      load_stage(A,B,sA,sB,nk%STAGES,nk*BK,bm,bn,M,K,tid);
    }
  }

  // store
  int grp = lane>>2;   // 0..7
  int tg  = lane&3;    // 0..3
  #pragma unroll
  for(int mt=0;mt<4;mt++){
    #pragma unroll
    for(int nt=0;nt<4;nt++){
      int m0 = bm*BM + warp_m*64 + mt*16 + grp;
      int m1 = m0 + 8;
      int n0 = bn*BN + warp_n*32 + nt*8 + tg*2;
      float* a = acc[mt][nt];
      if(m0 < M){
        __nv_bfloat162 v = __floats2bfloat162_rn(a[0], a[1]);
        *reinterpret_cast<__nv_bfloat162*>(C + (size_t)m0*N + n0) = v;
      }
      if(m1 < M){
        __nv_bfloat162 v = __floats2bfloat162_rn(a[2], a[3]);
        *reinterpret_cast<__nv_bfloat162*>(C + (size_t)m1*N + n0) = v;
      }
    }
  }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
  CUDA_CHECK(cudaSetDevice(A.device().device_id));
  int M = (int)A.size(0);
  int K = (int)A.size(1);
  int N = (int)B.size(0);

  const __nv_bfloat16* Ap = static_cast<const __nv_bfloat16*>(A.data_ptr());
  const __nv_bfloat16* Bp = static_cast<const __nv_bfloat16*>(B.data_ptr());
  __nv_bfloat16* Cp = static_cast<__nv_bfloat16*>(C.data_ptr());

  dim3 grid(N/BN, (M+BM-1)/BM);
  dim3 block(THREADS);
  size_t smem = (size_t)STAGES*(BM*BKP + BN*BKP)*sizeof(__nv_bfloat16);

  cudaFuncSetAttribute(gemm_kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
  gemm_kernel_fn<<<grid, block, smem, stream>>>(Ap, Bp, Cp, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_kernel::run);

}  // namespace gemm_kernel