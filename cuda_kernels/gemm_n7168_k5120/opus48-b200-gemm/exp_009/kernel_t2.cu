#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)

namespace gemm_kernel {

constexpr int BM=128, BN=128, BK=32;
constexpr int WARPS_M=2, WARPS_N=4;
constexpr int WM=BM/WARPS_M; // 64
constexpr int WN=BN/WARPS_N; // 32
constexpr int MI=WM/16;      // 4
constexpr int NI=WN/8;       // 4
constexpr int THREADS=WARPS_M*WARPS_N*32; // 256

__device__ __forceinline__ uint32_t cvta(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem){
  uint32_t s=cvta(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n"::: "memory"); }
template<int N> __device__ __forceinline__ void cp_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory"); }

__device__ __forceinline__ void ldm_x4(uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3,uint32_t a){
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a));
}
__device__ __forceinline__ void ldm_x2(uint32_t&r0,uint32_t&r1,uint32_t a){
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
    :"=r"(r0),"=r"(r1):"r"(a));
}
__device__ __forceinline__ void mma16816(float&c0,float&c1,float&c2,float&c3,
    uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

__device__ __forceinline__ void load_tile(
    __nv_bfloat16* As_buf, __nv_bfloat16* Bs_buf,
    const __nv_bfloat16* A, const __nv_bfloat16* B,
    int block_m, int block_n, int k0, int M, int K, int tid){
  #pragma unroll
  for(int i=0;i<2;i++){
    int li=tid+i*THREADS;
    int row=li>>2; int col=(li&3)*8;
    int grow=block_m*BM+row; if(grow>M-1)grow=M-1;
    cp_async16(&As_buf[row*BK+col], A+(size_t)grow*K+(k0+col));
  }
  #pragma unroll
  for(int i=0;i<2;i++){
    int li=tid+i*THREADS;
    int row=li>>2; int col=(li&3)*8;
    int grow=block_n*BN+row;
    cp_async16(&Bs_buf[row*BK+col], B+(size_t)grow*K+(k0+col));
  }
}

__global__ void __launch_bounds__(THREADS)
gemm_kernel_fn(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
               __nv_bfloat16* __restrict__ C, int M, int N, int K){
  __shared__ __nv_bfloat16 As[2][BM*BK];
  __shared__ __nv_bfloat16 Bs[2][BN*BK];
  int tid=threadIdx.x;
  int lane=tid&31;
  int wid=tid>>5;
  int warp_m=wid/WARPS_N, warp_n=wid%WARPS_N;
  int m_warp_base=warp_m*WM, n_warp_base=warp_n*WN;
  int block_m=blockIdx.y, block_n=blockIdx.x;

  float acc[MI][NI][4];
  #pragma unroll
  for(int i=0;i<MI;i++)
    #pragma unroll
    for(int j=0;j<NI;j++){acc[i][j][0]=0.f;acc[i][j][1]=0.f;acc[i][j][2]=0.f;acc[i][j][3]=0.f;}

  int num_k = K/BK;

  load_tile(As[0],Bs[0],A,B,block_m,block_n,0,M,K,tid);
  cp_commit();

  for(int t=0;t<num_k;t++){
    int nt=t+1;
    if(nt<num_k){ load_tile(As[nt&1],Bs[nt&1],A,B,block_m,block_n,nt*BK,M,K,tid); cp_commit(); cp_wait<1>(); }
    else { cp_wait<0>(); }
    __syncthreads();
    int buf=t&1;
    #pragma unroll
    for(int kk=0;kk<BK/16;kk++){
      int kb=kk*16;
      uint32_t a[MI][4];
      #pragma unroll
      for(int mi=0;mi<MI;mi++){
        int mrow=m_warp_base+mi*16 + (lane&15);
        int acol=kb + ((lane>>4)*8);
        uint32_t addr=cvta(&As[buf][mrow*BK+acol]);
        ldm_x4(a[mi][0],a[mi][1],a[mi][2],a[mi][3],addr);
      }
      uint32_t b[NI][2];
      #pragma unroll
      for(int ni=0;ni<NI;ni++){
        int nrow=n_warp_base+ni*8 + (lane&7);
        int bcol=kb + (((lane>>3)&1)*8);
        uint32_t addr=cvta(&Bs[buf][nrow*BK+bcol]);
        ldm_x2(b[ni][0],b[ni][1],addr);
      }
      #pragma unroll
      for(int mi=0;mi<MI;mi++)
        #pragma unroll
        for(int ni=0;ni<NI;ni++)
          mma16816(acc[mi][ni][0],acc[mi][ni][1],acc[mi][ni][2],acc[mi][ni][3],
                   a[mi][0],a[mi][1],a[mi][2],a[mi][3], b[ni][0],b[ni][1]);
    }
    __syncthreads();
  }

  int groupID=lane>>2; int tig=lane&3;
  #pragma unroll
  for(int mi=0;mi<MI;mi++){
    #pragma unroll
    for(int ni=0;ni<NI;ni++){
      int base_row=block_m*BM+m_warp_base+mi*16;
      int base_col=block_n*BN+n_warp_base+ni*8+tig*2;
      int r0=base_row+groupID, r1=base_row+groupID+8;
      if(r0<M){
        C[(size_t)r0*N+base_col+0]=__float2bfloat16(acc[mi][ni][0]);
        C[(size_t)r0*N+base_col+1]=__float2bfloat16(acc[mi][ni][1]);
      }
      if(r1<M){
        C[(size_t)r1*N+base_col+0]=__float2bfloat16(acc[mi][ni][2]);
        C[(size_t)r1*N+base_col+1]=__float2bfloat16(acc[mi][ni][3]);
      }
    }
  }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
  CUDA_CHECK(cudaSetDevice(A.device().device_id));
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(0);
  const __nv_bfloat16* a=static_cast<const __nv_bfloat16*>(A.data_ptr());
  const __nv_bfloat16* b=static_cast<const __nv_bfloat16*>(B.data_ptr());
  __nv_bfloat16* c=static_cast<__nv_bfloat16*>(C.data_ptr());
  dim3 grid(N/BN, (M+BM-1)/BM);
  dim3 block(THREADS);
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
  gemm_kernel_fn<<<grid, block, 0, stream>>>(a,b,c,M,N,K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_kernel::run);

}  // namespace gemm_kernel