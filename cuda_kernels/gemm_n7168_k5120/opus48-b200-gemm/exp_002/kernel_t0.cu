#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)

namespace gemm_kernel_ns {

constexpr int BM=128, BN=128, BK=32, PAD=8, SW=BK+PAD; // 40

__device__ __forceinline__ void cp_async_16(uint32_t smem, const void* gmem, int src_size){
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
    :: "r"(smem),"l"(gmem),"r"(src_size):"memory");
}
__device__ __forceinline__ void ldm_x4(uint32_t a, uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3){
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a));
}
__device__ __forceinline__ void ldm_x2(uint32_t a, uint32_t&r0,uint32_t&r1){
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
    :"=r"(r0),"=r"(r1):"r"(a));
}
__device__ __forceinline__ void mma16816(float*d, const uint32_t*a, const uint32_t*b){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

__device__ __forceinline__ void load_tiles(const __nv_bfloat16*A,const __nv_bfloat16*B,
    __nv_bfloat16 (*As)[SW], __nv_bfloat16 (*Bs)[SW],
    int block_m,int block_n,int k0,int M,int K,int tid){
  #pragma unroll
  for(int i=0;i<2;i++){
    int idx=tid+i*256; int r=idx>>2; int c=(idx&3)*8;
    int gm=block_m*BM+r; const __nv_bfloat16* src; int sz;
    if(gm<M){src=A+(long)gm*K+(k0+c); sz=16;} else {src=A; sz=0;}
    uint32_t d=(uint32_t)__cvta_generic_to_shared(&As[r][c]);
    cp_async_16(d,src,sz);
  }
  #pragma unroll
  for(int i=0;i<2;i++){
    int idx=tid+i*256; int r=idx>>2; int c=(idx&3)*8;
    int gn=block_n*BN+r; const __nv_bfloat16* src=B+(long)gn*K+(k0+c);
    uint32_t d=(uint32_t)__cvta_generic_to_shared(&Bs[r][c]);
    cp_async_16(d,src,16);
  }
}

__global__ void __launch_bounds__(256)
gemm_kernel(const __nv_bfloat16* A,const __nv_bfloat16* B,__nv_bfloat16* C,int M,int N,int K){
  int block_m=blockIdx.y, block_n=blockIdx.x;
  int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
  int warp_m=warp>>2, warp_n=warp&3; // 2 x 4 warp grid

  __shared__ __nv_bfloat16 As[2][BM][SW];
  __shared__ __nv_bfloat16 Bs[2][BN][SW];

  float acc[4][4][4];
  #pragma unroll
  for(int i=0;i<4;i++)
    #pragma unroll
    for(int j=0;j<4;j++)
      #pragma unroll
      for(int k=0;k<4;k++) acc[i][j][k]=0.f;

  int num_k=K/BK;
  load_tiles(A,B,As[0],Bs[0],block_m,block_n,0,M,K,tid);
  asm volatile("cp.async.commit_group;\n":::"memory");

  for(int kt=0;kt<num_k;kt++){
    int stage=kt&1;
    if(kt+1<num_k){
      load_tiles(A,B,As[(kt+1)&1],Bs[(kt+1)&1],block_m,block_n,(kt+1)*BK,M,K,tid);
      asm volatile("cp.async.commit_group;\n":::"memory");
      asm volatile("cp.async.wait_group 1;\n":::"memory");
    } else {
      asm volatile("cp.async.wait_group 0;\n":::"memory");
    }
    __syncthreads();

    uint32_t af[4][4], bf[4][2];
    #pragma unroll
    for(int kk=0;kk<BK/16;kk++){
      #pragma unroll
      for(int mi=0;mi<4;mi++){
        int ar=warp_m*64+mi*16+(lane&15);
        int ac=kk*16+((lane>>4)*8);
        uint32_t ad=(uint32_t)__cvta_generic_to_shared(&As[stage][ar][ac]);
        ldm_x4(ad,af[mi][0],af[mi][1],af[mi][2],af[mi][3]);
      }
      #pragma unroll
      for(int ni=0;ni<4;ni++){
        int ll=lane&15;
        int br=warp_n*32+ni*8+(ll&7);
        int bc=kk*16+((ll>>3)*8);
        uint32_t bd=(uint32_t)__cvta_generic_to_shared(&Bs[stage][br][bc]);
        ldm_x2(bd,bf[ni][0],bf[ni][1]);
      }
      #pragma unroll
      for(int mi=0;mi<4;mi++)
        #pragma unroll
        for(int ni=0;ni<4;ni++)
          mma16816(acc[mi][ni],af[mi],bf[ni]);
    }
    __syncthreads();
  }

  int groupID=lane>>2, tin=lane&3;
  #pragma unroll
  for(int mi=0;mi<4;mi++)
    #pragma unroll
    for(int ni=0;ni<4;ni++){
      int base_m=block_m*BM+warp_m*64+mi*16;
      int base_n=block_n*BN+warp_n*32+ni*8;
      float* c=acc[mi][ni];
      int m0=base_m+groupID, m1=base_m+8+groupID, n0=base_n+tin*2;
      if(m0<M) *reinterpret_cast<__nv_bfloat162*>(&C[(long)m0*N+n0])=__floats2bfloat162_rn(c[0],c[1]);
      if(m1<M) *reinterpret_cast<__nv_bfloat162*>(&C[(long)m1*N+n0])=__floats2bfloat162_rn(c[2],c[3]);
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
  CUDA_CHECK(cudaSetDevice(A.device().device_id));
  int M=(int)A.size(0), K=(int)A.size(1), N=(int)B.size(0);
  const __nv_bfloat16* Ap=static_cast<const __nv_bfloat16*>(A.data_ptr());
  const __nv_bfloat16* Bp=static_cast<const __nv_bfloat16*>(B.data_ptr());
  __nv_bfloat16* Cp=static_cast<__nv_bfloat16*>(C.data_ptr());
  dim3 grid((N+BN-1)/BN,(M+BM-1)/BM);
  dim3 block(256);
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type,A.device().device_id));
  gemm_kernel<<<grid,block,0,stream>>>(Ap,Bp,Cp,M,N,K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_kernel_ns::run);

}  // namespace gemm_kernel_ns