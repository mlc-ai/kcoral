#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_kernel {

#define NEG_INF (-__int_as_float(0x7f800000))

__device__ __forceinline__ void mma16816(float &d0,float&d1,float&d2,float&d3,
   uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3, uint32_t b0,uint32_t b1){
  asm volatile(
   "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
   "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
   : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
   : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

__device__ __forceinline__ uint32_t pack2v(const __nv_bfloat16* p, int i0, int i1){
  uint16_t a=*reinterpret_cast<const uint16_t*>(p+i0);
  uint16_t b=*reinterpret_cast<const uint16_t*>(p+i1);
  return (uint32_t)a | ((uint32_t)b<<16);
}
__device__ __forceinline__ uint32_t pack2f(float x,float y){
  __nv_bfloat16 a=__float2bfloat16(x), b=__float2bfloat16(y);
  uint16_t ai=*reinterpret_cast<uint16_t*>(&a), bi=*reinterpret_cast<uint16_t*>(&b);
  return (uint32_t)ai | ((uint32_t)bi<<16);
}

// BM=64 query rows per CTA, BN=64 key rows per iter, D=128. 4 warps (128 threads).
// warp w handles query rows [16*w, 16*w+16). Each thread owns 2 query rows (gid, gid+8).
__global__ __launch_bounds__(128) void mha_kernel_fn(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int H, int B){
  const int D=128;
  int qcta = blockIdx.x*64;
  int h = blockIdx.y;
  int b = blockIdx.z;
  int64_t bh = (int64_t)b*H + h;
  const __nv_bfloat16* Qb = Q + bh*(int64_t)S*D;
  const __nv_bfloat16* Kb = K + bh*(int64_t)S*D;
  const __nv_bfloat16* Vb = V + bh*(int64_t)S*D;
  __nv_bfloat16* Ob = O + bh*(int64_t)S*D;

  extern __shared__ char smem[];
  __nv_bfloat16* Qs=(__nv_bfloat16*)smem;
  __nv_bfloat16* Ks=Qs + 64*128;
  __nv_bfloat16* Vs=Ks + 64*128;

  int tid=threadIdx.x;
  int w=tid>>5; int lane=tid&31; int gid=lane>>2; int tig=lane&3;

  float scale = rsqrtf((float)D);

  // load Q tile once
  for(int i=tid;i<64*128;i+=128){
    int r=i>>7, d=i&127; int q=qcta+r;
    Qs[i] = (q<S)? Qb[(int64_t)q*128+d] : __float2bfloat16(0.f);
  }
  __syncthreads();

  float o[16][4];
  #pragma unroll
  for(int nt=0;nt<16;nt++){o[nt][0]=o[nt][1]=o[nt][2]=o[nt][3]=0.f;}
  float m0=NEG_INF, m1=NEG_INF, l0=0.f, l1=0.f;

  for(int kb=0;kb<S;kb+=64){
    // load K,V block
    for(int i=tid;i<64*128;i+=128){
      int r=i>>7, d=i&127; int key=kb+r;
      if(key<S){ Ks[i]=Kb[(int64_t)key*128+d]; Vs[i]=Vb[(int64_t)key*128+d]; }
      else { Ks[i]=__float2bfloat16(0.f); Vs[i]=__float2bfloat16(0.f); }
    }
    __syncthreads();

    // ---- S = Q @ K^T ----  s[nt] : query rows {gid,gid+8}, keys {8nt+2tig,+1}
    float s[8][4];
    #pragma unroll
    for(int nt=0;nt<8;nt++){s[nt][0]=s[nt][1]=s[nt][2]=s[nt][3]=0.f;}
    #pragma unroll
    for(int kt=0;kt<8;kt++){
      int qrow0=16*w+gid, qrow1=qrow0+8;
      int dcol=16*kt+2*tig;
      uint32_t a0=*reinterpret_cast<const uint32_t*>(&Qs[qrow0*128+dcol]);
      uint32_t a1=*reinterpret_cast<const uint32_t*>(&Qs[qrow1*128+dcol]);
      uint32_t a2=*reinterpret_cast<const uint32_t*>(&Qs[qrow0*128+dcol+8]);
      uint32_t a3=*reinterpret_cast<const uint32_t*>(&Qs[qrow1*128+dcol+8]);
      #pragma unroll
      for(int nt=0;nt<8;nt++){
        int krow=8*nt+gid;
        uint32_t b0=*reinterpret_cast<const uint32_t*>(&Ks[krow*128+dcol]);
        uint32_t b1=*reinterpret_cast<const uint32_t*>(&Ks[krow*128+dcol+8]);
        mma16816(s[nt][0],s[nt][1],s[nt][2],s[nt][3],a0,a1,a2,a3,b0,b1);
      }
    }

    // scale + mask out-of-range keys
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      int keyA=kb+8*nt+2*tig, keyB=keyA+1;
      s[nt][0]*=scale; s[nt][1]*=scale; s[nt][2]*=scale; s[nt][3]*=scale;
      if(keyA>=S){s[nt][0]=NEG_INF;s[nt][2]=NEG_INF;}
      if(keyB>=S){s[nt][1]=NEG_INF;s[nt][3]=NEG_INF;}
    }

    // block max per row (reduce across tig group of 4)
    float bm0=NEG_INF, bm1=NEG_INF;
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      bm0=fmaxf(bm0,fmaxf(s[nt][0],s[nt][1]));
      bm1=fmaxf(bm1,fmaxf(s[nt][2],s[nt][3]));
    }
    #pragma unroll
    for(int off=1;off<4;off<<=1){
      bm0=fmaxf(bm0,__shfl_xor_sync(0xffffffff,bm0,off));
      bm1=fmaxf(bm1,__shfl_xor_sync(0xffffffff,bm1,off));
    }
    float nm0=fmaxf(m0,bm0), nm1=fmaxf(m1,bm1);
    float corr0=__expf(m0-nm0), corr1=__expf(m1-nm1);

    // P = exp(s - nm) ; row sums
    float sum0=0.f, sum1=0.f;
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      float e0=__expf(s[nt][0]-nm0);
      float e1=__expf(s[nt][1]-nm0);
      float e2=__expf(s[nt][2]-nm1);
      float e3=__expf(s[nt][3]-nm1);
      s[nt][0]=e0;s[nt][1]=e1;s[nt][2]=e2;s[nt][3]=e3;
      sum0+=e0+e1; sum1+=e2+e3;
    }
    #pragma unroll
    for(int off=1;off<4;off<<=1){
      sum0+=__shfl_xor_sync(0xffffffff,sum0,off);
      sum1+=__shfl_xor_sync(0xffffffff,sum1,off);
    }
    l0=l0*corr0+sum0; l1=l1*corr1+sum1;
    m0=nm0; m1=nm1;

    // correct running O
    #pragma unroll
    for(int nt=0;nt<16;nt++){
      o[nt][0]*=corr0;o[nt][1]*=corr0;o[nt][2]*=corr1;o[nt][3]*=corr1;
    }

    // ---- O += P @ V ----  (P frag reused directly from s accumulator layout)
    #pragma unroll
    for(int kt=0;kt<4;kt++){
      uint32_t a0=pack2f(s[2*kt][0],s[2*kt][1]);
      uint32_t a1=pack2f(s[2*kt][2],s[2*kt][3]);
      uint32_t a2=pack2f(s[2*kt+1][0],s[2*kt+1][1]);
      uint32_t a3=pack2f(s[2*kt+1][2],s[2*kt+1][3]);
      int key0=16*kt+2*tig;
      #pragma unroll
      for(int nt=0;nt<16;nt++){
        int dcol=8*nt+gid;
        uint32_t b0=pack2v(Vs,(key0)*128+dcol,(key0+1)*128+dcol);
        uint32_t b1=pack2v(Vs,(key0+8)*128+dcol,(key0+9)*128+dcol);
        mma16816(o[nt][0],o[nt][1],o[nt][2],o[nt][3],a0,a1,a2,a3,b0,b1);
      }
    }
    __syncthreads();
  }

  int q0=qcta+16*w+gid, q1=q0+8;
  float inv0=1.f/l0, inv1=1.f/l1;
  #pragma unroll
  for(int nt=0;nt<16;nt++){
    int d=8*nt+2*tig;
    if(q0<S){ Ob[(int64_t)q0*128+d]  =__float2bfloat16(o[nt][0]*inv0);
              Ob[(int64_t)q0*128+d+1]=__float2bfloat16(o[nt][1]*inv0);}
    if(q1<S){ Ob[(int64_t)q1*128+d]  =__float2bfloat16(o[nt][2]*inv1);
              Ob[(int64_t)q1*128+d+1]=__float2bfloat16(o[nt][3]*inv1);}
  }
  if(tig==0){
    if(q0<S) LSE[bh*(int64_t)S+q0]=m0+logf(l0);
    if(q1<S) LSE[bh*(int64_t)S+q1]=m1+logf(l1);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bsz=(int)Q.size(0), Hsz=(int)Q.size(1), Ssz=(int)Q.size(2);

  const __nv_bfloat16* Qp=static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp=static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp=static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
  float* Lp=static_cast<float*>(LSE.data_ptr());

  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  int smem = 3*64*128*(int)sizeof(__nv_bfloat16); // 48KB
  static bool attr_set=false;
  if(!attr_set){
    cudaFuncSetAttribute(mha_kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr_set=true;
  }

  dim3 grid((Ssz+63)/64, Hsz, Bsz);
  mha_kernel_fn<<<grid,128,smem,stream>>>(Qp,Kp,Vp,Op,Lp,Ssz,Hsz,Bsz);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel