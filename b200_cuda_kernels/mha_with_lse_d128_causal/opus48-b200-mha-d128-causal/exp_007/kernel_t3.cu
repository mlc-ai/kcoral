#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
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

#define STRIDE 136

__device__ __forceinline__ void mma_m16n8k16(
    float &d0,float &d1,float &d2,float &d3,
    uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,
    uint32_t b0,uint32_t b1,
    float c0,float c1,float c2,float c3){
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
      "f"(c0),"f"(c1),"f"(c2),"f"(c3));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3,uint32_t a){
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a));
}

__device__ __forceinline__ uint32_t pack2bf16(float x, float y){
  __nv_bfloat162 v = __floats2bfloat162_rn(x,y);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ void cp_async_row(const __nv_bfloat16* g, __nv_bfloat16* s, bool valid){
  unsigned sa = (unsigned)__cvta_generic_to_shared(s);
  int src = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;\n" :: "r"(sa),"l"(g),"r"(src) : "memory");
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n":::"memory"); }
template<int N> __device__ __forceinline__ void cp_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory"); }

__device__ __forceinline__ void prefetch_kv(const __nv_bfloat16* Kb, const __nv_bfloat16* Vb,
    __nv_bfloat16* Ksh, __nv_bfloat16* Vsh, int row0, int S, int tid){
  #pragma unroll
  for(int idx=tid; idx<64*16; idx+=256){
    int r=idx>>4, c=idx&15;
    int gr=row0+r;
    bool v=(gr<S);
    cp_async_row(Kb+(int64_t)gr*128+c*8, Ksh+r*STRIDE+c*8, v);
    cp_async_row(Vb+(int64_t)gr*128+c*8, Vsh+r*STRIDE+c*8, v);
  }
}

__global__ void __launch_bounds__(256,2) attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S)
{
  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* Q_sh = smem;
  __nv_bfloat16* K_sh = Q_sh + 128*STRIDE;
  __nv_bfloat16* V_sh = K_sh + 2*64*STRIDE;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int w = tid >> 5;
  const int groupID = lane >> 2;
  const int threadID = lane & 3;

  const int qb = blockIdx.x;
  const int bh = blockIdx.y;
  const int q0 = qb * 128;
  if(q0 >= S) return;

  const int64_t base = (int64_t)bh * S * 128;
  const __nv_bfloat16* Qb = Q + base;
  const __nv_bfloat16* Kb = K + base;
  const __nv_bfloat16* Vb = V + base;
  __nv_bfloat16* Ob = O + base;

  const float SCALE = 0.08838834764831843f;
  const float NEG = -1e30f;

  const int base_row = w*16;
  const int qglob0 = q0 + base_row + groupID;
  const int qglob1 = qglob0 + 8;

  float oacc[16][4];
  #pragma unroll
  for(int nd=0;nd<16;nd++){ oacc[nd][0]=oacc[nd][1]=oacc[nd][2]=oacc[nd][3]=0.f; }
  float m_t=NEG, m_b=NEG, l_t=0.f, l_b=0.f;

  int jmax = 2*qb + 1;
  int maxkv = (S - 1) / 64;
  if(jmax > maxkv) jmax = maxkv;

  #pragma unroll
  for(int idx=tid; idx<128*16; idx+=256){
    int r=idx>>4, c=idx&15; int gr=q0+r;
    cp_async_row(Qb+(int64_t)gr*128+c*8, Q_sh+r*STRIDE+c*8, gr<S);
  }
  prefetch_kv(Kb, Vb, K_sh, V_sh, 0, S, tid);
  cp_commit();

  for(int j=0;j<=jmax;j++){
    if(j+1<=jmax){
      int nb=(j+1)&1;
      prefetch_kv(Kb, Vb, K_sh+nb*64*STRIDE, V_sh+nb*64*STRIDE, (j+1)*64, S, tid);
      cp_commit();
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();

    __nv_bfloat16* Kc = K_sh + (j&1)*64*STRIDE;
    __nv_bfloat16* Vc = V_sh + (j&1)*64*STRIDE;

    float sacc[8][4];
    #pragma unroll
    for(int nt=0;nt<8;nt++){ sacc[nt][0]=sacc[nt][1]=sacc[nt][2]=sacc[nt][3]=0.f; }
    #pragma unroll
    for(int kt=0;kt<8;kt++){
      int d0 = kt*16 + threadID*2;
      int d1 = d0 + 8;
      int qr0 = base_row + groupID;
      int qr1 = qr0 + 8;
      uint32_t a0 = *reinterpret_cast<uint32_t*>(&Q_sh[qr0*STRIDE + d0]);
      uint32_t a1 = *reinterpret_cast<uint32_t*>(&Q_sh[qr1*STRIDE + d0]);
      uint32_t a2 = *reinterpret_cast<uint32_t*>(&Q_sh[qr0*STRIDE + d1]);
      uint32_t a3 = *reinterpret_cast<uint32_t*>(&Q_sh[qr1*STRIDE + d1]);
      #pragma unroll
      for(int nt=0;nt<8;nt++){
        int key = nt*8 + groupID;
        uint32_t b0 = *reinterpret_cast<uint32_t*>(&Kc[key*STRIDE + d0]);
        uint32_t b1 = *reinterpret_cast<uint32_t*>(&Kc[key*STRIDE + d1]);
        mma_m16n8k16(sacc[nt][0],sacc[nt][1],sacc[nt][2],sacc[nt][3],
                     a0,a1,a2,a3,b0,b1,
                     sacc[nt][0],sacc[nt][1],sacc[nt][2],sacc[nt][3]);
      }
    }

    bool needmask = ((j*64 + 63) >= (q0 + base_row)) || ((j*64+64) > S);
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      sacc[nt][0]*=SCALE; sacc[nt][1]*=SCALE; sacc[nt][2]*=SCALE; sacc[nt][3]*=SCALE;
      if(needmask){
        int kc0 = j*64 + nt*8 + threadID*2;
        int kc1 = kc0+1;
        if(kc0 > qglob0 || kc0 >= S) sacc[nt][0]=NEG;
        if(kc1 > qglob0 || kc1 >= S) sacc[nt][1]=NEG;
        if(kc0 > qglob1 || kc0 >= S) sacc[nt][2]=NEG;
        if(kc1 > qglob1 || kc1 >= S) sacc[nt][3]=NEG;
      }
    }

    float rmax_t=NEG, rmax_b=NEG;
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      rmax_t = fmaxf(rmax_t, fmaxf(sacc[nt][0],sacc[nt][1]));
      rmax_b = fmaxf(rmax_b, fmaxf(sacc[nt][2],sacc[nt][3]));
    }
    rmax_t = fmaxf(rmax_t, __shfl_xor_sync(0xffffffffu, rmax_t, 1));
    rmax_t = fmaxf(rmax_t, __shfl_xor_sync(0xffffffffu, rmax_t, 2));
    rmax_b = fmaxf(rmax_b, __shfl_xor_sync(0xffffffffu, rmax_b, 1));
    rmax_b = fmaxf(rmax_b, __shfl_xor_sync(0xffffffffu, rmax_b, 2));

    float m_new_t = fmaxf(m_t, rmax_t);
    float m_new_b = fmaxf(m_b, rmax_b);
    float corr_t = __expf(m_t - m_new_t);
    float corr_b = __expf(m_b - m_new_b);

    float psum_t=0.f, psum_b=0.f;
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      float p0=__expf(sacc[nt][0]-m_new_t);
      float p1=__expf(sacc[nt][1]-m_new_t);
      float p2=__expf(sacc[nt][2]-m_new_b);
      float p3=__expf(sacc[nt][3]-m_new_b);
      sacc[nt][0]=p0; sacc[nt][1]=p1; sacc[nt][2]=p2; sacc[nt][3]=p3;
      psum_t += p0+p1; psum_b += p2+p3;
    }
    psum_t += __shfl_xor_sync(0xffffffffu,psum_t,1);
    psum_t += __shfl_xor_sync(0xffffffffu,psum_t,2);
    psum_b += __shfl_xor_sync(0xffffffffu,psum_b,1);
    psum_b += __shfl_xor_sync(0xffffffffu,psum_b,2);

    l_t = l_t*corr_t + psum_t;
    l_b = l_b*corr_b + psum_b;
    m_t = m_new_t; m_b = m_new_b;

    #pragma unroll
    for(int nd=0;nd<16;nd++){
      oacc[nd][0]*=corr_t; oacc[nd][1]*=corr_t;
      oacc[nd][2]*=corr_b; oacc[nd][3]*=corr_b;
    }

    #pragma unroll
    for(int kt=0;kt<4;kt++){
      uint32_t a0 = pack2bf16(sacc[2*kt][0],   sacc[2*kt][1]);
      uint32_t a1 = pack2bf16(sacc[2*kt][2],   sacc[2*kt][3]);
      uint32_t a2 = pack2bf16(sacc[2*kt+1][0], sacc[2*kt+1][1]);
      uint32_t a3 = pack2bf16(sacc[2*kt+1][2], sacc[2*kt+1][3]);

      int mt = lane >> 3;
      int rr = lane & 7;
      int keyhalf = mt & 1;
      int ndoff = mt >> 1;
      int key = kt*16 + keyhalf*8 + rr;

      #pragma unroll
      for(int ndp=0;ndp<8;ndp++){
        int nd = ndp*2;
        int dc = (nd + ndoff)*8;
        uint32_t addr = (uint32_t)__cvta_generic_to_shared(&Vc[key*STRIDE + dc]);
        uint32_t r0,r1,r2,r3;
        ldmatrix_x4_trans(r0,r1,r2,r3,addr);
        mma_m16n8k16(oacc[nd][0],oacc[nd][1],oacc[nd][2],oacc[nd][3],
                     a0,a1,a2,a3,r0,r1,
                     oacc[nd][0],oacc[nd][1],oacc[nd][2],oacc[nd][3]);
        mma_m16n8k16(oacc[nd+1][0],oacc[nd+1][1],oacc[nd+1][2],oacc[nd+1][3],
                     a0,a1,a2,a3,r2,r3,
                     oacc[nd+1][0],oacc[nd+1][1],oacc[nd+1][2],oacc[nd+1][3]);
      }
    }
    __syncthreads();
  }

  float inv_l_t = 1.f / l_t;
  float inv_l_b = 1.f / l_b;
  if(qglob0 < S){
    #pragma unroll
    for(int nd=0;nd<16;nd++){
      int d = nd*8 + threadID*2;
      float o0 = oacc[nd][0]*inv_l_t;
      float o1 = oacc[nd][1]*inv_l_t;
      __nv_bfloat162 pk = __floats2bfloat162_rn(o0,o1);
      *reinterpret_cast<__nv_bfloat162*>(&Ob[(int64_t)qglob0*128 + d]) = pk;
    }
    if(threadID==0) LSE[(int64_t)bh*S + qglob0] = m_t + logf(l_t);
  }
  if(qglob1 < S){
    #pragma unroll
    for(int nd=0;nd<16;nd++){
      int d = nd*8 + threadID*2;
      float o0 = oacc[nd][2]*inv_l_b;
      float o1 = oacc[nd][3]*inv_l_b;
      __nv_bfloat162 pk = __floats2bfloat162_rn(o0,o1);
      *reinterpret_cast<__nv_bfloat162*>(&Ob[(int64_t)qglob1*128 + d]) = pk;
    }
    if(threadID==0) LSE[(int64_t)bh*S + qglob1] = m_b + logf(l_b);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int64_t B = Q.size(0);
  int64_t H = Q.size(1);
  int64_t S = Q.size(2);

  const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* LSEp = static_cast<float*>(LSE.data_ptr());

  int num_qblocks = (int)((S + 127) / 128);
  dim3 grid(num_qblocks, (unsigned)(B*H), 1);
  dim3 block(256);
  size_t shmem = (size_t)(128 + 2*64 + 2*64)*STRIDE*sizeof(__nv_bfloat16);

  static bool attr_set = false;
  if(!attr_set){
    cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    attr_set = true;
  }

  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  attn_kernel<<<grid, block, shmem, stream>>>(Qp, Kp, Vp, Op, LSEp, (int)S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel